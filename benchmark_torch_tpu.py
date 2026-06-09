import time
from absl import app
from absl import logging
import torch
from torch import distributed as dist
from torch import nn
from torch.distributed import fsdp
import torch.distributed.tensor as dt
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal.utils import log_utils
import transformers

log_utils.log_to_stderr()

# Configuration Variables
MODEL_COMPILE = True
BATCH_SIZE = 1
SEQ_LEN = 2048
NUM_TRAIN_STEPS = 20

def _torch_compile_model(model):
  return torch.compile(
      model, dynamic=False, backend=torch_tpu_compile.TpuBackend()
  )

def _shard_and_materialize_model(
    model: nn.Module, device: str, config
) -> nn.Module:
  """Shards the meta model and materializes it on the target device."""
  for layer in model.model.layers:
    fsdp.fully_shard(layer)
  fsdp.fully_shard(model)
  model.to_empty(device=device)

  # Load model config and instantiate CPU model (random weights)
  with torch.device("cpu"):
    cpu_model = transformers.AutoModelForCausalLM.from_config(
        config, dtype=torch.bfloat16
    )

  # Re-init RoPE buffers (persistent=False) since to_empty wipes them.
  for mod, cpu_mod in zip(model.modules(), cpu_model.modules()):
    if hasattr(cpu_mod, "inv_freq"):
      mod.register_buffer(
          "inv_freq", cpu_mod.inv_freq.to(device), persistent=False
      )
      if hasattr(cpu_mod, "attention_scaling"):
        mod.attention_scaling = cpu_mod.attention_scaling

  # Shard the weights and load them onto the meta model.
  empty_sharded_sd = model.state_dict()
  sharded_sd = {}
  for param_name, full_cpu_tensor in cpu_model.state_dict().items():
    sharded_empty_param = empty_sharded_sd.get(param_name)
    if sharded_empty_param is None:
      continue
    sharded_tensor = dt.distribute_tensor(
        full_cpu_tensor,
        device_mesh=sharded_empty_param.device_mesh,
        placements=sharded_empty_param.placements,
        src_data_rank=None,
    )
    sharded_sd[param_name] = nn.Parameter(sharded_tensor)

  model.load_state_dict(sharded_sd, assign=True)
  torch.tpu.synchronize()
  return model

def worker_fn(argv=None):
  del argv  # Unused
  device = torch.device("tpu")
  dist.init_process_group(backend="tpu_dist")

  rank = dist.get_rank()
  world_size = dist.get_world_size()
  global_batch_size = BATCH_SIZE * world_size
  logging.info(
      "Using batch_size=%d (global=%d, world=%d)",
      BATCH_SIZE,
      global_batch_size,
      world_size,
  )

  # Manually construct Llama 3.2 1B config
  config = transformers.LlamaConfig(
      vocab_size=128256,
      hidden_size=2048,
      intermediate_size=8192,
      num_hidden_layers=16,
      num_attention_heads=32,
      num_key_value_heads=8,
      hidden_act="silu",
      max_position_embeddings=SEQ_LEN,
      initializer_range=0.02,
      rms_norm_eps=1e-05,
      use_cache=True,
      bos_token_id=128000,
      eos_token_id=128001,
      tie_word_embeddings=True,
      torch_dtype="bfloat16"
  )

  # Shard the model using FSDP strategy.
  logging.info("Instantiating meta-model for FSDP")
  with torch.device("meta"):
    model: nn.Module = transformers.AutoModelForCausalLM.from_config(
        config, dtype=torch.bfloat16
    )
  model = _shard_and_materialize_model(model, device, config)

  model.gradient_checkpointing_enable(
      gradient_checkpointing_kwargs={"use_reentrant": False}
  )
  vocab_size = model.config.vocab_size

  if MODEL_COMPILE:
    logging.info("Compiling model on rank %d", rank)
    model = _torch_compile_model(model)

  torch.manual_seed(rank)

  # Generate random input and target tokens.
  data = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN + 1), device=device)
  input_tokens = data[:, :SEQ_LEN]
  target_tokens = data[:, 1:]

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  model.train()

  # Warmup steps
  warmup = 5
  step_times = []

  for step in range(NUM_TRAIN_STEPS):
    optimizer.zero_grad()
    torch.tpu.synchronize()
    t0 = time.perf_counter()

    output = model(input_ids=input_tokens, labels=target_tokens)
    loss = output.loss
    loss.backward()
    optimizer.step()

    torch.tpu.synchronize()
    t1 = time.perf_counter()
    step_time = t1 - t0
    step_times.append(step_time)

    if rank == 0:
      throughput = (BATCH_SIZE * world_size * SEQ_LEN) / step_time
      print(f"Step {step+1}/{NUM_TRAIN_STEPS} | Loss: {loss.item():.4f} | Step Time: {step_time:.4f}s | Throughput: {throughput:.2f} tokens/s (total), {throughput/world_size:.2f} tokens/s/chip", flush=True)

  if len(step_times) > warmup:
    avg_step_time = sum(step_times[warmup:]) / (len(step_times) - warmup)
    avg_throughput = (BATCH_SIZE * world_size * SEQ_LEN) / avg_step_time
    if rank == 0:
      print(f"\n==== Training Completed ====", flush=True)
      print(f"Average Step Time (after {warmup} warmup steps): {avg_step_time:.4f}s", flush=True)
      print(f"Average Throughput (total): {avg_throughput:.2f} tokens/s", flush=True)
      print(f"Average Throughput (per chip): {avg_throughput/world_size:.2f} tokens/s/chip", flush=True)

if __name__ == "__main__":
  app.run(worker_fn)
