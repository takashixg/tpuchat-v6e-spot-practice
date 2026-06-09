# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import logging
import os
import time
from collections import defaultdict
from typing import Tuple

import helper
import jax
import jax.numpy as jnp
import numpy as np
import optax
import splash_attn
import torch
import torch.nn.functional
from jax.experimental import mesh_utils
from jax.experimental.pallas.ops.tpu import flash_attention
from jax.sharding import Mesh, NamedSharding
from torch.utils import _pytree as pytree
from torchtitan.models.llama3.model.model import TransformerModelArgs, Transformer as titan

import torchax as tx
import torchax.interop
import torchax.train
from torchax.interop import JittableModule, jax_view, torch_view

P = jax.sharding.PartitionSpec

num_global_devices = jax.device_count()
num_local_devices = jax.local_device_count()


def sharded_device_put(tensor: jax.Array, sharding) -> jax.Array:
  if isinstance(tensor, tuple):
    return tuple(sharded_device_put(t, sharding) for t in tensor)

  if num_global_devices == num_local_devices:
    return jax.device_put(tensor, sharding)

  # NOTE: at here, num_global_devices != num_local_devices
  # meaning we are in multi-host setup. Each host will run the same process
  # and each process only need to handle the devices accessible to this host.
  shape = tensor.shape
  x_split = [
    jax.device_put(tensor[i], device)
    for device, i in sharding.addressable_devices_indices_map(shape).items()
  ]
  return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


sharding_map_original = {
  "freqs_cis": (),  #  torch.complex64 (2048, 64)
  "tok_embeddings.weight": ("fsdp", "tp"),  #  torch.float32 (vocab_size, 4096)
  "layers.*.attention.wo.weight": ("fsdp", "tp"),  #  torch.int8 (4096, 4096)
  "layers.*.attention.wq.weight": ("tp", "fsdp"),  #  torch.int8 (4096, 4096)
  "layers.*.attention.wk.weight": ("tp", "fsdp"),  #  torch.int8 (4096, 4096)
  "layers.*.attention.wv.weight": ("tp", "fsdp"),  #  torch.int8 (4096, 4096)
  "layers.*.feed_forward.w1.weight": ("tp", "fsdp"),  #  torch.float32 (11008, 4096)
  "layers.*.feed_forward.w2.weight": ("fsdp", "tp"),  #  torch.float32 (4096, 11008)
  "layers.*.feed_forward.w3.weight": ("tp", "fsdp"),  #  torch.float32 (11008, 4096)
  "layers.*.attention_norm.weight": ("fsdp",),  #  torch.float32 (4096,)
  "layers.*.ffn_norm.weight": ("fsdp",),  #  torch.float32 (4096,)
  "norm.weight": ("fsdp",),  #  torch.float32 (4096,)
  "output.weight": ("tp", "fsdp"),  #  torch.float32 (vocab_size, 4096)
}

sharding_map_scan = {
  "freqs_cis": (),  #  torch.complex64 (2048, 64)
  "tok_embeddings.weight": (),  #  torch.float32 (vocab_size, 4096)
  "layers.params.attention___wo___weight": (
    None,
    "fsdp",
    "tp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wq___weight": (
    None,
    "tp",
    "fsdp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wk___weight": (
    None,
    "tp",
    "fsdp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wv___weight": (
    None,
    "tp",
    "fsdp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.feed_forward___w1___weight": (
    None,
    "tp",
    "fsdp",
  ),  #  torch.float32 (n, 11008, 4096)
  "layers.params.feed_forward___w2___weight": (
    None,
    "fsdp",
    "tp",
  ),  #  torch.float32 (n, 4096, 11008)
  "layers.params.feed_forward___w3___weight": (
    None,
    "tp",
    "fsdp",
  ),  #  torch.float32 (n, 11008, 4096)
  "layers.params.attention_norm___weight": (
    None,
  ),  #  torch.float32 (n, 4096,)
  "layers.params.ffn_norm___weight": (
    None,
  ),  #  torch.float32 (n, 4096,)
  "norm.weight": (),  #  torch.float32 (4096,)
  "output.weight": (),  #  torch.float32 (vocab_size, 4096)
}

sharding_map_scan_fsdp = {
  "freqs_cis": (),  #  torch.complex64 (2048, 64)
  "tok_embeddings.weight": (),  #  torch.float32 (vocab_size, 4096)
  "layers.params.attention___wo___weight": (
    None,
    "fsdp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wq___weight": (
    None,
    "fsdp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wk___weight": (
    None,
    "fsdp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wv___weight": (
    None,
    "fsdp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.feed_forward___w1___weight": (
    None,
    "fsdp",
  ),  #  torch.float32 (n, 11008, 4096)
  "layers.params.feed_forward___w2___weight": (
    None,
    "fsdp",
  ),  #  torch.float32 (n, 4096, 11008)
  "layers.params.feed_forward___w3___weight": (
    None,
    "fsdp",
  ),  #  torch.float32 (n, 11008, 4096)
  "layers.params.attention_norm___weight": (
    None,
  ),  #  torch.float32 (n, 4096,)
  "layers.params.ffn_norm___weight": (
    None,
  ),  #  torch.float32 (n, 4096,)
  "norm.weight": (),  #  torch.float32 (4096,)
  "output.weight": (),  #  torch.float32 (vocab_size, 4096)
}


class Trainer:
  def __init__(self, mesh):
    self.mesh = mesh
    self.x_sharding = jax.sharding.NamedSharding(self.mesh, P("fsdp"))
    self.replicated = jax.sharding.NamedSharding(self.mesh, P())

  def fit(self, model, loss_fn, data_loader, train_steps=20):
    xla_env = torchax.default_env()
    jax.config.update("jax_enable_x64", False)

    jittable_mod = JittableModule(model)

    def model_fn(weights, buffers, args):
      return jittable_mod.functional_call("forward", weights, buffers, args)

    # Use Adam Optimizer to match PyTorch TPU setup
    jax_optimizer = optax.adam(1e-4)
    opt_state = torch_view(jax_optimizer.init(jax_view(jittable_mod.params)))

    train_step = torchax.train.make_train_step(
      model_fn,
      loss_fn,
      jax_optimizer,
      remat_policy=jax.checkpoint_policies.nothing_saveable,
    )

    # Metrics tracking
    total_tokens_after_warmup = 0
    total_time_after_warmup = 0
    avg_throughput = 0.0
    warmup_steps = 5

    print("Begining training")
    s = time.perf_counter()
    jax.profiler.start_trace("/tmp/tensorboard")
    print("start training")
    min_loop_time = 10000
    for i, item in enumerate(data_loader):
      inputs, labels = item

      current_batch_size, current_seq_len = inputs.shape[0], inputs.shape[1]
      tokens_this_step = current_batch_size * current_seq_len * num_global_devices

      # Move them to jax device
      inputs = inputs.to("jax")
      labels = labels.to("jax")

      # Shard them on batch dim for fsdp
      inputs.apply_jax_(sharded_device_put, self.x_sharding)
      labels.apply_jax_(sharded_device_put, self.x_sharding)

      if i == 0:
        train_step = helper.compile_step_func(
          train_step,
          jittable_mod.params,
          jittable_mod.buffers,
          opt_state,
          inputs,
          labels,
          self.mesh,
        )

      print("INPUT shape", inputs.shape)
      step_start = time.perf_counter()
      loss, jittable_mod.params, opt_state = train_step(
        jittable_mod.params, jittable_mod.buffers, opt_state, inputs, labels
      )
      # wait for iteration to finish to measure time
      torchax.interop.call_jax(jax.block_until_ready, (loss, jittable_mod.params))
      step_end = time.perf_counter()
      loop_time = step_end - step_start
      current_throughput = tokens_this_step / loop_time
      print(
        f"Step {i + 1}/{train_steps} | Loss: {loss.item():.4f} | Step Time: {loop_time:.4f}s | Throughput:"
        f" {current_throughput:.2f} tokens/s (total), {current_throughput/num_global_devices:.2f} tokens/s/chip",
      )
      min_loop_time = min(min_loop_time, loop_time)

      if i >= warmup_steps:
        total_tokens_after_warmup += tokens_this_step
        total_time_after_warmup += loop_time
        avg_throughput = total_tokens_after_warmup / total_time_after_warmup

      if i >= train_steps - 1:
        break
    jax.profiler.stop_trace()

    print(f"\n==== Training Completed ====")
    print(f"Average Throughput (total): {avg_throughput:.2f} tokens/s")
    print(f"Average Throughput (per chip): {avg_throughput/num_global_devices:.2f} tokens/s/chip")

    return min_loop_time


def _process_sharding_name(name):
  def is_integer(t):
    try:
      int(t)
      return True
    except:  # noqa: E722
      return False

  tokens = name.split(".")
  for i, t in enumerate(tokens):
    if is_integer(t):
      tokens[i] = "*"
  return ".".join(tokens)


def _make_weight_shard(weight_meta, slice_index):
  shard_meta = weight_meta[slice_index]
  seed = hash(tuple((s.start, s.stop, s.step) for s in slice_index)) % (2**31 - 1)
  key = jax.random.PRNGKey(seed)
  dtype_map = {
    torch.bfloat16: jnp.bfloat16,
    torch.float16: jnp.float16,
    torch.float32: jnp.float32,
    torch.complex64: jnp.complex64,
    torch.complex128: jnp.complex128,
  }
  jax_dtype = dtype_map.get(shard_meta.dtype, jnp.bfloat16)
  return jax.random.normal(key, shard_meta.shape, dtype=jax_dtype)


def create_sharded_weights(model, mesh, sharding_map):
  res = {}
  env = torchax.default_env()
  for name, weight_meta in model.state_dict().items():
    sharding_spec = sharding_map.get(_process_sharding_name(name))
    if sharding_spec is None:
      print("Skipping weight:", name)
      continue
    sharding = NamedSharding(mesh, P(*sharding_spec))
    print(
      f"Initializing weight {name} w shape={weight_meta.shape} dtype={weight_meta.dtype} sharding={sharding}...."
    )
    res[name] = env.j2t_iso(
      jax.make_array_from_callback(
        weight_meta.shape,
        sharding,
        functools.partial(_make_weight_shard, weight_meta),
      )
    )
  return res


def fake_dataloader(size, seqlen, batch_size):
  for _ in range(size):
    x = torch.randint(0, 32000, (batch_size, seqlen), device="cpu")
    yield x, (x + 1) % 32000


def main(
  model_type="1B",
  batch_size=8,
  seqlen=2048,
  override_num_layers=-1,
  use_scan=True,
  tp_parallelism=1,
  train_steps=20,
  tpu_num_slices=1,
):
  torchax.enable_globally()
  torchax.enable_performance_mode()

  print(f"Running with parameters {locals()}", flush=True)

  num_hosts = jax.process_count()

  print(
    f"Global Devices: {num_global_devices}, Local Devices: {num_local_devices}, Hosts: {num_hosts}, Slices: {tpu_num_slices}",
    flush=True,
  )

  fsdp = num_global_devices // tp_parallelism

  print(f"Using FSDP parallelism: {fsdp}, TP parallelism: {tp_parallelism}", flush=True)

  if tpu_num_slices == 1:
    mesh = Mesh(np.array(jax.devices()).reshape(fsdp, tp_parallelism), ("fsdp", "tp"))
  else:
    dev_array = jax.experimental.mesh_utils.create_hybrid_device_mesh(
      (fsdp // tpu_num_slices, tp_parallelism),
      (tpu_num_slices, 1),
      jax.devices(),
      process_is_granule=False,
      allow_split_physical_axes=True,
    )
    mesh = Mesh(dev_array, ("fsdp", "tp"))

  print(f"Using mesh {mesh=}", flush=True)

  if use_scan:
    if tp_parallelism > 1:
      sharding_map = sharding_map_scan
    else:
      sharding_map = sharding_map_scan_fsdp
  else:
    sharding_map = sharding_map_original

  env = torchax.default_env()

  # Manually construct TransformerModelArgs for Llama 3.2 1B
  config = TransformerModelArgs(
      dim=2048,
      n_layers=16,
      n_heads=32,
      n_kv_heads=8,
      vocab_size=128256,
      multiple_of=256,
      ffn_dim_multiplier=1.5,
      norm_eps=1e-5,
      rope_theta=500000.0,
      max_seq_len=seqlen,
  )

  torch.set_default_dtype(torch.bfloat16)
  with torch.device("meta"):
    gpt = titan(config)
  print(
    f"Model initialized with {sum(p.numel() for p in gpt.parameters()) / 1e9:.2f} B parameters."
  )

  with torch.device("cpu"):
    freqs_cis = gpt._precompute_freqs_cis()

  if use_scan:
    checkpoint_policy = jax.checkpoint_policies.nothing_saveable
    gpt = TransfomerWithScan(gpt, checkpoint_policy)

  state_dict = dict(gpt.state_dict())
  state_dict.pop("freqs_cis")  # dont shard freqs_cis
  state_dict = create_sharded_weights(gpt, mesh, sharding_map)
  replicated = jax.sharding.NamedSharding(mesh, P())

  state_dict["freqs_cis"] = freqs_cis.to("jax").apply_jax(jax.device_put, replicated)
  gpt.load_state_dict(state_dict, assign=True)

  train_loader = fake_dataloader(train_steps, seqlen, batch_size)

  # NOTE: overriding attention to capture mesh and sharding info
  partition = P("fsdp", "tp", None, None)
  attention = functools.partial(splash_attn.tpu_splash_attention, mesh, partition, True)
  attention = jax.jit(attention)

  def custom_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
  ):
    jk, jq, jv = jax_view((query, key, value))
    res = attention(jk, jq, jv, None)
    return torch_view(res)

  env.override_op_definition(
    torch.nn.functional.scaled_dot_product_attention, custom_attention
  )

  def loss_fn(logits, y):
    num_tokens = logits.shape[-1]
    logits = logits.reshape(-1, num_tokens)
    y = y.reshape(-1)
    return torch.nn.functional.cross_entropy(logits, y)

  with jax.set_mesh(mesh):
    trainer = Trainer(mesh)
    return trainer.fit(gpt, loss_fn, train_loader, train_steps=train_steps)


class TransfomerWithScan(torch.nn.Module):
  def __init__(self, old_transformer, checkpoint_policy):
    super().__init__()
    self.tok_embeddings = old_transformer.tok_embeddings
    self.norm = old_transformer.norm
    self.output = old_transformer.output
    self.layers = torchax.train.ScannedModule(
      list(old_transformer.layers.values()), checkpoint_policy
    )

    self.register_buffer("freqs_cis", old_transformer.freqs_cis)

  def forward(self, tokens: torch.Tensor):
    h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
    h = self.layers(h, self.freqs_cis, None)
    h = self.norm(h) if self.norm else h
    output = self.output(h) if self.output else h
    return output


if __name__ == "__main__":
  import fire

  fire.Fire(main)
