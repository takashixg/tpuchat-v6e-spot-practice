import functools as ft
import itertools as it
import time
import os
import math
import queue
import threading
import json
import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tiktoken
from datasets import load_dataset

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    # Data
    num_shards: int = 50
    hf_repo_id: str = 'vorushin/tpuchat'

    # Model architecture
    n_head: int = 8
    n_kv_head: int = 2
    aspect_ratio: int = 64
    head_dim: int = 128
    vocab_size: int = 32768
    seq_len: int = 2048
    window_pattern: str = 'LLLL'
    softcap: float = 15.0
    attn_impl: str = 'einsum'  # einsum is standard for SPMD here
    splash_block_size: int = 1024

    # SFT Training Hyperparameters
    num_iterations: int = 500
    device_batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_ratio: float = 0.05
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0

    # Eval / Logging
    eval_every: int = 50
    eval_steps: int = 10
    log_every: int = 10
    save_every: int = -1
    sample_every: int = 100

    param_seed: int = 42

    @property
    def n_embd(self):
        return self.n_head * self.head_dim

    @property
    def depth(self):
        return self.n_embd // self.aspect_ratio

    @property
    def n_layer(self):
        return self.depth

# %%
# === Model Definition (matching 02_train.py but self-contained) ===

def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)

def precompute_rope(seq_len, head_dim, base=10000):
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(jnp.bfloat16)
    sin = jnp.sin(freqs).astype(jnp.bfloat16)
    return cos, sin

def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)

def compute_window_sizes(config: Config):
    pattern = config.window_pattern.upper()
    long_w = config.seq_len
    short_w = long_w // 2
    char_to_w = {'L': long_w, 'S': short_w}
    sizes = []
    for i in range(config.n_layer):
        c = pattern[i % len(pattern)]
        sizes.append(char_to_w[c])
    sizes[-1] = long_w
    return sizes

def _expand_kv(k, v, n_head, n_kv_head):
    if n_kv_head == n_head:
        return k, v
    ratio = n_head // n_kv_head
    return jnp.repeat(k, ratio, axis=1), jnp.repeat(v, ratio, axis=1)

def model_apply(config: Config, params: dict, tokens: jax.Array) -> jax.Array:
    B, T = tokens.shape
    n_head = config.n_head
    n_kv_head = config.n_kv_head
    head_dim = config.head_dim
    n_layer = config.n_layer
    window_sizes = compute_window_sizes(config)

    cos = params['rope_cos'][:T][None, None, :, :]
    sin = params['rope_sin'][:T][None, None, :, :]

    x = params['wte'][tokens]
    x = rms_norm(x)
    x0 = x

    for i in range(n_layer):
        layer = params['layers'][i]

        h = rms_norm(x)

        q = jnp.einsum('btd,dhk->bhtk', h, layer['c_q'])
        k = jnp.einsum('btd,dhk->bhtk', h, layer['c_k'])
        v = jnp.einsum('btd,dhk->bhtk', h, layer['c_v'])

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = rms_norm(q)
        k = rms_norm(k)

        w = window_sizes[i]

        if config.attn_impl == 'einsum':
            k_exp, v_exp = _expand_kv(k, v, n_head, n_kv_head)
            scale = head_dim ** -0.5
            scores = jnp.einsum('bhtd,bhsd->bhts', q, k_exp) * scale
            rows = jnp.arange(T)[:, None]
            cols = jnp.arange(T)[None, :]
            if w < T:
                mask = (cols <= rows) & (cols >= rows - w + 1)
            else:
                mask = cols <= rows
            scores = jnp.where(mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min)
            attn_weights = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum('bhts,bhsd->bhtd', attn_weights, v_exp)
        else:
            raise ValueError(f"attn_impl {config.attn_impl} not fully supported in SFT script yet.")

        attn_out = jnp.einsum('bhtd,hde->bte', attn_out, layer['c_proj'])

        x = params['resid_lambdas'][i] * x + params['x0_lambdas'][i] * x0
        x = x + attn_out

        h2 = rms_norm(x)
        mlp_out = jnp.einsum('btd,dh->bth', h2, layer['c_fc'])
        mlp_out = jax.nn.relu(mlp_out) ** 2
        mlp_out = jnp.einsum('bth,hd->btd', mlp_out, layer['mlp_proj'])
        x = x + mlp_out

    x = rms_norm(x)
    logits = jnp.einsum('btd,dv->btv', x, params['lm_head'], preferred_element_type=jnp.float32)
    logits = config.softcap * jnp.tanh(logits / config.softcap)
    return logits

# %%
# === Optimizer (AdamW) ===

def init_adam_state(param: jax.Array) -> dict:
    return {
        'mu': jnp.zeros_like(param),
        'nu': jnp.zeros_like(param),
        'count': jnp.array(0, dtype=jnp.int32),
    }

def adamw_step(config, lr_mult, param, grad, state):
    new_count = state['count'] + 1
    new_mu = config.beta1 * state['mu'] + (1 - config.beta1) * grad
    new_nu = config.beta2 * state['nu'] + (1 - config.beta2) * grad ** 2

    mu_hat = new_mu / (1 - config.beta1 ** new_count)
    nu_hat = new_nu / (1 - config.beta2 ** new_count)

    lr = config.learning_rate * lr_mult
    update = mu_hat / (jnp.sqrt(nu_hat) + config.eps)

    wd = jnp.where(param.ndim >= 2, config.weight_decay, 0.0)
    new_param = param - lr * (update + wd * param)

    new_state = {'mu': new_mu, 'nu': new_nu, 'count': new_count}
    return new_param, new_state

def get_lr_multiplier(step, num_iterations, config: Config):
    warmup_iters = int(config.warmup_ratio * num_iterations)
    warmdown_iters = int(config.warmdown_ratio * num_iterations)

    if step < warmup_iters:
        return (step + 1) / max(warmup_iters, 1)
    elif step <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - step) / max(warmdown_iters, 1)
        return progress * 1.0 + (1 - progress) * config.final_lr_frac

# %%
# === Checkpoint Loading ===

@jax.tree_util.register_pytree_with_keys_class
class dot_dict(dict):
    __setattr__ = dict.__setitem__
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def tree_flatten_with_keys(self):
        keys = tuple(sorted(self))
        return tuple((jax.tree_util.DictKey(k), self[k]) for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))

def load_param_state(checkpoint_dir: str) -> dict:
    with open(os.path.join(checkpoint_dir, 'params.pkl'), 'rb') as f:
        params_np = pickle.load(f)
    
    # Convert numpy dictionaries recursively to JAX arrays
    def to_jax(x):
        if isinstance(x, dict):
            return {k: to_jax(v) for k, v in x.items()}
        elif isinstance(x, np.ndarray):
            return jnp.array(x)
        else:
            return x
    return to_jax(params_np)

# %%
# === Dataloader & SFT Tokenization ===

def tokenize_sft_dataset(dataset, enc, batch_size, seq_len):
    bos_id = enc.encode_single_token('<|bos|>')
    user_start_id = enc.encode_single_token('<|user_start|>')
    user_end_id = enc.encode_single_token('<|user_end|>')
    assistant_start_id = enc.encode_single_token('<|assistant_start|>')
    assistant_end_id = enc.encode_single_token('<|assistant_end|>')

    token_buf = []
    mask_buf = []

    while True:
        for row in dataset:
            messages = row['messages']
            tokens = [bos_id]
            mask = [0]

            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    msg_tokens = [user_start_id] + enc.encode_ordinary(content) + [user_end_id]
                    tokens.extend(msg_tokens)
                    mask.extend([0] * len(msg_tokens))
                elif role == 'assistant':
                    msg_tokens = [assistant_start_id] + enc.encode_ordinary(content) + [assistant_end_id]
                    tokens.extend(msg_tokens)
                    mask.extend([1] * len(msg_tokens))

            token_buf.extend(tokens)
            mask_buf.extend(mask)

            tokens_per_batch = batch_size * (seq_len + 1)
            while len(token_buf) >= tokens_per_batch:
                tb = np.array(token_buf[:tokens_per_batch], dtype=np.int32).reshape(batch_size, seq_len + 1)
                mb = np.array(mask_buf[:tokens_per_batch], dtype=np.float32).reshape(batch_size, seq_len + 1)

                x = tb[:, :-1]
                y = tb[:, 1:]
                m = mb[:, 1:]

                token_buf = token_buf[tokens_per_batch:]
                mask_buf = mask_buf[tokens_per_batch:]

                yield x, y, m

@dataclass
class PrefetchDataLoader:
    iterator: any
    data_sharding: NamedSharding
    capacity: int = 2

    def __post_init__(self):
        self.queue = queue.Queue(maxsize=self.capacity)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            for item in self.iterator:
                if self.stop_event.is_set():
                    break
                x, y, mask = item
                x_dev = jax.device_put(jnp.array(x), self.data_sharding)
                y_dev = jax.device_put(jnp.array(y), self.data_sharding)
                mask_dev = jax.device_put(jnp.array(mask), self.data_sharding)
                self.queue.put((x_dev, y_dev, mask_dev))
        except Exception as e:
            print(f"Prefetch worker error: {e}")
            self.stop_event.set()
        finally:
            self.stop_event.set()

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_event.is_set() and self.queue.empty():
            raise StopIteration
        return self.queue.get()

    def stop(self):
        self.stop_event.set()

# %%
# === Trainable Params Helper ===

def split_trainable(params):
    trainable = {}
    static = {}
    for k, v in params.items():
        if k in ('rope_cos', 'rope_sin'):
            static[k] = v
        else:
            trainable[k] = v
    return trainable, static

def merge_params(trainable, static):
    merged = {}
    merged.update(trainable)
    merged.update(static)
    return merged

# %%
# === Main Executable ===

def main():
    print(f"TPUs available: {len(jax.devices())} devices")
    
    # Define distributed Mesh (SPMD)
    mesh = Mesh(jax.devices(), ('batch',))
    data_sharding = NamedSharding(mesh, P('batch', None))

    # Initialize Config
    config = Config()

    # Load tokenizer
    import pickle
    TOKENIZER_DIR = './content/tokenizer'
    with open(os.path.join(TOKENIZER_DIR, 'tokenizer', 'tokenizer.pkl'), 'rb') as f:
        enc = pickle.load(f)
    print(f"Loaded tokenizer with vocab size: {enc.n_vocab}")

    # Load CPT Checkpoint
    CHECKPOINT_DIR = './content/checkpoint'
    print(f"Loading pretrained weights from: {CHECKPOINT_DIR}")
    params = load_param_state(CHECKPOINT_DIR)
    print("Pretrained weights loaded successfully.")

    # SFT Dataset Setup
    print("Downloading SmolTalk dataset...")
    raw_dataset = load_dataset("HuggingFaceTB/smoltalk", name="everyday-conversations", split="train")
    val_dataset = load_dataset("HuggingFaceTB/smoltalk", name="everyday-conversations", split="test")
    print(f"SmolTalk everyday-conversations subset: train={len(raw_dataset)}, test={len(val_dataset)}")

    total_batch_size = config.device_batch_size * len(jax.devices())
    print(f"Total SFT Batch Size: {total_batch_size} ({config.device_batch_size} per device)")

    # Data loaders
    raw_train_loader = tokenize_sft_dataset(raw_dataset, enc, total_batch_size, config.seq_len)
    train_loader = PrefetchDataLoader(raw_train_loader, data_sharding, capacity=4)
    
    val_loader_fn = lambda: tokenize_sft_dataset(val_dataset, enc, total_batch_size, config.seq_len)

    # Initialize Optimizer
    trainable_params, static_params = split_trainable(params)
    opt_state = jax.tree.map(init_adam_state, trainable_params)
    print("Optimizer state initialized.")

    # JIT Compiled steps
    @jax.jit
    def train_step(config: Config, params: dict, opt_state: dict,
                   x: jax.Array, y: jax.Array, mask: jax.Array, lr_mult: jax.Array):
        trainable, static = split_trainable(params)

        def loss_fn(trainable_params):
            full_params = merge_params(trainable_params, static)
            logits = model_apply(config, full_params, x)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
            masked_ce = ce * mask
            loss = jnp.sum(masked_ce) / jnp.maximum(jnp.sum(mask), 1e-5)
            return loss

        with jax.named_scope('forward_backward'):
            loss, grads = jax.value_and_grad(loss_fn)(trainable)

        # Apply AdamW updates
        with jax.named_scope('optimizer'):
            is_opt_leaf = lambda x: isinstance(x, dict) and 'mu' in x
            t_leaves, t_treedef = jax.tree.flatten(trainable)
            g_leaves, _ = jax.tree.flatten(grads)
            o_leaves, o_treedef = jax.tree.flatten(opt_state, is_leaf=is_opt_leaf)

            new_t_leaves, new_o_leaves = [], []
            for p, g, s in zip(t_leaves, g_leaves, o_leaves):
                new_p, new_s = adamw_step(config, lr_mult, p, g, s)
                new_t_leaves.append(new_p)
                new_o_leaves.append(new_s)

            new_trainable = t_treedef.unflatten(new_t_leaves)
            new_opt_state = o_treedef.unflatten(new_o_leaves)
            new_params = merge_params(new_trainable, static)

        return loss, new_params, new_opt_state

    @jax.jit
    def eval_step(config: Config, params: dict, x: jax.Array, y: jax.Array, mask: jax.Array):
        logits = model_apply(config, params, x)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        masked_ce = ce * mask
        return jnp.sum(masked_ce) / jnp.maximum(jnp.sum(mask), 1e-5)

    @jax.jit
    def predict_step(config: Config, params: dict, x: jax.Array):
        return model_apply(config, params, x)

    # Chat Inference Helper
    def generate_chat(config, params, enc, prompt, max_new_tokens=100, seed=None):
        if seed is None:
            seed = 42
        key = jax.random.key(seed)

        bos_id = enc.encode_single_token('<|bos|>')
        user_start_id = enc.encode_single_token('<|user_start|>')
        user_end_id = enc.encode_single_token('<|user_end|>')
        assistant_start_id = enc.encode_single_token('<|assistant_start|>')
        assistant_end_id = enc.encode_single_token('<|assistant_end|>')

        # Format prompt with special ChatML tokens
        prompt_ids = [bos_id, user_start_id] + enc.encode_ordinary(prompt) + [user_end_id, assistant_start_id]
        ids = list(prompt_ids)

        for _ in range(max_new_tokens):
            context = ids[-config.seq_len:]
            pad_len = config.seq_len - len(context)
            padded_context = context + [0] * pad_len
            x_input = jnp.array([padded_context], dtype=jnp.int32)
            
            logits = predict_step(config, params, x_input)
            logits.block_until_ready()
            next_logits = logits[0, len(context) - 1, :]
            next_logits = next_logits.astype(jnp.float32)

            # Greedy decoding for stable eval
            next_id = int(jnp.argmax(next_logits))

            ids.append(next_id)
            if next_id == assistant_end_id:
                break

        return enc.decode(ids)

    # Prompts for Chat SFT Evaluation
    SFT_PROMPTS = [
        "Explain what gravity is.",
        "Write a python function to compute factorial.",
        "Hello, who are you?",
    ]

    # Training loop
    print(f"\n=== Starting SFT Fine-Tuning for {config.num_iterations} steps ===\n")
    smooth_loss = 0.0
    best_val_loss = float('inf')
    
    for step in range(config.num_iterations + 1):
        last_step = (step == config.num_iterations)

        # === SFT Evaluation ===
        if config.eval_every > 0 and (last_step or step % config.eval_every == 0):
            val_loader = val_loader_fn()
            val_losses = []
            for ei in range(config.eval_steps):
                vx, vy, vm = next(val_loader)
                # Transfer to devices
                vx = jax.device_put(jnp.array(vx), data_sharding)
                vy = jax.device_put(jnp.array(vy), data_sharding)
                vm = jax.device_put(jnp.array(vm), data_sharding)
                vl = eval_step(config, params, vx, vy, vm)
                val_losses.append(float(vl))
            avg_val_loss = sum(val_losses) / len(val_losses)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            print(f'SFT Step {step:05d} | Val loss: {avg_val_loss:.4f} (best: {best_val_loss:.4f})')

        # === SFT Inference Sampling ===
        if config.sample_every > 0 and step > 0 and (last_step or step % config.sample_every == 0):
            print(f"\n--- SFT Chat Samples (step {step}) ---")
            for prompt in SFT_PROMPTS:
                sample_text = generate_chat(config, params, enc, prompt, max_new_tokens=80)
                print(f"User: {prompt}\nResponse: {sample_text}\n")
            print("--------------------------------------")

        if last_step:
            break

        # === SFT Train step ===
        lr_mult = jnp.array(get_lr_multiplier(step, config.num_iterations, config), dtype=jnp.float32)
        t0 = time.time()

        x_batch, y_batch, mask_batch = next(train_loader)
        loss, params, opt_state = train_step(config, params, opt_state, x_batch, y_batch, mask_batch, lr_mult)
        loss.block_until_ready()
        dt = time.time() - t0

        loss_val = float(loss)
        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

        if step % config.log_every == 0:
            pct = 100 * step / config.num_iterations
            print(f'SFT step {step:05d}/{config.num_iterations:05d} ({pct:.1f}%) | loss: {debiased_loss:.4f} | lr_mult: {float(lr_mult):.3f} | dt: {dt*1000:.0f}ms')

    print("\nSFT Training completed.")

    # Save final SFT checkpoint
    print("Saving final SFT checkpoint...")
    SFT_CHECKPOINT_DIR = './content/checkpoint_sft'
    os.makedirs(SFT_CHECKPOINT_DIR, exist_ok=True)
    params_np = jax.tree.map(lambda x: np.array(x) if isinstance(x, jax.Array) else x, params)
    with open(os.path.join(SFT_CHECKPOINT_DIR, 'params.pkl'), 'wb') as f:
        pickle.dump(params_np, f)
    
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    with open(os.path.join(SFT_CHECKPOINT_DIR, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"SFT checkpoint saved to: {SFT_CHECKPOINT_DIR}")

    # Generate final report summary info to a file
    with open('./sft_results.json', 'w') as f:
        json.dump({
            "sft_final_val_loss": best_val_loss,
            "sft_iterations": config.num_iterations
        }, f, indent=2)

if __name__ == '__main__':
    main()
