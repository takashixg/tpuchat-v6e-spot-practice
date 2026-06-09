# Llama 3.2 1B Training Benchmark: TorchTPU vs TorchAX on Cloud TPU v6e-8

This report summarizes the benchmark results for training the Llama 3.2 1B model on a single host Cloud TPU v6e-8 (8 TPU chips) using **TorchTPU** and **TorchAX**.

## Benchmark Setup
* **Model**: Llama 3.2 1B
  * Vocab Size: 128,256
  * Hidden Size: 2,048
  * FFN Intermediate Size: 8,192
  * Number of Layers: 16
  * Attention Heads / KV Heads: 32 / 8
* **Sequence Length**: 2,048
* **Global Batch Size**: 8 (Local batch size of 1 per chip sharded via FSDP)
* **Precision**: Bfloat16 (`bfloat16`)
* **Hardware**: Cloud TPU v6e-8 (8 chips, 27GB HBM per chip)
* **Warmup Steps**: 5
* **Total Training Steps**: 20
* **Optimizer**: Adam (`lr=1e-4`)

---

## Benchmark Results Summary

| Framework / Mode | Step Time (after warmup) | Throughput per Chip (Actual) | Throughput per Chip (Reported) | Step 1 Time (Compile Overhead) |
| :--- | :---: | :---: | :---: | :---: |
| **TorchAX** (JAX-JIT Compiled) | **0.1377s** | **14,872 tokens/s/chip** | 118,976 tokens/s/chip *(Bug)* | 0.1627s |
| **TorchTPU** (Compiled - Entire Model) | **0.6587s** | **3,109 tokens/s/chip** | 3,109 tokens/s/chip | 7.3003s |
| **TorchTPU** (Compiled - Layer-wise) | **0.6744s** | **3,036 tokens/s/chip** | 3,036 tokens/s/chip | 2.9341s |
| **TorchTPU** (Eager FSDP) | **0.6734s** | **3,041 tokens/s/chip** | 3,041 tokens/s/chip | 2.9341s |

---

## Key Findings and Deep Dive

### 1. TorchAX Performance Advantage (~4.8x Speedup)
TorchAX achieves a step time of **0.1377 seconds**, compared to TorchTPU's **0.6587 seconds** in its best compiled configuration. This represents a **~4.8x speedup**. The primary reasons for this performance gap are:
* **Unified Graph JIT**: TorchAX compiles the entire training step (including forward, backward, loss computation, and optimizer updates) into a single JAX-compiled function (`jax.jit`).
* **Optimized Attention Kernel**: TorchAX overrides PyTorch's native attention with `tpu_splash_attention`, an optimized Flash Attention implementation written using JAX Pallas specifically for TPUs.
* **Scan Optimization**: It utilizes `ScannedModule` to scan over layers, which reduces the XLA compilation time and memory usage.

### 2. TorchAX Throughput Double-Counting Bug
A critical bug was discovered in the throughput calculation code of `benchmark_torchax.py`.
```python
current_batch_size = inputs.shape[0] # returns the global batch size (8)
tokens_this_step = current_batch_size * current_seq_len * num_global_devices
```
In JAX/TorchAX, the shape of a sharded tensor represents its **global shape**. Therefore, `inputs.shape[0]` returns the global batch size (8), not the local batch size per device (1).
* By multiplying by `num_global_devices` (8) again, the script calculated the tokens processed in a single step as `8 * 2,048 * 8 = 131,072` instead of the actual `16,384`.
* This led to an 8x over-estimation of both total throughput and per-chip throughput (reporting **118,976 tokens/s/chip** instead of the actual **14,872 tokens/s/chip**).
* The table above includes the corrected (actual) throughput for a fair comparison.

### 3. TorchTPU Compilation Challenges
Native PyTorch compilation (`torch.compile`) on TPU still shows limited performance gains compared to eager FSDP:
* **Entire Model Compile**: Compiling the top-level FSDP model via `torch.compile(model, backend="tpu")` only improves step time from `0.6734s` to `0.6587s` (~2% improvement) while introducing a high initial compilation time of **7.3 seconds**.
* **Layer-wise Compile Mismatch**: 
  * Compiling individual FSDP layers in a loop by instantiating `TpuBackend()` on each call hits PyTorch Dynamo's recompilation limit (default: 8) because Dynamo detects a different backend callable instance for each layer. This triggers a silent fallback to eager execution.
  * Reusing a single `TpuBackend` instance prevents the recompilation warning but does not yield performance improvements (step time remains at `0.6744s`), suggesting that compiling individual FSDP layers does not allow the XLA compiler to optimize across layer boundaries.
