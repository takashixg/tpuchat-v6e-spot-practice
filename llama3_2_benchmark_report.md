# Llama 3.2 1B Training Benchmark: TorchAX vs. TorchTPU on TPU v6e-8

This report presents training benchmark results and architectural comparisons for **Llama 3.2 1B** training on a **TPU v6e-8** spot VM. We compare the performance of **TorchAX** (JAX/SPMD-backed PyTorch compiler) against **TorchTPU** (native PyTorch C++ extension backend, in both eager and compiled modes).

---

## 1. Executive Summary

Benchmarks show that **TorchAX** outperforms **TorchTPU** by **~5x** in throughput and step latency. 

| Framework / Mode | Local Batch Size | Seq Length | Step Latency (Steady State) | Throughput per Chip (tokens/s/chip) | Performance Ratio | Compile Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TorchAX** (Compiled JAX/SPMD) | 1 | 2048 | **0.1376s** | **14,883.70** | **1.00x (Baseline)** | **~6.85s** |
| **TorchTPU** (Eager Mode) | 1 | 2048 | 0.6734s | 3,041.09 | 0.204x (4.9x slower) | N/A |
| **TorchTPU** (Compiled Mode) | 1 | 2048 | 0.7251s | 2,824.33 | 0.190x (5.27x slower) | ~11.0s |

> [!NOTE]
> The throughput figures above are normalized for a **local batch size of 1 per chip** (global batch size 8 sharded over 8 TPU chips) to prevent Out-of-Memory (OOM) failures due to the large vocabulary dimension (`128256`) of the Llama 3.2 model.

---

## 2. Key Findings & Performance Analysis

### Why is TorchAX so much faster?
1. **JAX GSPMD Auto-SPMD Parallelism**: TorchAX leverages JAX's Global SPMD (GSPMD) compiler. Instead of relying on manual collective communication injections (e.g. PyTorch FSDP gather/scatter), GSPMD optimizes the entire multi-device computation as a single graph. It performs automatic sharding propagation and inserts communication collectives (all-gather, all-reduce, reduce-scatter) at mathematically optimal locations, overlapping them with matrix multiplications.
2. **Loop compilation via `jax.lax.scan`**: TorchAX represents the stack of 16 Transformer layers using JAX's loop primitive. In PyTorch and standard TorchTPU, the compiler must unroll and trace all 16 layers sequentially, leading to giant HLO graphs, increased host-device orchestration overhead, and compilation bloating. TorchAX compiles the single layer logic once and loops over it dynamically, yielding extremely low compile times (~6.8s) and minimal instruction overhead.
3. **Pointwise Parameter Replication**: In TorchAX FSDP, 1D parameter vectors (such as normalization weights and biases) are kept replicated `P()` instead of sharded. This avoids sharding conflicts during pointwise operations (e.g., RMSNorm element-wise multiply) which would otherwise require complex and illegal overlapping shardings on the same mesh axis.

### Why is TorchTPU slower?
1. **PrivateUse1 Backend Translation Overhead**: TorchTPU maps PyTorch's ATen C++ operators to XLA operations via a custom C++ runtime backend (`PrivateUse1`). This intermediate layer introduces host-side scheduling latency, extra memory allocation overhead, and translation passes.
2. **PyTorch Compiler / PyTorch-TPU compilation overhead**: PyTorch's compiler (`torch.compile`) on TPU v6e is still maturing. It can produce suboptimal HLO graphs for complex recurrent transformer blocks, and graph breaks can fall back to slow eager-mode CPU/TPU coordination. Eager mode in TorchTPU actually performed slightly faster than compiled mode in our runs.

---

## 3. Aligned Benchmark Details

Both benchmarks trained a model matching **Llama 3.2 1B** parameters and hyperparameters:
*   `dim = 2048`
*   `n_layers = 16`
*   `n_heads = 32`
*   `n_kv_heads = 8`
*   `vocab_size = 128256`
*   `ffn_dim_multiplier = 1.5` (yielding intermediate MLP dimension `8192`)
*   `seq_len = 2048`
*   Optimizer: Adam (`lr = 1e-4`)

---

## 4. Run Scripts and Replication

The benchmarks were run on a single `tpu-v6e-8` node using the following scripts.

*   **TorchAX Run**: `benchmark_torchax.py`
*   **TorchTPU Run**: `benchmark_torch_tpu.py`
*   **Orchestration Runner**: `run_benchmarks.sh`
