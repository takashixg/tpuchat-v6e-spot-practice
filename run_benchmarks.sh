#!/bin/bash
# Script to run Llama 3.2 1B training benchmarks on TPU v6e-8 (building torch_tpu from source)
set -ex

# 1. Create a fresh Python 3.12 virtual environment (Python 3.12 is already installed!)
rm -rf ~/venv
python3.12 -m venv ~/venv

# 2. Upgrade pip using virtualenv pip
~/venv/bin/pip install --upgrade pip

# 3. Install JAX and XLA dependencies
~/venv/bin/pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
~/venv/bin/pip install optax fire flax triton portpicker keyrings.google-artifactregistry-auth transformers datasets

# 4. Install PyTorch CPU (MUST BE 2.10.0+cpu to match torch_tpu's Bazel build requirements!)
~/venv/bin/pip install torch==2.10.0+cpu torchvision==0.25.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 5. Build torch_tpu from source using Bazel (Bazel is already installed!)
cd ~/torch_tpu
bazel build -c opt //ci/wheel:torch_tpu_wheel --config=no_rbe
~/venv/bin/pip install bazel-bin/ci/wheel/*.whl
cd ~

# 6. Install torchtitan 0.2.0
~/venv/bin/pip install torchtitan==0.2.0

# 7. Install torchax from source
cd ~/torchax
~/venv/bin/pip install -e .
cd ~

# 8. Run TorchAX benchmark (Disable PyTorch auto-loading to prevent conflict with torch_tpu!)
echo "=== Starting TorchAX Benchmark ==="
cp ~/benchmark_torchax.py ~/torchax/examples/train_llama_torchtitan/benchmark_torchax.py
cd ~/torchax/examples/train_llama_torchtitan
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ~/venv/bin/python3 benchmark_torchax.py --model_type=1B --batch_size=8 --seqlen=2048 --train_steps=20 > ~/torchax_benchmark.log 2>&1
cd ~
echo "=== TorchAX Benchmark Completed ==="

# 9. Run TorchTPU (Compiled) benchmark
echo "=== Starting TorchTPU (Compiled) Benchmark ==="
eval $(~/venv/bin/python3 -m torch_tpu._internal.distributed.launchers.singlehost_wrapper)
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE
export MASTER_PORT=$(~/venv/bin/python3 -c 'import portpicker; print(portpicker.pick_unused_port())')

# Explicitly enable backend autoloading for TorchTPU
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
~/venv/bin/python3 -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=$WORLD_SIZE ~/benchmark_torch_tpu.py > ~/torch_tpu_compiled_benchmark.log 2>&1
echo "=== TorchTPU (Compiled) Benchmark Completed ==="

# 10. Run TorchTPU (Eager) benchmark
echo "=== Starting TorchTPU (Eager) Benchmark ==="
# Modify script to disable compilation
sed -i 's/MODEL_COMPILE = True/MODEL_COMPILE = False/g' ~/benchmark_torch_tpu.py
~/venv/bin/python3 -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=$WORLD_SIZE ~/benchmark_torch_tpu.py > ~/torch_tpu_eager_benchmark.log 2>&1
echo "=== TorchTPU (Eager) Benchmark Completed ==="

echo "=== All Benchmarks Completed Successfully ==="
