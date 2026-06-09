#!/usr/bin/env bash
set -e

# ==============================================================================
# TPU v6e-8 E2E CPT & SFT 1-Cycle Verification & Comparison Orchestration Script
# ==============================================================================

PROJECT_ID="takashix-tpu"
ZONE="asia-northeast1-b"
TPU_NAME="tpuchat-v6e-8x-sft-e2e"
HF_TOKEN="${HF_TOKEN:-your_hf_token_here}"

echo "===================================================================="
echo "🚀 Step 1: Provisioning Trillium v6e-8 Spot Instance"
echo "===================================================================="
if ! gcloud alpha compute tpus tpu-vm describe "${TPU_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" 2>/dev/null; then
    echo "Creating new TPU v6e-8 spot instance..."
    gcloud alpha compute tpus tpu-vm create "${TPU_NAME}" \
        --project="${PROJECT_ID}" --zone="${ZONE}" \
        --accelerator-type="v6e-8" --version="v2-alpha-tpuv6e" --spot
else
    echo "TPU Instance ${TPU_NAME} is already running."
fi

echo "===================================================================="
echo "🧪 Step 2: Running E2E CPT (Pretraining) on TPU VM"
echo "===================================================================="
# Run CPT on TPU VM
gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    --command="
        set -e
        rm -rf tpuchat_e2e
        git clone https://github.com/vorushin/tpuchat.git tpuchat_e2e
        cd tpuchat_e2e
        
        # Install basic packages + datasets
        pip install -q 'jax[tpu]' optax huggingface_hub tiktoken pyarrow requests torch tensorboard tensorboard-plugin-profile matplotlib datasets
        
        # Apply JAX SPMD Distributed patch to 02_train.py
        cp 02_train.py 02_train_distributed.py
        sed -i 's/\/content/.\/content/g' 02_train_distributed.py
        sed -i 's/^[[:space:]]*%/# %/g' 02_train_distributed.py
        sed -i 's/from google.colab import userdata/# from google.colab import userdata/g' 02_train_distributed.py
        sed -i 's/userdata.get(\"HF_TOKEN\")/os.environ.get(\"HF_TOKEN\")/g' 02_train_distributed.py
        sed -i 's/device_batch_size: int = 8/device_batch_size: int = 4/g' 02_train_distributed.py
        sed -i 's/head_dim: int = 256/head_dim: int = 128/g' 02_train_distributed.py
        sed -i 's/attn_impl: str = '\''splash'\''/attn_impl: str = '\''einsum'\''/g' 02_train_distributed.py
        
        # Force checkpoint saving at the end of CPT pretraining
        sed -i 's/save_checkpoint = False/save_checkpoint = True/g' 02_train_distributed.py

        python3 -c \"
with open('02_train_distributed.py', 'r') as f:
    c = f.read()
c = c.replace('import jax\nimport jax.numpy as jnp', 'import jax\nimport jax.numpy as jnp\nfrom jax.sharding import Mesh, PartitionSpec as P, NamedSharding\nmesh = Mesh(jax.devices(), (\'batch\',))\ndata_sharding = NamedSharding(mesh, P(\'batch\', None))\n')
c = c.replace('config.device_batch_size, config.seq_len', 'config.device_batch_size * len(jax.devices()), config.seq_len')
c = c.replace('loss, params, opt_state = train_step(config, params, opt_state, x_batch, y_batch, lr_mult)', 'x_batch = jax.device_put(x_batch, data_sharding)\n    y_batch = jax.device_put(y_batch, data_sharding)\n    loss, params, opt_state = train_step(config, params, opt_state, x_batch, y_batch, lr_mult)')
c = c.replace('__getattr__ = dict.__getitem__', 'def __getattr__(self, name):\n        try:\n            return self[name]\n        except KeyError:\n            raise AttributeError(name)')
c = c.replace('with open(os.path.join(CHECKPOINT_DIR, \'config.json\'), \'w\') as f:\n        json.dump(config_dict, f, indent=2, default=str)', 'with open(os.path.join(CHECKPOINT_DIR, \'config.json\'), \'w\') as f:\n        json.dump(config_dict, f, indent=2, default=str)\n    import sys; sys.exit(0)')
with open('02_train_distributed.py', 'w') as f:
    f.write(c)
\"

        export HF_TOKEN=\"${HF_TOKEN}\"
        
        echo \"=== Running CPT (Pretraining) Phase ===\"
        python3 02_train_distributed.py > cpt_training.log 2>&1
        echo \"CPT Phase completed successfully.\"
    "

echo "===================================================================="
echo "🧪 Step 3: Copying & Running SFT (Fine-Tuning) on TPU VM"
echo "===================================================================="
# Copy local JAX SFT and helper scripts to the TPU VM
gcloud alpha compute tpus tpu-vm scp \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    ./modified_tpuchat/03_sft_distributed.py ./modified_tpuchat/compile_report.sh "${TPU_NAME}:~/tpuchat_e2e/"

# Run SFT on TPU VM
gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    --command="
        set -e
        cd tpuchat_e2e
        
        export HF_TOKEN=\"${HF_TOKEN}\"
        
        echo \"=== Running SFT (Fine-Tuning) Phase ===\"
        python3 03_sft_distributed.py > sft_training.log 2>&1
        echo \"SFT Phase completed successfully.\"
    "

echo "===================================================================="
echo "📊 Step 4: Compiling E2E Report on TPU VM"
echo "===================================================================="
# Compile report on the TPU VM
gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    --command="
        set -e
        cd tpuchat_e2e
        bash compile_report.sh
    "

# Fetch report and logs back to local workspace
gcloud alpha compute tpus tpu-vm scp \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    "${TPU_NAME}:~/tpuchat_e2e/cpt_sft_e2e_report.md" ./tpu_v6e8_cpt_sft_report.md

# Delete TPU VM Spot to save billing
echo "Deleting TPU Spot Instance to prevent billing charges..."
gcloud alpha compute tpus tpu-vm delete "${TPU_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --quiet

echo "===================================================================="
echo "📦 Step 5: Syncing and Pushing Report to GitHub"
echo "===================================================================="
git add tpu_v6e8_cpt_sft_report.md ./modified_tpuchat/03_sft_distributed.py
git commit -m "docs: add E2E CPT + SFT JAX distributed report and SFT implementation script" || echo "No changes to commit"
git pull --rebase origin main
git push origin main

echo "🎉 E2E CPT & SFT 1-Cycle Verification successfully completed!"
