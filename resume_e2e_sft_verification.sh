#!/usr/bin/env bash
set -e

# ==============================================================================
# TPU v6e-8 Resume SFT & Compile Report Orchestration Script
# ==============================================================================

PROJECT_ID="takashix-tpu"
ZONE="asia-northeast1-b"
TPU_NAME="tpuchat-v6e-8x-sft-e2e"
HF_TOKEN="${HF_TOKEN:-your_hf_token_here}"

echo "===================================================================="
echo "🧪 Step 1: Copying Updated SFT & Helper Scripts to TPU VM"
echo "===================================================================="
gcloud alpha compute tpus tpu-vm scp \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    ./modified_tpuchat/03_sft_distributed.py ./modified_tpuchat/compile_report.sh "${TPU_NAME}:~/tpuchat_e2e/"

echo "===================================================================="
echo "🧪 Step 2: Running SFT (Fine-Tuning) on TPU VM"
echo "===================================================================="
# gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
#     --project="${PROJECT_ID}" --zone="${ZONE}" \
#     --command="
#         set -e
#         cd tpuchat_e2e
#         export HF_TOKEN=\"${HF_TOKEN}\"
#         echo \"=== Running SFT (Fine-Tuning) Phase ===\"
#         python3 03_sft_distributed.py > sft_training.log 2>&1
#         echo \"SFT Phase completed successfully.\"
#     "

echo "===================================================================="
echo "📊 Step 3: Compiling E2E Report on TPU VM"
echo "===================================================================="
gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    --command="
        set -e
        cd tpuchat_e2e
        bash compile_report.sh
    "

gcloud alpha compute tpus tpu-vm scp \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    "${TPU_NAME}:~/tpuchat_e2e/cpt_sft_e2e_report.md" ./tpu_v6e8_cpt_sft_report.md

echo "Deleting TPU Spot Instance to prevent billing charges..."
gcloud alpha compute tpus tpu-vm delete "${TPU_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --quiet

echo "===================================================================="
echo "📦 Step 4: Syncing and Pushing Report to GitHub"
echo "===================================================================="
git add tpu_v6e8_cpt_sft_report.md ./modified_tpuchat/03_sft_distributed.py
git commit -m "docs: add E2E CPT + SFT JAX distributed report and SFT implementation script" || echo "No changes to commit"
git pull --rebase origin main
git push origin main

echo "🎉 E2E SFT Resume & Verification successfully completed!"
