#!/usr/bin/env bash
# SmolVLA fine-tuning on single RTX 5090 (24GB), GPU 1.
# Model: 450M params (bf16), full fine-tuning.
# Dataset: green_in_box_v3 (80 eps, 27009 frames, left_wrist camera only).
# Anti-freeze: single process, nice 19, OMP=4, batch=8.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/output_smolvla_1x5090_${TIMESTAMP}"
LOG_FILE="${ROOT_DIR}/logs/smolvla_1x5090_${TIMESTAMP}.log"
cd "$ROOT_DIR"

export PATH="/home/zhoulin/miniconda3/envs/hupy3.12/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MALLOC_TRIM_THRESHOLD_=0

mkdir -p "$ROOT_DIR/logs"

echo "[$(date)] Starting SmolVLA training on 1x RTX 5090 GPU1 (bf16, batch=8)" | tee "$LOG_FILE"
echo "Output: $OUT_DIR" | tee -a "$LOG_FILE"

exec nice -n 19 /home/zhoulin/miniconda3/envs/hupy3.12/bin/lerobot-train \
  --dataset.repo_id=hu43/green_in_box_v3_20260921_100555 \
  --dataset.root=/home/zhoulin/.cache/huggingface/lerobot/hu43/green_in_box_v3_20260921_100555 \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.image_transforms.enable=true \
  --rename_map='{"observation.images.left_wrist": "observation.images.camera1"}' \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=5000 \
  --num_workers=2 \
  --log_freq=50 \
  --output_dir="$OUT_DIR" \
  --job_name=smolvla_1x5090 \
  --wandb.enable=false
