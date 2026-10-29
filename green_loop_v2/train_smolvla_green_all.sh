#!/usr/bin/env bash
# SmolVLA 微调 —— 数据集 hu43/green_loop_v2_all（149 集 / 96726 帧，双臂 12 维动作 + 左右腕双相机）
# 任务：把绿色方块放进盒子里，然后拿出来，循环往复
# GPU 0（当前空闲；GPU 1 在跑 qwen3-tts）。
#
# 用法：
#   ./train_smolvla_green_all.sh              # 默认 20000 步
#   STEPS=5000 ./train_smolvla_green_all.sh   # 改步数
#   TRAIN_GPU=1 ./train_smolvla_green_all.sh  # 换 GPU
set -euo pipefail

PY=/home/zhoulin/miniconda3/envs/hupy3.12/bin/python
export PATH="/home/zhoulin/miniconda3/envs/hupy3.12/bin:$PATH"
cd /home/zhoulin/Huang/lerobot

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/home/zhoulin/Huang/lerobot/output_smolvla_green_all_${TIMESTAMP}"
LOG_DIR="/home/zhoulin/Huang/lerobot/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/smolvla_green_all_${TIMESTAMP}.log"

DATASET_ID="hu43/green_loop_v2_all"
DATASET_ROOT="/home/zhoulin/.cache/huggingface/lerobot/hu43/green_loop_v2_all"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_GPU="${TRAIN_GPU:-0}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MALLOC_TRIM_THRESHOLD_=0

echo "=== SmolVLA 训练 ===" | tee "$LOG_FILE"
echo "数据集: $DATASET_ID (149 集)" | tee -a "$LOG_FILE"
echo "输出: $OUT_DIR" | tee -a "$LOG_FILE"
echo "GPU: $TRAIN_GPU  steps: $STEPS  batch: $BATCH_SIZE" | tee -a "$LOG_FILE"

exec nice -n 10 "$PY" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.image_transforms.enable=true \
  --rename_map='{"observation.images.left_wrist": "observation.images.camera1", "observation.images.right_wrist": "observation.images.camera2"}' \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --num_workers=2 \
  --log_freq=50 \
  --output_dir="$OUT_DIR" \
  --job_name=smolvla_green_all \
  --wandb.enable=false
