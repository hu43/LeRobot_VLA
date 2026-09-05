#!/usr/bin/env bash
# pi0.5 (LoRA) fine-tune on SO-101 catchBottle (握手) dataset.
# Env: hupy3.12 (hu/lerobot 0.5.2 needs Python 3.12 syntax; hupy3.11 breaks).
# Recipe (verified by smoke test):
#   - pi05_base weights loaded via --policy.path (fp32 blob, cast to bf16 with --policy.dtype=bfloat16)
#   - LoRA PEFT (default targets: action-expert q/v + state/action proj), frees ~16GB vs full FT
#   - single camera: rename front -> base_0_rgb (pi05 expects 3 cams; missing 2 auto-masked)
#   - --policy.push_to_hub=false (no HF upload; tokenizer points to local copy)
#   - runs on GPU 1 (GPU 0 is busy with the CosyVoice TTS service)
set -euo pipefail

PY=/home/zhoulin/miniconda3/envs/hupy3.12/bin/python
cd /home/zhoulin/hu/lerobot

PI05_BASE=/home/zhoulin/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba
if [ ! -d "$PI05_BASE" ]; then
  echo "ERROR: pi05_base weights not found at $PI05_BASE" >&2
  exit 1
fi

HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
  "$PY" -m lerobot.scripts.lerobot_train \
  --policy.path="$PI05_BASE" \
  --policy.repo_id=lingbot/pi05-catchbottle \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --peft.method_type=lora \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb"}' \
  --dataset.repo_id /home/zhoulin/.cache/huggingface/lerobot/catchBottle \
  --env null \
  --output_dir /home/zhoulin/hu/lerobot/pi05/output \
  --job_name catchBottle_pi05 \
  --steps 50000 \
  --batch_size 4 \
  --save_freq 5000 \
  --log_freq 100 \
  --num_workers 4 \
  --use_policy_training_preset true \
  --wandb.enable false
