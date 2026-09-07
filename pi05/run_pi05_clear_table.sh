#!/usr/bin/env bash
# pi0.5 (LoRA) fine-tune on XLeRobot clear_table dataset (104 eps / 101334 frames).
# Based on verified run_pi05_catchbottle.sh recipe.
#   - pi05_base weights via --policy.path (cast to bf16)
#   - LoRA PEFT, single GPU (GPU 1; GPU 0 reserved for TTS service)
#   - cameras renamed to openpi convention: head->base_0_rgb, wrists -> *_wrist_0_rgb
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
  --policy.repo_id=hu43/pi05-clear_table \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --peft.method_type=lora \
  --rename_map='{"observation.images.head": "observation.images.base_0_rgb", "observation.images.left_wrist": "observation.images.left_wrist_0_rgb", "observation.images.right_wrist": "observation.images.right_wrist_0_rgb"}' \
  --dataset.repo_id=hu43/clear_table \
  --dataset.root=/home/zhoulin/datasets/clear_table \
  --env null \
  --output_dir /home/zhoulin/hu/lerobot/pi05/output_clear_table \
  --job_name clear_table_pi05 \
  --steps 50000 \
  --batch_size 4 \
  --save_freq 5000 \
  --log_freq 100 \
  --num_workers 4 \
  --use_policy_training_preset true \
  --wandb.enable false
