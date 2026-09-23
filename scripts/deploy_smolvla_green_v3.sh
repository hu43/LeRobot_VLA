#!/usr/bin/env bash
# Deploy SmolVLA model trained on green_in_box_v3 dataset.
# Single LEFT arm (so101_follower) + left wrist camera.
# Usage:
#   ./run_smolvla_green_v3.sh                  # default 60s on GPU 1
#   DURATION_SEC=120 ./run_smolvla_green_v3.sh # run 120s
#   POLICY_GPU=0 ./run_smolvla_green_v3.sh      # use GPU 0
#   CKPT_STEP=010000 ./run_smolvla_green_v3.sh  # use specific checkpoint
set -euo pipefail

PY=/home/zhoulin/miniconda3/envs/hupy3.12/bin/python
export PATH="/home/zhoulin/miniconda3/envs/hupy3.12/bin:$PATH"
cd /home/zhoulin/hu/lerobot

# Checkpoint selection: CKPT_STEP=last (default) or specific step like 010000
CKPT_STEP=${CKPT_STEP:-last}
CKPT="/home/zhoulin/hu/lerobot/pi05/output_smolvla_1x5090_20260923_094001/checkpoints/${CKPT_STEP}/pretrained_model"

if [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint not found at $CKPT" >&2
  echo "Available checkpoints:"
  ls /home/zhoulin/hu/lerobot/pi05/output_smolvla_1x5090_20260923_094001/checkpoints/ 2>/dev/null
  exit 1
fi

# DURATION_SEC overridable
DURATION_SEC=${DURATION_SEC:-120}

echo "=== SmolVLA Deployment ==="
echo "Checkpoint: $CKPT"
echo "GPU: ${POLICY_GPU:-1}"
echo "Duration: ${DURATION_SEC}s"
echo "Task: don't anything"
echo "============================"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=${POLICY_GPU:-1} \
  "$PY" -m lerobot.scripts.lerobot_rollout \
  --strategy.type=base \
  --policy.path="$CKPT" \
  --policy.device=cuda \
  --rename_map='{"observation.images.left_wrist": "observation.images.camera1"}' \
  --robot.type=so101_follower \
  --robot.port=/dev/follower_arm \
  --robot.id=xlerobot_left_arm \
  --robot.cameras='{ left_wrist: {type: opencv, index_or_path: /dev/left_wrist_cam, width: 640, height: 480, fps: 30, color_mode: RGB, rotation: 0, backend: V4L2, fourcc: MJPG}}' \
  --task="don't anything" \
  --duration="$DURATION_SEC" \
  --display_data=false
