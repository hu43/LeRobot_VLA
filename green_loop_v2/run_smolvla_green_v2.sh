#!/usr/bin/env bash
# SmolVLA 部署 —— 数据集 hu43/green_loop_v2_merged（双臂 12 维动作 + 左右腕双相机）
# 任务：把绿色方块放进盒子里，然后拿出来，循环往复
# 模型：output_smolvla_green_v2_20261028_113853
#
# 用法：
#   ./run_smolvla_green_v2.sh                    # 默认 last 检查点，跑 60s，GPU 1
#   DURATION_SEC=120 ./run_smolvla_green_v2.sh   # 跑 120s
#   CKPT_STEP=015000 ./run_smolvla_green_v2.sh   # 指定检查点（005000/010000/015000/020000/last）
#   POLICY_GPU=0 ./run_smolvla_green_v2.sh       # 换 GPU
set -euo pipefail

PY=/home/zhoulin/miniconda3/envs/hupy3.12/bin/python
export PATH="/home/zhoulin/miniconda3/envs/hupy3.12/bin:$PATH"
export PYTHONPATH="/home/zhoulin/hu/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
cd /home/zhoulin/Huang/lerobot

# ====== 参数（环境变量可覆盖）======
OUT_DIR="/home/zhoulin/Huang/lerobot/output_smolvla_green_v2_20261028_113853"
CKPT_STEP="${CKPT_STEP:-last}"
CKPT="${OUT_DIR}/checkpoints/${CKPT_STEP}/pretrained_model"
DURATION_SEC="${DURATION_SEC:-60}"
POLICY_GPU="${POLICY_GPU:-1}"
TASK="${TASK:-把绿色方块放进盒子里，然后拿出来，循环往复}"
CAM_FPS="${CAM_FPS:-30}"
# ===================================

if [ ! -d "$CKPT" ]; then
  echo "[错误] 找不到检查点: $CKPT" >&2
  echo "可用检查点:"
  ls "${OUT_DIR}/checkpoints/" 2>/dev/null
  exit 1
fi

# 清理上次会话残留的 rerun（它会占住相机导致 select() timeout）
pkill -9 -f "rerun --port=9876" 2>/dev/null || true
sleep 1

# 解析设备（按 USB ID_PATH 稳定映射）
eval "$("$PY" resolve_ports.py --export)"
for var in LEFT_ARM RIGHT_ARM LEFT_WRIST RIGHT_WRIST; do
  [ -z "${!var:-}" ] && { echo "[错误] 未能解析设备 $var，请检查接线后运行 python resolve_ports.py" >&2; exit 1; }
done
# 相机优先用 udev 稳定符号链接
[ -e /dev/left_wrist_cam ] && LEFT_WRIST=/dev/left_wrist_cam
[ -e /dev/right_wrist_cam ] && RIGHT_WRIST=/dev/right_wrist_cam
echo "设备：左臂=$LEFT_ARM 右臂=$RIGHT_ARM 左腕=$LEFT_WRIST 右腕=$RIGHT_WRIST"

# 相机预检（3 秒内必须读到帧）
"$PY" - "$LEFT_WRIST" "$RIGHT_WRIST" <<'PYEOF'
import sys, cv2, time
for d in sys.argv[1:]:
    c = cv2.VideoCapture(d, cv2.CAP_V4L2)
    ok = False
    if c.isOpened():
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        t0 = time.time()
        while time.time() - t0 < 3.0:
            r, f = c.read()
            if r and f is not None:
                ok = True; break
    c.release()
    if not ok:
        sys.exit(f"[错误] 相机 3 秒内读不到帧: {d}（先执行: pkill -9 -f rerun 再重试）")
    print(f"  相机可读 OK: {d}")
PYEOF

CAMS="{ left_wrist: {type: opencv, index_or_path: $LEFT_WRIST, width: 640, height: 480, fps: $CAM_FPS, color_mode: RGB, rotation: 0, backend: V4L2, fourcc: MJPG}, right_wrist: {type: opencv, index_or_path: $RIGHT_WRIST, width: 640, height: 480, fps: $CAM_FPS, color_mode: RGB, rotation: 0, backend: V4L2, fourcc: MJPG}}"

echo "=== SmolVLA 部署 ==="
echo "检查点: $CKPT"
echo "GPU: $POLICY_GPU  时长: ${DURATION_SEC}s"
echo "任务: $TASK"
echo "===================="

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES="$POLICY_GPU" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" -m lerobot.scripts.lerobot_rollout \
  --strategy.type=base \
  --policy.path="$CKPT" \
  --policy.device=cuda \
  --rename_map='{"observation.images.left_wrist": "observation.images.camera1", "observation.images.right_wrist": "observation.images.camera2"}' \
  --robot.type=xlerobot \
  --robot.port1="$LEFT_ARM" \
  --robot.port2="$RIGHT_ARM" \
  --robot.use_head=false \
  --robot.use_base=false \
  --robot.cameras="$CAMS" \
  --task="$TASK" \
  --duration="$DURATION_SEC" \
  --display_data=false
