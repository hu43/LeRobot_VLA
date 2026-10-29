#!/usr/bin/env bash
# XLeRobot 双臂 VR 示教采集 —— 全新数据集
# 任务：把绿色方块放进盒子，再拿出来，循环往复（放进去 -> 拿出来 -> 再放进去 …）
#
# 设备：由 resolve_ports.py 按 USB 端口(ID_PATH)自动解析；相机用 udev 稳定符号链接
#   （/dev/left_wrist_cam、/dev/right_wrist_cam）。运行前先：python resolve_ports.py 核对。
#
# VR 左手柄摇杆：右=结束本集   左=重录本集   上=停止录制   下=复位零位
# 头显浏览器连终端打印的 https://<本机IP>:8443
set -euo pipefail

PY=/home/zhoulin/miniconda3/envs/hupy3.12/bin/python
XLEVR=/home/zhoulin/Tang/XLeRobot/XLeVR
export PATH="/home/zhoulin/miniconda3/envs/hupy3.12/bin:$PATH"
export PYTHONPATH="/home/zhoulin/hu/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"

# ====== 参数（环境变量可覆盖）======
REPO_ID="${REPO_ID:-hu43/green_loop_v2}"              # 全新数据集名（自动追加时间戳）
TASK="${TASK:-把绿色方块放进盒子里，然后拿出来，循环往复}"  # 训练/部署必须与此完全一致
NUM_EPISODES="${NUM_EPISODES:-40}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"   # 单集上限 60s（提前完成按摇杆右结束）
RESET_TIME_S="${RESET_TIME_S:-1}"        # 集间复位 1s（不记录）
FPS="${FPS:-30}"
CAM_FPS="${CAM_FPS:-20}"                 # 相机帧率：官方文档建议 20
DISPLAY_DATA="${DISPLAY_DATA:-false}"    # 关闭可视化（rerun 残留进程会占相机导致启动失败）
USE_HEAD_CAM="${USE_HEAD_CAM:-false}"    # 头部广角相机（默认不录）
# ===================================

cd "$(dirname "$0")"

# 清理上次会话残留的 rerun（它会占住相机导致 select() timeout）
pkill -9 -f "rerun --port=9876" 2>/dev/null || true
sleep 1

# 解析设备
eval "$("$PY" resolve_ports.py --export)"
REQUIRED="LEFT_ARM RIGHT_ARM LEFT_WRIST RIGHT_WRIST"
[ "$USE_HEAD_CAM" = "true" ] && REQUIRED="$REQUIRED HEAD"
for var in $REQUIRED; do
  [ -z "${!var:-}" ] && { echo "[错误] 未能解析设备 $var，请检查接线后运行 python resolve_ports.py" >&2; exit 1; }
done
# 相机优先用 udev 稳定符号链接
[ -e /dev/left_wrist_cam ] && LEFT_WRIST=/dev/left_wrist_cam
[ -e /dev/right_wrist_cam ] && RIGHT_WRIST=/dev/right_wrist_cam
[ -e /dev/head_cam ] && HEAD=/dev/head_cam
echo "设备：左臂=$LEFT_ARM 右臂=$RIGHT_ARM 左腕=$LEFT_WRIST 右腕=$RIGHT_WRIST 头=${HEAD:-不用}"

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

CAMS="{ left_wrist: {type: opencv, index_or_path: $LEFT_WRIST, width: 640, height: 480, fps: $CAM_FPS, color_mode: RGB, rotation: 0, backend: V4L2, fourcc: MJPG}, right_wrist: {type: opencv, index_or_path: $RIGHT_WRIST, width: 640, height: 480, fps: $CAM_FPS, color_mode: RGB, rotation: 0, backend: V4L2, fourcc: MJPG}"
if [ "$USE_HEAD_CAM" = "true" ]; then
  CAMS="$CAMS, head: {type: opencv, index_or_path: $HEAD, width: 640, height: 480, fps: $CAM_FPS, color_mode: RGB, rotation: 0, backend: V4L2, fourcc: MJPG}"
fi
CAMS="$CAMS}"

"$PY" -m lerobot.scripts.lerobot_record \
  --robot.type=xlerobot \
  --robot.port1="$LEFT_ARM" \
  --robot.port2="$RIGHT_ARM" \
  --robot.use_head=false \
  --robot.use_base=false \
  --robot.cameras="$CAMS" \
  --teleop.type=xlerobot_vr \
  --teleop.xlevr_path="$XLEVR" \
  --display_data="$DISPLAY_DATA" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.fps="$FPS" \
  --dataset.push_to_hub=false
  --display_data=false
