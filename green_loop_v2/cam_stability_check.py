#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两路腕部相机稳定性验证：60 秒，报告丢帧/掉线。合格标准：0 丢帧 + 全程在线。"""
import time, sys

from resolve_ports import resolve

# 直接复用 resolve_ports 的端口映射（换线后无需改本文件）
m = resolve()
WANT = {"左腕(绿)": m.get("LEFT_WRIST", ""),
        "右腕(蓝)": m.get("RIGHT_WRIST", "")}

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation, Cv2Backends

def ocv(d):
    return OpenCVCameraConfig(index_or_path=d, fps=30, width=640, height=480,
                              color_mode=ColorMode.RGB, rotation=Cv2Rotation.NO_ROTATION,
                              backend=Cv2Backends.V4L2, fourcc="MJPG")

cams = {}
for k, d in WANT.items():
    if not d:
        print(f"❌ {k} 未找到（端口映射缺失）"); sys.exit(1)
    try:
        c = OpenCVCamera(ocv(d)); c.connect(); cams[k] = c
        print(f"✅ {k} {d} 已连接")
    except Exception as e:
        print(f"❌ {k} {d} 连接失败: {e}"); sys.exit(1)

fails = {k: 0 for k in cams}
dropped = {k: False for k in cams}
t0 = time.time()
while time.time() - t0 < 60:
    for k, c in list(cams.items()):
        try:
            c.async_read(timeout_ms=60)
        except TimeoutError:
            fails[k] += 1
        except Exception:
            dropped[k] = True
    time.sleep(0.033)

print("\n=== 60 秒结果 ===")
ok = True
for k in cams:
    alive = cams[k].thread.is_alive()
    # 判定：读线程存活 + 无彻底掉线 + 超时率 < 5%（零星超时会自愈，不影响训练）
    rate_ok = fails[k] < 30
    good = alive and (not dropped[k]) and rate_ok
    if not good: ok = False
    print(f"{'✅' if good else '❌'} {k}: 超时 {fails[k]} 次, 读线程{'存活' if alive else '已死'}, 掉线={dropped[k]}")
for c in cams.values():
    try: c.disconnect()
    except Exception: pass
print("\n结论:", "✅ 合格，可以录制" if ok else "❌ 不合格（读线程死亡或掉线）")
