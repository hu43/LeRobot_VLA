#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""相机 / 机械臂端口对照工具（自己看画面判断匹配关系）。

为什么要按 ID_PATH 解析：USB 重新枚举后 /dev/videoN 会漂移（video0/1/4 互换），
但 USB 物理端口的 ID_PATH 稳定。本脚本用 sysfs index==0 找到每个端口的“捕获节点”，
再打开实时预览，窗口标题里同时写 /dev/videoN 和端口后缀，方便你对照。

用法（在 hupy3.12 环境下）：
    python camera_preview.py                 # 实时预览 + 打印端口映射
    python camera_preview.py --snap          # 不弹窗，各存一张快照到 ./camcheck
    python camera_preview.py --wiggle ACM0   # 预览时轻微转动 ttyACM0 的 wrist_roll 帮助辨认
    python camera_preview.py --wiggle ACM1

预览窗口操作：按 s 保存当前三路快照；按 q 或 ESC 退出。
判断方法：手动摆动机器人某条臂（或加 --wiggle），看哪个窗口画面跟着动。
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

import cv2


def id_path(video_node: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["udevadm", "info", "-q", "property", "-n", video_node], stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("ID_PATH="):
            return line.split("=", 1)[1]
    return None


def v4l_index(video_node: str) -> int | None:
    n = os.path.basename(video_node)
    try:
        return int(open(f"/sys/class/video4linux/{n}/index").read().strip())
    except Exception:
        return None


def short_port(p: str) -> str:
    # pci-0000:74:00.4-usb-0:1.4:1.0 -> 74:00.4-1.4
    try:
        s = p.replace("pci-0000:", "").replace("usb-0:", "").replace(":1.0", "")
        return s
    except Exception:
        return p or "?"


def enumerate_capture_nodes() -> dict[str, str]:
    """返回 {ID_PATH: /dev/videoN}，只取 index==0 的捕获节点。"""
    out: dict[str, str] = {}
    for v in sorted(glob.glob("/dev/video*")):
        if v4l_index(v) != 0:
            continue
        p = id_path(v)
        if p and p not in out:
            out[p] = v
    return out


def enumerate_tty() -> dict[str, str]:
    """返回 {/dev/ttyACMn: 端口后缀}。"""
    out = {}
    for t in sorted(glob.glob("/dev/ttyACM*")):
        p = id_path(t)
        out[t] = short_port(p) if p else "?"
    return out


def open_cap(node: str) -> cv2.VideoCapture:
    c = cv2.VideoCapture(node, cv2.CAP_V4L2)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return c


def wiggle(port_alias: str, offset: int = 450) -> None:
    """轻微转动某总线的 wrist_roll（id5），结束后复位并关闭扭矩。"""
    port = "/dev/ttyACM0" if port_alias.upper() == "ACM0" else "/dev/ttyACM1"
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    try:
        bus = FeetechMotorsBus(port, {f"m{i}": Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i in range(1, 6)})
        bus.connect(handshake=False)
        roll = bus.read("Present_Position", "m5", normalize=False)
        print(f"[wiggle] {port} wrist_roll 基准={roll}，缓慢转动 ±{offset} …")
        bus.write("Torque_Enable", "m5", 1, normalize=False)
        try:
            bus.write("Goal_Position", "m5", roll + offset, normalize=False)
            time.sleep(1.2)
            bus.write("Goal_Position", "m5", roll - offset, normalize=False)
            time.sleep(1.2)
        finally:
            bus.write("Goal_Position", "m5", roll, normalize=False)
            time.sleep(0.8)
            bus.write("Torque_Enable", "m5", 0, normalize=False)
            bus.disconnect()
        print("[wiggle] 已复位并关闭扭矩")
    except Exception as e:
        print(f"[wiggle] 跳过（{type(e).__name__}: {e}）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", action="store_true", help="不弹窗，各存一张快照")
    ap.add_argument("--out-dir", default="./camcheck")
    ap.add_argument("--wiggle", default=None, help="ACM0 或 ACM1，轻微转动 wrist_roll 帮助辨认")
    args = ap.parse_args()

    nodes = enumerate_capture_nodes()
    ttys = enumerate_tty()

    print("=" * 64)
    print("摄像头捕获节点（按稳定 USB 端口 ID_PATH，index==0）：")
    for p, v in sorted(nodes.items(), key=lambda kv: kv[1]):
        print(f"  {v:<14}  {short_port(p):<16}  ({p})")
    print("机械臂串口：")
    for t, s in ttys.items():
        print(f"  {t:<14}  {s}")
    print("=" * 64)
    sys.stdout.flush()

    if not nodes:
        print("没找到任何摄像头捕获节点。")
        return

    if args.wiggle:
        wiggle(args.wiggle)

    caps: dict[str, cv2.VideoCapture] = {}
    labels: dict[str, str] = {}
    for p, v in sorted(nodes.items(), key=lambda kv: kv[1]):
        c = open_cap(v)
        if not c.isOpened():
            print(f"[warn] 打不开 {v}，跳过")
            continue
        key = short_port(p)
        caps[key] = c
        labels[key] = f"{v}  | port {key}"
    if not caps:
        print("所有摄像头都打不开。")
        return

    for _ in range(10):
        for c in caps.values():
            c.read()
    time.sleep(0.3)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    def snapshot():
        for key, c in caps.items():
            for _ in range(6):
                r, f = c.read()
                if r and f is not None:
                    path = os.path.join(out_dir, f"cam_{key.replace(':', '-')}.jpg")
                    cv2.imwrite(path, f)
                    print(f"  已保存 {path}")
                    break

    if args.snap:
        snapshot()
        for c in caps.values():
            c.release()
        return

    for key, c in caps.items():
        cv2.namedWindow(key, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(key, 640, 480)

    print("预览中：按 s 保存快照，按 q/ESC 退出。请摆动机器人手臂观察哪个窗口跟着动。")
    while True:
        frames = {}
        for key, c in caps.items():
            r, f = c.read()
            if r and f is not None:
                f = f.copy()
                cv2.putText(f, labels[key], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                frames[key] = f
        for key, f in frames.items():
            cv2.imshow(key, f)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            snapshot()

    for c in caps.values():
        c.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
