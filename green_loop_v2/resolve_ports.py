#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按稳定 USB 端口（ID_PATH）解析设备节点，供采集脚本使用。

原因：USB 重枚举后 /dev/videoN 与 /dev/ttyACMn 会漂移，但物理端口的 ID_PATH 稳定。
    ACM1 / video(绿腕) = 左臂侧   port 74:00.4
    ACM0 / video(蓝腕) = 右臂侧   port 0a:00.0
    head 广角相机      = port 0a:00.0-5

用法：
    python resolve_ports.py            # 打印映射
    python resolve_ports.py --export    # 输出 shell 变量（供 record 脚本 eval）
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess

# 物理端口 -> 角色。以当前实测为准（每次重插后跑 resolve_ports.py 核对并按需更新）。
# 判定依据：port1(LEFT_ARM) 是含 head 电机(1-8) 的总线；腕相机以画面方块颜色区分。
# 注意：不要登记"历史端口"——端口复用会串角色，换口后直接改这里的值。
PORT_ROLES_LIST = [
    # --- tty (机械臂) ---
    ("pci-0000:0a:00.0-usb-0:8:1.0", "LEFT_ARM"),     # ACM1 左臂（含 head）
    ("pci-0000:75:00.0-usb-0:1:1.0", "RIGHT_ARM"),    # ACM0 右臂
    # --- video (相机, index==0) ---
    ("pci-0000:0a:00.0-usb-0:6:1.0", "LEFT_WRIST"),   # 绿方块近景 = 左腕
    ("pci-0000:74:00.4-usb-0:1:1.0", "RIGHT_WRIST"),  # 蓝方块近景 = 右腕
    ("pci-0000:74:00.3-usb-0:1:1.0", "HEAD"),          # 广角
]
PORT_ROLES = dict(PORT_ROLES_LIST)


def id_path(node: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["udevadm", "info", "-q", "property", "-n", node], stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("ID_PATH="):
            return line.split("=", 1)[1]
    return None


def v4l_index(node: str) -> int | None:
    n = os.path.basename(node)
    try:
        return int(open(f"/sys/class/video4linux/{n}/index").read().strip())
    except Exception:
        return None


def resolve() -> dict[str, str]:
    found: dict[str, str] = {}
    for t in sorted(glob.glob("/dev/ttyACM*")):
        role = PORT_ROLES.get(id_path(t) or "")
        if role:
            found[role] = t
    for v in sorted(glob.glob("/dev/video*")):
        if v4l_index(v) != 0:
            continue
        role = PORT_ROLES.get(id_path(v) or "")
        if role and role not in found:
            found[role] = v
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()

    m = resolve()
    order = ["LEFT_ARM", "RIGHT_ARM", "LEFT_WRIST", "RIGHT_WRIST", "HEAD"]
    if args.export:
        for k in order:
            print(f'{k}="{m.get(k, "")}"')
        return

    print("设备解析结果（按 ID_PATH）：")
    for k in order:
        print(f"  {k:<12} -> {m.get(k, '（未找到）')}")
    missing = [k for k in order if not m.get(k)]
    if missing:
        print("\n[警告] 缺失：", ", ".join(missing), "（可能是 USB 未插稳或端口变了）")


if __name__ == "__main__":
    main()
