#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""采集并校验「左腕」摄像头数据（无界面，不弹窗显示图像）。

只读取视频帧、打印帧信息并把图像保存到磁盘，供后续查看/用于数据采集测试。
对应机器人 config 里的摄像头 key: left_wrist

用法示例:
    uv run python collect_camera_left_wrist.py                        # 存 5 帧 jpg
    uv run python collect_camera_left_wrist.py --num-frames 30        # 存 30 帧
    uv run python collect_camera_left_wrist.py --video 90             # 再录 90 帧 -> mp4
    uv run python collect_camera_left_wrist.py --device /dev/video2   # 换设备
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

CAMERA_KEY = "left_wrist"
DEVICE = "/dev/left_wrist_cam"
OUT_DIR = "camera_frames/left_wrist"


def build_config(args: argparse.Namespace) -> OpenCVCameraConfig:
    return OpenCVCameraConfig(
        index_or_path=args.device,
        fps=args.fps,
        width=args.width,
        height=args.height,
        color_mode=ColorMode.BGR,
        rotation=Cv2Rotation.NO_ROTATION,
        fourcc="MJPG",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=f"采集 {CAMERA_KEY} 摄像头数据")
    parser.add_argument("--device", default=DEVICE, help=f"视频设备路径，默认 {DEVICE}")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=5, help="保存为 jpg 的帧数")
    parser.add_argument("--video", type=int, default=0, help=">0 时额外录制这么多帧为 mp4")
    parser.add_argument("--out-dir", default=OUT_DIR, help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)
    print(f"[{CAMERA_KEY}] 打开摄像头 {args.device} ({args.width}x{args.height}@{args.fps})")

    cam = OpenCVCamera(cfg)
    cam.connect()
    try:
        for i in range(args.num_frames):
            frame = cam.read()
            if frame is None:
                print(f"[{CAMERA_KEY}] 第 {i} 帧读取失败")
                continue
            print(
                f"[{CAMERA_KEY}] 帧 {i:3d}  shape={frame.shape}  "
                f"min={frame.min()} max={frame.max()} mean={frame.mean():.1f}"
            )
            path = out_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"        已保存 {path}")

        if args.video > 0:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(out_dir / "capture.mp4"), fourcc, args.fps, (args.width, args.height)
            )
            for _ in range(args.video):
                frame = cam.read()
                if frame is None:
                    continue
                writer.write(frame)
            writer.release()
            print(f"[{CAMERA_KEY}] 已录制 {args.video} 帧 -> {out_dir / 'capture.mp4'}")
    finally:
        cam.disconnect()
        print(f"[{CAMERA_KEY}] 采集完成，输出目录: {out_dir}")


if __name__ == "__main__":
    main()
