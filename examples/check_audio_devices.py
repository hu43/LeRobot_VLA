#!/usr/bin/env python3
"""
检查可用的音频设备
"""

import sounddevice as sd

print("="*60)
print("可用的音频设备")
print("="*60)
print()

devices = sd.query_devices()
for i, dev in enumerate(devices):
    print(f"[{i}] {dev['name']}")
    print(f"    输入通道: {dev['max_input_channels']}")
    print(f"    输出通道: {dev['max_output_channels']}")
    print(f"    默认采样率: {dev['default_samplerate']}")
    print()

print("="*60)
print(f"默认输入设备索引: {sd.default.device[0]}")
print(f"默认输出设备索引: {sd.default.device[1]}")
print("="*60)
