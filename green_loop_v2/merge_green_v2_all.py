#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""合并 4 个 green_loop_v2 数据集为一个（green_loop_v2_all）。"""
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.utils.constants import HF_LEROBOT_HOME

ROOT = Path(HF_LEROBOT_HOME) / "hu43"
SRC = [
    "green_loop_v2_20261028_141505",  # 12 集
    "green_loop_v2_20261028_143528",  # 19 集
    "green_loop_v2_20261028_150311",  # 40 集
    "green_loop_v2_merged",           # 78 集（之前合并的）
]
OUT_ID = "green_loop_v2_all"

dsets = [LeRobotDataset(repo_id=f"hu43/{s}", root=str(ROOT / s)) for s in SRC]
for d in dsets:
    print(f"载入 {d.repo_id}: {d.num_episodes} 集, {len(d)} 帧")

out_dir = ROOT / OUT_ID
if out_dir.exists():
    print(f"输出目录已存在，删除重建: {out_dir}")
    shutil.rmtree(out_dir)

merged = merge_datasets(dsets, output_repo_id=f"hu43/{OUT_ID}", output_dir=out_dir)
print(f"\n合并完成: {OUT_ID}")
print(f"  集数={merged.num_episodes}  总帧={len(merged)}  fps={merged.fps}")
print(f"  路径={out_dir}")
