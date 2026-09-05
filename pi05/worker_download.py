#!/usr/bin/env python3
"""One-shot pi05_base download worker (resumes from incomplete blob)."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import snapshot_download
snapshot_download("lerobot/pi05_base")
print("WORKER_DONE")