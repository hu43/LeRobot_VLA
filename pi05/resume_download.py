#!/usr/bin/env python3
"""Resumable pi05_base download: keep calling snapshot_download until fully cached."""
import os, sys, time, signal

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# use the configured mirror (direct HF became unreachable mid-download)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

TARGET = 14467165872
BLOB_DIR = "/home/zhoulin/.cache/huggingface/hub/models--lerobot--pi05_base/blobs"
ATTEMPT_TIMEOUT = 150  # seconds per attempt; a hung connection is interrupted and retried

def cur_size():
    import glob
    tot = 0
    for f in glob.glob(f"{BLOB_DIR}/*.incomplete"):
        try:
            tot += os.path.getsize(f)
        except OSError:
            pass
    return tot

def handler(signum, frame):
    raise TimeoutError("attempt timed out")

from huggingface_hub import snapshot_download

for attempt in range(1, 2000):
    sz = cur_size()
    print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}  downloaded={sz/1e9:.2f} GB / {TARGET/1e9:.2f} GB ({sz/TARGET*100:.1f}%)", flush=True)
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(ATTEMPT_TIMEOUT)
    try:
        p = snapshot_download("lerobot/pi05_base")
        signal.alarm(0)
        print("SNAPSHOT COMPLETE:", p, flush=True)
        sys.exit(0)
    except Exception as e:
        signal.alarm(0)
        print(f"  retry: {type(e).__name__}: {str(e)[:160]}", flush=True)
        time.sleep(2)

print("gave up")
sys.exit(1)