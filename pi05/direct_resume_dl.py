#!/usr/bin/env python3
"""Resumable direct download of lerobot/pi05_base model.safetensors.

Uses huggingface_hub.http_get (10s read timeout + auto-resume on httpx errors),
appends to the existing .incomplete blob so we resume from the last byte.
Verifies sha256 then renames to the final content-addressed blob.
"""
import os, sys, time, hashlib

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

BLOB = "/home/zhoulin/.cache/huggingface/hub/models--lerobot--pi05_base/blobs"
HASH = "0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f"
INCOMPLETE = f"{BLOB}/{HASH}.8d1f2787.incomplete"
FINAL = f"{BLOB}/{HASH}"
TARGET = 14467165872

URLS = [
    "https://hf-mirror.com/lerobot/pi05_base/resolve/main/model.safetensors",
    "https://huggingface.co/lerobot/pi05_base/resolve/main/model.safetensors",
]

from huggingface_hub.file_download import http_get


def cur_size():
    if os.path.exists(FINAL):
        return os.path.getsize(FINAL)
    return os.path.getsize(INCOMPLETE) if os.path.exists(INCOMPLETE) else 0


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


for attempt in range(1, 5000):
    sz = cur_size()
    print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}  {sz/1e9:.2f}/{TARGET/1e9:.2f} GB ({sz/TARGET*100:.1f}%)", flush=True)
    if sz == TARGET:
        path = INCOMPLETE if os.path.exists(INCOMPLETE) else FINAL
        h = sha256_of(path)
        if h == HASH:
            if os.path.exists(INCOMPLETE):
                os.rename(INCOMPLETE, FINAL)
            print("DOWNLOAD COMPLETE + sha256 OK", flush=True)
            sys.exit(0)
        else:
            print(f"  size match but sha256 MISMATCH ({h}) -> restart from 0", flush=True)
            os.remove(path)
            continue
    got = False
    for url in URLS:
        try:
            with open(INCOMPLETE, "ab") as f:
                http_get(url, f, resume_size=sz, expected_size=TARGET)
            got = True
            break
        except Exception as e:
            print(f"  {url.split('/')[2]} err: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(2)
    if not got:
        print("  all endpoints failed this attempt", flush=True)
    time.sleep(2)

print("gave up")
sys.exit(1)
