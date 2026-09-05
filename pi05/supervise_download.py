#!/usr/bin/env python3
"""Watchdog supervisor: run download worker, kill+restart if no byte progress for STALL_SECS."""
import os, sys, time, subprocess, glob

TARGET = 14467165872
BLOB = "/home/zhoulin/.cache/huggingface/hub/models--lerobot--pi05_base/blobs"
WORKER = "/home/zhoulin/hu/lerobot/pi05/worker_download.py"
STALL_SECS = 60
PY = sys.executable

def cur_size():
    tot = 0
    for f in glob.glob(f"{BLOB}/*.incomplete"):
        try:
            tot += os.path.getsize(f)
        except OSError:
            pass
    return tot

for round_ in range(1, 2000):
    before = cur_size()
    print(f"[{time.strftime('%H:%M:%S')}] round {round_}  {before/1e9:.2f}/{TARGET/1e9:.2f} GB ({before/TARGET*100:.1f}%)", flush=True)
    proc = subprocess.Popen([PY, WORKER])
    last, last_update = before, time.time()
    while proc.poll() is None:
        time.sleep(8)
        s = cur_size()
        if s != last:
            last, last_update = s, time.time()
        if time.time() - last_update > STALL_SECS:
            print("  stalled -> kill worker", flush=True)
            proc.kill()
            proc.wait()
            break
    rc = proc.poll()
    if rc == 0:
        print("SNAPSHOT COMPLETE", flush=True)
        sys.exit(0)
    time.sleep(2)

print("gave up")
sys.exit(1)