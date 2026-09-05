#!/usr/bin/env bash
# Resumable loop: keep calling snapshot_download until pi05_base fully cached.
# Uses direct HF (no mirror) for speed; resumes the same .incomplete blob on each retry.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hupy3.12
unset HF_ENDPOINT
export HF_HUB_DISABLE_TELEMETRY=1
TARGET=14467165872   # bytes of model.safetensors (lerobot/pi05_base)
BLOB="/home/zhoulin/.cache/huggingface/hub/models--lerobot--pi05_base/blobs"
for attempt in $(seq 1 500); do
  SZ=$(ls -la "$BLOB"/*.incomplete 2>/dev/null | awk '{print $5}' | head -1)
  [ -z "$SZ" ] && SZ=0
  PCT=$(python3 -c "print(f'{$SZ/$TARGET*100:.1f}%')" 2>/dev/null || echo "?")
  echo "[$(date +%H:%M:%S)] attempt $attempt  downloaded=$SZ ($PCT)"
  if python - <<'PY' 2>&1; then
    from huggingface_hub import snapshot_download
    p = snapshot_download("lerobot/pi05_base")
    print("COMPLETE:", p)
    import os
    os._exit(0)
  PY
    echo "[$(date +%H:%M:%S)] SNAPSHOT COMPLETE"
    exit 0
  fi
  sleep 3
done
echo "gave up after 500 attempts"
exit 1