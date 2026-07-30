#!/usr/bin/env bash
set -Eeuo pipefail

BASE_PID=""
LAUNCHER_PID=""

shutdown() {
  trap - SIGTERM SIGINT EXIT
  if [[ -n "$LAUNCHER_PID" ]]; then
    kill -TERM "$LAUNCHER_PID" 2>/dev/null || true
  fi
  if [[ -n "$BASE_PID" ]]; then
    kill -TERM "$BASE_PID" 2>/dev/null || true
  fi
  wait "$LAUNCHER_PID" "$BASE_PID" 2>/dev/null || true
}

trap shutdown SIGTERM SIGINT EXIT

echo "Starting stock RunPod ComfyUI services..."
/usr/local/bin/runpod-base-start.sh &
BASE_PID=$!

echo "Starting 10sorLabs Model Grabber on port ${LAUNCHER_PORT:-3000}..."
python3.12 -m launcher.bootstrap &
LAUNCHER_PID=$!

wait -n "$BASE_PID" "$LAUNCHER_PID"

