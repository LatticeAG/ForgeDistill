#!/usr/bin/env bash
# safe_launch.sh - NEVER deletes data. Archives old traces to
# data/archive/YYYYMMDD-HHMMSS/ before starting a fresh run, then
# launches the distiller with a raised fd limit.
#
# Usage: ./safe_launch.sh [--count N] [--no-archive]
set -euo pipefail
cd "$(dirname "$0")"

# Load provider credentials if a local .env exists (never committed).
if [ -f ".env" ]; then
  set -a; . ./.env; set +a
  echo "[safe_launch] loaded credentials from .env"
fi

COUNT="${1:-}"
if [ "$COUNT" = "--count" ]; then COUNT="$2"; fi
ARCHIVE=1
for a in "$@"; do
  [ "$a" = "--no-archive" ] && ARCHIVE=0
done
[ -n "$COUNT" ] || COUNT=15000

RAW="data/raw"
if [ "$ARCHIVE" = "1" ] && ls "$RAW"/traces_*.jsonl >/dev/null 2>&1; then
  TS=$(date +%Y%m%d-%H%M%S)
  DEST="data/archive/$TS"
  mkdir -p "$DEST"
  mv "$RAW"/traces_*.jsonl "$RAW"/checkpoint_*.json "$DEST"/ 2>/dev/null || true
  echo "[safe_launch] archived existing traces -> $DEST"
else
  echo "[safe_launch] no existing traces to archive (or --no-archive)"
fi

ulimit -n 65536
echo "[safe_launch] launching: count=$COUNT, fd_limit=$(ulimit -n)"

PY=""
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"
elif [ -x "/home/ubuntu/.hermes/hermes-agent/venv/bin/python" ]; then PY="/home/ubuntu/.hermes/hermes-agent/venv/bin/python"
else
  echo "[safe_launch] ERROR: no python found" >&2
  exit 1
fi
exec env PYTHONUNBUFFERED=1 "$PY" src/distill_tools.py --count "$COUNT"
