#!/usr/bin/env bash
# safe_launch.sh - NEVER deletes data. Archives old traces to
# data/archive/YYYYMMDD-HHMMSS/ before starting a fresh run, then
# launches the distiller with a raised fd limit.
#
# Usage: ./safe_launch.sh [--count N] [--out-dir DIR] [--no-archive] [other distill flags]
set -euo pipefail
cd "$(dirname "$0")"

# Load provider credentials if a local .env exists (never committed).
if [ -f ".env" ]; then
  set -a; . ./.env; set +a
  echo "[safe_launch] loaded credentials from .env"
fi

COUNT=15000
ARCHIVE=1
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --count)
      COUNT="$2"
      shift 2
      ;;
    --no-archive)
      ARCHIVE=0
      shift
      ;;
    --out-dir)
      EXTRA+=("$1" "$2")
      shift 2
      ;;
    --wipe)
      echo "[safe_launch] refusing --wipe; archive via src/archive_data.py instead" >&2
      exit 1
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

PY=""
if [ -n "${FORGE_PYTHON:-}" ] && [ -x "$FORGE_PYTHON" ]; then
  PY="$FORGE_PYTHON"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="$(command -v python3 || true)"
  if [ -z "$PY" ]; then
    echo "[safe_launch] ERROR: no python found" >&2
    exit 1
  fi
fi

RAW="data/raw"
if [ "$ARCHIVE" = "1" ]; then
  "$PY" src/archive_data.py --label pre-launch || true
else
  echo "[safe_launch] no existing traces to archive (or --no-archive)"
fi

ulimit -n 65536
echo "[safe_launch] launching: count=$COUNT, fd_limit=$(ulimit -n)"

exec env PYTHONUNBUFFERED=1 "$PY" src/distill_tools.py --count "$COUNT" "${EXTRA[@]}"
