#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <checkpoint.pt>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT="$1"
RUN_DIR="$(cd "$(dirname "$CHECKPOINT")/.." && pwd)"
RUN_NAME="$(basename "$RUN_DIR")"

cd "$REPO_ROOT"
./run_train.sh full \
  --env devbox \
  --devices "${DEVICES:-0}" \
  --output-dir "$RUN_DIR" \
  --run-name "$RUN_NAME" \
  --resume "$CHECKPOINT"
