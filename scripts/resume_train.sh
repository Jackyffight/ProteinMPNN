#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <checkpoint.pt> [preset]   preset: full|v100|a100 (default: inferred from run name)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT="$1"
RUN_DIR="$(cd "$(dirname "$CHECKPOINT")/.." && pwd)"
RUN_NAME="$(basename "$RUN_DIR")"

# Preserve the original run's token budget: resuming a *-v100 run with the 'full'/a100
# preset (10000 tokens) can OOM a V100 that started at 6000. Infer the preset from the
# run name, overridable by a positional arg or the PRESET env var.
PRESET="${2:-${PRESET:-}}"
if [ -z "$PRESET" ]; then
  case "$RUN_NAME" in
    *v100*) PRESET="v100" ;;
    *a100*) PRESET="a100" ;;
    *) PRESET="full" ;;
  esac
fi

cd "$REPO_ROOT"
./run_train.sh "$PRESET" \
  --env devbox \
  --devices "${DEVICES:-0}" \
  --output-dir "$RUN_DIR" \
  --run-name "$RUN_NAME" \
  --resume "$CHECKPOINT"
