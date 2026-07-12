#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

if [ $# -gt 1 ]; then
  echo "Usage: scripts/promote_2026_stage2a.sh [formal-run-dir]" >&2
  exit 2
fi

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$(
    find "$PROTEINMPNN_OUTPUT_ROOT" \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name 'proteinmpnn-2026-stage2a-v48-a100-*' \
      | sort \
      | tail -n 1
  )"
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "Error: formal stage2a run directory not found: ${RUN_DIR:-none}" >&2
  exit 1
fi

PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"
PROMOTION_DIR="${PROMOTION_DIR:-$PROTEINMPNN_OUTPUT_ROOT/promoted/proteinmpnn-2026-stage2a}"

exec "$PYTHON_BIN" "$REPO_ROOT/repo/training/promote_stage2a_checkpoint.py" \
  --run-dir "$RUN_DIR" \
  --destination-dir "$PROMOTION_DIR" \
  --model-id proteinmpnn-2026-stage2a
