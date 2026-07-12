#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

if [ $# -gt 1 ]; then
  echo "Usage: scripts/validate_2026_stage2a_dataset.sh [dataset-dir]" >&2
  exit 2
fi

DATA_DIR="${1:-$PROTEINMPNN_STAGE2A_DATA_DIR}"
PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: stage2a dataset directory not found: $DATA_DIR" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$REPO_ROOT/repo/training/validate_tar_shard_dataset.py" \
  --dataset-dir "$DATA_DIR" \
  --output "$DATA_DIR/validation.json"
