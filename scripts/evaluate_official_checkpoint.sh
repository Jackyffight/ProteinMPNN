#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/scripts/env_nas.sh" ]; then
  source "$ROOT/scripts/env_nas.sh"
fi

PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"
DEFAULT_CHECKPOINT="$ROOT/repo/vanilla_model_weights/v_48_020.pt"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"
SPLIT="${SPLIT:-valid}"
OUTPUT="${OUTPUT:-$ROOT/runs/baselines/official-v48-020-${SPLIT}.json}"

if [ -z "${DATA_DIR:-}" ]; then
  NAS_DATA_DIR="${PROTEINMPNN_DATA_ROOT:-}/pdb_2021aug02"
  LOCAL_DATA_DIR="$(cd "$ROOT/.." && pwd)/datasets/proteinmpnn/pdb_2021aug02"
  if [ -d "$NAS_DATA_DIR" ]; then
    DATA_DIR="$NAS_DATA_DIR"
  else
    DATA_DIR="$LOCAL_DATA_DIR"
  fi
fi

LOCAL_DEPS="$(cd "$ROOT/.." && pwd)/.pdbbuild_deps"
if [ -d "$LOCAL_DEPS" ]; then
  export PYTHONPATH="$LOCAL_DEPS${PYTHONPATH:+:$PYTHONPATH}"
fi

if [ "$CHECKPOINT" = "$DEFAULT_CHECKPOINT" ]; then
  "$ROOT/scripts/ensure_official_checkpoint.sh" "$CHECKPOINT"
fi

exec "$PYTHON_BIN" "$ROOT/repo/training/evaluate_checkpoint.py" \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --split "$SPLIT" \
  --max-examples "$MAX_EXAMPLES" \
  --output "$OUTPUT" \
  "$@"
