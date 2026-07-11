#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conservative first continuation stage: consume every available training
# cluster, cross the optimizer warmup, then evaluate before training longer.
export RUN_NAME="${RUN_NAME:-proteinmpnn-2026-v1-stage1-v48-a100-$(date +%Y%m%d%H%M%S)}"
export NUM_EPOCHS="${NUM_EPOCHS:-20}"
export NUM_EXAMPLES="${NUM_EXAMPLES:-1000000}"
export BATCH_TOKENS="${BATCH_TOKENS:-10000}"
export MAX_PROTEIN_LENGTH="${MAX_PROTEIN_LENGTH:-2000}"
export SAVE_EVERY="${SAVE_EVERY:-5}"
export RELOAD_EVERY="${RELOAD_EVERY:-2}"
export LOADER_WORKERS="${LOADER_WORKERS:-0}"
export PREFETCH_WORKERS="${PREFETCH_WORKERS:-1}"
export PREFETCH_BATCHES="${PREFETCH_BATCHES:-1}"

exec "$SCRIPT_DIR/run_2026_v1_pilot_a100.sh" "$@"
