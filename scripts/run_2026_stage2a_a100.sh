#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full stage2a continuation remains deliberately short. Every epoch is retained
# and the dual validation gate chooses a checkpoint after training.
export RUN_NAME="${RUN_NAME:-proteinmpnn-2026-stage2a-v48-a100-$(date +%Y%m%d%H%M%S)}"
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export NUM_EXAMPLES="${NUM_EXAMPLES:-1000000}"
export LR_FACTOR="${LR_FACTOR:-0.25}"
export WARMUP_STEPS="${WARMUP_STEPS:-4000}"
export SAVE_EVERY="${SAVE_EVERY:-1}"
export RELOAD_EVERY="${RELOAD_EVERY:-1}"

exec "$SCRIPT_DIR/run_2026_stage2a_pilot_a100.sh" "$@"
