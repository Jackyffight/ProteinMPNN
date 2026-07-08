#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
./run_train.sh full \
  --env devbox \
  --devices "${DEVICES:-0}" \
  --num-epochs "${NUM_EPOCHS:-1}" \
  --num-examples "${NUM_EXAMPLES:-1000}" \
  --batch-tokens "${BATCH_TOKENS:-6000}" \
  --run-name "${RUN_NAME:-proteinmpnn-full-sanity}"
