#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
./run_train.sh a100 \
  --env devbox \
  --devices "${DEVICES:-0}" \
  --run-name "${RUN_NAME:-proteinmpnn-v48-noise020-a100}" \
  --batch-tokens "${BATCH_TOKENS:-10000}" \
  --loader-workers "${LOADER_WORKERS:-0}" \
  --prefetch-workers "${PREFETCH_WORKERS:-2}"
