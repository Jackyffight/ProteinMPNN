#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
./run_train.sh smoke \
  --env local \
  --devices "${DEVICES:-0}" \
  --run-name "${RUN_NAME:-proteinmpnn-smoke}"
