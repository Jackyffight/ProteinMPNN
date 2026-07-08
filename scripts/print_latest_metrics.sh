#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 2
fi

RUN_DIR="$1"
METRICS="$RUN_DIR/metrics.jsonl"
EVAL_RESULTS="$RUN_DIR/eval_results.json"

if [ ! -f "$METRICS" ]; then
  echo "Missing metrics file: $METRICS" >&2
  exit 1
fi

echo "Latest metrics:"
tail -n 1 "$METRICS"

if [ -f "$EVAL_RESULTS" ]; then
  echo "Eval results (best epoch by validation loss):"
  cat "$EVAL_RESULTS"
fi
