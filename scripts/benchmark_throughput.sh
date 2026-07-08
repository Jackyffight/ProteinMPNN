#!/usr/bin/env bash
# Benchmark ProteinMPNN training throughput with short controlled runs.
#
# Usage:
#   scripts/benchmark_throughput.sh
#   scripts/benchmark_throughput.sh smoke
#   scripts/benchmark_throughput.sh full
#
# Modes:
#   smoke: sample-data cases, 50 examples each
#   quick: default, full-data V100/A100 parameter probes, 1000 examples each
#   full : quick + larger token-budget probes, 5000 examples each

set -u

MODE="${1:-quick}"
if [ "$MODE" != "smoke" ] && [ "$MODE" != "quick" ] && [ "$MODE" != "full" ]; then
  echo "Usage: $0 [smoke|quick|full]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"
TIMESTAMP="$(date +%Y%m%d%H%M%S)"

DATA_ROOT="$PROTEINMPNN_DATA_ROOT"
SAMPLE_DATA_DIR="$DATA_ROOT/pdb_2021aug02_sample"
FULL_DATA_DIR="$DATA_ROOT/pdb_2021aug02"
BENCH_ROOT="${PROTEINMPNN_BENCH_ROOT:-$PROTEINMPNN_OUTPUT_ROOT/benchmarks/throughput-${MODE}-${TIMESTAMP}}"
SUMMARY_FILE="${BENCH_ROOT}/summary.tsv"
DEVICES="${DEVICES:-0}"

case "$MODE" in
  smoke) DEFAULT_EXAMPLES=50 ;;
  quick) DEFAULT_EXAMPLES=1000 ;;
  full) DEFAULT_EXAMPLES=5000 ;;
esac
NUM_EXAMPLES="${NUM_EXAMPLES:-$DEFAULT_EXAMPLES}"

# Smoke benchmark uses the sample dir if provisioned, else the full dataset (smoke
# preset runs in debug mode, 50 examples). quick/full only need the full dataset.
if [ ! -d "$SAMPLE_DATA_DIR" ]; then
  SAMPLE_DATA_DIR="$FULL_DATA_DIR"
fi
if [ "$MODE" = "smoke" ]; then
  if [ ! -d "$SAMPLE_DATA_DIR" ]; then
    echo "No data dir found for smoke benchmark: $SAMPLE_DATA_DIR" >&2
    exit 1
  fi
elif [ ! -d "$FULL_DATA_DIR" ]; then
  echo "Full data dir not found: $FULL_DATA_DIR" >&2
  exit 1
fi

mkdir -p "$BENCH_ROOT"
printf "case\tstatus\tpreset\tdevices\tbatch_tokens\tloader_workers\tprefetch_workers\tnum_examples\tseconds\texamples_per_second\ttrain_perplexity\tvalid_perplexity\ttrain_acc\tvalid_acc\toutput_dir\n" > "$SUMMARY_FILE"

echo "Benchmark root: $BENCH_ROOT"
echo "Summary: $SUMMARY_FILE"
echo "Mode: $MODE, num_examples: $NUM_EXAMPLES"

run_case() {
  local name="$1"
  local preset="$2"
  local data_dir="$3"
  local batch_tokens="$4"
  local loader_workers="$5"
  local prefetch_workers="$6"

  local run_name="bench-${name}"
  local output_dir="${BENCH_ROOT}/${run_name}"
  local log_file="${BENCH_ROOT}/${run_name}.log"
  local gpu_log="${BENCH_ROOT}/${run_name}.gpu.csv"
  local monitor_pid=""

  mkdir -p "$output_dir"
  echo ""
  echo "===== case: $name ====="
  echo "log: $log_file"
  echo "gpu_log: $gpu_log"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
      --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
      --format=csv \
      -l 5 > "$gpu_log" 2>/dev/null &
    monitor_pid="$!"
  fi

  local cmd=(
    "$REPO_ROOT/run_train.sh"
    "$preset"
    --env benchmark
    --data-dir "$data_dir"
    --output-dir "$output_dir"
    --run-name "$run_name"
    --devices "$DEVICES"
    --num-epochs 1
    --num-examples "$NUM_EXAMPLES"
    --batch-tokens "$batch_tokens"
    --loader-workers "$loader_workers"
    --prefetch-workers "$prefetch_workers"
    --reload-every 1000000
    --save-every 1000000
    --seed 42
  )

  set +e
  "${cmd[@]}" > "$log_file" 2>&1
  local status=$?
  set -e

  if [ -n "$monitor_pid" ]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
  fi

  if [ "$status" -ne 0 ]; then
    printf "%s\tfailed:%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t\t\t\t\t\t%s\n" \
      "$name" "$status" "$preset" "$DEVICES" "$batch_tokens" "$loader_workers" "$prefetch_workers" "$NUM_EXAMPLES" "$output_dir" >> "$SUMMARY_FILE"
    echo "case failed: $name status=$status"
    tail -n 40 "$log_file" || true
    return 0
  fi

  python - "$name" "$preset" "$DEVICES" "$batch_tokens" "$loader_workers" "$prefetch_workers" "$NUM_EXAMPLES" "$output_dir" "$SUMMARY_FILE" <<'PY'
import json
import sys
from pathlib import Path

name, preset, devices, batch_tokens, loader_workers, prefetch_workers, num_examples, output_dir, summary_file = sys.argv[1:]
metrics_file = Path(output_dir) / "metrics.jsonl"
metrics = {}
if metrics_file.exists():
    lines = [line for line in metrics_file.read_text().splitlines() if line.strip()]
    if lines:
        metrics = json.loads(lines[-1])

seconds = float(metrics.get("seconds") or 0.0)
examples = int(metrics.get("num_examples_per_epoch") or num_examples)
examples_per_second = examples / seconds if seconds > 0 else 0.0
row = [
    name,
    "ok",
    preset,
    devices,
    batch_tokens,
    loader_workers,
    prefetch_workers,
    str(examples),
    str(seconds),
    str(examples_per_second),
    str(metrics.get("train_perplexity", "")),
    str(metrics.get("validation_perplexity", "")),
    str(metrics.get("train_accuracy", "")),
    str(metrics.get("validation_accuracy", "")),
    output_dir,
]
with open(summary_file, "a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
  tail -n 20 "$log_file" || true
}

set -e

if [ "$MODE" = "smoke" ]; then
  run_case "sample_b1000_lw0_pw4" "smoke" "$SAMPLE_DATA_DIR" "1000" "0" "4"
  run_case "sample_b1000_lw4_pw4" "smoke" "$SAMPLE_DATA_DIR" "1000" "4" "4"
else
  run_case "v100_b4000_lw4_pw8" "v100" "$FULL_DATA_DIR" "4000" "4" "8"
  run_case "v100_b6000_lw4_pw8" "v100" "$FULL_DATA_DIR" "6000" "4" "8"
  run_case "v100_b8000_lw4_pw8" "v100" "$FULL_DATA_DIR" "8000" "4" "8"
  run_case "a100_b10000_lw4_pw12" "a100" "$FULL_DATA_DIR" "10000" "4" "12"
  run_case "a100_b10000_lw8_pw16" "a100" "$FULL_DATA_DIR" "10000" "8" "16"
fi

if [ "$MODE" = "full" ]; then
  run_case "a100_b12000_lw4_pw12" "a100" "$FULL_DATA_DIR" "12000" "4" "12"
  run_case "a100_b16000_lw4_pw12" "a100" "$FULL_DATA_DIR" "16000" "4" "12"
  run_case "v100_b6000_lw0_pw8" "v100" "$FULL_DATA_DIR" "6000" "0" "8"
fi

echo ""
echo "===== throughput summary ====="
cat "$SUMMARY_FILE"
echo ""
echo "Benchmark root: $BENCH_ROOT"
