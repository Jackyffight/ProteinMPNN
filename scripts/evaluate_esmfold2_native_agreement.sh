#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

EXPECTED_BENCHMARK_ID="${EXPECTED_BENCHMARK_ID:-pdb-valid-7136a4ecae1956027aa6}"
RUNTIME_ROOT="${STRUCTURE_METRICS_RUNTIME_ROOT:-$MPNN_WORKSPACE/structure_runtime/native-structure-metrics}"
METRICS_THREADS="${STRUCTURE_METRICS_THREADS:-1}"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate_esmfold2_native_agreement.sh [--dry-run]

Compare the fixed 40-record ESMFold2-Fast full run with the experimental
C-alpha coordinates in proteinmpnn_tar_shards_v1. The run is CPU-only and
resumable. Set RETRY_FAILED=1 to retry records already marked failed.

Optional environment overrides:
  BENCHMARK_DIR PREDICTION_RUN OUTPUT_DIR STRUCTURE_METRICS_RUNTIME_ROOT
  PROTEINMPNN_V1_DATA_DIR EXPECTED_BENCHMARK_ID STRUCTURE_METRICS_THREADS
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|--dry_run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$METRICS_THREADS" =~ ^[1-4]$ ]]; then
  echo "Error: STRUCTURE_METRICS_THREADS must be an integer from 1 to 4." >&2
  exit 2
fi

if [ -z "${BENCHMARK_DIR:-}" ]; then
  shopt -s nullglob
  candidates=("$PROTEINMPNN_OUTPUT_ROOT"/benchmarks/structure-input-valid-n40-s42-*)
  shopt -u nullglob
  if [ "${#candidates[@]}" -eq 0 ]; then
    echo "Error: no 40-record structure benchmark found under runs/benchmarks." >&2
    exit 1
  fi
  BENCHMARK_DIR="$(printf '%s\n' "${candidates[@]}" | sort -r | head -1)"
fi
SUITE_PATH="$BENCHMARK_DIR/benchmark-suite.json"
if [ ! -f "$SUITE_PATH" ]; then
  echo "Error: benchmark suite not found: $SUITE_PATH" >&2
  exit 1
fi

INSPECT_PYTHON="${PROTEINMPNN_PYTHON:-python}"
observed_benchmark_id="$("$INSPECT_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["benchmark_id"])' "$SUITE_PATH")"
if [ "$observed_benchmark_id" != "$EXPECTED_BENCHMARK_ID" ]; then
  echo "Error: benchmark ID mismatch." >&2
  echo "expected: $EXPECTED_BENCHMARK_ID" >&2
  echo "observed: $observed_benchmark_id" >&2
  exit 1
fi

RUN_STEM="esmfold2-fast-${EXPECTED_BENCHMARK_ID}"
PREDICTION_RUN="${PREDICTION_RUN:-$PROTEINMPNN_OUTPUT_ROOT/benchmarks/${RUN_STEM}-full-l3-s50-seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-$PREDICTION_RUN/evaluations/native-structure-agreement-v1}"
RUNTIME_PYTHON="$RUNTIME_ROOT/venv/bin/python"

command=(
  "$RUNTIME_PYTHON"
  -m protein_mrna_pipeline
  evaluate-esmfold2-native
  --suite "$SUITE_PATH"
  --prediction-run "$PREDICTION_RUN"
  --dataset-dir "$PROTEINMPNN_V1_DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --metrics-runtime-root "$RUNTIME_ROOT"
)
if [ "${RETRY_FAILED:-0}" = 1 ]; then
  command+=(--retry-failed)
fi

echo "benchmark_id: $observed_benchmark_id"
echo "suite: $SUITE_PATH"
echo "prediction_run: $PREDICTION_RUN"
echo "dataset_dir: $PROTEINMPNN_V1_DATA_DIR"
echo "metrics_runtime_root: $RUNTIME_ROOT"
echo "output_dir: $OUTPUT_DIR"
echo "gpu_used: false"
echo "cpu_threads: $METRICS_THREADS"

if [ "$DRY_RUN" = true ]; then
  printf 'command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi
if [ ! -x "$RUNTIME_PYTHON" ] || [ ! -f "$RUNTIME_ROOT/runtime-manifest.json" ]; then
  echo "Error: pinned structure metrics runtime is not ready: $RUNTIME_ROOT" >&2
  echo "Run scripts/setup_structure_metrics_runtime.sh first." >&2
  exit 1
fi
if [ ! -f "$PREDICTION_RUN/run-manifest.json" ]; then
  echo "Error: ESMFold2 full run not found: $PREDICTION_RUN" >&2
  exit 1
fi
if [ ! -f "$PROTEINMPNN_V1_DATA_DIR/manifest.json" ]; then
  echo "Error: fixed v1 dataset not found: $PROTEINMPNN_V1_DATA_DIR" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$METRICS_THREADS"
export OPENBLAS_NUM_THREADS="$METRICS_THREADS"
export MKL_NUM_THREADS="$METRICS_THREADS"
export PYTHONPATH="$REPO_ROOT/protein_mrna_pipeline/src${PYTHONPATH:+:$PYTHONPATH}"
"${command[@]}"

echo "summary: $OUTPUT_DIR/summary.json"
