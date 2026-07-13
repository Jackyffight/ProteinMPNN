#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

EXPECTED_BENCHMARK_ID="${EXPECTED_BENCHMARK_ID:-pdb-valid-7136a4ecae1956027aa6}"
RUNTIME_ROOT="${ESMFOLD2_RUNTIME_ROOT:-$MPNN_WORKSPACE/structure_runtime/esmfold2-fast}"
MODE="${1:-smoke}"
DRY_RUN=false
if [ $# -gt 2 ]; then
  echo "Usage: scripts/run_esmfold2_fast.sh [smoke|full] [--dry-run]" >&2
  exit 2
fi
if [ "$MODE" = "--dry-run" ] || [ "$MODE" = "--dry_run" ]; then
  MODE="smoke"
  DRY_RUN=true
elif [ "${2:-}" = "--dry-run" ] || [ "${2:-}" = "--dry_run" ]; then
  DRY_RUN=true
elif [ $# -gt 1 ]; then
  echo "Usage: scripts/run_esmfold2_fast.sh [smoke|full] [--dry-run]" >&2
  exit 2
fi
if [ "$MODE" != "smoke" ] && [ "$MODE" != "full" ]; then
  echo "Error: mode must be smoke or full, got: $MODE" >&2
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
observed_benchmark_id="$($INSPECT_PYTHON -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["benchmark_id"])' "$SUITE_PATH")"
if [ "$observed_benchmark_id" != "$EXPECTED_BENCHMARK_ID" ]; then
  echo "Error: benchmark ID mismatch." >&2
  echo "expected: $EXPECTED_BENCHMARK_ID" >&2
  echo "observed: $observed_benchmark_id" >&2
  exit 1
fi

RUN_STEM="esmfold2-fast-${EXPECTED_BENCHMARK_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROTEINMPNN_OUTPUT_ROOT/benchmarks/${RUN_STEM}-${MODE}-l3-s50-seed42}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$PROTEINMPNN_OUTPUT_ROOT/benchmarks/${RUN_STEM}-smoke-l3-s50-seed42}"
RUNTIME_PYTHON="$RUNTIME_ROOT/venv/bin/python"

if [ "$MODE" = "full" ]; then
  smoke_manifest="$SMOKE_OUTPUT_DIR/run-manifest.json"
  if [ ! -f "$smoke_manifest" ]; then
    echo "Error: smoke run not found: $SMOKE_OUTPUT_DIR" >&2
    echo "Run scripts/run_esmfold2_fast.sh smoke first." >&2
    exit 1
  fi
fi

command=(
  "$RUNTIME_PYTHON"
  -m protein_mrna_pipeline
  run-esmfold2-benchmark
  --suite "$SUITE_PATH"
  --output-dir "$OUTPUT_DIR"
  --runtime-root "$RUNTIME_ROOT"
  --mode "$MODE"
  --seed 42
  --chunk-size 64
  --num-loops 3
  --num-sampling-steps 50
)
if [ "${RETRY_FAILED:-0}" = 1 ]; then
  command+=(--retry-failed)
fi

echo "mode: $MODE"
echo "benchmark_id: $observed_benchmark_id"
echo "suite: $SUITE_PATH"
echo "runtime_root: $RUNTIME_ROOT"
echo "output_dir: $OUTPUT_DIR"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-0}"
echo "parameters: loops=3 sampling_steps=50 diffusion_samples=1 chunk_size=64 seed=42"

if [ "$DRY_RUN" = true ]; then
  printf 'command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi
if [ ! -x "$RUNTIME_PYTHON" ] || [ ! -f "$RUNTIME_ROOT/runtime-manifest.json" ]; then
  echo "Error: pinned ESMFold2 runtime is not ready: $RUNTIME_ROOT" >&2
  echo "Run scripts/setup_esmfold2_fast_runtime.sh first." >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-0}" == *,* ]]; then
  echo "Error: this bounded runner is single-GPU; expose exactly one device." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$RUNTIME_ROOT/hf-home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO_ROOT/protein_mrna_pipeline/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$MODE" = "full" ]; then
  "$RUNTIME_PYTHON" -m protein_mrna_pipeline verify-esmfold2-run \
    --suite "$SUITE_PATH" \
    --output-dir "$SMOKE_OUTPUT_DIR" \
    --mode smoke
fi
"${command[@]}"
