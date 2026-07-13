#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

EXPECTED_BENCHMARK_ID="${EXPECTED_BENCHMARK_ID:-pdb-valid-7136a4ecae1956027aa6}"
METRICS_RUNTIME_ROOT="${STRUCTURE_METRICS_RUNTIME_ROOT:-$MPNN_WORKSPACE/structure_runtime/native-structure-metrics}"
ESMFOLD2_RUNTIME_ROOT="${ESMFOLD2_RUNTIME_ROOT:-$MPNN_WORKSPACE/structure_runtime/esmfold2-fast}"
OFFICIAL_CHECKPOINT="${OFFICIAL_CHECKPOINT:-$REPO_ROOT/repo/vanilla_model_weights/v_48_020.pt}"
STAGE2A_CHECKPOINT="${STAGE2A_CHECKPOINT:-$PROTEINMPNN_OUTPUT_ROOT/promoted/proteinmpnn-2026-stage2a/model.pt}"
OFFICIAL_SHA256="c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
STAGE2A_SHA256="08fc2549004d0e8a8b1ac1983dd4e94772f15445732926d8f7e677a4464ba6f7"
MODE="${1:-all}"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  scripts/run_proteinmpnn_refold_pilot.sh [all|generate|refold|evaluate] [--dry-run]

Generate 32 paired official-v48-020/Stage2a designs on four deterministic valid
backbones, refold them with ESMFold2-Fast, and compute dual-reference metrics.
Every phase resumes its existing identity-bound output.

Optional environment overrides:
  BENCHMARK_DIR NATIVE_PREDICTION_RUN NATIVE_AGREEMENT_DIR PILOT_DIR
  REFOLD_DIR EVALUATION_DIR CUDA_VISIBLE_DEVICES RETRY_FAILED=1
EOF
}

if [ "$MODE" = "-h" ] || [ "$MODE" = "--help" ]; then
  usage
  exit 0
fi
if [ "$MODE" = "--dry-run" ] || [ "$MODE" = "--dry_run" ]; then
  MODE="all"
  DRY_RUN=true
elif [ "${2:-}" = "--dry-run" ] || [ "${2:-}" = "--dry_run" ]; then
  DRY_RUN=true
elif [ $# -gt 1 ]; then
  usage >&2
  exit 2
fi
case "$MODE" in
  all|generate|refold|evaluate) ;;
  *) echo "Error: unknown mode: $MODE" >&2; usage >&2; exit 2 ;;
esac

if [ -z "${BENCHMARK_DIR:-}" ]; then
  shopt -s nullglob
  candidates=("$PROTEINMPNN_OUTPUT_ROOT"/benchmarks/structure-input-valid-n40-s42-*)
  shopt -u nullglob
  if [ "${#candidates[@]}" -eq 0 ]; then
    echo "Error: fixed 40-record benchmark not found." >&2
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
  echo "Error: benchmark ID mismatch: $observed_benchmark_id" >&2
  exit 1
fi

RUN_STEM="esmfold2-fast-${EXPECTED_BENCHMARK_ID}"
NATIVE_PREDICTION_RUN="${NATIVE_PREDICTION_RUN:-$PROTEINMPNN_OUTPUT_ROOT/benchmarks/${RUN_STEM}-full-l3-s50-seed42}"
NATIVE_AGREEMENT_DIR="${NATIVE_AGREEMENT_DIR:-$NATIVE_PREDICTION_RUN/evaluations/native-structure-agreement-v1}"
PILOT_DIR="${PILOT_DIR:-$PROTEINMPNN_OUTPUT_ROOT/benchmarks/proteinmpnn-refold-pilot-${EXPECTED_BENCHMARK_ID}-t010-s4}"
REFOLD_DIR="${REFOLD_DIR:-$PILOT_DIR/refolds/esmfold2-fast-l3-s50-seed42}"
EVALUATION_DIR="${EVALUATION_DIR:-$PILOT_DIR/evaluations/dual-reference-v1}"
METRICS_PYTHON="$METRICS_RUNTIME_ROOT/venv/bin/python"
ESMFOLD2_PYTHON="$ESMFOLD2_RUNTIME_ROOT/venv/bin/python"

generate_command=(
  "$METRICS_PYTHON" -m protein_mrna_pipeline generate-proteinmpnn-refold-pilot
  --suite "$SUITE_PATH"
  --native-agreement-dir "$NATIVE_AGREEMENT_DIR"
  --native-prediction-run "$NATIVE_PREDICTION_RUN"
  --dataset-dir "$PROTEINMPNN_V1_DATA_DIR"
  --official-checkpoint "$OFFICIAL_CHECKPOINT"
  --stage2a-checkpoint "$STAGE2A_CHECKPOINT"
  --output-dir "$PILOT_DIR"
  --metrics-runtime-root "$METRICS_RUNTIME_ROOT"
  --repository-root "$REPO_ROOT"
  --seeds 11 23 42 67
  --temperature 0.1
  --device cuda
)
refold_command=(
  "$ESMFOLD2_PYTHON" -m protein_mrna_pipeline run-proteinmpnn-refolds
  --pilot-dir "$PILOT_DIR"
  --output-dir "$REFOLD_DIR"
  --runtime-root "$ESMFOLD2_RUNTIME_ROOT"
  --seed 42
)
evaluate_command=(
  "$METRICS_PYTHON" -m protein_mrna_pipeline evaluate-proteinmpnn-refolds
  --pilot-dir "$PILOT_DIR"
  --refold-dir "$REFOLD_DIR"
  --output-dir "$EVALUATION_DIR"
  --metrics-runtime-root "$METRICS_RUNTIME_ROOT"
)
if [ "${RETRY_FAILED:-0}" = 1 ]; then
  refold_command+=(--retry-failed)
  evaluate_command+=(--retry-failed)
fi

echo "mode: $MODE"
echo "benchmark_id: $observed_benchmark_id"
echo "suite: $SUITE_PATH"
echo "dataset_dir: $PROTEINMPNN_V1_DATA_DIR"
echo "native_prediction_run: $NATIVE_PREDICTION_RUN"
echo "native_agreement_dir: $NATIVE_AGREEMENT_DIR"
echo "official_checkpoint: $OFFICIAL_CHECKPOINT"
echo "stage2a_checkpoint: $STAGE2A_CHECKPOINT"
echo "pilot_dir: $PILOT_DIR"
echo "refold_dir: $REFOLD_DIR"
echo "evaluation_dir: $EVALUATION_DIR"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-0}"
echo "designs: 4 backbones x 2 models x 4 paired seeds = 32"

if [ "$DRY_RUN" = true ]; then
  for command_name in generate_command refold_command evaluate_command; do
    declare -n command_ref="$command_name"
    printf '%s:' "$command_name"
    printf ' %q' "${command_ref[@]}"
    printf '\n'
  done
  exit 0
fi

if [[ "${CUDA_VISIBLE_DEVICES:-0}" == *,* ]]; then
  echo "Error: expose exactly one GPU for the bounded pilot." >&2
  exit 1
fi
for path in \
  "$METRICS_PYTHON" \
  "$ESMFOLD2_PYTHON" \
  "$METRICS_RUNTIME_ROOT/runtime-manifest.json" \
  "$ESMFOLD2_RUNTIME_ROOT/runtime-manifest.json" \
  "$NATIVE_AGREEMENT_DIR/summary.json" \
  "$NATIVE_PREDICTION_RUN/run-manifest.json"; do
  if [ ! -e "$path" ]; then
    echo "Error: required pilot input not found: $path" >&2
    exit 1
  fi
done

"$SCRIPT_DIR/ensure_official_checkpoint.sh" "$OFFICIAL_CHECKPOINT"
if [ ! -f "$STAGE2A_CHECKPOINT" ]; then
  echo "Error: promoted Stage2a checkpoint not found: $STAGE2A_CHECKPOINT" >&2
  exit 1
fi
observed_official_sha="$(sha256sum "$OFFICIAL_CHECKPOINT" | awk '{print $1}')"
observed_stage2a_sha="$(sha256sum "$STAGE2A_CHECKPOINT" | awk '{print $1}')"
if [ "$observed_official_sha" != "$OFFICIAL_SHA256" ]; then
  echo "Error: official checkpoint SHA256 mismatch." >&2
  exit 1
fi
if [ "$observed_stage2a_sha" != "$STAGE2A_SHA256" ]; then
  echo "Error: Stage2a checkpoint SHA256 mismatch." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$ESMFOLD2_RUNTIME_ROOT/hf-home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO_ROOT/protein_mrna_pipeline/src:$REPO_ROOT/repo${PYTHONPATH:+:$PYTHONPATH}"

if [ "$MODE" = "all" ] || [ "$MODE" = "generate" ]; then
  "${generate_command[@]}"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "refold" ]; then
  "${refold_command[@]}"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "evaluate" ]; then
  CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
    "${evaluate_command[@]}"
  "$SCRIPT_DIR/report_proteinmpnn_refold_pilot.sh"
fi

echo "pilot_complete: $PILOT_DIR"
echo "summary: $EVALUATION_DIR/summary.json"
