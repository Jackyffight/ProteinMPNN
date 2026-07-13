#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/env_nas.sh"

if [ "$#" -ne 1 ]; then
  echo "Usage: design_flow_stage3/run_stage3_esmfold2.sh /absolute/path/to/stage3-job.tar.gz" >&2
  exit 2
fi

JOB_ARCHIVE="$1"
if [ "${JOB_ARCHIVE#/}" = "$JOB_ARCHIVE" ] || [ ! -f "$JOB_ARCHIVE" ]; then
  echo "Error: job archive must be an existing absolute path: $JOB_ARCHIVE" >&2
  exit 1
fi

RUNTIME_ROOT="/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/structure_runtime/esmfold2-fast"
WORK_ROOT="/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/runs/design-flow-stage3"
RUNTIME_PYTHON="$RUNTIME_ROOT/venv/bin/python"
RUNNER="$SCRIPT_DIR/esmfold2_job_runner.py"

if [ ! -x "$RUNTIME_PYTHON" ] || [ ! -f "$RUNTIME_ROOT/runtime-manifest.json" ]; then
  echo "Error: pinned ESMFold2-Fast runtime is not ready: $RUNTIME_ROOT" >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-0}" == *,* ]]; then
  echo "Error: expose exactly one GPU; this worker is sequential and resumable." >&2
  exit 1
fi

ARCHIVE_SHA256="$(sha256sum "$JOB_ARCHIVE" | awk '{print $1}')"
JOB_DIR="$WORK_ROOT/jobs/${ARCHIVE_SHA256:0:16}"
mkdir -p "$WORK_ROOT/jobs" "$WORK_ROOT/results" "$WORK_ROOT/exports"

if [ ! -d "$JOB_DIR" ]; then
  "$RUNTIME_PYTHON" "$RUNNER" unpack-job \
    --archive "$JOB_ARCHIVE" \
    --destination "$JOB_DIR"
fi
JOB_IDENTITY="$($RUNTIME_PYTHON "$RUNNER" inspect-job --job-dir "$JOB_DIR")"
OUTPUT_DIR="$WORK_ROOT/results/$JOB_IDENTITY"
RESULT_ARCHIVE="$WORK_ROOT/exports/${JOB_IDENTITY}-results.tar.gz"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$RUNTIME_ROOT/hf-home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "job_archive: $JOB_ARCHIVE"
echo "job_archive_sha256: $ARCHIVE_SHA256"
echo "job_identity: $JOB_IDENTITY"
echo "runtime_root: $RUNTIME_ROOT"
echo "output_dir: $OUTPUT_DIR"
echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"

run_args=(
  "$RUNTIME_PYTHON"
  "$RUNNER"
  run
  --job-dir "$JOB_DIR"
  --output-dir "$OUTPUT_DIR"
  --runtime-root "$RUNTIME_ROOT"
)
if [ "${RETRY_FAILED:-0}" = 1 ]; then
  run_args+=(--retry-failed)
fi
"${run_args[@]}"
"$RUNTIME_PYTHON" "$RUNNER" verify \
  --job-dir "$JOB_DIR" \
  --output-dir "$OUTPUT_DIR"
"$RUNTIME_PYTHON" "$RUNNER" pack-results \
  --job-dir "$JOB_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --archive "$RESULT_ARCHIVE"

echo "stage3_result_archive: $RESULT_ARCHIVE"
echo "stage3_result_archive_sha256: $(sha256sum "$RESULT_ARCHIVE" | awk '{print $1}')"
