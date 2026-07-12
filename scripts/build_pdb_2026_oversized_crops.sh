#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/datasets/proteinmpnn_custom}"
VERSION_ID="${VERSION_ID:-proteinmpnn_pdb_20260708}"
DATASET_ID="${DATASET_ID:-}"
REFERENCE_DATASET="${REFERENCE_DATASET:-}"
DEFERRED_MANIFEST="${DEFERRED_MANIFEST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-2000}"
MIN_CONTEXT_CROP_LENGTH="${MIN_CONTEXT_CROP_LENGTH:-30}"
WORKER_RECYCLE_TASKS="${WORKER_RECYCLE_TASKS:-25}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-1g}"
MIN_AVAILABLE_MEMORY_GB="${MIN_AVAILABLE_MEMORY_GB:-8}"
LIMIT="${LIMIT:-}"
FORCE=false
VALIDATE=true

usage() {
  cat <<'EOF'
Usage:
  scripts/build_pdb_2026_oversized_crops.sh [options]

Build stage2a spatial crops from structures deferred by the validated v1 build.
The parser is intentionally fixed at one worker and one in-flight file.

Options:
  --data-root <dir>         Dataset root containing the snapshot.
  --version-id <id>         Source snapshot id. Default: proteinmpnn_pdb_20260708.
  --dataset-id <id>         Output artifact version id.
  --reference-dataset <dir> Validated v1 dataset used to inherit splits.
  --deferred-manifest <file>
                            v1 build_deferred_oversized.jsonl.
  --output-dir <dir>        Default: processed/proteinmpnn_tar_shards_stage2a_v1.
  --max-context-length <n>  Total retained residue budget. Default: 2000.
  --min-context-crop-length <n>
                            Minimum contiguous context window. Default: 30.
  --worker-recycle-tasks <n>
                            Restart the parser after n files. Default: 25.
  --min-available-memory-gb <n>
                            Refuse to start below this MemAvailable value. Default: 8.
  --max-shard-size <size>   Tar shard target size. Default: 1g.
  --limit <n>               Debug: process only the first n deferred files.
  --force                   Replace the output directory.
  --no-validate             Skip semantic validation after the build.
  -h, --help                Show this help.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root|--data_root) DATA_ROOT="$2"; shift 2 ;;
    --version-id|--version_id) VERSION_ID="$2"; shift 2 ;;
    --dataset-id|--dataset_id) DATASET_ID="$2"; shift 2 ;;
    --reference-dataset|--reference_dataset) REFERENCE_DATASET="$2"; shift 2 ;;
    --deferred-manifest|--deferred_manifest) DEFERRED_MANIFEST="$2"; shift 2 ;;
    --output-dir|--output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --max-context-length|--max_context_length) MAX_CONTEXT_LENGTH="$2"; shift 2 ;;
    --min-context-crop-length|--min_context_crop_length) MIN_CONTEXT_CROP_LENGTH="$2"; shift 2 ;;
    --worker-recycle-tasks|--worker_recycle_tasks) WORKER_RECYCLE_TASKS="$2"; shift 2 ;;
    --min-available-memory-gb|--min_available_memory_gb) MIN_AVAILABLE_MEMORY_GB="$2"; shift 2 ;;
    --max-shard-size|--max_shard_size) MAX_SHARD_SIZE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    --no-validate|--no_validate) VALIDATE=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

VERSION_DIR="$DATA_ROOT/$VERSION_ID"
if [ -z "$DATASET_ID" ]; then
  DATASET_ID="${VERSION_ID}_stage2a_spatial_crop_v1"
fi
if [ -z "$REFERENCE_DATASET" ]; then
  REFERENCE_DATASET="$VERSION_DIR/processed/proteinmpnn_tar_shards_v1"
fi
if [ -z "$DEFERRED_MANIFEST" ]; then
  DEFERRED_MANIFEST="$REFERENCE_DATASET/build_deferred_oversized.jsonl"
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$VERSION_DIR/processed/proteinmpnn_tar_shards_stage2a_v1"
fi
CLUSTER_FILE="${CLUSTER_FILE:-$VERSION_DIR/raw/sequence_clusters/clusters-by-entity-30.txt}"
ENTRIES_INDEX="${ENTRIES_INDEX:-$VERSION_DIR/raw/metadata/entries.idx}"
LOG_DIR="$VERSION_DIR/logs"

for path in "$DEFERRED_MANIFEST" "$REFERENCE_DATASET/manifest.json" \
  "$REFERENCE_DATASET/validation.json" \
  "$REFERENCE_DATASET/list.csv" "$REFERENCE_DATASET/valid_clusters.txt" \
  "$REFERENCE_DATASET/test_clusters.txt" "$CLUSTER_FILE" "$ENTRIES_INDEX"; do
  if [ ! -s "$path" ]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

if ! [[ "$MIN_AVAILABLE_MEMORY_GB" =~ ^[0-9]+$ ]]; then
  echo "--min-available-memory-gb must be a non-negative integer" >&2
  exit 2
fi
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
required_kib=$((MIN_AVAILABLE_MEMORY_GB * 1024 * 1024))
if [ "$available_kib" -lt "$required_kib" ]; then
  echo "Refusing to start: MemAvailable is below ${MIN_AVAILABLE_MEMORY_GB} GiB." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
args=(
  --deferred-manifest "$DEFERRED_MANIFEST"
  --reference-dataset "$REFERENCE_DATASET"
  --out-dir "$OUTPUT_DIR"
  --version-id "$DATASET_ID"
  --cluster-file "$CLUSTER_FILE"
  --entries-index "$ENTRIES_INDEX"
  --max-context-length "$MAX_CONTEXT_LENGTH"
  --min-context-crop-length "$MIN_CONTEXT_CROP_LENGTH"
  --worker-recycle-tasks "$WORKER_RECYCLE_TASKS"
  --max-shard-size "$MAX_SHARD_SIZE"
)
if [ -n "$LIMIT" ]; then
  args+=(--limit "$LIMIT")
fi
if [ "$FORCE" = true ]; then
  args+=(--force)
fi

echo "source_snapshot: $VERSION_DIR"
echo "reference_dataset: $REFERENCE_DATASET"
echo "deferred_manifest: $DEFERRED_MANIFEST"
echo "output_dir: $OUTPUT_DIR"
echo "parser_workers: 1"
echo "max_in_flight: 1"
echo "worker_recycle_tasks: $WORKER_RECYCLE_TASKS"
echo "MemAvailable_kib: $available_kib"

EXTRA_PYTHONPATH="$REPO_ROOT/repo/training"
LOCAL_DEPS="$WORKSPACE_ROOT/.pdbbuild_deps"
if [ -d "$LOCAL_DEPS" ]; then
  EXTRA_PYTHONPATH="$LOCAL_DEPS:$EXTRA_PYTHONPATH"
fi
export PYTHONPATH="${PYTHONPATH:-}:$EXTRA_PYTHONPATH"

log_path="$LOG_DIR/build_oversized_crops_$(date +%Y%m%d%H%M%S).log"
if [ -x /usr/bin/time ]; then
  /usr/bin/time -v \
    python "$REPO_ROOT/repo/training/build_pdb_oversized_crop_tar_dataset.py" \
      "${args[@]}" 2>&1 | tee "$log_path"
else
  python "$REPO_ROOT/repo/training/build_pdb_oversized_crop_tar_dataset.py" \
    "${args[@]}" 2>&1 | tee "$log_path"
fi

if [ "$VALIDATE" = true ]; then
  python "$REPO_ROOT/repo/training/validate_tar_shard_dataset.py" \
    --dataset-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/validation.json"
fi

echo "stage2a_dataset: $OUTPUT_DIR"
echo "build_log: $log_path"
