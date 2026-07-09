#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

DATA_ROOT="${DATA_ROOT:-$PROTEINMPNN_CUSTOM_DATA_ROOT}"
VERSION_ID="${VERSION_ID:-proteinmpnn_pdb_20260708}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROTEINMPNN_TAR_SHARD_DATA_DIR}"
WORKERS="${WORKERS:-16}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-}"
ASSEMBLY_ID="${ASSEMBLY_ID:-all}"
MAX_RESOLUTION="${MAX_RESOLUTION:-3.5}"
MIN_DATE="${MIN_DATE:-}"
MAX_DATE="${MAX_DATE:-2026-07-08}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-1g}"
LIMIT="${LIMIT:-}"
FORCE=false

usage() {
  cat <<'EOF'
Usage:
  scripts/build_pdb_2026_tar_shards.sh [options]

Build a ProteinMPNN tar-shard dataset directly from existing wwPDB biological
assembly mmCIF files.

Options:
  --version-id <id>       Dataset version id. Default: proteinmpnn_pdb_20260708.
  --data-root <dir>       Custom dataset root. Default: NAS MPNN custom root.
  --output-dir <dir>      Tar-shard output dir. Default: $PROTEINMPNN_TAR_SHARD_DATA_DIR.
  --workers <n>           mmCIF parser workers. Default: 16.
  --max-in-flight <n>     ProcessPool futures in flight. Default: workers * 2.
  --assembly-id <id|all>  Assembly file selector. Default: all.
  --max-resolution <a>    Resolution cutoff. Default: 3.5.
  --min-date <YYYY-MM-DD> Optional deposition lower bound.
  --max-date <YYYY-MM-DD> Deposition upper bound. Default: 2026-07-08.
  --max-shard-size <size> Shard target size. Default: 1g.
  --limit <n>             Debug: process only first n assembly files.
  --force                 Replace output dir if it exists.
  -h, --help              Show this help.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version-id|--version_id) VERSION_ID="$2"; shift 2 ;;
    --data-root|--data_root) DATA_ROOT="$2"; shift 2 ;;
    --output-dir|--output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --max-in-flight|--max_in_flight) MAX_IN_FLIGHT="$2"; shift 2 ;;
    --assembly-id|--assembly_id) ASSEMBLY_ID="$2"; shift 2 ;;
    --max-resolution|--max_resolution) MAX_RESOLUTION="$2"; shift 2 ;;
    --min-date|--min_date) MIN_DATE="$2"; shift 2 ;;
    --max-date|--max_date) MAX_DATE="$2"; shift 2 ;;
    --max-shard-size|--max_shard_size) MAX_SHARD_SIZE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

VERSION_DIR="$DATA_ROOT/$VERSION_ID"
RAW_DIR="$VERSION_DIR/raw/assemblies_mmcif"
CLUSTER_FILE="$VERSION_DIR/raw/sequence_clusters/clusters-by-entity-30.txt"
ENTRIES_INDEX="$VERSION_DIR/raw/metadata/entries.idx"
LOG_DIR="$VERSION_DIR/logs"

if [ ! -d "$RAW_DIR" ]; then
  echo "Missing raw mmCIF dir: $RAW_DIR" >&2
  exit 1
fi
if [ ! -f "$ENTRIES_INDEX" ]; then
  echo "Missing entries index: $ENTRIES_INDEX" >&2
  exit 1
fi
if [ ! -f "$CLUSTER_FILE" ]; then
  echo "Warning: missing cluster file, builder will use sequence-hash fallback: $CLUSTER_FILE" >&2
fi
mkdir -p "$LOG_DIR"

args=(
  --raw-dir "$RAW_DIR"
  --out-dir "$OUTPUT_DIR"
  --version-id "$VERSION_ID"
  --cluster-file "$CLUSTER_FILE"
  --entries-index "$ENTRIES_INDEX"
  --workers "$WORKERS"
  --assembly-id "$ASSEMBLY_ID"
  --max-resolution "$MAX_RESOLUTION"
  --max-date "$MAX_DATE"
  --max-shard-size "$MAX_SHARD_SIZE"
)
if [ -n "$MAX_IN_FLIGHT" ]; then
  args+=(--max-in-flight "$MAX_IN_FLIGHT")
fi
if [ -n "$MIN_DATE" ]; then
  args+=(--min-date "$MIN_DATE")
fi
if [ -n "$LIMIT" ]; then
  args+=(--limit "$LIMIT")
fi
if [ "$FORCE" = true ]; then
  args+=(--force)
fi

echo "version_id: $VERSION_ID"
echo "raw_dir: $RAW_DIR"
echo "output_dir: $OUTPUT_DIR"
echo "workers: $WORKERS"

PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/repo/training" \
python "$REPO_ROOT/repo/training/build_pdb_mmcif_tar_shard_dataset.py" "${args[@]}" \
  2>&1 | tee "$LOG_DIR/build_tar_shards_$(date +%Y%m%d%H%M%S).log"

echo "tar_shard_dataset: $OUTPUT_DIR"
