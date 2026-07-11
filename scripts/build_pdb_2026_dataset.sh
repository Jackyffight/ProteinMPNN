#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

DATA_ROOT="${DATA_ROOT:-$PROTEINMPNN_CUSTOM_DATA_ROOT}"
VERSION_ID="${VERSION_ID:-proteinmpnn_pdb_20260708}"
WORKERS="${WORKERS:-2}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-2}"
ASSEMBLY_ID="${ASSEMBLY_ID:-all}"
MAX_RESOLUTION="${MAX_RESOLUTION:-3.5}"
MIN_DATE="${MIN_DATE:-}"
MAX_DATE="${MAX_DATE:-2026-07-08}"
LIMIT="${LIMIT:-}"
SKIP_SYNC=false
SKIP_CLUSTERS=false
SKIP_METADATA=false

usage() {
  cat <<'EOF'
Usage:
  scripts/build_pdb_2026_dataset.sh [options]

Build an owned ProteinMPNN dataset from the current 2026 wwPDB biological
assembly mmCIF snapshot.

Options:
  --version-id <id>       Dataset version id. Default: proteinmpnn_pdb_20260708.
  --data-root <dir>       Custom dataset root. Default: NAS MPNN custom root.
  --workers <n>           mmCIF parser workers. Default: 2.
  --max-in-flight <n>     Submitted parser jobs retained in memory. Default: 2.
  --assembly-id <id|all>  Assembly file selector. Default: all.
  --max-resolution <a>    Resolution cutoff. Default: 3.5.
  --min-date <YYYY-MM-DD> Optional deposition lower bound.
  --max-date <YYYY-MM-DD> Deposition upper bound. Default: 2026-07-08.
  --limit <n>             Debug: process only first n assembly files.
  --skip-sync             Use existing raw/assemblies_mmcif.
  --skip-clusters         Use existing raw/sequence_clusters.
  --skip-metadata         Use existing raw/metadata/entries.idx.
  -h, --help              Show this help.

Examples:
  # Full 2026-07-08 snapshot.
  scripts/build_pdb_2026_dataset.sh

  # Only structures deposited in 2026.
  scripts/build_pdb_2026_dataset.sh --min-date 2026-01-01

  # Debug build against assembly1 only.
  scripts/build_pdb_2026_dataset.sh --assembly-id 1 --limit 1000 --skip-sync
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version-id|--version_id) VERSION_ID="$2"; shift 2 ;;
    --data-root|--data_root) DATA_ROOT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --max-in-flight|--max_in_flight) MAX_IN_FLIGHT="$2"; shift 2 ;;
    --assembly-id|--assembly_id) ASSEMBLY_ID="$2"; shift 2 ;;
    --max-resolution|--max_resolution) MAX_RESOLUTION="$2"; shift 2 ;;
    --min-date|--min_date) MIN_DATE="$2"; shift 2 ;;
    --max-date|--max_date) MAX_DATE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --skip-sync|--skip_sync) SKIP_SYNC=true; shift ;;
    --skip-clusters|--skip_clusters) SKIP_CLUSTERS=true; shift ;;
    --skip-metadata|--skip_metadata) SKIP_METADATA=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

VERSION_DIR="$DATA_ROOT/$VERSION_ID"
RAW_DIR="$VERSION_DIR/raw/assemblies_mmcif"
CLUSTER_FILE="$VERSION_DIR/raw/sequence_clusters/clusters-by-entity-30.txt"
ENTRIES_INDEX="$VERSION_DIR/raw/metadata/entries.idx"
PROCESSED_DIR="$VERSION_DIR/processed/proteinmpnn"

DATA_ROOT="$DATA_ROOT" "$SCRIPT_DIR/init_dataset_version.sh" "$VERSION_ID" >/dev/null

if [ "$SKIP_SYNC" = false ]; then
  DATA_ROOT="$DATA_ROOT" VERSION_ID="$VERSION_ID" "$SCRIPT_DIR/sync_latest_pdb_assemblies.sh"
fi

if [ "$SKIP_CLUSTERS" = false ]; then
  DATA_ROOT="$DATA_ROOT" VERSION_ID="$VERSION_ID" "$SCRIPT_DIR/download_rcsb_sequence_clusters.sh"
fi

if [ "$SKIP_METADATA" = false ]; then
  DATA_ROOT="$DATA_ROOT" VERSION_ID="$VERSION_ID" "$SCRIPT_DIR/download_wwpdb_entries_index.sh"
fi

args=(
  --raw-dir "$RAW_DIR"
  --out-dir "$PROCESSED_DIR"
  --version-id "$VERSION_ID"
  --cluster-file "$CLUSTER_FILE"
  --entries-index "$ENTRIES_INDEX"
  --workers "$WORKERS"
  --max-in-flight "$MAX_IN_FLIGHT"
  --assembly-id "$ASSEMBLY_ID"
  --max-resolution "$MAX_RESOLUTION"
  --max-date "$MAX_DATE"
)
if [ -n "$MIN_DATE" ]; then
  args+=(--min-date "$MIN_DATE")
fi
if [ -n "$LIMIT" ]; then
  args+=(--limit "$LIMIT")
fi

python "$REPO_ROOT/repo/training/build_pdb_mmcif_dataset.py" "${args[@]}" \
  2>&1 | tee "$VERSION_DIR/logs/build_$(date +%Y%m%d%H%M%S).log"

DATA_DIR="$PROCESSED_DIR" "$SCRIPT_DIR/validate_dataset.sh"

echo "processed_dataset: $PROCESSED_DIR"
