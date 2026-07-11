#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

DATA_ROOT="${DATA_ROOT:-$PROTEINMPNN_CUSTOM_DATA_ROOT}"
VERSION_ID="${VERSION_ID:-proteinmpnn_pdb_20260708}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
WORKERS="${WORKERS:-2}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-2}"
ASSEMBLY_ID="${ASSEMBLY_ID:-all}"
ASSEMBLY_POLICY="${ASSEMBLY_POLICY:-first}"
MAX_RESOLUTION="${MAX_RESOLUTION:-3.5}"
MIN_RESOLVED_RESIDUES="${MIN_RESOLVED_RESIDUES:-30}"
MIN_BACKBONE_COVERAGE="${MIN_BACKBONE_COVERAGE:-0.5}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-2000}"
MIN_DATE="${MIN_DATE:-2021-08-03}"
MAX_DATE="${MAX_DATE:-2026-07-08}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-1g}"
MAX_RAW_FILE_SIZE="${MAX_RAW_FILE_SIZE:-50m}"
LIMIT="${LIMIT:-}"
FORCE=false
VALIDATE=true

usage() {
  cat <<'EOF'
Usage:
  scripts/build_pdb_2026_tar_shards.sh [options]

Build a ProteinMPNN tar-shard dataset directly from existing wwPDB biological
assembly mmCIF files.

Options:
  --version-id <id>       Dataset version id. Default: proteinmpnn_pdb_20260708.
  --data-root <dir>       Custom dataset root. Default: NAS MPNN custom root.
  --output-dir <dir>      Tar-shard output dir. Default: VERSION/processed/proteinmpnn_tar_shards_v1.
  --workers <n>           mmCIF parser workers. Default: 2.
  --max-in-flight <n>     ProcessPool futures in flight. Default: 2.
  --assembly-id <id|all>  Assembly file selector. Default: all.
  --assembly-policy <first|all>
                          Canonical assembly policy. Default: first.
  --max-resolution <a>    Resolution cutoff. Default: 3.5.
  --min-resolved-residues <n>
                          Minimum complete-backbone residues per context chain. Default: 30.
  --min-backbone-coverage <f>
                          Minimum complete-backbone fraction per context chain. Default: 0.5.
  --max-context-length <n>
                          Defer assemblies above this total length. Default: 2000.
  --min-date <YYYY-MM-DD> Deposition lower bound. Default: 2021-08-03.
  --max-date <YYYY-MM-DD> Deposition upper bound. Default: 2026-07-08.
  --max-shard-size <size> Shard target size. Default: 1g.
  --max-raw-file-size <size>
                          Skip compressed raw mmCIF files larger than this.
                          Default: 50m. Use 0 to disable.
  --limit <n>             Debug: process only first n assembly files.
  --force                 Replace output dir if it exists.
  --no-validate           Skip the post-build v2 payload validator.
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
    --assembly-policy|--assembly_policy) ASSEMBLY_POLICY="$2"; shift 2 ;;
    --max-resolution|--max_resolution) MAX_RESOLUTION="$2"; shift 2 ;;
    --min-resolved-residues|--min_resolved_residues) MIN_RESOLVED_RESIDUES="$2"; shift 2 ;;
    --min-backbone-coverage|--min_backbone_coverage) MIN_BACKBONE_COVERAGE="$2"; shift 2 ;;
    --max-context-length|--max_context_length) MAX_CONTEXT_LENGTH="$2"; shift 2 ;;
    --min-date|--min_date) MIN_DATE="$2"; shift 2 ;;
    --max-date|--max_date) MAX_DATE="$2"; shift 2 ;;
    --max-shard-size|--max_shard_size) MAX_SHARD_SIZE="$2"; shift 2 ;;
    --max-raw-file-size|--max_raw_file_size) MAX_RAW_FILE_SIZE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    --no-validate|--no_validate) VALIDATE=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$DATA_ROOT/$VERSION_ID/processed/proteinmpnn_tar_shards_v1"
fi

VERSION_DIR="$DATA_ROOT/$VERSION_ID"
RAW_DIR="$VERSION_DIR/raw/assemblies_mmcif"
CLUSTER_FILE="${CLUSTER_FILE:-$VERSION_DIR/raw/sequence_clusters/clusters-by-entity-30.txt}"
ENTRIES_INDEX="$VERSION_DIR/raw/metadata/entries.idx"
LOG_DIR="$VERSION_DIR/logs"

if [ ! -d "$RAW_DIR" ]; then
  echo "Missing raw mmCIF dir: $RAW_DIR" >&2
  exit 1
fi
if [ ! -s "$ENTRIES_INDEX" ]; then
  echo "Missing entries index: $ENTRIES_INDEX" >&2
  exit 1
fi
if [ ! -s "$CLUSTER_FILE" ]; then
  echo "Missing sequence cluster file: $CLUSTER_FILE" >&2
  echo "Refusing to build production splits without homology clusters." >&2
  exit 1
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
  --assembly-policy "$ASSEMBLY_POLICY"
  --max-resolution "$MAX_RESOLUTION"
  --min-resolved-residues "$MIN_RESOLVED_RESIDUES"
  --min-backbone-coverage "$MIN_BACKBONE_COVERAGE"
  --max-context-length "$MAX_CONTEXT_LENGTH"
  --max-date "$MAX_DATE"
  --max-shard-size "$MAX_SHARD_SIZE"
  --max-raw-file-size "$MAX_RAW_FILE_SIZE"
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
echo "max_in_flight: $MAX_IN_FLIGHT"
echo "assembly_policy: $ASSEMBLY_POLICY"
echo "date_range: $MIN_DATE..$MAX_DATE"
echo "min_resolved_residues: $MIN_RESOLVED_RESIDUES"
echo "min_backbone_coverage: $MIN_BACKBONE_COVERAGE"
echo "max_context_length: $MAX_CONTEXT_LENGTH"
echo "max_raw_file_size: $MAX_RAW_FILE_SIZE"

EXTRA_PYTHONPATH="$REPO_ROOT/repo/training"
LOCAL_DEPS="$(cd "$REPO_ROOT/.." && pwd)/.pdbbuild_deps"
if [ -d "$LOCAL_DEPS" ]; then
  EXTRA_PYTHONPATH="$LOCAL_DEPS:$EXTRA_PYTHONPATH"
fi

PYTHONPATH="${PYTHONPATH:-}:$EXTRA_PYTHONPATH" \
python "$REPO_ROOT/repo/training/build_pdb_mmcif_tar_shard_dataset.py" "${args[@]}" \
  2>&1 | tee "$LOG_DIR/build_tar_shards_$(date +%Y%m%d%H%M%S).log"

if [ "$VALIDATE" = true ]; then
  PYTHONPATH="${PYTHONPATH:-}:$EXTRA_PYTHONPATH" \
  python "$REPO_ROOT/repo/training/validate_tar_shard_dataset.py" \
    --dataset-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/validation.json"
fi

echo "tar_shard_dataset: $OUTPUT_DIR"
