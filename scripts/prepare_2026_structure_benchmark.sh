#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"
DATA_DIR="${DATA_DIR:-$PROTEINMPNN_V1_DATA_DIR}"
COUNT="${COUNT:-40}"
SEED="${SEED:-42}"
MIN_LENGTH="${MIN_LENGTH:-50}"
MAX_LENGTH="${MAX_LENGTH:-800}"
LENGTH_BINS="${LENGTH_BINS:-4}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROTEINMPNN_OUTPUT_ROOT/benchmarks/structure-input-valid-n${COUNT}-s${SEED}-${RUN_TAG}}"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_2026_structure_benchmark.sh [--dry-run]

Create a deterministic, metadata-only structure-throughput benchmark from the
validated 2026 ProteinMPNN valid split. This command does not read tar shards,
use a GPU, mutate proteins, or consume the held-out test split.

Optional environment overrides:
  DATA_DIR COUNT SEED MIN_LENGTH MAX_LENGTH LENGTH_BINS OUTPUT_DIR PYTHON_BIN
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|--dry_run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command=(
  "$PYTHON_BIN"
  -m protein_mrna_pipeline
  make-benchmark
  --dataset-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --count "$COUNT"
  --seed "$SEED"
  --min-length "$MIN_LENGTH"
  --max-length "$MAX_LENGTH"
  --length-bins "$LENGTH_BINS"
)

echo "dataset_dir: $DATA_DIR"
echo "output_dir: $OUTPUT_DIR"
echo "split: valid"
echo "count: $COUNT"
echo "seed: $SEED"
echo "length_range: $MIN_LENGTH..$MAX_LENGTH"
echo "length_bins: $LENGTH_BINS"

if [ "$DRY_RUN" = true ]; then
  printf 'command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

export PYTHONPATH="$REPO_ROOT/protein_mrna_pipeline/src${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PYTHON_BIN" -c \
  'from importlib.metadata import version; assert tuple(map(int, version("jsonschema").split(".")[:2])) >= (4, 18)' \
  >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN must provide jsonschema>=4.18." >&2
  echo "Install the local project dependencies before generating the benchmark." >&2
  exit 1
fi
"${command[@]}"
"$PYTHON_BIN" -m protein_mrna_pipeline verify-benchmark \
  --suite "$OUTPUT_DIR/benchmark-suite.json"

echo "benchmark_ready: $OUTPUT_DIR"
echo "gpu_started: false"
echo "next: pin a structure-model runtime, then enqueue these FASTA records"
