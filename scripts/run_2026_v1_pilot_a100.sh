#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"
DEVICES="${DEVICES:-0}"
DATA_DIR="${DATA_DIR:-$PROTEINMPNN_V1_DATA_DIR}"
DEFAULT_INIT_CHECKPOINT="$REPO_ROOT/repo/vanilla_model_weights/v_48_020.pt"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$DEFAULT_INIT_CHECKPOINT}"
RUN_NAME="${RUN_NAME:-proteinmpnn-2026-v1-pilot-v48-$(date +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROTEINMPNN_OUTPUT_ROOT/$RUN_NAME}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1000}"
BATCH_TOKENS="${BATCH_TOKENS:-10000}"
MAX_PROTEIN_LENGTH="${MAX_PROTEIN_LENGTH:-2000}"
SEED="${SEED:-42}"
LOADER_WORKERS="${LOADER_WORKERS:-0}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-1}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-1}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RELOAD_EVERY="${RELOAD_EVERY:-2}"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  scripts/run_2026_v1_pilot_a100.sh [--dry-run]

Run the guarded single-A100 ProteinMPNN v1 continuation launcher from the
official v_48_020 weights. Its defaults are a one-epoch pilot. Override settings
with environment variables such as DEVICES, RUN_NAME, NUM_EPOCHS, NUM_EXAMPLES,
BATCH_TOKENS, DATA_DIR, or OUTPUT_DIR.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|--dry_run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$DEVICES" =~ ^[0-9]+$ ]]; then
  echo "Error: DEVICES must contain exactly one numeric GPU index, got: $DEVICES" >&2
  echo "This launcher is single-GPU; do not pass 0,1,2,3." >&2
  exit 1
fi

for required in \
  manifest.json \
  validation.json \
  list.csv \
  index.jsonl \
  records.jsonl \
  valid_clusters.txt \
  test_clusters.txt; do
  if [ ! -s "$DATA_DIR/$required" ]; then
    echo "Error: missing or empty dataset file: $DATA_DIR/$required" >&2
    exit 1
  fi
done
if [ ! -d "$DATA_DIR/shards" ] || ! find "$DATA_DIR/shards" -type f -name '*.tar' -print -quit | grep -q .; then
  echo "Error: no tar shards found under: $DATA_DIR/shards" >&2
  exit 1
fi
if [ "$INIT_CHECKPOINT" = "$DEFAULT_INIT_CHECKPOINT" ]; then
  "$SCRIPT_DIR/ensure_official_checkpoint.sh" "$INIT_CHECKPOINT"
fi
if [ ! -s "$INIT_CHECKPOINT" ]; then
  echo "Error: initialization checkpoint not found: $INIT_CHECKPOINT" >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "Error: output path already exists: $OUTPUT_DIR" >&2
  echo "Set a different RUN_NAME or OUTPUT_DIR to avoid mixing runs." >&2
  exit 1
fi

"$PYTHON_BIN" - "$DATA_DIR/manifest.json" "$DATA_DIR/validation.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if manifest.get("format") != "proteinmpnn.tar_shard.v2":
    raise SystemExit("dataset manifest is not proteinmpnn.tar_shard.v2")
if manifest.get("payload_schema") != "structure_with_target_chain_ids":
    raise SystemExit("dataset payload schema is not the validated v2 schema")
if validation.get("status") != "ok":
    raise SystemExit("dataset validation status is not ok")
if validation.get("records") != manifest.get("record_count"):
    raise SystemExit("dataset validation/manifest record counts differ")
print(
    "dataset_ok "
    f"records={validation['records']} "
    f"shards={validation.get('shards_checked', 'unknown')}"
)
PY

command=(
  "$REPO_ROOT/run_train.sh"
  a100
  --env gpu-server
  --devices "$DEVICES"
  --data-dir "$DATA_DIR"
  --dataset-format tar
  --init-checkpoint "$INIT_CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --num-epochs "$NUM_EPOCHS"
  --num-examples "$NUM_EXAMPLES"
  --batch-tokens "$BATCH_TOKENS"
  --max-protein-length "$MAX_PROTEIN_LENGTH"
  --save-every "$SAVE_EVERY"
  --reload-every "$RELOAD_EVERY"
  --seed "$SEED"
  --loader-workers "$LOADER_WORKERS"
  --prefetch-workers "$PREFETCH_WORKERS"
  --prefetch-batches "$PREFETCH_BATCHES"
)

echo "repo: $REPO_ROOT"
echo "data_dir: $DATA_DIR"
echo "checkpoint: $INIT_CHECKPOINT"
echo "output_dir: $OUTPUT_DIR"
echo "device: $DEVICES"
echo "epochs: $NUM_EPOCHS"
echo "examples: $NUM_EXAMPLES"
echo "batch_tokens: $BATCH_TOKENS"
echo "save_every: $SAVE_EVERY"
echo "reload_every: $RELOAD_EVERY"

if [ "$DRY_RUN" = true ]; then
  printf 'command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="$DEVICES" "$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the selected Python environment")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
print(f"cuda_ok torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
PY

cd "$REPO_ROOT"
exec "${command[@]}"
