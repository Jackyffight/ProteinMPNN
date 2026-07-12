#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"
DEVICES="${DEVICES:-0}"
DATA_DIR="${DATA_DIR:-$PROTEINMPNN_STAGE2A_DATA_DIR}"
PROMOTION_DIR="${PROMOTION_DIR:-$PROTEINMPNN_OUTPUT_ROOT/promoted/proteinmpnn-2026-v1-stage1}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$PROMOTION_DIR/model.pt}"
PROMOTION_MANIFEST="${PROMOTION_MANIFEST:-$PROMOTION_DIR/promotion.json}"
RUN_NAME="${RUN_NAME:-proteinmpnn-2026-stage2a-pilot-v48-a100-$(date +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROTEINMPNN_OUTPUT_ROOT/$RUN_NAME}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1000}"
BATCH_TOKENS="${BATCH_TOKENS:-10000}"
MAX_PROTEIN_LENGTH="${MAX_PROTEIN_LENGTH:-2000}"
LR_FACTOR="${LR_FACTOR:-0.25}"
WARMUP_STEPS="${WARMUP_STEPS:-4000}"
GRADIENT_NORM="${GRADIENT_NORM:-1.0}"
SEED="${SEED:-42}"
SAVE_EVERY="${SAVE_EVERY:-1}"
RELOAD_EVERY="${RELOAD_EVERY:-1}"
LOADER_WORKERS="${LOADER_WORKERS:-0}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-1}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-1}"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  scripts/run_2026_stage2a_pilot_a100.sh [--dry-run]

Run a guarded single-A100 stage2a pilot initialized from the promoted stage-1
weights. Defaults: 1,000 examples, 1 epoch, Noam factor 0.25. Override settings
with environment variables such as DEVICES, RUN_NAME, NUM_EXAMPLES, or DATA_DIR.
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
  build_manifest.json \
  validation.json \
  list.csv \
  index.jsonl \
  records.jsonl \
  valid_clusters.txt \
  test_clusters.txt; do
  if [ ! -s "$DATA_DIR/$required" ]; then
    echo "Error: missing or empty stage2a dataset file: $DATA_DIR/$required" >&2
    exit 1
  fi
done
if [ ! -d "$DATA_DIR/shards" ] || ! find "$DATA_DIR/shards" -type f -name '*.tar' -print -quit | grep -q .; then
  echo "Error: no stage2a tar shards found under: $DATA_DIR/shards" >&2
  exit 1
fi
if [ ! -s "$INIT_CHECKPOINT" ] || [ ! -s "$PROMOTION_MANIFEST" ]; then
  echo "Error: promoted stage-1 model or manifest is missing under: $PROMOTION_DIR" >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "Error: output path already exists: $OUTPUT_DIR" >&2
  echo "Set a different RUN_NAME or OUTPUT_DIR to avoid mixing runs." >&2
  exit 1
fi

"$PYTHON_BIN" - \
  "$DATA_DIR/manifest.json" \
  "$DATA_DIR/validation.json" \
  "$PROMOTION_MANIFEST" \
  "$INIT_CHECKPOINT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, validation_path, promotion_path, checkpoint_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
validation = json.loads(validation_path.read_text(encoding="utf-8"))
promotion = json.loads(promotion_path.read_text(encoding="utf-8"))

if manifest.get("format") != "proteinmpnn.tar_shard.v2":
    raise SystemExit("stage2a dataset is not proteinmpnn.tar_shard.v2")
if manifest.get("payload_schema") != "structure_with_target_chain_ids_spatial_crop":
    raise SystemExit("stage2a dataset has an unexpected payload schema")
if manifest.get("crop_policy") != "full_target_nearest_chain_windows_v1":
    raise SystemExit("stage2a dataset has an unexpected crop policy")
if validation.get("status") != "ok":
    raise SystemExit("stage2a validation status is not ok")
records = int(manifest.get("record_count", 0))
if records <= 0 or validation.get("records") != records:
    raise SystemExit("stage2a validation/manifest record counts differ")
if validation.get("payloads_checked") != records:
    raise SystemExit("stage2a validation does not cover every payload")
if validation.get("exact_sequence_split_leaks") != 0 or validation.get("pdb_split_leaks") != 0:
    raise SystemExit("stage2a validation reports internal split leakage")
if validation.get("reference_pdb_overlaps") != 0:
    raise SystemExit("stage2a validation reports overlap with the v1 reference")

if promotion.get("schema") != "proteinmpnn.promoted_checkpoint.v1":
    raise SystemExit("stage-1 promotion manifest has an unexpected schema")
if promotion.get("model_id") != "proteinmpnn-2026-v1-stage1":
    raise SystemExit("stage-1 promotion manifest has an unexpected model id")
intended_use = promotion.get("intended_use", {})
if intended_use.get("checkpoint_mode") != "weight_initialization":
    raise SystemExit("promoted model is not approved for weight initialization")
if intended_use.get("restore_optimizer") is not False:
    raise SystemExit("stage2a must not restore the stage-1 optimizer")

digest = hashlib.sha256()
with checkpoint_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
expected = promotion.get("checkpoint", {}).get("sha256")
if not expected or digest.hexdigest() != expected:
    raise SystemExit("promoted stage-1 checkpoint checksum mismatch")
print(
    f"stage2a_inputs_ok records={records} "
    f"shards={validation.get('shards_checked', 'unknown')} "
    f"checkpoint_sha256={expected}"
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
  --lr-factor "$LR_FACTOR"
  --warmup-steps "$WARMUP_STEPS"
  --gradient-norm "$GRADIENT_NORM"
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
echo "lr_factor: $LR_FACTOR"
echo "warmup_steps: $WARMUP_STEPS"
echo "gradient_norm: $GRADIENT_NORM"

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
