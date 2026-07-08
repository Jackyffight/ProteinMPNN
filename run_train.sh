#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$ROOT/scripts/env_nas.sh" ]; then
  source "$ROOT/scripts/env_nas.sh"
fi
MODE="${PROTEINMPNN_MODE:-full}"
ENV_NAME="${PROTEINMPNN_ENV:-devbox}"
PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"
DATA_ROOT="$PROTEINMPNN_DATA_ROOT"
DATA_DIR="${DATA_DIR:-}"
OUTPUT_ROOT="${PROTEINMPNN_OUTPUT_ROOT:-$ROOT/runs}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RUN_NAME="${RUN_NAME:-}"
DEVICES="${DEVICES:-}"
RESUME="${PREVIOUS_CHECKPOINT:-}"

NUM_EPOCHS="${NUM_EPOCHS:-}"
NUM_EXAMPLES="${NUM_EXAMPLES:-}"
BATCH_TOKENS="${BATCH_TOKENS:-}"
MAX_PROTEIN_LENGTH="${MAX_PROTEIN_LENGTH:-}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
ENCODER_LAYERS="${ENCODER_LAYERS:-3}"
DECODER_LAYERS="${DECODER_LAYERS:-3}"
NUM_NEIGHBORS="${NUM_NEIGHBORS:-48}"
DROPOUT="${DROPOUT:-0.1}"
BACKBONE_NOISE="${BACKBONE_NOISE:-0.2}"
RESCUT="${RESCUT:-3.5}"
SEED="${SEED:-42}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RELOAD_EVERY="${RELOAD_EVERY:-2}"
LOADER_WORKERS="${LOADER_WORKERS:-4}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-12}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-3}"
GRADIENT_NORM="${GRADIENT_NORM:--1.0}"
MIXED_PRECISION="${MIXED_PRECISION:-True}"
TF32="${TF32:-True}"
SAVE_BEST="${SAVE_BEST:-True}"
DEBUG="${DEBUG:-}"
INSTALL_DEPS=false

usage() {
  cat <<'EOF'
Usage:
  ./ProteinMPNN/run_train.sh [smoke|full|v100|a100] [launcher args]

Launcher args:
  --env <devbox|online|local>        Environment label for run naming. Default: devbox.
  --mode <smoke|full|v100|a100>      Training preset. Positional mode is also accepted.
  --data-dir <path>                  Dataset root containing list.csv and pdb/.
  --output-root <dir>                Workspace root. Default: ProteinMPNN/runs.
  --output-dir <dir>                 Exact output directory. Overrides --output-root/--run-name.
  --run-name <name>                  Run name under --output-root.
  --devices <list>                   CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1.
  --python <path>                    Python binary. Default: python.
  --resume <checkpoint>              Resume from model_weights/epoch_last.pt or another checkpoint.
  --install-deps                     pip install ProteinMPNN/requirements.txt before launch.

Training args:
  --num-epochs <n>                   Epoch count. full/a100 default: 150; smoke default: 1.
  --num-examples <n>                 Structures sampled per epoch. full default: 1000000.
  --batch-tokens <n>                 Token budget per batch. V100 default: 6000; A100/full: 10000.
  --max-protein-length <n>           Length filter. full default: 10000; smoke default: 1000.
  --hidden-dim <n>                   ProteinMPNN hidden dim. Default: 128.
  --encoder-layers <n>               Encoder layers. Default: 3.
  --decoder-layers <n>               Decoder layers. Default: 3.
  --num-neighbors <n>                Sparse graph neighbors. Default: 48.
  --dropout <float>                  Dropout. Default: 0.1.
  --backbone-noise <float>           Backbone noise. Default: 0.2.
  --rescut <float>                   PDB resolution cutoff. Default: 3.5.
  --seed <n>                         RNG seed. Default: 42.
  --loader-workers <n>               DataLoader workers. Default: 4.
  --prefetch-workers <n>             ProcessPool workers for structure loading. Default: 12.
  --prefetch-batches <n>             Prefetched train/valid batches. Default: 3.
  --gradient-norm <float>            Clip norm; negative disables. Default: -1.0.
  --mixed-precision / --no-mixed-precision
  --tf32 / --no-tf32
  --debug / --no-debug
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    smoke|full|v100|a100) MODE="$1"; shift ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --data-dir|--data_dir) DATA_DIR="$2"; shift 2 ;;
    --output-root|--output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --output-dir|--output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --run-name|--run_name) RUN_NAME="$2"; shift 2 ;;
    --devices|--cuda-visible-devices|--cuda_visible_devices) DEVICES="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --resume|--previous-checkpoint|--previous_checkpoint) RESUME="$2"; shift 2 ;;
    --install-deps|--install_deps) INSTALL_DEPS=true; shift ;;
    --num-epochs|--num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
    --num-examples|--num_examples) NUM_EXAMPLES="$2"; shift 2 ;;
    --batch-tokens|--batch_tokens|--batch-size|--batch_size) BATCH_TOKENS="$2"; shift 2 ;;
    --max-protein-length|--max_protein_length) MAX_PROTEIN_LENGTH="$2"; shift 2 ;;
    --hidden-dim|--hidden_dim) HIDDEN_DIM="$2"; shift 2 ;;
    --encoder-layers|--encoder_layers|--num-encoder-layers|--num_encoder_layers) ENCODER_LAYERS="$2"; shift 2 ;;
    --decoder-layers|--decoder_layers|--num-decoder-layers|--num_decoder_layers) DECODER_LAYERS="$2"; shift 2 ;;
    --num-neighbors|--num_neighbors) NUM_NEIGHBORS="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --backbone-noise|--backbone_noise) BACKBONE_NOISE="$2"; shift 2 ;;
    --rescut) RESCUT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --save-every|--save_every) SAVE_EVERY="$2"; shift 2 ;;
    --reload-every|--reload_every) RELOAD_EVERY="$2"; shift 2 ;;
    --loader-workers|--loader_workers|--num-loader-workers|--num_loader_workers) LOADER_WORKERS="$2"; shift 2 ;;
    --prefetch-workers|--prefetch_workers) PREFETCH_WORKERS="$2"; shift 2 ;;
    --prefetch-batches|--prefetch_batches) PREFETCH_BATCHES="$2"; shift 2 ;;
    --gradient-norm|--gradient_norm) GRADIENT_NORM="$2"; shift 2 ;;
    --mixed-precision|--mixed_precision) MIXED_PRECISION=True; shift ;;
    --no-mixed-precision|--no_mixed_precision) MIXED_PRECISION=False; shift ;;
    --tf32) TF32=True; shift ;;
    --no-tf32|--no_tf32) TF32=False; shift ;;
    --debug) DEBUG=True; shift ;;
    --no-debug|--no_debug) DEBUG=False; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$MODE" != "smoke" ] && [ "$MODE" != "full" ] && [ "$MODE" != "v100" ] && [ "$MODE" != "a100" ]; then
  echo "Error: --mode must be smoke, full, v100, or a100." >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d%H%M%S)"
case "$MODE" in
  smoke)
    DATA_DIR="${DATA_DIR:-$DATA_ROOT/pdb_2021aug02_sample}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    NUM_EXAMPLES="${NUM_EXAMPLES:-50}"
    BATCH_TOKENS="${BATCH_TOKENS:-1000}"
    MAX_PROTEIN_LENGTH="${MAX_PROTEIN_LENGTH:-1000}"
    DEBUG="${DEBUG:-True}"
    ;;
  v100)
    DATA_DIR="${DATA_DIR:-$DATA_ROOT/pdb_2021aug02}"
    NUM_EPOCHS="${NUM_EPOCHS:-150}"
    NUM_EXAMPLES="${NUM_EXAMPLES:-1000000}"
    BATCH_TOKENS="${BATCH_TOKENS:-6000}"
    MAX_PROTEIN_LENGTH="${MAX_PROTEIN_LENGTH:-10000}"
    DEBUG="${DEBUG:-False}"
    ;;
  a100|full)
    DATA_DIR="${DATA_DIR:-$DATA_ROOT/pdb_2021aug02}"
    NUM_EPOCHS="${NUM_EPOCHS:-150}"
    NUM_EXAMPLES="${NUM_EXAMPLES:-1000000}"
    BATCH_TOKENS="${BATCH_TOKENS:-10000}"
    MAX_PROTEIN_LENGTH="${MAX_PROTEIN_LENGTH:-10000}"
    DEBUG="${DEBUG:-False}"
    ;;
esac

if [ -z "$RUN_NAME" ]; then
  RUN_NAME="proteinmpnn-${MODE}-${ENV_NAME}-${TIMESTAMP}"
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
fi

if [ ! -d "$DATA_DIR" ]; then
  echo "Error: data directory not found: $DATA_DIR" >&2
  exit 1
fi
for required in list.csv valid_clusters.txt test_clusters.txt; do
  if [ ! -f "$DATA_DIR/$required" ]; then
    echo "Error: missing dataset file: $DATA_DIR/$required" >&2
    exit 1
  fi
done
if [ -n "$RESUME" ] && [ ! -f "$RESUME" ]; then
  echo "Error: resume checkpoint not found: $RESUME" >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
  else
    echo "Error: python is not executable or on PATH: $PYTHON_BIN" >&2
    exit 1
  fi
fi

if [ "$INSTALL_DEPS" = true ]; then
  "$PYTHON_BIN" -m pip install -r "$ROOT/requirements.txt"
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [name for name in ("numpy", "dateutil", "torch") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing) + ". Install dependencies or rerun with --install-deps.")
PY

if [ -n "$DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES="$DEVICES"
fi

mkdir -p "$OUTPUT_DIR"

echo "=== ProteinMPNN training ==="
echo "env: $ENV_NAME"
echo "mode: $MODE"
echo "data_dir: $DATA_DIR"
echo "output_dir: $OUTPUT_DIR"
echo "devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "resume: ${RESUME:-none}"
echo "epochs: $NUM_EPOCHS"
echo "num_examples_per_epoch: $NUM_EXAMPLES"
echo "batch_tokens: $BATCH_TOKENS"
echo "max_protein_length: $MAX_PROTEIN_LENGTH"
echo "hidden_dim: $HIDDEN_DIM"
echo "encoder_layers: $ENCODER_LAYERS"
echo "decoder_layers: $DECODER_LAYERS"
echo "num_neighbors: $NUM_NEIGHBORS"
echo "backbone_noise: $BACKBONE_NOISE"
echo "seed: $SEED"
echo "loader_workers: $LOADER_WORKERS"
echo "prefetch_workers: $PREFETCH_WORKERS"
echo "mixed_precision: $MIXED_PRECISION"

cd "$ROOT/repo/training"
exec "$PYTHON_BIN" training.py \
  --path_for_training_data "$DATA_DIR" \
  --path_for_outputs "$OUTPUT_DIR" \
  --previous_checkpoint "$RESUME" \
  --num_epochs "$NUM_EPOCHS" \
  --save_model_every_n_epochs "$SAVE_EVERY" \
  --reload_data_every_n_epochs "$RELOAD_EVERY" \
  --num_examples_per_epoch "$NUM_EXAMPLES" \
  --batch_size "$BATCH_TOKENS" \
  --max_protein_length "$MAX_PROTEIN_LENGTH" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_encoder_layers "$ENCODER_LAYERS" \
  --num_decoder_layers "$DECODER_LAYERS" \
  --num_neighbors "$NUM_NEIGHBORS" \
  --dropout "$DROPOUT" \
  --backbone_noise "$BACKBONE_NOISE" \
  --rescut "$RESCUT" \
  --debug "$DEBUG" \
  --gradient_norm "$GRADIENT_NORM" \
  --mixed_precision "$MIXED_PRECISION" \
  --seed "$SEED" \
  --num_loader_workers "$LOADER_WORKERS" \
  --prefetch_workers "$PREFETCH_WORKERS" \
  --prefetch_batches "$PREFETCH_BATCHES" \
  --tf32 "$TF32" \
  --save_best "$SAVE_BEST"
