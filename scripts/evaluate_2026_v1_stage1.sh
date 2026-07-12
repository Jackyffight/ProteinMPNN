#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

if [ $# -gt 1 ]; then
  echo "Usage: scripts/evaluate_2026_v1_stage1.sh [run-dir]" >&2
  exit 2
fi

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$(
    find "$PROTEINMPNN_OUTPUT_ROOT" \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name 'proteinmpnn-2026-v1-stage1-v48-a100-*' \
      | sort \
      | tail -n 1
  )"
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "Error: stage-1 run directory not found: ${RUN_DIR:-none}" >&2
  exit 1
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

DATA_DIR="${DATA_DIR:-$PROTEINMPNN_V1_DATA_DIR}"
OFFICIAL_CHECKPOINT="$REPO_ROOT/repo/vanilla_model_weights/v_48_020.pt"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-$RUN_DIR/model_weights/best.pt}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$RUN_DIR/evaluations}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-42}"
DEVICES="${DEVICES:-0}"

if ! [[ "$DEVICES" =~ ^[0-9]+$ ]]; then
  echo "Error: DEVICES must contain one numeric GPU index, got: $DEVICES" >&2
  exit 1
fi
if [ "$SPLIT" != "valid" ] && [ "$SPLIT" != "test" ]; then
  echo "Error: SPLIT must be valid or test, got: $SPLIT" >&2
  exit 1
fi
if [ ! -s "$CANDIDATE_CHECKPOINT" ]; then
  echo "Error: candidate checkpoint not found: $CANDIDATE_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$EVAL_OUTPUT_DIR"
OFFICIAL_OUTPUT="$EVAL_OUTPUT_DIR/official-v48-020-${SPLIT}.json"
CANDIDATE_OUTPUT="$EVAL_OUTPUT_DIR/stage1-best-${SPLIT}.json"

"$SCRIPT_DIR/ensure_official_checkpoint.sh" "$OFFICIAL_CHECKPOINT"

echo "Evaluating official checkpoint"
CUDA_VISIBLE_DEVICES="$DEVICES" \
DATA_DIR="$DATA_DIR" \
CHECKPOINT="$OFFICIAL_CHECKPOINT" \
MAX_EXAMPLES="$MAX_EXAMPLES" \
SPLIT="$SPLIT" \
OUTPUT="$OFFICIAL_OUTPUT" \
  "$SCRIPT_DIR/evaluate_official_checkpoint.sh" \
    --dataset-format tar \
    --seed "$SEED" \
    >/dev/null

echo "Evaluating stage-1 best checkpoint"
CUDA_VISIBLE_DEVICES="$DEVICES" \
DATA_DIR="$DATA_DIR" \
CHECKPOINT="$CANDIDATE_CHECKPOINT" \
MAX_EXAMPLES="$MAX_EXAMPLES" \
SPLIT="$SPLIT" \
OUTPUT="$CANDIDATE_OUTPUT" \
  "$SCRIPT_DIR/evaluate_official_checkpoint.sh" \
    --dataset-format tar \
    --seed "$SEED" \
    >/dev/null

python - "$OFFICIAL_OUTPUT" "$CANDIDATE_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

official_path, candidate_path = map(Path, sys.argv[1:])
official = json.loads(official_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

official_ids = official["data"]["evaluated_structure_ids_sha256"]
candidate_ids = candidate["data"]["evaluated_structure_ids_sha256"]
if official_ids != candidate_ids:
    raise SystemExit("evaluation mismatch: checkpoints were not scored on identical structures")

official_metrics = official["metrics"]
candidate_metrics = candidate["metrics"]
perplexity_delta = candidate_metrics["perplexity"] - official_metrics["perplexity"]
accuracy_delta = candidate_metrics["accuracy"] - official_metrics["accuracy"]
candidate_metadata = candidate["checkpoint"]["metadata"]
status = "improved" if perplexity_delta < 0 else "not_improved"

print(f"split: {candidate['data']['split']}")
print(f"structures: {candidate['data']['evaluated_structures']}")
print(
    "official: "
    f"perplexity={official_metrics['perplexity']:.6f} "
    f"accuracy={official_metrics['accuracy']:.6f}"
)
print(
    "stage1_best: "
    f"epoch={candidate_metadata.get('epoch')} "
    f"step={candidate_metadata.get('step')} "
    f"perplexity={candidate_metrics['perplexity']:.6f} "
    f"accuracy={candidate_metrics['accuracy']:.6f}"
)
print(f"delta: perplexity={perplexity_delta:+.6f} accuracy={accuracy_delta:+.6f}")
print(f"status: {status}")
print(f"official_result: {official_path}")
print(f"candidate_result: {candidate_path}")
PY
