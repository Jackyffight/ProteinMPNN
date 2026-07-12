#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

if [ $# -gt 2 ]; then
  echo "Usage: scripts/evaluate_2026_v1_selected_test.sh [run-dir] [gpu-index]" >&2
  exit 2
fi

RUN_DIR="${1:-}"
DEVICE="${2:-0}"
PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"
if ! [[ "$DEVICE" =~ ^[0-9]+$ ]]; then
  echo "Error: gpu-index must be one integer, got: $DEVICE" >&2
  exit 1
fi
if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$({
    find "$PROTEINMPNN_OUTPUT_ROOT" \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name 'proteinmpnn-2026-v1-stage1-v48-a100-*' \
      | sort \
      | tail -n 1
  })"
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "Error: stage-1 run directory not found: ${RUN_DIR:-none}" >&2
  exit 1
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

VALID_SUMMARY="$RUN_DIR/evaluations/fixed-valid-records/summary.json"
if [ ! -s "$VALID_SUMMARY" ]; then
  echo "Error: fixed-valid summary not found: $VALID_SUMMARY" >&2
  echo "Run scripts/evaluate_2026_v1_stage1_checkpoints.sh first." >&2
  exit 1
fi

readarray -t SELECTION < <("$PYTHON_BIN" - "$VALID_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(summary["best_candidate"]["checkpoint"]["path"])
print(summary["best_candidate"]["label"])
PY
)
SELECTED_CHECKPOINT="${SELECTION[0]}"
SELECTED_LABEL="${SELECTION[1]}"
if [ ! -s "$SELECTED_CHECKPOINT" ]; then
  echo "Error: selected checkpoint not found: $SELECTED_CHECKPOINT" >&2
  exit 1
fi

RUN_MANIFEST="$RUN_DIR/run_manifest.json"
DATA_DIR="$($PYTHON_BIN - "$RUN_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest["data"]["path_for_training_data"])
PY
)"
OFFICIAL_CHECKPOINT="$REPO_ROOT/repo/vanilla_model_weights/v_48_020.pt"
"$SCRIPT_DIR/ensure_official_checkpoint.sh" "$OFFICIAL_CHECKPOINT"

OUTPUT_DIR="$RUN_DIR/evaluations/selected-test-records"
mkdir -p "$OUTPUT_DIR"
OFFICIAL_OUTPUT="$OUTPUT_DIR/official-v48-020.json"
SELECTED_OUTPUT="$OUTPUT_DIR/${SELECTED_LABEL}.json"

evaluate_one() {
  local checkpoint="$1"
  local output="$2"
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" \
    "$REPO_ROOT/repo/training/evaluate_checkpoint.py" \
    --checkpoint "$checkpoint" \
    --data-dir "$DATA_DIR" \
    --dataset-format tar \
    --split test \
    --evaluation-unit records \
    --max-examples 0 \
    --require-complete \
    --batch-tokens 10000 \
    --max-protein-length 2000 \
    --seed 42 \
    --device cuda \
    --output "$output" \
    >/dev/null
}

echo "Evaluating official checkpoint once on complete 2026 v1 test records"
evaluate_one "$OFFICIAL_CHECKPOINT" "$OFFICIAL_OUTPUT"
echo "Evaluating selected checkpoint once on complete 2026 v1 test records: $SELECTED_LABEL"
evaluate_one "$SELECTED_CHECKPOINT" "$SELECTED_OUTPUT"

"$PYTHON_BIN" - "$OFFICIAL_OUTPUT" "$SELECTED_OUTPUT" "$OUTPUT_DIR/summary.json" <<'PY'
import json
import sys
from pathlib import Path

official_path, selected_path, summary_path = map(Path, sys.argv[1:])
official = json.loads(official_path.read_text(encoding="utf-8"))
selected = json.loads(selected_path.read_text(encoding="utf-8"))
for result, path in ((official, official_path), (selected, selected_path)):
    data = result["data"]
    if data.get("split") != "test" or data.get("evaluation_unit") != "records":
        raise SystemExit(f"unexpected test population: {path}")
    if data.get("record_count") != 461:
        raise SystemExit(f"expected 461 fixed test records: {path}")
    if data.get("evaluated_structures") != data.get("record_count"):
        raise SystemExit(f"incomplete test evaluation: {path}")
if (
    official["data"]["evaluated_structure_ids_sha256"]
    != selected["data"]["evaluated_structure_ids_sha256"]
):
    raise SystemExit("official and selected checkpoints used different test records")

official_metrics = official["metrics"]
selected_metrics = selected["metrics"]
summary = {
    "schema": "proteinmpnn.selected_test_summary.v1",
    "official": official_metrics,
    "selected_checkpoint": selected["checkpoint"],
    "selected": selected_metrics,
    "delta": {
        "nll": selected_metrics["nll"] - official_metrics["nll"],
        "perplexity": selected_metrics["perplexity"] - official_metrics["perplexity"],
        "accuracy": selected_metrics["accuracy"] - official_metrics["accuracy"],
    },
    "records": selected["data"]["evaluated_structures"],
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
print(f"summary: {summary_path}")
PY
