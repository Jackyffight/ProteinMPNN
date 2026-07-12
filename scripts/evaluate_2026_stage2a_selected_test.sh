#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

if [ $# -gt 2 ]; then
  echo "Usage: scripts/evaluate_2026_stage2a_selected_test.sh [run-dir] [gpu-index]" >&2
  exit 2
fi

RUN_DIR="${1:-}"
DEVICE="${2:-0}"
PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"
V1_DATA_DIR="${V1_DATA_DIR:-$PROTEINMPNN_V1_DATA_DIR}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$PROTEINMPNN_OUTPUT_ROOT/promoted/proteinmpnn-2026-v1-stage1/model.pt}"
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
      -name 'proteinmpnn-2026-stage2a-v48-a100-*' \
      | sort \
      | tail -n 1
  })"
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "Error: formal stage2a run directory not found: ${RUN_DIR:-none}" >&2
  exit 1
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
VALID_SUMMARY="$RUN_DIR/evaluations/dual-valid/summary.json"
if [ ! -s "$VALID_SUMMARY" ]; then
  echo "Error: dual-valid summary not found; run evaluate_2026_stage2a_checkpoints.sh first" >&2
  exit 1
fi

readarray -t SELECTION < <("$PYTHON_BIN" - "$VALID_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("schema") != "proteinmpnn.stage2a_dual_valid_summary.v1":
    raise SystemExit("unexpected dual-valid summary schema")
if summary.get("status") != "passed" or not summary.get("selected"):
    raise SystemExit("no stage2a checkpoint passed the dual-valid gate")
print(summary["selected"]["checkpoint"]["path"])
print(summary["selected"]["label"])
print(summary["max_v1_nll_regression"])
PY
)
SELECTED_CHECKPOINT="${SELECTION[0]}"
SELECTED_LABEL="${SELECTION[1]}"
MAX_V1_NLL_REGRESSION="${SELECTION[2]}"
if [ ! -s "$SELECTED_CHECKPOINT" ] || [ ! -s "$BASELINE_CHECKPOINT" ]; then
  echo "Error: selected stage2a checkpoint or stage-1 baseline is missing" >&2
  exit 1
fi

STAGE2_DATA_DIR="$($PYTHON_BIN - "$RUN_DIR/run_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest["data"]["path_for_training_data"])
PY
)"
OUTPUT_DIR="$RUN_DIR/evaluations/selected-test-records"
SUMMARY_PATH="$OUTPUT_DIR/summary.json"
if [ -e "$SUMMARY_PATH" ]; then
  echo "Error: stage2a selected-test summary already exists: $SUMMARY_PATH" >&2
  echo "The held-out test gate is intentionally one-shot." >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

evaluate_one() {
  local dataset_label="$1"
  local data_dir="$2"
  local checkpoint_label="$3"
  local checkpoint="$4"
  local output="$OUTPUT_DIR/${dataset_label}--${checkpoint_label}.json"
  echo "===== $checkpoint_label on complete $dataset_label test records ====="
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" \
    "$REPO_ROOT/repo/training/evaluate_checkpoint.py" \
    --checkpoint "$checkpoint" \
    --data-dir "$data_dir" \
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

evaluate_one stage2a "$STAGE2_DATA_DIR" baseline-stage1 "$BASELINE_CHECKPOINT"
evaluate_one stage2a "$STAGE2_DATA_DIR" "$SELECTED_LABEL" "$SELECTED_CHECKPOINT"
evaluate_one v1 "$V1_DATA_DIR" baseline-stage1 "$BASELINE_CHECKPOINT"
evaluate_one v1 "$V1_DATA_DIR" "$SELECTED_LABEL" "$SELECTED_CHECKPOINT"

"$PYTHON_BIN" - \
  "$OUTPUT_DIR" \
  "$SUMMARY_PATH" \
  "$SELECTED_LABEL" \
  "$MAX_V1_NLL_REGRESSION" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
selected_label = sys.argv[3]
max_v1_regression = float(sys.argv[4])

results = {}
for dataset_label in ("stage2a", "v1"):
    results[dataset_label] = {}
    for checkpoint_label in ("baseline-stage1", selected_label):
        path = result_dir / f"{dataset_label}--{checkpoint_label}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        data = result["data"]
        if data.get("split") != "test" or data.get("evaluation_unit") != "records":
            raise SystemExit(f"unexpected test population: {path}")
        if data.get("record_count", 0) <= 0:
            raise SystemExit(f"empty test population: {path}")
        if dataset_label == "v1" and data.get("record_count") != 461:
            raise SystemExit(f"expected 461 fixed v1 test records: {path}")
        if data.get("evaluated_structures") != data.get("record_count"):
            raise SystemExit(f"incomplete test evaluation: {path}")
        results[dataset_label][checkpoint_label] = result

    baseline = results[dataset_label]["baseline-stage1"]
    selected = results[dataset_label][selected_label]
    if (
        baseline["data"]["evaluated_structure_ids_sha256"]
        != selected["data"]["evaluated_structure_ids_sha256"]
    ):
        raise SystemExit(f"baseline/selected population mismatch: {dataset_label}")

stage2a_baseline = results["stage2a"]["baseline-stage1"]
stage2a_selected = results["stage2a"][selected_label]
v1_baseline = results["v1"]["baseline-stage1"]
v1_selected = results["v1"][selected_label]
delta_stage2a = stage2a_selected["metrics"]["nll"] - stage2a_baseline["metrics"]["nll"]
delta_v1 = v1_selected["metrics"]["nll"] - v1_baseline["metrics"]["nll"]
passed = delta_stage2a < 0.0 and delta_v1 <= max_v1_regression
summary = {
    "schema": "proteinmpnn.stage2a_dual_test_summary.v1",
    "selected_label": selected_label,
    "selected_checkpoint": stage2a_selected["checkpoint"],
    "max_v1_nll_regression": max_v1_regression,
    "stage2a": {
        "records": stage2a_selected["data"]["evaluated_structures"],
        "baseline": stage2a_baseline["metrics"],
        "selected": stage2a_selected["metrics"],
        "delta_nll": delta_stage2a,
    },
    "v1": {
        "records": v1_selected["data"]["evaluated_structures"],
        "baseline": v1_baseline["metrics"],
        "selected": v1_selected["metrics"],
        "delta_nll": delta_v1,
    },
    "status": "passed" if passed else "failed",
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
print(f"summary: {summary_path}")
if not passed:
    raise SystemExit("selected stage2a checkpoint failed the held-out dual test gate")
PY
