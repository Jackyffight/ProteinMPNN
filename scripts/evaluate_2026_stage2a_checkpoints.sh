#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

if [ $# -gt 2 ]; then
  echo "Usage: scripts/evaluate_2026_stage2a_checkpoints.sh [run-dir] [gpu-index]" >&2
  exit 2
fi

RUN_DIR="${1:-}"
DEVICE="${2:-0}"
PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"
V1_DATA_DIR="${V1_DATA_DIR:-$PROTEINMPNN_V1_DATA_DIR}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$PROTEINMPNN_OUTPUT_ROOT/promoted/proteinmpnn-2026-v1-stage1/model.pt}"
MAX_V1_NLL_REGRESSION="${MAX_V1_NLL_REGRESSION:-0.001}"

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
      -name 'proteinmpnn-2026-stage2a-*-a100-*' \
      -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  })"
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "Error: stage2a run directory not found: ${RUN_DIR:-none}" >&2
  exit 1
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
RUN_MANIFEST="$RUN_DIR/run_manifest.json"
if [ ! -s "$RUN_MANIFEST" ] || [ ! -s "$BASELINE_CHECKPOINT" ]; then
  echo "Error: run manifest or promoted stage-1 baseline is missing" >&2
  exit 1
fi

STAGE2_DATA_DIR="$($PYTHON_BIN - "$RUN_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("initialization", {}).get("mode") != "checkpoint":
    raise SystemExit("stage2a run was not initialized from promoted weights")
optimizer = manifest.get("optimizer", {})
if float(optimizer.get("factor", 0.0)) >= 2.0:
    raise SystemExit("stage2a run did not use a conservative learning-rate factor")
print(manifest["data"]["path_for_training_data"])
PY
)"
if [ ! -d "$STAGE2_DATA_DIR" ] || [ ! -d "$V1_DATA_DIR" ]; then
  echo "Error: stage2a or v1 validation dataset is missing" >&2
  exit 1
fi

mapfile -t CANDIDATE_CHECKPOINTS < <(
  find "$RUN_DIR/model_weights" \
    -maxdepth 1 \
    -type f \
    \( -name 'epoch*.pt' -o -name 'best.pt' \) \
    -print \
    | sort -V
)
if [ "${#CANDIDATE_CHECKPOINTS[@]}" -eq 0 ]; then
  echo "Error: no stage2a checkpoints found under $RUN_DIR/model_weights" >&2
  exit 1
fi

EVAL_OUTPUT_DIR="$RUN_DIR/evaluations/dual-valid"
mkdir -p "$EVAL_OUTPUT_DIR"

evaluate_one() {
  local dataset_label="$1"
  local data_dir="$2"
  local checkpoint_label="$3"
  local checkpoint="$4"
  local output="$EVAL_OUTPUT_DIR/${dataset_label}--${checkpoint_label}.json"
  echo "===== $checkpoint_label on complete $dataset_label valid records ====="
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" \
    "$REPO_ROOT/repo/training/evaluate_checkpoint.py" \
    --checkpoint "$checkpoint" \
    --data-dir "$data_dir" \
    --dataset-format tar \
    --split valid \
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
evaluate_one v1 "$V1_DATA_DIR" baseline-stage1 "$BASELINE_CHECKPOINT"
for checkpoint in "${CANDIDATE_CHECKPOINTS[@]}"; do
  label="$(basename "${checkpoint%.pt}")"
  evaluate_one stage2a "$STAGE2_DATA_DIR" "$label" "$checkpoint"
  evaluate_one v1 "$V1_DATA_DIR" "$label" "$checkpoint"
done

SUMMARY_PATH="$EVAL_OUTPUT_DIR/summary.json"
"$PYTHON_BIN" - "$EVAL_OUTPUT_DIR" "$SUMMARY_PATH" "$MAX_V1_NLL_REGRESSION" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
max_v1_regression = float(sys.argv[3])
if max_v1_regression < 0.0:
    raise SystemExit("MAX_V1_NLL_REGRESSION must be non-negative")

rows = {}
population_ids = {}
for path in sorted(result_dir.glob("*.json")):
    if path.name == "summary.json":
        continue
    dataset_label, checkpoint_label = path.stem.split("--", 1)
    result = json.loads(path.read_text(encoding="utf-8"))
    data = result["data"]
    if data.get("split") != "valid" or data.get("evaluation_unit") != "records":
        raise SystemExit(f"unexpected validation population: {path}")
    if data.get("record_count", 0) <= 0:
        raise SystemExit(f"empty validation population: {path}")
    if data.get("evaluated_structures") != data.get("record_count"):
        raise SystemExit(f"incomplete validation evaluation: {path}")
    structure_ids = data["evaluated_structure_ids_sha256"]
    expected_ids = population_ids.setdefault(dataset_label, structure_ids)
    if structure_ids != expected_ids:
        raise SystemExit(f"checkpoint population mismatch: {path}")
    rows.setdefault(checkpoint_label, {})[dataset_label] = {
        "result": str(path),
        "checkpoint": result["checkpoint"],
        "records": data["evaluated_structures"],
        **result["metrics"],
    }

baseline = rows.pop("baseline-stage1")
if set(baseline) != {"stage2a", "v1"}:
    raise SystemExit("baseline did not cover both validation datasets")
candidates = []
for label, datasets in rows.items():
    if set(datasets) != {"stage2a", "v1"}:
        raise SystemExit(f"candidate did not cover both datasets: {label}")
    if datasets["stage2a"]["checkpoint"]["sha256"] != datasets["v1"]["checkpoint"]["sha256"]:
        raise SystemExit(f"candidate checkpoint mismatch between datasets: {label}")
    delta_stage2a = datasets["stage2a"]["nll"] - baseline["stage2a"]["nll"]
    delta_v1 = datasets["v1"]["nll"] - baseline["v1"]["nll"]
    candidates.append(
        {
            "label": label,
            "checkpoint": datasets["stage2a"]["checkpoint"],
            "stage2a": datasets["stage2a"],
            "v1": datasets["v1"],
            "delta": {"stage2a_nll": delta_stage2a, "v1_nll": delta_v1},
            "passes": delta_stage2a < 0.0 and delta_v1 <= max_v1_regression,
        }
    )

eligible = [row for row in candidates if row["passes"]]
selected = min(eligible, key=lambda row: row["stage2a"]["nll"]) if eligible else None
summary = {
    "schema": "proteinmpnn.stage2a_dual_valid_summary.v1",
    "selection_metric": "stage2a_valid_nll_with_v1_regression_gate",
    "max_v1_nll_regression": max_v1_regression,
    "population_ids": population_ids,
    "baseline": baseline,
    "candidates": sorted(candidates, key=lambda row: row["stage2a"]["nll"]),
    "selected": selected,
    "status": "passed" if selected is not None else "no_candidate_passed",
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print()
print("===== stage2a dual-valid gate =====")
print("checkpoint\tstage2a_nll\tdelta_stage2a\tv1_nll\tdelta_v1\tpasses")
for row in summary["candidates"]:
    print(
        f"{row['label']}\t{row['stage2a']['nll']:.6f}\t"
        f"{row['delta']['stage2a_nll']:+.6f}\t{row['v1']['nll']:.6f}\t"
        f"{row['delta']['v1_nll']:+.6f}\t{row['passes']}"
    )
print(f"status: {summary['status']}")
print(f"selected: {selected['label'] if selected else 'none'}")
print(f"summary: {summary_path}")
if selected is None:
    raise SystemExit("no stage2a checkpoint passed the dual validation gate")
PY
