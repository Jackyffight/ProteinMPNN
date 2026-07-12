#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

if [ $# -gt 2 ]; then
  echo "Usage: scripts/evaluate_2026_v1_stage1_checkpoints.sh [run-dir] [gpu-index]" >&2
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
RUN_MANIFEST="$RUN_DIR/run_manifest.json"
if [ ! -s "$RUN_MANIFEST" ]; then
  echo "Error: run manifest not found: $RUN_MANIFEST" >&2
  exit 1
fi

DATA_DIR="$($PYTHON_BIN - "$RUN_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest["data"]["path_for_training_data"])
PY
)"
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: training dataset directory not found: $DATA_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" - "$DATA_DIR/manifest.json" "$DATA_DIR/validation.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if manifest.get("format") != "proteinmpnn.tar_shard.v2":
    raise SystemExit("dataset is not proteinmpnn.tar_shard.v2")
if manifest.get("version_id") != "proteinmpnn_pdb_20260708":
    raise SystemExit("dataset is not the pinned proteinmpnn_pdb_20260708 version")
if validation.get("status") != "ok":
    raise SystemExit("dataset validation status is not ok")
if validation.get("records") != manifest.get("record_count"):
    raise SystemExit("dataset validation/manifest record counts differ")
PY

OFFICIAL_CHECKPOINT="$REPO_ROOT/repo/vanilla_model_weights/v_48_020.pt"
"$SCRIPT_DIR/ensure_official_checkpoint.sh" "$OFFICIAL_CHECKPOINT"

mapfile -t CANDIDATE_CHECKPOINTS < <(
  find "$RUN_DIR/model_weights" \
    -maxdepth 1 \
    -type f \
    \( -name 'epoch*.pt' -o -name 'best.pt' \) \
    -print \
    | sort -V
)
if [ "${#CANDIDATE_CHECKPOINTS[@]}" -eq 0 ]; then
  echo "Error: no stage-1 checkpoints found under $RUN_DIR/model_weights" >&2
  exit 1
fi

EVAL_OUTPUT_DIR="$RUN_DIR/evaluations/fixed-valid-records"
mkdir -p "$EVAL_OUTPUT_DIR"
RESULTS=()

evaluate_one() {
  local label="$1"
  local checkpoint="$2"
  local output="$EVAL_OUTPUT_DIR/${label}.json"
  echo "===== evaluating $label on complete 2026 v1 valid records ====="
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" \
    "$REPO_ROOT/repo/training/evaluate_checkpoint.py" \
    --checkpoint "$checkpoint" \
    --data-dir "$DATA_DIR" \
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
  RESULTS+=("$output")
}

evaluate_one "official-v48-020" "$OFFICIAL_CHECKPOINT"
for checkpoint in "${CANDIDATE_CHECKPOINTS[@]}"; do
  label="$(basename "${checkpoint%.pt}")"
  evaluate_one "$label" "$checkpoint"
done

SUMMARY_PATH="$EVAL_OUTPUT_DIR/summary.json"
"$PYTHON_BIN" - "$SUMMARY_PATH" "${RESULTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
result_paths = [Path(value) for value in sys.argv[2:]]
rows = []
reference_ids = None
for path in result_paths:
    result = json.loads(path.read_text(encoding="utf-8"))
    data = result["data"]
    if result.get("schema") != "proteinmpnn.checkpoint_evaluation.v2":
        raise SystemExit(f"unexpected evaluation schema: {path}")
    if data.get("split") != "valid" or data.get("evaluation_unit") != "records":
        raise SystemExit(f"unexpected evaluation population: {path}")
    if data.get("record_count") != 426:
        raise SystemExit(f"expected 426 fixed validation records: {path}")
    if data.get("evaluated_structures") != data.get("record_count"):
        raise SystemExit(f"incomplete held-out evaluation: {path}")
    structure_ids = data["evaluated_structure_ids_sha256"]
    if reference_ids is None:
        reference_ids = structure_ids
    elif structure_ids != reference_ids:
        raise SystemExit(f"checkpoint evaluated on different structures: {path}")
    metrics = result["metrics"]
    rows.append(
        {
            "label": path.stem,
            "result": str(path),
            "checkpoint": result["checkpoint"],
            "records": data["evaluated_structures"],
            "nll": metrics["nll"],
            "perplexity": metrics["perplexity"],
            "accuracy": metrics["accuracy"],
        }
    )

official = next(row for row in rows if row["label"] == "official-v48-020")
candidates = [row for row in rows if row is not official]
best = min(candidates, key=lambda row: row["nll"])
ranked = sorted(rows, key=lambda row: row["nll"])
summary = {
    "schema": "proteinmpnn.stage1_fixed_valid_summary.v1",
    "selection_metric": "valid_nll",
    "evaluated_structure_ids_sha256": reference_ids,
    "official": official,
    "best_candidate": best,
    "best_candidate_delta": {
        "nll": best["nll"] - official["nll"],
        "perplexity": best["perplexity"] - official["perplexity"],
        "accuracy": best["accuracy"] - official["accuracy"],
    },
    "ranked": ranked,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print()
print("===== fixed complete-valid checkpoint ranking =====")
print("rank\tcheckpoint\tepoch\tstep\tnll\tperplexity\taccuracy\trecords")
for rank, row in enumerate(ranked, start=1):
    metadata = row["checkpoint"]["metadata"]
    print(
        f"{rank}\t{row['label']}\t{metadata.get('epoch')}\t{metadata.get('step')}\t"
        f"{row['nll']:.6f}\t{row['perplexity']:.6f}\t{row['accuracy']:.6f}\t{row['records']}"
    )
delta = summary["best_candidate_delta"]
print()
print(f"selected: {best['label']}")
print(
    "delta_vs_official: "
    f"nll={delta['nll']:+.6f} perplexity={delta['perplexity']:+.6f} "
    f"accuracy={delta['accuracy']:+.6f}"
)
print(f"summary: {summary_path}")
PY
