#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

if [ $# -gt 1 ]; then
  echo "Usage: scripts/evaluate_2026_v1_stage1_multiseed.sh [run-dir]" >&2
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

SPLIT="${SPLIT:-valid}"
SEEDS="${SEEDS:-11 23 42 67 101}"
PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"
read -r -a seed_values <<< "$SEEDS"
if [ "$SPLIT" != "valid" ]; then
  echo "Error: multi-seed sensitivity evaluation is valid-only, got: $SPLIT" >&2
  exit 1
fi
if [ "${#seed_values[@]}" -lt 2 ]; then
  echo "Error: SEEDS must contain at least two integer seeds." >&2
  exit 1
fi
for seed in "${seed_values[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "Error: invalid seed: $seed" >&2
    exit 1
  fi
done

for seed in "${seed_values[@]}"; do
  output_dir="$RUN_DIR/evaluations/${SPLIT}-seed-${seed}"
  echo
  echo "===== paired evaluation seed=$seed split=$SPLIT ====="
  SEED="$seed" \
  SPLIT="$SPLIT" \
  EVAL_OUTPUT_DIR="$output_dir" \
    "$SCRIPT_DIR/evaluate_2026_v1_stage1.sh" "$RUN_DIR"
done

"$PYTHON_BIN" - "$RUN_DIR" "$SPLIT" "${seed_values[@]}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
split = sys.argv[2]
seeds = [int(value) for value in sys.argv[3:]]
rows = []

for seed in seeds:
    result_dir = run_dir / "evaluations" / f"{split}-seed-{seed}"
    official_path = result_dir / f"official-v48-020-{split}.json"
    candidate_path = result_dir / f"stage1-best-{split}.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        official["data"]["evaluated_structure_ids_sha256"]
        != candidate["data"]["evaluated_structure_ids_sha256"]
    ):
        raise SystemExit(f"seed {seed}: official/candidate structure IDs differ")

    official_metrics = official["metrics"]
    candidate_metrics = candidate["metrics"]
    rows.append(
        {
            "seed": seed,
            "official_perplexity": official_metrics["perplexity"],
            "candidate_perplexity": candidate_metrics["perplexity"],
            "perplexity_delta": (
                candidate_metrics["perplexity"] - official_metrics["perplexity"]
            ),
            "accuracy_delta": (
                candidate_metrics["accuracy"] - official_metrics["accuracy"]
            ),
        }
    )

perplexity_deltas = [row["perplexity_delta"] for row in rows]
accuracy_deltas = [row["accuracy_delta"] for row in rows]
wins = sum(delta < 0 for delta in perplexity_deltas)
mean_perplexity_delta = statistics.fmean(perplexity_deltas)
mean_accuracy_delta = statistics.fmean(accuracy_deltas)
if wins == len(rows):
    status = "consistently_improved"
elif mean_perplexity_delta < 0:
    status = "mean_improved_but_mixed"
else:
    status = "not_improved"

print()
print("===== multi-seed summary =====")
print(f"split: {split}")
print("seed\tofficial_ppl\tstage1_best_ppl\tdelta_ppl\tdelta_accuracy")
for row in rows:
    print(
        f"{row['seed']}\t"
        f"{row['official_perplexity']:.6f}\t"
        f"{row['candidate_perplexity']:.6f}\t"
        f"{row['perplexity_delta']:+.6f}\t"
        f"{row['accuracy_delta']:+.6f}"
    )
print(f"wins: {wins}/{len(rows)}")
print(f"mean_delta: perplexity={mean_perplexity_delta:+.6f} accuracy={mean_accuracy_delta:+.6f}")
print(f"range_delta_perplexity: {min(perplexity_deltas):+.6f}..{max(perplexity_deltas):+.6f}")
print(f"status: {status}")
PY
