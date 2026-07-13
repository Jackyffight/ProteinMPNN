#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

EXPECTED_BENCHMARK_ID="${EXPECTED_BENCHMARK_ID:-pdb-valid-7136a4ecae1956027aa6}"
RUN_STEM="esmfold2-fast-${EXPECTED_BENCHMARK_ID}"
DEFAULT_PREDICTION_RUN="$PROTEINMPNN_OUTPUT_ROOT/benchmarks/${RUN_STEM}-full-l3-s50-seed42"
PREDICTION_RUN="${PREDICTION_RUN:-$DEFAULT_PREDICTION_RUN}"
EVAL_DIR="${EVAL_DIR:-$PREDICTION_RUN/evaluations/native-structure-agreement-v1}"
REPORT_PYTHON="${PROTEINMPNN_PYTHON:-python}"
LIMIT="${1:-8}"

if [ "$LIMIT" = "-h" ] || [ "$LIMIT" = "--help" ]; then
  echo "Usage: scripts/report_esmfold2_native_agreement.sh [lowest_lddt_count]"
  exit 0
fi
if [ $# -gt 1 ]; then
  echo "Usage: scripts/report_esmfold2_native_agreement.sh [lowest_lddt_count]" >&2
  exit 2
fi
if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || [ "$LIMIT" -gt 40 ]; then
  echo "Error: lowest_lddt_count must be an integer from 1 to 40." >&2
  exit 2
fi

SUMMARY_PATH="$EVAL_DIR/summary.json"
RECORDS_PATH="$EVAL_DIR/records.jsonl"
if [ ! -f "$SUMMARY_PATH" ]; then
  echo "Error: native agreement summary not found: $SUMMARY_PATH" >&2
  exit 1
fi
if [ ! -f "$RECORDS_PATH" ]; then
  echo "Error: native agreement records not found: $RECORDS_PATH" >&2
  exit 1
fi

"$REPORT_PYTHON" - "$SUMMARY_PATH" "$RECORDS_PATH" "$LIMIT" <<'PY'
import json
import math
from pathlib import Path
import sys


summary_path = Path(sys.argv[1])
records_path = Path(sys.argv[2])
limit = int(sys.argv[3])


def read_document(path):
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"JSON root is not an object: {path}")
    return value


def finite(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"invalid numeric field {label}: {value}") from error
    if not math.isfinite(number):
        raise SystemExit(f"non-finite numeric field {label}: {value}")
    return number


summary = read_document(summary_path)
if summary.get("schema_version") != "protein-mrna.native-agreement-summary.v1":
    raise SystemExit("unexpected native agreement summary schema")
evaluation_identity = summary.get("evaluation_identity")
records = []
with records_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"invalid records JSON at line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise SystemExit(f"record at line {line_number} is not an object")
        if record.get("evaluation_identity") != evaluation_identity:
            raise SystemExit(f"evaluation identity mismatch at line {line_number}")
        if record.get("status") == "succeeded":
            records.append(record)

counts = summary.get("records", {})
if len(records) != int(counts.get("succeeded", -1)):
    raise SystemExit("summary and records.jsonl succeeded counts differ")
records.sort(key=lambda row: finite(row["metrics"]["ca_lddt"], "ca_lddt"))
overall = summary["overall"]
correlations = summary["confidence_correlations"]

print("=== native structure agreement ===")
print(f"status: {summary['status']}")
print(
    "records: "
    f"{counts['succeeded']}/{counts['selected']} succeeded, "
    f"{counts['failed']} failed, {counts['pending']} pending"
)
for field, label in (
    ("ca_lddt", "CA lDDT"),
    ("ca_tm_score_resolved", "CA TM resolved"),
    ("ca_tm_score_full_length", "CA TM full"),
    ("ca_rmsd_angstrom", "CA RMSD A"),
    ("native_ca_coverage", "native CA coverage"),
):
    values = overall[field]
    print(
        f"{label}: mean={finite(values['mean'], field):.4f} "
        f"median={finite(values['median'], field):.4f} "
        f"min={finite(values['min'], field):.4f} "
        f"max={finite(values['max'], field):.4f}"
    )
print(
    "confidence Pearson: "
    f"pLDDT/lDDT={finite(correlations['mean_plddt_vs_ca_lddt_pearson'], 'pLDDT correlation'):.4f} "
    "pTM/TM-full="
    f"{finite(correlations['predicted_ptm_vs_ca_tm_full_length_pearson'], 'pTM correlation'):.4f}"
)

print()
print(f"=== lowest CA lDDT records ({min(limit, len(records))}) ===")
print(
    f"{'record':<12} {'source_chain':<16} {'len':>5} {'lDDT':>7} "
    f"{'TM-res':>7} {'TM-full':>7} {'RMSD':>8} {'coverage':>9} "
    f"{'pLDDT':>7} {'pTM':>7}"
)
for record in records[:limit]:
    metrics = record["metrics"]
    confidence = record["prediction_confidence"]
    print(
        f"{record['benchmark_record_id']:<12} "
        f"{record['source_chain_id']:<16} "
        f"{int(record['sequence_length']):>5} "
        f"{finite(metrics['ca_lddt'], 'ca_lddt'):>7.4f} "
        f"{finite(metrics['ca_tm_score_resolved'], 'tm_resolved'):>7.4f} "
        f"{finite(metrics['ca_tm_score_full_length'], 'tm_full'):>7.4f} "
        f"{finite(metrics['ca_rmsd_angstrom'], 'rmsd'):>8.3f} "
        f"{finite(metrics['native_ca_coverage'], 'coverage'):>9.4f} "
        f"{finite(confidence['mean_plddt'], 'pLDDT'):>7.4f} "
        f"{finite(confidence['ptm'], 'pTM'):>7.4f}"
    )

print()
print(f"summary: {summary_path}")
print(f"records: {records_path}")
PY
