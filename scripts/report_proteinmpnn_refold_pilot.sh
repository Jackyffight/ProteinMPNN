#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

EXPECTED_BENCHMARK_ID="${EXPECTED_BENCHMARK_ID:-pdb-valid-7136a4ecae1956027aa6}"
PILOT_DIR="${PILOT_DIR:-$PROTEINMPNN_OUTPUT_ROOT/benchmarks/proteinmpnn-refold-pilot-${EXPECTED_BENCHMARK_ID}-t010-s4}"
EVALUATION_DIR="${EVALUATION_DIR:-$PILOT_DIR/evaluations/dual-reference-v1}"
REPORT_PYTHON="${PROTEINMPNN_PYTHON:-python}"
SUMMARY_PATH="$EVALUATION_DIR/summary.json"
MANIFEST_PATH="$PILOT_DIR/pilot-manifest.json"

if [ $# -ne 0 ]; then
  echo "Usage: scripts/report_proteinmpnn_refold_pilot.sh" >&2
  exit 2
fi
for path in "$SUMMARY_PATH" "$MANIFEST_PATH"; do
  if [ ! -f "$path" ]; then
    echo "Error: pilot report input not found: $path" >&2
    exit 1
  fi
done

"$REPORT_PYTHON" - "$SUMMARY_PATH" "$MANIFEST_PATH" <<'PY'
import json
import math
from pathlib import Path
import sys


summary_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if summary.get("schema_version") != "protein-mrna.proteinmpnn-refold-evaluation-summary.v1":
    raise SystemExit("unexpected ProteinMPNN refold summary schema")
if summary.get("evaluation_identity") is None:
    raise SystemExit("ProteinMPNN refold summary has no identity")


def number(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise SystemExit(f"non-finite report value: {label}")
    return result


def mean(label, field):
    return number(summary["by_model"][label][field]["mean"], f"{label}/{field}")


def backbone_mean(record_id, label, field):
    return number(
        summary["by_backbone"][record_id][label][field]["mean"],
        f"{record_id}/{label}/{field}",
    )


official = "official-v48-020"
stage2a = "stage2a"
counts = summary["records"]
paired = summary["paired"]
if (
    summary.get("status") != "passed"
    or counts != {"selected": 32, "succeeded": 32, "failed": 0, "pending": 0}
    or paired.get("pairs") != 16
    or len(manifest.get("selection", {}).get("records", [])) != 4
):
    raise SystemExit("ProteinMPNN paired refold pilot is incomplete")
print("=== ProteinMPNN paired refold pilot ===")
print(f"status: {summary['status']}")
print(
    f"records: {counts['succeeded']}/{counts['selected']} succeeded, "
    f"{counts['failed']} failed, {counts['pending']} pending"
)
print(f"paired comparisons: {paired['pairs']}")
print()
print(f"{'metric':<38} {'official':>10} {'stage2a':>10} {'delta':>10}")
for field, display in (
    ("sequence_recovery", "sequence recovery"),
    ("sampled_nll", "ProteinMPNN self-scored sampled NLL"),
    ("experimental_ca_lddt", "experimental CA lDDT"),
    ("experimental_ca_tm_score_resolved", "experimental CA TM resolved"),
    ("experimental_ca_tm_score_full_length", "experimental CA TM full"),
    ("experimental_ca_rmsd_angstrom", "experimental CA RMSD A"),
    (
        "delta_experimental_ca_lddt_vs_native_sequence",
        "lDDT delta vs native-sequence fold",
    ),
    (
        "delta_experimental_tm_resolved_vs_native_sequence",
        "TM-res delta vs native-sequence fold",
    ),
    (
        "delta_experimental_tm_full_length_vs_native_sequence",
        "TM-full delta vs native-sequence fold",
    ),
    (
        "delta_experimental_ca_rmsd_vs_native_sequence",
        "RMSD delta vs native-sequence fold",
    ),
    ("native_prediction_ca_lddt", "native-prediction-reference lDDT"),
    ("native_prediction_ca_tm_score", "native-prediction-reference TM"),
    ("native_prediction_ca_rmsd_angstrom", "native-prediction-reference RMSD A"),
    ("refold_mean_plddt", "refold mean pLDDT"),
    ("refold_ptm", "refold pTM"),
):
    official_value = mean(official, field)
    stage2a_value = mean(stage2a, field)
    print(
        f"{display:<38} {official_value:>10.4f} "
        f"{stage2a_value:>10.4f} {stage2a_value - official_value:>+10.4f}"
    )

print()
print("stage2a higher-is-better structural wins:")
for field, wins in paired["stage2a_wins"].items():
    print(f"  {field}: {wins}/{paired['pairs']}")

print("stage2a lower-is-better wins:")
for field, wins in paired["stage2a_lower_is_better_wins"].items():
    print(f"  {field}: {wins}/{paired['pairs']}")

print()
print("selected backbones:")
for record in manifest["selection"]["records"]:
    print(
        f"  {record['selection_role']}: {record['benchmark_record_id']} "
        f"chain={record['source_chain_id']} length={record['sequence_length']}"
    )

print()
print("per-backbone delta (stage2a - official):")
print(
    f"{'record':<12} {'len':>5} {'lDDT':>8} {'TM-res':>8} "
    f"{'TM-full':>8} {'RMSD':>8} {'ref-TM':>8} {'ref-RMSD':>9}"
)
for record in manifest["selection"]["records"]:
    record_id = record["benchmark_record_id"]
    deltas = {
        field: backbone_mean(record_id, stage2a, field)
        - backbone_mean(record_id, official, field)
        for field in (
            "experimental_ca_lddt",
            "experimental_ca_tm_score_resolved",
            "experimental_ca_tm_score_full_length",
            "experimental_ca_rmsd_angstrom",
            "native_prediction_ca_tm_score",
            "native_prediction_ca_rmsd_angstrom",
        )
    }
    print(
        f"{record_id:<12} {record['sequence_length']:>5} "
        f"{deltas['experimental_ca_lddt']:>+8.4f} "
        f"{deltas['experimental_ca_tm_score_resolved']:>+8.4f} "
        f"{deltas['experimental_ca_tm_score_full_length']:>+8.4f} "
        f"{deltas['experimental_ca_rmsd_angstrom']:>+8.3f} "
        f"{deltas['native_prediction_ca_tm_score']:>+8.4f} "
        f"{deltas['native_prediction_ca_rmsd_angstrom']:>+9.3f}"
    )

print()
print(f"summary: {summary_path}")
print(f"pilot_manifest: {manifest_path}")
PY
