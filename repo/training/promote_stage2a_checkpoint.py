#!/usr/bin/env python3
"""Promote a stage2a checkpoint that passed the dual valid/test gates."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

from promote_checkpoint import (
    PROMOTION_SCHEMA,
    copy_checkpoint_atomic,
    load_json,
    sha256_file,
    write_json_atomic,
)


VALID_SCHEMA = "proteinmpnn.stage2a_dual_valid_summary.v1"
TEST_SCHEMA = "proteinmpnn.stage2a_dual_test_summary.v1"
VALID_SELECTION_METRIC = "stage2a_valid_nll_with_v1_regression_gate"
EXPECTED_RECORDS = {
    "valid": {"stage2a": 19, "v1": 426},
    "test": {"stage2a": 13, "v1": 461},
}


def require_record_count(metrics: dict, expected: int, label: str) -> None:
    if not isinstance(metrics, dict) or int(metrics.get("records", 0)) != expected:
        raise ValueError(f"promotion requires all {expected} {label} records")


def require_consistent_delta(
    baseline: dict,
    selected: dict,
    recorded_delta: float,
    label: str,
) -> None:
    if not isinstance(baseline, dict) or not isinstance(selected, dict):
        raise ValueError(f"missing {label} metrics")
    observed_delta = float(selected["nll"]) - float(baseline["nll"])
    if not math.isclose(observed_delta, recorded_delta, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{label} NLL delta mismatch: recorded={recorded_delta} "
            f"observed={observed_delta}"
        )


def require_dual_gate(
    stage2a_delta: float,
    v1_delta: float,
    max_v1_regression: float,
    label: str,
) -> None:
    if stage2a_delta >= 0.0:
        raise ValueError(f"{label} stage2a NLL did not improve over the stage-1 baseline")
    if v1_delta > max_v1_regression:
        raise ValueError(
            f"{label} v1 NLL regression exceeds the gate: "
            f"{v1_delta} > {max_v1_regression}"
        )


def promote_stage2a_checkpoint(
    run_dir: Path,
    destination_dir: Path,
    model_id: str,
) -> dict:
    run_dir = run_dir.resolve()
    destination_dir = destination_dir.resolve()
    valid_summary_path = run_dir / "evaluations/dual-valid/summary.json"
    test_summary_path = run_dir / "evaluations/selected-test-records/summary.json"
    valid_summary = load_json(valid_summary_path)
    test_summary = load_json(test_summary_path)

    if valid_summary.get("schema") != VALID_SCHEMA:
        raise ValueError(f"unexpected valid summary schema: {valid_summary.get('schema')}")
    if test_summary.get("schema") != TEST_SCHEMA:
        raise ValueError(f"unexpected test summary schema: {test_summary.get('schema')}")
    if valid_summary.get("selection_metric") != VALID_SELECTION_METRIC:
        raise ValueError("stage2a checkpoint was not selected by the dual validation gate")
    if valid_summary.get("status") != "passed" or test_summary.get("status") != "passed":
        raise ValueError("stage2a promotion requires passed valid and test gates")

    max_v1_regression = float(valid_summary.get("max_v1_nll_regression", -1.0))
    test_max_v1_regression = float(test_summary.get("max_v1_nll_regression", -1.0))
    if max_v1_regression < 0.0 or test_max_v1_regression != max_v1_regression:
        raise ValueError("valid and test summaries use different v1 regression gates")

    valid_selected = valid_summary.get("selected")
    if not isinstance(valid_selected, dict) or valid_selected.get("passes") is not True:
        raise ValueError("valid summary does not contain a passing selected checkpoint")
    selected_label = str(valid_selected.get("label", ""))
    if not selected_label or selected_label != str(test_summary.get("selected_label", "")):
        raise ValueError("valid and test summaries selected different checkpoint labels")

    valid_checkpoint = valid_selected.get("checkpoint")
    test_checkpoint = test_summary.get("selected_checkpoint")
    if not isinstance(valid_checkpoint, dict) or not isinstance(test_checkpoint, dict):
        raise ValueError("valid or test summary is missing checkpoint metadata")
    expected_sha256 = str(valid_checkpoint.get("sha256", ""))
    if not expected_sha256 or expected_sha256 != str(test_checkpoint.get("sha256", "")):
        raise ValueError("valid and test summaries selected different checkpoint hashes")

    source_path = Path(str(valid_checkpoint.get("path", ""))).expanduser().resolve()
    test_source_path = Path(str(test_checkpoint.get("path", ""))).expanduser().resolve()
    if source_path != test_source_path:
        raise ValueError("valid and test summaries selected different checkpoint paths")
    if not source_path.is_file():
        raise ValueError(f"selected checkpoint not found: {source_path}")
    if run_dir not in source_path.parents:
        raise ValueError(f"selected checkpoint is outside the source run: {source_path}")
    actual_sha256 = sha256_file(source_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"selected checkpoint checksum mismatch: {actual_sha256} != {expected_sha256}"
        )

    valid_baseline = valid_summary.get("baseline", {})
    valid_deltas = valid_selected.get("delta", {})
    valid_stage2a = valid_selected.get("stage2a", {})
    valid_v1 = valid_selected.get("v1", {})
    require_record_count(
        valid_baseline.get("stage2a", {}),
        EXPECTED_RECORDS["valid"]["stage2a"],
        "baseline stage2a valid",
    )
    require_record_count(
        valid_baseline.get("v1", {}),
        EXPECTED_RECORDS["valid"]["v1"],
        "baseline v1 valid",
    )
    require_record_count(
        valid_stage2a,
        EXPECTED_RECORDS["valid"]["stage2a"],
        "stage2a valid",
    )
    require_record_count(valid_v1, EXPECTED_RECORDS["valid"]["v1"], "v1 valid")
    valid_stage2a_delta = float(valid_deltas.get("stage2a_nll", 0.0))
    valid_v1_delta = float(valid_deltas.get("v1_nll", float("inf")))
    require_consistent_delta(
        valid_baseline.get("stage2a", {}),
        valid_stage2a,
        valid_stage2a_delta,
        "valid stage2a",
    )
    require_consistent_delta(
        valid_baseline.get("v1", {}),
        valid_v1,
        valid_v1_delta,
        "valid v1",
    )
    require_dual_gate(
        valid_stage2a_delta,
        valid_v1_delta,
        max_v1_regression,
        "valid",
    )

    test_stage2a = test_summary.get("stage2a", {})
    test_v1 = test_summary.get("v1", {})
    require_record_count(
        test_stage2a,
        EXPECTED_RECORDS["test"]["stage2a"],
        "stage2a test",
    )
    require_record_count(test_v1, EXPECTED_RECORDS["test"]["v1"], "v1 test")
    test_stage2a_delta = float(test_stage2a.get("delta_nll", 0.0))
    test_v1_delta = float(test_v1.get("delta_nll", float("inf")))
    require_consistent_delta(
        test_stage2a.get("baseline", {}),
        test_stage2a.get("selected", {}),
        test_stage2a_delta,
        "test stage2a",
    )
    require_consistent_delta(
        test_v1.get("baseline", {}),
        test_v1.get("selected", {}),
        test_v1_delta,
        "test v1",
    )
    require_dual_gate(test_stage2a_delta, test_v1_delta, max_v1_regression, "test")

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_checkpoint = destination_dir / "model.pt"
    destination_manifest = destination_dir / "promotion.json"
    if destination_manifest.is_file():
        existing_manifest = load_json(destination_manifest)
        if existing_manifest.get("schema") != PROMOTION_SCHEMA:
            raise ValueError("existing promotion manifest has an unexpected schema")
        if existing_manifest.get("model_id") != model_id:
            raise ValueError("existing promotion manifest has a different model ID")
        existing_sha256 = existing_manifest.get("checkpoint", {}).get("sha256")
        if existing_sha256 != expected_sha256:
            raise ValueError(
                "promotion manifest already points to different weights: "
                f"{existing_sha256} != {expected_sha256}"
            )
        copy_checkpoint_atomic(source_path, destination_checkpoint, expected_sha256)
        write_json_atomic(destination_dir / "dual-valid-summary.json", valid_summary)
        write_json_atomic(destination_dir / "dual-test-summary.json", test_summary)
        return existing_manifest

    copy_checkpoint_atomic(source_path, destination_checkpoint, expected_sha256)
    write_json_atomic(destination_dir / "dual-valid-summary.json", valid_summary)
    write_json_atomic(destination_dir / "dual-test-summary.json", test_summary)

    manifest = {
        "schema": PROMOTION_SCHEMA,
        "model_id": model_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "checkpoint": {
            "path": str(destination_checkpoint),
            "sha256": expected_sha256,
            "bytes": destination_checkpoint.stat().st_size,
            "metadata": test_checkpoint.get("metadata"),
            "source_path": str(source_path),
        },
        "selection": {
            "metric": VALID_SELECTION_METRIC,
            "max_v1_nll_regression": max_v1_regression,
            "validation": {
                "stage2a_records": EXPECTED_RECORDS["valid"]["stage2a"],
                "v1_records": EXPECTED_RECORDS["valid"]["v1"],
                "stage2a_delta_nll": valid_stage2a_delta,
                "v1_delta_nll": valid_v1_delta,
            },
            "test": {
                "stage2a_records": EXPECTED_RECORDS["test"]["stage2a"],
                "v1_records": EXPECTED_RECORDS["test"]["v1"],
                "stage2a_delta_nll": test_stage2a_delta,
                "v1_delta_nll": test_v1_delta,
            },
        },
        "intended_use": {
            "checkpoint_mode": "weight_initialization",
            "restore_optimizer": False,
            "inference": True,
        },
        "evidence": {
            "dual_valid_summary": "dual-valid-summary.json",
            "dual_test_summary": "dual-test-summary.json",
        },
    }
    write_json_atomic(destination_manifest, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a stage2a checkpoint that passed dual valid/test gates."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--destination-dir", required=True)
    parser.add_argument("--model-id", default="proteinmpnn-2026-stage2a")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = promote_stage2a_checkpoint(
        Path(args.run_dir), Path(args.destination_dir), args.model_id
    )
    checkpoint = manifest["checkpoint"]
    print(f"promoted_checkpoint: {checkpoint['path']}")
    print(f"sha256: {checkpoint['sha256']}")
    print(f"source_run: {manifest['source_run']}")
    print(f"manifest: {Path(args.destination_dir).resolve() / 'promotion.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
