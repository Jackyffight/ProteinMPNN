#!/usr/bin/env python3
"""Promote a validated stage-1 checkpoint into a stable model artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


VALID_SCHEMA = "proteinmpnn.stage1_fixed_valid_summary.v1"
TEST_SCHEMA = "proteinmpnn.selected_test_summary.v1"
PROMOTION_SCHEMA = "proteinmpnn.promoted_checkpoint.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_checkpoint_atomic(source: Path, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        actual_sha256 = sha256_file(destination)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "promotion destination contains different weights: "
                f"{destination} sha256={actual_sha256} expected={expected_sha256}"
            )
        return

    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        copied_sha256 = sha256_file(temporary)
        if copied_sha256 != expected_sha256:
            raise ValueError(
                f"copied checkpoint checksum mismatch: {copied_sha256} != {expected_sha256}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def require_negative_delta(summary: dict, label: str) -> None:
    delta = summary.get("delta") if label == "test" else summary.get("best_candidate_delta")
    if not isinstance(delta, dict) or float(delta.get("nll", 0.0)) >= 0.0:
        raise ValueError(f"{label} NLL did not improve over the official checkpoint")


def promote_checkpoint(run_dir: Path, destination_dir: Path, model_id: str) -> dict:
    run_dir = run_dir.resolve()
    destination_dir = destination_dir.resolve()
    valid_summary_path = run_dir / "evaluations/fixed-valid-records/summary.json"
    test_summary_path = run_dir / "evaluations/selected-test-records/summary.json"
    valid_summary = load_json(valid_summary_path)
    test_summary = load_json(test_summary_path)

    if valid_summary.get("schema") != VALID_SCHEMA:
        raise ValueError(f"unexpected valid summary schema: {valid_summary.get('schema')}")
    if test_summary.get("schema") != TEST_SCHEMA:
        raise ValueError(f"unexpected test summary schema: {test_summary.get('schema')}")
    if valid_summary.get("selection_metric") != "valid_nll":
        raise ValueError("stage-1 checkpoint was not selected by validation NLL")

    valid_candidate = valid_summary.get("best_candidate", {})
    valid_official = valid_summary.get("official", {})
    valid_checkpoint = valid_candidate.get("checkpoint", {})
    test_checkpoint = test_summary.get("selected_checkpoint", {})
    if (
        int(valid_official.get("records", 0)) != 426
        or int(valid_candidate.get("records", 0)) != 426
    ):
        raise ValueError("promotion requires all 426 fixed validation records")
    if int(test_summary.get("records", 0)) != 461:
        raise ValueError("promotion requires all 461 fixed test records")
    require_negative_delta(valid_summary, "valid")
    require_negative_delta(test_summary, "test")

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
        return existing_manifest

    copy_checkpoint_atomic(source_path, destination_checkpoint, expected_sha256)
    write_json_atomic(destination_dir / "fixed-valid-summary.json", valid_summary)
    write_json_atomic(destination_dir / "selected-test-summary.json", test_summary)

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
            "metric": "valid_nll",
            "valid_records": valid_candidate["records"],
            "valid_official": {
                key: valid_summary["official"][key]
                for key in ("nll", "perplexity", "accuracy")
            },
            "valid_selected": {
                key: valid_candidate[key] for key in ("nll", "perplexity", "accuracy")
            },
            "valid_delta": valid_summary["best_candidate_delta"],
            "test_records": test_summary["records"],
            "test_official": test_summary["official"],
            "test_selected": test_summary["selected"],
            "test_delta": test_summary["delta"],
        },
        "intended_use": {
            "checkpoint_mode": "weight_initialization",
            "restore_optimizer": False,
        },
        "evidence": {
            "fixed_valid_summary": "fixed-valid-summary.json",
            "selected_test_summary": "selected-test-summary.json",
        },
    }
    write_json_atomic(destination_manifest, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a stage-1 checkpoint that passed fixed valid/test gates."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--destination-dir", required=True)
    parser.add_argument("--model-id", default="proteinmpnn-2026-v1-stage1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = promote_checkpoint(
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
