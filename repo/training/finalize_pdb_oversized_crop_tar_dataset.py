#!/usr/bin/env python3
"""Finalize completed stage2a shards after an interrupted metadata phase."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import shutil
import time
from collections import Counter
from pathlib import Path

import build_pdb_mmcif_dataset as base
import build_pdb_mmcif_tar_shard_dataset as tar_builder
import build_pdb_oversized_crop_tar_dataset as crop_builder
from tar_shard_utils import TarShardStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--deferred-manifest", required=True)
    parser.add_argument("--reference-dataset", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--cluster-file", required=True)
    parser.add_argument("--entries-index", required=True)
    parser.add_argument("--max-shard-size", default="1g")
    return parser.parse_args()


def copy_recovery_source(path: Path, recovery_dir: Path) -> Path:
    source_copy = recovery_dir / path.name
    if not source_copy.exists():
        shutil.copy2(path, source_copy)
    return source_copy


def reconstruct_rows(
    out_dir: Path,
    source_index_path: Path,
) -> tuple[list[dict], dict[str, dict]]:
    shutil.copy2(source_index_path, out_dir / "index.jsonl")
    store = TarShardStore(out_dir)
    index_rows = crop_builder.read_jsonl(source_index_path)
    rows = []
    crop_stats_by_chain = {}
    for number, index_row in enumerate(index_rows, start=1):
        chain_id = index_row["chain_id"]
        payload = store.load_payload_for_chain(chain_id)
        if payload.get("format") != base.TAR_SHARD_FORMAT:
            raise ValueError(f"unexpected payload format: {chain_id}")
        crop = payload.get("meta", {}).get("crop")
        if crop is None or crop.get("policy") != base.SPATIAL_CROP_POLICY:
            raise ValueError(f"payload is not a stage2a crop: {chain_id}")
        target_chain = payload["meta"]["target_chain"]
        rows.append(
            {
                "CHAINID": chain_id,
                "DEPOSITION": index_row["date"],
                "RESOLUTION": str(float(index_row["resolution"])),
                "HASH": index_row["hash"],
                "CLUSTER": str(index_row["cluster"]),
                "SEQUENCE": payload["chains"][target_chain]["seq"],
            }
        )
        crop_stats_by_chain[chain_id] = {
            "context_chains": len(payload["chains"]),
            **crop,
        }
        if number % 1000 == 0 or number == len(index_rows):
            print(f"reconstructed={number}/{len(index_rows)}", flush=True)
    return rows, crop_stats_by_chain


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    started = time.time()
    out_dir = Path(args.out_dir).expanduser().resolve()
    deferred_manifest = Path(args.deferred_manifest).expanduser().resolve()
    reference_dataset = Path(args.reference_dataset).expanduser().resolve()
    chain_index_path = out_dir / "index.jsonl"
    record_index_path = out_dir / "records.jsonl"
    shards_dir = out_dir / "shards"
    for path in (
        deferred_manifest,
        reference_dataset / "manifest.json",
        reference_dataset / "validation.json",
        reference_dataset / "list.csv",
        reference_dataset / "valid_clusters.txt",
        reference_dataset / "test_clusters.txt",
        chain_index_path,
        record_index_path,
    ):
        if not path.is_file():
            raise SystemExit(f"required recovery input not found: {path}")
    if not shards_dir.is_dir():
        raise SystemExit(f"shards directory not found: {shards_dir}")

    recovery_dir = out_dir / "_recovery"
    recovery_dir.mkdir(exist_ok=True)
    source_chain_index = copy_recovery_source(chain_index_path, recovery_dir)
    source_record_index = copy_recovery_source(record_index_path, recovery_dir)
    source_chain_rows = crop_builder.read_jsonl(source_chain_index)
    source_record_rows = crop_builder.read_jsonl(source_record_index)
    if not source_chain_rows or len(source_chain_rows) != len(source_record_rows):
        raise SystemExit("recovery source index counts are empty or inconsistent")

    rows, crop_stats_by_chain = reconstruct_rows(out_dir, source_chain_index)
    split_stats, quarantined = crop_builder.inherit_reference_splits(
        rows, reference_dataset
    )
    shutil.copy2(source_record_index, record_index_path)
    crop_builder.filter_and_rewrite_indexes(
        chain_index_path,
        record_index_path,
        rows,
    )

    rows.sort(key=lambda row: row["CHAINID"])
    with (out_dir / "list.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "CHAINID",
                "DEPOSITION",
                "RESOLUTION",
                "HASH",
                "CLUSTER",
                "SEQUENCE",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    crop_builder.write_cluster_ids(
        out_dir / "valid_clusters.txt",
        split_stats.pop("stage_valid_cluster_ids"),
    )
    crop_builder.write_cluster_ids(
        out_dir / "test_clusters.txt",
        split_stats.pop("stage_test_cluster_ids"),
    )
    write_jsonl(out_dir / "build_quarantined.jsonl", quarantined)

    deferred_rows = crop_builder.read_jsonl(deferred_manifest)
    parsed_entry_ids = {row["entry_id"] for row in source_record_rows}
    skipped = []
    for row in deferred_rows:
        if row.get("entry_id") in parsed_entry_ids:
            continue
        skipped.append(
            {
                **row,
                "original_deferred_reason": row.get("reason"),
                "reason": "target_too_long_for_complete_crop",
                "status": "skipped",
                "recovered_from_completed_indexes": True,
            }
        )
    if len(source_record_rows) + len(skipped) != len(deferred_rows):
        raise ValueError("deferred manifest could not be reconciled with parsed records")
    write_jsonl(out_dir / "build_skipped.jsonl", skipped)

    kept_chain_ids = {row["CHAINID"] for row in rows}
    record_count = len(rows)
    context_chain_count = sum(
        crop_stats_by_chain[chain_id]["context_chains"]
        for chain_id in kept_chain_ids
    )
    original_context_residues = sum(
        crop_stats_by_chain[chain_id]["original_context_length"]
        for chain_id in kept_chain_ids
    )
    retained_context_residues = sum(
        crop_stats_by_chain[chain_id]["retained_context_length"]
        for chain_id in kept_chain_ids
    )
    shard_summaries = crop_builder.summarize_existing_shards(
        shards_dir, record_index_path
    )

    base.CLUSTER_MAP = base.load_cluster_map(args.cluster_file)
    base.ENTRY_METADATA = base.load_entry_metadata(args.entries_index)
    if not base.CLUSTER_MAP or not base.ENTRY_METADATA:
        raise ValueError("cluster map or entry metadata is empty")
    max_shard_size = tar_builder.parse_size(args.max_shard_size)
    source_reason_counts = Counter(
        row.get("reason", "unknown") for row in deferred_rows
    )
    build_manifest = {
        "version_id": args.version_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deferred_manifest": str(deferred_manifest),
        "deferred_manifest_sha256": tar_builder.sha256_file(deferred_manifest),
        "reference_dataset": str(reference_dataset),
        "reference_files_sha256": {
            filename: tar_builder.sha256_file(reference_dataset / filename)
            for filename in (
                "manifest.json",
                "validation.json",
                "list.csv",
                "valid_clusters.txt",
                "test_clusters.txt",
            )
        },
        "cluster_file": args.cluster_file,
        "entries_index": args.entries_index,
        "cluster_map_entries": len(base.CLUSTER_MAP),
        "entry_metadata_records": len(base.ENTRY_METADATA),
        "concurrency": {
            "workers": 1,
            "max_in_flight": 1,
            "worker_recycle_tasks": 1,
            "start_method": "fork",
        },
        "resources": {
            "worker_peak_rss_kib": None,
            "finalizer_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "crop_policy": base.SPATIAL_CROP_POLICY,
        "split_policy": (
            "inherit_v1_reference_clusters_stage2_only_clusters_train_"
            "quarantine_conflicting_components"
        ),
        "filters": {
            "max_resolution": 3.5,
            "min_date": "2021-08-03",
            "max_date": "2026-07-08",
            "min_chain_length": 30,
            "max_chain_length": 10000,
            "max_context_length": 2000,
            "min_context_crop_length": 30,
            "min_resolved_residues": 30,
            "min_backbone_coverage": 0.5,
            "max_chains": len(base.CHAIN_IDS),
            "method_allow": ["ELECTRON MICROSCOPY", "X-RAY DIFFRACTION"],
        },
        "counts": {
            "source_manifest_rows": len(deferred_rows),
            "input_files": len(deferred_rows),
            "parsed_crop_payloads": len(source_record_rows),
            "published_records": record_count,
            "records": record_count,
            "context_chains": context_chain_count,
            "original_context_residues": original_context_residues,
            "retained_context_residues": retained_context_residues,
            "failures": 0,
            "ok": len(source_record_rows),
            "target_too_long_for_complete_crop": len(skipped),
            **split_stats,
        },
        "source_reason_counts": dict(source_reason_counts),
        "skip_reasons": {
            "target_too_long_for_complete_crop": len(skipped),
        },
        "recovery": {
            "mode": "finalize_completed_shards_after_split_conflict",
            "source_chain_index": str(source_chain_index.relative_to(out_dir)),
            "source_chain_index_sha256": tar_builder.sha256_file(source_chain_index),
            "source_record_index": str(source_record_index.relative_to(out_dir)),
            "source_record_index_sha256": tar_builder.sha256_file(source_record_index),
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "format": base.TAR_SHARD_FORMAT,
        "version_id": args.version_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset": str(deferred_manifest),
        "reference_dataset": str(reference_dataset),
        "record_granularity": crop_builder.RECORD_GRANULARITY,
        "index_granularity": "target_chain",
        "payload_schema": crop_builder.PAYLOAD_SCHEMA,
        "target_selection_policy": "max_resolved_backbone_then_coverage_then_source_id",
        "crop_policy": base.SPATIAL_CROP_POLICY,
        "quarantine_policy": "exclude_reference_split_conflict_components",
        "quarantined_payload_count": len(quarantined),
        "record_count": record_count,
        "target_count": record_count,
        "context_chain_count": context_chain_count,
        "max_shard_size": max_shard_size,
        "shards": shard_summaries,
        "files": {
            "chain_index": "index.jsonl",
            "record_index": "records.jsonl",
            "list": "list.csv",
            "valid_clusters": "valid_clusters.txt",
            "test_clusters": "test_clusters.txt",
            "build_manifest": "build_manifest.json",
            "quarantine": "build_quarantined.jsonl",
        },
        "build": build_manifest,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README").write_text(
        f"{args.version_id}\nProteinMPNN oversized spatial-crop continuation dataset.\n",
        encoding="utf-8",
    )
    print(json.dumps(build_manifest["counts"], indent=2, sort_keys=True))
    print(f"manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
