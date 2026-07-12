#!/usr/bin/env python3
"""Build bounded spatial crops from assemblies deferred by the 2026 v1 build."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import resource
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import build_pdb_mmcif_dataset as base
import build_pdb_mmcif_tar_shard_dataset as tar_builder


PAYLOAD_SCHEMA = "structure_with_target_chain_ids_spatial_crop"
RECORD_GRANULARITY = "pdb_canonical_assembly_spatial_crop"


def parse_one_with_usage(path: str, config: dict) -> dict:
    result = base.parse_one(path, config)
    result["_worker_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--deferred-manifest", required=True)
    parser.add_argument("--reference-dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--cluster-file", required=True)
    parser.add_argument("--entries-index", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-resolution", type=float, default=3.5)
    parser.add_argument("--min-date", default="2021-08-03")
    parser.add_argument("--max-date", default="2026-07-08")
    parser.add_argument("--min-chain-length", type=int, default=30)
    parser.add_argument("--max-chain-length", type=int, default=10000)
    parser.add_argument("--max-context-length", type=int, default=2000)
    parser.add_argument("--min-context-crop-length", type=int, default=30)
    parser.add_argument("--min-resolved-residues", type=int, default=30)
    parser.add_argument("--min-backbone-coverage", type=float, default=0.5)
    parser.add_argument("--max-chains", type=int, default=len(base.CHAIN_IDS))
    parser.add_argument(
        "--method-allow", default="X-RAY DIFFRACTION,ELECTRON MICROSCOPY"
    )
    parser.add_argument("--max-shard-size", default="1g")
    parser.add_argument(
        "--worker-recycle-tasks",
        type=int,
        default=25,
        help="restart the single parser process after this many files",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_deferred_paths(path: Path, limit: int) -> tuple[list[Path], Counter]:
    rows = read_jsonl(path)
    stats = Counter(row.get("reason", "unknown") for row in rows)
    paths = []
    seen = set()
    for row in rows:
        if row.get("reason") not in {"context_too_long", "too_many_chains"}:
            raise ValueError(f"unexpected deferred reason: {row}")
        raw_path = Path(row["path"]).expanduser().resolve()
        if raw_path in seen:
            raise ValueError(f"duplicate deferred path: {raw_path}")
        if not raw_path.is_file():
            raise ValueError(f"deferred raw file not found: {raw_path}")
        seen.add(raw_path)
        paths.append(raw_path)
    if limit > 0:
        paths = paths[:limit]
    return paths, stats


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_cluster_ids(path: Path) -> set[int]:
    return {
        int(value)
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }


def split_name(cluster: int, valid: set[int], test: set[int]) -> str:
    if cluster in valid:
        return "valid"
    if cluster in test:
        return "test"
    return "train"


def inherit_reference_splits(rows: list[dict], reference_dataset: Path) -> dict:
    reference_rows = read_csv_rows(reference_dataset / "list.csv")
    reference_valid = read_cluster_ids(reference_dataset / "valid_clusters.txt")
    reference_test = read_cluster_ids(reference_dataset / "test_clusters.txt")
    if reference_valid & reference_test:
        raise ValueError("reference valid/test clusters overlap")

    combined_reference = []
    for row in reference_rows:
        copied = dict(row)
        copied["_reference_split"] = split_name(
            int(row["CLUSTER"]), reference_valid, reference_test
        )
        combined_reference.append(copied)
    combined = combined_reference + rows
    reconciliation = base.reconcile_exact_sequence_clusters(combined)

    reference_splits_by_cluster = defaultdict(set)
    for row in combined_reference:
        reference_splits_by_cluster[int(row["CLUSTER"])].add(row["_reference_split"])
    conflicts = {
        cluster: splits
        for cluster, splits in reference_splits_by_cluster.items()
        if len(splits) > 1
    }
    if conflicts:
        raise ValueError(f"reference split conflict after reconciliation: {conflicts}")
    inherited_split = {
        cluster: next(iter(splits))
        for cluster, splits in reference_splits_by_cluster.items()
    }

    stage_valid = {
        int(row["CLUSTER"])
        for row in rows
        if inherited_split.get(int(row["CLUSTER"]), "train") == "valid"
    }
    stage_test = {
        int(row["CLUSTER"])
        for row in rows
        if inherited_split.get(int(row["CLUSTER"]), "train") == "test"
    }
    sequence_splits = defaultdict(set)
    for row in combined:
        cluster = int(row["CLUSTER"])
        sequence_splits[row["SEQUENCE"]].add(inherited_split.get(cluster, "train"))
    leaks = [sequence for sequence, splits in sequence_splits.items() if len(splits) > 1]
    if leaks:
        raise ValueError(f"exact sequences cross inherited splits: {len(leaks)}")

    return {
        **reconciliation,
        "reference_rows": len(reference_rows),
        "reference_valid_clusters": len(reference_valid),
        "reference_test_clusters": len(reference_test),
        "stage_valid_cluster_ids": sorted(stage_valid),
        "stage_test_cluster_ids": sorted(stage_test),
        "stage_valid_clusters": len(stage_valid),
        "stage_test_clusters": len(stage_test),
        "stage_train_rows": sum(
            inherited_split.get(int(row["CLUSTER"]), "train") == "train"
            for row in rows
        ),
        "stage_valid_rows": sum(int(row["CLUSTER"]) in stage_valid for row in rows),
        "stage_test_rows": sum(int(row["CLUSTER"]) in stage_test for row in rows),
    }


def write_cluster_ids(path: Path, values: list[int]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit cannot be negative")
    if args.max_context_length <= 0:
        raise SystemExit("--max-context-length must be positive")
    if args.min_context_crop_length < args.min_chain_length:
        raise SystemExit("--min-context-crop-length cannot be below --min-chain-length")
    if args.max_chains < 1 or args.max_chains > len(base.CHAIN_IDS):
        raise SystemExit(f"--max-chains must be in [1, {len(base.CHAIN_IDS)}]")
    if args.worker_recycle_tasks < 1:
        raise SystemExit("--worker-recycle-tasks must be positive")
    if "fork" not in multiprocessing.get_all_start_methods():
        raise SystemExit("oversized crop building requires the Linux fork start method")

    deferred_manifest = Path(args.deferred_manifest).expanduser().resolve()
    reference_dataset = Path(args.reference_dataset).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not deferred_manifest.is_file():
        raise SystemExit(f"Deferred manifest not found: {deferred_manifest}")
    for filename in (
        "manifest.json",
        "validation.json",
        "list.csv",
        "valid_clusters.txt",
        "test_clusters.txt",
    ):
        if not (reference_dataset / filename).is_file():
            raise SystemExit(f"Reference dataset file not found: {reference_dataset / filename}")

    reference_manifest = json.loads(
        (reference_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    reference_validation = json.loads(
        (reference_dataset / "validation.json").read_text(encoding="utf-8")
    )
    reference_records = int(reference_manifest.get("record_count", 0))
    if reference_manifest.get("payload_schema") != "structure_with_target_chain_ids":
        raise SystemExit("Reference dataset is not the validated v1 payload schema")
    if reference_validation.get("status") != "ok":
        raise SystemExit("Reference dataset validation status is not ok")
    if int(reference_validation.get("payloads_checked", 0)) != reference_records:
        raise SystemExit("Reference validation does not cover every v1 payload")

    files, source_reason_counts = load_deferred_paths(deferred_manifest, args.limit)
    if not files:
        raise SystemExit("Deferred manifest selected no files")
    max_shard_size = tar_builder.parse_size(args.max_shard_size)
    if max_shard_size <= 0:
        raise SystemExit("--max-shard-size must be positive")
    method_allow = {
        value.strip().upper() for value in args.method_allow.split(",") if value.strip()
    }
    base.CLUSTER_MAP = base.load_cluster_map(args.cluster_file)
    base.ENTRY_METADATA = base.load_entry_metadata(args.entries_index)
    if not base.CLUSTER_MAP:
        raise SystemExit("Sequence cluster map is empty")
    if not base.ENTRY_METADATA:
        raise SystemExit("Entry metadata index is empty")

    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"Output dir exists, pass --force to replace: {out_dir}")
        shutil.rmtree(out_dir)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True)
    config = {
        "out_dir": str(out_dir),
        "max_resolution": args.max_resolution,
        "min_date": args.min_date,
        "max_date": args.max_date,
        "min_chain_length": args.min_chain_length,
        "max_chain_length": args.max_chain_length,
        "max_context_length": args.max_context_length,
        "min_context_crop_length": args.min_context_crop_length,
        "min_resolved_residues": args.min_resolved_residues,
        "min_backbone_coverage": args.min_backbone_coverage,
        "max_chains": args.max_chains,
        "method_allow": method_allow,
        "write_pt": False,
        "return_payload": True,
        "spatial_crop": True,
        "require_oversized": True,
    }

    print(f"deferred_manifest: {deferred_manifest}")
    print(f"reference_dataset: {reference_dataset}")
    print(f"out_dir: {out_dir}")
    print(f"files: {len(files)}")
    print("parser_workers: 1")
    print("max_in_flight: 1")
    print(f"worker_recycle_tasks: {args.worker_recycle_tasks}")
    print(f"max_context_length: {args.max_context_length}")
    print(f"crop_policy: {base.SPATIAL_CROP_POLICY}")

    started = time.time()
    stats = Counter()
    failures = []
    skipped = []
    rows = []
    shard_summaries = []
    record_count = 0
    context_chain_count = 0
    original_context_residues = 0
    retained_context_residues = 0
    worker_peak_rss_kib = 0
    shard_index = 0
    shard_payload_bytes = 0
    shard_record_count = 0
    chain_index_path = out_dir / "index.jsonl"
    record_index_path = out_dir / "records.jsonl"
    tar, shard_path, shard_name = tar_builder.open_shard(shards_dir, shard_index)

    def close_current_shard() -> None:
        nonlocal tar
        tar.close()
        shard_summaries.append(
            {
                "name": shard_name,
                "records": shard_record_count,
                "bytes": shard_path.stat().st_size,
                "sha256": tar_builder.sha256_file(shard_path),
            }
        )

    def consume_result(result: dict, record_index, chain_index) -> None:
        nonlocal tar, shard_index, shard_name, shard_path
        nonlocal shard_payload_bytes, shard_record_count
        nonlocal record_count, context_chain_count
        nonlocal original_context_residues, retained_context_residues
        nonlocal worker_peak_rss_kib
        worker_peak_rss_kib = max(
            worker_peak_rss_kib,
            int(result.pop("_worker_max_rss_kib", 0)),
        )
        status_key = result["status"] if result["status"] == "ok" else result.get("reason", "unknown")
        stats[status_key] += 1
        if result["status"] == "failed":
            failures.append(result)
            return
        if result["status"] != "ok":
            skipped.append(result)
            return

        payload = result.pop("payload")
        record_size = tar_builder.padded_tar_record_size(len(payload))
        if shard_record_count > 0 and shard_payload_bytes + record_size > max_shard_size:
            close_current_shard()
            shard_index += 1
            tar, shard_path, shard_name = tar_builder.open_shard(shards_dir, shard_index)
            shard_payload_bytes = 0
            shard_record_count = 0
        record_row, written_size = tar_builder.write_payload(
            tar, shard_name, result["entry_id"], payload
        )
        tar_builder.write_indexes(record_index, chain_index, record_row, result["rows"])
        rows.extend(result["rows"])
        record_count += 1
        context_chain_count += result["chains"]
        original_context_residues += result["crop"]["original_context_length"]
        retained_context_residues += result["crop"]["retained_context_length"]
        shard_record_count += 1
        shard_payload_bytes += written_size

    fork_context = multiprocessing.get_context("fork")
    with chain_index_path.open("w", encoding="utf-8") as chain_index, record_index_path.open(
        "w", encoding="utf-8"
    ) as record_index:
        for batch_start in range(0, len(files), args.worker_recycle_tasks):
            batch = files[batch_start : batch_start + args.worker_recycle_tasks]
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=fork_context,
            ) as executor:
                for batch_offset, path in enumerate(batch, start=1):
                    index = batch_start + batch_offset
                    result = executor.submit(parse_one_with_usage, str(path), config).result()
                    consume_result(result, record_index, chain_index)
                    if index % 100 == 0 or index == len(files):
                        print(
                            f"processed={index}/{len(files)} ok={stats['ok']} "
                            f"target_too_long={stats['target_too_long_for_complete_crop']} "
                            f"failures={len(failures)}",
                            flush=True,
                        )

    close_current_shard()
    if not rows:
        raise SystemExit("No spatial crop records were produced")

    split_stats = inherit_reference_splits(rows, reference_dataset)
    tar_builder.rewrite_index_clusters(chain_index_path, record_index_path, rows)
    rows.sort(key=lambda row: row["CHAINID"])
    with (out_dir / "list.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["CHAINID", "DEPOSITION", "RESOLUTION", "HASH", "CLUSTER", "SEQUENCE"],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_cluster_ids(
        out_dir / "valid_clusters.txt", split_stats.pop("stage_valid_cluster_ids")
    )
    write_cluster_ids(
        out_dir / "test_clusters.txt", split_stats.pop("stage_test_cluster_ids")
    )

    (out_dir / "README").write_text(
        f"{args.version_id}\nProteinMPNN oversized spatial-crop continuation dataset.\n",
        encoding="utf-8",
    )
    if failures:
        with (out_dir / "build_failures.jsonl").open("w", encoding="utf-8") as handle:
            for result in failures:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
    if skipped:
        with (out_dir / "build_skipped.jsonl").open("w", encoding="utf-8") as handle:
            for result in skipped:
                handle.write(json.dumps(result, sort_keys=True) + "\n")

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
            "worker_recycle_tasks": args.worker_recycle_tasks,
            "start_method": "fork",
        },
        "resources": {
            "worker_peak_rss_kib": worker_peak_rss_kib,
            "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "crop_policy": base.SPATIAL_CROP_POLICY,
        "split_policy": "inherit_v1_reference_clusters_stage2_only_clusters_train",
        "filters": {
            "max_resolution": args.max_resolution,
            "min_date": args.min_date,
            "max_date": args.max_date,
            "min_chain_length": args.min_chain_length,
            "max_chain_length": args.max_chain_length,
            "max_context_length": args.max_context_length,
            "min_context_crop_length": args.min_context_crop_length,
            "min_resolved_residues": args.min_resolved_residues,
            "min_backbone_coverage": args.min_backbone_coverage,
            "max_chains": args.max_chains,
            "method_allow": sorted(method_allow),
        },
        "counts": {
            "source_manifest_rows": sum(source_reason_counts.values()),
            "input_files": len(files),
            "records": record_count,
            "context_chains": context_chain_count,
            "original_context_residues": original_context_residues,
            "retained_context_residues": retained_context_residues,
            "failures": len(failures),
            **dict(stats),
            **split_stats,
        },
        "source_reason_counts": dict(source_reason_counts),
        "skip_reasons": {key: value for key, value in stats.items() if key != "ok"},
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "format": base.TAR_SHARD_FORMAT,
        "version_id": args.version_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset": str(deferred_manifest),
        "reference_dataset": str(reference_dataset),
        "record_granularity": RECORD_GRANULARITY,
        "index_granularity": "target_chain",
        "payload_schema": PAYLOAD_SCHEMA,
        "target_selection_policy": "max_resolved_backbone_then_coverage_then_source_id",
        "crop_policy": base.SPATIAL_CROP_POLICY,
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
        },
        "build": build_manifest,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(build_manifest["counts"], indent=2, sort_keys=True))
    print(json.dumps(build_manifest["resources"], indent=2, sort_keys=True))
    if failures:
        raise SystemExit(f"Spatial crop build had {len(failures)} parser failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
