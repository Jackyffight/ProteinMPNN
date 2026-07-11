#!/usr/bin/env python3
"""Build a ProteinMPNN tar-shard dataset directly from wwPDB assembly mmCIF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import tarfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import build_pdb_mmcif_dataset as base


FORMAT = base.TAR_SHARD_FORMAT


def parse_size(value: str) -> int:
    value = value.strip().lower()
    units = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda item: -len(item[0])):
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * multiplier)
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--cluster-file", default="")
    parser.add_argument("--entries-index", default="")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-in-flight", type=int, default=0)
    parser.add_argument("--assembly-id", default="all")
    parser.add_argument(
        "--assembly-policy",
        choices=["all", "first"],
        default="first",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-resolution", type=float, default=3.5)
    parser.add_argument("--min-date", default="")
    parser.add_argument("--max-date", default="2026-07-08")
    parser.add_argument("--min-chain-length", type=int, default=30)
    parser.add_argument("--max-chain-length", type=int, default=10000)
    parser.add_argument("--max-context-length", type=int, default=2000)
    parser.add_argument("--min-resolved-residues", type=int, default=30)
    parser.add_argument("--min-backbone-coverage", type=float, default=0.5)
    parser.add_argument("--max-chains", type=int, default=len(base.CHAIN_IDS))
    parser.add_argument(
        "--method-allow",
        default="X-RAY DIFFRACTION,ELECTRON MICROSCOPY",
        help="Comma-separated experimental methods. Empty string allows all.",
    )
    parser.add_argument("--valid-frac", type=float, default=0.01)
    parser.add_argument("--test-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-shard-size", default="1g")
    parser.add_argument(
        "--max-raw-file-size",
        default="50m",
        help="Skip raw mmCIF files larger than this compressed size. Use 0 to disable.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_shard(shards_dir: Path, shard_index: int) -> tuple[tarfile.TarFile, Path, str]:
    shard_name = f"shard_{shard_index:06d}.tar"
    shard_path = shards_dir / shard_name
    tar = tarfile.open(shard_path, "w", format=tarfile.GNU_FORMAT)
    return tar, shard_path, shard_name


def padded_tar_record_size(size: int) -> int:
    return 512 + int(math.ceil(size / 512.0) * 512)


def write_payload(
    tar: tarfile.TarFile,
    shard_name: str,
    entry_id: str,
    payload: bytes,
) -> tuple[dict, int]:
    member = f"{entry_id}.pt"
    header_offset = tar.fileobj.tell()
    info = tarfile.TarInfo(member)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(payload))
    data_offset = header_offset + 512
    row = {
        "entry_id": entry_id,
        "shard": f"shards/{shard_name}",
        "member": member,
        "offset": data_offset,
        "size": len(payload),
    }
    return row, padded_tar_record_size(len(payload))


def write_indexes(record_index, chain_index, record_row: dict, rows: list[dict]) -> None:
    record_index_row = {
        **record_row,
        "chains": [row["CHAINID"] for row in rows],
        "clusters": sorted({int(row["CLUSTER"]) for row in rows}),
        "date": rows[0]["DEPOSITION"],
        "resolution": float(rows[0]["RESOLUTION"]),
    }
    record_index.write(json.dumps(record_index_row, sort_keys=True) + "\n")
    for row in rows:
        chain_index.write(
            json.dumps(
                {
                    "chain_id": row["CHAINID"],
                    "entry_id": record_row["entry_id"],
                    "cluster": int(row["CLUSTER"]),
                    "hash": row["HASH"],
                    "sequence_length": len(row["SEQUENCE"]),
                    "date": row["DEPOSITION"],
                    "resolution": float(row["RESOLUTION"]),
                    "shard": record_row["shard"],
                    "member": record_row["member"],
                    "offset": record_row["offset"],
                    "size": record_row["size"],
                },
                sort_keys=True,
            )
            + "\n"
        )


def rewrite_index_clusters(
    chain_index_path: Path,
    record_index_path: Path,
    rows: list[dict],
) -> None:
    cluster_by_chain = {row["CHAINID"]: int(row["CLUSTER"]) for row in rows}

    chain_tmp = chain_index_path.with_suffix(".jsonl.tmp")
    with chain_index_path.open("r", encoding="utf-8") as source, chain_tmp.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            index_row = json.loads(line)
            index_row["cluster"] = cluster_by_chain[index_row["chain_id"]]
            destination.write(json.dumps(index_row, sort_keys=True) + "\n")
    chain_tmp.replace(chain_index_path)

    record_tmp = record_index_path.with_suffix(".jsonl.tmp")
    with record_index_path.open("r", encoding="utf-8") as source, record_tmp.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            index_row = json.loads(line)
            index_row["clusters"] = sorted(
                {cluster_by_chain[chain_id] for chain_id in index_row["chains"]}
            )
            destination.write(json.dumps(index_row, sort_keys=True) + "\n")
    record_tmp.replace(record_index_path)


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "shards"
    max_shard_size = parse_size(args.max_shard_size)
    max_raw_file_size = parse_size(args.max_raw_file_size)
    if max_shard_size <= 0:
        raise SystemExit("--max-shard-size must be positive")
    if max_raw_file_size < 0:
        raise SystemExit("--max-raw-file-size cannot be negative")

    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"Output dir exists, pass --force to replace: {out_dir}")
        shutil.rmtree(out_dir)
    shards_dir.mkdir(parents=True)

    base.CLUSTER_MAP = base.load_cluster_map(args.cluster_file)
    base.ENTRY_METADATA = base.load_entry_metadata(args.entries_index)

    discovered_files = base.discover_files(
        raw_dir, args.assembly_id, 0, assembly_policy="all"
    )
    discovered_file_count = len(discovered_files)
    canonical_files = base.discover_files(
        raw_dir,
        args.assembly_id,
        0,
        assembly_policy=args.assembly_policy,
    )
    canonical_file_count = len(canonical_files)
    files = canonical_files[: args.limit] if args.limit > 0 else canonical_files
    skipped_oversized_files = []
    if max_raw_file_size > 0:
        kept_files = []
        for path in files:
            size = path.stat().st_size
            if size > max_raw_file_size:
                skipped_oversized_files.append({"path": str(path), "bytes": size})
            else:
                kept_files.append(path)
        files = kept_files
    if not files:
        raise SystemExit(f"No mmCIF files found under {raw_dir}")

    method_allow = {item.strip().upper() for item in args.method_allow.split(",") if item.strip()}
    config = {
        "out_dir": str(out_dir),
        "max_resolution": args.max_resolution,
        "min_date": args.min_date,
        "max_date": args.max_date,
        "min_chain_length": args.min_chain_length,
        "max_chain_length": args.max_chain_length,
        "max_context_length": args.max_context_length,
        "min_resolved_residues": args.min_resolved_residues,
        "min_backbone_coverage": args.min_backbone_coverage,
        "max_chains": args.max_chains,
        "method_allow": method_allow,
        "write_pt": False,
        "return_payload": True,
    }

    print(f"raw_dir: {raw_dir}")
    print(f"out_dir: {out_dir}")
    print(f"discovered_files: {discovered_file_count}")
    print(f"canonical_files: {canonical_file_count}")
    print(f"files: {len(files)}")
    print(f"max_raw_file_size: {max_raw_file_size}")
    print(f"skipped_oversized_files: {len(skipped_oversized_files)}")
    print(f"cluster_map_entries: {len(base.CLUSTER_MAP)}")
    print(f"entry_metadata_records: {len(base.ENTRY_METADATA)}")
    print(f"max_shard_size: {max_shard_size}")

    started = time.time()
    stats = Counter()
    failures = []
    deferred_oversized = []
    rows: list[dict] = []
    shard_summaries = []
    record_count = 0
    target_count = 0
    context_chain_count = 0
    shard_index = 0
    shard_payload_bytes = 0
    shard_record_count = 0
    processed = 0

    chain_index_path = out_dir / "index.jsonl"
    record_index_path = out_dir / "records.jsonl"
    tar, shard_path, shard_name = open_shard(shards_dir, shard_index)

    def close_current_shard() -> None:
        nonlocal tar
        tar.close()
        shard_summaries.append(
            {
                "name": shard_name,
                "records": shard_record_count,
                "bytes": shard_path.stat().st_size,
                "sha256": sha256_file(shard_path),
            }
        )

    max_in_flight = args.max_in_flight if args.max_in_flight > 0 else args.workers
    max_in_flight = max(args.workers, max_in_flight)

    def consume_result(result: dict) -> None:
        nonlocal tar
        nonlocal shard_index, shard_name, shard_path
        nonlocal shard_payload_bytes, shard_record_count
        nonlocal processed, record_count, target_count, context_chain_count

        processed += 1
        status_key = (
            result["status"]
            if result["status"] == "ok"
            else result.get("reason", "unknown")
        )
        stats[status_key] += 1
        if result["status"] == "ok":
            payload = result.pop("payload")
            record_size = padded_tar_record_size(len(payload))
            if (
                shard_record_count > 0
                and shard_payload_bytes + record_size > max_shard_size
            ):
                close_current_shard()
                shard_index += 1
                tar, shard_path, shard_name = open_shard(shards_dir, shard_index)
                shard_payload_bytes = 0
                shard_record_count = 0
            record_row, written_size = write_payload(
                tar, shard_name, result["entry_id"], payload
            )
            write_indexes(record_index, chain_index, record_row, result["rows"])
            rows.extend(result["rows"])
            record_count += 1
            target_count += len(result["rows"])
            context_chain_count += result["chains"]
            shard_record_count += 1
            shard_payload_bytes += written_size
        elif result["status"] == "failed":
            failures.append(result)
        elif result.get("reason") in {"context_too_long", "too_many_chains"}:
            deferred_oversized.append(result)
        if processed % 1000 == 0:
            print(
                f"processed={processed}/{len(files)} ok_entries={stats['ok']} "
                f"rows={len(rows)} shards={shard_index + 1}",
                flush=True,
            )

    with chain_index_path.open("w", encoding="utf-8") as chain_index, record_index_path.open(
        "w", encoding="utf-8"
    ) as record_index:
        if args.workers <= 1:
            iterator = (base.parse_one(str(path), config) for path in files)
            for result in iterator:
                consume_result(result)
        else:
            file_iter = iter(enumerate(files))
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures: dict = {}
                ready: dict[int, dict] = {}
                next_to_write = 0

                def submit_until_full() -> None:
                    while len(futures) + len(ready) < max_in_flight:
                        try:
                            file_index, path = next(file_iter)
                        except StopIteration:
                            break
                        future = executor.submit(base.parse_one, str(path), config)
                        futures[future] = file_index

                submit_until_full()
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        file_index = futures.pop(future)
                        ready[file_index] = future.result()
                    while next_to_write in ready:
                        consume_result(ready.pop(next_to_write))
                        next_to_write += 1
                    submit_until_full()
                if ready:
                    raise RuntimeError("internal ordering error: completed results were not written")

    close_current_shard()

    reconciliation_stats = base.reconcile_exact_sequence_clusters(rows)
    rewrite_index_clusters(chain_index_path, record_index_path, rows)
    rows.sort(key=lambda row: row["CHAINID"])
    with (out_dir / "list.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["CHAINID", "DEPOSITION", "RESOLUTION", "HASH", "CLUSTER", "SEQUENCE"],
        )
        writer.writeheader()
        writer.writerows(rows)

    split_stats = base.write_splits(rows, out_dir, args.valid_frac, args.test_frac, args.seed)
    with (out_dir / "README").open("w", encoding="utf-8") as handle:
        handle.write(f"{args.version_id}\n")
        handle.write("ProteinMPNN tar-shard dataset built directly from wwPDB biological assembly mmCIF files.\n")

    if failures:
        with (out_dir / "build_failures.jsonl").open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, sort_keys=True) + "\n")
    if skipped_oversized_files:
        with (out_dir / "build_skipped_oversized.jsonl").open("w", encoding="utf-8") as handle:
            for skipped in skipped_oversized_files:
                handle.write(json.dumps(skipped, sort_keys=True) + "\n")
    if deferred_oversized:
        with (out_dir / "build_deferred_oversized.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for deferred in deferred_oversized:
                handle.write(json.dumps(deferred, sort_keys=True) + "\n")

    skip_reasons = dict(stats)
    if skipped_oversized_files:
        skip_reasons["raw_file_size"] = len(skipped_oversized_files)
    build_manifest = {
        "version_id": args.version_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "assembly_id": args.assembly_id,
        "assembly_policy": args.assembly_policy,
        "cluster_file": args.cluster_file,
        "entries_index": args.entries_index,
        "cluster_map_entries": len(base.CLUSTER_MAP),
        "entry_metadata_records": len(base.ENTRY_METADATA),
        "concurrency": {
            "workers": args.workers,
            "max_in_flight": max_in_flight,
        },
        "filters": {
            "max_resolution": args.max_resolution,
            "min_date": args.min_date,
            "max_date": args.max_date,
            "min_chain_length": args.min_chain_length,
            "max_chain_length": args.max_chain_length,
            "max_context_length": args.max_context_length,
            "min_resolved_residues": args.min_resolved_residues,
            "min_backbone_coverage": args.min_backbone_coverage,
            "max_chains": args.max_chains,
            "max_raw_file_size": max_raw_file_size,
            "method_allow": sorted(method_allow),
        },
        "counts": {
            "discovered_files": discovered_file_count,
            "canonical_files": canonical_file_count,
            "alternative_assemblies_skipped": discovered_file_count - canonical_file_count,
            "input_files": len(files),
            "skipped_oversized_files": len(skipped_oversized_files),
            "list_rows": len(rows),
            "ok_entries": stats["ok"],
            "failures": len(failures),
            "deferred_oversized": len(deferred_oversized),
            "context_chains": context_chain_count,
            **reconciliation_stats,
            **split_stats,
        },
        "skip_reasons": skip_reasons,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with (out_dir / "build_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(build_manifest, handle, indent=2, sort_keys=True)

    manifest = {
        "format": FORMAT,
        "version_id": args.version_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset": str(raw_dir),
        "record_granularity": "pdb_canonical_assembly",
        "index_granularity": "target_chain",
        "payload_schema": "structure_with_target_chain_ids",
        "target_selection_policy": "max_resolved_backbone_then_coverage_then_source_id",
        "record_count": record_count,
        "target_count": target_count,
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
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(build_manifest["counts"], indent=2, sort_keys=True))
    if not rows:
        raise SystemExit("No training rows were produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
