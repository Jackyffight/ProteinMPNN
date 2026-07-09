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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-resolution", type=float, default=3.5)
    parser.add_argument("--min-date", default="")
    parser.add_argument("--max-date", default="2026-07-08")
    parser.add_argument("--min-chain-length", type=int, default=30)
    parser.add_argument("--max-chain-length", type=int, default=10000)
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


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "shards"
    max_shard_size = parse_size(args.max_shard_size)

    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"Output dir exists, pass --force to replace: {out_dir}")
        shutil.rmtree(out_dir)
    shards_dir.mkdir(parents=True)

    base.CLUSTER_MAP = base.load_cluster_map(args.cluster_file)
    base.ENTRY_METADATA = base.load_entry_metadata(args.entries_index)

    files = base.discover_files(raw_dir, args.assembly_id, args.limit)
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
        "max_chains": args.max_chains,
        "method_allow": method_allow,
        "write_pt": False,
        "return_payload": True,
    }

    print(f"raw_dir: {raw_dir}")
    print(f"out_dir: {out_dir}")
    print(f"files: {len(files)}")
    print(f"cluster_map_entries: {len(base.CLUSTER_MAP)}")
    print(f"entry_metadata_records: {len(base.ENTRY_METADATA)}")
    print(f"max_shard_size: {max_shard_size}")

    started = time.time()
    stats = Counter()
    failures = []
    rows: list[dict] = []
    shard_summaries = []
    record_count = 0
    chain_count = 0
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

    def submit_until_full(executor: ProcessPoolExecutor, futures: dict, file_iter) -> None:
        max_in_flight = args.max_in_flight if args.max_in_flight > 0 else args.workers * 2
        max_in_flight = max(args.workers, max_in_flight)
        while len(futures) < max_in_flight:
            try:
                path = next(file_iter)
            except StopIteration:
                break
            futures[executor.submit(base.parse_one, str(path), config)] = path

    with chain_index_path.open("w", encoding="utf-8") as chain_index, record_index_path.open(
        "w", encoding="utf-8"
    ) as record_index:
        if args.workers <= 1:
            iterator = (base.parse_one(str(path), config) for path in files)
            for result in iterator:
                processed += 1
                status_key = result["status"] if result["status"] == "ok" else result.get("reason", "unknown")
                stats[status_key] += 1
                if result["status"] == "ok":
                    payload = result.pop("payload")
                    record_size = padded_tar_record_size(len(payload))
                    if shard_record_count > 0 and shard_payload_bytes + record_size > max_shard_size:
                        close_current_shard()
                        shard_index += 1
                        tar, shard_path, shard_name = open_shard(shards_dir, shard_index)
                        shard_payload_bytes = 0
                        shard_record_count = 0
                    record_row, written_size = write_payload(tar, shard_name, result["entry_id"], payload)
                    write_indexes(record_index, chain_index, record_row, result["rows"])
                    rows.extend(result["rows"])
                    record_count += 1
                    chain_count += len(result["rows"])
                    shard_record_count += 1
                    shard_payload_bytes += written_size
                elif result["status"] == "failed":
                    failures.append(result)
                if processed % 1000 == 0:
                    print(
                        f"processed={processed}/{len(files)} ok_entries={stats['ok']} "
                        f"rows={len(rows)} shards={shard_index + 1}",
                        flush=True,
                    )
        else:
            file_iter = iter(files)
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures: dict = {}
                submit_until_full(executor, futures, file_iter)
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.pop(future)
                        result = future.result()
                        processed += 1
                        status_key = result["status"] if result["status"] == "ok" else result.get("reason", "unknown")
                        stats[status_key] += 1
                        if result["status"] == "ok":
                            payload = result.pop("payload")
                            record_size = padded_tar_record_size(len(payload))
                            if shard_record_count > 0 and shard_payload_bytes + record_size > max_shard_size:
                                close_current_shard()
                                shard_index += 1
                                tar, shard_path, shard_name = open_shard(shards_dir, shard_index)
                                shard_payload_bytes = 0
                                shard_record_count = 0
                            record_row, written_size = write_payload(tar, shard_name, result["entry_id"], payload)
                            write_indexes(record_index, chain_index, record_row, result["rows"])
                            rows.extend(result["rows"])
                            record_count += 1
                            chain_count += len(result["rows"])
                            shard_record_count += 1
                            shard_payload_bytes += written_size
                        elif result["status"] == "failed":
                            failures.append(result)
                        if processed % 1000 == 0:
                            print(
                                f"processed={processed}/{len(files)} ok_entries={stats['ok']} "
                                f"rows={len(rows)} shards={shard_index + 1}",
                                flush=True,
                            )
                    submit_until_full(executor, futures, file_iter)

    close_current_shard()

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

    build_manifest = {
        "version_id": args.version_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "assembly_id": args.assembly_id,
        "cluster_file": args.cluster_file,
        "entries_index": args.entries_index,
        "cluster_map_entries": len(base.CLUSTER_MAP),
        "entry_metadata_records": len(base.ENTRY_METADATA),
        "filters": {
            "max_resolution": args.max_resolution,
            "min_date": args.min_date,
            "max_date": args.max_date,
            "min_chain_length": args.min_chain_length,
            "max_chain_length": args.max_chain_length,
            "max_chains": args.max_chains,
            "method_allow": sorted(method_allow),
        },
        "counts": {
            "input_files": len(files),
            "list_rows": len(rows),
            "ok_entries": stats["ok"],
            "failures": len(failures),
            **split_stats,
        },
        "skip_reasons": dict(stats),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with (out_dir / "build_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(build_manifest, handle, indent=2, sort_keys=True)

    manifest = {
        "format": FORMAT,
        "version_id": args.version_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset": str(raw_dir),
        "record_granularity": "assembly",
        "index_granularity": "chain",
        "record_count": record_count,
        "chain_count": chain_count,
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
