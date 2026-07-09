#!/usr/bin/env python3
"""Pack a ProteinMPNN-compatible .pt dataset into random-access tar shards."""

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
from collections import defaultdict
from pathlib import Path

import torch


FORMAT = "proteinmpnn.tar_shard.v1"


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
    parser.add_argument("--input-dir", required=True, help="ProteinMPNN dataset dir with list.csv and pdb/.")
    parser.add_argument("--output-dir", required=True, help="Output tar-shard dataset directory.")
    parser.add_argument("--version-id", default="")
    parser.add_argument("--max-shard-size", default="1g")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_list_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["CLUSTER"] = int(row["CLUSTER"])
        row["RESOLUTION"] = float(row["RESOLUTION"])
        row["SEQ_LEN"] = len(row["SEQUENCE"])
        row["ENTRY_ID"], row["CHAIN_ID_SHORT"] = row["CHAINID"].rsplit("_", 1)
    return rows


def load_entry_payload(input_dir: Path, entry_id: str, rows: list[dict]) -> bytes:
    prefix = input_dir / "pdb" / entry_id[1:3] / entry_id
    meta = torch.load(f"{prefix}.pt", map_location="cpu")
    chains = {}
    for chain_id in meta["chains"]:
        chains[chain_id] = torch.load(f"{prefix}_{chain_id}.pt", map_location="cpu")
    payload = {
        "format": FORMAT,
        "entry_id": entry_id,
        "rows": rows,
        "meta": meta,
        "chains": chains,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


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


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "shards"
    max_shard_size = parse_size(args.max_shard_size)

    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output dir exists, pass --force to replace: {output_dir}")
        shutil.rmtree(output_dir)
    shards_dir.mkdir(parents=True)

    rows = read_list_csv(input_dir / "list.csv")
    rows_by_entry: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_entry[row["ENTRY_ID"]].append(row)

    shutil.copy2(input_dir / "list.csv", output_dir / "list.csv")
    shutil.copy2(input_dir / "valid_clusters.txt", output_dir / "valid_clusters.txt")
    shutil.copy2(input_dir / "test_clusters.txt", output_dir / "test_clusters.txt")

    chain_index_path = output_dir / "index.jsonl"
    record_index_path = output_dir / "records.jsonl"
    tar, shard_path, shard_name = open_shard(shards_dir, 0)
    shard_index = 0
    shard_payload_bytes = 0
    shard_record_count = 0
    shard_summaries = []
    record_count = 0
    chain_count = 0

    with chain_index_path.open("w", encoding="utf-8") as chain_index, record_index_path.open(
        "w", encoding="utf-8"
    ) as record_index:
        for entry_id in sorted(rows_by_entry):
            entry_rows = rows_by_entry[entry_id]
            payload = load_entry_payload(input_dir, entry_id, entry_rows)
            record_size = padded_tar_record_size(len(payload))
            if shard_record_count > 0 and shard_payload_bytes + record_size > max_shard_size:
                tar.close()
                shard_summaries.append(
                    {
                        "name": shard_name,
                        "records": shard_record_count,
                        "bytes": shard_path.stat().st_size,
                        "sha256": sha256_file(shard_path),
                    }
                )
                shard_index += 1
                tar, shard_path, shard_name = open_shard(shards_dir, shard_index)
                shard_payload_bytes = 0
                shard_record_count = 0

            member = f"{entry_id}.pt"
            header_offset = tar.fileobj.tell()
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))
            data_offset = header_offset + 512

            record_index_row = {
                "entry_id": entry_id,
                "shard": f"shards/{shard_name}",
                "member": member,
                "offset": data_offset,
                "size": len(payload),
                "chains": [row["CHAINID"] for row in entry_rows],
                "clusters": sorted({row["CLUSTER"] for row in entry_rows}),
                "date": entry_rows[0]["DEPOSITION"],
                "resolution": entry_rows[0]["RESOLUTION"],
            }
            record_index.write(json.dumps(record_index_row, sort_keys=True) + "\n")
            for row in entry_rows:
                chain_index.write(
                    json.dumps(
                        {
                            "chain_id": row["CHAINID"],
                            "entry_id": entry_id,
                            "cluster": row["CLUSTER"],
                            "hash": row["HASH"],
                            "sequence_length": row["SEQ_LEN"],
                            "date": row["DEPOSITION"],
                            "resolution": row["RESOLUTION"],
                            "shard": f"shards/{shard_name}",
                            "member": member,
                            "offset": data_offset,
                            "size": len(payload),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            record_count += 1
            chain_count += len(entry_rows)
            shard_record_count += 1
            shard_payload_bytes += record_size

    tar.close()
    shard_summaries.append(
        {
            "name": shard_name,
            "records": shard_record_count,
            "bytes": shard_path.stat().st_size,
            "sha256": sha256_file(shard_path),
        }
    )

    manifest = {
        "format": FORMAT,
        "version_id": args.version_id or input_dir.name,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset": str(input_dir),
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
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps({"output_dir": str(output_dir), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
