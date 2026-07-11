#!/usr/bin/env python3
"""Validate ProteinMPNN v2 tar shards and their training semantics."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

from tar_shard_utils import TarShardStore


ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")
LOCATION_FIELDS = ("shard", "member", "offset", "size")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="limit payload reads; metadata, checksums, and split checks still cover all rows",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def split_name(cluster, valid_clusters, test_clusters):
    if cluster in valid_clusters:
        return "valid"
    if cluster in test_clusters:
        return "test"
    return "train"


def read_cluster_ids(path):
    return {
        int(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shards(dataset_dir, manifest, record_index_rows):
    summaries = manifest.get("shards", [])
    require(summaries, "manifest contains no shards")
    summary_by_path = {}
    for summary in summaries:
        name = summary["name"]
        require(Path(name).name == name, f"invalid shard name: {name}")
        relative_path = f"shards/{name}"
        require(relative_path not in summary_by_path, f"duplicate shard: {name}")
        summary_by_path[relative_path] = summary

    actual_paths = {
        str(path.relative_to(dataset_dir))
        for path in (dataset_dir / "shards").glob("*.tar")
    }
    require(actual_paths == set(summary_by_path), "manifest/shard file set mismatch")

    records_per_shard = Counter(row["shard"] for row in record_index_rows)
    require(
        set(records_per_shard).issubset(summary_by_path),
        "record index references a shard outside the manifest",
    )
    for relative_path, summary in sorted(summary_by_path.items()):
        shard_path = dataset_dir / relative_path
        require(shard_path.is_file(), f"missing shard: {relative_path}")
        require(
            shard_path.stat().st_size == int(summary["bytes"]),
            f"shard size mismatch: {relative_path}",
        )
        require(
            records_per_shard[relative_path] == int(summary["records"]),
            f"shard record count mismatch: {relative_path}",
        )
        require(
            sha256_file(shard_path) == summary["sha256"],
            f"shard checksum mismatch: {relative_path}",
        )
    return len(summary_by_path)


def validate_indexes(dataset_dir, manifest, rows, chain_index_rows, record_index_rows):
    require(len(rows) == len(chain_index_rows), "list/index row count mismatch")
    require(len(rows) == len(record_index_rows), "list/record row count mismatch")
    require(len(rows) == manifest["record_count"], "manifest record_count mismatch")
    require(len(rows) == manifest["target_count"], "manifest target_count mismatch")

    row_by_chain = {row["CHAINID"]: row for row in rows}
    require(len(row_by_chain) == len(rows), "duplicate target chain IDs")
    pdb_ids = [row["CHAINID"][:4].lower() for row in rows]
    require(len(set(pdb_ids)) == len(pdb_ids), "more than one target row per PDB")

    chain_index_by_id = {row["chain_id"]: row for row in chain_index_rows}
    require(len(chain_index_by_id) == len(rows), "duplicate chain index IDs")
    record_index_by_entry = {row["entry_id"]: row for row in record_index_rows}
    require(len(record_index_by_entry) == len(record_index_rows), "duplicate record IDs")

    for chain_id, list_row in row_by_chain.items():
        require(chain_id in chain_index_by_id, f"target missing from index: {chain_id}")
        chain_row = chain_index_by_id[chain_id]
        entry_id = chain_id.split("_", 1)[0]
        require(chain_row["entry_id"] == entry_id, f"chain entry mismatch: {chain_id}")
        require(entry_id in record_index_by_entry, f"record missing from index: {entry_id}")
        record_row = record_index_by_entry[entry_id]
        require(record_row["chains"] == [chain_id], f"record target mismatch: {entry_id}")
        require(
            record_row["clusters"] == [int(list_row["CLUSTER"])],
            f"record cluster mismatch: {entry_id}",
        )
        require(record_row["date"] == list_row["DEPOSITION"], f"record date mismatch: {entry_id}")
        require(
            math.isclose(float(record_row["resolution"]), float(list_row["RESOLUTION"])),
            f"record resolution mismatch: {entry_id}",
        )
        require(
            int(list_row["CLUSTER"]) == chain_row["cluster"],
            f"chain cluster mismatch: {chain_id}",
        )
        require(chain_row["hash"] == list_row["HASH"], f"hash mismatch: {chain_id}")
        require(
            chain_row["sequence_length"] == len(list_row["SEQUENCE"]),
            f"sequence length mismatch: {chain_id}",
        )
        for field in LOCATION_FIELDS:
            require(
                chain_row[field] == record_row[field],
                f"chain/record {field} mismatch: {chain_id}",
            )
        shard_path = dataset_dir / chain_row["shard"]
        require(shard_path.is_file(), f"indexed shard missing: {chain_row['shard']}")
        require(chain_row["offset"] >= 512, f"invalid payload offset: {chain_id}")
        require(chain_row["size"] > 0, f"invalid payload size: {chain_id}")
        require(
            chain_row["offset"] + chain_row["size"] <= shard_path.stat().st_size,
            f"payload exceeds shard bounds: {chain_id}",
        )
        require(
            chain_row["member"] == f"{entry_id}.pt",
            f"member name mismatch: {chain_id}",
        )
    return row_by_chain, chain_index_by_id


def validate_payload(chain_id, row, payload, build_filters):
    require(payload.get("format") == "proteinmpnn.tar_shard.v2", "payload is not v2")
    entry_id = chain_id.split("_", 1)[0]
    require(payload["entry_id"] == entry_id, f"entry mismatch: {chain_id}")
    require(
        payload.get("target_chain_ids") == [chain_id],
        f"payload target list mismatch: {chain_id}",
    )

    meta = payload["meta"]
    chains = payload["chains"]
    require(meta["chains"] == list(chains), f"meta/payload chain order mismatch: {chain_id}")
    require(set(meta["source_chain_map"]) == set(chains), f"source chain map mismatch: {chain_id}")
    target_chain = meta["target_chain"]
    require(f"{entry_id}_{target_chain}" == chain_id, f"meta target mismatch: {chain_id}")
    require(target_chain in chains, f"target payload missing: {chain_id}")
    require(chains[target_chain]["seq"] == row["SEQUENCE"], f"target sequence mismatch: {chain_id}")
    require(meta["source_pdb_id"].lower() == chain_id[:4].lower(), f"source PDB mismatch: {chain_id}")

    context_length = 0
    retained_missing_positions = 0
    for context_chain_id, chain in chains.items():
        sequence = chain["seq"]
        require(
            sequence and set(sequence).issubset(ALPHABET),
            f"bad sequence alphabet: {chain_id}/{context_chain_id}",
        )
        length = len(sequence)
        require(
            int(build_filters["min_chain_length"])
            <= length
            <= int(build_filters["max_chain_length"]),
            f"chain length filter violation: {chain_id}/{context_chain_id}",
        )
        context_length += length
        require(
            tuple(chain["xyz"].shape) == (length, 14, 3),
            f"xyz shape mismatch: {chain_id}/{context_chain_id}",
        )
        for tensor_name in ("mask", "bfac", "occ"):
            require(
                tuple(chain[tensor_name].shape) == (length, 14),
                f"{tensor_name} shape mismatch: {chain_id}/{context_chain_id}",
            )
        require(chain["mask"].dtype == torch.bool, f"mask dtype mismatch: {chain_id}/{context_chain_id}")
        finite_components = torch.isfinite(chain["xyz"])
        expected_finite = chain["mask"].unsqueeze(-1).expand_as(finite_components)
        require(
            torch.equal(finite_components, expected_finite),
            f"coordinate/mask mismatch: {chain_id}/{context_chain_id}",
        )
        require(torch.isfinite(chain["bfac"]).all(), f"non-finite B factors: {chain_id}/{context_chain_id}")
        require(torch.isfinite(chain["occ"]).all(), f"non-finite occupancies: {chain_id}/{context_chain_id}")

        complete_backbone = chain["mask"][:, :4].all(dim=1)
        resolved_count = int(complete_backbone.sum().item())
        require(
            resolved_count == int(chain["resolved_residue_count"]),
            f"resolved residue count mismatch: {chain_id}/{context_chain_id}",
        )
        coverage = resolved_count / float(length)
        require(
            math.isclose(coverage, float(chain["backbone_coverage"]), abs_tol=1e-9),
            f"backbone coverage mismatch: {chain_id}/{context_chain_id}",
        )
        require(
            resolved_count >= int(build_filters["min_resolved_residues"]),
            f"resolved residue filter violation: {chain_id}/{context_chain_id}",
        )
        require(
            coverage >= float(build_filters["min_backbone_coverage"]),
            f"coverage filter violation: {chain_id}/{context_chain_id}",
        )
        require(
            meta["source_chain_map"][context_chain_id] == chain["source_chain_id"],
            f"source chain mapping mismatch: {chain_id}/{context_chain_id}",
        )
        retained_missing_positions += int((~complete_backbone).sum().item())

    require(
        context_length <= int(build_filters["max_context_length"]),
        f"context exceeds limit: {chain_id}",
    )
    require(len(chains) <= int(build_filters["max_chains"]), f"chain count exceeds limit: {chain_id}")
    tm = meta["tm"]
    chain_count = len(chains)
    require(tuple(tm.shape) == (chain_count, chain_count, 3), f"tm shape mismatch: {chain_id}")
    require(torch.isfinite(tm).all(), f"non-finite tm values: {chain_id}")
    require(bool(((tm >= 0.0) & (tm <= 1.0)).all()), f"tm values outside [0, 1]: {chain_id}")
    target_index = meta["chains"].index(target_chain)
    require(float(tm[target_index, target_index, 1]) == 1.0, f"target identity mismatch: {chain_id}")
    expected_target = min(
        chains,
        key=lambda context_chain_id: (
            -int(chains[context_chain_id]["resolved_residue_count"]),
            -float(chains[context_chain_id]["backbone_coverage"]),
            chains[context_chain_id]["source_chain_id"],
        ),
    )
    require(target_chain == expected_target, f"target selection mismatch: {chain_id}")
    return context_length, len(chains), retained_missing_positions


def main():
    args = parse_args()
    require(args.max_records >= 0, "--max-records cannot be negative")
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    required_files = (
        "manifest.json",
        "build_manifest.json",
        "list.csv",
        "index.jsonl",
        "records.jsonl",
        "valid_clusters.txt",
        "test_clusters.txt",
    )
    for filename in required_files:
        require((dataset_dir / filename).is_file(), f"missing dataset file: {filename}")
    require((dataset_dir / "shards").is_dir(), "missing shards directory")

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    build_manifest = json.loads(
        (dataset_dir / "build_manifest.json").read_text(encoding="utf-8")
    )
    require(manifest.get("format") == "proteinmpnn.tar_shard.v2", "dataset is not v2")
    require(
        manifest.get("payload_schema") == "structure_with_target_chain_ids",
        "unexpected payload schema",
    )
    require(manifest.get("build") == build_manifest, "embedded build manifest mismatch")
    require(build_manifest.get("cluster_map_entries", 0) > 0, "missing homology clusters")
    require(build_manifest.get("entry_metadata_records", 0) > 0, "missing entry metadata")
    require(
        manifest.get("record_granularity") == "pdb_canonical_assembly",
        "unexpected record granularity",
    )
    require(
        manifest.get("index_granularity") == "target_chain",
        "unexpected index granularity",
    )

    with (dataset_dir / "list.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    chain_index_rows = read_jsonl(dataset_dir / "index.jsonl")
    record_index_rows = read_jsonl(dataset_dir / "records.jsonl")
    require(rows, "dataset contains no targets")
    row_by_chain, _ = validate_indexes(
        dataset_dir, manifest, rows, chain_index_rows, record_index_rows
    )
    shard_count = validate_shards(dataset_dir, manifest, record_index_rows)

    valid_clusters = read_cluster_ids(dataset_dir / "valid_clusters.txt")
    test_clusters = read_cluster_ids(dataset_dir / "test_clusters.txt")
    all_clusters = {int(row["CLUSTER"]) for row in rows}
    require(not valid_clusters.intersection(test_clusters), "valid/test clusters overlap")
    require(valid_clusters.issubset(all_clusters), "valid split contains unknown clusters")
    require(test_clusters.issubset(all_clusters), "test split contains unknown clusters")
    sequence_splits = defaultdict(set)
    pdb_splits = defaultdict(set)
    for row in rows:
        split = split_name(int(row["CLUSTER"]), valid_clusters, test_clusters)
        sequence_splits[row["SEQUENCE"]].add(split)
        pdb_splits[row["CHAINID"][:4].lower()].add(split)
    exact_sequence_split_leaks = sum(
        len(splits) > 1 for splits in sequence_splits.values()
    )
    pdb_split_leaks = sum(len(splits) > 1 for splits in pdb_splits.values())
    require(exact_sequence_split_leaks == 0, "exact sequences cross data splits")
    require(pdb_split_leaks == 0, "PDB IDs cross data splits")

    store = TarShardStore(dataset_dir)
    require(len(store.index_by_chain) == len(rows), "TarShardStore index count mismatch")
    payload_rows = rows[: args.max_records] if args.max_records > 0 else rows
    retained_missing_positions = 0
    context_chain_count = 0
    max_observed_context_length = 0
    for row in payload_rows:
        chain_id = row["CHAINID"]
        payload = store.load_payload_for_chain(chain_id)
        context_length, chain_count, missing_positions = validate_payload(
            chain_id, row_by_chain[chain_id], payload, build_manifest["filters"]
        )
        context_chain_count += chain_count
        retained_missing_positions += missing_positions
        max_observed_context_length = max(max_observed_context_length, context_length)

    if args.max_records == 0:
        require(
            context_chain_count == manifest["context_chain_count"],
            "manifest context_chain_count mismatch",
        )

    result = {
        "schema": "proteinmpnn.tar_shard_validation.v2",
        "dataset_dir": str(dataset_dir),
        "records": len(rows),
        "payloads_checked": len(payload_rows),
        "shards_checked": shard_count,
        "context_chains_checked": context_chain_count,
        "max_observed_context_length": max_observed_context_length,
        "retained_missing_positions": retained_missing_positions,
        "exact_sequence_split_leaks": exact_sequence_split_leaks,
        "pdb_split_leaks": pdb_split_leaks,
        "status": "ok",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
