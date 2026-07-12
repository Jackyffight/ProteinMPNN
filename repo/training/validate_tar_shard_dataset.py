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
STANDARD_PAYLOAD_SCHEMA = "structure_with_target_chain_ids"
SPATIAL_CROP_PAYLOAD_SCHEMA = "structure_with_target_chain_ids_spatial_crop"
SPATIAL_CROP_POLICY = "full_target_nearest_chain_windows_v1"
RECORD_GRANULARITY_BY_SCHEMA = {
    STANDARD_PAYLOAD_SCHEMA: "pdb_canonical_assembly",
    SPATIAL_CROP_PAYLOAD_SCHEMA: "pdb_canonical_assembly_spatial_crop",
}


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


def validate_spatial_crop(chain_id, target_chain_id, chains, meta, build_manifest):
    filters = build_manifest["filters"]
    crop_meta = meta.get("crop")
    require(isinstance(crop_meta, dict), f"missing crop metadata: {chain_id}")
    require(
        build_manifest.get("crop_policy") == SPATIAL_CROP_POLICY,
        "unexpected build crop policy",
    )
    require(
        crop_meta.get("policy") == SPATIAL_CROP_POLICY,
        f"unexpected payload crop policy: {chain_id}",
    )
    require(crop_meta.get("target_was_cropped") is False, f"target was cropped: {chain_id}")
    require(
        crop_meta.get("target_source_chain_id")
        == chains[target_chain_id]["source_chain_id"],
        f"crop target source mismatch: {chain_id}",
    )

    original_length = int(crop_meta["original_context_length"])
    original_chains = int(crop_meta["original_context_chains"])
    retained_length = sum(len(chain["seq"]) for chain in chains.values())
    retained_chains = len(chains)
    max_context_length = int(filters["max_context_length"])
    max_chains = int(filters["max_chains"])
    require(
        int(crop_meta["max_context_length"]) == max_context_length,
        f"crop budget mismatch: {chain_id}",
    )
    require(
        original_length > max_context_length or original_chains > max_chains,
        f"source was not oversized: {chain_id}",
    )
    require(original_length >= retained_length, f"crop length grew: {chain_id}")
    require(original_chains >= retained_chains, f"crop chain count grew: {chain_id}")
    require(
        retained_length == int(crop_meta["retained_context_length"]),
        f"retained crop length mismatch: {chain_id}",
    )
    require(
        retained_chains == int(crop_meta["retained_context_chains"]),
        f"retained crop chain count mismatch: {chain_id}",
    )
    require(
        original_chains - retained_chains == int(crop_meta["dropped_context_chains"]),
        f"dropped crop chain count mismatch: {chain_id}",
    )

    ranked_context = []
    target_chain = chains[target_chain_id]
    for context_chain_id, chain in chains.items():
        crop = chain.get("crop")
        require(isinstance(crop, dict), f"missing chain crop metadata: {chain_id}/{context_chain_id}")
        length = len(chain["seq"])
        source_length = int(crop["source_sequence_length"])
        source_start = int(crop["source_residue_start"])
        source_end = int(crop["source_residue_end"])
        source_resolved = int(crop["source_resolved_residue_count"])
        source_coverage = float(crop["source_backbone_coverage"])
        require(0 <= source_resolved <= source_length, f"bad source resolved count: {chain_id}/{context_chain_id}")
        require(0.0 <= source_coverage <= 1.0, f"bad source coverage: {chain_id}/{context_chain_id}")
        require(
            math.isclose(
                source_resolved / float(source_length),
                source_coverage,
                abs_tol=1e-9,
            ),
            f"source coverage mismatch: {chain_id}/{context_chain_id}",
        )
        require(
            0 <= source_start < source_end <= source_length,
            f"bad source crop interval: {chain_id}/{context_chain_id}",
        )
        require(
            source_end - source_start == length,
            f"source crop length mismatch: {chain_id}/{context_chain_id}",
        )
        require(
            source_resolved >= int(chain["resolved_residue_count"]),
            f"crop gained resolved residues: {chain_id}/{context_chain_id}",
        )

        if context_chain_id == target_chain_id:
            require(crop.get("kind") == "full_target", f"target crop kind mismatch: {chain_id}")
            require(int(crop.get("spatial_rank", -1)) == 0, f"target crop rank mismatch: {chain_id}")
            require(source_start == 0 and source_end == source_length, f"target is not complete: {chain_id}")
            require(source_length == length, f"target source length mismatch: {chain_id}")
            require(source_resolved == int(chain["resolved_residue_count"]), f"target resolved count changed: {chain_id}")
            require(
                math.isclose(source_coverage, float(chain["backbone_coverage"]), abs_tol=1e-9),
                f"target source coverage changed: {chain_id}",
            )
            require(crop.get("nearest_target_residue") is None, f"target has nearest target index: {chain_id}")
            require(crop.get("nearest_source_residue") is None, f"target has nearest source index: {chain_id}")
            require(float(crop["distance_to_target"]) == 0.0, f"target crop distance changed: {chain_id}")
            continue

        kind = crop.get("kind")
        require(kind in {"full_context", "context_window"}, f"bad crop kind: {chain_id}/{context_chain_id}")
        rank = int(crop["spatial_rank"])
        distance = float(crop["distance_to_target"])
        target_index = int(crop["nearest_target_residue"])
        source_index = int(crop["nearest_source_residue"])
        require(rank > 0, f"bad spatial rank: {chain_id}/{context_chain_id}")
        require(math.isfinite(distance) and distance >= 0.0, f"bad crop distance: {chain_id}/{context_chain_id}")
        require(0 <= target_index < len(target_chain["seq"]), f"bad nearest target index: {chain_id}/{context_chain_id}")
        require(0 <= source_index < source_length, f"bad nearest source index: {chain_id}/{context_chain_id}")
        require(source_start <= source_index < source_end, f"nearest residue outside crop: {chain_id}/{context_chain_id}")
        if kind == "full_context":
            require(source_start == 0 and source_end == source_length, f"full context was cropped: {chain_id}/{context_chain_id}")
            require(
                source_resolved == int(chain["resolved_residue_count"]),
                f"full context resolved count changed: {chain_id}/{context_chain_id}",
            )
            require(
                math.isclose(
                    source_coverage,
                    float(chain["backbone_coverage"]),
                    abs_tol=1e-9,
                ),
                f"full context coverage changed: {chain_id}/{context_chain_id}",
            )
        else:
            require(length < source_length, f"context window is not shorter: {chain_id}/{context_chain_id}")

        retained_source_index = source_index - source_start
        target_ca = target_chain["xyz"][target_index, 1]
        context_ca = chain["xyz"][retained_source_index, 1]
        require(torch.isfinite(target_ca).all(), f"nearest target CA missing: {chain_id}/{context_chain_id}")
        require(torch.isfinite(context_ca).all(), f"nearest context CA missing: {chain_id}/{context_chain_id}")
        observed_distance = float(torch.linalg.vector_norm(target_ca - context_ca).item())
        require(
            math.isclose(observed_distance, distance, rel_tol=1e-5, abs_tol=1e-5),
            f"crop distance mismatch: {chain_id}/{context_chain_id}",
        )
        ranked_context.append((rank, distance, chain["source_chain_id"]))

    ranks = [rank for rank, _, _ in ranked_context]
    require(len(ranks) == len(set(ranks)), f"duplicate spatial ranks: {chain_id}")
    by_rank = sorted(ranked_context)
    require(
        [(distance, source_id) for _, distance, source_id in by_rank]
        == sorted((distance, source_id) for _, distance, source_id in by_rank),
        f"spatial crop order mismatch: {chain_id}",
    )


def validate_payload(chain_id, row, payload, build_manifest, payload_schema):
    build_filters = build_manifest["filters"]
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
    if payload_schema == SPATIAL_CROP_PAYLOAD_SCHEMA:
        validate_spatial_crop(
            chain_id,
            target_chain,
            chains,
            meta,
            build_manifest,
        )
        expected_target = min(
            chains,
            key=lambda context_chain_id: (
                -int(chains[context_chain_id]["crop"]["source_resolved_residue_count"]),
                -float(chains[context_chain_id]["crop"]["source_backbone_coverage"]),
                chains[context_chain_id]["source_chain_id"],
            ),
        )
    else:
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


def validate_reference_split_inheritance(
    dataset_dir,
    build_manifest,
    rows,
    valid_clusters,
    test_clusters,
):
    recorded_reference = Path(build_manifest["reference_dataset"]).expanduser()
    candidates = (
        recorded_reference,
        dataset_dir.parent / recorded_reference.name,
    )
    reference_dataset = next(
        (candidate.resolve() for candidate in candidates if candidate.is_dir()),
        recorded_reference.resolve(),
    )
    expected_hashes = build_manifest.get("reference_files_sha256", {})
    immutable_files = (
        "manifest.json",
        "list.csv",
        "valid_clusters.txt",
        "test_clusters.txt",
    )
    require(
        set(immutable_files).issubset(expected_hashes),
        "missing reference dataset checksums",
    )
    for filename in immutable_files:
        path = reference_dataset / filename
        require(path.is_file(), f"missing reference dataset file: {path}")
        require(
            sha256_file(path) == expected_hashes[filename],
            f"reference dataset checksum mismatch: {filename}",
        )

    reference_validation_path = reference_dataset / "validation.json"
    require(
        reference_validation_path.is_file(),
        f"missing reference dataset file: {reference_validation_path}",
    )
    reference_manifest = json.loads(
        (reference_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    reference_validation = json.loads(
        reference_validation_path.read_text(encoding="utf-8")
    )
    reference_record_count = int(reference_manifest.get("record_count", 0))
    require(reference_validation.get("status") == "ok", "reference validation is not ok")
    require(
        int(reference_validation.get("payloads_checked", 0)) == reference_record_count,
        "reference validation does not cover every payload",
    )

    with (reference_dataset / "list.csv").open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    reference_valid = read_cluster_ids(reference_dataset / "valid_clusters.txt")
    reference_test = read_cluster_ids(reference_dataset / "test_clusters.txt")
    require(reference_rows, "reference dataset contains no targets")
    require(
        len(reference_rows) == reference_record_count,
        "reference manifest/list record counts differ",
    )
    require(not reference_valid.intersection(reference_test), "reference valid/test overlap")

    reference_cluster_splits = defaultdict(set)
    reference_sequence_splits = defaultdict(set)
    reference_pdb_ids = set()
    for row in reference_rows:
        split = split_name(int(row["CLUSTER"]), reference_valid, reference_test)
        reference_cluster_splits[int(row["CLUSTER"])].add(split)
        reference_sequence_splits[row["SEQUENCE"]].add(split)
        reference_pdb_ids.add(row["CHAINID"][:4].lower())
    require(
        all(len(splits) == 1 for splits in reference_cluster_splits.values()),
        "reference clusters cross splits",
    )
    require(
        all(len(splits) == 1 for splits in reference_sequence_splits.values()),
        "reference exact sequences cross splits",
    )
    cluster_split = {
        cluster: next(iter(splits))
        for cluster, splits in reference_cluster_splits.items()
    }
    sequence_split = {
        sequence: next(iter(splits))
        for sequence, splits in reference_sequence_splits.items()
    }

    stage_rows_by_cluster = defaultdict(list)
    for row in rows:
        stage_rows_by_cluster[int(row["CLUSTER"])].append(row)
    inherited_clusters = 0
    exact_sequence_overlaps = 0
    for cluster, cluster_rows in stage_rows_by_cluster.items():
        anchors = set()
        if cluster in cluster_split:
            anchors.add(cluster_split[cluster])
        for row in cluster_rows:
            inherited = sequence_split.get(row["SEQUENCE"])
            if inherited is not None:
                anchors.add(inherited)
                exact_sequence_overlaps += 1
        require(len(anchors) <= 1, f"conflicting reference split anchors: cluster {cluster}")
        expected_split = next(iter(anchors)) if anchors else "train"
        actual_split = split_name(cluster, valid_clusters, test_clusters)
        require(
            actual_split == expected_split,
            f"reference split inheritance mismatch: cluster {cluster}",
        )
        inherited_clusters += bool(anchors)

    stage_pdb_ids = {row["CHAINID"][:4].lower() for row in rows}
    require(
        not stage_pdb_ids.intersection(reference_pdb_ids),
        "stage and reference datasets contain overlapping PDB IDs",
    )
    return {
        "reference_records": len(reference_rows),
        "reference_anchored_clusters": inherited_clusters,
        "reference_exact_sequence_overlaps": exact_sequence_overlaps,
        "reference_pdb_overlaps": 0,
    }


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
    payload_schema = manifest.get("payload_schema")
    require(
        payload_schema in RECORD_GRANULARITY_BY_SCHEMA,
        "unexpected payload schema",
    )
    require(manifest.get("build") == build_manifest, "embedded build manifest mismatch")
    require(build_manifest.get("cluster_map_entries", 0) > 0, "missing homology clusters")
    require(build_manifest.get("entry_metadata_records", 0) > 0, "missing entry metadata")
    require(
        manifest.get("record_granularity") == RECORD_GRANULARITY_BY_SCHEMA[payload_schema],
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
    reference_split_stats = {}
    quarantine_stats = {"quarantined_payloads": 0}
    if payload_schema == SPATIAL_CROP_PAYLOAD_SCHEMA:
        require(
            manifest.get("crop_policy") == SPATIAL_CROP_POLICY,
            "unexpected manifest crop policy",
        )
        reference_split_stats = validate_reference_split_inheritance(
            dataset_dir,
            build_manifest,
            rows,
            valid_clusters,
            test_clusters,
        )
        quarantined_payloads = int(manifest.get("quarantined_payload_count", 0))
        require(
            quarantined_payloads
            == int(build_manifest.get("counts", {}).get("split_conflict_rows_quarantined", 0)),
            "manifest/build quarantine counts differ",
        )
        if quarantined_payloads > 0:
            require(
                manifest.get("quarantine_policy")
                == "exclude_reference_split_conflict_components",
                "unexpected quarantine policy",
            )
            quarantine_file = manifest.get("files", {}).get("quarantine")
            require(quarantine_file, "manifest does not name the quarantine file")
            quarantine_path = dataset_dir / quarantine_file
            require(quarantine_path.is_file(), "quarantine file is missing")
            quarantine_rows = read_jsonl(quarantine_path)
            quarantine_ids = [row.get("chain_id") for row in quarantine_rows]
            require(
                len(quarantine_rows) == quarantined_payloads,
                "quarantine row count mismatch",
            )
            require(
                len(set(quarantine_ids)) == len(quarantine_ids),
                "duplicate quarantine chain IDs",
            )
            require(
                set(quarantine_ids).isdisjoint(row_by_chain),
                "quarantined chains remain in the training index",
            )
            require(
                all(
                    row.get("reason") == "reference_split_conflict_component"
                    for row in quarantine_rows
                ),
                "unexpected quarantine reason",
            )
        quarantine_stats = {"quarantined_payloads": quarantined_payloads}

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
            chain_id,
            row_by_chain[chain_id],
            payload,
            build_manifest,
            payload_schema,
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
        "payload_schema": payload_schema,
        "records": len(rows),
        "payloads_checked": len(payload_rows),
        "shards_checked": shard_count,
        "context_chains_checked": context_chain_count,
        "max_observed_context_length": max_observed_context_length,
        "retained_missing_positions": retained_missing_positions,
        "exact_sequence_split_leaks": exact_sequence_split_leaks,
        "pdb_split_leaks": pdb_split_leaks,
        **reference_split_stats,
        **quarantine_stats,
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
