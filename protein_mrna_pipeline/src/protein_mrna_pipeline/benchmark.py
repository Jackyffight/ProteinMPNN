"""Deterministic, metadata-only engineering benchmark selection."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    ContractError,
    canonical_json_bytes,
    derive_benchmark_id,
    read_json,
    text_sha256,
    validate_benchmark_suite,
)


CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUIRED_DATASET_FILES = (
    "manifest.json",
    "validation.json",
    "list.csv",
    "valid_clusters.txt",
    "test_clusters.txt",
)
REQUIRED_LIST_COLUMNS = (
    "CHAINID",
    "DEPOSITION",
    "RESOLUTION",
    "HASH",
    "CLUSTER",
    "SEQUENCE",
)


@dataclass(frozen=True)
class LengthBin:
    label: str
    min_length: int
    max_length: int

    def contains(self, length: int) -> bool:
        return self.min_length <= length <= self.max_length


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_length_bins(min_length: int, max_length: int, count: int) -> list[LengthBin]:
    if min_length < 1 or max_length < min_length:
        raise ContractError("benchmark length bounds are invalid")
    width = max_length - min_length + 1
    if count < 1 or count > width:
        raise ContractError("length bin count must fit within the selected length range")
    bins = []
    for index in range(count):
        start = min_length + (index * width) // count
        end = min_length + ((index + 1) * width) // count - 1
        bins.append(LengthBin(f"length-{start:04d}-{end:04d}", start, end))
    return bins


def _read_cluster_ids(path: Path) -> set[int]:
    cluster_ids = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value:
            continue
        try:
            cluster_ids.append(int(value))
        except ValueError as error:
            raise ContractError(
                f"invalid cluster ID in {path} at line {line_number}: {value}"
            ) from error
    if not cluster_ids:
        raise ContractError(f"cluster split is empty: {path}")
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ContractError(f"cluster split contains duplicate IDs: {path}")
    if any(cluster_id < 0 for cluster_id in cluster_ids):
        raise ContractError(f"cluster split contains a negative ID: {path}")
    return set(cluster_ids)


def _row_rank(seed: int, row: dict) -> str:
    identity = f"{seed}\0{row['source_chain_id']}\0{row['sequence_sha256']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _parse_resolution(value: str, line_number: int) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        resolution = float(stripped)
    except ValueError as error:
        raise ContractError(
            f"invalid resolution in list.csv at line {line_number}: {value}"
        ) from error
    if not math.isfinite(resolution) or resolution <= 0:
        raise ContractError(
            f"non-finite or non-positive resolution in list.csv at line "
            f"{line_number}: {value}"
        )
    return resolution


def _required_csv_value(source: dict, column: str, line_number: int) -> str:
    value = source.get(column)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"empty {column} in list.csv at line {line_number}")
    return value.strip()


def _load_eligible_rows(
    list_path: Path,
    valid_clusters: set[int],
    min_length: int,
    max_length: int,
) -> tuple[list[dict], dict[str, int]]:
    counts = {
        "list_rows": 0,
        "valid_split_rows": 0,
        "length_eligible_rows": 0,
        "canonical_rows": 0,
        "unique_cluster_sequence_rows": 0,
    }
    rows = []
    with list_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(REQUIRED_LIST_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ContractError(f"list.csv is missing required columns: {missing}")
        for line_number, source in enumerate(reader, 2):
            counts["list_rows"] += 1
            cluster_value = _required_csv_value(source, "CLUSTER", line_number)
            try:
                cluster = int(cluster_value)
            except (TypeError, ValueError) as error:
                raise ContractError(
                    f"invalid cluster in list.csv at line {line_number}: "
                    f"{cluster_value}"
                ) from error
            if cluster not in valid_clusters:
                continue
            counts["valid_split_rows"] += 1
            sequence = _required_csv_value(source, "SEQUENCE", line_number).upper()
            if not min_length <= len(sequence) <= max_length:
                continue
            counts["length_eligible_rows"] += 1
            if not sequence or not set(sequence) <= CANONICAL_AMINO_ACIDS:
                continue
            counts["canonical_rows"] += 1
            chain_id = _required_csv_value(source, "CHAINID", line_number)
            resolution_value = source.get("RESOLUTION")
            if not isinstance(resolution_value, str):
                raise ContractError(f"missing RESOLUTION in list.csv at line {line_number}")
            rows.append(
                {
                    "source_chain_id": chain_id,
                    "source_cluster": cluster,
                    "source_sequence_hash": _required_csv_value(
                        source, "HASH", line_number
                    ),
                    "deposition_date": _required_csv_value(
                        source, "DEPOSITION", line_number
                    ),
                    "resolution": _parse_resolution(resolution_value, line_number),
                    "length": len(sequence),
                    "sequence_sha256": text_sha256(sequence),
                    "sequence": sequence,
                }
            )
    return rows, counts


def _deduplicate_rows(rows: list[dict], seed: int) -> list[dict]:
    selected = []
    clusters = set()
    sequences = set()
    for row in sorted(rows, key=lambda item: (_row_rank(seed, item), item["source_chain_id"])):
        if row["source_cluster"] in clusters or row["sequence_sha256"] in sequences:
            continue
        selected.append(row)
        clusters.add(row["source_cluster"])
        sequences.add(row["sequence_sha256"])
    return selected


def _stratified_select(
    rows: list[dict],
    length_bins: list[LengthBin],
    requested_count: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if requested_count < 1 or requested_count > 1000:
        raise ContractError("benchmark count must be between 1 and 1000")
    unique_capacity = len(_deduplicate_rows(rows, seed))
    if requested_count > unique_capacity:
        raise ContractError(
            "not enough unique valid-split clusters for benchmark: "
            f"requested={requested_count} eligible={unique_capacity}"
        )

    rows_by_bin = {length_bin.label: [] for length_bin in length_bins}
    for row in rows:
        matching = [length_bin for length_bin in length_bins if length_bin.contains(row["length"])]
        if len(matching) != 1:
            raise ContractError(
                f"eligible row length is not covered by exactly one bin: {row['length']}"
            )
        row["length_bin"] = matching[0].label
        rows_by_bin[matching[0].label].append(row)

    base, remainder = divmod(requested_count, len(length_bins))
    selected = []
    selected_clusters = set()
    selected_sequences = set()
    selected_chains = set()
    quotas = {
        length_bin.label: base + (1 if index < remainder else 0)
        for index, length_bin in enumerate(length_bins)
    }
    candidates_by_bin = {
        length_bin.label: _deduplicate_rows(rows_by_bin[length_bin.label], seed)
        for length_bin in length_bins
    }

    # Reserve scarce length buckets first so a multi-length cluster cannot be
    # consumed by an abundant bucket before its rarer representative is chosen.
    selection_order = sorted(
        length_bins,
        key=lambda length_bin: (
            len(candidates_by_bin[length_bin.label]),
            length_bin.min_length,
        ),
    )
    for length_bin in selection_order:
        selected_for_bin = 0
        for row in candidates_by_bin[length_bin.label]:
            if selected_for_bin >= quotas[length_bin.label]:
                break
            if (
                row["source_cluster"] in selected_clusters
                or row["sequence_sha256"] in selected_sequences
            ):
                continue
            selected.append(row)
            selected_clusters.add(row["source_cluster"])
            selected_sequences.add(row["sequence_sha256"])
            selected_chains.add(row["source_chain_id"])
            selected_for_bin += 1

    if len(selected) < requested_count:
        remaining = sorted(
            rows,
            key=lambda item: (_row_rank(seed, item), item["source_chain_id"]),
        )
        for row in remaining:
            if len(selected) >= requested_count:
                break
            if (
                row["source_chain_id"] in selected_chains
                or row["source_cluster"] in selected_clusters
                or row["sequence_sha256"] in selected_sequences
            ):
                continue
            selected.append(row)
            selected_clusters.add(row["source_cluster"])
            selected_sequences.add(row["sequence_sha256"])
            selected_chains.add(row["source_chain_id"])

    if len(selected) != requested_count:
        raise ContractError(
            "benchmark stratification could not satisfy the requested count: "
            f"requested={requested_count} selected={len(selected)}"
        )

    bin_order = {length_bin.label: index for index, length_bin in enumerate(length_bins)}
    selected.sort(
        key=lambda item: (
            bin_order[item["length_bin"]],
            item["length"],
            item["source_chain_id"],
        )
    )
    for index, row in enumerate(selected, 1):
        row["benchmark_record_id"] = f"record-{index:04d}"

    bin_summaries = []
    for length_bin in length_bins:
        bin_summaries.append(
            {
                "label": length_bin.label,
                "min_length": length_bin.min_length,
                "max_length": length_bin.max_length,
                "eligible": len(candidates_by_bin[length_bin.label]),
                "selected": sum(
                    row["length_bin"] == length_bin.label for row in selected
                ),
            }
        )
    return selected, bin_summaries


def _fasta_bytes(records: list[dict]) -> bytes:
    lines = []
    for record in records:
        lines.append(
            f">{record['benchmark_record_id']} "
            f"source_chain={record['source_chain_id']} "
            f"cluster={record['source_cluster']} length={record['length']}"
        )
        sequence = record["sequence"]
        lines.extend(sequence[offset : offset + 80] for offset in range(0, len(sequence), 80))
    return ("\n".join(lines) + "\n").encode("ascii")


def verify_benchmark_suite_files(suite_path: str | Path) -> dict:
    document_path = Path(suite_path).expanduser().resolve()
    suite = read_json(document_path)
    validate_benchmark_suite(suite)
    fasta_path = document_path.parent / suite["fasta"]["path"]
    if not fasta_path.is_file():
        raise ContractError(f"benchmark FASTA not found: {fasta_path}")
    fasta = fasta_path.read_bytes()
    if len(fasta) != int(suite["fasta"]["bytes"]):
        raise ContractError("benchmark FASTA byte size mismatch")
    if hashlib.sha256(fasta).hexdigest() != suite["fasta"]["sha256"]:
        raise ContractError("benchmark FASTA SHA256 mismatch")
    if fasta != _fasta_bytes(suite["records"]):
        raise ContractError("benchmark FASTA content does not match suite records")
    return {
        "benchmark_id": suite["benchmark_id"],
        "suite_path": str(document_path),
        "suite_sha256": sha256_file(document_path),
        "fasta_path": str(fasta_path),
        "fasta_sha256": suite["fasta"]["sha256"],
        "records": len(suite["records"]),
        "status": "ok",
    }


def generate_benchmark_suite(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    requested_count: int = 40,
    seed: int = 42,
    min_length: int = 50,
    max_length: int = 800,
    length_bin_count: int = 4,
) -> dict:
    if seed < 0:
        raise ContractError("benchmark seed must be non-negative")
    source_dir = Path(dataset_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_DATASET_FILES if not (source_dir / name).is_file()]
    if missing:
        raise ContractError(f"benchmark dataset is missing required files: {missing}")

    manifest = read_json(source_dir / "manifest.json")
    validation = read_json(source_dir / "validation.json")
    if manifest.get("format") != "proteinmpnn.tar_shard.v2":
        raise ContractError("benchmark source is not proteinmpnn.tar_shard.v2")
    if validation.get("schema") != "proteinmpnn.tar_shard_validation.v2":
        raise ContractError("benchmark source has an unexpected validation schema")
    if validation.get("status") != "ok":
        raise ContractError("benchmark source validation status is not ok")
    if validation.get("exact_sequence_split_leaks") != 0:
        raise ContractError("benchmark source reports exact-sequence split leakage")
    if validation.get("pdb_split_leaks") != 0:
        raise ContractError("benchmark source reports PDB split leakage")
    expected_records = manifest.get("record_count")
    if (
        not isinstance(expected_records, int)
        or isinstance(expected_records, bool)
        or expected_records <= 0
    ):
        raise ContractError("benchmark source manifest has no records")
    if validation.get("records") != expected_records:
        raise ContractError("benchmark source manifest/validation record counts differ")
    if validation.get("payloads_checked") != expected_records:
        raise ContractError("benchmark source validation does not cover every payload")

    valid_path = source_dir / "valid_clusters.txt"
    test_path = source_dir / "test_clusters.txt"
    valid_clusters = _read_cluster_ids(valid_path)
    test_clusters = _read_cluster_ids(test_path)
    overlap = valid_clusters & test_clusters
    if overlap:
        raise ContractError(f"valid/test cluster files overlap: {sorted(overlap)[:20]}")

    length_bins = make_length_bins(min_length, max_length, length_bin_count)
    rows, source_counts = _load_eligible_rows(
        source_dir / "list.csv",
        valid_clusters,
        min_length,
        max_length,
    )
    if source_counts["list_rows"] != expected_records:
        raise ContractError(
            "benchmark source list.csv row count does not match the manifest: "
            f"list={source_counts['list_rows']} manifest={expected_records}"
        )
    unique_rows = _deduplicate_rows(rows, seed)
    source_counts["unique_cluster_sequence_rows"] = len(unique_rows)
    selected, bin_summaries = _stratified_select(
        rows,
        length_bins,
        requested_count,
        seed,
    )

    source = {
        "dataset_id": manifest.get("version_id", "unknown-dataset"),
        "dataset_path": str(source_dir),
        "dataset_format": manifest["format"],
        "record_count": expected_records,
        "split": "valid",
        "manifest_sha256": sha256_file(source_dir / "manifest.json"),
        "validation_sha256": sha256_file(source_dir / "validation.json"),
        "list_sha256": sha256_file(source_dir / "list.csv"),
        "split_sha256": sha256_file(valid_path),
        "test_split_sha256": sha256_file(test_path),
    }
    suite = {
        "schema_version": "protein-mrna.benchmark-suite.v1",
        "benchmark_id": "pending",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "engineering_structure_throughput",
        "source": source,
        "selection": {
            "seed": seed,
            "requested_count": requested_count,
            "selected_count": len(selected),
            "min_length": min_length,
            "max_length": max_length,
            "canonical_amino_acids_only": True,
            "one_record_per_cluster": True,
            "exact_sequence_unique": True,
            "length_bins": bin_summaries,
            "source_counts": source_counts,
        },
        "records": selected,
        "fasta": {},
        "limitations": [
            "Engineering benchmark only; records are not a biological design target.",
            "The valid split is used for throughput calibration; the test split remains unused.",
            "Structure-model outputs are proxy labels, not wet-lab efficacy evidence.",
        ],
    }
    suite["benchmark_id"] = derive_benchmark_id(suite)
    fasta = _fasta_bytes(selected)
    suite["fasta"] = {
        "path": "sequences.fasta",
        "sha256": hashlib.sha256(fasta).hexdigest(),
        "bytes": len(fasta),
        "records": len(selected),
    }
    validate_benchmark_suite(suite)

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise ContractError(f"benchmark output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ContractError(f"temporary benchmark directory already exists: {temporary}")
    try:
        temporary.mkdir()
        (temporary / "sequences.fasta").write_bytes(fasta)
        (temporary / "benchmark-suite.json").write_bytes(canonical_json_bytes(suite) + b"\n")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    suite_path = destination / "benchmark-suite.json"
    return {
        "benchmark_id": suite["benchmark_id"],
        "output_dir": str(destination),
        "suite_path": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "fasta_path": str(destination / "sequences.fasta"),
        "records": len(selected),
        "min_observed_length": min(row["length"] for row in selected),
        "max_observed_length": max(row["length"] for row in selected),
        "split": "valid",
    }
