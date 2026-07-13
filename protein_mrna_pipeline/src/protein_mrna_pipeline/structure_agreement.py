"""Native-structure agreement evaluation for the fixed ESMFold2 benchmark."""

from __future__ import annotations

import importlib.metadata
import io
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .benchmark import sha256_file, verify_benchmark_suite_files
from .contracts import ContractError, document_sha256, read_json
from .esmfold2_runner import verify_esmfold2_benchmark_run
from .run_store import write_json_atomic, write_jsonl_atomic


METRICS_RUNTIME_VERSIONS = {
    "biotite": "1.6.0",
    "biotraj": "1.2.2",
    "jsonschema": "4.25.1",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
}
AA1_TO_3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}
AA3_TO_1 = {value: key for key, value in AA1_TO_3.items()}
METRIC_FIELDS = (
    "ca_lddt",
    "ca_rmsd_angstrom",
    "ca_tm_score_full_length",
    "ca_tm_score_resolved",
    "native_ca_coverage",
    "native_complete_backbone_coverage",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity(document: dict, field: str) -> str:
    payload = dict(document)
    payload.pop(field, None)
    payload.pop("created_at_utc", None)
    return document_sha256(payload)


def _distribution_versions() -> dict[str, str | None]:
    names = (
        "biotite",
        "biotraj",
        "jsonschema",
        "numpy",
        "scipy",
        "torch",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def create_metrics_runtime_manifest(runtime_root: str | Path) -> dict:
    root = Path(runtime_root).expanduser().resolve()
    versions = _distribution_versions()
    for name, expected in METRICS_RUNTIME_VERSIONS.items():
        if versions[name] != expected:
            raise ContractError(
                f"structure metrics dependency mismatch for {name}: "
                f"expected={expected} observed={versions[name]}"
            )
    if versions["torch"] is None:
        raise ContractError("structure metrics runtime cannot find host Torch")
    try:
        import biotite.structure  # noqa: F401
        import biotraj  # noqa: F401
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import torch  # noqa: F401
    except Exception as error:
        raise ContractError(f"structure metrics runtime import failed: {error}") from error

    manifest = {
        "schema_version": "protein-mrna.structure-metrics-runtime.v1",
        "runtime_identity": "pending",
        "created_at_utc": utc_now(),
        "runtime_root": str(root),
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "distributions": versions,
        },
        "implementation": {
            "library": "Biotite",
            "version": METRICS_RUNTIME_VERSIONS["biotite"],
            "documentation": "https://www.biotite-python.org/1.6.0/apidoc/",
        },
    }
    manifest["runtime_identity"] = _identity(manifest, "runtime_identity")
    write_json_atomic(root / "runtime-manifest.json", manifest)
    return manifest


def _validate_metrics_runtime_document(manifest: dict) -> None:
    if (
        manifest.get("schema_version") != "protein-mrna.structure-metrics-runtime.v1"
        or manifest.get("runtime_identity")
        != _identity(manifest, "runtime_identity")
    ):
        raise ContractError("structure metrics runtime manifest identity mismatch")


def load_metrics_runtime_manifest(runtime_root: str | Path) -> dict:
    root = Path(runtime_root).expanduser().resolve()
    path = root / "runtime-manifest.json"
    if not path.is_file():
        raise ContractError(f"structure metrics runtime manifest not found: {path}")
    manifest = read_json(path)
    _validate_metrics_runtime_document(manifest)
    if manifest.get("runtime_root") != str(root):
        raise ContractError("structure metrics runtime manifest identity mismatch")
    observed = _distribution_versions()
    recorded = manifest.get("environment", {}).get("distributions")
    if observed != recorded:
        raise ContractError("active structure metrics environment differs from its manifest")
    for name, expected in METRICS_RUNTIME_VERSIONS.items():
        if observed[name] != expected:
            raise ContractError(f"active structure metrics dependency changed: {name}")
    return manifest


def _verify_dataset(dataset_dir: Path, suite: dict) -> dict:
    required = (
        "manifest.json",
        "validation.json",
        "list.csv",
        "index.jsonl",
        "valid_clusters.txt",
        "test_clusters.txt",
    )
    missing = [name for name in required if not (dataset_dir / name).is_file()]
    if missing:
        raise ContractError(f"native structure dataset is missing files: {missing}")
    manifest = read_json(dataset_dir / "manifest.json")
    validation = read_json(dataset_dir / "validation.json")
    if manifest.get("format") != "proteinmpnn.tar_shard.v2":
        raise ContractError("native structure dataset has the wrong format")
    if manifest.get("payload_schema") != "structure_with_target_chain_ids":
        raise ContractError("native structure dataset is not the fixed v1 payload schema")
    if validation.get("schema") != "proteinmpnn.tar_shard_validation.v2":
        raise ContractError("native structure dataset has the wrong validation schema")
    if validation.get("status") != "ok":
        raise ContractError("native structure dataset validation did not pass")
    if validation.get("exact_sequence_split_leaks") != 0:
        raise ContractError("native structure dataset reports sequence split leakage")
    if validation.get("pdb_split_leaks") != 0:
        raise ContractError("native structure dataset reports PDB split leakage")
    if validation.get("payloads_checked") != manifest.get("record_count"):
        raise ContractError("native structure validation does not cover all payloads")
    source = suite["source"]
    if (
        manifest.get("version_id") != source["dataset_id"]
        or manifest.get("format") != source["dataset_format"]
        or manifest.get("record_count") != source["record_count"]
    ):
        raise ContractError("native structure dataset identity differs from benchmark source")
    expected_hashes = {
        "manifest.json": source["manifest_sha256"],
        "validation.json": source["validation_sha256"],
        "list.csv": source["list_sha256"],
        "valid_clusters.txt": source["split_sha256"],
        "test_clusters.txt": source["test_split_sha256"],
    }
    for name, expected in expected_hashes.items():
        observed = sha256_file(dataset_dir / name)
        if observed != expected:
            raise ContractError(
                f"native structure dataset differs from benchmark source: {name}"
            )
    return {
        "dataset_id": source["dataset_id"],
        "dataset_dir": str(dataset_dir),
        "manifest_sha256": expected_hashes["manifest.json"],
        "validation_sha256": expected_hashes["validation.json"],
        "index_sha256": sha256_file(dataset_dir / "index.jsonl"),
        "payload_schema": manifest.get("payload_schema"),
        "record_count": manifest["record_count"],
        "shards": manifest.get("shards", []),
        "shard_integrity": "manifest_SHA256_recorded_and_file_size_verified",
    }


def _selected_index_rows(dataset_dir: Path, chain_ids: set[str]) -> dict[str, dict]:
    rows = {}
    with (dataset_dir / "index.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(
                    f"invalid native structure index JSON at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ContractError(
                    f"native structure index row is not an object: {line_number}"
                )
            chain_id = row.get("chain_id")
            if chain_id in chain_ids:
                if chain_id in rows:
                    raise ContractError(f"duplicate native structure index row: {chain_id}")
                rows[chain_id] = row
    missing = sorted(chain_ids - set(rows))
    if missing:
        raise ContractError(f"benchmark chains missing from native structure index: {missing}")
    return rows


def _validate_selected_index_rows(
    dataset_dir: Path,
    rows: dict[str, dict],
    dataset: dict,
) -> None:
    shards = {
        f"shards/{item['name']}": item
        for item in dataset["shards"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if len(shards) != len(dataset["shards"]):
        raise ContractError("native structure manifest has malformed shard metadata")
    checked_shards = set()
    for chain_id, row in rows.items():
        if row.get("chain_id") != chain_id:
            raise ContractError(f"native index chain identity mismatch: {chain_id}")
        shard_name = row.get("shard")
        if shard_name not in shards:
            raise ContractError(f"native index references an unknown shard: {chain_id}")
        try:
            offset = int(row["offset"])
            size = int(row["size"])
            shard_size = int(shards[shard_name]["bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"native index has invalid offsets: {chain_id}") from error
        if offset < 512 or size <= 0 or offset + size > shard_size:
            raise ContractError(f"native index payload is outside its shard: {chain_id}")
        if shard_name not in checked_shards:
            shard_path = dataset_dir / shard_name
            if not shard_path.is_file() or shard_path.stat().st_size != shard_size:
                raise ContractError(f"native shard size mismatch: {shard_name}")
            checked_shards.add(shard_name)


def _load_payload(dataset_dir: Path, index_row: dict) -> dict:
    try:
        import torch
    except Exception as error:
        raise ContractError(f"Torch is required to read native payloads: {error}") from error
    shard_path = dataset_dir / index_row["shard"]
    with shard_path.open("rb") as handle:
        handle.seek(int(index_row["offset"]))
        data = handle.read(int(index_row["size"]))
    if len(data) != int(index_row["size"]):
        raise ContractError(f"truncated native payload read: {index_row['chain_id']}")
    try:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(io.BytesIO(data), map_location="cpu")


def _to_numpy(value):
    try:
        value = value.detach().cpu()
    except AttributeError:
        pass
    try:
        return value.numpy()
    except AttributeError:
        import numpy as np

        return np.asarray(value)


def extract_native_ca(
    payload: dict,
    source_chain_id: str,
    expected_sequence: str,
) -> dict:
    import numpy as np

    if payload.get("format") != "proteinmpnn.tar_shard.v2":
        raise ContractError(f"native payload format mismatch: {source_chain_id}")
    if payload.get("target_chain_ids") != [source_chain_id]:
        raise ContractError(f"native payload target mismatch: {source_chain_id}")
    meta = payload.get("meta", {})
    target_chain_id = meta.get("target_chain")
    chains = payload.get("chains", {})
    if target_chain_id not in chains:
        raise ContractError(f"native target chain is absent: {source_chain_id}")
    chain = chains[target_chain_id]
    if chain.get("seq") != expected_sequence:
        raise ContractError(f"native target sequence mismatch: {source_chain_id}")
    xyz = np.asarray(_to_numpy(chain["xyz"]), dtype=np.float64)
    mask = np.asarray(_to_numpy(chain["mask"]), dtype=bool)
    length = len(expected_sequence)
    if xyz.shape != (length, 14, 3) or mask.shape != (length, 14):
        raise ContractError(f"native target tensor shape mismatch: {source_chain_id}")
    ca_mask = mask[:, 1] & np.isfinite(xyz[:, 1, :]).all(axis=1)
    complete_mask = mask[:, :4].all(axis=1) & np.isfinite(xyz[:, :4, :]).all(
        axis=(1, 2)
    )
    ca_count = int(ca_mask.sum())
    if ca_count < 3:
        raise ContractError(f"native target has fewer than three resolved CAs: {source_chain_id}")
    return {
        "ca_coordinates": xyz[:, 1, :],
        "ca_mask": ca_mask,
        "ca_residues": ca_count,
        "ca_coverage": ca_count / float(length),
        "complete_backbone_residues": int(complete_mask.sum()),
        "complete_backbone_coverage": float(complete_mask.mean()),
        "source_pdb_id": chain.get("source_pdb_id"),
        "source_assembly_id": chain.get("source_assembly_id"),
        "source_mmcif_chain_id": chain.get("source_chain_id"),
        "source_entity_id": chain.get("source_entity_id"),
    }


def parse_prediction_ca(pdb_path: str | Path, expected_sequence: str):
    import numpy as np
    from biotite.structure.io.pdb import PDBFile

    path = Path(pdb_path)
    try:
        atoms = PDBFile.read(path).get_structure(model=1)
    except Exception as error:
        raise ContractError(f"cannot parse predicted PDB {path}: {error}") from error
    ca = atoms[(atoms.atom_name == "CA") & ~atoms.hetero]
    length = len(expected_sequence)
    if len(ca) != length:
        raise ContractError(
            f"predicted PDB CA count mismatch: expected={length} observed={len(ca)}"
        )
    chain_ids = sorted(set(str(value) for value in ca.chain_id))
    if len(chain_ids) != 1:
        raise ContractError(f"predicted PDB is not single-chain: {chain_ids}")
    if not np.array_equal(ca.res_id, np.arange(1, length + 1)):
        raise ContractError("predicted PDB residue IDs do not cover 1..L")
    if hasattr(ca, "ins_code") and any(str(value).strip() for value in ca.ins_code):
        raise ContractError("predicted PDB contains residue insertion codes")
    try:
        observed_sequence = "".join(AA3_TO_1[str(name).upper()] for name in ca.res_name)
    except KeyError as error:
        raise ContractError(f"predicted PDB has an unknown residue: {error}") from error
    if observed_sequence != expected_sequence:
        raise ContractError("predicted PDB sequence differs from benchmark sequence")
    coordinates = np.asarray(ca.coord, dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ContractError("predicted PDB contains non-finite CA coordinates")
    return coordinates


def compute_ca_metrics(
    native_coordinates,
    prediction_coordinates,
    native_mask,
    sequence: str,
) -> dict[str, float]:
    import numpy as np
    import biotite.structure as struc

    native = np.asarray(native_coordinates, dtype=np.float32)
    prediction = np.asarray(prediction_coordinates, dtype=np.float32)
    mask = np.asarray(native_mask, dtype=bool)
    length = len(sequence)
    if native.shape != (length, 3) or prediction.shape != (length, 3):
        raise ContractError("CA metric coordinate shapes do not match sequence length")
    if mask.shape != (length,) or int(mask.sum()) < 3:
        raise ContractError("CA metric mask is invalid")
    positions = np.flatnonzero(mask)
    reference = struc.AtomArray(len(positions))
    reference.coord = native[positions]
    reference.chain_id[:] = "A"
    reference.res_id = positions + 1
    reference.atom_name[:] = "CA"
    reference.res_name = np.asarray([AA1_TO_3[sequence[index]] for index in positions])
    reference.element[:] = "C"
    subject = reference.copy()
    subject.coord = prediction[positions]

    fitted, _ = struc.superimpose(reference, subject)
    indices = np.arange(len(positions), dtype=int)
    metrics = {
        "ca_lddt": float(
            struc.lddt(
                reference,
                fitted,
                inclusion_radius=15.0,
                distance_bins=(0.5, 1.0, 2.0, 4.0),
            )
        ),
        "ca_rmsd_angstrom": float(struc.rmsd(reference, fitted)),
        "ca_tm_score_full_length": float(
            struc.tm_score(
                reference,
                fitted,
                indices,
                indices,
                reference_length=length,
            )
        ),
        "ca_tm_score_resolved": float(
            struc.tm_score(
                reference,
                fitted,
                indices,
                indices,
                reference_length=len(positions),
            )
        ),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ContractError(f"non-finite native agreement metrics: {metrics}")
    return metrics


def _record_result_path(output_dir: Path, record_id: str) -> Path:
    return output_dir / "records" / f"{record_id}.json"


def _valid_record_result(output_dir: Path, record: dict, evaluation_identity: str):
    path = _record_result_path(output_dir, record["benchmark_record_id"])
    if not path.is_file():
        return None
    try:
        result = read_json(path)
    except ContractError:
        return None
    if (
        result.get("schema_version") != "protein-mrna.native-agreement-record.v1"
        or result.get("result_identity") != _identity(result, "result_identity")
        or result.get("evaluation_identity") != evaluation_identity
        or result.get("benchmark_record_id") != record["benchmark_record_id"]
        or result.get("sequence_sha256") != record["sequence_sha256"]
        or result.get("status") not in {"succeeded", "failed"}
    ):
        return None
    return result


def _metric_stats(results: list[dict]) -> dict:
    if not results:
        return {"count": 0}
    summary = {"count": len(results)}
    for field in METRIC_FIELDS:
        values = [float(result["metrics"][field]) for result in results]
        summary[field] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def _summarize(
    output_dir: Path,
    records: list[dict],
    evaluation_identity: str,
) -> dict:
    results = [
        result
        for record in records
        if (
            result := _valid_record_result(output_dir, record, evaluation_identity)
        )
        is not None
    ]
    succeeded = [result for result in results if result["status"] == "succeeded"]
    failed = [result for result in results if result["status"] == "failed"]
    by_bin = {}
    for length_bin in dict.fromkeys(record["length_bin"] for record in records):
        by_bin[length_bin] = _metric_stats(
            [result for result in succeeded if result["length_bin"] == length_bin]
        )
    mean_plddt = [float(result["prediction_confidence"]["mean_plddt"]) for result in succeeded]
    predicted_ptm = [float(result["prediction_confidence"]["ptm"]) for result in succeeded]
    ca_lddt = [float(result["metrics"]["ca_lddt"]) for result in succeeded]
    ca_tm = [float(result["metrics"]["ca_tm_score_full_length"]) for result in succeeded]
    summary = {
        "schema_version": "protein-mrna.native-agreement-summary.v1",
        "evaluation_identity": evaluation_identity,
        "updated_at_utc": utc_now(),
        "status": "passed" if len(succeeded) == len(records) else "failed" if failed else "running",
        "records": {
            "selected": len(records),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "pending": len(records) - len(results),
        },
        "overall": _metric_stats(succeeded),
        "by_length_bin": by_bin,
        "confidence_correlations": {
            "mean_plddt_vs_ca_lddt_pearson": _pearson(mean_plddt, ca_lddt),
            "predicted_ptm_vs_ca_tm_full_length_pearson": _pearson(
                predicted_ptm, ca_tm
            ),
        },
        "limitations": [
            "C-alpha-only metrics use sequence-position correspondence, not a "
            "free structural alignment.",
            "Only experimentally observed C-alpha positions contribute coordinate errors.",
            "Predictions are single-chain, while native chains come from biological "
            "assemblies and may contain interface-stabilized conformations.",
            "Possible overlap with ESMFold2 training data has not been audited; this "
            "is not a strict generalization benchmark.",
            "No quality threshold is used to select models or checkpoints.",
        ],
    }
    write_json_atomic(output_dir / "summary.json", summary)
    write_jsonl_atomic(output_dir / "records.jsonl", results)
    return summary


def _initialize_output(output_dir: Path, manifest: dict) -> None:
    path = output_dir / "evaluation-manifest.json"
    if output_dir.exists():
        if not path.is_file():
            raise ContractError(f"existing native agreement output has no manifest: {output_dir}")
        existing = read_json(path)
        if (
            existing.get("evaluation_identity") != manifest["evaluation_identity"]
            or existing.get("evaluation_identity")
            != _identity(existing, "evaluation_identity")
        ):
            raise ContractError("existing native agreement output has different inputs")
        return
    output_dir.mkdir(parents=True)
    write_json_atomic(path, manifest)


def evaluate_native_structure_agreement(
    suite_path: str | Path,
    prediction_run: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    metrics_runtime_root: str | Path,
    *,
    retry_failed: bool = False,
    payload_loader: Callable[[Path, dict], dict] = _load_payload,
    native_extractor: Callable = extract_native_ca,
    prediction_parser: Callable = parse_prediction_ca,
    metric_function: Callable = compute_ca_metrics,
    metrics_runtime_document: dict | None = None,
) -> dict:
    suite_document_path = Path(suite_path).expanduser().resolve()
    verify_benchmark_suite_files(suite_document_path)
    suite = read_json(suite_document_path)
    if suite["source"]["split"] != "valid":
        raise ContractError("native agreement evaluation requires the valid split")
    prediction_dir = Path(prediction_run).expanduser().resolve()
    verify_esmfold2_benchmark_run(suite_document_path, prediction_dir, mode="full")
    prediction_manifest_path = prediction_dir / "run-manifest.json"
    prediction_manifest = read_json(prediction_manifest_path)
    source_dir = Path(dataset_dir).expanduser().resolve()
    dataset = _verify_dataset(source_dir, suite)
    if metrics_runtime_document is None:
        runtime = load_metrics_runtime_manifest(metrics_runtime_root)
    else:
        runtime = metrics_runtime_document
        _validate_metrics_runtime_document(runtime)
    records = list(suite["records"])
    index_rows = _selected_index_rows(
        source_dir, {record["source_chain_id"] for record in records}
    )
    _validate_selected_index_rows(source_dir, index_rows, dataset)

    manifest = {
        "schema_version": "protein-mrna.native-agreement-evaluation.v1",
        "evaluation_identity": "pending",
        "created_at_utc": utc_now(),
        "benchmark": {
            "benchmark_id": suite["benchmark_id"],
            "suite_path": str(suite_document_path),
            "suite_sha256": sha256_file(suite_document_path),
            "records": len(records),
            "split": "valid",
        },
        "prediction": {
            "run_dir": str(prediction_dir),
            "run_identity": prediction_manifest["run_identity"],
            "manifest_sha256": sha256_file(prediction_manifest_path),
        },
        "native_dataset": dataset,
        "metrics_runtime_identity": runtime["runtime_identity"],
        "implementation": {
            "module": "protein_mrna_pipeline.structure_agreement",
            "module_sha256": sha256_file(Path(__file__).resolve()),
        },
        "metrics": {
            "library": "Biotite 1.6.0",
            "correspondence": "exact_sequence_position",
            "atoms": "C-alpha",
            "superimposition": "global_Kabsch_over_observed_native_CA",
            "lddt": {
                "inclusion_radius_angstrom": 15.0,
                "distance_bins_angstrom": [0.5, 1.0, 2.0, 4.0],
            },
            "tm_score_normalizations": ["observed_native_CA", "full_sequence"],
        },
    }
    manifest["evaluation_identity"] = _identity(manifest, "evaluation_identity")
    destination = Path(output_dir).expanduser().resolve()
    _initialize_output(destination, manifest)

    pending = []
    for record in records:
        existing = _valid_record_result(
            destination, record, manifest["evaluation_identity"]
        )
        if existing is None or (retry_failed and existing["status"] == "failed"):
            pending.append(record)
    for index, record in enumerate(pending, 1):
        record_id = record["benchmark_record_id"]
        started_at = utc_now()
        started = time.monotonic()
        try:
            index_row = index_rows[record["source_chain_id"]]
            if int(index_row.get("sequence_length", -1)) != int(record["length"]):
                raise ContractError(f"native index length mismatch: {record_id}")
            payload = payload_loader(source_dir, index_row)
            native = native_extractor(
                payload, record["source_chain_id"], record["sequence"]
            )
            prediction_result = read_json(
                prediction_dir / "records" / record_id / "result.json"
            )
            if (
                prediction_result.get("schema_version")
                != "protein-mrna.esmfold2-result.v1"
                or prediction_result.get("run_identity")
                != prediction_manifest["run_identity"]
                or prediction_result.get("benchmark_record_id") != record_id
                or prediction_result.get("sequence_sha256")
                != record["sequence_sha256"]
                or prediction_result.get("status") != "succeeded"
            ):
                raise ContractError(f"prediction result identity mismatch: {record_id}")
            prediction_path = prediction_dir / prediction_result["artifact"]["path"]
            if sha256_file(prediction_path) != prediction_result["artifact"]["sha256"]:
                raise ContractError(f"prediction PDB checksum mismatch: {record_id}")
            prediction_ca = prediction_parser(prediction_path, record["sequence"])
            metrics = metric_function(
                native["ca_coordinates"],
                prediction_ca,
                native["ca_mask"],
                record["sequence"],
            )
            metrics.update(
                {
                    "native_ca_coverage": native["ca_coverage"],
                    "native_complete_backbone_coverage": native[
                        "complete_backbone_coverage"
                    ],
                }
            )
            if set(metrics) != set(METRIC_FIELDS) or not all(
                math.isfinite(float(value)) for value in metrics.values()
            ):
                raise ContractError(f"invalid native agreement metric set: {record_id}")
            metrics = {name: float(metrics[name]) for name in METRIC_FIELDS}
            result = {
                "schema_version": "protein-mrna.native-agreement-record.v1",
                "result_identity": "pending",
                "evaluation_identity": manifest["evaluation_identity"],
                "benchmark_record_id": record_id,
                "source_chain_id": record["source_chain_id"],
                "sequence_sha256": record["sequence_sha256"],
                "sequence_length": record["length"],
                "length_bin": record["length_bin"],
                "status": "succeeded",
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.monotonic() - started,
                "native": {
                    "ca_residues": native["ca_residues"],
                    "complete_backbone_residues": native[
                        "complete_backbone_residues"
                    ],
                    "source_pdb_id": native["source_pdb_id"],
                    "source_assembly_id": native["source_assembly_id"],
                    "source_mmcif_chain_id": native["source_mmcif_chain_id"],
                    "source_entity_id": native["source_entity_id"],
                    "shard": index_row["shard"],
                    "payload_offset": index_row["offset"],
                    "payload_size": index_row["size"],
                },
                "prediction": {
                    "pdb_path": str(prediction_path.relative_to(prediction_dir)),
                    "pdb_sha256": prediction_result["artifact"]["sha256"],
                },
                "prediction_confidence": prediction_result["metrics"],
                "metrics": metrics,
            }
        except Exception as error:
            result = {
                "schema_version": "protein-mrna.native-agreement-record.v1",
                "result_identity": "pending",
                "evaluation_identity": manifest["evaluation_identity"],
                "benchmark_record_id": record_id,
                "source_chain_id": record["source_chain_id"],
                "sequence_sha256": record["sequence_sha256"],
                "sequence_length": record["length"],
                "length_bin": record["length_bin"],
                "status": "failed",
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.monotonic() - started,
                "error": {"type": type(error).__name__, "message": str(error)[:4000]},
            }
            print(f"{record_id} failed: {type(error).__name__}: {error}", file=sys.stderr)
        result["result_identity"] = _identity(result, "result_identity")
        write_json_atomic(_record_result_path(destination, record_id), result)
        summary = _summarize(destination, records, manifest["evaluation_identity"])
        print(
            f"[{index}/{len(pending)}] {record_id}: {result['status']} "
            f"completed={summary['records']['succeeded']} "
            f"failed={summary['records']['failed']}",
            flush=True,
        )
    return _summarize(destination, records, manifest["evaluation_identity"])
