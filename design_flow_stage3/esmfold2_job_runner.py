#!/usr/bin/env python3
"""Run checksum-bound design-flow candidates through pinned ESMFold2-Fast."""

from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import time
from typing import Callable

from protein_mrna_pipeline.benchmark import sha256_file
from protein_mrna_pipeline.contracts import (
    ContractError,
    document_sha256,
    read_json,
    text_sha256,
)
from protein_mrna_pipeline.esmfold2_runner import (
    BIOHUB_TRANSFORMERS_COMMIT,
    DEFAULT_PARAMETERS,
    ESMC_6B_REVISION,
    ESMFOLD2_FAST_REVISION,
    ESMFold2FastBackend,
    FoldBackend,
    load_runtime_manifest,
)
from protein_mrna_pipeline.run_store import write_json_atomic


JOB_SCHEMA = "vaxflow.esmfold2-job.v1"
RUN_SCHEMA = "vaxflow.esmfold2-run.v1"
RESULT_SCHEMA = "vaxflow.esmfold2-result.v1"
SUMMARY_SCHEMA = "vaxflow.esmfold2-summary.v1"
CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
JOB_FILES = frozenset({"job-manifest.json", "sequences.fasta"})


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _identity(document: dict, identity_field: str) -> str:
    payload = dict(document)
    payload.pop(identity_field, None)
    payload.pop("created_at_utc", None)
    return document_sha256(payload)


def _wrap_fasta(sequence: str, width: int = 80) -> str:
    return "\n".join(
        sequence[offset : offset + width]
        for offset in range(0, len(sequence), width)
    )


def job_fasta_bytes(records: list[dict]) -> bytes:
    text = "".join(
        f">{record['candidate_id']} key={record['candidate_key']} "
        f"length={record['length']}\n{_wrap_fasta(record['sequence'])}\n"
        for record in records
    )
    return text.encode("ascii")


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _validate_requested_model(model: object) -> None:
    expected = {
        "name": "ESMFold2-Fast",
        "source_revision": BIOHUB_TRANSFORMERS_COMMIT,
        "structure_revision": ESMFOLD2_FAST_REVISION,
        "language_model_revision": ESMC_6B_REVISION,
    }
    if model != expected:
        raise ContractError(
            "job requests an ESMFold2 model identity that differs from the pinned worker"
        )


def _validate_parameters(execution: object) -> dict[str, int]:
    if not isinstance(execution, dict):
        raise ContractError("job execution must be an object")
    if execution.get("seed") != 42:
        raise ContractError("exploratory Stage 3 requires seed=42")
    parameters = execution.get("parameters")
    if not isinstance(parameters, dict):
        raise ContractError("job execution.parameters must be an object")
    expected_names = set(DEFAULT_PARAMETERS)
    if set(parameters) != expected_names:
        raise ContractError(
            f"job parameter names must be exactly {sorted(expected_names)}"
        )
    normalized = {
        name: _positive_int(parameters[name], f"execution.parameters.{name}")
        for name in sorted(expected_names)
    }
    if normalized["num_diffusion_samples"] != 1:
        raise ContractError("Stage 3 worker supports one diffusion sample per candidate")
    return normalized


def validate_job_directory(job_dir: str | Path) -> dict:
    root = Path(job_dir).expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"job directory not found: {root}")
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ContractError(f"job directory contains symlinks: {symlinks}")
    if present != JOB_FILES:
        raise ContractError(
            f"job directory must contain exactly {sorted(JOB_FILES)}; observed={sorted(present)}"
        )

    manifest_path = root / "job-manifest.json"
    job = read_json(manifest_path)
    if job.get("schema_version") != JOB_SCHEMA:
        raise ContractError(f"unexpected Stage 3 job schema: {job.get('schema_version')}")
    identity = job.get("job_identity")
    if not isinstance(identity, str) or identity != _identity(job, "job_identity"):
        raise ContractError("Stage 3 job identity mismatch")
    source = job.get("source")
    required_source = {
        "project_id",
        "stage2_run_id",
        "stage2_artifact_index_sha256",
        "candidate_batch_sha256",
    }
    if not isinstance(source, dict) or not all(
        isinstance(source.get(name), str) and source[name]
        for name in required_source
    ):
        raise ContractError("job source lineage is incomplete")
    _validate_requested_model(job.get("model"))
    parameters = _validate_parameters(job.get("execution"))

    maximum_length = _positive_int(job.get("maximum_sequence_length"), "maximum_sequence_length")
    records = job.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("job records must be a non-empty array")
    candidate_ids: set[str] = set()
    candidate_keys: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ContractError(f"job record {index} must be an object")
        candidate_id = record.get("candidate_id")
        candidate_key = record.get("candidate_key")
        if not isinstance(candidate_id, str) or not IDENTIFIER_PATTERN.fullmatch(candidate_id):
            raise ContractError(f"record {index} has an invalid candidate_id")
        if not isinstance(candidate_key, str) or not IDENTIFIER_PATTERN.fullmatch(candidate_key):
            raise ContractError(f"record {index} has an invalid candidate_key")
        if candidate_id in candidate_ids or candidate_key in candidate_keys:
            raise ContractError(f"duplicate candidate ID or key: {candidate_id}/{candidate_key}")
        candidate_ids.add(candidate_id)
        candidate_keys.add(candidate_key)
        sequence = record.get("sequence")
        if (
            not isinstance(sequence, str)
            or not sequence
            or not set(sequence) <= CANONICAL_AMINO_ACIDS
        ):
            raise ContractError(f"record {candidate_id} has a non-canonical sequence")
        if record.get("length") != len(sequence) or len(sequence) > maximum_length:
            raise ContractError(
                f"record {candidate_id} length is inconsistent or exceeds {maximum_length}"
            )
        if record.get("sequence_sha256") != text_sha256(sequence):
            raise ContractError(f"record {candidate_id} sequence SHA256 mismatch")

    fasta = job_fasta_bytes(records)
    fasta_document = job.get("fasta")
    fasta_path = root / "sequences.fasta"
    if not isinstance(fasta_document, dict) or fasta_document != {
        "path": "sequences.fasta",
        "sha256": __import__("hashlib").sha256(fasta).hexdigest(),
        "bytes": len(fasta),
        "records": len(records),
    }:
        raise ContractError("job FASTA manifest is inconsistent")
    if fasta_path.read_bytes() != fasta:
        raise ContractError("job FASTA content differs from manifest records")
    return {"root": root, "manifest_path": manifest_path, "job": job, "parameters": parameters}


def _result_path(output_dir: Path, record: dict) -> Path:
    return output_dir / "records" / record["candidate_id"] / "result.json"


def _valid_result(
    output_dir: Path,
    record: dict,
    run_identity: str,
) -> dict | None:
    path = _result_path(output_dir, record)
    if not path.is_file():
        return None
    try:
        result = read_json(path)
    except ContractError:
        return None
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("run_identity") != run_identity
        or result.get("candidate_id") != record["candidate_id"]
        or result.get("candidate_key") != record["candidate_key"]
        or result.get("sequence_sha256") != record["sequence_sha256"]
        or result.get("status") not in {"succeeded", "failed"}
    ):
        return None
    if result["status"] == "failed":
        return result
    artifact = result.get("artifact")
    relative_path = f"records/{record['candidate_id']}/prediction.pdb"
    if not isinstance(artifact, dict) or artifact.get("path") != relative_path:
        return None
    artifact_path = output_dir / relative_path
    if (
        not artifact_path.is_file()
        or artifact_path.stat().st_size != artifact.get("bytes")
        or sha256_file(artifact_path) != artifact.get("sha256")
    ):
        return None
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not all(
        isinstance(metrics.get(name), (int, float))
        and not isinstance(metrics.get(name), bool)
        and math.isfinite(float(metrics[name]))
        for name in ("mean_plddt", "ptm")
    ):
        return None
    return result


def _build_run_manifest(job: dict, runtime: dict) -> dict:
    manifest = {
        "schema_version": RUN_SCHEMA,
        "run_identity": "pending",
        "created_at_utc": utc_now(),
        "job_identity": job["job_identity"],
        "job_manifest_sha256": document_sha256(job),
        "source": job["source"],
        "runtime_identity": runtime["runtime_identity"],
        "model": job["model"],
        "weight_files": runtime["models"]["weight_files"],
        "execution": {
            **job["execution"],
            "device": "cuda:0",
            "sequential": True,
        },
        "candidate_ids": [record["candidate_id"] for record in job["records"]],
        "limitations": [
            "Exploratory single-sequence structure predictions; not experimental folding evidence.",
            "No MSA, template, oligomer, membrane, glycosylation, or expression context is modeled.",
        ],
    }
    manifest["run_identity"] = _identity(manifest, "run_identity")
    return manifest


def _initialize_output(output_dir: Path, manifest: dict) -> None:
    manifest_path = output_dir / "run-manifest.json"
    if output_dir.exists():
        if not manifest_path.is_file():
            raise ContractError(f"existing output has no run manifest: {output_dir}")
        existing = read_json(manifest_path)
        if (
            existing.get("run_identity") != manifest["run_identity"]
            or existing.get("run_identity") != _identity(existing, "run_identity")
        ):
            raise ContractError("existing output was created for another Stage 3 job/runtime")
        return
    output_dir.mkdir(parents=True)
    write_json_atomic(manifest_path, manifest)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _summarize(output_dir: Path, records: list[dict], run_identity: str, load_seconds: float) -> dict:
    results = [
        result
        for record in records
        if (result := _valid_result(output_dir, record, run_identity)) is not None
    ]
    succeeded = [result for result in results if result["status"] == "succeeded"]
    failed = [result for result in results if result["status"] == "failed"]
    pending = len(records) - len(results)
    prior_load_seconds = 0.0
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        try:
            prior_load_seconds = float(
                read_json(summary_path).get("timing", {}).get(
                    "model_load_seconds_max_observed", 0.0
                )
            )
        except (ContractError, TypeError, ValueError):
            pass
    status = "passed" if len(succeeded) == len(records) else "failed" if failed else "running"
    success_seconds = sum(float(result["runtime_seconds"]) for result in succeeded)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "run_identity": run_identity,
        "updated_at_utc": utc_now(),
        "status": status,
        "records": {
            "selected": len(records),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "pending": pending,
        },
        "timing": {
            "model_load_seconds_this_process": load_seconds,
            "model_load_seconds_max_observed": max(prior_load_seconds, load_seconds),
            "record_runtime_seconds": sum(float(result["runtime_seconds"]) for result in results),
            "mean_seconds_per_success": success_seconds / len(succeeded) if succeeded else None,
        },
        "peak_gpu_memory_allocated_bytes": max(
            (int(result["peak_gpu_memory_allocated_bytes"]) for result in succeeded),
            default=0,
        ),
        "peak_gpu_memory_reserved_bytes": max(
            (int(result["peak_gpu_memory_reserved_bytes"]) for result in succeeded),
            default=0,
        ),
        "result_paths": [
            str(_result_path(output_dir, record).relative_to(output_dir))
            for record in records
            if _result_path(output_dir, record).is_file()
        ],
    }
    write_json_atomic(summary_path, summary)
    return summary


def run_job(
    job_dir: str | Path,
    output_dir: str | Path,
    runtime_root: str | Path,
    *,
    retry_failed: bool = False,
    backend_factory: Callable[[Path], FoldBackend] = ESMFold2FastBackend,
    runtime_document: dict | None = None,
) -> dict:
    validated = validate_job_directory(job_dir)
    job = validated["job"]
    runtime_path = Path(runtime_root).expanduser().resolve()
    runtime = runtime_document or load_runtime_manifest(runtime_path)
    destination = Path(output_dir).expanduser().resolve()
    manifest = _build_run_manifest(job, runtime)
    _initialize_output(destination, manifest)
    records = job["records"]
    pending = []
    for record in records:
        result = _valid_result(destination, record, manifest["run_identity"])
        if result is None or (retry_failed and result["status"] == "failed"):
            pending.append(record)
    if not pending:
        return _summarize(destination, records, manifest["run_identity"], 0.0)

    backend = backend_factory(runtime_path)
    load_seconds = float(getattr(backend, "load_seconds", 0.0))
    for index, record in enumerate(pending, 1):
        record_dir = destination / "records" / record["candidate_id"]
        pdb_path = record_dir / "prediction.pdb"
        pdb_path.unlink(missing_ok=True)
        started_at = utc_now()
        started = time.monotonic()
        print(
            f"[{index}/{len(pending)}] folding {record['candidate_id']} "
            f"key={record['candidate_key']} length={record['length']}",
            flush=True,
        )
        try:
            folded = backend.fold(
                record["sequence"],
                job["execution"]["parameters"],
                job["execution"]["seed"],
            )
            runtime_seconds = time.monotonic() - started
            _write_text_atomic(pdb_path, folded.pdb_text)
            result = {
                "schema_version": RESULT_SCHEMA,
                "run_identity": manifest["run_identity"],
                "candidate_id": record["candidate_id"],
                "candidate_key": record["candidate_key"],
                "sequence_sha256": record["sequence_sha256"],
                "length": record["length"],
                "status": "succeeded",
                "seed": job["execution"]["seed"],
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "runtime_seconds": runtime_seconds,
                "parameters": job["execution"]["parameters"],
                "metrics": folded.metrics,
                "peak_gpu_memory_allocated_bytes": folded.peak_gpu_memory_allocated_bytes,
                "peak_gpu_memory_reserved_bytes": folded.peak_gpu_memory_reserved_bytes,
                "artifact": {
                    "path": str(pdb_path.relative_to(destination)),
                    "media_type": "chemical/x-pdb",
                    "bytes": pdb_path.stat().st_size,
                    "sha256": sha256_file(pdb_path),
                },
            }
        except Exception as error:
            result = {
                "schema_version": RESULT_SCHEMA,
                "run_identity": manifest["run_identity"],
                "candidate_id": record["candidate_id"],
                "candidate_key": record["candidate_key"],
                "sequence_sha256": record["sequence_sha256"],
                "length": record["length"],
                "status": "failed",
                "seed": job["execution"]["seed"],
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.monotonic() - started,
                "parameters": job["execution"]["parameters"],
                "error": {"type": type(error).__name__, "message": str(error)[:4000]},
            }
            print(
                f"{record['candidate_id']} failed: {type(error).__name__}: {error}",
                file=__import__("sys").stderr,
                flush=True,
            )
        write_json_atomic(_result_path(destination, record), result)
        backend.recover_after_failure()
        summary = _summarize(destination, records, manifest["run_identity"], load_seconds)
        print(
            f"progress: succeeded={summary['records']['succeeded']} "
            f"failed={summary['records']['failed']} pending={summary['records']['pending']}",
            flush=True,
        )
    return _summarize(destination, records, manifest["run_identity"], load_seconds)


def verify_run(job_dir: str | Path, output_dir: str | Path) -> dict:
    validated = validate_job_directory(job_dir)
    job = validated["job"]
    destination = Path(output_dir).expanduser().resolve()
    manifest_path = destination / "run-manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"run manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("run_identity") != _identity(manifest, "run_identity")
        or manifest.get("job_identity") != job["job_identity"]
        or manifest.get("job_manifest_sha256") != document_sha256(job)
        or manifest.get("candidate_ids")
        != [record["candidate_id"] for record in job["records"]]
    ):
        raise ContractError("run manifest does not match the supplied Stage 3 job")
    invalid = [
        record["candidate_id"]
        for record in job["records"]
        if (result := _valid_result(destination, record, manifest["run_identity"])) is None
        or result["status"] != "succeeded"
    ]
    if invalid:
        raise ContractError(f"run is incomplete or contains invalid results: {invalid}")
    summary = read_json(destination / "summary.json")
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("run_identity") != manifest["run_identity"]
        or summary.get("status") != "passed"
        or summary.get("records", {}).get("succeeded") != len(job["records"])
    ):
        raise ContractError("run summary is incomplete or inconsistent")
    return {
        "status": "passed",
        "job_identity": job["job_identity"],
        "run_identity": manifest["run_identity"],
        "records": len(job["records"]),
        "output_dir": str(destination),
    }


def unpack_job_archive(archive: str | Path, destination: str | Path) -> dict:
    source = Path(archive).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise ContractError(f"job archive not found: {source}")
    if target.exists():
        return validate_job_directory(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ContractError(f"temporary extraction path already exists: {temporary}")
    temporary.mkdir()
    try:
        with tarfile.open(source, "r:gz") as bundle:
            members = bundle.getmembers()
            names = {member.name for member in members if member.isfile()}
            if names != JOB_FILES or any(
                member.issym()
                or member.islnk()
                or member.isdev()
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
                for member in members
            ):
                raise ContractError("job archive has unexpected or unsafe members")
            for member in members:
                if member.isdir():
                    continue
                source_handle = bundle.extractfile(member)
                if source_handle is None:
                    raise ContractError(f"cannot extract archive member: {member.name}")
                with source_handle, (temporary / member.name).open("wb") as output_handle:
                    shutil.copyfileobj(source_handle, output_handle)
        validated = validate_job_directory(temporary)
        os.replace(temporary, target)
        validated["root"] = target
        validated["manifest_path"] = target / "job-manifest.json"
        return validated
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def pack_results(job_dir: str | Path, output_dir: str | Path, archive: str | Path) -> dict:
    verified = verify_run(job_dir, output_dir)
    source = Path(output_dir).expanduser().resolve()
    destination = Path(archive).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with tarfile.open(temporary, "w:gz") as bundle:
            for path in sorted(source.rglob("*")):
                if path.is_symlink():
                    raise ContractError(f"result directory contains symlink: {path}")
                if path.is_file():
                    bundle.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **verified,
        "archive": str(destination),
        "archive_bytes": destination.stat().st_size,
        "archive_sha256": sha256_file(destination),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect-job")
    inspect_parser.add_argument("--job-dir", type=Path, required=True)
    unpack_parser = subparsers.add_parser("unpack-job")
    unpack_parser.add_argument("--archive", type=Path, required=True)
    unpack_parser.add_argument("--destination", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--job-dir", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--runtime-root", type=Path, required=True)
    run_parser.add_argument("--retry-failed", action="store_true")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--job-dir", type=Path, required=True)
    verify_parser.add_argument("--output-dir", type=Path, required=True)
    pack_parser = subparsers.add_parser("pack-results")
    pack_parser.add_argument("--job-dir", type=Path, required=True)
    pack_parser.add_argument("--output-dir", type=Path, required=True)
    pack_parser.add_argument("--archive", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "inspect-job":
            print(validate_job_directory(args.job_dir)["job"]["job_identity"])
        elif args.command == "unpack-job":
            print(unpack_job_archive(args.archive, args.destination)["job"]["job_identity"])
        elif args.command == "run":
            print(
                __import__("json").dumps(
                    run_job(
                        args.job_dir,
                        args.output_dir,
                        args.runtime_root,
                        retry_failed=args.retry_failed,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "verify":
            print(__import__("json").dumps(verify_run(args.job_dir, args.output_dir), indent=2, sort_keys=True))
        else:
            print(
                __import__("json").dumps(
                    pack_results(args.job_dir, args.output_dir, args.archive),
                    indent=2,
                    sort_keys=True,
                )
            )
        return 0
    except (ContractError, OSError, ValueError) as error:
        print(f"design-flow-stage3: {error}", file=__import__("sys").stderr)
        return 1
    finally:
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
