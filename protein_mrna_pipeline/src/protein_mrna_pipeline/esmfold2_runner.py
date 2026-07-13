"""Pinned, resumable ESMFold2-Fast engineering benchmark runner."""

from __future__ import annotations

import gc
import importlib.metadata
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from .benchmark import sha256_file, verify_benchmark_suite_files
from .contracts import ContractError, document_sha256, read_json
from .run_store import write_json_atomic


BIOHUB_TRANSFORMERS_COMMIT = "ef32577f55da19a4989cd7b22e004dc43a4998cb"
BIOHUB_TRANSFORMERS_VERSION = "4.57.6"
ESMFOLD2_FAST_REPOSITORY = "biohub/ESMFold2-Fast"
ESMFOLD2_FAST_REVISION = "b28d8ace5e05e61e5bec1e6820cfd3e221819d12"
ESMC_6B_REPOSITORY = "biohub/ESMC-6B"
ESMC_6B_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
DEFAULT_PARAMETERS = {
    "chunk_size": 64,
    "num_diffusion_samples": 1,
    "num_loops": 3,
    "num_sampling_steps": 50,
}


@dataclass(frozen=True)
class WeightFile:
    model: str
    path: str
    bytes: int
    sha256: str


WEIGHT_FILES = (
    WeightFile(
        "ESMFold2-Fast",
        "model.safetensors",
        755416924,
        "60ca19f2898188beba92944365f7b909efd9c99212f5018af75cc47cd9a6184a",
    ),
    WeightFile(
        "ESMC-6B",
        "model-00001-of-00006.safetensors",
        4864457920,
        "bd90149ff223e6ac1a0cac6147a5ae0df20d3a21df4f65356a1f19cd14f4aa8a",
    ),
    WeightFile(
        "ESMC-6B",
        "model-00002-of-00006.safetensors",
        4971211344,
        "f75e2144d8269fe2eb4b3e0823fb089b94f176d8024153e85b8fb573a42294fa",
    ),
    WeightFile(
        "ESMC-6B",
        "model-00003-of-00006.safetensors",
        4863752992,
        "f699f01ecc9691d9c6470492765fe54b8b5d2e9f277c139e89427433ffdfe0b2",
    ),
    WeightFile(
        "ESMC-6B",
        "model-00004-of-00006.safetensors",
        4971211344,
        "46add1b7be098bbfdc3073884851ba3057f1b33ea23a158b650a37007dabd13d",
    ),
    WeightFile(
        "ESMC-6B",
        "model-00005-of-00006.safetensors",
        4863752992,
        "1e1cb62f060a34e18f54a31a76683ef888b8cec59e73315f5b31d25d45a1f88c",
    ),
    WeightFile(
        "ESMC-6B",
        "model-00006-of-00006.safetensors",
        873762296,
        "56c73e13ae96e777ce65eee99364056069ef93b646470f352f83c5f1037b1b18",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity(document: dict, identity_field: str) -> str:
    payload = dict(document)
    payload.pop(identity_field, None)
    payload.pop("created_at_utc", None)
    return document_sha256(payload)


def _weight_manifest() -> list[dict]:
    return [
        {
            "model": item.model,
            "path": f"models/{item.model}/{item.path}",
            "bytes": item.bytes,
            "sha256": item.sha256,
        }
        for item in WEIGHT_FILES
    ]


def _check_weight_files(runtime_root: Path, verify_hashes: bool) -> None:
    for item in WEIGHT_FILES:
        path = runtime_root / "models" / item.model / item.path
        if not path.is_file():
            raise ContractError(f"ESMFold2 runtime weight not found: {path}")
        observed_size = path.stat().st_size
        if observed_size != item.bytes:
            raise ContractError(
                f"ESMFold2 runtime weight size mismatch for {path}: "
                f"expected={item.bytes} observed={observed_size}"
            )
        if verify_hashes:
            observed_hash = sha256_file(path)
            if observed_hash != item.sha256:
                raise ContractError(
                    f"ESMFold2 runtime weight SHA256 mismatch for {path}: "
                    f"expected={item.sha256} observed={observed_hash}"
                )


def create_runtime_manifest(runtime_root: str | Path, verify_hashes: bool = True) -> dict:
    root = Path(runtime_root).expanduser().resolve()
    _check_weight_files(root, verify_hashes=verify_hashes)
    try:
        transformers_version = importlib.metadata.version("transformers")
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise ContractError(f"ESMFold2 runtime dependency is missing: {error}") from error
    if transformers_version != BIOHUB_TRANSFORMERS_VERSION:
        raise ContractError(
            "unexpected transformers version in ESMFold2 runtime: "
            f"expected={BIOHUB_TRANSFORMERS_VERSION} observed={transformers_version}"
        )
    try:
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: F401
    except Exception as error:
        raise ContractError(f"Biohub ESMFold2Model import failed: {error}") from error

    dependency_versions = {}
    for distribution in (
        "huggingface-hub",
        "numpy",
        "safetensors",
        "tokenizers",
        "torch",
        "transformers",
    ):
        try:
            dependency_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution] = None

    manifest = {
        "schema_version": "protein-mrna.esmfold2-runtime.v1",
        "runtime_identity": "pending",
        "created_at_utc": utc_now(),
        "runtime_root": str(root),
        "source": {
            "repository": "https://github.com/Biohub/transformers.git",
            "revision": BIOHUB_TRANSFORMERS_COMMIT,
            "transformers_version": transformers_version,
        },
        "models": {
            "structure": {
                "repository": ESMFOLD2_FAST_REPOSITORY,
                "revision": ESMFOLD2_FAST_REVISION,
                "local_path": "models/ESMFold2-Fast",
            },
            "language_model": {
                "repository": ESMC_6B_REPOSITORY,
                "revision": ESMC_6B_REVISION,
                "local_path": "models/ESMC-6B",
                "precision": "bfloat16",
            },
            "weight_files": _weight_manifest(),
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch_version,
            "distributions": dependency_versions,
        },
    }
    manifest["runtime_identity"] = _identity(manifest, "runtime_identity")
    write_json_atomic(root / "runtime-manifest.json", manifest)
    return manifest


def _validate_runtime_document(document: dict) -> None:
    if document.get("schema_version") != "protein-mrna.esmfold2-runtime.v1":
        raise ContractError("unexpected ESMFold2 runtime manifest schema")
    if document.get("runtime_identity") != _identity(document, "runtime_identity"):
        raise ContractError("ESMFold2 runtime manifest identity mismatch")
    source = document.get("source", {})
    if source.get("revision") != BIOHUB_TRANSFORMERS_COMMIT:
        raise ContractError("ESMFold2 runtime source revision is not pinned")
    if source.get("transformers_version") != BIOHUB_TRANSFORMERS_VERSION:
        raise ContractError("ESMFold2 runtime transformers version is not pinned")
    models = document.get("models", {})
    structure = models.get("structure", {})
    language_model = models.get("language_model", {})
    if (
        structure.get("repository") != ESMFOLD2_FAST_REPOSITORY
        or structure.get("revision") != ESMFOLD2_FAST_REVISION
        or language_model.get("repository") != ESMC_6B_REPOSITORY
        or language_model.get("revision") != ESMC_6B_REVISION
        or models.get("weight_files") != _weight_manifest()
    ):
        raise ContractError("ESMFold2 runtime model identity does not match the pin")


def load_runtime_manifest(
    runtime_root: str | Path,
    *,
    verify_hashes: bool = False,
) -> dict:
    root = Path(runtime_root).expanduser().resolve()
    manifest_path = root / "runtime-manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"ESMFold2 runtime manifest not found: {manifest_path}")
    document = read_json(manifest_path)
    _validate_runtime_document(document)
    if document.get("runtime_root") != str(root):
        raise ContractError("ESMFold2 runtime root differs from its manifest")
    _check_weight_files(root, verify_hashes=verify_hashes)
    try:
        observed = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as error:
        raise ContractError("transformers is not installed in the active runtime") from error
    if observed != BIOHUB_TRANSFORMERS_VERSION:
        raise ContractError(
            f"active transformers version differs from runtime manifest: {observed}"
        )
    return document


def select_benchmark_records(suite: dict, mode: str) -> list[dict]:
    if mode == "full":
        return list(suite["records"])
    if mode != "smoke":
        raise ContractError(f"unknown ESMFold2 benchmark mode: {mode}")
    selected = []
    labels = [item["label"] for item in suite["selection"]["length_bins"]]
    for label in labels:
        candidates = [row for row in suite["records"] if row["length_bin"] == label]
        if not candidates:
            raise ContractError(f"benchmark length bin has no records: {label}")
        selected.append(
            max(candidates, key=lambda row: (row["length"], row["benchmark_record_id"]))
        )
    return selected


@dataclass
class FoldOutput:
    pdb_text: str
    metrics: dict[str, float]
    peak_gpu_memory_allocated_bytes: int
    peak_gpu_memory_reserved_bytes: int


class FoldBackend(Protocol):
    load_seconds: float

    def fold(self, sequence: str, parameters: dict, seed: int) -> FoldOutput: ...

    def recover_after_failure(self) -> None: ...


class ESMFold2FastBackend:
    """Lazy Biohub runtime binding; importing this module does not import torch."""

    def __init__(self, runtime_root: Path):
        started = time.monotonic()
        try:
            import torch
            from transformers.models.esmc.modeling_esmc import ESMCModel
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
        except Exception as error:
            raise ContractError(f"cannot import the pinned ESMFold2 runtime: {error}") from error
        if not torch.cuda.is_available():
            raise ContractError("ESMFold2-Fast requires an available CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise ContractError("ESMC-6B bfloat16 inference requires a bf16-capable GPU")

        self.torch = torch
        self.device = torch.device("cuda:0")
        fast_path = runtime_root / "models/ESMFold2-Fast"
        esmc_path = runtime_root / "models/ESMC-6B"
        print(f"Loading ESMFold2-Fast from {fast_path}", file=sys.stderr, flush=True)
        model = ESMFold2Model.from_pretrained(
            fast_path,
            load_esmc=False,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        print(f"Loading ESMC-6B in bfloat16 from {esmc_path}", file=sys.stderr, flush=True)
        esmc = ESMCModel.from_pretrained(
            esmc_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        esmc.eval()
        for parameter in esmc.parameters():
            parameter.requires_grad_(False)
        model._esmc = esmc
        model._esmc_fp8 = False
        model.set_kernel_backend(None)
        model.set_chunk_size(DEFAULT_PARAMETERS["chunk_size"])
        self.model = model.eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.synchronize(self.device)
        self.load_seconds = time.monotonic() - started

    @staticmethod
    def _finite_scalar(value, label: str) -> float:
        result = float(value.detach().float().mean().cpu())
        if not math.isfinite(result):
            raise RuntimeError(f"ESMFold2 returned non-finite {label}: {result}")
        return result

    def fold(self, sequence: str, parameters: dict, seed: int) -> FoldOutput:
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model.set_chunk_size(int(parameters["chunk_size"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            output = self.model.infer_protein(
                sequence,
                num_diffusion_samples=int(parameters["num_diffusion_samples"]),
                num_loops=int(parameters["num_loops"]),
                num_sampling_steps=int(parameters["num_sampling_steps"]),
            )
            torch.cuda.synchronize(self.device)
            pdb_text = self.model.output_to_pdb(output)
            metrics = {
                "mean_plddt": self._finite_scalar(output["plddt"], "pLDDT"),
                "ptm": self._finite_scalar(output["ptm"], "pTM"),
            }
            if "complex_plddt" in output:
                metrics["complex_plddt"] = self._finite_scalar(
                    output["complex_plddt"], "complex pLDDT"
                )
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
            del output
        if not pdb_text.strip() or "ATOM" not in pdb_text:
            raise RuntimeError("ESMFold2 produced an empty or invalid PDB artifact")
        return FoldOutput(pdb_text, metrics, peak_allocated, peak_reserved)

    def recover_after_failure(self) -> None:
        gc.collect()
        self.torch.cuda.empty_cache()


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="ascii")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _result_path(output_dir: Path, record: dict) -> Path:
    return output_dir / "records" / record["benchmark_record_id"] / "result.json"


def _valid_result(
    output_dir: Path,
    record: dict,
    run_identity: str,
) -> dict | None:
    result_path = _result_path(output_dir, record)
    if not result_path.is_file():
        return None
    try:
        result = read_json(result_path)
    except ContractError:
        return None
    if (
        result.get("schema_version") != "protein-mrna.esmfold2-result.v1"
        or result.get("run_identity") != run_identity
        or result.get("benchmark_record_id") != record["benchmark_record_id"]
        or result.get("sequence_sha256") != record["sequence_sha256"]
        or result.get("status") not in {"succeeded", "failed"}
    ):
        return None
    if result["status"] == "failed":
        return result
    artifact = result.get("artifact", {})
    expected_relative = f"records/{record['benchmark_record_id']}/prediction.pdb"
    if artifact.get("path") != expected_relative:
        return None
    artifact_path = output_dir / expected_relative
    if not artifact_path.is_file() or artifact_path.stat().st_size != artifact.get("bytes"):
        return None
    if sha256_file(artifact_path) != artifact.get("sha256"):
        return None
    return result


def _build_run_manifest(
    suite: dict,
    suite_path: Path,
    selected: list[dict],
    runtime: dict,
    mode: str,
    parameters: dict,
    seed: int,
) -> dict:
    manifest = {
        "schema_version": "protein-mrna.esmfold2-benchmark-run.v1",
        "run_identity": "pending",
        "created_at_utc": utc_now(),
        "benchmark": {
            "benchmark_id": suite["benchmark_id"],
            "suite_path": str(suite_path),
            "suite_sha256": sha256_file(suite_path),
            "source_split": suite["source"]["split"],
            "selected_record_ids": [row["benchmark_record_id"] for row in selected],
        },
        "runtime_identity": runtime["runtime_identity"],
        "model": {
            "name": "ESMFold2-Fast",
            "source_revision": BIOHUB_TRANSFORMERS_COMMIT,
            "structure_revision": ESMFOLD2_FAST_REVISION,
            "language_model_revision": ESMC_6B_REVISION,
            "weight_files": _weight_manifest(),
        },
        "execution": {
            "mode": mode,
            "device": "cuda:0",
            "sequential": True,
            "seed": seed,
            "parameters": parameters,
        },
        "limitations": [
            "Engineering valid-split benchmark; not a biological design target.",
            "Predicted structures and confidence values are computational hypotheses.",
        ],
    }
    manifest["run_identity"] = _identity(manifest, "run_identity")
    return manifest


def _initialize_run(output_dir: Path, manifest: dict) -> None:
    manifest_path = output_dir / "run-manifest.json"
    if output_dir.exists():
        if not manifest_path.is_file():
            raise ContractError(
                f"existing ESMFold2 output has no run manifest: {output_dir}"
            )
        existing = read_json(manifest_path)
        if (
            existing.get("run_identity") != manifest["run_identity"]
            or existing.get("run_identity") != _identity(existing, "run_identity")
        ):
            raise ContractError(
                "existing ESMFold2 output was created with different inputs or parameters"
            )
        return
    output_dir.mkdir(parents=True)
    write_json_atomic(manifest_path, manifest)


def _summarize(
    output_dir: Path,
    selected: list[dict],
    run_identity: str,
    model_load_seconds: float,
) -> dict:
    results = [
        result
        for record in selected
        if (result := _valid_result(output_dir, record, run_identity)) is not None
    ]
    succeeded = [result for result in results if result["status"] == "succeeded"]
    failed = [result for result in results if result["status"] == "failed"]
    pending = len(selected) - len(results)
    record_seconds = sum(float(result["runtime_seconds"]) for result in results)
    success_seconds = sum(float(result["runtime_seconds"]) for result in succeeded)
    previous_load_seconds = 0.0
    previous_summary_path = output_dir / "summary.json"
    if previous_summary_path.is_file():
        try:
            previous = read_json(previous_summary_path)
            previous_load_seconds = float(
                previous.get("timing", {}).get("model_load_seconds_max_observed", 0.0)
            )
        except (ContractError, TypeError, ValueError):
            previous_load_seconds = 0.0
    status = "passed" if len(succeeded) == len(selected) else "failed" if failed else "running"
    summary = {
        "schema_version": "protein-mrna.esmfold2-benchmark-summary.v1",
        "run_identity": run_identity,
        "updated_at_utc": utc_now(),
        "status": status,
        "records": {
            "selected": len(selected),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "pending": pending,
        },
        "timing": {
            "model_load_seconds_this_process": model_load_seconds,
            "model_load_seconds_max_observed": max(
                previous_load_seconds, model_load_seconds
            ),
            "record_runtime_seconds": record_seconds,
            "mean_seconds_per_success": (
                success_seconds / len(succeeded) if succeeded else None
            ),
        },
        "peak_gpu_memory_allocated_bytes": max(
            (int(result.get("peak_gpu_memory_allocated_bytes", 0)) for result in succeeded),
            default=0,
        ),
        "peak_gpu_memory_reserved_bytes": max(
            (int(result.get("peak_gpu_memory_reserved_bytes", 0)) for result in succeeded),
            default=0,
        ),
        "result_paths": [
            str(_result_path(output_dir, record).relative_to(output_dir))
            for record in selected
            if _result_path(output_dir, record).is_file()
        ],
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def verify_esmfold2_benchmark_run(
    suite_path: str | Path,
    output_dir: str | Path,
    *,
    mode: str,
) -> dict:
    suite_document_path = Path(suite_path).expanduser().resolve()
    verify_benchmark_suite_files(suite_document_path)
    suite = read_json(suite_document_path)
    selected = select_benchmark_records(suite, mode)
    destination = Path(output_dir).expanduser().resolve()
    manifest_path = destination / "run-manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"ESMFold2 run manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    run_identity = manifest.get("run_identity")
    if (
        manifest.get("schema_version") != "protein-mrna.esmfold2-benchmark-run.v1"
        or run_identity != _identity(manifest, "run_identity")
        or manifest.get("benchmark", {}).get("benchmark_id") != suite["benchmark_id"]
        or manifest.get("benchmark", {}).get("suite_sha256") != sha256_file(suite_document_path)
        or manifest.get("execution", {}).get("mode") != mode
        or manifest.get("benchmark", {}).get("selected_record_ids")
        != [record["benchmark_record_id"] for record in selected]
    ):
        raise ContractError("ESMFold2 run manifest does not match the requested benchmark")
    results = [
        _valid_result(destination, record, run_identity)
        for record in selected
    ]
    invalid = [
        record["benchmark_record_id"]
        for record, result in zip(selected, results)
        if result is None or result["status"] != "succeeded"
    ]
    if invalid:
        raise ContractError(
            f"ESMFold2 run is incomplete or has invalid artifacts: {invalid}"
        )
    return {
        "benchmark_id": suite["benchmark_id"],
        "mode": mode,
        "output_dir": str(destination),
        "records": len(results),
        "run_identity": run_identity,
        "status": "passed",
    }


def run_esmfold2_benchmark(
    suite_path: str | Path,
    output_dir: str | Path,
    runtime_root: str | Path,
    *,
    mode: str = "smoke",
    seed: int = 42,
    parameters: dict | None = None,
    retry_failed: bool = False,
    backend_factory: Callable[[Path], FoldBackend] = ESMFold2FastBackend,
    runtime_document: dict | None = None,
) -> dict:
    if seed < 0:
        raise ContractError("ESMFold2 benchmark seed must be non-negative")
    suite_document_path = Path(suite_path).expanduser().resolve()
    verify_benchmark_suite_files(suite_document_path)
    suite = read_json(suite_document_path)
    selected = select_benchmark_records(suite, mode)
    resolved_parameters = dict(DEFAULT_PARAMETERS)
    if parameters:
        resolved_parameters.update(parameters)
    for name in ("chunk_size", "num_diffusion_samples", "num_loops", "num_sampling_steps"):
        value = resolved_parameters.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ContractError(f"ESMFold2 parameter {name} must be a positive integer")
    if resolved_parameters["num_diffusion_samples"] != 1:
        raise ContractError("the bounded benchmark requires num_diffusion_samples=1")

    root = Path(runtime_root).expanduser().resolve()
    if runtime_document is None:
        runtime = load_runtime_manifest(root)
    else:
        _validate_runtime_document(runtime_document)
        runtime = runtime_document
    destination = Path(output_dir).expanduser().resolve()
    manifest = _build_run_manifest(
        suite,
        suite_document_path,
        selected,
        runtime,
        mode,
        resolved_parameters,
        seed,
    )
    _initialize_run(destination, manifest)

    pending = []
    for record in selected:
        result = _valid_result(destination, record, manifest["run_identity"])
        if result is None or (retry_failed and result["status"] == "failed"):
            pending.append(record)
    if not pending:
        return _summarize(destination, selected, manifest["run_identity"], 0.0)

    backend = backend_factory(root)
    load_seconds = float(getattr(backend, "load_seconds", 0.0))
    for index, record in enumerate(pending, 1):
        record_id = record["benchmark_record_id"]
        record_dir = destination / "records" / record_id
        artifact_path = record_dir / "prediction.pdb"
        artifact_path.unlink(missing_ok=True)
        started_at = utc_now()
        started = time.monotonic()
        print(
            f"[{index}/{len(pending)}] folding {record_id} "
            f"length={record['length']} bin={record['length_bin']}",
            flush=True,
        )
        try:
            folded = backend.fold(record["sequence"], resolved_parameters, seed)
            runtime_seconds = time.monotonic() - started
            _write_text_atomic(artifact_path, folded.pdb_text)
            result = {
                "schema_version": "protein-mrna.esmfold2-result.v1",
                "run_identity": manifest["run_identity"],
                "benchmark_record_id": record_id,
                "sequence_sha256": record["sequence_sha256"],
                "length": record["length"],
                "length_bin": record["length_bin"],
                "status": "succeeded",
                "seed": seed,
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "runtime_seconds": runtime_seconds,
                "parameters": resolved_parameters,
                "metrics": folded.metrics,
                "peak_gpu_memory_allocated_bytes": folded.peak_gpu_memory_allocated_bytes,
                "peak_gpu_memory_reserved_bytes": folded.peak_gpu_memory_reserved_bytes,
                "artifact": {
                    "path": str(artifact_path.relative_to(destination)),
                    "media_type": "chemical/x-pdb",
                    "bytes": artifact_path.stat().st_size,
                    "sha256": sha256_file(artifact_path),
                },
            }
        except Exception as error:
            runtime_seconds = time.monotonic() - started
            result = {
                "schema_version": "protein-mrna.esmfold2-result.v1",
                "run_identity": manifest["run_identity"],
                "benchmark_record_id": record_id,
                "sequence_sha256": record["sequence_sha256"],
                "length": record["length"],
                "length_bin": record["length_bin"],
                "status": "failed",
                "seed": seed,
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "runtime_seconds": runtime_seconds,
                "parameters": resolved_parameters,
                "error": {"type": type(error).__name__, "message": str(error)[:4000]},
            }
            print(f"{record_id} failed: {type(error).__name__}: {error}", file=sys.stderr)
        write_json_atomic(_result_path(destination, record), result)
        backend.recover_after_failure()
        summary = _summarize(
            destination,
            selected,
            manifest["run_identity"],
            load_seconds,
        )
        print(
            f"progress: succeeded={summary['records']['succeeded']} "
            f"failed={summary['records']['failed']} pending={summary['records']['pending']}",
            flush=True,
        )
    return _summarize(destination, selected, manifest["run_identity"], load_seconds)
