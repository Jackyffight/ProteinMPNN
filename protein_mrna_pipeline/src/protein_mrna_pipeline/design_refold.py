"""Resumable ESMFold2 refolding and dual-reference evaluation for pilot designs."""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

from .benchmark import sha256_file, verify_benchmark_suite_files
from .contracts import ContractError, read_json
from .esmfold2_runner import (
    DEFAULT_PARAMETERS,
    ESMFold2FastBackend,
    FoldBackend,
    _validate_runtime_document,
    _write_text_atomic,
    load_runtime_manifest,
    verify_esmfold2_benchmark_run,
)
from .proteinmpnn_pilot import (
    _identity,
    _validate_native_agreement,
    load_generated_pilot,
)
from .run_store import write_json_atomic, write_jsonl_atomic
from .structure_agreement import (
    _load_payload,
    _selected_index_rows,
    _validate_selected_index_rows,
    _validate_metrics_runtime_document,
    _verify_dataset,
    compute_ca_metrics,
    extract_native_ca,
    load_metrics_runtime_manifest,
    parse_prediction_ca,
    utc_now,
)


REFOLD_METRIC_FIELDS = (
    "sequence_recovery",
    "mutation_fraction",
    "sampled_nll",
    "experimental_ca_lddt",
    "experimental_ca_tm_score_resolved",
    "experimental_ca_tm_score_full_length",
    "experimental_ca_rmsd_angstrom",
    "delta_experimental_ca_lddt_vs_native_sequence",
    "delta_experimental_tm_resolved_vs_native_sequence",
    "delta_experimental_tm_full_length_vs_native_sequence",
    "delta_experimental_ca_rmsd_vs_native_sequence",
    "native_prediction_ca_lddt",
    "native_prediction_ca_tm_score",
    "native_prediction_ca_rmsd_angstrom",
    "refold_mean_plddt",
    "refold_ptm",
)


def _initialize_directory(path: Path, manifest_name: str, manifest: dict, field: str) -> None:
    manifest_path = path / manifest_name
    if path.exists():
        if not manifest_path.is_file():
            raise ContractError(f"existing output has no {manifest_name}: {path}")
        existing = read_json(manifest_path)
        if (
            existing.get(field) != manifest[field]
            or existing.get(field) != _identity(existing, field)
        ):
            raise ContractError(f"existing output has different inputs: {path}")
        return
    path.mkdir(parents=True)
    write_json_atomic(manifest_path, manifest)


def _refold_result_path(output_dir: Path, design_id: str) -> Path:
    return output_dir / "records" / design_id / "result.json"


def _valid_refold_result(
    output_dir: Path,
    design: dict,
    refold_identity: str,
) -> dict | None:
    path = _refold_result_path(output_dir, design["design_id"])
    if not path.is_file():
        return None
    try:
        result = read_json(path)
    except ContractError:
        return None
    if (
        result.get("schema_version") != "protein-mrna.esmfold2-design-refold.v1"
        or result.get("result_identity") != _identity(result, "result_identity")
        or result.get("refold_identity") != refold_identity
        or result.get("design_id") != design["design_id"]
        or result.get("design_identity") != design["design_identity"]
        or result.get("sequence_sha256") != design["sequence_sha256"]
        or result.get("sequence_length") != design["sequence_length"]
        or result.get("model_label") != design["model_label"]
        or result.get("sampling_seed") != design["seed"]
        or result.get("status") not in {"succeeded", "failed"}
    ):
        return None
    try:
        runtime_seconds = float(result["runtime_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(runtime_seconds) or runtime_seconds < 0.0:
        return None
    if result["status"] == "failed":
        return result
    metrics = result.get("metrics", {})
    try:
        confidence = [float(metrics[field]) for field in ("mean_plddt", "ptm")]
        peak_allocated = int(result["peak_gpu_memory_allocated_bytes"])
        peak_reserved = int(result["peak_gpu_memory_reserved_bytes"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(value) for value in confidence)
        or peak_allocated < 0
        or peak_reserved < peak_allocated
    ):
        return None
    artifact = result.get("artifact", {})
    expected_relative = f"records/{design['design_id']}/prediction.pdb"
    artifact_path = output_dir / expected_relative
    if (
        artifact.get("path") != expected_relative
        or not artifact_path.is_file()
        or artifact.get("media_type") != "chemical/x-pdb"
        or not isinstance(artifact.get("bytes"), int)
        or artifact.get("bytes") <= 0
        or artifact_path.stat().st_size != artifact.get("bytes")
        or sha256_file(artifact_path) != artifact.get("sha256")
    ):
        return None
    return result


def _summarize_refolds(
    output_dir: Path,
    designs: list[dict],
    refold_identity: str,
    model_load_seconds: float,
) -> dict:
    results = [
        result
        for design in designs
        if (
            result := _valid_refold_result(output_dir, design, refold_identity)
        )
        is not None
    ]
    succeeded = [result for result in results if result["status"] == "succeeded"]
    failed = [result for result in results if result["status"] == "failed"]
    pending = len(designs) - len(results)
    previous_load_seconds = 0.0
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        try:
            previous_load_seconds = float(
                read_json(summary_path)
                .get("timing", {})
                .get("model_load_seconds_max_observed", 0.0)
            )
        except (ContractError, TypeError, ValueError):
            previous_load_seconds = 0.0
    summary = {
        "schema_version": "protein-mrna.esmfold2-design-refold-summary.v1",
        "refold_identity": refold_identity,
        "updated_at_utc": utc_now(),
        "status": (
            "passed"
            if len(succeeded) == len(designs)
            else "failed" if failed else "running"
        ),
        "records": {
            "selected": len(designs),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "pending": pending,
        },
        "timing": {
            "model_load_seconds_this_process": model_load_seconds,
            "model_load_seconds_max_observed": max(
                previous_load_seconds, model_load_seconds
            ),
            "record_runtime_seconds": sum(
                float(result["runtime_seconds"]) for result in results
            ),
            "mean_seconds_per_success": (
                statistics.fmean(
                    float(result["runtime_seconds"]) for result in succeeded
                )
                if succeeded
                else None
            ),
        },
        "peak_gpu_memory_allocated_bytes": max(
            (
                int(result.get("peak_gpu_memory_allocated_bytes", 0))
                for result in succeeded
            ),
            default=0,
        ),
        "peak_gpu_memory_reserved_bytes": max(
            (
                int(result.get("peak_gpu_memory_reserved_bytes", 0))
                for result in succeeded
            ),
            default=0,
        ),
    }
    write_json_atomic(summary_path, summary)
    return summary


def load_completed_refolds(
    pilot_dir: str | Path,
    refold_dir: str | Path,
) -> tuple[dict, list[dict], dict, dict[str, dict]]:
    pilot_manifest, designs, _ = load_generated_pilot(pilot_dir)
    output_dir = Path(refold_dir).expanduser().resolve()
    manifest = read_json(output_dir / "refold-manifest.json")
    if (
        manifest.get("schema_version")
        != "protein-mrna.esmfold2-design-refold-run.v1"
        or manifest.get("refold_identity")
        != _identity(manifest, "refold_identity")
        or manifest.get("pilot_identity") != pilot_manifest["pilot_identity"]
        or manifest.get("designs_sha256")
        != sha256_file(Path(pilot_dir).resolve() / "designs.jsonl")
    ):
        raise ContractError("ESMFold2 design refold manifest identity mismatch")
    summary = read_json(output_dir / "summary.json")
    if (
        summary.get("schema_version")
        != "protein-mrna.esmfold2-design-refold-summary.v1"
        or summary.get("refold_identity") != manifest["refold_identity"]
        or summary.get("status") != "passed"
        or summary.get("records", {}).get("succeeded") != len(designs)
    ):
        raise ContractError("ESMFold2 design refold run is incomplete")
    results = {}
    for design in designs:
        result = _valid_refold_result(
            output_dir, design, manifest["refold_identity"]
        )
        if result is None or result["status"] != "succeeded":
            raise ContractError(f"invalid design refold result: {design['design_id']}")
        results[design["design_id"]] = result
    return pilot_manifest, designs, manifest, results


def run_design_refolds(
    pilot_dir: str | Path,
    output_dir: str | Path,
    runtime_root: str | Path,
    *,
    retry_failed: bool = False,
    seed: int = 42,
    backend_factory: Callable[[Path], FoldBackend] = ESMFold2FastBackend,
    runtime_document: dict | None = None,
) -> dict:
    if seed < 0:
        raise ContractError("ESMFold2 design refold seed must be non-negative")
    pilot_root = Path(pilot_dir).expanduser().resolve()
    pilot_manifest, designs, generation_summary = load_generated_pilot(pilot_root)
    runtime_path = Path(runtime_root).expanduser().resolve()
    runtime = (
        load_runtime_manifest(runtime_path)
        if runtime_document is None
        else runtime_document
    )
    if runtime_document is not None:
        _validate_runtime_document(runtime)
    parameters = dict(DEFAULT_PARAMETERS)
    manifest = {
        "schema_version": "protein-mrna.esmfold2-design-refold-run.v1",
        "refold_identity": "pending",
        "created_at_utc": utc_now(),
        "pilot_identity": pilot_manifest["pilot_identity"],
        "pilot_dir": str(pilot_root),
        "designs_sha256": generation_summary["designs_sha256"],
        "design_ids": [design["design_id"] for design in designs],
        "runtime_identity": runtime["runtime_identity"],
        "model": {
            "name": "ESMFold2-Fast",
            "structure_revision": runtime["models"]["structure"]["revision"],
            "language_model_revision": runtime["models"]["language_model"][
                "revision"
            ],
        },
        "execution": {
            "device": "cuda:0",
            "sequential": True,
            "seed": seed,
            "parameters": parameters,
        },
        "implementation": {
            "module": "protein_mrna_pipeline.design_refold",
            "module_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    manifest["refold_identity"] = _identity(manifest, "refold_identity")
    destination = Path(output_dir).expanduser().resolve()
    _initialize_directory(
        destination, "refold-manifest.json", manifest, "refold_identity"
    )
    pending = []
    for design in designs:
        existing = _valid_refold_result(
            destination, design, manifest["refold_identity"]
        )
        if existing is None or (retry_failed and existing["status"] == "failed"):
            pending.append(design)
    if not pending:
        return _summarize_refolds(
            destination, designs, manifest["refold_identity"], 0.0
        )

    backend = backend_factory(runtime_path)
    load_seconds = float(getattr(backend, "load_seconds", 0.0))
    for index, design in enumerate(pending, 1):
        design_id = design["design_id"]
        record_dir = destination / "records" / design_id
        artifact_path = record_dir / "prediction.pdb"
        artifact_path.unlink(missing_ok=True)
        started_at = utc_now()
        started = time.monotonic()
        print(
            f"[{index}/{len(pending)}] refolding {design_id} "
            f"length={design['sequence_length']}",
            flush=True,
        )
        try:
            folded = backend.fold(design["sequence"], parameters, seed)
            _write_text_atomic(artifact_path, folded.pdb_text)
            result = {
                "schema_version": "protein-mrna.esmfold2-design-refold.v1",
                "result_identity": "pending",
                "refold_identity": manifest["refold_identity"],
                "design_id": design_id,
                "design_identity": design["design_identity"],
                "sequence_sha256": design["sequence_sha256"],
                "sequence_length": design["sequence_length"],
                "model_label": design["model_label"],
                "sampling_seed": design["seed"],
                "status": "succeeded",
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.monotonic() - started,
                "metrics": folded.metrics,
                "peak_gpu_memory_allocated_bytes": (
                    folded.peak_gpu_memory_allocated_bytes
                ),
                "peak_gpu_memory_reserved_bytes": (
                    folded.peak_gpu_memory_reserved_bytes
                ),
                "artifact": {
                    "path": str(artifact_path.relative_to(destination)),
                    "media_type": "chemical/x-pdb",
                    "bytes": artifact_path.stat().st_size,
                    "sha256": sha256_file(artifact_path),
                },
            }
        except Exception as error:
            recover = getattr(backend, "recover_after_failure", None)
            if recover is not None:
                recover()
            result = {
                "schema_version": "protein-mrna.esmfold2-design-refold.v1",
                "result_identity": "pending",
                "refold_identity": manifest["refold_identity"],
                "design_id": design_id,
                "design_identity": design["design_identity"],
                "sequence_sha256": design["sequence_sha256"],
                "sequence_length": design["sequence_length"],
                "model_label": design["model_label"],
                "sampling_seed": design["seed"],
                "status": "failed",
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "runtime_seconds": time.monotonic() - started,
                "error": {"type": type(error).__name__, "message": str(error)[:4000]},
            }
            print(
                f"{design_id} failed: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        result["result_identity"] = _identity(result, "result_identity")
        write_json_atomic(_refold_result_path(destination, design_id), result)
        summary = _summarize_refolds(
            destination, designs, manifest["refold_identity"], load_seconds
        )
        print(
            f"progress: succeeded={summary['records']['succeeded']} "
            f"failed={summary['records']['failed']} "
            f"pending={summary['records']['pending']}",
            flush=True,
        )
    return _summarize_refolds(
        destination, designs, manifest["refold_identity"], load_seconds
    )


def _evaluation_record_path(output_dir: Path, design_id: str) -> Path:
    return output_dir / "records" / f"{design_id}.json"


def _valid_evaluation_result(
    output_dir: Path,
    design: dict,
    evaluation_identity: str,
) -> dict | None:
    path = _evaluation_record_path(output_dir, design["design_id"])
    if not path.is_file():
        return None
    try:
        result = read_json(path)
    except ContractError:
        return None
    if (
        result.get("schema_version")
        != "protein-mrna.proteinmpnn-refold-evaluation-record.v1"
        or result.get("result_identity") != _identity(result, "result_identity")
        or result.get("evaluation_identity") != evaluation_identity
        or result.get("design_id") != design["design_id"]
        or result.get("design_identity") != design["design_identity"]
        or result.get("benchmark_record_id") != design["benchmark_record_id"]
        or result.get("selection_role") != design["selection_role"]
        or result.get("model_label") != design["model_label"]
        or result.get("sampling_seed") != design["seed"]
        or result.get("status") not in {"succeeded", "failed"}
    ):
        return None
    try:
        runtime_seconds = float(result["runtime_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(runtime_seconds) or runtime_seconds < 0.0:
        return None
    if result["status"] == "succeeded":
        nested_design = result.get("design", {})
        if (
            nested_design.get("sequence_sha256") != design["sequence_sha256"]
            or nested_design.get("sequence_length") != design["sequence_length"]
            or nested_design.get("designable_positions")
            != design["designable_positions"]
            or nested_design.get("mutation_count") != design["mutation_count"]
        ):
            return None
        try:
            metrics = _flat_metrics(result)
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in metrics.values()):
            return None
    return result


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _flat_metrics(result: dict) -> dict[str, float]:
    design = result["design"]
    experimental = result["experimental_native"]
    native_prediction = result["native_sequence_prediction_reference"]
    delta = result["delta_vs_native_sequence_baseline"]
    confidence = result["refold_confidence"]
    return {
        "sequence_recovery": float(design["sequence_recovery"]),
        "mutation_fraction": 1.0 - float(design["sequence_recovery"]),
        "sampled_nll": float(design["sampled_nll"]),
        "experimental_ca_lddt": float(experimental["ca_lddt"]),
        "experimental_ca_tm_score_resolved": float(
            experimental["ca_tm_score_resolved"]
        ),
        "experimental_ca_tm_score_full_length": float(
            experimental["ca_tm_score_full_length"]
        ),
        "experimental_ca_rmsd_angstrom": float(
            experimental["ca_rmsd_angstrom"]
        ),
        "delta_experimental_ca_lddt_vs_native_sequence": float(
            delta["ca_lddt"]
        ),
        "delta_experimental_tm_resolved_vs_native_sequence": float(
            delta["ca_tm_score_resolved"]
        ),
        "delta_experimental_tm_full_length_vs_native_sequence": float(
            delta["ca_tm_score_full_length"]
        ),
        "delta_experimental_ca_rmsd_vs_native_sequence": float(
            delta["ca_rmsd_angstrom"]
        ),
        "native_prediction_ca_lddt": float(native_prediction["ca_lddt"]),
        "native_prediction_ca_tm_score": float(
            native_prediction["ca_tm_score_full_length"]
        ),
        "native_prediction_ca_rmsd_angstrom": float(
            native_prediction["ca_rmsd_angstrom"]
        ),
        "refold_mean_plddt": float(confidence["mean_plddt"]),
        "refold_ptm": float(confidence["ptm"]),
    }


def _summarize_evaluation(
    output_dir: Path,
    designs: list[dict],
    evaluation_identity: str,
) -> dict:
    results = [
        result
        for design in designs
        if (
            result := _valid_evaluation_result(
                output_dir, design, evaluation_identity
            )
        )
        is not None
    ]
    succeeded = [result for result in results if result["status"] == "succeeded"]
    failed = [result for result in results if result["status"] == "failed"]
    by_model = {}
    for label in ("official-v48-020", "stage2a"):
        model_results = [row for row in succeeded if row["model_label"] == label]
        flattened = [_flat_metrics(row) for row in model_results]
        by_model[label] = {
            field: _stats([row[field] for row in flattened])
            for field in REFOLD_METRIC_FIELDS
        }
    by_backbone = {}
    for record_id in dict.fromkeys(
        design["benchmark_record_id"] for design in designs
    ):
        by_backbone[record_id] = {}
        for label in ("official-v48-020", "stage2a"):
            rows = [
                _flat_metrics(result)
                for result in succeeded
                if result["benchmark_record_id"] == record_id
                and result["model_label"] == label
            ]
            by_backbone[record_id][label] = {
                field: _stats([row[field] for row in rows])
                for field in REFOLD_METRIC_FIELDS
            }

    paired = []
    result_by_key = {
        (
            result["benchmark_record_id"],
            int(result["sampling_seed"]),
            result["model_label"],
        ): result
        for result in succeeded
    }
    for design in designs:
        if design["model_label"] != "official-v48-020":
            continue
        key = (design["benchmark_record_id"], int(design["seed"]))
        official = result_by_key.get((*key, "official-v48-020"))
        stage2a = result_by_key.get((*key, "stage2a"))
        if official is None or stage2a is None:
            continue
        official_metrics = _flat_metrics(official)
        stage2a_metrics = _flat_metrics(stage2a)
        paired.append(
            {
                "benchmark_record_id": key[0],
                "seed": key[1],
                "delta_stage2a_minus_official": {
                    field: stage2a_metrics[field] - official_metrics[field]
                    for field in REFOLD_METRIC_FIELDS
                },
            }
        )
    paired_summary = {
        "pairs": len(paired),
        "mean_delta_stage2a_minus_official": {
            field: (
                statistics.fmean(
                    row["delta_stage2a_minus_official"][field] for row in paired
                )
                if paired
                else None
            )
            for field in REFOLD_METRIC_FIELDS
        },
        "stage2a_wins": {
            field: sum(
                row["delta_stage2a_minus_official"][field] > 0.0 for row in paired
            )
            for field in (
                "experimental_ca_lddt",
                "experimental_ca_tm_score_resolved",
                "experimental_ca_tm_score_full_length",
                "native_prediction_ca_lddt",
                "native_prediction_ca_tm_score",
            )
        },
        "stage2a_lower_is_better_wins": {
            field: sum(
                row["delta_stage2a_minus_official"][field] < 0.0 for row in paired
            )
            for field in (
                "experimental_ca_rmsd_angstrom",
                "native_prediction_ca_rmsd_angstrom",
            )
        },
    }
    summary = {
        "schema_version": "protein-mrna.proteinmpnn-refold-evaluation-summary.v1",
        "evaluation_identity": evaluation_identity,
        "updated_at_utc": utc_now(),
        "status": (
            "passed"
            if len(succeeded) == len(designs)
            else "failed" if failed else "running"
        ),
        "records": {
            "selected": len(designs),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "pending": len(designs) - len(results),
        },
        "by_model": by_model,
        "by_backbone": by_backbone,
        "paired": paired_summary,
        "limitations": [
            "Four-backbone valid-split engineering pilot; no release gate is defined.",
            "ESMFold2 is a shared computational oracle for both model arms.",
            "Target-only refolds may omit assembly-stabilized native conformations.",
            "Possible overlap with ESMFold2 training data has not been audited.",
            "Sequence recovery is descriptive and is not a biological quality metric.",
        ],
    }
    write_json_atomic(output_dir / "summary.json", summary)
    write_jsonl_atomic(output_dir / "records.jsonl", results)
    return summary


def evaluate_design_refolds(
    pilot_dir: str | Path,
    refold_dir: str | Path,
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
    pilot_root = Path(pilot_dir).expanduser().resolve()
    pilot_manifest, designs, refold_manifest, refold_results = (
        load_completed_refolds(pilot_root, refold_dir)
    )
    suite_path = Path(pilot_manifest["benchmark"]["suite_path"])
    verify_benchmark_suite_files(suite_path)
    suite = read_json(suite_path)
    suite_by_id = {row["benchmark_record_id"]: row for row in suite["records"]}
    native_prediction_dir = Path(pilot_manifest["native_prediction_run"])
    verify_esmfold2_benchmark_run(suite_path, native_prediction_dir, mode="full")
    agreement_dir = Path(pilot_manifest["native_agreement"]["directory"])
    agreement_summary, agreement_records = _validate_native_agreement(
        agreement_dir, suite
    )
    agreement_by_id = {
        row["benchmark_record_id"]: row for row in agreement_records
    }
    source_dir = Path(pilot_manifest["native_dataset"]["dataset_dir"])
    dataset = _verify_dataset(source_dir, suite)
    selected_ids = {design["benchmark_record_id"] for design in designs}
    index_rows = _selected_index_rows(
        source_dir,
        {suite_by_id[record_id]["source_chain_id"] for record_id in selected_ids},
    )
    _validate_selected_index_rows(source_dir, index_rows, dataset)
    runtime = (
        load_metrics_runtime_manifest(metrics_runtime_root)
        if metrics_runtime_document is None
        else metrics_runtime_document
    )
    if metrics_runtime_document is not None:
        _validate_metrics_runtime_document(runtime)
    manifest = {
        "schema_version": "protein-mrna.proteinmpnn-refold-evaluation.v1",
        "evaluation_identity": "pending",
        "created_at_utc": utc_now(),
        "pilot_identity": pilot_manifest["pilot_identity"],
        "refold_identity": refold_manifest["refold_identity"],
        "metrics_runtime_identity": runtime["runtime_identity"],
        "native_agreement_identity": agreement_summary["evaluation_identity"],
        "references": ["experimental_native", "native_sequence_esmfold2_prediction"],
        "implementation": {
            "module": "protein_mrna_pipeline.design_refold",
            "module_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    manifest["evaluation_identity"] = _identity(manifest, "evaluation_identity")
    destination = Path(output_dir).expanduser().resolve()
    _initialize_directory(
        destination,
        "evaluation-manifest.json",
        manifest,
        "evaluation_identity",
    )

    pending = []
    for design in designs:
        existing = _valid_evaluation_result(
            destination, design, manifest["evaluation_identity"]
        )
        if existing is None or (retry_failed and existing["status"] == "failed"):
            pending.append(design)
    payload_cache = {}
    native_cache = {}
    native_prediction_cache = {}
    for index, design in enumerate(pending, 1):
        design_id = design["design_id"]
        record_id = design["benchmark_record_id"]
        benchmark_record = suite_by_id[record_id]
        started = time.monotonic()
        try:
            if record_id not in payload_cache:
                index_row = index_rows[benchmark_record["source_chain_id"]]
                payload_cache[record_id] = payload_loader(source_dir, index_row)
                native_cache[record_id] = native_extractor(
                    payload_cache[record_id],
                    benchmark_record["source_chain_id"],
                    benchmark_record["sequence"],
                )
                native_result = read_json(
                    native_prediction_dir / "records" / record_id / "result.json"
                )
                native_pdb = native_prediction_dir / native_result["artifact"]["path"]
                native_prediction_cache[record_id] = prediction_parser(
                    native_pdb, benchmark_record["sequence"]
                )
            native = native_cache[record_id]
            refold_result = refold_results[design_id]
            refold_pdb = Path(refold_dir).resolve() / refold_result["artifact"]["path"]
            design_coordinates = prediction_parser(refold_pdb, design["sequence"])
            experimental_metrics = metric_function(
                native["ca_coordinates"],
                design_coordinates,
                native["ca_mask"],
                design["sequence"],
            )
            native_prediction_coordinates = native_prediction_cache[record_id]
            native_prediction_metrics = metric_function(
                native_prediction_coordinates,
                design_coordinates,
                [True] * int(design["sequence_length"]),
                design["sequence"],
            )
            baseline = agreement_by_id[record_id]["metrics"]
            delta = {
                "ca_lddt": (
                    float(experimental_metrics["ca_lddt"])
                    - float(baseline["ca_lddt"])
                ),
                "ca_tm_score_resolved": (
                    float(experimental_metrics["ca_tm_score_resolved"])
                    - float(baseline["ca_tm_score_resolved"])
                ),
                "ca_tm_score_full_length": (
                    float(experimental_metrics["ca_tm_score_full_length"])
                    - float(baseline["ca_tm_score_full_length"])
                ),
                "ca_rmsd_angstrom": (
                    float(experimental_metrics["ca_rmsd_angstrom"])
                    - float(baseline["ca_rmsd_angstrom"])
                ),
            }
            result = {
                "schema_version": (
                    "protein-mrna.proteinmpnn-refold-evaluation-record.v1"
                ),
                "result_identity": "pending",
                "evaluation_identity": manifest["evaluation_identity"],
                "design_id": design_id,
                "design_identity": design["design_identity"],
                "benchmark_record_id": record_id,
                "selection_role": design["selection_role"],
                "model_label": design["model_label"],
                "sampling_seed": design["seed"],
                "status": "succeeded",
                "runtime_seconds": time.monotonic() - started,
                "design": {
                    "sequence_sha256": design["sequence_sha256"],
                    "sequence_length": design["sequence_length"],
                    "designable_positions": design["designable_positions"],
                    "mutation_count": design["mutation_count"],
                    "sequence_recovery": design["sequence_recovery"],
                    "sampled_nll": design["sampled_nll"],
                    "native_nll_same_order": design["native_nll_same_order"],
                },
                "experimental_native": {
                    **{key: float(value) for key, value in experimental_metrics.items()},
                    "native_ca_coverage": native["ca_coverage"],
                },
                "native_sequence_prediction_reference": {
                    key: float(value)
                    for key, value in native_prediction_metrics.items()
                },
                "native_sequence_experimental_baseline": baseline,
                "delta_vs_native_sequence_baseline": delta,
                "refold_confidence": refold_result["metrics"],
                "artifacts": {
                    "refold_pdb": refold_result["artifact"],
                },
            }
        except Exception as error:
            result = {
                "schema_version": (
                    "protein-mrna.proteinmpnn-refold-evaluation-record.v1"
                ),
                "result_identity": "pending",
                "evaluation_identity": manifest["evaluation_identity"],
                "design_id": design_id,
                "design_identity": design["design_identity"],
                "benchmark_record_id": record_id,
                "selection_role": design["selection_role"],
                "model_label": design["model_label"],
                "sampling_seed": design["seed"],
                "status": "failed",
                "runtime_seconds": time.monotonic() - started,
                "error": {"type": type(error).__name__, "message": str(error)[:4000]},
            }
            print(
                f"{design_id} evaluation failed: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        result["result_identity"] = _identity(result, "result_identity")
        write_json_atomic(_evaluation_record_path(destination, design_id), result)
        summary = _summarize_evaluation(
            destination, designs, manifest["evaluation_identity"]
        )
        print(
            f"[{index}/{len(pending)}] {design_id}: {result['status']} "
            f"completed={summary['records']['succeeded']} "
            f"failed={summary['records']['failed']}",
            flush=True,
        )
    return _summarize_evaluation(
        destination, designs, manifest["evaluation_identity"]
    )
