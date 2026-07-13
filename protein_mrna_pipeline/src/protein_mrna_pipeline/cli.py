"""Command-line interface for contracts and resumable pipeline runs."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .benchmark import generate_benchmark_suite, verify_benchmark_suite_files
from .contracts import (
    ContractError,
    SCHEMA_FILES,
    read_json,
    validate_document,
)
from .esmfold2_runner import (
    create_runtime_manifest,
    run_esmfold2_benchmark,
    verify_esmfold2_benchmark_run,
)
from .run_store import RunStore, initialize_run


KINDS = tuple(SCHEMA_FILES)


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def command_validate(args: argparse.Namespace) -> int:
    document = read_json(args.path)
    validate_document(document, args.kind)
    print_json({"kind": args.kind, "path": str(Path(args.path).resolve()), "status": "ok"})
    return 0


def command_init_run(args: argparse.Namespace) -> int:
    manifest = initialize_run(
        args.target,
        args.run_dir,
        allow_unreviewed=args.allow_unreviewed,
    )
    print_json(manifest)
    return 0


def command_make_benchmark(args: argparse.Namespace) -> int:
    summary = generate_benchmark_suite(
        args.dataset_dir,
        args.output_dir,
        requested_count=args.count,
        seed=args.seed,
        min_length=args.min_length,
        max_length=args.max_length,
        length_bin_count=args.length_bins,
    )
    print_json(summary)
    return 0


def command_verify_benchmark(args: argparse.Namespace) -> int:
    print_json(verify_benchmark_suite_files(args.suite))
    return 0


def command_verify_esmfold2_runtime(args: argparse.Namespace) -> int:
    print_json(
        create_runtime_manifest(
            args.runtime_root,
            verify_hashes=not args.skip_weight_hashes,
        )
    )
    return 0


def command_run_esmfold2_benchmark(args: argparse.Namespace) -> int:
    summary = run_esmfold2_benchmark(
        args.suite,
        args.output_dir,
        args.runtime_root,
        mode=args.mode,
        seed=args.seed,
        parameters={
            "chunk_size": args.chunk_size,
            "num_diffusion_samples": 1,
            "num_loops": args.num_loops,
            "num_sampling_steps": args.num_sampling_steps,
        },
        retry_failed=args.retry_failed,
    )
    print_json(summary)
    return 0 if summary["status"] == "passed" else 1


def command_verify_esmfold2_run(args: argparse.Namespace) -> int:
    print_json(
        verify_esmfold2_benchmark_run(
            args.suite,
            args.output_dir,
            mode=args.mode,
        )
    )
    return 0


def command_enqueue(args: argparse.Namespace) -> int:
    store = RunStore(args.run_dir)
    item, created = store.enqueue(read_json(args.item))
    print_json({"created": created, "work_item": item})
    return 0


def command_claim(args: argparse.Namespace) -> int:
    store = RunStore(args.run_dir)
    claim = store.claim(args.worker_id, args.lease_seconds)
    print_json({"claimed": claim})
    return 0


def command_finish(args: argparse.Namespace) -> int:
    store = RunStore(args.run_dir)
    result = read_json(args.result)
    state = store.finish(result, args.worker_id)
    print_json({"state": state, "work_id": result["work_id"]})
    return 0


def command_renew(args: argparse.Namespace) -> int:
    store = RunStore(args.run_dir)
    lease = store.renew(
        args.work_id,
        args.worker_id,
        args.attempt,
        args.lease_seconds,
    )
    print_json(lease)
    return 0


def command_status(args: argparse.Namespace) -> int:
    print_json(RunStore(args.run_dir).status())
    return 0


def command_export(args: argparse.Namespace) -> int:
    print_json(RunStore(args.run_dir).export())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protein-mrna-pipeline",
        description="Validate and orchestrate auditable protein-to-mRNA runs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate one JSON contract")
    validate_parser.add_argument("--kind", required=True, choices=KINDS)
    validate_parser.add_argument("path")
    validate_parser.set_defaults(handler=command_validate)

    init_parser = subparsers.add_parser("init-run", help="atomically initialize a run")
    init_parser.add_argument("--target", required=True)
    init_parser.add_argument("--run-dir", required=True)
    init_parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="engineering-only override; denied targets remain forbidden",
    )
    init_parser.set_defaults(handler=command_init_run)

    benchmark_parser = subparsers.add_parser(
        "make-benchmark",
        help="select a bounded engineering benchmark from the PDB valid split",
    )
    benchmark_parser.add_argument("--dataset-dir", required=True)
    benchmark_parser.add_argument("--output-dir", required=True)
    benchmark_parser.add_argument("--count", type=int, default=40)
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.add_argument("--min-length", type=int, default=50)
    benchmark_parser.add_argument("--max-length", type=int, default=800)
    benchmark_parser.add_argument("--length-bins", type=int, default=4)
    benchmark_parser.set_defaults(handler=command_make_benchmark)

    verify_benchmark_parser = subparsers.add_parser(
        "verify-benchmark",
        help="verify a benchmark suite and its companion FASTA",
    )
    verify_benchmark_parser.add_argument("--suite", required=True)
    verify_benchmark_parser.set_defaults(handler=command_verify_benchmark)

    runtime_parser = subparsers.add_parser(
        "verify-esmfold2-runtime",
        help="verify pinned ESMFold2-Fast weights and write its runtime manifest",
    )
    runtime_parser.add_argument("--runtime-root", required=True)
    runtime_parser.add_argument(
        "--skip-weight-hashes",
        action="store_true",
        help="check weight sizes only; intended for repeat inventory, not installation",
    )
    runtime_parser.set_defaults(handler=command_verify_esmfold2_runtime)

    structure_parser = subparsers.add_parser(
        "run-esmfold2-benchmark",
        help="run the resumable pinned ESMFold2-Fast engineering benchmark",
    )
    structure_parser.add_argument("--suite", required=True)
    structure_parser.add_argument("--output-dir", required=True)
    structure_parser.add_argument("--runtime-root", required=True)
    structure_parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    structure_parser.add_argument("--seed", type=int, default=42)
    structure_parser.add_argument("--chunk-size", type=int, default=64)
    structure_parser.add_argument("--num-loops", type=int, default=3)
    structure_parser.add_argument("--num-sampling-steps", type=int, default=50)
    structure_parser.add_argument("--retry-failed", action="store_true")
    structure_parser.set_defaults(handler=command_run_esmfold2_benchmark)

    verify_structure_parser = subparsers.add_parser(
        "verify-esmfold2-run",
        help="verify every result and PDB checksum in an ESMFold2 benchmark run",
    )
    verify_structure_parser.add_argument("--suite", required=True)
    verify_structure_parser.add_argument("--output-dir", required=True)
    verify_structure_parser.add_argument(
        "--mode", choices=("smoke", "full"), required=True
    )
    verify_structure_parser.set_defaults(handler=command_verify_esmfold2_run)

    enqueue_parser = subparsers.add_parser("enqueue", help="enqueue one work item")
    enqueue_parser.add_argument("--run-dir", required=True)
    enqueue_parser.add_argument("--item", required=True)
    enqueue_parser.set_defaults(handler=command_enqueue)

    claim_parser = subparsers.add_parser("claim", help="claim the next leased item")
    claim_parser.add_argument("--run-dir", required=True)
    claim_parser.add_argument("--worker-id", required=True)
    claim_parser.add_argument("--lease-seconds", type=int, default=3600)
    claim_parser.set_defaults(handler=command_claim)

    finish_parser = subparsers.add_parser("finish", help="finish the active attempt")
    finish_parser.add_argument("--run-dir", required=True)
    finish_parser.add_argument("--worker-id", required=True)
    finish_parser.add_argument("--result", required=True)
    finish_parser.set_defaults(handler=command_finish)

    renew_parser = subparsers.add_parser("renew", help="renew an active work lease")
    renew_parser.add_argument("--run-dir", required=True)
    renew_parser.add_argument("--work-id", required=True)
    renew_parser.add_argument("--worker-id", required=True)
    renew_parser.add_argument("--attempt", required=True, type=int)
    renew_parser.add_argument("--lease-seconds", type=int, default=3600)
    renew_parser.set_defaults(handler=command_renew)

    status_parser = subparsers.add_parser("status", help="show queue counts")
    status_parser.add_argument("--run-dir", required=True)
    status_parser.set_defaults(handler=command_status)

    export_parser = subparsers.add_parser("export", help="export queue and attempt JSONL")
    export_parser.add_argument("--run-dir", required=True)
    export_parser.set_defaults(handler=command_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ContractError, OSError, sqlite3.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
