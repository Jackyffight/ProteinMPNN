from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "protein_mrna_pipeline" / "src"))

from design_flow_stage3.esmfold2_job_runner import (  # noqa: E402
    JOB_SCHEMA,
    _identity,
    job_fasta_bytes,
    pack_results,
    run_job,
    unpack_job_archive,
    validate_job_directory,
    verify_run,
)
from protein_mrna_pipeline.esmfold2_runner import (  # noqa: E402
    BIOHUB_TRANSFORMERS_COMMIT,
    BIOHUB_TRANSFORMERS_VERSION,
    DEFAULT_PARAMETERS,
    ESMC_6B_REPOSITORY,
    ESMC_6B_REVISION,
    ESMFOLD2_FAST_REPOSITORY,
    ESMFOLD2_FAST_REVISION,
    FoldOutput,
    _identity as runtime_identity,
    _weight_manifest,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runtime_fixture() -> dict:
    document = {
        "schema_version": "protein-mrna.esmfold2-runtime.v1",
        "runtime_identity": "pending",
        "created_at_utc": "2026-07-13T00:00:00+00:00",
        "runtime_root": "/fixture/runtime",
        "source": {
            "repository": "https://github.com/Biohub/transformers.git",
            "revision": BIOHUB_TRANSFORMERS_COMMIT,
            "transformers_version": BIOHUB_TRANSFORMERS_VERSION,
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
        "environment": {"python": "3.11.2", "torch": "2.7.1"},
    }
    document["runtime_identity"] = runtime_identity(document, "runtime_identity")
    return document


def write_job(root: Path) -> Path:
    root.mkdir()
    records = [
        {
            "candidate_id": "candidate-alpha",
            "candidate_key": "alpha",
            "display_name": "Alpha",
            "candidate_type": "source_control",
            "sequence": "ACDEFGHIK",
            "sequence_sha256": hashlib.sha256(b"ACDEFGHIK").hexdigest(),
            "length": 9,
            "release_status": "provisional",
            "inferred_components": [],
        },
        {
            "candidate_id": "candidate-beta",
            "candidate_key": "beta",
            "display_name": "Beta",
            "candidate_type": "fusion",
            "sequence": "LMNPQRSTV",
            "sequence_sha256": hashlib.sha256(b"LMNPQRSTV").hexdigest(),
            "length": 9,
            "release_status": "quarantined",
            "inferred_components": [],
        },
    ]
    fasta = job_fasta_bytes(records)
    (root / "sequences.fasta").write_bytes(fasta)
    job = {
        "schema_version": JOB_SCHEMA,
        "job_identity": "pending",
        "created_at_utc": "2026-07-13T12:00:00+00:00",
        "source": {
            "project_id": "fixture-project",
            "stage2_run_id": "fixture-stage2",
            "stage2_artifact_index_sha256": "1" * 64,
            "candidate_batch_sha256": "2" * 64,
        },
        "model": {
            "name": "ESMFold2-Fast",
            "source_revision": BIOHUB_TRANSFORMERS_COMMIT,
            "structure_revision": ESMFOLD2_FAST_REVISION,
            "language_model_revision": ESMC_6B_REVISION,
        },
        "execution": {"seed": 42, "parameters": dict(DEFAULT_PARAMETERS)},
        "maximum_sequence_length": 1024,
        "records": records,
        "fasta": {
            "path": "sequences.fasta",
            "sha256": hashlib.sha256(fasta).hexdigest(),
            "bytes": len(fasta),
            "records": len(records),
        },
    }
    job["job_identity"] = _identity(job, "job_identity")
    write_json(root / "job-manifest.json", job)
    return root


class FakeBackend:
    calls: list[str] = []
    load_seconds = 1.5

    def __init__(self, runtime_root: Path):
        self.runtime_root = runtime_root

    def fold(self, sequence: str, parameters: dict, seed: int) -> FoldOutput:
        self.calls.append(sequence)
        return FoldOutput(
            pdb_text=(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
                "  1.00 75.00           C\nEND\n"
            ),
            metrics={"mean_plddt": 0.75, "ptm": 0.6},
            peak_gpu_memory_allocated_bytes=1024,
            peak_gpu_memory_reserved_bytes=2048,
        )

    def recover_after_failure(self) -> None:
        return None


class Stage3WorkerTests(unittest.TestCase):
    def test_job_run_is_resumable_and_rechecks_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            job_dir = write_job(root / "job")
            output_dir = root / "output"
            FakeBackend.calls = []
            summary = run_job(
                job_dir,
                output_dir,
                root / "runtime",
                backend_factory=FakeBackend,
                runtime_document=runtime_fixture(),
            )
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(len(FakeBackend.calls), 2)
            self.assertEqual(verify_run(job_dir, output_dir)["records"], 2)

            def must_not_load(_runtime_root: Path):
                raise AssertionError("completed records must not reload the model")

            resumed = run_job(
                job_dir,
                output_dir,
                root / "runtime",
                backend_factory=must_not_load,
                runtime_document=runtime_fixture(),
            )
            self.assertEqual(resumed["status"], "passed")

            pdb_path = output_dir / "records/candidate-alpha/prediction.pdb"
            pdb_path.write_text("tampered\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "invalid results"):
                verify_run(job_dir, output_dir)
            FakeBackend.calls = []
            run_job(
                job_dir,
                output_dir,
                root / "runtime",
                backend_factory=FakeBackend,
                runtime_document=runtime_fixture(),
            )
            self.assertEqual(FakeBackend.calls, ["ACDEFGHIK"])

    def test_job_identity_and_fasta_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            job_dir = write_job(Path(temporary_dir) / "job")
            (job_dir / "sequences.fasta").write_text(
                ">candidate-alpha\nACDEFGHIK\n", encoding="ascii"
            )
            with self.assertRaisesRegex(ValueError, "FASTA content differs"):
                validate_job_directory(job_dir)

    def test_job_and_result_archives_are_bounded_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_job = write_job(root / "source-job")
            job_archive = root / "job.tar.gz"
            with tarfile.open(job_archive, "w:gz") as bundle:
                for name in ("job-manifest.json", "sequences.fasta"):
                    bundle.add(source_job / name, arcname=name)
            unpacked = root / "unpacked"
            self.assertEqual(
                unpack_job_archive(job_archive, unpacked)["job"]["job_identity"],
                validate_job_directory(source_job)["job"]["job_identity"],
            )
            output_dir = root / "output"
            run_job(
                unpacked,
                output_dir,
                root / "runtime",
                backend_factory=FakeBackend,
                runtime_document=runtime_fixture(),
            )
            result_archive = root / "results.tar.gz"
            packed = pack_results(unpacked, output_dir, result_archive)
            self.assertTrue(result_archive.is_file())
            self.assertEqual(packed["status"], "passed")
            with tarfile.open(result_archive, "r:gz") as bundle:
                self.assertIn("run-manifest.json", bundle.getnames())
                self.assertIn("records/candidate-alpha/prediction.pdb", bundle.getnames())

    def test_launcher_uses_separate_absolute_runtime_and_single_gpu(self) -> None:
        launcher = (REPO_ROOT / "design_flow_stage3/run_stage3_esmfold2.sh").read_text()
        self.assertIn("/MPNN/structure_runtime/esmfold2-fast", launcher)
        self.assertIn("expose exactly one GPU", launcher)
        self.assertIn("HF_HUB_OFFLINE=1", launcher)
        self.assertIn("pack-results", launcher)


if __name__ == "__main__":
    unittest.main()
