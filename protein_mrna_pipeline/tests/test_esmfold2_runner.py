import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from protein_mrna_pipeline.benchmark import generate_benchmark_suite  # noqa: E402
from protein_mrna_pipeline.contracts import read_json  # noqa: E402
from protein_mrna_pipeline.esmfold2_runner import (  # noqa: E402
    BIOHUB_TRANSFORMERS_COMMIT,
    BIOHUB_TRANSFORMERS_VERSION,
    ESMC_6B_REPOSITORY,
    ESMC_6B_REVISION,
    ESMFOLD2_FAST_REPOSITORY,
    ESMFOLD2_FAST_REVISION,
    FoldOutput,
    _identity,
    _weight_manifest,
    run_esmfold2_benchmark,
    select_benchmark_records,
    verify_esmfold2_benchmark_run,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_benchmark_dataset(root):
    root.mkdir()
    write_json(
        root / "manifest.json",
        {
            "format": "proteinmpnn.tar_shard.v2",
            "version_id": "fixture-pdb-2026",
            "record_count": 9,
        },
    )
    write_json(
        root / "validation.json",
        {
            "schema": "proteinmpnn.tar_shard_validation.v2",
            "status": "ok",
            "exact_sequence_split_leaks": 0,
            "pdb_split_leaks": 0,
            "records": 9,
            "payloads_checked": 9,
        },
    )
    (root / "valid_clusters.txt").write_text(
        "".join(f"{cluster}\n" for cluster in range(1, 9)), encoding="utf-8"
    )
    (root / "test_clusters.txt").write_text("99\n", encoding="utf-8")
    alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def sequence(length, offset):
        rotated = alphabet[offset:] + alphabet[:offset]
        return (rotated * (length // len(rotated) + 1))[:length]

    rows = ["CHAINID,DEPOSITION,RESOLUTION,HASH,CLUSTER,SEQUENCE\n"]
    for index, length in enumerate((60, 120, 250, 350, 450, 550, 650, 750), 1):
        rows.append(
            f"fixture{index}_A,2026-01-{index:02d},2.00,h{index},{index},"
            f"{sequence(length, index)}\n"
        )
    rows.append(f"test_A,2026-01-09,2.00,h9,99,{sequence(100, 9)}\n")
    (root / "list.csv").write_text("".join(rows), encoding="utf-8")


def runtime_fixture():
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
        "environment": {
            "python": "3.11.2",
            "python_executable": "/fixture/runtime/venv/bin/python",
            "platform": "fixture",
            "torch": "2.7.1",
        },
    }
    document["runtime_identity"] = _identity(document, "runtime_identity")
    return document


class FakeBackend:
    calls = []
    load_seconds = 1.25

    def __init__(self, runtime_root):
        self.runtime_root = runtime_root

    def fold(self, sequence, parameters, seed):
        self.calls.append((len(sequence), dict(parameters), seed))
        return FoldOutput(
            pdb_text=(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
                "  1.00  0.50           C\nEND\n"
            ),
            metrics={"mean_plddt": 0.5, "ptm": 0.4},
            peak_gpu_memory_allocated_bytes=1024,
            peak_gpu_memory_reserved_bytes=2048,
        )

    def recover_after_failure(self):
        return None


class FailFirstBackend(FakeBackend):
    failures_remaining = 1

    def fold(self, sequence, parameters, seed):
        if self.failures_remaining:
            type(self).failures_remaining -= 1
            raise RuntimeError("fixture failure")
        return super().fold(sequence, parameters, seed)


class ESMFold2RunnerTest(unittest.TestCase):
    def make_suite(self, root, count=8):
        dataset = root / "dataset"
        write_benchmark_dataset(dataset)
        summary = generate_benchmark_suite(dataset, root / "suite", requested_count=count)
        return Path(summary["suite_path"])

    def test_smoke_selects_the_longest_record_from_each_length_bin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            suite_path = self.make_suite(Path(temp_dir))
            selected = select_benchmark_records(read_json(suite_path), "smoke")
            self.assertEqual([record["length"] for record in selected], [120, 350, 550, 750])

    def test_fake_benchmark_is_atomic_resumable_and_rechecks_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite_path = self.make_suite(root, count=4)
            output_dir = root / "run"
            FakeBackend.calls = []
            summary = run_esmfold2_benchmark(
                suite_path,
                output_dir,
                root / "runtime",
                mode="smoke",
                backend_factory=FakeBackend,
                runtime_document=runtime_fixture(),
            )
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["records"]["succeeded"], 4)
            self.assertEqual(len(FakeBackend.calls), 4)
            self.assertTrue((output_dir / "run-manifest.json").is_file())
            verified = verify_esmfold2_benchmark_run(
                suite_path, output_dir, mode="smoke"
            )
            self.assertEqual(verified["records"], 4)

            def must_not_load(_runtime_root):
                raise AssertionError("completed run should not reload the model")

            resumed = run_esmfold2_benchmark(
                suite_path,
                output_dir,
                root / "runtime",
                mode="smoke",
                backend_factory=must_not_load,
                runtime_document=runtime_fixture(),
            )
            self.assertEqual(resumed["status"], "passed")

            artifact = output_dir / "records/record-0001/prediction.pdb"
            artifact.write_text("tampered\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "invalid artifacts"):
                verify_esmfold2_benchmark_run(suite_path, output_dir, mode="smoke")
            FakeBackend.calls = []
            repaired = run_esmfold2_benchmark(
                suite_path,
                output_dir,
                root / "runtime",
                mode="smoke",
                backend_factory=FakeBackend,
                runtime_document=runtime_fixture(),
            )
            self.assertEqual(repaired["status"], "passed")
            self.assertEqual(len(FakeBackend.calls), 1)

    def test_failed_records_are_terminal_until_retry_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite_path = self.make_suite(root, count=4)
            output_dir = root / "run"
            FailFirstBackend.calls = []
            FailFirstBackend.failures_remaining = 1
            failed = run_esmfold2_benchmark(
                suite_path,
                output_dir,
                root / "runtime",
                mode="smoke",
                backend_factory=FailFirstBackend,
                runtime_document=runtime_fixture(),
            )
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["records"]["failed"], 1)

            def must_not_load(_runtime_root):
                raise AssertionError("failed records require an explicit retry")

            unchanged = run_esmfold2_benchmark(
                suite_path,
                output_dir,
                root / "runtime",
                mode="smoke",
                backend_factory=must_not_load,
                runtime_document=runtime_fixture(),
            )
            self.assertEqual(unchanged["status"], "failed")

            FakeBackend.calls = []
            retried = run_esmfold2_benchmark(
                suite_path,
                output_dir,
                root / "runtime",
                mode="smoke",
                retry_failed=True,
                backend_factory=FakeBackend,
                runtime_document=runtime_fixture(),
            )
            self.assertEqual(retried["status"], "passed")
            self.assertEqual(len(FakeBackend.calls), 1)

    def test_launchers_pin_revisions_and_enforce_single_gpu(self):
        setup = (PROJECT_ROOT.parent / "scripts/setup_esmfold2_fast_runtime.sh").read_text()
        runner = (PROJECT_ROOT.parent / "scripts/run_esmfold2_fast.sh").read_text()
        self.assertIn(BIOHUB_TRANSFORMERS_COMMIT, setup)
        self.assertIn(ESMFOLD2_FAST_REVISION, setup)
        self.assertIn(ESMC_6B_REVISION, setup)
        self.assertIn("verify-esmfold2-runtime", setup)
        self.assertNotIn("pip install esm@", setup)
        self.assertIn("this bounded runner is single-GPU", runner)
        self.assertIn("HF_HUB_OFFLINE=1", runner)


if __name__ == "__main__":
    unittest.main()
