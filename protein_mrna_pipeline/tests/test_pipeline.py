from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from protein_mrna_pipeline.contracts import (  # noqa: E402
    ContractError,
    SCHEMA_FILES,
    load_schema,
    read_json,
    target_sha256,
    text_sha256,
    validate_candidate,
    validate_target,
)
from protein_mrna_pipeline.run_store import RunStore, initialize_run  # noqa: E402


EXAMPLE_TARGET = PROJECT_ROOT / "examples/target-package.example.json"


def approved_target():
    target = read_json(EXAMPLE_TARGET)
    target["safety"] = {
        "status": "approved",
        "review_id": "review-contract-fixture",
        "reviewed_by": "unit-test",
        "reviewed_at_utc": "2026-07-12T00:00:00+00:00",
        "notes": "Test fixture only.",
        "prohibited_modifications": [],
    }
    return target


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def work_item(manifest):
    return {
        "schema_version": "protein-mrna.work-item.v1",
        "target_id": manifest["target"]["target_id"],
        "target_sha256": manifest["target"]["sha256"],
        "execution_class": "formal",
        "stage": "initial_fold",
        "tool": {
            "name": "structure-oracle-fixture",
            "kind": "structure",
            "revision": "fixture-revision",
            "weights_sha256": "1" * 64,
            "environment_id": "fixture-environment",
        },
        "parameters": {"recycles": 1},
        "seed": 42,
        "priority": 10,
        "max_attempts": 3,
        "inputs": [],
    }


class ContractTest(unittest.TestCase):
    def test_bundled_schemas_are_valid(self):
        for kind in SCHEMA_FILES:
            with self.subTest(kind=kind):
                self.assertIsInstance(load_schema(kind), dict)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate.json"
            path.write_text('{"target_id":"a","target_id":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "duplicate JSON object key"):
                read_json(path)

    def test_example_target_passes_structural_and_semantic_validation(self):
        target = read_json(EXAMPLE_TARGET)
        validate_target(target)
        self.assertEqual(len(target_sha256(target)), 64)

    def test_mutable_region_cannot_overlap_immutable_region(self):
        target = read_json(EXAMPLE_TARGET)
        target["design_constraints"]["mutable_regions"].append(
            {"domain_id": "domain-a", "start": 3, "end": 8}
        )
        with self.assertRaisesRegex(ContractError, "overlaps fixed/immutable"):
            validate_target(target)

    def test_every_adjacent_pair_requires_a_linker_when_direct_fusion_is_disabled(self):
        target = read_json(EXAMPLE_TARGET)
        target["architecture"]["linker_options"] = []
        with self.assertRaisesRegex(ContractError, "no linker option"):
            validate_target(target)

    def test_retained_mrna_candidate_requires_a_matching_translation_hash(self):
        target = approved_target()
        target_hash = target_sha256(target)
        protein = "ACDE"
        candidate = {
            "schema_version": "protein-mrna.candidate-record.v1",
            "candidate_id": "candidate-fixture",
            "target_id": target["target_id"],
            "target_sha256": target_hash,
            "stage": "mrna",
            "protein_sequence": protein,
            "mrna": {
                "cds_sequence": "GCTTGTGATGAA",
                "translated_protein_sha256": "2" * 64,
                "translation_check": "passed",
            },
            "provenance": [
                {
                    "work_id": "mrna-generate-fixture",
                    "tool_name": "generator-fixture",
                    "revision": "fixture",
                    "weights_sha256": None,
                    "parameters": {},
                    "seed": 42,
                }
            ],
            "hard_checks": {
                "immutable_residues": "passed",
                "maximum_length": "passed",
                "translation_preservation": "passed",
                "safety": "passed",
            },
            "scores": {},
            "artifacts": [],
            "decision": {"status": "retained", "reasons": ["fixture"]},
            "created_at_utc": "2026-07-12T00:00:00+00:00",
        }
        with self.assertRaisesRegex(ContractError, "does not match"):
            validate_candidate(candidate)
        candidate["mrna"]["translated_protein_sha256"] = text_sha256(protein)
        validate_candidate(candidate)
        candidate["scores"]["non_finite"] = float("nan")
        with self.assertRaisesRegex(ContractError, "not canonical JSON"):
            validate_candidate(candidate)


class RunStoreTest(unittest.TestCase):
    def initialize_approved_run(self, root):
        target_path = root / "target.json"
        write_json(target_path, approved_target())
        run_dir = root / "run"
        manifest = initialize_run(target_path, run_dir)
        return run_dir, manifest

    def test_unreviewed_target_requires_an_explicit_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ContractError, "approved review is required"):
                initialize_run(EXAMPLE_TARGET, root / "rejected")

            manifest = initialize_run(
                EXAMPLE_TARGET,
                root / "smoke",
                allow_unreviewed=True,
            )
            self.assertTrue(manifest["safety_gate"]["override_used"])
            self.assertTrue((root / "smoke/queue.sqlite3").is_file())
            smoke_store = RunStore(root / "smoke")
            formal_item = work_item(manifest)
            with self.assertRaisesRegex(ContractError, "smoke work only"):
                smoke_store.enqueue(formal_item)
            formal_item["execution_class"] = "smoke"
            _, created = smoke_store.enqueue(formal_item)
            self.assertTrue(created)

    def test_repeated_initialization_has_unique_run_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target_path = root / "target.json"
            write_json(target_path, approved_target())
            first = initialize_run(target_path, root / "run-1")
            second = initialize_run(target_path, root / "run-2")
            self.assertNotEqual(first["run_id"], second["run_id"])

    def test_missing_run_does_not_create_an_empty_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "missing"
            with self.assertRaisesRegex(ContractError, "queue database not found"):
                RunStore(run_dir).status()
            self.assertFalse((run_dir / "queue.sqlite3").exists())

    def test_denied_target_cannot_use_the_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = read_json(EXAMPLE_TARGET)
            target["safety"]["status"] = "denied"
            target_path = root / "denied.json"
            write_json(target_path, target)
            with self.assertRaisesRegex(ContractError, "denied target"):
                initialize_run(target_path, root / "run", allow_unreviewed=True)

    def test_queue_completes_and_exports_a_verified_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, manifest = self.initialize_approved_run(Path(temp_dir))
            store = RunStore(run_dir)
            prepared, created = store.enqueue(work_item(manifest))
            self.assertTrue(created)
            existing, created_again = store.enqueue(work_item(manifest))
            self.assertFalse(created_again)
            self.assertEqual(existing["work_id"], prepared["work_id"])

            claim = store.claim("worker-0", lease_seconds=3600)
            self.assertIsNotNone(claim)
            work_dir = run_dir / claim["queue"]["work_dir"]
            artifact_path = work_dir / "structure.pdb"
            artifact_bytes = b"HEADER    CONTRACT FIXTURE\nEND\n"
            artifact_path.write_bytes(artifact_bytes)
            relative_artifact = artifact_path.relative_to(run_dir)
            result = {
                "schema_version": "protein-mrna.tool-result.v1",
                "work_id": prepared["work_id"],
                "worker_id": "worker-0",
                "attempt": claim["queue"]["attempt"],
                "status": "succeeded",
                "started_at_utc": claim["queue"]["claimed_at_utc"],
                "finished_at_utc": claim["queue"]["claimed_at_utc"],
                "runtime_seconds": 1.0,
                "artifacts": [
                    {
                        "artifact_id": "structure-fixture",
                        "path": str(relative_artifact),
                        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                        "bytes": len(artifact_bytes),
                        "media_type": "chemical/x-pdb",
                    }
                ],
                "metrics": {"confidence": 0.5},
                "candidate_ids": [],
            }
            self.assertEqual(store.finish(result, "worker-0"), "completed")
            status = store.status()
            self.assertEqual(status["counts"]["completed"], 1)
            self.assertEqual(status["attempts"], 1)

            exported = store.export()
            self.assertEqual(exported["work_items"], 1)
            self.assertEqual(exported["attempts"], 1)
            self.assertTrue(Path(exported["work_items_path"]).is_file())
            self.assertTrue(Path(exported["attempts_path"]).is_file())

    def test_expired_lease_is_reclaimed_and_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, manifest = self.initialize_approved_run(Path(temp_dir))
            store = RunStore(run_dir)
            prepared, _ = store.enqueue(work_item(manifest))
            first = store.claim("worker-0", lease_seconds=3600)
            self.assertEqual(first["queue"]["attempt"], 1)
            renewed = store.renew(
                prepared["work_id"], "worker-0", attempt=1, lease_seconds=7200
            )
            self.assertEqual(renewed["attempt"], 1)
            with self.assertRaisesRegex(ContractError, "another worker"):
                store.renew(
                    prepared["work_id"], "worker-1", attempt=1, lease_seconds=7200
                )
            with sqlite3.connect(run_dir / "queue.sqlite3") as connection:
                connection.execute(
                    "UPDATE work_items SET lease_expires_at = 0 WHERE work_id = ?",
                    (prepared["work_id"],),
                )
            with self.assertRaisesRegex(ContractError, "lease has expired"):
                store.renew(
                    prepared["work_id"], "worker-0", attempt=1, lease_seconds=7200
                )
            expired_result = {
                "schema_version": "protein-mrna.tool-result.v1",
                "work_id": prepared["work_id"],
                "worker_id": "worker-0",
                "attempt": 1,
                "status": "retryable",
                "started_at_utc": first["queue"]["claimed_at_utc"],
                "finished_at_utc": first["queue"]["claimed_at_utc"],
                "runtime_seconds": 1.0,
                "artifacts": [],
                "metrics": {},
                "candidate_ids": [],
                "error": {"type": "LeaseFixture", "message": "expired lease"},
            }
            with self.assertRaisesRegex(ContractError, "lease has expired"):
                store.finish(expired_result, "worker-0")
            second = store.claim("worker-1", lease_seconds=3600)
            self.assertEqual(second["queue"]["attempt"], 2)
            with sqlite3.connect(run_dir / "queue.sqlite3") as connection:
                statuses = connection.execute(
                    "SELECT status FROM attempts ORDER BY attempt"
                ).fetchall()
            self.assertEqual(statuses, [("lease_expired",), ("running",)])

    def test_concurrent_workers_cannot_claim_the_same_attempt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, manifest = self.initialize_approved_run(Path(temp_dir))
            store = RunStore(run_dir)
            store.enqueue(work_item(manifest))

            def claim(worker_id):
                return RunStore(run_dir).claim(worker_id, lease_seconds=3600)

            with ThreadPoolExecutor(max_workers=2) as executor:
                claims = list(executor.map(claim, ("worker-0", "worker-1")))
            self.assertEqual(sum(claimed is not None for claimed in claims), 1)
            self.assertEqual(store.status()["attempts"], 1)

    def test_retryable_result_stops_at_max_attempts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, manifest = self.initialize_approved_run(Path(temp_dir))
            store = RunStore(run_dir)
            item = work_item(manifest)
            item["max_attempts"] = 1
            prepared, _ = store.enqueue(item)
            claim = store.claim("worker-0", lease_seconds=3600)
            result = {
                "schema_version": "protein-mrna.tool-result.v1",
                "work_id": prepared["work_id"],
                "worker_id": "worker-0",
                "attempt": 1,
                "status": "retryable",
                "started_at_utc": claim["queue"]["claimed_at_utc"],
                "finished_at_utc": claim["queue"]["claimed_at_utc"],
                "runtime_seconds": 1.0,
                "artifacts": [],
                "metrics": {},
                "candidate_ids": [],
                "error": {"type": "FixtureError", "message": "retry fixture"},
            }
            self.assertEqual(store.finish(result, "worker-0"), "failed")
            self.assertIsNone(store.claim("worker-1", lease_seconds=3600))
            self.assertEqual(store.status()["counts"]["failed"], 1)

    def test_bad_artifact_hash_cannot_complete_work(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, manifest = self.initialize_approved_run(Path(temp_dir))
            store = RunStore(run_dir)
            prepared, _ = store.enqueue(work_item(manifest))
            claim = store.claim("worker-0", lease_seconds=3600)
            work_dir = run_dir / claim["queue"]["work_dir"]
            artifact_path = work_dir / "bad.pdb"
            artifact_path.write_bytes(b"fixture")
            result = {
                "schema_version": "protein-mrna.tool-result.v1",
                "work_id": prepared["work_id"],
                "worker_id": "worker-0",
                "attempt": 1,
                "status": "succeeded",
                "started_at_utc": claim["queue"]["claimed_at_utc"],
                "finished_at_utc": claim["queue"]["claimed_at_utc"],
                "runtime_seconds": 1.0,
                "artifacts": [
                    {
                        "artifact_id": "bad-fixture",
                        "path": str(artifact_path.relative_to(run_dir)),
                        "sha256": "0" * 64,
                        "bytes": len(b"fixture"),
                        "media_type": "chemical/x-pdb",
                    }
                ],
                "metrics": {},
                "candidate_ids": [],
            }
            with self.assertRaisesRegex(ContractError, "SHA256 mismatch"):
                store.finish(result, "worker-0")
            self.assertEqual(store.status()["counts"]["running"], 1)


if __name__ == "__main__":
    unittest.main()
