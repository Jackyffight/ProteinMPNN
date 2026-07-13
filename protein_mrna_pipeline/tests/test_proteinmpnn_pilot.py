import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from protein_mrna_pipeline.design_refold import (  # noqa: E402
    _summarize_evaluation,
    evaluate_design_refolds,
    load_completed_refolds,
    run_design_refolds,
)
from protein_mrna_pipeline.benchmark import sha256_file  # noqa: E402
from protein_mrna_pipeline.contracts import ContractError  # noqa: E402
from protein_mrna_pipeline.esmfold2_runner import (  # noqa: E402
    BIOHUB_TRANSFORMERS_COMMIT,
    BIOHUB_TRANSFORMERS_VERSION,
    ESMC_6B_REPOSITORY,
    ESMC_6B_REVISION,
    ESMFOLD2_FAST_REPOSITORY,
    ESMFOLD2_FAST_REVISION,
    FoldOutput,
    _identity as esmfold2_identity,
    _weight_manifest,
)
from protein_mrna_pipeline.proteinmpnn_pilot import (  # noqa: E402
    _identity,
    _protein_dict_from_payload,
    generate_paired_design_pilot,
    load_generated_pilot,
    select_pilot_backbones,
)
from protein_mrna_pipeline.run_store import (  # noqa: E402
    write_json_atomic,
    write_jsonl_atomic,
)


try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metrics_runtime_fixture():
    document = {
        "schema_version": "protein-mrna.structure-metrics-runtime.v1",
        "runtime_identity": "pending",
        "created_at_utc": "2026-07-13T00:00:00+00:00",
        "runtime_root": "/fixture/metrics",
        "environment": {"distributions": {}},
        "implementation": {"library": "Biotite", "version": "1.6.0"},
    }
    document["runtime_identity"] = _identity(document, "runtime_identity")
    return document


def esmfold2_runtime_fixture():
    document = {
        "schema_version": "protein-mrna.esmfold2-runtime.v1",
        "runtime_identity": "pending",
        "created_at_utc": "2026-07-13T00:00:00+00:00",
        "runtime_root": "/fixture/esmfold2",
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
        "environment": {"python": "fixture", "torch": "fixture"},
    }
    document["runtime_identity"] = esmfold2_identity(document, "runtime_identity")
    return document


def agreement_record(record_id, chain_id, length, lddt, tm, coverage=1.0):
    return {
        "benchmark_record_id": record_id,
        "source_chain_id": chain_id,
        "sequence_length": length,
        "length_bin": "fixture-bin",
        "metrics": {
            "ca_lddt": lddt,
            "ca_tm_score_resolved": tm,
            "ca_tm_score_full_length": tm * coverage,
            "ca_rmsd_angstrom": 1.0,
            "native_ca_coverage": coverage,
            "native_complete_backbone_coverage": coverage,
        },
    }


class FakeDesignBackend:
    calls = []

    def __init__(self, repository, checkpoint_path, checkpoint, device):
        self.label = checkpoint["label"]

    def generate(self, payload, source_chain_id, native_sequence, seed, temperature):
        type(self).calls.append((self.label, source_chain_id, seed, temperature))
        replacement = "A" if native_sequence[0] != "A" else "C"
        sequence = replacement + native_sequence[1:]
        return {
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            "designable_positions": len(sequence),
            "fixed_missing_positions": 0,
            "mutation_count": 1,
            "sequence_recovery": 1.0 - 1.0 / len(sequence),
            "sampled_nll": 1.0 if self.label == "official-v48-020" else 0.9,
            "native_nll_same_order": 0.8,
            "runtime_seconds": 0.01,
        }

    def close(self):
        return None


class FakeFoldBackend:
    calls = []
    load_seconds = 1.0

    def __init__(self, runtime_root):
        self.runtime_root = runtime_root

    def fold(self, sequence, parameters, seed):
        type(self).calls.append((sequence, dict(parameters), seed))
        return FoldOutput(
            pdb_text=(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
                "  1.00  0.50           C\nEND\n"
            ),
            metrics={"mean_plddt": 0.8, "ptm": 0.7},
            peak_gpu_memory_allocated_bytes=100,
            peak_gpu_memory_reserved_bytes=200,
        )

    def recover_after_failure(self):
        return None


class ProteinMPNNPilotTest(unittest.TestCase):
    def test_selection_covers_four_distinct_failure_modes(self):
        records = [
            agreement_record("low-lddt", "1aaa_A", 100, 0.5, 0.7, 0.9),
            agreement_record("low-tm", "1aab_A", 200, 0.7, 0.4, 0.99),
            agreement_record("longest", "1aac_A", 800, 0.8, 0.8),
            agreement_record("best", "1aad_A", 150, 0.99, 0.95),
            agreement_record("other", "1aae_A", 300, 0.9, 0.9),
        ]
        selected = select_pilot_backbones(records)
        self.assertEqual(
            [row["benchmark_record_id"] for row in selected],
            ["low-lddt", "low-tm", "longest", "best"],
        )
        self.assertEqual(len({row["benchmark_record_id"] for row in selected}), 4)

    def test_generation_and_refolding_are_identity_bound_and_resumable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sequence = "C" * 60
            suite_records = [
                {
                    "benchmark_record_id": f"record-{index:04d}",
                    "source_chain_id": f"fixture{index}_A",
                    "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                    "length": len(sequence),
                    "length_bin": "fixture-bin",
                }
                for index in range(1, 6)
            ]
            suite = {
                "benchmark_id": "fixture-benchmark",
                "source": {"split": "valid"},
                "records": suite_records,
            }
            suite_path = root / "suite.json"
            write_json(suite_path, suite)
            agreement_rows = [
                agreement_record("record-0001", "fixture1_A", 60, 0.5, 0.7, 0.9),
                agreement_record("record-0002", "fixture2_A", 60, 0.7, 0.4, 0.99),
                agreement_record("record-0003", "fixture3_A", 60, 0.8, 0.8),
                agreement_record("record-0004", "fixture4_A", 60, 0.99, 0.95),
                agreement_record("record-0005", "fixture5_A", 60, 0.9, 0.9),
            ]
            agreement_dir = root / "agreement"
            write_json(agreement_dir / "summary.json", {"fixture": True})
            (agreement_dir / "records.jsonl").write_text("fixture\n", encoding="utf-8")
            official = root / "official.pt"
            stage2a = root / "stage2a.pt"
            official.write_bytes(b"official")
            stage2a.write_bytes(b"stage2a")
            expected_hashes = {
                "official-v48-020": hashlib.sha256(official.read_bytes()).hexdigest(),
                "stage2a": hashlib.sha256(stage2a.read_bytes()).hexdigest(),
            }

            def checkpoint_loader(path):
                return {
                    "model_state_dict": {},
                    "label": (
                        "official-v48-020" if path == official else "stage2a"
                    ),
                    "num_edges": 48,
                }

            FakeDesignBackend.calls = []
            with (
                mock.patch(
                    "protein_mrna_pipeline.proteinmpnn_pilot.verify_benchmark_suite_files"
                ),
                mock.patch(
                    "protein_mrna_pipeline.proteinmpnn_pilot.verify_esmfold2_benchmark_run"
                ),
                mock.patch(
                    "protein_mrna_pipeline.proteinmpnn_pilot._validate_native_agreement",
                    return_value=(
                        {"evaluation_identity": "fixture-agreement"},
                        agreement_rows,
                    ),
                ),
                mock.patch(
                    "protein_mrna_pipeline.proteinmpnn_pilot._verify_dataset",
                    return_value={"dataset_dir": str(root / "dataset"), "shards": []},
                ),
                mock.patch(
                    "protein_mrna_pipeline.proteinmpnn_pilot._selected_index_rows",
                    return_value={
                        row["source_chain_id"]: {
                            "chain_id": row["source_chain_id"],
                            "sequence_length": 60,
                        }
                        for row in suite_records
                    },
                ),
                mock.patch(
                    "protein_mrna_pipeline.proteinmpnn_pilot._validate_selected_index_rows"
                ),
            ):
                summary = generate_paired_design_pilot(
                    suite_path,
                    agreement_dir,
                    root / "native-prediction",
                    root / "dataset",
                    official,
                    stage2a,
                    root / "pilot",
                    root / "metrics-runtime",
                    REPOSITORY_ROOT,
                    payload_loader=lambda dataset, row: {"fixture": row["chain_id"]},
                    backend_factory=FakeDesignBackend,
                    checkpoint_loader=checkpoint_loader,
                    expected_checkpoint_sha256=expected_hashes,
                    metrics_runtime_document=metrics_runtime_fixture(),
                )
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["designs"], 32)
            self.assertEqual(len(FakeDesignBackend.calls), 32)
            pilot_manifest, designs, _ = load_generated_pilot(root / "pilot")
            self.assertEqual(len(designs), 32)
            self.assertEqual(
                {design["seed"] for design in designs}, {11, 23, 42, 67}
            )

            FakeFoldBackend.calls = []
            refold_summary = run_design_refolds(
                root / "pilot",
                root / "refolds",
                root / "esmfold2-runtime",
                backend_factory=FakeFoldBackend,
                runtime_document=esmfold2_runtime_fixture(),
            )
            self.assertEqual(refold_summary["status"], "passed")
            self.assertEqual(len(FakeFoldBackend.calls), 32)

            def must_not_load(runtime_root):
                raise AssertionError("completed refolds must not reload ESMFold2")

            resumed = run_design_refolds(
                root / "pilot",
                root / "refolds",
                root / "esmfold2-runtime",
                backend_factory=must_not_load,
                runtime_document=esmfold2_runtime_fixture(),
            )
            self.assertEqual(resumed["status"], "passed")

            first_refold_path = (
                root
                / "refolds"
                / "records"
                / designs[0]["design_id"]
                / "result.json"
            )
            first_refold = json.loads(first_refold_path.read_text(encoding="utf-8"))
            tampered_refold = {**first_refold, "model_label": "stage2a"}
            tampered_refold["result_identity"] = _identity(
                tampered_refold, "result_identity"
            )
            write_json_atomic(first_refold_path, tampered_refold)
            with self.assertRaises(ContractError):
                load_completed_refolds(root / "pilot", root / "refolds")
            write_json_atomic(first_refold_path, first_refold)

            for record in suite_records:
                write_json(
                    root
                    / "native-prediction"
                    / "records"
                    / record["benchmark_record_id"]
                    / "result.json",
                    {"artifact": {"path": "fixture-native.pdb"}},
                )

            def native_extractor(payload, source_chain_id, expected_sequence):
                return {
                    "ca_coordinates": None,
                    "ca_mask": None,
                    "ca_coverage": 1.0,
                }

            def metric_function(reference, subject, mask, sequence):
                return {
                    "ca_lddt": 0.8,
                    "ca_rmsd_angstrom": 2.0,
                    "ca_tm_score_full_length": 0.8,
                    "ca_tm_score_resolved": 0.8,
                }

            with (
                mock.patch(
                    "protein_mrna_pipeline.design_refold.verify_benchmark_suite_files"
                ),
                mock.patch(
                    "protein_mrna_pipeline.design_refold.verify_esmfold2_benchmark_run"
                ),
                mock.patch(
                    "protein_mrna_pipeline.design_refold._validate_native_agreement",
                    return_value=(
                        {"evaluation_identity": "fixture-agreement"},
                        agreement_rows,
                    ),
                ),
                mock.patch(
                    "protein_mrna_pipeline.design_refold._verify_dataset",
                    return_value={"dataset_dir": str(root / "dataset"), "shards": []},
                ),
                mock.patch(
                    "protein_mrna_pipeline.design_refold._selected_index_rows",
                    return_value={
                        row["source_chain_id"]: {"chain_id": row["source_chain_id"]}
                        for row in suite_records
                    },
                ),
                mock.patch(
                    "protein_mrna_pipeline.design_refold._validate_selected_index_rows"
                ),
            ):
                evaluated = evaluate_design_refolds(
                    root / "pilot",
                    root / "refolds",
                    root / "evaluated",
                    root / "metrics-runtime",
                    payload_loader=lambda dataset, row: {"fixture": True},
                    native_extractor=native_extractor,
                    prediction_parser=lambda path, sequence: None,
                    metric_function=metric_function,
                    metrics_runtime_document=metrics_runtime_fixture(),
                )
            self.assertEqual(evaluated["status"], "passed")
            self.assertEqual(evaluated["records"]["succeeded"], 32)
            self.assertEqual(evaluated["paired"]["pairs"], 16)

            evaluation_dir = root / "evaluation"
            evaluation_dir.mkdir()
            evaluation_identity = "fixture-evaluation"
            for design in designs:
                lddt = 0.8 if design["model_label"] == "official-v48-020" else 0.85
                rmsd = 2.0 if design["model_label"] == "official-v48-020" else 1.8
                result = {
                    "schema_version": (
                        "protein-mrna.proteinmpnn-refold-evaluation-record.v1"
                    ),
                    "result_identity": "pending",
                    "evaluation_identity": evaluation_identity,
                    "design_id": design["design_id"],
                    "design_identity": design["design_identity"],
                    "benchmark_record_id": design["benchmark_record_id"],
                    "selection_role": design["selection_role"],
                    "model_label": design["model_label"],
                    "sampling_seed": design["seed"],
                    "status": "succeeded",
                    "runtime_seconds": 0.01,
                    "design": {
                        "sequence_sha256": design["sequence_sha256"],
                        "sequence_length": design["sequence_length"],
                        "designable_positions": design["designable_positions"],
                        "mutation_count": design["mutation_count"],
                        "sequence_recovery": design["sequence_recovery"],
                        "sampled_nll": design["sampled_nll"],
                    },
                    "experimental_native": {
                        "ca_lddt": lddt,
                        "ca_tm_score_resolved": lddt,
                        "ca_tm_score_full_length": lddt,
                        "ca_rmsd_angstrom": rmsd,
                    },
                    "native_sequence_prediction_reference": {
                        "ca_lddt": lddt,
                        "ca_tm_score_full_length": lddt,
                        "ca_rmsd_angstrom": 1.5,
                    },
                    "delta_vs_native_sequence_baseline": {
                        "ca_lddt": lddt - 0.9,
                        "ca_tm_score_resolved": lddt - 0.9,
                        "ca_tm_score_full_length": lddt - 0.9,
                        "ca_rmsd_angstrom": rmsd - 1.0,
                    },
                    "refold_confidence": {"mean_plddt": 0.8, "ptm": 0.7},
                }
                result["result_identity"] = _identity(result, "result_identity")
                write_json_atomic(
                    evaluation_dir / "records" / f"{design['design_id']}.json",
                    result,
                )
            evaluation_summary = _summarize_evaluation(
                evaluation_dir, designs, evaluation_identity
            )
            self.assertEqual(evaluation_summary["paired"]["pairs"], 16)
            self.assertAlmostEqual(
                evaluation_summary["paired"]["mean_delta_stage2a_minus_official"][
                    "experimental_ca_lddt"
                ],
                0.05,
            )
            self.assertEqual(
                evaluation_summary["paired"]["stage2a_wins"][
                    "experimental_ca_lddt"
                ],
                16,
            )
            self.assertEqual(
                evaluation_summary["paired"]["stage2a_lower_is_better_wins"][
                    "experimental_ca_rmsd_angstrom"
                ],
                16,
            )
            self.assertEqual(pilot_manifest["sampling"]["paired_seed_policy"], True)

            tampered_designs = list(designs)
            tampered_designs[0] = {
                **tampered_designs[0],
                "model_label": "stage2a",
            }
            tampered_designs[0]["design_identity"] = _identity(
                tampered_designs[0], "design_identity"
            )
            designs_path = root / "pilot/designs.jsonl"
            write_jsonl_atomic(designs_path, tampered_designs)
            generation_summary = json.loads(
                (root / "pilot/generation-summary.json").read_text(encoding="utf-8")
            )
            generation_summary["designs_sha256"] = sha256_file(designs_path)
            write_json_atomic(
                root / "pilot/generation-summary.json", generation_summary
            )
            with self.assertRaises(ContractError):
                load_generated_pilot(root / "pilot")

    @unittest.skipUnless(HAS_NUMPY, "NumPy is not installed")
    def test_payload_adapter_preserves_context_and_missing_target_positions(self):
        sequence = "ACDE"
        xyz = np.zeros((4, 14, 3), dtype=np.float32)
        mask = np.ones((4, 14), dtype=bool)
        xyz[2, :4] = np.nan
        mask[2, :4] = False
        payload = {
            "format": "proteinmpnn.tar_shard.v2",
            "target_chain_ids": ["1abc_B"],
            "meta": {"target_chain": "B", "chains": ["A", "B"]},
            "chains": {
                "A": {"seq": sequence, "xyz": np.zeros_like(xyz), "mask": np.ones_like(mask)},
                "B": {"seq": sequence, "xyz": xyz, "mask": mask},
            },
        }
        protein, chain_dict, target = _protein_dict_from_payload(
            payload, "1abc_B", sequence
        )
        self.assertEqual(target, "B")
        self.assertEqual(chain_dict["1abc_B"], (["B"], ["A"]))
        self.assertEqual(protein["num_of_chains"], 2)
        self.assertTrue(np.isnan(protein["coords_chain_B"]["N_chain_B"][2]).all())

    def test_one_command_launcher_is_bounded_and_valid_only(self):
        launcher = (
            REPOSITORY_ROOT / "scripts/run_proteinmpnn_refold_pilot.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("4 backbones x 2 models x 4 paired seeds = 32", launcher)
        self.assertIn("--seeds 11 23 42 67", launcher)
        self.assertIn("--temperature 0.1", launcher)
        self.assertIn("PROTEINMPNN_V1_DATA_DIR", launcher)
        self.assertIn("promoted/proteinmpnn-2026-stage2a/model.pt", launcher)
        self.assertIn("expose exactly one GPU", launcher)
        self.assertNotIn("test_clusters", launcher)

    def test_compact_report_reads_paired_summary(self):
        report = REPOSITORY_ROOT / "scripts/report_proteinmpnn_refold_pilot.sh"
        fields = (
            "sequence_recovery",
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
        with tempfile.TemporaryDirectory() as temp_dir:
            pilot_dir = Path(temp_dir) / "pilot"
            evaluation_dir = pilot_dir / "evaluations/dual-reference-v1"
            write_json(
                pilot_dir / "pilot-manifest.json",
                {
                    "selection": {
                        "records": [
                            {
                                "selection_role": "lowest_ca_lddt",
                                "benchmark_record_id": f"record-{index:04d}",
                                "source_chain_id": f"fixture{index}_A",
                                "sequence_length": 100 + index,
                            }
                            for index in range(1, 5)
                        ]
                    }
                },
            )
            write_json(
                evaluation_dir / "summary.json",
                {
                    "schema_version": (
                        "protein-mrna.proteinmpnn-refold-evaluation-summary.v1"
                    ),
                    "evaluation_identity": "fixture",
                    "status": "passed",
                    "records": {
                        "selected": 32,
                        "succeeded": 32,
                        "failed": 0,
                        "pending": 0,
                    },
                    "by_model": {
                        label: {
                            field: {"mean": 0.8 if label == "official-v48-020" else 0.85}
                            for field in fields
                        }
                        for label in ("official-v48-020", "stage2a")
                    },
                    "by_backbone": {
                        f"record-{index:04d}": {
                            label: {
                                field: {
                                    "mean": (
                                        0.8
                                        if label == "official-v48-020"
                                        else 0.85
                                    )
                                }
                                for field in fields
                            }
                            for label in ("official-v48-020", "stage2a")
                        }
                        for index in range(1, 5)
                    },
                    "paired": {
                        "pairs": 16,
                        "stage2a_wins": {"experimental_ca_lddt": 12},
                        "stage2a_lower_is_better_wins": {
                            "experimental_ca_rmsd_angstrom": 10
                        },
                    },
                },
            )
            completed = subprocess.run(
                [report],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PILOT_DIR": str(pilot_dir),
                    "EVALUATION_DIR": str(evaluation_dir),
                    "PROTEINMPNN_PYTHON": sys.executable,
                },
            )
            self.assertIn("paired comparisons: 16", completed.stdout)
            self.assertIn("experimental CA lDDT", completed.stdout)
            self.assertIn("+0.0500", completed.stdout)
            self.assertIn("per-backbone delta", completed.stdout)
            self.assertIn("record-0004", completed.stdout)


if __name__ == "__main__":
    unittest.main()
