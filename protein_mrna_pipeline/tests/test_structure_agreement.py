import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from protein_mrna_pipeline.benchmark import generate_benchmark_suite  # noqa: E402
from protein_mrna_pipeline.contracts import ContractError, read_json  # noqa: E402
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
    run_esmfold2_benchmark,
)
from protein_mrna_pipeline.structure_agreement import (  # noqa: E402
    METRICS_RUNTIME_VERSIONS,
    _identity as metrics_identity,
    compute_ca_metrics,
    evaluate_native_structure_agreement,
    extract_native_ca,
    parse_prediction_ca,
)


try:
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    import numpy as np

    HAS_STRUCTURE_METRICS = True
except ImportError:
    HAS_STRUCTURE_METRICS = False


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_benchmark_dataset(root):
    root.mkdir()
    (root / "shards").mkdir()
    shard = root / "shards/shard_000000.tar"
    shard.write_bytes(b"\0" * 4096)
    lengths = (60, 120, 250, 350, 450, 550, 650, 750)
    record_count = len(lengths) + 1
    write_json(
        root / "manifest.json",
        {
            "format": "proteinmpnn.tar_shard.v2",
            "version_id": "fixture-pdb-2026",
            "record_count": record_count,
            "payload_schema": "structure_with_target_chain_ids",
            "shards": [
                {
                    "name": shard.name,
                    "bytes": shard.stat().st_size,
                    "records": record_count,
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    write_json(
        root / "validation.json",
        {
            "schema": "proteinmpnn.tar_shard_validation.v2",
            "status": "ok",
            "exact_sequence_split_leaks": 0,
            "pdb_split_leaks": 0,
            "records": record_count,
            "payloads_checked": record_count,
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
    index_rows = []
    for index, length in enumerate(lengths, 1):
        chain_id = f"fixture{index}_A"
        rows.append(
            f"{chain_id},2026-01-{index:02d},2.00,h{index},{index},"
            f"{sequence(length, index)}\n"
        )
        index_rows.append(
            {
                "chain_id": chain_id,
                "entry_id": f"fixture{index}",
                "sequence_length": length,
                "shard": "shards/shard_000000.tar",
                "offset": 512,
                "size": 1,
            }
        )
    test_sequence = sequence(100, 9)
    rows.append(f"test_A,2026-01-09,2.00,h9,99,{test_sequence}\n")
    index_rows.append(
        {
            "chain_id": "test_A",
            "entry_id": "test",
            "sequence_length": len(test_sequence),
            "shard": "shards/shard_000000.tar",
            "offset": 512,
            "size": 1,
        }
    )
    (root / "list.csv").write_text("".join(rows), encoding="utf-8")
    (root / "index.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows),
        encoding="utf-8",
    )


def esmfold2_runtime_fixture():
    document = {
        "schema_version": "protein-mrna.esmfold2-runtime.v1",
        "runtime_identity": "pending",
        "created_at_utc": "2026-07-13T00:00:00+00:00",
        "runtime_root": "/fixture/esmfold2-runtime",
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


def metrics_runtime_fixture():
    document = {
        "schema_version": "protein-mrna.structure-metrics-runtime.v1",
        "runtime_identity": "pending",
        "created_at_utc": "2026-07-13T00:00:00+00:00",
        "runtime_root": "/fixture/metrics-runtime",
        "environment": {"distributions": METRICS_RUNTIME_VERSIONS},
        "implementation": {"library": "Biotite", "version": "1.6.0"},
    }
    document["runtime_identity"] = metrics_identity(document, "runtime_identity")
    return document


class FakeBackend:
    load_seconds = 0.0

    def __init__(self, runtime_root):
        self.runtime_root = runtime_root

    def fold(self, sequence, parameters, seed):
        return FoldOutput(
            pdb_text=(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
                "  1.00  0.50           C\nEND\n"
            ),
            metrics={"mean_plddt": len(sequence) / 1000.0, "ptm": 0.5},
            peak_gpu_memory_allocated_bytes=0,
            peak_gpu_memory_reserved_bytes=0,
        )

    def recover_after_failure(self):
        return None


def fake_native_extractor(payload, source_chain_id, sequence):
    return {
        "ca_coordinates": None,
        "ca_mask": None,
        "ca_residues": len(sequence),
        "ca_coverage": 1.0,
        "complete_backbone_residues": len(sequence),
        "complete_backbone_coverage": 1.0,
        "source_pdb_id": source_chain_id[:4],
        "source_assembly_id": "1",
        "source_mmcif_chain_id": "A",
        "source_entity_id": "1",
    }


def fake_metric_function(native, prediction, mask, sequence):
    return {
        "ca_lddt": len(sequence) / 1000.0,
        "ca_rmsd_angstrom": 1.0,
        "ca_tm_score_full_length": 0.5,
        "ca_tm_score_resolved": 0.5,
    }


class StructureAgreementTest(unittest.TestCase):
    def test_evaluation_is_identity_bound_and_resumable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            write_benchmark_dataset(dataset)
            suite = Path(
                generate_benchmark_suite(
                    dataset,
                    root / "suite",
                    requested_count=4,
                )["suite_path"]
            )
            prediction_run = root / "prediction"
            run_esmfold2_benchmark(
                suite,
                prediction_run,
                root / "esmfold2-runtime",
                mode="full",
                backend_factory=FakeBackend,
                runtime_document=esmfold2_runtime_fixture(),
            )

            calls = []

            def payload_loader(dataset_dir, index_row):
                calls.append(index_row["chain_id"])
                return {"fixture": True}

            output = root / "agreement"
            summary = evaluate_native_structure_agreement(
                suite,
                prediction_run,
                dataset,
                output,
                root / "metrics-runtime",
                payload_loader=payload_loader,
                native_extractor=fake_native_extractor,
                prediction_parser=lambda path, sequence: None,
                metric_function=fake_metric_function,
                metrics_runtime_document=metrics_runtime_fixture(),
            )
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["records"]["succeeded"], 4)
            self.assertEqual(len(calls), 4)
            self.assertEqual(len((output / "records.jsonl").read_text().splitlines()), 4)

            def must_not_load(*args):
                raise AssertionError("completed evaluation must resume without rereading payloads")

            resumed = evaluate_native_structure_agreement(
                suite,
                prediction_run,
                dataset,
                output,
                root / "metrics-runtime",
                payload_loader=must_not_load,
                native_extractor=must_not_load,
                prediction_parser=must_not_load,
                metric_function=must_not_load,
                metrics_runtime_document=metrics_runtime_fixture(),
            )
            self.assertEqual(resumed["status"], "passed")

            record_path = output / "records/record-0001.json"
            tampered = read_json(record_path)
            tampered["metrics"]["ca_lddt"] = 0.0
            write_json(record_path, tampered)
            calls.clear()
            repaired = evaluate_native_structure_agreement(
                suite,
                prediction_run,
                dataset,
                output,
                root / "metrics-runtime",
                payload_loader=payload_loader,
                native_extractor=fake_native_extractor,
                prediction_parser=lambda path, sequence: None,
                metric_function=fake_metric_function,
                metrics_runtime_document=metrics_runtime_fixture(),
            )
            self.assertEqual(repaired["status"], "passed")
            self.assertEqual(len(calls), 1)

    @unittest.skipUnless(HAS_STRUCTURE_METRICS, "pinned Biotite runtime is not installed")
    def test_rigid_transform_scores_as_identical_with_missing_native_positions(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        index = np.arange(len(sequence), dtype=np.float32)
        native = np.stack((index * 3.8, np.sin(index), np.cos(index)), axis=1)
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        prediction = native @ rotation.T + np.asarray([20.0, -7.0, 3.0])
        mask = np.ones(len(sequence), dtype=bool)
        mask[-2:] = False
        metrics = compute_ca_metrics(native, prediction, mask, sequence)
        self.assertAlmostEqual(metrics["ca_lddt"], 1.0, places=6)
        self.assertLess(metrics["ca_rmsd_angstrom"], 1e-4)
        self.assertAlmostEqual(metrics["ca_tm_score_resolved"], 1.0, places=6)
        self.assertAlmostEqual(metrics["ca_tm_score_full_length"], 0.9, places=6)

    @unittest.skipUnless(HAS_STRUCTURE_METRICS, "pinned Biotite runtime is not installed")
    def test_native_and_prediction_parsers_preserve_sequence_positions(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        length = len(sequence)
        xyz = np.full((length, 14, 3), np.nan, dtype=np.float32)
        mask = np.zeros((length, 14), dtype=bool)
        xyz[:, :4, :] = 0.0
        mask[:, :4] = True
        payload = {
            "format": "proteinmpnn.tar_shard.v2",
            "target_chain_ids": ["1abc_A"],
            "meta": {"target_chain": "A"},
            "chains": {
                "A": {
                    "seq": sequence,
                    "xyz": xyz,
                    "mask": mask,
                    "source_pdb_id": "1abc",
                    "source_assembly_id": "1",
                    "source_chain_id": "A",
                    "source_entity_id": "1",
                }
            },
        }
        native = extract_native_ca(payload, "1abc_A", sequence)
        self.assertEqual(native["ca_residues"], length)
        self.assertEqual(native["complete_backbone_residues"], length)

        atoms = struc.AtomArray(length)
        atoms.coord = np.stack(
            (np.arange(length) * 3.8, np.zeros(length), np.zeros(length)), axis=1
        )
        atoms.chain_id[:] = "A"
        atoms.res_id = np.arange(1, length + 1)
        atoms.atom_name[:] = "CA"
        atoms.res_name = np.asarray(
            [
                {
                    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU",
                    "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
                    "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
                    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
                    "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
                }[residue]
                for residue in sequence
            ]
        )
        atoms.element[:] = "C"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prediction.pdb"
            pdb = PDBFile()
            pdb.set_structure(atoms)
            pdb.write(path)
            parsed = parse_prediction_ca(path, sequence)
            self.assertEqual(parsed.shape, (length, 3))
            with self.assertRaisesRegex(ContractError, "sequence differs"):
                parse_prediction_ca(path, sequence[::-1])

    def test_launchers_pin_metrics_and_disable_gpu_use(self):
        setup = (PROJECT_ROOT.parent / "scripts/setup_structure_metrics_runtime.sh").read_text()
        runner = (PROJECT_ROOT.parent / "scripts/evaluate_esmfold2_native_agreement.sh").read_text()
        for name, version in METRICS_RUNTIME_VERSIONS.items():
            self.assertIn(f'"{name}=={version}"', setup)
        self.assertIn("link_base_torch_site.py", setup)
        self.assertIn("verify-structure-metrics-runtime", setup)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', runner)
        self.assertIn('METRICS_THREADS="${STRUCTURE_METRICS_THREADS:-1}"', runner)
        self.assertIn('export OMP_NUM_THREADS="$METRICS_THREADS"', runner)
        self.assertIn('[[ "$METRICS_THREADS" =~ ^[1-4]$ ]]', runner)
        self.assertIn("evaluate-esmfold2-native", runner)

    def test_runtime_venvs_do_not_require_system_ensurepip(self):
        setup_paths = (
            PROJECT_ROOT.parent / "scripts/setup_structure_metrics_runtime.sh",
            PROJECT_ROOT.parent / "scripts/setup_esmfold2_fast_runtime.sh",
        )
        for path in setup_paths:
            setup = path.read_text(encoding="utf-8")
            self.assertIn(
                '-m venv --without-pip --system-site-packages',
                setup,
                msg=f"{path.name} must bootstrap pip after venv creation",
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir = Path(temp_dir) / "venv"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "venv",
                    "--without-pip",
                    "--system-site-packages",
                    str(venv_dir),
                ],
                check=True,
            )
            observed_prefix = subprocess.run(
                [
                    venv_dir / "bin/python",
                    "-c",
                    "import sys; print(sys.prefix)",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(Path(observed_prefix).resolve(), venv_dir.resolve())

    def test_native_agreement_report_prints_lowest_lddt_first(self):
        report = PROJECT_ROOT.parent / "scripts/report_esmfold2_native_agreement.sh"

        def stats(first, second):
            return {
                "mean": (first + second) / 2,
                "median": (first + second) / 2,
                "min": min(first, second),
                "max": max(first, second),
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluation_dir = Path(temp_dir)
            identity = "fixture-evaluation"
            summary = {
                "schema_version": "protein-mrna.native-agreement-summary.v1",
                "evaluation_identity": identity,
                "status": "passed",
                "records": {
                    "selected": 2,
                    "succeeded": 2,
                    "failed": 0,
                    "pending": 0,
                },
                "overall": {
                    "ca_lddt": stats(0.7, 0.9),
                    "ca_tm_score_resolved": stats(0.8, 0.95),
                    "ca_tm_score_full_length": stats(0.75, 0.9),
                    "ca_rmsd_angstrom": stats(1.0, 4.0),
                    "native_ca_coverage": stats(0.9, 1.0),
                },
                "confidence_correlations": {
                    "mean_plddt_vs_ca_lddt_pearson": 0.3,
                    "predicted_ptm_vs_ca_tm_full_length_pearson": 0.7,
                },
            }
            write_json(evaluation_dir / "summary.json", summary)
            records = []
            for record_id, lddt in (("record-high", 0.9), ("record-low", 0.7)):
                records.append(
                    {
                        "evaluation_identity": identity,
                        "benchmark_record_id": record_id,
                        "source_chain_id": f"{record_id}_A",
                        "sequence_length": 100,
                        "status": "succeeded",
                        "metrics": {
                            "ca_lddt": lddt,
                            "ca_tm_score_resolved": 0.8,
                            "ca_tm_score_full_length": 0.75,
                            "ca_rmsd_angstrom": 2.0,
                            "native_ca_coverage": 0.9,
                        },
                        "prediction_confidence": {
                            "mean_plddt": 0.85,
                            "ptm": 0.75,
                        },
                    }
                )
            (evaluation_dir / "records.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [report, "1"],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "EVAL_DIR": str(evaluation_dir),
                    "PROTEINMPNN_PYTHON": sys.executable,
                },
            )
            self.assertIn("record-low", completed.stdout)
            self.assertNotIn("record-high", completed.stdout)
            self.assertNotIn("jq ", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
