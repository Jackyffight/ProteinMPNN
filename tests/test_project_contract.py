from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherContractTest(unittest.TestCase):
    def test_launcher_has_reproducible_training_outputs(self):
        launcher = (ROOT / "run_train.sh").read_text(encoding="utf-8")

        self.assertIn("--seed", launcher)
        self.assertIn("--num_loader_workers", launcher)
        self.assertIn("--prefetch_workers", launcher)
        self.assertIn("--save_best", launcher)
        self.assertIn("run_name", launcher.lower())

    def test_v100_preset_uses_smaller_token_budget_than_a100(self):
        launcher = (ROOT / "run_train.sh").read_text(encoding="utf-8")

        self.assertIn("BATCH_TOKENS=\"${BATCH_TOKENS:-6000}\"", launcher)
        self.assertIn("BATCH_TOKENS=\"${BATCH_TOKENS:-10000}\"", launcher)

    def test_dataset_download_script_uses_range_parts_and_checksum(self):
        script = (ROOT / "scripts/download_dataset_parts.sh").read_text(encoding="utf-8")
        stage_script = (ROOT / "scripts/stage_existing_dataset.sh").read_text(encoding="utf-8")

        self.assertIn("--range", script)
        self.assertIn("EXPECTED_SHA256", script)
        self.assertIn("EXPECTED_SIZE", script)
        self.assertIn("--http1.1", script)
        self.assertIn("download_logs", script)
        self.assertIn("pdb_2021aug02.tar.gz", script)
        self.assertIn("18037128263", script)
        self.assertIn("/data00/home/wangzhi.wit/models/datasets/proteinmpnn", stage_script)
        self.assertIn("sha256sum", stage_script)

    def test_latest_pdb_dataset_track_is_scaffolded(self):
        sync_script = (ROOT / "scripts/sync_latest_pdb_assemblies.sh").read_text(encoding="utf-8")
        build_script = (ROOT / "scripts/build_pdb_2026_dataset.sh").read_text(encoding="utf-8")
        cluster_script = (ROOT / "scripts/download_rcsb_sequence_clusters.sh").read_text(encoding="utf-8")
        builder = (ROOT / "repo/training/build_pdb_mmcif_dataset.py").read_text(encoding="utf-8")
        versions_doc = (ROOT / "DATASET_VERSIONS.md").read_text(encoding="utf-8")

        self.assertIn("files.wwpdb.org/pub/pdb/data/assemblies/mmCIF/divided", sync_script)
        self.assertIn("rsync.rcsb.org", sync_script)
        self.assertIn("--dry-run", sync_script)
        self.assertIn("proteinmpnn_pdb_20260708", build_script)
        self.assertIn("clusters-by-entity-${SEQ_ID}.txt", cluster_script)
        self.assertIn("MMCIF2Dict", builder)
        self.assertIn("list.csv", builder)
        self.assertIn("valid_clusters.txt", builder)
        self.assertIn("asmb_xform0", builder)
        self.assertIn("proteinmpnn_pdb_latest_<YYYYMMDD>", versions_doc)
        self.assertIn("Upstream Reference Baseline", versions_doc)

    def test_throughput_benchmark_sweeps_mpnn_parameters(self):
        script = (ROOT / "scripts/benchmark_throughput.sh").read_text(encoding="utf-8")
        printer = (ROOT / "scripts/print_throughput_benchmark.sh").read_text(encoding="utf-8")

        self.assertIn("batch_tokens", script)
        self.assertIn("loader_workers", script)
        self.assertIn("prefetch_workers", script)
        self.assertIn("examples_per_second", script)
        self.assertIn("metrics.jsonl", script)
        self.assertIn("examples_per_second", printer)

    def test_baseline_from_scratch_runs_all_required_phases(self):
        script = (ROOT / "scripts/run_baseline_from_scratch.sh").read_text(encoding="utf-8")
        runbook = (ROOT / "BASELINE_RUNBOOK.md").read_text(encoding="utf-8")

        self.assertIn("download_dataset_parts.sh --extract", script)
        self.assertIn("validate_dataset.sh", script)
        self.assertIn("smoke_train.sh", script)
        self.assertIn("full_sanity.sh", script)
        self.assertIn("benchmark_throughput.sh", script)
        self.assertIn("full_train_${PROFILE}.sh", script)
        self.assertIn("Baseline From-Scratch Runbook", runbook)

    def test_nas_environment_paths_are_pinned(self):
        env_script = (ROOT / "scripts/env_nas.sh").read_text(encoding="utf-8")
        runbook = (ROOT / "BASELINE_RUNBOOK.md").read_text(encoding="utf-8")

        self.assertIn("/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN", env_script)
        self.assertIn("PROTEINMPNN_DATA_ROOT", env_script)
        self.assertIn("PROTEINMPNN_OUTPUT_ROOT", env_script)
        self.assertIn("/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN", runbook)


class TrainingContractTest(unittest.TestCase):
    def test_training_writes_manifest_metrics_and_best_checkpoint(self):
        training = (ROOT / "repo/training/training.py").read_text(encoding="utf-8")

        self.assertIn("run_manifest.json", training)
        self.assertIn("metrics.jsonl", training)
        self.assertIn("eval_results.json", training)
        self.assertIn("model_weights/best.pt", training)
        self.assertIn("best_validation_loss", training)

    def test_training_exposes_seed_and_runtime_controls(self):
        training = (ROOT / "repo/training/training.py").read_text(encoding="utf-8")

        self.assertIn("--seed", training)
        self.assertIn("--num_loader_workers", training)
        self.assertIn("--prefetch_workers", training)
        self.assertIn("--tf32", training)


if __name__ == "__main__":
    unittest.main()
