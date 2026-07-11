import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None
HAS_TRAINING_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("dateutil", "numpy", "torch")
)


class LauncherContractTest(unittest.TestCase):
    def test_launcher_has_reproducible_training_outputs(self):
        launcher = (ROOT / "run_train.sh").read_text(encoding="utf-8")

        self.assertIn("--seed", launcher)
        self.assertIn("--num_loader_workers", launcher)
        self.assertIn("--dataset-format", launcher)
        self.assertIn("--dataset_format", launcher)
        self.assertIn("--prefetch_workers", launcher)
        self.assertIn("--save_best", launcher)
        self.assertIn("run_name", launcher.lower())

    def test_launcher_separates_weight_initialization_from_resume(self):
        launcher = (ROOT / "run_train.sh").read_text(encoding="utf-8")
        training = (ROOT / "repo/training/training.py").read_text(encoding="utf-8")

        self.assertIn("--init-checkpoint", launcher)
        self.assertIn("--init_checkpoint", launcher)
        self.assertIn("--resume and --init-checkpoint are mutually exclusive", launcher)
        self.assertIn("add_mutually_exclusive_group", training)
        self.assertIn("require_resume_state", training)

    def test_launcher_passes_only_the_selected_checkpoint_argument(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            (data_dir / "pdb").mkdir(parents=True)
            (data_dir / "list.csv").write_text(
                "CHAINID,DEPOSITION,RESOLUTION,HASH,CLUSTER,SEQUENCE\n",
                encoding="utf-8",
            )
            (data_dir / "valid_clusters.txt").touch()
            (data_dir / "test_clusters.txt").touch()
            checkpoint = root / "official.pt"
            checkpoint.touch()
            capture_path = root / "args.txt"
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "if [ \"${1:-}\" = - ]; then cat >/dev/null; exit 0; fi\n"
                "printf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PROTEINMPNN_PYTHON"] = str(fake_python)
            env["CAPTURE_ARGS"] = str(capture_path)

            subprocess.run(
                [
                    str(ROOT / "run_train.sh"),
                    "smoke",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(root / "output"),
                    "--init-checkpoint",
                    str(checkpoint),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            arguments = capture_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("--init_checkpoint", arguments)
        self.assertNotIn("--previous_checkpoint", arguments)

    def test_v100_preset_uses_smaller_token_budget_than_a100(self):
        launcher = (ROOT / "run_train.sh").read_text(encoding="utf-8")

        self.assertIn("BATCH_TOKENS=\"${BATCH_TOKENS:-6000}\"", launcher)
        self.assertIn("BATCH_TOKENS=\"${BATCH_TOKENS:-10000}\"", launcher)

    def test_training_process_defaults_are_memory_bounded(self):
        launcher = (ROOT / "run_train.sh").read_text(encoding="utf-8")
        v100 = (ROOT / "scripts/full_train_v100.sh").read_text(encoding="utf-8")
        a100 = (ROOT / "scripts/full_train_a100.sh").read_text(encoding="utf-8")

        self.assertIn('LOADER_WORKERS="${LOADER_WORKERS:-0}"', launcher)
        self.assertIn('PREFETCH_WORKERS="${PREFETCH_WORKERS:-1}"', launcher)
        self.assertIn('--loader-workers "${LOADER_WORKERS:-0}"', v100)
        self.assertIn('--prefetch-workers "${PREFETCH_WORKERS:-1}"', v100)
        self.assertIn('--loader-workers "${LOADER_WORKERS:-0}"', a100)
        self.assertIn('--prefetch-workers "${PREFETCH_WORKERS:-2}"', a100)

    def test_a100_v1_pilot_script_is_guarded_and_dry_runnable(self):
        script_path = ROOT / "scripts/run_2026_v1_pilot_a100.sh"
        script = script_path.read_text(encoding="utf-8")
        checkpoint_script = (ROOT / "scripts/ensure_official_checkpoint.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("PROTEINMPNN_V1_DATA_DIR", script)
        self.assertIn("proteinmpnn.tar_shard.v2", script)
        self.assertIn("structure_with_target_chain_ids", script)
        self.assertIn("do not pass 0,1,2,3", script)
        self.assertIn("--init-checkpoint", script)
        self.assertIn("--dry-run", script)
        self.assertIn("ensure_official_checkpoint.sh", script)
        self.assertIn("dauparas/ProteinMPNN", checkpoint_script)
        self.assertIn("8907e6671bfbfc92303b5f79c4b5e6ce47cdef57", checkpoint_script)
        self.assertIn("6681301", checkpoint_script)
        self.assertIn(
            "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd",
            checkpoint_script,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            shards_dir = data_dir / "shards"
            shards_dir.mkdir(parents=True)
            for filename in (
                "list.csv",
                "index.jsonl",
                "records.jsonl",
                "valid_clusters.txt",
                "test_clusters.txt",
            ):
                (data_dir / filename).write_text("fixture\n", encoding="utf-8")
            (shards_dir / "shard_000000.tar").touch()
            (data_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "proteinmpnn.tar_shard.v2",
                        "payload_schema": "structure_with_target_chain_ids",
                        "record_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (data_dir / "validation.json").write_text(
                json.dumps({"status": "ok", "records": 1, "shards_checked": 1}),
                encoding="utf-8",
            )
            checkpoint = root / "official.pt"
            checkpoint.write_text("fixture", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "DATA_DIR": str(data_dir),
                    "INIT_CHECKPOINT": str(checkpoint),
                    "OUTPUT_DIR": str(root / "output"),
                    "RUN_NAME": "contract-pilot",
                    "PYTHON_BIN": sys.executable,
                    "DEVICES": "0",
                }
            )
            result = subprocess.run(
                [str(script_path), "--dry-run"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(f"--data-dir {data_dir}", result.stdout)
            self.assertIn(f"--init-checkpoint {checkpoint}", result.stdout)

            env["DEVICES"] = "0,1"
            rejected = subprocess.run(
                [str(script_path), "--dry-run"],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("single-GPU", rejected.stderr)

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
        # The staging script verifies integrity; it may probe several host-specific
        # source paths, so assert the integrity check rather than an exact host path.
        self.assertIn("sha256sum", stage_script)
        self.assertIn("pdb_2021aug02", stage_script)

    def test_latest_pdb_dataset_track_is_scaffolded(self):
        sync_script = (ROOT / "scripts/sync_latest_pdb_assemblies.sh").read_text(encoding="utf-8")
        build_script = (ROOT / "scripts/build_pdb_2026_dataset.sh").read_text(encoding="utf-8")
        cluster_script = (ROOT / "scripts/download_rcsb_sequence_clusters.sh").read_text(encoding="utf-8")
        metadata_script = (ROOT / "scripts/download_wwpdb_entries_index.sh").read_text(encoding="utf-8")
        pack_script = (ROOT / "scripts/pack_proteinmpnn_tar_shards.py").read_text(encoding="utf-8")
        build_tar_script = (ROOT / "scripts/build_pdb_2026_tar_shards.sh").read_text(encoding="utf-8")
        shard_reader = (ROOT / "repo/training/tar_shard_utils.py").read_text(encoding="utf-8")
        builder = (ROOT / "repo/training/build_pdb_mmcif_dataset.py").read_text(encoding="utf-8")
        tar_builder = (ROOT / "repo/training/build_pdb_mmcif_tar_shard_dataset.py").read_text(encoding="utf-8")
        versions_doc = (ROOT / "DATASET_VERSIONS.md").read_text(encoding="utf-8")

        self.assertIn("files.wwpdb.org/pub/pdb/data/assemblies/mmCIF/divided", sync_script)
        self.assertIn("rsync.rcsb.org", sync_script)
        self.assertIn("--dry-run", sync_script)
        self.assertIn("proteinmpnn_pdb_20260708", build_script)
        self.assertIn("clusters-by-entity-${SEQ_ID}.txt", cluster_script)
        self.assertIn("--continue-at -", cluster_script)
        self.assertIn("--http1.1", cluster_script)
        self.assertIn("entries.idx", metadata_script)
        self.assertIn("load_entry_metadata", builder)
        self.assertIn("MMCIF2Dict", builder)
        self.assertIn("list.csv", builder)
        self.assertIn("valid_clusters.txt", builder)
        self.assertIn("asmb_xform0", builder)
        self.assertIn("proteinmpnn.tar_shard.v1", pack_script)
        self.assertIn("index.jsonl", pack_script)
        self.assertIn("offset", pack_script)
        self.assertIn("loader_tar_pdb", shard_reader)
        self.assertIn("build_pdb_mmcif_tar_shard_dataset.py", build_tar_script)
        self.assertIn("write_pt", builder)
        self.assertIn("return_payload", builder)
        self.assertIn("ProcessPoolExecutor", tar_builder)
        self.assertIn("max_in_flight", tar_builder)
        self.assertIn("--max-in-flight", builder)
        self.assertIn("FIRST_COMPLETED", builder)
        self.assertIn("manifest.json", tar_builder)
        self.assertIn("proteinmpnn.tar_shard.v2", builder)
        self.assertIn("target_chain_ids", builder)
        self.assertIn("validate_tar_shard_dataset.py", build_tar_script)
        self.assertIn("--max-context-length", build_tar_script)
        self.assertIn("structure_with_target_chain_ids", tar_builder)
        self.assertIn("Refusing to build production splits without homology clusters", build_tar_script)
        self.assertIn("dataset_format", (ROOT / "repo/training/training.py").read_text(encoding="utf-8"))
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
        self.assertIn("Baseline and Continued-Training Runbook", runbook)

    def test_nas_environment_paths_are_pinned(self):
        env_script = (ROOT / "scripts/env_nas.sh").read_text(encoding="utf-8")
        runbook = (ROOT / "BASELINE_RUNBOOK.md").read_text(encoding="utf-8")

        self.assertIn("/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN", env_script)
        self.assertIn("PROTEINMPNN_DATA_ROOT", env_script)
        self.assertIn("PROTEINMPNN_TAR_SHARD_DATA_DIR", env_script)
        self.assertIn("PROTEINMPNN_V1_DATA_DIR", env_script)
        self.assertIn("proteinmpnn_tar_shards_v1", env_script)
        self.assertIn("proteinmpnn_tar_shards", env_script)
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
        self.assertIn("--dataset_format", training)
        self.assertIn("loader_tar_pdb", training)
        self.assertIn("--num_loader_workers", training)
        self.assertIn("--prefetch_workers", training)
        self.assertIn("--tf32", training)

    def test_official_checkpoint_evaluator_uses_training_data_path(self):
        evaluator = (ROOT / "repo/training/evaluate_checkpoint.py").read_text(encoding="utf-8")
        wrapper = (ROOT / "scripts/evaluate_official_checkpoint.sh").read_text(encoding="utf-8")

        self.assertIn("build_training_clusters", evaluator)
        self.assertIn("get_pdbs", evaluator)
        self.assertIn("featurize", evaluator)
        self.assertIn("loss_nll", evaluator)
        self.assertIn("checkpoint_evaluation.v1", evaluator)
        self.assertIn("v_48_020.pt", wrapper)

    def test_amp_gradient_clip_unscales_before_clipping(self):
        # Regression: clipping must run on real (unscaled) gradients under AMP.
        training = (ROOT / "repo/training/training.py").read_text(encoding="utf-8")
        unscale = training.index("scaler.unscale_(optimizer)")
        clip = training.index("clip_grad_norm_", unscale)
        step = training.index("scaler.step(optimizer)", clip)
        self.assertTrue(unscale < clip < step, "expected order: unscale_ -> clip_grad_norm_ -> scaler.step")


class PrefetchQueueTest(unittest.TestCase):
    def test_prefetch_processes_use_spawn_after_torch_initialization(self):
        sys.path.insert(0, str(ROOT / "repo/training"))
        try:
            from training import get_prefetch_context
        finally:
            sys.path.pop(0)

        self.assertEqual(get_prefetch_context().get_start_method(), "spawn")

    def test_consuming_one_prefetch_immediately_schedules_the_next(self):
        import queue

        sys.path.insert(0, str(ROOT / "repo/training"))
        try:
            from training import get_next_prefetched_pdbs, submit_prefetched_pdbs
        finally:
            sys.path.pop(0)

        class ImmediateFuture:
            def __init__(self, value):
                self.value = value

            def result(self):
                return self.value

        class ImmediateExecutor:
            def __init__(self):
                self.calls = 0

            def submit(self, fn, *args):
                self.calls += 1
                return ImmediateFuture(fn(*args))

        def fake_get_pdbs(data_loader, repeat, max_length, num_examples):
            return {
                "data_loader": data_loader,
                "repeat": repeat,
                "max_length": max_length,
                "num_examples": num_examples,
            }

        work_queue = queue.Queue(maxsize=1)
        executor = ImmediateExecutor()
        submit_prefetched_pdbs(work_queue, executor, fake_get_pdbs, "train", 10000, 5000)

        first = get_next_prefetched_pdbs(work_queue, executor, fake_get_pdbs, "train", 10000, 5000)
        self.assertEqual(first["num_examples"], 5000)
        self.assertFalse(work_queue.empty(), "prefetch queue must be refilled after consumption")

        second = get_next_prefetched_pdbs(work_queue, executor, fake_get_pdbs, "train", 10000, 5000)
        self.assertEqual(second["max_length"], 10000)
        self.assertFalse(work_queue.empty(), "prefetch_batches=1 must not deadlock on later reloads")
        self.assertEqual(executor.calls, 3)


@unittest.skipUnless(HAS_TRAINING_DEPS, "training dependencies not installed")
class StructureLoaderBatchingTest(unittest.TestCase):
    def test_sequences_larger_than_token_budget_form_singleton_batches(self):
        sys.path.insert(0, str(ROOT / "repo/training"))
        try:
            from utils import StructureLoader
        finally:
            sys.path.pop(0)

        dataset = [
            {"name": "short", "seq": "A" * 10},
            {"name": "long_a", "seq": "A" * 7000},
            {"name": "long_b", "seq": "A" * 8000},
        ]
        loader = StructureLoader(dataset, batch_size=6000)
        batches = list(loader)

        self.assertTrue(all(batches), "StructureLoader must never yield empty batches")
        names = sorted(item["name"] for batch in batches for item in batch)
        self.assertEqual(names, ["long_a", "long_b", "short"])


class DesignManifestSchemaTest(unittest.TestCase):
    """Actually exercise the ProteinMPNN->mRNABERT contract, not just its file text."""

    SCHEMA_PATH = ROOT / "design_manifest.schema.json"
    EXAMPLE_PATH = ROOT / "design_manifest.example.json"

    def _load(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_schema_and_example_are_valid_json(self):
        schema = self._load(self.SCHEMA_PATH)
        example = self._load(self.EXAMPLE_PATH)
        self.assertEqual(schema.get("$schema"), "https://json-schema.org/draft/2020-12/schema")
        self.assertIn("design_id", schema.get("required", []))
        self.assertIsInstance(example, dict)

    def test_example_satisfies_the_contract_dependency_free(self):
        # Always-on structural guard (no third-party dependency): the real rules the
        # downstream mRNABERT consumer relies on.
        schema = self._load(self.SCHEMA_PATH)
        example = self._load(self.EXAMPLE_PATH)
        for key in schema["required"]:
            self.assertIn(key, example, f"example missing required field: {key}")
        self.assertRegex(example["protein_sequence"], r"^[ACDEFGHIKLMNPQRSTVWYX]+$")
        self.assertIn("species", example["mrna_objective"])
        self.assertIn("cds_policy", example["mrna_objective"])
        self.assertIn(example["axis_method"], schema["properties"]["axis_method"]["enum"])

    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema not installed")
    def test_example_validates_against_schema_and_bad_instance_is_rejected(self):
        from jsonschema import Draft202012Validator

        schema = self._load(self.SCHEMA_PATH)
        example = self._load(self.EXAMPLE_PATH)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        self.assertEqual(list(validator.iter_errors(example)), [], "canonical example must validate")

        bad = dict(example)
        bad["protein_sequence"] = "M...invalid"   # '.' not in the AA alphabet
        del bad["mrna_objective"]                  # drop a required field
        self.assertTrue(list(validator.iter_errors(bad)), "a malformed manifest must be rejected")


if __name__ == "__main__":
    unittest.main()
