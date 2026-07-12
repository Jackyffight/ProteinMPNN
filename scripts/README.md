# Operational Scripts

These scripts wrap `ProteinMPNN/run_train.sh` with stable local paths and presets.
They are intentionally thin; the main launcher owns validation, manifests, and
runtime checks.

Pinned NAS workspace:

```text
/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN
```

Shared path variables live in `scripts/env_nas.sh`.

Recommended order:

```bash
scripts/run_baseline_from_scratch.sh --profile v100 --devices 0 --no-full
scripts/stage_existing_dataset.sh
scripts/download_dataset_parts.sh --extract
scripts/validate_dataset.sh
scripts/smoke_train.sh
scripts/full_sanity.sh
scripts/full_train_v100.sh
scripts/benchmark_throughput.sh smoke
scripts/benchmark_throughput.sh quick
scripts/print_throughput_benchmark.sh runs/benchmarks/<benchmark-dir>
scripts/print_latest_metrics.sh runs/<run-name>
```

Prefer `stage_existing_dataset.sh` when an existing `/data00` copy is visible on
the host. The public IPD archive is 17GB and can be slow or return transient
504 errors; `download_dataset_parts.sh` is resumable, but staging a local copy is
usually much faster.

Latest PDB owned dataset track:

```bash
scripts/init_dataset_version.sh
scripts/sync_latest_pdb_assemblies.sh --dry-run
scripts/inspect_dataset_version.sh /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn_custom/proteinmpnn_pdb_latest_YYYYMMDD
```

For A100:

```bash
scripts/full_train_a100.sh
```

Validated 2026 v1 continuation pilot on one A100:

```bash
scripts/run_2026_v1_pilot_a100.sh --dry-run
scripts/run_2026_v1_pilot_a100.sh
scripts/run_2026_v1_stage1_a100.sh --dry-run
scripts/run_2026_v1_stage1_a100.sh
```

The pilot script uses the pinned GPU-server paths from `env_nas.sh`, validates
the dataset manifest and validation result, and refuses multi-GPU device lists or
an existing output directory. It verifies the ignored official `v_48_020.pt`
file by size and SHA256 before training, downloading a clean copy from the
upstream ProteinMPNN repository when the file is absent or invalid.

The stage-1 script reuses the same guards and runs 20 continuation epochs over
all available v1 training clusters, saving periodic checkpoints every 5 epochs.
It is intentionally single-GPU because this training loop does not yet implement
DDP or distributed metric reduction.

After stage 1 finishes, rank all retained checkpoints and the official weights on
the complete fixed 2026 v1 validation population:

```bash
scripts/evaluate_2026_v1_stage1_checkpoints.sh
```

After the fixed-valid summary selects one checkpoint, run the test comparison once:

```bash
scripts/evaluate_2026_v1_selected_test.sh
```

The paired and multi-seed stage-1 scripts now use complete validation records only.
The multi-seed script measures evaluation sensitivity; promotion across training
seeds still requires independently trained runs.

Current unmeasured presets mirror the mRNABERT launcher style: use `v100`
for conservative token budgets and `a100` for larger token budgets. Run the
throughput benchmark before long training on a new host, then keep the measured
winner in the run name and manifest.

Resume:

```bash
scripts/resume_train.sh runs/<run-name>/model_weights/epoch_last.pt
```

Expected training outputs:

```text
<run>/run_manifest.json
<run>/metrics.jsonl
<run>/eval_results.json
<run>/log.txt
<run>/model_weights/epoch_last.pt
<run>/model_weights/best.pt
```
