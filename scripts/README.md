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

Prepare the research-independent structure throughput benchmark:

```bash
scripts/prepare_2026_structure_benchmark.sh --dry-run
scripts/prepare_2026_structure_benchmark.sh
scripts/inspect_structure_runtime.sh
scripts/setup_esmfold2_fast_runtime.sh --dry-run
scripts/setup_esmfold2_fast_runtime.sh
CUDA_VISIBLE_DEVICES=0 scripts/run_esmfold2_fast.sh smoke
CUDA_VISIBLE_DEVICES=0 scripts/run_esmfold2_fast.sh full
scripts/setup_structure_metrics_runtime.sh --dry-run
scripts/setup_structure_metrics_runtime.sh
scripts/evaluate_esmfold2_native_agreement.sh
scripts/report_esmfold2_native_agreement.sh
scripts/run_proteinmpnn_refold_pilot.sh --dry-run
scripts/run_proteinmpnn_refold_pilot.sh
scripts/report_proteinmpnn_refold_pilot.sh
```

It reads only the validated 2026 v1 metadata, selects 40 cluster-unique native
sequences from `valid`, writes a checksummed suite plus FASTA, and does not start
a GPU job. A real fusion target is a separate research input and is not required
for this engineering preflight.
`inspect_structure_runtime.sh` is read-only: it reports GPUs, relevant Python
packages, candidate executables, and cache roots without installing or running a
structure model.
The ESMFold2-Fast setup pins the Biohub source plus both Hugging Face revisions,
downloads about 24.4 GiB of checksummed weights into an isolated runtime, and is
safe to rerun after an interrupted download. The runner uses one GPU and one
record at a time; `full` is blocked until the four-length-bin smoke passes. See
`protein_mrna_pipeline/docs/ESMFOLD2_FAST_RUNBOOK.md`.
The native-agreement launcher is CPU-only and resumable. It pins a separate
Biotite metrics environment, verifies exact sequence-position correspondence,
and compares the 40 predicted PDBs with resolved C-alpha coordinates in the v1
tar shards. It leaves failed records explicit and uses no quality threshold for
model or checkpoint selection.
The paired refold pilot then compares official `v_48_020` and promoted Stage2a
on four deterministic valid backbones, four paired seeds, and a shared
ESMFold2-Fast oracle. It retains complete assembly context for ProteinMPNN but
refolds only the designed target chain. See
`protein_mrna_pipeline/docs/PROTEINMPNN_REFOLD_PILOT.md`.

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
scripts/promote_2026_v1_stage1.sh
```

Promotion requires negative valid and test NLL deltas, matching selected
checkpoint paths and SHA256 values, and the complete 426/461 record populations.
It atomically copies the checkpoint and writes a provenance manifest under
`runs/promoted/proteinmpnn-2026-v1-stage1/`.

Build the separate stage2a dataset from v1's deferred oversized manifest on the
host that stores the raw mmCIF snapshot:

```bash
scripts/build_pdb_2026_oversized_crops.sh
```

This launcher intentionally exposes no worker-count override: it keeps one file
in flight, recycles the parser every 25 files, checks available memory, records
peak RSS, and validates all generated payloads. Its default output is
`processed/proteinmpnn_tar_shards_stage2a_v1`. Do not start stage-two training
until that directory contains `validation.json` with `"status": "ok"`.

After syncing the validated artifact to the GPU workspace, use:

```bash
scripts/validate_2026_stage2a_dataset.sh
DEVICES=0 scripts/run_2026_stage2a_pilot_a100.sh --dry-run
DEVICES=0 scripts/run_2026_stage2a_pilot_a100.sh
scripts/evaluate_2026_stage2a_checkpoints.sh
```

The pilot starts from the promoted stage-1 weights with a conservative Noam
factor and never restores the stage-1 optimizer. The checkpoint gate requires
stage2a validation improvement while bounding v1 validation regression. See
`STAGE2A_RUNBOOK.md` for the formal run and one-shot test sequence.

After the formal checkpoint passes both the dual-valid and one-shot dual-test
gates, run `scripts/promote_2026_stage2a.sh`. It verifies the selected checkpoint
path and SHA256 plus the complete gate populations before writing the stable
artifact under `runs/promoted/proteinmpnn-2026-stage2a/`.

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
