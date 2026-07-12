# ProteinMPNN Trainable Component

This repository packages a trainable ProteinMPNN component for the four-stage
protein-to-mRNA pipeline:

```text
ESMFold2 geometry -> ProteinMPNN sequence expansion -> ESMFold2 refold -> mRNABERT regulation
```

The goal is full local retraining and iteration, not remote model invocation.

## Attribution

The model architecture and training code under `repo/` are **ProteinMPNN** by
Dauparas et al., *Science* 2022 (github.com/dauparas/ProteinMPNN,
DOI 10.1126/science.add2187). `repo/` is a **subset** of that repository — the
training code plus the inference entry point `protein_mpnn_run.py`; the examples
and colab directories are not vendored. This
project adds the local retraining launcher, operational scripts, dataset
provenance, and the mRNABERT design-manifest bridge; it does not modify the
ProteinMPNN model. Training data is the IPD `pdb_2021aug02` reference set.

## Repository Status

- Git remote: `https://github.com/Jackyffight/ProteinMPNN`
- Upstream source subset: `repo/`
- Launcher: `run_train.sh`
- Operational wrappers: `scripts/`
- mRNA bridge contract: `design_manifest.schema.json`
- Training data location: `/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn/pdb_2021aug02`

The local `repo/` tree contains the upstream ProteinMPNN training/inference files
needed for from-scratch training. Large datasets and checkpoint weights are kept
out of git.

The upstream `v_48_020.pt` checkpoint is used in two distinct ways: unchanged as
the published-model baseline, and as model-weight initialization for a new
continued-training run. It is not a resumable training checkpoint because it
does not contain optimizer, step, or epoch state.

## Data

Full training data is already present locally:

```text
/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn/pdb_2021aug02
```

Archive:

```text
/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn/pdb_2021aug02.tar.gz
```

Known archive SHA256:

```text
84d51d0b9224011db8deeab8b83e96f092830aaf6a1f538b1d94b0144f295714
```

See `DATASET.md` for provenance and expected layout.

The launcher accepts both the upstream small-file `.pt` layout and the new
tar-shard layout used for the 2026 dataset track. It auto-detects tar shards when
`manifest.json`, `index.jsonl`, and `shards/` are present:

```bash
ProteinMPNN/run_train.sh smoke \
  --data-dir /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn/proteinmpnn_tar_shards \
  --dataset-format tar
```

When `scripts/env_nas.sh` is sourced, the same path is available as
`$PROTEINMPNN_TAR_SHARD_DATA_DIR`.

The existing 2026 tar-shard artifact is `prototype-v0` and is not training-ready.
See `DATASET_VERSIONS.md` for the conformance failures that must be fixed before
it is used for continued training.

The replacement v1 continuation dataset is written under the owned snapshot at
`processed/proteinmpnn_tar_shards_v1`. It covers post-2021 entries, uses one
canonical assembly and target per PDB, preserves missing-residue masks, and
defers contexts above 2,000 residues. `scripts/build_pdb_2026_tar_shards.sh`
runs its full semantic validator automatically. The 2026-07-11 production build
passed validation with 46,619 target records, zero parser failures, and zero
exact-sequence or PDB split leaks.

To download or rebuild the upstream reference archive from HTTP range parts:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/stage_existing_dataset.sh
scripts/download_dataset_parts.sh --extract
```

Prefer `stage_existing_dataset.sh` if this host can see an existing local copy.
The public IPD source can be slow and occasionally returns transient 504 errors.

For our own latest-PDB dataset track:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/init_dataset_version.sh
scripts/sync_latest_pdb_assemblies.sh --dry-run
```

## Environment

Use a separate environment from mRNABERT:

```bash
conda create -n proteinmpnn python=3.10 -y
conda activate proteinmpnn
pip install -r ProteinMPNN/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use the CUDA PyTorch build appropriate for the target machine.

## Run

Evaluate the unchanged official checkpoint first. This uses zero coordinate
noise and zero dropout while retaining the checkpoint's 48-neighbor model:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
MAX_EXAMPLES=1000 SPLIT=valid scripts/evaluate_official_checkpoint.sh
```

Start the guarded single-A100 pilot from the official model weights:

```bash
scripts/run_2026_v1_pilot_a100.sh --dry-run
scripts/run_2026_v1_pilot_a100.sh
```

The script checks the pinned v1 dataset, validation result, checkpoint, selected
GPU, and output path before delegating to `run_train.sh`. The official checkpoint
is checksum-verified and downloaded from the upstream ProteinMPNN repository when
it is absent or invalid. `--init-checkpoint` starts step and epoch at zero with a
new optimizer.
`--resume` is reserved for `epoch_last.pt` or another checkpoint written by this
training loop.

After the pilot passes, start the conservative formal continuation stage:

```bash
scripts/run_2026_v1_stage1_a100.sh --dry-run
scripts/run_2026_v1_stage1_a100.sh
```

This stage uses one A100, all available v1 training clusters, 20 epochs, a
10,000-token batch budget, bounded single-process prefetch, and checkpoints every
5 epochs. The current training loop does not implement DDP; a comma-separated
GPU list is rejected instead of silently using only one of the requested GPUs.

Compare the finished run's `best.pt` with the official checkpoint on identical
held-out v1 test structures before selecting it for a later stage:

```bash
scripts/evaluate_2026_v1_stage1.sh
scripts/evaluate_2026_v1_stage1_multiseed.sh
```

The multi-seed comparison is the promotion gate. It repeats the paired held-out
test evaluation with seeds `11 23 42 67 101` and reports the mean delta plus the
number of seeds on which the stage-1 checkpoint wins.

Full baseline from data download through training:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/run_baseline_from_scratch.sh --profile v100 --devices 0
```

See `BASELINE_RUNBOOK.md` for the manual sequence.

Smoke (uses `pdb_2021aug02_sample` if provisioned, else falls back to the full
dataset in debug mode — 50 examples — so it works right after the full download):

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN
ProteinMPNN/scripts/smoke_train.sh
```

Full-data sanity:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN
ProteinMPNN/scripts/full_sanity.sh
```

V100 preset:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN
ProteinMPNN/scripts/full_train_v100.sh
```

A100 preset:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN
ProteinMPNN/scripts/full_train_a100.sh
```

Throughput benchmark before a long run:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/benchmark_throughput.sh quick
scripts/print_throughput_benchmark.sh runs/benchmarks/<benchmark-dir>
```

Direct launcher example:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN
ProteinMPNN/run_train.sh v100 --devices 0 --run-name proteinmpnn-v48-noise020-v100
```

Resume:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/resume_train.sh runs/proteinmpnn-v48-noise020-v100/model_weights/epoch_last.pt
```

## Outputs

Each run writes:

```text
runs/<run-name>/run_manifest.json
runs/<run-name>/metrics.jsonl
runs/<run-name>/eval_results.json
runs/<run-name>/log.txt
runs/<run-name>/model_weights/epoch_last.pt
runs/<run-name>/model_weights/best.pt
```

`run_manifest.json` captures args, data paths, cluster counts, model config, and
runtime versions. `metrics.jsonl` is one JSON row per epoch.

## V100 Notes

ProteinMPNN is small enough for V100. The `v100` preset uses a lower token budget
than A100:

```text
V100: batch_tokens=6000
A100/full: batch_tokens=10000
```

If V100 memory is tight, lower `--batch-tokens` to `4000`.

Structure prefetch uses the multiprocessing `spawn` context because forking after
PyTorch model initialization can deadlock native thread pools. The launcher keeps
nested DataLoader workers at zero and starts one prefetch process by default;
increase these only after measuring host memory and throughput.

Use `scripts/benchmark_throughput.sh quick` on each new host before committing to
a long run. The benchmark sweeps token budget plus loader/prefetch workers and
writes a sortable `summary.tsv`.

## Validation

Lightweight checks that do not require torch:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
python -m unittest discover -s tests -v
```

Syntax checks:

```bash
bash -n run_train.sh scripts/*.sh
python -m py_compile repo/training/training.py repo/training/utils.py repo/training/model_utils.py repo/protein_mpnn_run.py repo/protein_mpnn_utils.py
```
