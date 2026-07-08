# ProteinMPNN Trainable Component

This repository packages a trainable ProteinMPNN component for the four-stage
protein-to-mRNA pipeline:

```text
ESMFold2 geometry -> ProteinMPNN sequence expansion -> ESMFold2 refold -> mRNABERT regulation
```

The goal is full local retraining and iteration, not remote model invocation.

## Repository Status

- Git remote: `https://github.com/Jackyffight/ProteinMPNN`
- Upstream source subset: `repo/`
- Launcher: `run_train.sh`
- Operational wrappers: `scripts/`
- mRNA bridge contract: `design_manifest.schema.json`
- Training data location: `../datasets/proteinmpnn/pdb_2021aug02`

The local `repo/` tree contains the upstream ProteinMPNN training/inference files
needed for from-scratch training. Large datasets and checkpoint weights are kept
out of git.

## Data

Full training data is already present locally:

```text
/data00/home/wangzhi.wit/models/datasets/proteinmpnn/pdb_2021aug02
```

Archive:

```text
/data00/home/wangzhi.wit/models/datasets/proteinmpnn/pdb_2021aug02.tar.gz
```

Known archive SHA256:

```text
84d51d0b9224011db8deeab8b83e96f092830aaf6a1f538b1d94b0144f295714
```

See `DATASET.md` for provenance and expected layout.

To download or rebuild the archive from HTTP range parts:

```bash
cd /data00/home/wangzhi.wit/models/ProteinMPNN
scripts/download_dataset_parts.sh --extract
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

Smoke:

```bash
cd /data00/home/wangzhi.wit/models
ProteinMPNN/scripts/smoke_train.sh
```

Full-data sanity:

```bash
cd /data00/home/wangzhi.wit/models
ProteinMPNN/scripts/full_sanity.sh
```

V100 preset:

```bash
cd /data00/home/wangzhi.wit/models
ProteinMPNN/scripts/full_train_v100.sh
```

A100 preset:

```bash
cd /data00/home/wangzhi.wit/models
ProteinMPNN/scripts/full_train_a100.sh
```

Direct launcher example:

```bash
cd /data00/home/wangzhi.wit/models
ProteinMPNN/run_train.sh v100 --devices 0 --run-name proteinmpnn-v48-noise020-v100
```

Resume:

```bash
cd /data00/home/wangzhi.wit/models/ProteinMPNN
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

## Validation

Lightweight checks that do not require torch:

```bash
cd /data00/home/wangzhi.wit/models/ProteinMPNN
python -m unittest discover -s tests -v
```

Syntax checks:

```bash
bash -n run_train.sh scripts/*.sh
python -m py_compile repo/training/training.py repo/training/utils.py repo/training/model_utils.py repo/protein_mpnn_run.py repo/protein_mpnn_utils.py
```
