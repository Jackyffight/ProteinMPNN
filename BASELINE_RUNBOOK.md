# Baseline and Continued-Training Runbook

The primary baseline is the unchanged published ProteinMPNN checkpoint. A
from-scratch run remains useful as a reproduction/ablation, but the main 2026
route starts from the published weights after the replacement dataset passes its
semantic checks.

## 0. Environment

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN
conda create -n proteinmpnn python=3.10 -y
conda activate proteinmpnn
pip install -r ProteinMPNN/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use the CUDA PyTorch wheel that matches the training host.

## 1. Evaluate Published Weights

Evaluate the official `v_48_020.pt` checkpoint on the upstream 2021 validation
split before training:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
MAX_EXAMPLES=1000 SPLIT=valid scripts/evaluate_official_checkpoint.sh
```

The result is written to
`runs/baselines/official-v48-020-valid.json`. Use the same command and seed for
later checkpoints so NLL, perplexity, and accuracy are directly comparable.

## 2. Continued Training

The corrected 2026 v1 dataset passed its full conformance validator on
2026-07-11. Start the guarded single-A100 pilot from the official model weights:

```bash
scripts/run_2026_v1_pilot_a100.sh --dry-run
scripts/run_2026_v1_pilot_a100.sh
```

Do not use the quarantined `prototype-v0` 2026 tar shards for this run. The eventual
training mix should retain a fixed replay sample from the upstream 2021 dataset;
the exact replay ratio is selected with validation rather than baked into the
loader prematurely.

## 3. From-Scratch Reproduction

V100:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/run_baseline_from_scratch.sh --profile v100 --devices 0
```

A100:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/run_baseline_from_scratch.sh --profile a100 --devices 0
```

Dry operational pass without the long full train:

```bash
scripts/run_baseline_from_scratch.sh --profile v100 --devices 0 --no-full
```

## 4. Manual From-Scratch Steps

Fast path if the host can see an existing local copy:

```bash
scripts/stage_existing_dataset.sh
```

Download and extract upstream reference data:

```bash
scripts/download_dataset_parts.sh --extract
```

If the public IPD source is unstable, lower parallelism:

```bash
scripts/download_dataset_parts.sh --parallel 1 --extract
```

Validate dataset layout:

```bash
scripts/validate_dataset.sh
```

Smoke train (uses the sample dataset if present, else the full dataset in debug
mode — 50 examples — so this works immediately after the full download):

```bash
DEVICES=0 scripts/smoke_train.sh
```

Full-data sanity train:

```bash
DEVICES=0 scripts/full_sanity.sh
```

Benchmark host throughput:

```bash
DEVICES=0 scripts/benchmark_throughput.sh quick
scripts/print_throughput_benchmark.sh runs/benchmarks/<benchmark-dir>
```

Start full training:

```bash
DEVICES=0 RUN_NAME=proteinmpnn-baseline-v100 scripts/full_train_v100.sh
```

or:

```bash
DEVICES=0 RUN_NAME=proteinmpnn-baseline-a100 scripts/full_train_a100.sh
```

## 5. Outputs

Each run writes:

```text
runs/<run-name>/run_manifest.json
runs/<run-name>/metrics.jsonl
runs/<run-name>/eval_results.json
runs/<run-name>/log.txt
runs/<run-name>/model_weights/epoch_last.pt
runs/<run-name>/model_weights/best.pt
```

Latest metrics:

```bash
scripts/print_latest_metrics.sh runs/<run-name>
```

Resume:

```bash
scripts/resume_train.sh runs/<run-name>/model_weights/epoch_last.pt
```
