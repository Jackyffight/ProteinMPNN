# Baseline From-Scratch Runbook

This runbook trains the upstream-reference ProteinMPNN baseline from zero.

## 0. Environment

```bash
cd /data00/home/wangzhi.wit/models
conda create -n proteinmpnn python=3.10 -y
conda activate proteinmpnn
pip install -r ProteinMPNN/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use the CUDA PyTorch wheel that matches the training host.

## 1. End-to-End Script

V100:

```bash
cd /data00/home/wangzhi.wit/models/ProteinMPNN
scripts/run_baseline_from_scratch.sh --profile v100 --devices 0
```

A100:

```bash
cd /data00/home/wangzhi.wit/models/ProteinMPNN
scripts/run_baseline_from_scratch.sh --profile a100 --devices 0
```

Dry operational pass without the long full train:

```bash
scripts/run_baseline_from_scratch.sh --profile v100 --devices 0 --no-full
```

## 2. Manual Step-by-Step

Download and extract upstream reference data:

```bash
scripts/download_dataset_parts.sh --extract
```

Validate dataset layout:

```bash
scripts/validate_dataset.sh
```

Smoke train:

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

## 3. Outputs

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
