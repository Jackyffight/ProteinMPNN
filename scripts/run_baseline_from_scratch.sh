#!/usr/bin/env bash
# End-to-end upstream-reference ProteinMPNN baseline run.
#
# This starts from data download and ends with a full from-scratch training run.
# Use --no-full to stop after download, validation, smoke, sanity, and benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE="v100"
RUN_FULL=true
RUN_BENCHMARK=true
DOWNLOAD=true
EXTRACT=true
BENCHMARK_MODE="quick"
DEVICES="${DEVICES:-0}"
RUN_NAME=""

usage() {
  cat <<'EOF'
Usage:
  scripts/run_baseline_from_scratch.sh [options]

Options:
  --profile <v100|a100>   Full-train preset. Default: v100.
  --devices <list>        CUDA_VISIBLE_DEVICES for training scripts. Default: 0.
  --run-name <name>       Full-train run name. Default: proteinmpnn-baseline-<profile>-<timestamp>.
  --no-full               Stop before the long full training run.
  --no-benchmark          Skip throughput benchmark.
  --skip-download         Skip range download/extract step.
  --no-extract            Download and verify archive but do not extract.
  --benchmark-mode <mode> smoke, quick, or full. Default: quick.
  -h, --help              Show this help.

Typical:
  scripts/run_baseline_from_scratch.sh --profile v100 --devices 0
  scripts/run_baseline_from_scratch.sh --profile a100 --devices 0 --no-full
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --devices|--cuda-visible-devices|--cuda_visible_devices) DEVICES="$2"; shift 2 ;;
    --run-name|--run_name) RUN_NAME="$2"; shift 2 ;;
    --no-full|--no_full) RUN_FULL=false; shift ;;
    --no-benchmark|--no_benchmark) RUN_BENCHMARK=false; shift ;;
    --skip-download|--skip_download) DOWNLOAD=false; shift ;;
    --no-extract|--no_extract) EXTRACT=false; shift ;;
    --benchmark-mode|--benchmark_mode) BENCHMARK_MODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$PROFILE" != "v100" ] && [ "$PROFILE" != "a100" ]; then
  echo "Error: --profile must be v100 or a100." >&2
  exit 1
fi
if [ "$BENCHMARK_MODE" != "smoke" ] && [ "$BENCHMARK_MODE" != "quick" ] && [ "$BENCHMARK_MODE" != "full" ]; then
  echo "Error: --benchmark-mode must be smoke, quick, or full." >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d%H%M%S)"
if [ -z "$RUN_NAME" ]; then
  RUN_NAME="proteinmpnn-baseline-${PROFILE}-${TIMESTAMP}"
fi

cd "$REPO_ROOT"

echo "=== ProteinMPNN baseline from scratch ==="
echo "repo_root: $REPO_ROOT"
echo "profile: $PROFILE"
echo "devices: $DEVICES"
echo "run_name: $RUN_NAME"
echo "download: $DOWNLOAD"
echo "extract: $EXTRACT"
echo "benchmark: $RUN_BENCHMARK"
echo "run_full: $RUN_FULL"

if [ "$DOWNLOAD" = true ]; then
  if [ "$EXTRACT" = true ]; then
    scripts/download_dataset_parts.sh --extract
  else
    scripts/download_dataset_parts.sh
  fi
fi

scripts/validate_dataset.sh

echo "=== smoke train ==="
DEVICES="$DEVICES" RUN_NAME="proteinmpnn-smoke-${TIMESTAMP}" scripts/smoke_train.sh

echo "=== full-data sanity train ==="
DEVICES="$DEVICES" RUN_NAME="proteinmpnn-full-sanity-${TIMESTAMP}" scripts/full_sanity.sh

if [ "$RUN_BENCHMARK" = true ]; then
  echo "=== throughput benchmark: $BENCHMARK_MODE ==="
  DEVICES="$DEVICES" scripts/benchmark_throughput.sh "$BENCHMARK_MODE"
  LATEST_BENCHMARK="$(find runs/benchmarks -maxdepth 1 -type d -name "throughput-${BENCHMARK_MODE}-*" | sort | tail -n 1)"
  if [ -n "$LATEST_BENCHMARK" ]; then
    scripts/print_throughput_benchmark.sh "$LATEST_BENCHMARK"
  fi
fi

if [ "$RUN_FULL" = false ]; then
  echo "Stopped before full training because --no-full was set."
  echo "Run full training later with:"
  echo "  DEVICES=$DEVICES RUN_NAME=$RUN_NAME scripts/full_train_${PROFILE}.sh"
  exit 0
fi

echo "=== full baseline training: $PROFILE ==="
DEVICES="$DEVICES" RUN_NAME="$RUN_NAME" "scripts/full_train_${PROFILE}.sh"
