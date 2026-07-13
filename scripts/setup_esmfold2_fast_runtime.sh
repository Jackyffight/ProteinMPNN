#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

TRANSFORMERS_COMMIT="ef32577f55da19a4989cd7b22e004dc43a4998cb"
FAST_REVISION="b28d8ace5e05e61e5bec1e6820cfd3e221819d12"
ESMC_REVISION="45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
EXPECTED_WEIGHT_BYTES=26163565812
RESERVE_BYTES=$((6 * 1024 * 1024 * 1024))
RUNTIME_ROOT="${ESMFOLD2_RUNTIME_ROOT:-$MPNN_WORKSPACE/structure_runtime/esmfold2-fast}"
BASE_PYTHON="${ESMFOLD2_BASE_PYTHON:-${PROTEINMPNN_PYTHON:-python}}"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_esmfold2_fast_runtime.sh [--dry-run]

Create an isolated Python environment, install the pinned Biohub transformers
fork, download the pinned ESMFold2-Fast and ESMC-6B snapshots, and verify every
weight SHA256. Interrupted Hugging Face downloads can be resumed by rerunning.

Optional environment overrides:
  ESMFOLD2_RUNTIME_ROOT ESMFOLD2_BASE_PYTHON ALLOW_LOW_DISK=1
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|--dry_run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

echo "runtime_root: $RUNTIME_ROOT"
echo "base_python: $BASE_PYTHON"
echo "biohub_transformers_commit: $TRANSFORMERS_COMMIT"
echo "esmfold2_fast_revision: $FAST_REVISION"
echo "esmc_6b_revision: $ESMC_REVISION"
echo "weight_bytes: $EXPECTED_WEIGHT_BYTES"
echo "weight_gib: $((EXPECTED_WEIGHT_BYTES / 1024 / 1024 / 1024))"

if [ "$DRY_RUN" = true ]; then
  echo "download_started: false"
  exit 0
fi

command -v "$BASE_PYTHON" >/dev/null 2>&1 || {
  echo "Error: base Python not found: $BASE_PYTHON" >&2
  exit 1
}
"$BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10)' || {
  echo "Error: the ESMFold2-Fast runner requires Python 3.10 or newer." >&2
  exit 1
}
BASE_TORCH_VERSION="$("$BASE_PYTHON" - <<'PY'
import importlib.metadata
import torch

if not torch.cuda.is_available():
    raise SystemExit("base Python Torch cannot access CUDA")
print(importlib.metadata.version("torch"))
PY
)" || {
  echo "Error: base Python must provide CUDA-enabled Torch with package metadata." >&2
  exit 1
}
echo "base_torch_version: $BASE_TORCH_VERSION"

mkdir -p "$RUNTIME_ROOT" "$RUNTIME_ROOT/models" "$RUNTIME_ROOT/tmp" "$RUNTIME_ROOT/hf-home"
existing_bytes="$(du -sb "$RUNTIME_ROOT/models" | awk '{print $1}')"
remaining_bytes=$((EXPECTED_WEIGHT_BYTES - existing_bytes))
if [ "$remaining_bytes" -lt 0 ]; then
  remaining_bytes=0
fi
available_bytes="$(df -PB1 "$RUNTIME_ROOT" | awk 'NR == 2 {print $4}')"
required_bytes=$((remaining_bytes + RESERVE_BYTES))
echo "existing_model_bytes: $existing_bytes"
echo "available_bytes: $available_bytes"
echo "required_available_bytes: $required_bytes"
if [ "$available_bytes" -lt "$required_bytes" ] && [ "${ALLOW_LOW_DISK:-0}" != 1 ]; then
  echo "Error: insufficient free space for pinned models plus 6 GiB headroom." >&2
  echo "Set ALLOW_LOW_DISK=1 only after checking the filesystem manually." >&2
  exit 1
fi

if [ ! -x "$RUNTIME_ROOT/venv/bin/python" ]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$RUNTIME_ROOT/venv"
fi
RUNTIME_PYTHON="$RUNTIME_ROOT/venv/bin/python"
export TMPDIR="$RUNTIME_ROOT/tmp"
export HF_HOME="$RUNTIME_ROOT/hf-home"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

"$SCRIPT_DIR/ensure_venv_pip.sh" "$RUNTIME_PYTHON"
VENV_SITE_PACKAGES="$("$RUNTIME_PYTHON" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
"$BASE_PYTHON" "$SCRIPT_DIR/link_base_torch_site.py" \
  --target-site-packages "$VENV_SITE_PACKAGES"
RUNTIME_TORCH_VERSION="$("$RUNTIME_PYTHON" - <<'PY'
import importlib.metadata
import torch

if not torch.cuda.is_available():
    raise SystemExit("runtime venv Torch cannot access CUDA")
print(importlib.metadata.version("torch"))
PY
)" || {
  echo "Error: runtime venv cannot reuse the base CUDA Torch installation." >&2
  exit 1
}
if [ "$RUNTIME_TORCH_VERSION" != "$BASE_TORCH_VERSION" ]; then
  echo "Error: runtime/base Torch versions differ." >&2
  echo "base: $BASE_TORCH_VERSION" >&2
  echo "runtime: $RUNTIME_TORCH_VERSION" >&2
  exit 1
fi
echo "runtime_torch_version: $RUNTIME_TORCH_VERSION"
"$RUNTIME_PYTHON" -m pip install --no-cache-dir --upgrade \
  "jsonschema==4.25.1" \
  "$REPO_ROOT/protein_mrna_pipeline"
"$RUNTIME_PYTHON" -m pip install --no-cache-dir --upgrade \
  "transformers @ https://github.com/Biohub/transformers/archive/${TRANSFORMERS_COMMIT}.zip"

FAST_REVISION="$FAST_REVISION" ESMC_REVISION="$ESMC_REVISION" \
RUNTIME_ROOT="$RUNTIME_ROOT" "$RUNTIME_PYTHON" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(os.environ["RUNTIME_ROOT"])
jobs = (
    (
        "biohub/ESMFold2-Fast",
        os.environ["FAST_REVISION"],
        root / "models/ESMFold2-Fast",
        ["config.json", "model.safetensors"],
    ),
    (
        "biohub/ESMC-6B",
        os.environ["ESMC_REVISION"],
        root / "models/ESMC-6B",
        ["config.json", "model.safetensors.index.json", "model-*.safetensors"],
    ),
)
for repository, revision, destination, patterns in jobs:
    print(f"Downloading {repository}@{revision} -> {destination}", flush=True)
    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=destination,
        allow_patterns=patterns,
        max_workers=4,
    )
PY

export PYTHONPATH="$REPO_ROOT/protein_mrna_pipeline/src${PYTHONPATH:+:$PYTHONPATH}"
"$RUNTIME_PYTHON" -m protein_mrna_pipeline verify-esmfold2-runtime \
  --runtime-root "$RUNTIME_ROOT"

"$RUNTIME_PYTHON" - <<'PY'
import json
import torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

print(json.dumps({
    "cuda_available": torch.cuda.is_available(),
    "cuda_devices": [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ],
    "esmfold2_import": ESMFold2Model.__name__,
}, indent=2, sort_keys=True))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the pinned ESMFold2 runtime")
PY

echo "runtime_ready: $RUNTIME_ROOT"
echo "next: scripts/run_esmfold2_fast.sh smoke"
