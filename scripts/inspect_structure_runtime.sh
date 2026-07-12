#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

PYTHON_BIN="${PROTEINMPNN_PYTHON:-${PYTHON_BIN:-python}}"

echo "=== structure runtime inventory ==="
echo "python: $PYTHON_BIN"
echo "python_path: $(command -v "$PYTHON_BIN" 2>/dev/null || echo not-found)"
echo "workspace: $MPNN_WORKSPACE"
echo "hf_home: ${HF_HOME:-$HOME/.cache/huggingface}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPUs ==="
  if ! nvidia-smi \
    --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader; then
    echo "nvidia_smi: query-failed"
  fi
else
  echo "nvidia_smi: not-found"
fi

echo "=== Python packages ==="
"$PYTHON_BIN" - <<'PY'
import importlib.metadata
import importlib.util
import json
import platform
import sys

distributions = [
    "torch",
    "transformers",
    "jsonschema",
    "fair-esm",
    "openfold",
    "colabfold",
    "accelerate",
    "biotite",
]
modules = ["torch", "transformers", "jsonschema", "esm", "openfold", "colabfold"]

versions = {}
for name in distributions:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None

available_modules = {
    name: importlib.util.find_spec(name) is not None
    for name in modules
}
report = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "distributions": versions,
    "modules": available_modules,
}

if available_modules["torch"]:
    try:
        import torch

        report["torch_cuda"] = {
            "available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as error:
        report["torch_cuda_error"] = f"{type(error).__name__}: {error}"

print(json.dumps(report, indent=2, sort_keys=True))
PY

echo "=== Candidate executables ==="
for executable in \
  esm-fold \
  esmfold \
  colabfold_batch \
  foldseek \
  python; do
  if command -v "$executable" >/dev/null 2>&1; then
    printf '%s: %s\n' "$executable" "$(command -v "$executable")"
  else
    printf '%s: not-found\n' "$executable"
  fi
done

echo "=== Cache roots ==="
for cache_root in \
  "${HF_HOME:-$HOME/.cache/huggingface}" \
  "$HOME/.cache/torch" \
  "$HOME/.cache/colabfold"; do
  if [ -d "$cache_root" ]; then
    printf '%s: present\n' "$cache_root"
  else
    printf '%s: absent\n' "$cache_root"
  fi
done

echo "inventory_only: true"
echo "inference_started: false"
