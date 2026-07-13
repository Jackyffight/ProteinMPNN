#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

RUNTIME_ROOT="${STRUCTURE_METRICS_RUNTIME_ROOT:-$MPNN_WORKSPACE/structure_runtime/native-structure-metrics}"
BASE_PYTHON="${STRUCTURE_METRICS_BASE_PYTHON:-${PROTEINMPNN_PYTHON:-python}}"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_structure_metrics_runtime.sh [--dry-run]

Create a CPU-only metrics environment with pinned Biotite, NumPy, SciPy, and
the base interpreter's Torch installation. This does not download model weights
or use a GPU.

Optional environment overrides:
  STRUCTURE_METRICS_RUNTIME_ROOT STRUCTURE_METRICS_BASE_PYTHON
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
echo "biotite: 1.6.0"
echo "biotraj: 1.2.2"
echo "jsonschema: 4.25.1"
echo "numpy: 2.4.6"
echo "scipy: 1.17.1"
echo "gpu_required: false"

if [ "$DRY_RUN" = true ]; then
  echo "installation_started: false"
  exit 0
fi

command -v "$BASE_PYTHON" >/dev/null 2>&1 || {
  echo "Error: base Python not found: $BASE_PYTHON" >&2
  exit 1
}
"$BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3, 11)' || {
  echo "Error: the pinned Biotite runtime requires Python 3.11 or newer." >&2
  exit 1
}
BASE_TORCH_VERSION="$("$BASE_PYTHON" - <<'PY'
import importlib.metadata
import torch

print(importlib.metadata.version("torch"))
PY
)" || {
  echo "Error: base Python must provide Torch with package metadata." >&2
  exit 1
}
echo "base_torch_version: $BASE_TORCH_VERSION"

mkdir -p "$RUNTIME_ROOT" "$RUNTIME_ROOT/tmp"
if [ ! -x "$RUNTIME_ROOT/venv/bin/python" ]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$RUNTIME_ROOT/venv"
fi
RUNTIME_PYTHON="$RUNTIME_ROOT/venv/bin/python"
"$RUNTIME_PYTHON" -c 'import sys; assert sys.version_info >= (3, 11)' || {
  echo "Error: existing metrics venv uses Python older than 3.11." >&2
  exit 1
}
export TMPDIR="$RUNTIME_ROOT/tmp"
export PIP_DISABLE_PIP_VERSION_CHECK=1

"$SCRIPT_DIR/ensure_venv_pip.sh" "$RUNTIME_PYTHON"
VENV_SITE_PACKAGES="$("$RUNTIME_PYTHON" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
"$BASE_PYTHON" "$SCRIPT_DIR/link_base_torch_site.py" \
  --target-site-packages "$VENV_SITE_PACKAGES"

RUNTIME_TORCH_VERSION="$("$RUNTIME_PYTHON" - <<'PY'
import importlib.metadata
import torch

print(importlib.metadata.version("torch"))
PY
)" || {
  echo "Error: metrics venv cannot reuse the base Torch installation." >&2
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
  "numpy==2.4.6" \
  "scipy==1.17.1" \
  "biotraj==1.2.2" \
  "biotite==1.6.0" \
  "jsonschema==4.25.1" \
  "$REPO_ROOT/protein_mrna_pipeline"

export PYTHONPATH="$REPO_ROOT/protein_mrna_pipeline/src${PYTHONPATH:+:$PYTHONPATH}"
"$RUNTIME_PYTHON" -m protein_mrna_pipeline verify-structure-metrics-runtime \
  --runtime-root "$RUNTIME_ROOT"

echo "runtime_ready: $RUNTIME_ROOT"
echo "next: scripts/evaluate_esmfold2_native_agreement.sh"
