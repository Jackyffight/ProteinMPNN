#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: scripts/ensure_venv_pip.sh <venv-python>" >&2
  exit 2
fi

TARGET_PYTHON="$1"
if [ ! -x "$TARGET_PYTHON" ]; then
  echo "Error: venv Python is not executable: $TARGET_PYTHON" >&2
  exit 1
fi
if "$TARGET_PYTHON" -m pip --version >/dev/null 2>&1; then
  "$TARGET_PYTHON" -m pip --version
  exit 0
fi

# Immutable PyPA get-pip artifact. This revision bundles pip 26.1.2 and supports
# Python 3.10+, including distro Python builds that omit ensurepip.
BOOTSTRAP_URL="${VENV_PIP_BOOTSTRAP_URL:-https://raw.githubusercontent.com/pypa/get-pip/3b73145063be545b649ad9ca83ea8da5fc915a4f/public/get-pip.py}"
BOOTSTRAP_SHA256="${VENV_PIP_BOOTSTRAP_SHA256:-a341e1a43e38001c551a1508a73ff23636a11970b61d901d9a1cad2a18f57055}"
PIP_VERSION="${VENV_PIP_VERSION:-26.1.2}"
BOOTSTRAP_DIR="${TMPDIR:-/tmp}/proteinmpnn-pip-bootstrap"
BOOTSTRAP_PATH="$BOOTSTRAP_DIR/get-pip-$BOOTSTRAP_SHA256.py"

command -v curl >/dev/null 2>&1 || {
  echo "Error: curl is required to bootstrap pip in the ESMFold2 venv." >&2
  exit 1
}
command -v sha256sum >/dev/null 2>&1 || {
  echo "Error: sha256sum is required to verify the pip bootstrap." >&2
  exit 1
}
mkdir -p "$BOOTSTRAP_DIR"

if [ ! -f "$BOOTSTRAP_PATH" ] || \
  [ "$(sha256sum "$BOOTSTRAP_PATH" | awk '{print $1}')" != "$BOOTSTRAP_SHA256" ]; then
  partial="$BOOTSTRAP_PATH.partial"
  rm -f "$partial"
  curl --http1.1 -fL --retry 3 --retry-delay 2 \
    "$BOOTSTRAP_URL" -o "$partial"
  observed_sha256="$(sha256sum "$partial" | awk '{print $1}')"
  if [ "$observed_sha256" != "$BOOTSTRAP_SHA256" ]; then
    rm -f "$partial"
    echo "Error: get-pip.py SHA256 mismatch." >&2
    echo "expected: $BOOTSTRAP_SHA256" >&2
    echo "observed: $observed_sha256" >&2
    exit 1
  fi
  mv "$partial" "$BOOTSTRAP_PATH"
fi

echo "Bootstrapping pip $PIP_VERSION into $TARGET_PYTHON"
"$TARGET_PYTHON" "$BOOTSTRAP_PATH" --disable-pip-version-check
"$TARGET_PYTHON" -m pip --version
