#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DESTINATION="${1:-$REPO_ROOT/repo/vanilla_model_weights/v_48_020.pt}"
OFFICIAL_COMMIT="8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
URL="${PROTEINMPNN_OFFICIAL_CHECKPOINT_URL:-https://raw.githubusercontent.com/dauparas/ProteinMPNN/$OFFICIAL_COMMIT/vanilla_model_weights/v_48_020.pt}"
EXPECTED_SIZE=6681301
EXPECTED_SHA256="c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"

for required_command in curl sha256sum stat; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "Error: $required_command is required to fetch the official checkpoint." >&2
    exit 1
  fi
done

verify_checkpoint() {
  local path="$1"
  [ -f "$path" ] || return 1
  [ "$(stat -c%s "$path")" = "$EXPECTED_SIZE" ] || return 1
  [ "$(sha256sum "$path" | awk '{print $1}')" = "$EXPECTED_SHA256" ]
}

if verify_checkpoint "$DESTINATION"; then
  echo "official_checkpoint_ok: $DESTINATION"
  exit 0
fi

mkdir -p "$(dirname "$DESTINATION")"
temporary="${DESTINATION}.download.$$"
trap 'rm -f "$temporary"' EXIT

echo "Downloading official ProteinMPNN v_48_020 checkpoint"
echo "source: $URL"
echo "destination: $DESTINATION"
curl \
  --fail \
  --location \
  --retry 5 \
  --retry-all-errors \
  --retry-delay 2 \
  --connect-timeout 30 \
  --output "$temporary" \
  "$URL"

if ! verify_checkpoint "$temporary"; then
  actual_size="$(stat -c%s "$temporary" 2>/dev/null || echo missing)"
  actual_sha="$(sha256sum "$temporary" 2>/dev/null | awk '{print $1}' || echo missing)"
  echo "Error: downloaded checkpoint failed verification." >&2
  echo "size: got=$actual_size expected=$EXPECTED_SIZE" >&2
  echo "sha256: got=$actual_sha expected=$EXPECTED_SHA256" >&2
  exit 1
fi

mv "$temporary" "$DESTINATION"
trap - EXIT
echo "official_checkpoint_ready: $DESTINATION"
