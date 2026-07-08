#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

URL="https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz"
DATA_ROOT="${DATA_ROOT:-$PROTEINMPNN_DATA_ROOT}"
ARCHIVE_NAME="pdb_2021aug02.tar.gz"
EXPECTED_SIZE="18037128263"
EXPECTED_SHA256="84d51d0b9224011db8deeab8b83e96f092830aaf6a1f538b1d94b0144f295714"
METHOD="auto"
EXTRACT=false
FORCE=false

ARIA2_CONNECTIONS="${ARIA2_CONNECTIONS:-8}"
ARIA2_SPLITS="${ARIA2_SPLITS:-50}"
PART_COUNT="${PART_COUNT:-50}"
PARALLEL="${PARALLEL:-6}"
CURL_RETRIES="${CURL_RETRIES:-50}"
CURL_RETRY_DELAY="${CURL_RETRY_DELAY:-5}"

usage() {
  cat <<'EOF'
Usage:
  scripts/download_dataset_fast.sh [options]

Fast downloader for the upstream ProteinMPNN PDB training archive. It uses
aria2c multi-connection download when available and otherwise falls back to the
repo's curl range-part downloader. Safe to rerun after interruption.

Options:
  --data-root <dir>            Dataset root. Default: NAS MPNN datasets/proteinmpnn.
  --method <auto|aria2|curl>   Download backend. Default: auto.
  --aria2-connections <n>      aria2 max connections per server. Default: 8.
  --aria2-splits <n>           aria2 split count. Default: 50.
  --part-count <n>             curl fallback byte ranges. Default: 50.
  --parallel <n>               curl fallback concurrent ranges. Default: 6.
  --extract                    Extract archive after checksum verification.
  --force                      Redownload from scratch.
  -h, --help                   Show this help.

Environment overrides:
  DATA_ROOT, ARIA2_CONNECTIONS, ARIA2_SPLITS, PART_COUNT, PARALLEL,
  CURL_RETRIES, CURL_RETRY_DELAY
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root|--data_root) DATA_ROOT="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --aria2-connections|--aria2_connections) ARIA2_CONNECTIONS="$2"; shift 2 ;;
    --aria2-splits|--aria2_splits) ARIA2_SPLITS="$2"; shift 2 ;;
    --part-count|--part_count) PART_COUNT="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --extract) EXTRACT=true; shift ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$METHOD" in
  auto|aria2|curl) ;;
  *) echo "Error: --method must be one of auto, aria2, curl." >&2; exit 2 ;;
esac

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
    echo "Error: $name must be a positive integer." >&2
    exit 1
  fi
}

require_positive_int "--aria2-connections" "$ARIA2_CONNECTIONS"
require_positive_int "--aria2-splits" "$ARIA2_SPLITS"
require_positive_int "--part-count" "$PART_COUNT"
require_positive_int "--parallel" "$PARALLEL"

ARCHIVE="$DATA_ROOT/$ARCHIVE_NAME"
LOG_DIR="$DATA_ROOT/download_logs"
mkdir -p "$DATA_ROOT" "$LOG_DIR"

file_size() {
  if [ -f "$1" ]; then
    stat -c%s "$1"
  else
    echo 0
  fi
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

verify_archive() {
  local size
  size="$(file_size "$ARCHIVE")"
  if [ "$size" != "$EXPECTED_SIZE" ]; then
    return 1
  fi
  if [ "$(sha256_of "$ARCHIVE")" != "$EXPECTED_SHA256" ]; then
    return 1
  fi
  return 0
}

extract_archive() {
  echo "Extracting archive into: $DATA_ROOT"
  tar -xzf "$ARCHIVE" -C "$DATA_ROOT"
}

if [ "$FORCE" = false ] && [ -f "$ARCHIVE" ] && verify_archive; then
  echo "Archive already verified: $ARCHIVE"
  if [ "$EXTRACT" = true ]; then
    extract_archive
  fi
  exit 0
fi

run_aria2() {
  if ! command -v aria2c >/dev/null 2>&1; then
    return 127
  fi
  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "Error: sha256sum is required." >&2
    exit 1
  fi

  if [ "$FORCE" = true ]; then
    rm -f "$ARCHIVE" "$ARCHIVE.aria2"
  fi

  echo "Downloading ProteinMPNN dataset with aria2c"
  echo "url: $URL"
  echo "archive: $ARCHIVE"
  echo "connections: $ARIA2_CONNECTIONS"
  echo "splits: $ARIA2_SPLITS"
  echo "log: $LOG_DIR/aria2.log"

  aria2c \
    --continue=true \
    --max-connection-per-server="$ARIA2_CONNECTIONS" \
    --split="$ARIA2_SPLITS" \
    --min-split-size=64M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --max-tries=0 \
    --retry-wait=10 \
    --timeout=60 \
    --connect-timeout=30 \
    --summary-interval=30 \
    --log="$LOG_DIR/aria2.log" \
    --dir="$DATA_ROOT" \
    --out="$ARCHIVE_NAME" \
    "$URL"

  if ! verify_archive; then
    echo "Error: aria2 download failed checksum/size verification." >&2
    echo "Actual size: $(file_size "$ARCHIVE")" >&2
    if [ -f "$ARCHIVE" ]; then
      echo "Actual sha256: $(sha256_of "$ARCHIVE")" >&2
    fi
    exit 1
  fi
}

run_curl_parts() {
  echo "Downloading ProteinMPNN dataset with curl range parts"
  local args=(
    --data-root "$DATA_ROOT" \
    --url "$URL" \
    --archive-name "$ARCHIVE_NAME" \
    --expected-size "$EXPECTED_SIZE" \
    --sha256 "$EXPECTED_SHA256" \
    --part-count "$PART_COUNT" \
    --parallel "$PARALLEL" \
    --curl-retries "$CURL_RETRIES" \
    --retry-delay "$CURL_RETRY_DELAY"
  )
  if [ "$FORCE" = true ]; then
    args+=(--force)
  fi
  "$SCRIPT_DIR/download_dataset_parts.sh" "${args[@]}"
}

if [ "$METHOD" = "aria2" ]; then
  run_aria2
elif [ "$METHOD" = "curl" ]; then
  run_curl_parts
elif command -v aria2c >/dev/null 2>&1; then
  run_aria2
else
  echo "aria2c not found; falling back to curl range parts."
  run_curl_parts
fi

echo "Archive verified:"
echo "  path: $ARCHIVE"
echo "  size: $(file_size "$ARCHIVE")"
echo "  sha256: $(sha256_of "$ARCHIVE")"

if [ "$EXTRACT" = true ]; then
  extract_archive
fi
