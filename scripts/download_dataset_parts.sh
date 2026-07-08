#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

URL="https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz"
DATA_ROOT="$PROTEINMPNN_DATA_ROOT"
ARCHIVE_NAME="pdb_2021aug02.tar.gz"
EXPECTED_SIZE="18037128263"
EXPECTED_SHA256="84d51d0b9224011db8deeab8b83e96f092830aaf6a1f538b1d94b0144f295714"
PART_COUNT=8
PARALLEL=4
EXTRACT=false
FORCE=false

usage() {
  cat <<'EOF'
Usage:
  scripts/download_dataset_parts.sh [options]

Downloads the upstream reference ProteinMPNN PDB training archive using HTTP range
requests, verifies the merged SHA256, and optionally extracts it.

Options:
  --data-root <dir>       Dataset root. Default: NAS MPNN datasets/proteinmpnn.
  --url <url>             Archive URL. Default: upstream pdb_2021aug02 tarball.
  --archive-name <name>   Final archive name. Default: pdb_2021aug02.tar.gz.
  --expected-size <bytes> Expected archive size. Required for range math.
  --sha256 <hex>          Expected final archive SHA256.
  --part-count <n>        Number of byte ranges. Default: 8.
  --parallel <n>          Concurrent range downloads. Default: 4.
  --extract               Extract archive under --data-root after verification.
  --force                 Redownload parts and rebuild archive.
  -h, --help              Show this help.

Environment overrides:
  DATA_ROOT, PART_COUNT, PARALLEL
EOF
}

DATA_ROOT="${DATA_ROOT:-$DATA_ROOT}"
PART_COUNT="${PART_COUNT:-$PART_COUNT}"
PARALLEL="${PARALLEL:-$PARALLEL}"

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root|--data_root) DATA_ROOT="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --archive-name|--archive_name) ARCHIVE_NAME="$2"; shift 2 ;;
    --expected-size|--expected_size) EXPECTED_SIZE="$2"; shift 2 ;;
    --sha256) EXPECTED_SHA256="$2"; shift 2 ;;
    --part-count|--part_count) PART_COUNT="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --extract) EXTRACT=true; shift ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required." >&2
  exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  echo "Error: sha256sum is required." >&2
  exit 1
fi
if ! [[ "$EXPECTED_SIZE" =~ ^[0-9]+$ ]] || [ "$EXPECTED_SIZE" -lt 1 ]; then
  echo "Error: --expected-size must be a positive integer." >&2
  exit 1
fi
if ! [[ "$PART_COUNT" =~ ^[0-9]+$ ]] || [ "$PART_COUNT" -lt 1 ]; then
  echo "Error: --part-count must be a positive integer." >&2
  exit 1
fi
if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [ "$PARALLEL" -lt 1 ]; then
  echo "Error: --parallel must be a positive integer." >&2
  exit 1
fi

ARCHIVE="$DATA_ROOT/$ARCHIVE_NAME"
PART_DIR="$DATA_ROOT/parts"
LOCK_DIR="$DATA_ROOT/.download-${ARCHIVE_NAME}.lock"
PART_SIZE=$(( (EXPECTED_SIZE + PART_COUNT - 1) / PART_COUNT ))
DIGITS=${#PART_COUNT}
if [ "$DIGITS" -lt 2 ]; then
  DIGITS=2
fi

mkdir -p "$DATA_ROOT" "$PART_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Error: another download appears to be active: $LOCK_DIR" >&2
  exit 1
fi
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

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
  if [ -n "$EXPECTED_SHA256" ] && [ "$(sha256_of "$ARCHIVE")" != "$EXPECTED_SHA256" ]; then
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

if [ "$FORCE" = true ]; then
  rm -f "$ARCHIVE" "$ARCHIVE.tmp"
  rm -f "$PART_DIR"/part_*
fi

download_part() {
  local index="$1"
  local start=$(( index * PART_SIZE ))
  local end=$(( start + PART_SIZE - 1 ))
  if [ "$start" -ge "$EXPECTED_SIZE" ]; then
    return 0
  fi
  if [ "$end" -ge "$EXPECTED_SIZE" ]; then
    end=$(( EXPECTED_SIZE - 1 ))
  fi
  local expected=$(( end - start + 1 ))
  local suffix
  printf -v suffix "%0${DIGITS}d" "$index"
  local part="$PART_DIR/part_$suffix"
  local tmp="$part.tmp.$$"
  local current
  current="$(file_size "$part")"

  if [ "$current" -eq "$expected" ]; then
    echo "part_$suffix already complete ($expected bytes)"
    return 0
  fi
  if [ "$current" -gt "$expected" ]; then
    echo "part_$suffix is too large; redownloading"
    rm -f "$part"
    current=0
  fi

  local range_start=$(( start + current ))
  local range="${range_start}-${end}"
  echo "Downloading part_$suffix range=$range current=$current expected=$expected"
  rm -f "$tmp"
  curl \
    --fail \
    --location \
    --retry 8 \
    --retry-delay 5 \
    --connect-timeout 30 \
    --range "$range" \
    --output "$tmp" \
    "$URL"

  if [ "$current" -gt 0 ]; then
    cat "$tmp" >> "$part"
    rm -f "$tmp"
  else
    mv "$tmp" "$part"
  fi

  current="$(file_size "$part")"
  if [ "$current" -ne "$expected" ]; then
    echo "Error: part_$suffix size mismatch: got $current expected $expected" >&2
    exit 1
  fi
}

echo "Downloading ProteinMPNN dataset archive in parts"
echo "url: $URL"
echo "data_root: $DATA_ROOT"
echo "archive: $ARCHIVE"
echo "expected_size: $EXPECTED_SIZE"
echo "part_count: $PART_COUNT"
echo "parallel: $PARALLEL"

pids=()
for ((i = 0; i < PART_COUNT; i++)); do
  download_part "$i" &
  pids+=("$!")
  if [ "${#pids[@]}" -ge "$PARALLEL" ]; then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Merging parts into: $ARCHIVE"
rm -f "$ARCHIVE.tmp"
: > "$ARCHIVE.tmp"
for ((i = 0; i < PART_COUNT; i++)); do
  start=$(( i * PART_SIZE ))
  if [ "$start" -ge "$EXPECTED_SIZE" ]; then
    continue
  fi
  printf -v suffix "%0${DIGITS}d" "$i"
  part="$PART_DIR/part_$suffix"
  if [ ! -f "$part" ]; then
    echo "Error: missing part: $part" >&2
    exit 1
  fi
  cat "$part" >> "$ARCHIVE.tmp"
done
mv "$ARCHIVE.tmp" "$ARCHIVE"

size="$(file_size "$ARCHIVE")"
if [ "$size" != "$EXPECTED_SIZE" ]; then
  echo "Error: merged archive size mismatch: got $size expected $EXPECTED_SIZE" >&2
  exit 1
fi

actual_sha="$(sha256_of "$ARCHIVE")"
if [ -n "$EXPECTED_SHA256" ] && [ "$actual_sha" != "$EXPECTED_SHA256" ]; then
  echo "Error: SHA256 mismatch: got $actual_sha expected $EXPECTED_SHA256" >&2
  exit 1
fi

echo "Archive verified:"
echo "  path: $ARCHIVE"
echo "  size: $size"
echo "  sha256: $actual_sha"

if [ "$EXTRACT" = true ]; then
  extract_archive
fi
