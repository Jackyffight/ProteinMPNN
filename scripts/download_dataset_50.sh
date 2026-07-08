#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

URL="https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz"
DATA_ROOT="${DATA_ROOT:-$PROTEINMPNN_DATA_ROOT}"
ARCHIVE="$DATA_ROOT/pdb_2021aug02.tar.gz"
EXPECTED_SIZE=18037128263
EXPECTED_SHA256="84d51d0b9224011db8deeab8b83e96f092830aaf6a1f538b1d94b0144f295714"
PART_COUNT=50
PARALLEL="${PARALLEL:-10}"
EXTRACT=false

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root|--data_root) DATA_ROOT="$2"; ARCHIVE="$DATA_ROOT/pdb_2021aug02.tar.gz"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --extract) EXTRACT=true; shift ;;
    -h|--help)
      echo "Usage: scripts/download_dataset_50.sh [--data-root DIR] [--parallel N] [--extract]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [ "$PARALLEL" -lt 1 ]; then
  echo "Error: --parallel must be a positive integer." >&2
  exit 1
fi

PART_DIR="$DATA_ROOT/parts_50"
LOG_DIR="$DATA_ROOT/download_logs"
PART_SIZE=$(( (EXPECTED_SIZE + PART_COUNT - 1) / PART_COUNT ))

mkdir -p "$DATA_ROOT" "$PART_DIR" "$LOG_DIR"

file_size() {
  if [ -f "$1" ]; then
    stat -c%s "$1"
  else
    echo 0
  fi
}

download_part() {
  local i="$1"
  local start=$(( i * PART_SIZE ))
  local end=$(( start + PART_SIZE - 1 ))
  if [ "$end" -ge "$EXPECTED_SIZE" ]; then
    end=$(( EXPECTED_SIZE - 1 ))
  fi

  local suffix
  printf -v suffix "%02d" "$i"
  local part="$PART_DIR/part_$suffix"
  local log="$LOG_DIR/part_$suffix.log"

  echo "part_$suffix $start-$end"
  curl \
    --fail \
    --location \
    --http1.1 \
    --retry 30 \
    --retry-delay 5 \
    --retry-connrefused \
    --connect-timeout 30 \
    --speed-limit 1024 \
    --speed-time 300 \
    --range "$start-$end" \
    --output "$part" \
    "$URL" > "$log" 2>&1
}

echo "Downloading ProteinMPNN dataset in 50 parts"
echo "url: $URL"
echo "data_root: $DATA_ROOT"
echo "archive: $ARCHIVE"
echo "parallel: $PARALLEL"
echo "parts: $PART_DIR"

pids=()
failures=0
for ((i = 0; i < PART_COUNT; i++)); do
  download_part "$i" &
  pids+=("$!")
  if [ "${#pids[@]}" -ge "$PARALLEL" ]; then
    if ! wait "${pids[0]}"; then
      failures=$(( failures + 1 ))
    fi
    pids=("${pids[@]:1}")
  fi
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$(( failures + 1 ))
  fi
done

if [ "$failures" -gt 0 ]; then
  echo "Error: $failures part download(s) failed. Check $LOG_DIR/part_*.log" >&2
  exit 1
fi

echo "Merging parts"
rm -f "$ARCHIVE" "$ARCHIVE.tmp"
: > "$ARCHIVE.tmp"
for ((i = 0; i < PART_COUNT; i++)); do
  printf -v suffix "%02d" "$i"
  cat "$PART_DIR/part_$suffix" >> "$ARCHIVE.tmp"
done
mv "$ARCHIVE.tmp" "$ARCHIVE"

size="$(file_size "$ARCHIVE")"
if [ "$size" != "$EXPECTED_SIZE" ]; then
  echo "Error: size mismatch: got $size expected $EXPECTED_SIZE" >&2
  exit 1
fi

sha256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [ "$sha256" != "$EXPECTED_SHA256" ]; then
  echo "Error: sha256 mismatch: got $sha256 expected $EXPECTED_SHA256" >&2
  exit 1
fi

echo "Archive verified: $ARCHIVE"
echo "sha256: $sha256"

if [ "$EXTRACT" = true ]; then
  tar -xzf "$ARCHIVE" -C "$DATA_ROOT"
fi
