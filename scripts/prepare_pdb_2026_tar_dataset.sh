#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

DATA_ROOT="${DATA_ROOT:-$PROTEINMPNN_CUSTOM_DATA_ROOT}"
VERSION_ID="${VERSION_ID:-proteinmpnn_pdb_20260708}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

ASSEMBLY_ID="${ASSEMBLY_ID:-all}"
ASSEMBLY_POLICY="${ASSEMBLY_POLICY:-first}"
RAW_WORKERS="${RAW_WORKERS:-64}"
RAW_MAX_IN_FLIGHT="${RAW_MAX_IN_FLIGHT:-256}"
RAW_RETRIES="${RAW_RETRIES:-20}"
RAW_RETRY_DELAY="${RAW_RETRY_DELAY:-5}"
RAW_ROUNDS="${RAW_ROUNDS:-5}"
RAW_LIMIT="${RAW_LIMIT:-}"
FORCE_LIST=false

SEQ_ID="${SEQ_ID:-30}"
CLUSTER_WORKERS="${CLUSTER_WORKERS:-12}"
CLUSTER_PART_BYTES="${CLUSTER_PART_BYTES:-524288}"
CLUSTER_RETRIES="${CLUSTER_RETRIES:-200}"
CLUSTER_URL="${CLUSTER_URL:-https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-${SEQ_ID}.txt}"

BUILD_WORKERS="${BUILD_WORKERS:-2}"
BUILD_MAX_IN_FLIGHT="${BUILD_MAX_IN_FLIGHT:-2}"
MAX_RESOLUTION="${MAX_RESOLUTION:-3.5}"
MIN_RESOLVED_RESIDUES="${MIN_RESOLVED_RESIDUES:-30}"
MIN_BACKBONE_COVERAGE="${MIN_BACKBONE_COVERAGE:-0.5}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-2000}"
MIN_DATE="${MIN_DATE:-2021-08-03}"
MAX_DATE="${MAX_DATE:-2026-07-08}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-1g}"
BUILD_MAX_RAW_FILE_SIZE="${BUILD_MAX_RAW_FILE_SIZE:-50m}"
FORCE_BUILD=false

MODE="all"

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_pdb_2026_tar_dataset.sh [options]

Foreground, tmux-friendly pipeline for the owned 2026 ProteinMPNN tar-shard
dataset:
  1. initialize dataset version
  2. download wwPDB biological assembly mmCIF files
  3. download wwPDB entries.idx metadata
  4. download RCSB sequence clusters
  5. build ProteinMPNN tar shards directly from mmCIF

The script is resumable. Re-running it skips already downloaded raw files and
existing metadata/cluster files, then rebuilds shards only when requested.

Options:
  --version-id <id>        Dataset version id. Default: proteinmpnn_pdb_20260708.
  --data-root <dir>        Custom dataset root. Default: $PROTEINMPNN_CUSTOM_DATA_ROOT.
  --output-dir <dir>       Tar-shard output dir. Default: VERSION/processed/proteinmpnn_tar_shards_v1.
  --assembly-id <id|all>   Assembly selector for raw download/build. Default: all.
  --assembly-policy <first|all>
                           Canonical assembly policy. Default: first.
  --raw-workers <n>        Raw mmCIF download workers. Default: 64.
  --raw-max-in-flight <n>  Raw futures in flight. Default: 256.
  --raw-rounds <n>         Retry whole raw sync up to n rounds. Default: 5.
  --raw-limit <n>          Debug: download/build only first n assembly files.
  --force-list             Rebuild raw download manifest.
  --cluster-workers <n>    Parallel Range workers for cluster download. Default: 12.
  --build-workers <n>      mmCIF parser workers. Default: 2.
  --build-max-in-flight <n> Build futures in flight. Default: 2.
  --build-max-raw-file-size <size>
                            Skip compressed raw mmCIF files larger than this.
                            Default: 50m. Use 0 to disable.
  --max-resolution <a>     Resolution cutoff. Default: 3.5.
  --min-resolved-residues <n>
                           Minimum complete-backbone residues per chain. Default: 30.
  --min-backbone-coverage <f>
                           Minimum complete-backbone fraction per chain. Default: 0.5.
  --max-context-length <n> Defer assemblies above this total length. Default: 2000.
  --min-date <YYYY-MM-DD>  Deposition lower bound. Default: 2021-08-03.
  --max-date <YYYY-MM-DD>  Deposition upper bound. Default: 2026-07-08.
  --max-shard-size <size>  Shard target size. Default: 1g.
  --force-build            Replace output dir when building.
  --download-only          Stop after raw/metadata/cluster downloads.
  --build-only             Only build tar shards from existing raw data.
  --status                 Print current dataset progress and exit.
  -h, --help               Show this help.

Common usage:
  tmux new -s pdb2026
  cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
  ./scripts/prepare_pdb_2026_tar_dataset.sh --force-build

Local dev2 override:
  DATA_ROOT=/data00/home/wangzhi.wit/models/datasets/proteinmpnn_custom \
  OUTPUT_DIR=/data00/home/wangzhi.wit/models/datasets/proteinmpnn_custom/proteinmpnn_pdb_20260708/processed/proteinmpnn_tar_shards_v1 \
  ./scripts/prepare_pdb_2026_tar_dataset.sh --force-build
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version-id|--version_id) VERSION_ID="$2"; shift 2 ;;
    --data-root|--data_root) DATA_ROOT="$2"; shift 2 ;;
    --output-dir|--output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --assembly-id|--assembly_id) ASSEMBLY_ID="$2"; shift 2 ;;
    --assembly-policy|--assembly_policy) ASSEMBLY_POLICY="$2"; shift 2 ;;
    --raw-workers|--raw_workers) RAW_WORKERS="$2"; shift 2 ;;
    --raw-max-in-flight|--raw_max_in_flight) RAW_MAX_IN_FLIGHT="$2"; shift 2 ;;
    --raw-rounds|--raw_rounds) RAW_ROUNDS="$2"; shift 2 ;;
    --raw-limit|--raw_limit) RAW_LIMIT="$2"; shift 2 ;;
    --force-list|--force_list) FORCE_LIST=true; shift ;;
    --cluster-workers|--cluster_workers) CLUSTER_WORKERS="$2"; shift 2 ;;
    --build-workers|--build_workers) BUILD_WORKERS="$2"; shift 2 ;;
    --build-max-in-flight|--build_max_in_flight) BUILD_MAX_IN_FLIGHT="$2"; shift 2 ;;
    --build-max-raw-file-size|--build_max_raw_file_size) BUILD_MAX_RAW_FILE_SIZE="$2"; shift 2 ;;
    --max-resolution|--max_resolution) MAX_RESOLUTION="$2"; shift 2 ;;
    --min-resolved-residues|--min_resolved_residues) MIN_RESOLVED_RESIDUES="$2"; shift 2 ;;
    --min-backbone-coverage|--min_backbone_coverage) MIN_BACKBONE_COVERAGE="$2"; shift 2 ;;
    --max-context-length|--max_context_length) MAX_CONTEXT_LENGTH="$2"; shift 2 ;;
    --min-date|--min_date) MIN_DATE="$2"; shift 2 ;;
    --max-date|--max_date) MAX_DATE="$2"; shift 2 ;;
    --max-shard-size|--max_shard_size) MAX_SHARD_SIZE="$2"; shift 2 ;;
    --force-build|--force_build) FORCE_BUILD=true; shift ;;
    --download-only|--download_only) MODE="download"; shift ;;
    --build-only|--build_only) MODE="build"; shift ;;
    --status) MODE="status"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$DATA_ROOT/$VERSION_ID/processed/proteinmpnn_tar_shards_v1"
fi

VERSION_DIR="$DATA_ROOT/$VERSION_ID"
RAW_DIR="$VERSION_DIR/raw/assemblies_mmcif"
RAW_MANIFEST="$VERSION_DIR/raw/assembly_download_manifest_${ASSEMBLY_ID}.jsonl"
ENTRIES_INDEX="$VERSION_DIR/raw/metadata/entries.idx"
CLUSTER_FILE="$VERSION_DIR/raw/sequence_clusters/clusters-by-entity-${SEQ_ID}.txt"
LOG_DIR="$VERSION_DIR/logs"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

count_raw_files() {
  if [ -d "$RAW_DIR" ]; then
    find "$RAW_DIR" -type f -name '*.cif.gz' | wc -l | tr -d ' '
  else
    echo 0
  fi
}

raw_size_human() {
  if [ -d "$RAW_DIR" ]; then
    du -sh "$RAW_DIR" 2>/dev/null | awk '{print $1}'
  else
    echo 0
  fi
}

manifest_count() {
  if [ -f "$RAW_MANIFEST" ]; then
    wc -l < "$RAW_MANIFEST" | tr -d ' '
  else
    echo 0
  fi
}

print_config() {
  cat <<EOF
=== ProteinMPNN 2026 tar dataset ===
repo: $REPO_ROOT
version_id: $VERSION_ID
data_root: $DATA_ROOT
raw_dir: $RAW_DIR
raw_manifest: $RAW_MANIFEST
entries_index: $ENTRIES_INDEX
cluster_file: $CLUSTER_FILE
output_dir: $OUTPUT_DIR
assembly_id: $ASSEMBLY_ID
assembly_policy: $ASSEMBLY_POLICY
raw_workers: $RAW_WORKERS
raw_max_in_flight: $RAW_MAX_IN_FLIGHT
raw_rounds: $RAW_ROUNDS
cluster_workers: $CLUSTER_WORKERS
build_workers: $BUILD_WORKERS
build_max_in_flight: $BUILD_MAX_IN_FLIGHT
build_max_raw_file_size: $BUILD_MAX_RAW_FILE_SIZE
max_resolution: $MAX_RESOLUTION
min_resolved_residues: $MIN_RESOLVED_RESIDUES
min_backbone_coverage: $MIN_BACKBONE_COVERAGE
max_context_length: $MAX_CONTEXT_LENGTH
min_date: $MIN_DATE
max_date: $MAX_DATE
mode: $MODE
EOF
}

print_status() {
  local expected actual
  expected="$(manifest_count)"
  actual="$(count_raw_files)"
  print_config
  echo "raw_expected_files: $expected"
  echo "raw_actual_files: $actual"
  echo "raw_size: $(raw_size_human)"
  if [ "$expected" != "0" ]; then
    python3 - "$expected" "$actual" <<'PY'
import sys
expected = int(sys.argv[1])
actual = int(sys.argv[2])
print(f"raw_file_progress: {actual / expected * 100:.2f}%")
PY
  fi
  if [ -f "$ENTRIES_INDEX" ]; then
    echo "entries_index: ready ($(du -h "$ENTRIES_INDEX" | cut -f1))"
  else
    echo "entries_index: missing"
  fi
  if [ -f "$CLUSTER_FILE" ]; then
    echo "cluster_file: ready ($(du -h "$CLUSTER_FILE" | cut -f1), lines=$(wc -l < "$CLUSTER_FILE"))"
  else
    echo "cluster_file: missing"
  fi
  if [ -d "$OUTPUT_DIR" ]; then
    echo "tar_output: $(du -sh "$OUTPUT_DIR" | awk '{print $1}')"
    echo "tar_shards: $(find "$OUTPUT_DIR/shards" -type f -name '*.tar' 2>/dev/null | wc -l | tr -d ' ')"
    if [ -f "$OUTPUT_DIR/manifest.json" ]; then
      echo "tar_manifest: ready"
    else
      echo "tar_manifest: missing"
    fi
  else
    echo "tar_output: missing"
  fi
}

init_version() {
  if [ ! -f "$VERSION_DIR/dataset_manifest.json" ]; then
    log "initializing dataset version"
    DATA_ROOT="$DATA_ROOT" "$SCRIPT_DIR/init_dataset_version.sh" "$VERSION_ID" >/dev/null
  fi
  mkdir -p "$RAW_DIR" "$VERSION_DIR/raw/metadata" "$VERSION_DIR/raw/sequence_clusters" "$LOG_DIR"
}

download_raw() {
  local round rc expected actual list_flag limit_args
  list_flag=()
  limit_args=()
  if [ "$FORCE_LIST" = true ]; then
    list_flag=(--force-list)
  fi
  if [ -n "$RAW_LIMIT" ]; then
    limit_args=(--limit "$RAW_LIMIT")
  fi

  for round in $(seq 1 "$RAW_ROUNDS"); do
    log "raw sync round $round/$RAW_ROUNDS"
    set +e
    PYTHONUNBUFFERED=1 python3 "$SCRIPT_DIR/download_wwpdb_assemblies_https.py" \
      --dest "$RAW_DIR" \
      --workers "$RAW_WORKERS" \
      --max-in-flight "$RAW_MAX_IN_FLIGHT" \
      --retries "$RAW_RETRIES" \
      --retry-delay "$RAW_RETRY_DELAY" \
      --assembly-id "$ASSEMBLY_ID" \
      --manifest "$RAW_MANIFEST" \
      "${limit_args[@]}" \
      "${list_flag[@]}" \
      2>&1 | tee "$LOG_DIR/raw_download_round_${round}_$(date +%Y%m%d%H%M%S).log"
    rc=${PIPESTATUS[0]}
    set -e
    expected="$(manifest_count)"
    actual="$(count_raw_files)"
    log "raw status actual=$actual expected=$expected size=$(raw_size_human) rc=$rc"
    if [ "$expected" != "0" ] && [ "$actual" = "$expected" ] && [ "$rc" = "0" ]; then
      return 0
    fi
    list_flag=()
  done

  expected="$(manifest_count)"
  actual="$(count_raw_files)"
  echo "ERROR raw sync incomplete after $RAW_ROUNDS rounds: actual=$actual expected=$expected" >&2
  return 1
}

download_entries_index() {
  if [ -s "$ENTRIES_INDEX" ]; then
    log "entries index already exists: $ENTRIES_INDEX"
    return 0
  fi
  log "downloading entries.idx"
  DATA_ROOT="$DATA_ROOT" VERSION_ID="$VERSION_ID" "$SCRIPT_DIR/download_wwpdb_entries_index.sh"
}

download_sequence_clusters() {
  log "checking/downloading sequence clusters"
  python3 - "$CLUSTER_URL" "$CLUSTER_FILE" "$VERSION_DIR" "$SEQ_ID" "$CLUSTER_WORKERS" "$CLUSTER_PART_BYTES" "$CLUSTER_RETRIES" <<'PY'
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

url = sys.argv[1]
dest = Path(sys.argv[2])
version_dir = Path(sys.argv[3])
seq_id = sys.argv[4]
workers = int(sys.argv[5])
part_bytes = int(sys.argv[6])
retries = int(sys.argv[7])
ua = "ProteinMPNN-dataset-builder/1.0"


def log(message: str) -> None:
    print(f"[cluster] {message}", flush=True)


def remote_size() -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": ua})
    with urllib.request.urlopen(request, timeout=60) as response:
        value = response.headers.get("Content-Length")
        if not value:
            raise RuntimeError("missing Content-Length for cluster file")
        return int(value)


def local_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def fetch_range(index: int, start: int, end: int, part_dir: Path) -> None:
    part = part_dir / f"part_{index:05d}"
    tmp = part.with_suffix(".tmp")
    expected = end - start + 1
    if local_size(part) == expected:
        return
    if local_size(tmp) == expected:
        tmp.replace(part)
        return
    if tmp.exists():
        tmp.unlink()

    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": ua,
                "Range": f"bytes={start}-{end}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 256)
            got = local_size(tmp)
            if got == expected:
                tmp.replace(part)
                return
            log(f"part={index} bad_size got={got} expected={expected}")
            tmp.unlink(missing_ok=True)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt % 10 == 0 or attempt == 1:
                log(f"part={index} attempt={attempt} error={type(exc).__name__}: {exc}")
        time.sleep(min(30, 1 + attempt // 10))
    raise RuntimeError(f"failed to download cluster part {index}")


dest.parent.mkdir(parents=True, exist_ok=True)
version_dir.mkdir(parents=True, exist_ok=True)
size = remote_size()
if dest.exists() and dest.stat().st_size == size:
    log(f"ready path={dest} size={size}")
else:
    part_dir = dest.parent / f".{dest.name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    count = math.ceil(size / part_bytes)
    ranges = []
    for index in range(count):
        start = index * part_bytes
        end = min(size - 1, start + part_bytes - 1)
        ranges.append((index, start, end))
    log(f"downloading url={url} size={size} parts={count} workers={workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_range, index, start, end, part_dir) for index, start, end in ranges]
        done = 0
        for future in concurrent.futures.as_completed(futures):
            future.result()
            done += 1
            if done % 10 == 0 or done == count:
                bytes_done = sum(local_size(part_dir / f"part_{i:05d}") for i in range(count))
                log(f"parts_done={done}/{count} bytes_done={bytes_done}/{size}")

    merged = dest.with_suffix(dest.suffix + ".merge")
    with merged.open("wb") as out:
        for index in range(count):
            part = part_dir / f"part_{index:05d}"
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, out, length=1024 * 1024)
    if merged.stat().st_size != size:
        raise RuntimeError(f"merged cluster size mismatch: {merged.stat().st_size} != {size}")
    merged.replace(dest)
    log(f"merged path={dest} size={size}")

with dest.open("rb") as handle:
    line_count = sum(1 for _ in handle)

manifest = {
    "version_id": version_dir.name,
    "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "sequence_identity": seq_id,
    "url": url,
    "path": str(dest),
    "bytes": dest.stat().st_size,
    "line_count": line_count,
}
(version_dir / "sequence_cluster_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
log(f"manifest={version_dir / 'sequence_cluster_manifest.json'} lines={line_count}")
PY
}

build_tar_shards() {
  local force_args min_date_args limit_args
  force_args=()
  min_date_args=()
  limit_args=()
  if [ "$FORCE_BUILD" = true ]; then
    force_args=(--force)
  fi
  if [ -n "$MIN_DATE" ]; then
    min_date_args=(--min-date "$MIN_DATE")
  fi
  if [ -n "$RAW_LIMIT" ]; then
    limit_args=(--limit "$RAW_LIMIT")
  fi
  log "building tar shards"
  DATA_ROOT="$DATA_ROOT" VERSION_ID="$VERSION_ID" OUTPUT_DIR="$OUTPUT_DIR" \
  WORKERS="$BUILD_WORKERS" MAX_IN_FLIGHT="$BUILD_MAX_IN_FLIGHT" \
  ASSEMBLY_ID="$ASSEMBLY_ID" ASSEMBLY_POLICY="$ASSEMBLY_POLICY" \
  MAX_RESOLUTION="$MAX_RESOLUTION" \
  MIN_RESOLVED_RESIDUES="$MIN_RESOLVED_RESIDUES" \
  MIN_BACKBONE_COVERAGE="$MIN_BACKBONE_COVERAGE" \
  MAX_CONTEXT_LENGTH="$MAX_CONTEXT_LENGTH" MAX_DATE="$MAX_DATE" \
  MAX_SHARD_SIZE="$MAX_SHARD_SIZE" MAX_RAW_FILE_SIZE="$BUILD_MAX_RAW_FILE_SIZE" \
    "$SCRIPT_DIR/build_pdb_2026_tar_shards.sh" \
      "${force_args[@]}" \
      "${min_date_args[@]}" \
      "${limit_args[@]}"
}

main() {
  cd "$REPO_ROOT"
  if [ "$MODE" = "status" ]; then
    print_status
    return 0
  fi
  print_config
  init_version
  if [ "$MODE" != "build" ]; then
    download_raw
    download_entries_index
    download_sequence_clusters
  else
    log "build-only mode: using existing raw data, metadata, and sequence clusters"
  fi
  if [ "$MODE" = "download" ]; then
    print_status
    return 0
  fi
  build_tar_shards
  print_status
}

main "$@"
