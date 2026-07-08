#!/usr/bin/env bash
# Stage an already-downloaded ProteinMPNN dataset into the NAS MPNN workspace.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

DATA_ROOT="${DATA_ROOT:-$PROTEINMPNN_DATA_ROOT}"
ARCHIVE_NAME="pdb_2021aug02.tar.gz"
EXPECTED_SHA256="84d51d0b9224011db8deeab8b83e96f092830aaf6a1f538b1d94b0144f295714"
SOURCE_ARCHIVE="${SOURCE_ARCHIVE:-}"
SOURCE_DIR="${SOURCE_DIR:-}"
EXTRACT=true

usage() {
  cat <<'EOF'
Usage:
  scripts/stage_existing_dataset.sh [options]

Copies an already-downloaded ProteinMPNN upstream reference archive or extracted
directory into the NAS MPNN dataset root. Use this before falling back to the
slow public IPD download.

Options:
  --source-archive <path> Existing pdb_2021aug02.tar.gz.
  --source-dir <path>     Existing extracted pdb_2021aug02 directory.
  --data-root <dir>       Destination dataset root. Default: NAS MPNN datasets/proteinmpnn.
  --no-extract            Do not extract after staging archive.
  -h, --help              Show this help.

Default source candidates:
  /data00/home/wangzhi.wit/models/datasets/proteinmpnn/pdb_2021aug02.tar.gz
  /data00/home/wangzhi.wit/models/datasets/proteinmpnn/pdb_2021aug02
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --source-archive|--source_archive) SOURCE_ARCHIVE="$2"; shift 2 ;;
    --source-dir|--source_dir) SOURCE_DIR="$2"; shift 2 ;;
    --data-root|--data_root) DATA_ROOT="$2"; shift 2 ;;
    --no-extract|--no_extract) EXTRACT=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

copy_path() {
  local src="$1"
  local dst="$2"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$src" "$dst"
  else
    cp -a "$src" "$dst"
  fi
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

if [ -z "$SOURCE_ARCHIVE" ]; then
  candidate="/data00/home/wangzhi.wit/models/datasets/proteinmpnn/$ARCHIVE_NAME"
  if [ -f "$candidate" ]; then
    SOURCE_ARCHIVE="$candidate"
  fi
fi
if [ -z "$SOURCE_DIR" ]; then
  candidate="/data00/home/wangzhi.wit/models/datasets/proteinmpnn/pdb_2021aug02"
  if [ -d "$candidate" ]; then
    SOURCE_DIR="$candidate"
  fi
fi

mkdir -p "$DATA_ROOT"
DEST_ARCHIVE="$DATA_ROOT/$ARCHIVE_NAME"
DEST_DIR="$DATA_ROOT/pdb_2021aug02"

if [ -n "$SOURCE_ARCHIVE" ]; then
  if [ ! -f "$SOURCE_ARCHIVE" ]; then
    echo "Source archive not found: $SOURCE_ARCHIVE" >&2
    exit 1
  fi
  echo "Staging archive:"
  echo "  source: $SOURCE_ARCHIVE"
  echo "  target: $DEST_ARCHIVE"
  copy_path "$SOURCE_ARCHIVE" "$DEST_ARCHIVE.tmp"
  actual_sha="$(sha256_of "$DEST_ARCHIVE.tmp")"
  if [ "$actual_sha" != "$EXPECTED_SHA256" ]; then
    echo "SHA256 mismatch after copy: got $actual_sha expected $EXPECTED_SHA256" >&2
    exit 1
  fi
  mv "$DEST_ARCHIVE.tmp" "$DEST_ARCHIVE"
fi

if [ "$EXTRACT" = true ] && [ -f "$DEST_ARCHIVE" ]; then
  if [ -d "$DEST_DIR" ]; then
    echo "Extracted dataset already exists: $DEST_DIR"
  else
    echo "Extracting archive into: $DATA_ROOT"
    tar -xzf "$DEST_ARCHIVE" -C "$DATA_ROOT"
  fi
elif [ -n "$SOURCE_DIR" ]; then
  if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source dataset dir not found: $SOURCE_DIR" >&2
    exit 1
  fi
  if [ -d "$DEST_DIR" ]; then
    echo "Extracted dataset already exists: $DEST_DIR"
  else
    echo "Staging extracted dataset directory:"
    echo "  source: $SOURCE_DIR"
    echo "  target: $DEST_DIR"
    copy_path "$SOURCE_DIR" "$DEST_DIR"
  fi
fi

if [ ! -f "$DEST_ARCHIVE" ] && [ ! -d "$DEST_DIR" ]; then
  echo "No source archive or source dir was found." >&2
  echo "Pass --source-archive/--source-dir, or use scripts/download_dataset_parts.sh --extract." >&2
  exit 1
fi

echo "Dataset staged under: $DATA_ROOT"
echo "Run: scripts/validate_dataset.sh"
