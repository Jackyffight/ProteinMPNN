#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/../datasets/proteinmpnn_custom}"
VERSION_ID="${VERSION_ID:-proteinmpnn_pdb_latest_$(date +%Y%m%d)}"
VERSION_DIR="$DATA_ROOT/$VERSION_ID"
DEST="$VERSION_DIR/raw/assemblies_mmcif"

HTTPS_SOURCE="https://files.wwpdb.org/pub/pdb/data/assemblies/mmCIF/divided/"
RSYNC_SOURCE="${RSYNC_SOURCE:-rsync://rsync.rcsb.org/ftp_data/assemblies/mmCIF/divided/}"
METHOD="${METHOD:-auto}"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_latest_pdb_assemblies.sh [options]

Sync current wwPDB biological assembly mmCIF files for an owned ProteinMPNN
dataset version. The default destination is:
  ../datasets/proteinmpnn_custom/proteinmpnn_pdb_latest_<YYYYMMDD>/raw/assemblies_mmcif

Options:
  --version-id <id>       Dataset version id.
  --data-root <dir>       Custom dataset root.
  --method <auto|rsync|wget>
  --rsync-source <url>    Default: rsync://rsync.rcsb.org/ftp_data/assemblies/mmCIF/divided/
  --dry-run               Print commands without downloading.
  -h, --help              Show this help.

Notes:
  - rsync is preferred for maintaining a current mirror.
  - wget fallback uses HTTPS recursive download from files.wwpdb.org.
  - This script syncs raw data only; it does not yet build ProteinMPNN .pt files.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version-id|--version_id) VERSION_ID="$2"; shift 2 ;;
    --data-root|--data_root) DATA_ROOT="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --rsync-source|--rsync_source) RSYNC_SOURCE="$2"; shift 2 ;;
    --dry-run|--dry_run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$METHOD" != "auto" ] && [ "$METHOD" != "rsync" ] && [ "$METHOD" != "wget" ]; then
  echo "Error: --method must be auto, rsync, or wget." >&2
  exit 1
fi

VERSION_DIR="$DATA_ROOT/$VERSION_ID"
DEST="$VERSION_DIR/raw/assemblies_mmcif"

if [ ! -f "$VERSION_DIR/dataset_manifest.json" ]; then
  DATA_ROOT="$DATA_ROOT" "$SCRIPT_DIR/init_dataset_version.sh" "$VERSION_ID" >/dev/null
fi
mkdir -p "$DEST" "$VERSION_DIR/logs"

if [ "$METHOD" = "auto" ]; then
  if command -v rsync >/dev/null 2>&1; then
    METHOD="rsync"
  elif command -v wget >/dev/null 2>&1; then
    METHOD="wget"
  else
    echo "Error: install rsync or wget." >&2
    exit 1
  fi
fi

echo "version_id: $VERSION_ID"
echo "version_dir: $VERSION_DIR"
echo "method: $METHOD"
echo "dest: $DEST"

if [ "$METHOD" = "rsync" ]; then
  CMD=(rsync -av --partial --delete "$RSYNC_SOURCE" "$DEST/")
else
  CMD=(
    wget
    --recursive
    --no-parent
    --continue
    --no-host-directories
    --cut-dirs=6
    --accept "*.cif.gz"
    --directory-prefix "$DEST"
    "$HTTPS_SOURCE"
  )
fi

printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [ "$DRY_RUN" = false ]; then
  "${CMD[@]}" 2>&1 | tee "$VERSION_DIR/logs/sync_$(date +%Y%m%d%H%M%S).log"
fi

cat > "$VERSION_DIR/download_manifest.json" <<EOF
{
  "version_id": "$VERSION_ID",
  "synced_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "method": "$METHOD",
  "https_source": "$HTTPS_SOURCE",
  "rsync_source": "$RSYNC_SOURCE",
  "destination": "$DEST"
}
EOF

echo "download_manifest: $VERSION_DIR/download_manifest.json"
