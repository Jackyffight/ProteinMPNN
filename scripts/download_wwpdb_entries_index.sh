#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

DATA_ROOT="${DATA_ROOT:-$PROTEINMPNN_CUSTOM_DATA_ROOT}"
VERSION_ID="${VERSION_ID:-proteinmpnn_pdb_20260708}"
URL="${URL:-https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx}"

usage() {
  cat <<'EOF'
Usage:
  scripts/download_wwpdb_entries_index.sh [options]

Download wwPDB derived_data/index/entries.idx for deposition date, resolution,
and experimental method metadata.

Options:
  --version-id <id>       Dataset version id. Default: proteinmpnn_pdb_20260708.
  --data-root <dir>       Custom dataset root. Default: NAS MPNN custom root.
  --url <url>             Override source URL.
  -h, --help              Show this help.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version-id|--version_id) VERSION_ID="$2"; shift 2 ;;
    --data-root|--data_root) DATA_ROOT="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

VERSION_DIR="$DATA_ROOT/$VERSION_ID"
DEST_DIR="$VERSION_DIR/raw/metadata"
DEST="$DEST_DIR/entries.idx"
LOG_DIR="$VERSION_DIR/logs"

if [ ! -f "$VERSION_DIR/dataset_manifest.json" ]; then
  DATA_ROOT="$DATA_ROOT" "$SCRIPT_DIR/init_dataset_version.sh" "$VERSION_ID" >/dev/null
fi
mkdir -p "$DEST_DIR" "$LOG_DIR"

echo "version_id: $VERSION_ID"
echo "url: $URL"
echo "dest: $DEST"

curl \
  --fail \
  --location \
  --retry 20 \
  --retry-delay 10 \
  --retry-connrefused \
  --connect-timeout 30 \
  --output "$DEST.tmp" \
  "$URL" 2>&1 | tee "$LOG_DIR/entries_index_$(date +%Y%m%d%H%M%S).log"
mv "$DEST.tmp" "$DEST"

cat > "$VERSION_DIR/entries_index_manifest.json" <<EOF
{
  "version_id": "$VERSION_ID",
  "downloaded_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "url": "$URL",
  "path": "$DEST",
  "bytes": $(stat -c%s "$DEST")
}
EOF

echo "entries_index: $DEST"
echo "manifest: $VERSION_DIR/entries_index_manifest.json"
