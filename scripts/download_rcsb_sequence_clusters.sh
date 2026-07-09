#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_nas.sh"

DATA_ROOT="${DATA_ROOT:-$PROTEINMPNN_CUSTOM_DATA_ROOT}"
VERSION_ID="${VERSION_ID:-proteinmpnn_pdb_20260708}"
SEQ_ID="${SEQ_ID:-30}"
URL="${URL:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/download_rcsb_sequence_clusters.sh [options]

Download RCSB weekly polymer entity sequence clusters for custom ProteinMPNN
dataset splitting. The default sequence identity threshold is 30%.

Options:
  --version-id <id>       Dataset version id. Default: proteinmpnn_pdb_20260708.
  --data-root <dir>       Custom dataset root. Default: NAS MPNN custom root.
  --seq-id <n>            Cluster identity threshold. Default: 30.
  --url <url>             Override source URL.
  -h, --help              Show this help.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version-id|--version_id) VERSION_ID="$2"; shift 2 ;;
    --data-root|--data_root) DATA_ROOT="$2"; shift 2 ;;
    --seq-id|--seq_id) SEQ_ID="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

VERSION_DIR="$DATA_ROOT/$VERSION_ID"
DEST_DIR="$VERSION_DIR/raw/sequence_clusters"
DEST="$DEST_DIR/clusters-by-entity-${SEQ_ID}.txt"
LOG_DIR="$VERSION_DIR/logs"
if [ -z "$URL" ]; then
  URL="https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-${SEQ_ID}.txt"
fi

if [ ! -f "$VERSION_DIR/dataset_manifest.json" ]; then
  DATA_ROOT="$DATA_ROOT" "$SCRIPT_DIR/init_dataset_version.sh" "$VERSION_ID" >/dev/null
fi
mkdir -p "$DEST_DIR" "$LOG_DIR"

echo "version_id: $VERSION_ID"
echo "url: $URL"
echo "dest: $DEST"

curl \
  --fail \
  --http1.1 \
  --location \
  --retry 20 \
  --retry-all-errors \
  --retry-delay 10 \
  --retry-connrefused \
  --connect-timeout 30 \
  --continue-at - \
  --output "$DEST.tmp" \
  "$URL" 2>&1 | tee "$LOG_DIR/sequence_clusters_${SEQ_ID}_$(date +%Y%m%d%H%M%S).log"
mv "$DEST.tmp" "$DEST"

cat > "$VERSION_DIR/sequence_cluster_manifest.json" <<EOF
{
  "version_id": "$VERSION_ID",
  "downloaded_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sequence_identity": "$SEQ_ID",
  "url": "$URL",
  "path": "$DEST",
  "line_count": $(wc -l < "$DEST")
}
EOF

echo "sequence_clusters: $DEST"
echo "manifest: $VERSION_DIR/sequence_cluster_manifest.json"
