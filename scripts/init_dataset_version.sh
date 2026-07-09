#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"
DATA_ROOT="${DATA_ROOT:-$PROTEINMPNN_CUSTOM_DATA_ROOT}"
VERSION_ID="${1:-proteinmpnn_pdb_latest_$(date +%Y%m%d)}"
VERSION_DIR="$DATA_ROOT/$VERSION_ID"

mkdir -p \
  "$VERSION_DIR/raw/assemblies_mmcif" \
  "$VERSION_DIR/raw/sequence_clusters" \
  "$VERSION_DIR/raw/metadata" \
  "$VERSION_DIR/processed" \
  "$VERSION_DIR/splits" \
  "$VERSION_DIR/logs"

cat > "$VERSION_DIR/README.md" <<EOF
# $VERSION_ID

Owned ProteinMPNN dataset version.

## Raw Source

https://files.wwpdb.org/pub/pdb/data/assemblies/mmCIF/divided/

## Layout

\`\`\`text
raw/assemblies_mmcif/
raw/sequence_clusters/
raw/metadata/
processed/
splits/
logs/
dataset_manifest.json
\`\`\`

## Build Status

- [ ] raw wwPDB biological assembly mmCIF synced
- [ ] RCSB 30% sequence clusters downloaded
- [ ] wwPDB entries.idx metadata downloaded
- [ ] mmCIF parsed into ProteinMPNN records
- [ ] filters applied
- [ ] sequence clusters created
- [ ] train/valid/test split written
- [ ] smoke train passed
EOF

cat > "$VERSION_DIR/dataset_manifest.json" <<EOF
{
  "version_id": "$VERSION_ID",
  "created_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source": {
    "name": "wwPDB current biological assembly mmCIF archive",
    "url": "https://files.wwpdb.org/pub/pdb/data/assemblies/mmCIF/divided/",
    "access_method": "pending"
  },
  "status": "initialized",
  "paths": {
    "raw_assemblies_mmcif": "$VERSION_DIR/raw/assemblies_mmcif",
    "processed": "$VERSION_DIR/processed",
    "splits": "$VERSION_DIR/splits"
  }
}
EOF

echo "$VERSION_DIR"
