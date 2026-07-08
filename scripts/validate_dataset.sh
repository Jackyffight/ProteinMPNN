#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_nas.sh"
DATA_DIR="${DATA_DIR:-$PROTEINMPNN_DATA_ROOT/pdb_2021aug02}"

if [ ! -d "$DATA_DIR" ]; then
  echo "Missing dataset directory: $DATA_DIR" >&2
  exit 1
fi

for required in README list.csv valid_clusters.txt test_clusters.txt pdb; do
  if [ ! -e "$DATA_DIR/$required" ]; then
    echo "Missing required dataset entry: $DATA_DIR/$required" >&2
    exit 1
  fi
done

EXPECTED_STRUCTURE_FILES="${EXPECTED_STRUCTURE_FILES:-869544}"
STRUCTURE_FILES="$(find "$DATA_DIR/pdb" -type f | wc -l | tr -d ' ')"

echo "dataset_dir: $DATA_DIR"
echo "list_rows: $(wc -l < "$DATA_DIR/list.csv")"
echo "valid_clusters: $(wc -l < "$DATA_DIR/valid_clusters.txt")"
echo "test_clusters: $(wc -l < "$DATA_DIR/test_clusters.txt")"
echo "structure_files: $STRUCTURE_FILES (expected $EXPECTED_STRUCTURE_FILES)"

# Assert the pinned structure-file count so a truncated/interrupted extraction fails
# here instead of silently training on a partial dataset. The archive sha256 is
# verified at download/stage time; this guards the extracted result.
if [ "$STRUCTURE_FILES" != "$EXPECTED_STRUCTURE_FILES" ]; then
  echo "Error: structure file count $STRUCTURE_FILES != expected $EXPECTED_STRUCTURE_FILES;" \
       "extraction is incomplete or corrupt. Re-extract, or set EXPECTED_STRUCTURE_FILES to override." >&2
  exit 1
fi
echo "dataset validation OK"
