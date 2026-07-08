#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/../datasets/proteinmpnn/pdb_2021aug02}"

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

echo "dataset_dir: $DATA_DIR"
echo "list_rows: $(wc -l < "$DATA_DIR/list.csv")"
echo "valid_clusters: $(wc -l < "$DATA_DIR/valid_clusters.txt")"
echo "test_clusters: $(wc -l < "$DATA_DIR/test_clusters.txt")"
echo "structure_files: $(find "$DATA_DIR/pdb" -type f | wc -l)"
