#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <dataset_version_dir>" >&2
  exit 2
fi

VERSION_DIR="$1"
RAW_DIR="$VERSION_DIR/raw/assemblies_mmcif"

if [ ! -d "$VERSION_DIR" ]; then
  echo "Missing dataset version dir: $VERSION_DIR" >&2
  exit 1
fi

echo "version_dir: $VERSION_DIR"
if [ -f "$VERSION_DIR/dataset_manifest.json" ]; then
  echo "dataset_manifest: $VERSION_DIR/dataset_manifest.json"
fi
if [ -f "$VERSION_DIR/download_manifest.json" ]; then
  echo "download_manifest: $VERSION_DIR/download_manifest.json"
fi
if [ -d "$RAW_DIR" ]; then
  echo "raw_assembly_files: $(find "$RAW_DIR" -type f -name '*.cif.gz' | wc -l)"
  echo "raw_assembly_bytes: $(du -sb "$RAW_DIR" | awk '{print $1}')"
fi
