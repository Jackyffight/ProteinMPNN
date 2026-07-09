# Dataset Versions

We maintain two dataset tracks.

## Track A: Upstream Reference Baseline

Version id:

```text
proteinmpnn_upstream_pdb_2021aug02
```

Purpose:

- reproduce the upstream ProteinMPNN training setup
- validate launcher, metrics, checkpointing, and V100/A100 presets
- provide a stable baseline before changing data curation

Source archive:

```text
https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz
```

Local processed dataset:

```text
/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn/pdb_2021aug02
```

This is a ProteinMPNN upstream reference dataset, not the current complete PDB
archive.

## Track B: Owned Latest PDB Dataset

Version id pattern:

```text
proteinmpnn_pdb_latest_<YYYYMMDD>
```

Current 2026 build id:

```text
proteinmpnn_pdb_20260708
```

Purpose:

- build our own current PDB-derived ProteinMPNN training dataset
- keep provenance, filters, split policy, and preprocessing code owned by us
- support later refreshes without losing the upstream baseline

Raw source:

```text
https://files.wwpdb.org/pub/pdb/data/assemblies/mmCIF/divided/
```

The RCSB PDB file download documentation says the PDB archive is maintained by
wwPDB, is available over HTTPS, and includes biological assembly coordinate files
in mmCIF format under `/pub/pdb/data/assemblies/mmCIF`.

Local raw layout:

```text
/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn_custom/
  proteinmpnn_pdb_20260708/
    README.md
    dataset_manifest.json
    raw/
      assemblies_mmcif/
      sequence_clusters/
      metadata/
    processed/
      proteinmpnn/
    splits/
```

Initial build stages:

1. sync current wwPDB biological assembly mmCIF files
2. download RCSB 30% weekly sequence clusters for split grouping
3. download wwPDB `entries.idx` metadata for deposition date, resolution, and method
4. parse assembly mmCIF into ProteinMPNN-compatible structure records
5. filter proteins by experimental method, resolution, polymer type, missing atoms,
   chain length, and residue alphabet
6. create train/valid/test split by cluster id
7. write `list.csv`, `valid_clusters.txt`, `test_clusters.txt`, and `pdb/**/*.pt`
8. run smoke training against the new processed dataset

Build the current 2026 snapshot:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/build_pdb_2026_dataset.sh
```

Build the raw 2026 snapshot on dev2 for later transfer:

```bash
cd /data00/home/wangzhi.wit/models/ProteinMPNN
LOCAL_DATA=/data00/home/wangzhi.wit/models/datasets/proteinmpnn_custom
VERSION=proteinmpnn_pdb_20260708

python scripts/download_wwpdb_assemblies_https.py \
  --dest "$LOCAL_DATA/$VERSION/raw/assemblies_mmcif" \
  --workers 32

DATA_ROOT="$LOCAL_DATA" VERSION_ID="$VERSION" \
  scripts/download_rcsb_sequence_clusters.sh

DATA_ROOT="$LOCAL_DATA" VERSION_ID="$VERSION" \
  scripts/download_wwpdb_entries_index.sh

PYTHONPATH=/data00/home/wangzhi.wit/models/.pdbbuild_deps \
DATA_ROOT="$LOCAL_DATA" VERSION_ID="$VERSION" \
  scripts/build_pdb_2026_dataset.sh --skip-sync --skip-clusters --skip-metadata --workers 32
```

Build only structures deposited in 2026:

```bash
scripts/build_pdb_2026_dataset.sh --min-date 2026-01-01
```

Do not mix predicted structures into this dataset until the experimental-structure
baseline is trained and evaluated.
