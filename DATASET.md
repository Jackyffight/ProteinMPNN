# ProteinMPNN Dataset

## Upstream Reference Source

```text
https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz
```

The archive is the public upstream reference ProteinMPNN PDB training set used
by the upstream training scripts. It is not the current complete PDB archive.

## Local Layout

```text
/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn/
  pdb_2021aug02.tar.gz
  pdb_2021aug02/
    README
    list.csv
    valid_clusters.txt
    test_clusters.txt
    pdb/
  pdb_2021aug02_sample.tar.gz
  pdb_2021aug02_sample/
```

## Verified Local Facts

```text
archive_size_bytes: 18037128263
archive_sha256: 84d51d0b9224011db8deeab8b83e96f092830aaf6a1f538b1d94b0144f295714
expanded_size: about 62G
structure_files: 869544
```

Split files:

```text
list.csv
valid_clusters.txt
test_clusters.txt
```

The dataset is sufficient for retraining ProteinMPNN from scratch. It does not
depend on Hugging Face or on remote model APIs.

## Validation Commands

Download or rebuild the full archive from HTTP byte-range parts:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/download_dataset_parts.sh --extract
```

By default this writes parts under:

```text
/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn/parts/
```

and merges them into:

```text
/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn/pdb_2021aug02.tar.gz
```

Fast layout check:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/validate_dataset.sh
```

Full archive checksum, expensive because the file is 17G:

```bash
sha256sum /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn/pdb_2021aug02.tar.gz
```

## Latest PDB Track

Create and sync an owned latest-PDB dataset version:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/init_dataset_version.sh
scripts/sync_latest_pdb_assemblies.sh --dry-run
```

Remove `--dry-run` to start syncing current wwPDB biological assembly mmCIF
files. See `DATASET_VERSIONS.md` for the full two-track plan.
