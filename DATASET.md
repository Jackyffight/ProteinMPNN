# ProteinMPNN Dataset

## Source

```text
https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz
```

The archive is the public ProteinMPNN PDB training set used by the upstream
training scripts.

## Local Layout

```text
/data00/home/wangzhi.wit/models/datasets/proteinmpnn/
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
cd /data00/home/wangzhi.wit/models/ProteinMPNN
scripts/download_dataset_parts.sh --extract
```

By default this writes parts under:

```text
/data00/home/wangzhi.wit/models/datasets/proteinmpnn/parts/
```

and merges them into:

```text
/data00/home/wangzhi.wit/models/datasets/proteinmpnn/pdb_2021aug02.tar.gz
```

Fast layout check:

```bash
cd /data00/home/wangzhi.wit/models/ProteinMPNN
scripts/validate_dataset.sh
```

Full archive checksum, expensive because the file is 17G:

```bash
sha256sum /data00/home/wangzhi.wit/models/datasets/proteinmpnn/pdb_2021aug02.tar.gz
```
