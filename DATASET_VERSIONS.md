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
  proteinmpnn_pdb_latest_<YYYYMMDD>/
    README.md
    dataset_manifest.json
    raw/
      assemblies_mmcif/
    processed/
    splits/
```

Initial build stages:

1. sync current wwPDB biological assembly mmCIF files
2. parse assembly mmCIF into ProteinMPNN-compatible structure records
3. filter proteins by experimental method, resolution, polymer type, missing atoms,
   chain length, and residue alphabet
4. cluster sequences and create train/valid/test split
5. write `list.csv`, `valid_clusters.txt`, `test_clusters.txt`, and `pdb/**/*.pt`
6. run smoke training against the new processed dataset

Do not mix predicted structures into this dataset until the experimental-structure
baseline is trained and evaluated.
