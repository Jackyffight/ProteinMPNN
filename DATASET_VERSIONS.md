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

Current artifact status:

```text
v1 (validated 2026-07-11; training-ready main continuation set)
```

The older tar-shard build under
`datasets/proteinmpnn/proteinmpnn_tar_shards` is useful for storage and loader
testing only. Do not use it for model training or publish metrics from it. The
audit found these blocking semantic issues:

- residues missing any backbone atom were deleted and the remaining residues
  compacted, which can create false sequence adjacency
- each biological assembly file was emitted as a separate learning record,
  over-weighting PDB entries with many assemblies
- homology metadata was approximated by position-wise identity instead of a
  sequence alignment
- exact processed sequences and PDB entries cross train/valid/test boundaries
  under the current cluster assignment
- assemblies above 10,000 residues are silently excluded by the training length
  filter

The v1 replacement preserves sequence positions with coordinate masks, defines
one deterministic target-chain example policy, and regenerates splits from the
final processed sequences with explicit zero-leakage checks. Oversized
assemblies will be handled as target-chain plus spatial-neighbor/interface crops
under the token limit; they remain a separate stage until that policy is
implemented and validated.

### v1 Replacement

The v1 main dataset uses the 2026-07-08 wwPDB snapshot but includes only entries
deposited after the upstream baseline cutoff:

```text
2021-08-03 through 2026-07-08
```

This is the new-data continuation set. The upstream 2021 dataset remains a
separate replay source rather than being duplicated into v1. Each v1 record has:

- the lowest numbered available biological assembly for one PDB entry
- one deterministic target chain, selected by complete-backbone residue count,
  then coverage and source chain ID
- complete polymer sequence positions with missing coordinates represented by
  masks instead of deleted residues
- at most 2,000 total context residues; larger assemblies are written to
  `build_deferred_oversized.jsonl` for the later crop stage
- final exact-sequence cluster union followed by hard zero-leakage assertions

Production output:

```text
proteinmpnn_pdb_20260708/processed/proteinmpnn_tar_shards_v1
```

Build and automatically validate it with bounded concurrency:

```bash
DATA_ROOT=/data00/home/wangzhi.wit/models/datasets/proteinmpnn_custom \
VERSION_ID=proteinmpnn_pdb_20260708 \
WORKERS=2 MAX_IN_FLIGHT=2 \
  scripts/build_pdb_2026_tar_shards.sh
```

The post-build validator checks every payload, coordinate mask, target/index
mapping, shard checksum, context-length bound, and split assignment, and writes
`validation.json` only after all checks pass.

The completed production build contains 46,619 target records and 97,952
context chains across nine shards (8.3 GiB). It retained 2,687,809 unresolved
polymer positions as masks, deferred 10,166 parsed oversized assemblies for stage
two, and separately logged 101 compressed raw files above 50 MiB. It reported
zero parser failures and passed exact-sequence and PDB split leakage checks with
zero violations.

### stage2a Oversized Spatial Crops

Stage2a consumes only the 10,166 assemblies that v1 parsed successfully and
deferred for total length or chain count. It does not process the separate set
of 101 compressed mmCIF files above 50 MiB. For each assembly it:

- preserves the complete deterministic target chain
- ranks other chains by minimum resolved CA distance to the target
- adds complete nearby chains while they fit, then at most one contiguous
  nearest-residue window, under the same 2,000-residue and 62-chain bounds
- clones retained tensor slices so serialized crops do not retain large backing
  storage
- inherits v1 valid/test assignments by homology cluster and exact sequence;
  clusters found only in stage2a remain train
- defers targets longer than 2,000 residues instead of making a discontinuous or
  truncated target

The production builder is fixed at one parser worker and one in-flight file. It
restarts that worker every 25 files, records peak RSS in the manifest, and
refuses to start with less than 8 GiB available memory. Build and validate with:

```bash
scripts/build_pdb_2026_oversized_crops.sh
```

Production output:

```text
proteinmpnn_pdb_20260708/processed/proteinmpnn_tar_shards_stage2a_v1
```

A 42-input boundary pilot covered a 69,654-residue, 78-chain assembly and two
over-budget single-chain targets. It produced 40 valid crops with zero parser or
split-leak failures; parser peak RSS was 1,657,896 KiB. The complete production
artifact must pass the same exhaustive payload validator before stage-two
training starts. Separate worst-case preflights covered the largest parsed
context (357,240 residues, 780 chains) and largest compressed deferred input
(52,020,427 bytes, 1,080 chains); their parser peaks were 3,289,472 KiB and
3,654,784 KiB respectively. Long target chains and the 101 raw files above
50 MiB remain a separate later substage.

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
      proteinmpnn_tar_shards_v1/
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
7. write `manifest.json`, `index.jsonl`, `records.jsonl`, `list.csv`, split
   files, and `shards/*.tar`
8. pass residue-mask, assembly-weighting, homology, split-leakage, and oversized
   deferral conformance tests
9. run smoke training against the corrected tar-shard dataset

Run the resumable download/build pipeline:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
scripts/prepare_pdb_2026_tar_dataset.sh
```

Build tar shards directly from an existing raw snapshot:

```bash
scripts/build_pdb_2026_tar_shards.sh --force
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
WORKERS=2 MAX_IN_FLIGHT=2 \
  scripts/build_pdb_2026_tar_shards.sh
```

Build only structures deposited in 2026:

```bash
scripts/build_pdb_2026_tar_shards.sh --min-date 2026-01-01
```

Do not mix predicted structures into this dataset until the experimental-structure
baseline is trained and evaluated.
