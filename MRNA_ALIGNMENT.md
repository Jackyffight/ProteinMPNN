# Trainable ProteinMPNN Component and mRNA Alignment

## Local Layout

- ProteinMPNN code: `/data00/home/wangzhi.wit/models/ProteinMPNN/repo`
- Training launcher: `/data00/home/wangzhi.wit/models/ProteinMPNN/run_train.sh`
- Design manifest schema: `/data00/home/wangzhi.wit/models/ProteinMPNN/design_manifest.schema.json`
- Full training data: `/data00/home/wangzhi.wit/models/datasets/proteinmpnn/pdb_2021aug02`
- Smoke-test data: `/data00/home/wangzhi.wit/models/datasets/proteinmpnn/pdb_2021aug02_sample`
- Full tarball: `/data00/home/wangzhi.wit/models/datasets/proteinmpnn/pdb_2021aug02.tar.gz`

The GitHub git clone was unreliable in this environment, so the local repo is a
raw-file training subset, not a complete `.git` checkout. It contains the
training entrypoint and model/data utilities needed to train from scratch.

## Data Provenance

- Source: `https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz`
- Source date from upstream README: PDB biounits, 2021-08-02
- Archive size: `18037128263` bytes
- SHA256: `84d51d0b9224011db8deeab8b83e96f092830aaf6a1f538b1d94b0144f295714`
- Expanded size on this machine: about `62G`
- Structure files after extraction: `869544`
- Split files:
  - `list.csv`
  - `valid_clusters.txt`
  - `test_clusters.txt`

## A100 Environment

Use a separate environment from mRNABERT. ProteinMPNN only needs a small Python
stack, but it should use the CUDA PyTorch build installed for the target host.

```bash
conda create -n proteinmpnn python=3.10 -y
conda activate proteinmpnn
pip install -r /data00/home/wangzhi.wit/models/ProteinMPNN/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If PyTorch is not already installed on the GPU machine, install the matching CUDA
wheel from the official PyTorch instructions for that host.

## Run Commands

Smoke test on the small sample:

```bash
cd /data00/home/wangzhi.wit/models
ProteinMPNN/scripts/smoke_train.sh
```

Short full-data sanity run:

```bash
cd /data00/home/wangzhi.wit/models
ProteinMPNN/scripts/full_sanity.sh
```

V100 full training preset:

```bash
cd /data00/home/wangzhi.wit/models
ProteinMPNN/scripts/full_train_v100.sh
```

A100 full training preset:

```bash
cd /data00/home/wangzhi.wit/models
ProteinMPNN/scripts/full_train_a100.sh
```

Resume:

```bash
cd /data00/home/wangzhi.wit/models/ProteinMPNN
scripts/resume_train.sh runs/proteinmpnn-v48-noise020-v100/model_weights/epoch_last.pt
```

Outputs are written under:

```text
<OUTPUT_DIR>/log.txt
<OUTPUT_DIR>/run_manifest.json
<OUTPUT_DIR>/metrics.jsonl
<OUTPUT_DIR>/eval_results.json
<OUTPUT_DIR>/model_weights/epoch_last.pt
<OUTPUT_DIR>/model_weights/best.pt
<OUTPUT_DIR>/model_weights/epoch<N>_step<M>.pt
```

## Architecture Boundary With mRNABERT

Keep ProteinMPNN and mRNABERT as separate trainable components. The bridge
between them should be an immutable design manifest, not a coupled training
loop.

ProteinMPNN learns:

```text
backbone geometry + chain context -> amino-acid sequence distribution
```

mRNABERT learns:

```text
mRNA/CDS/codon/UTR sequence -> expression, stability, translation, or other regulatory signal
```

The shared contract should be one JSONL row per designed protein sequence:

```json
{
  "design_id": "targetA_axis0_mpnn000001",
  "target_id": "targetA",
  "backbone_id": "targetA_esmfold2_seed0",
  "axis_method": "pca_ca",
  "axis_vector": [0.12, -0.98, 0.15],
  "anchor_residues": [{"chain": "A", "residue": 42}, {"chain": "A", "residue": 117}],
  "fixed_positions": [{"chain": "A", "residue": 23}, {"chain": "A", "residue": 24}],
  "designed_chains": ["A"],
  "mpnn_checkpoint": "/data00/home/wangzhi.wit/models/ProteinMPNN/runs/v48_noise020/model_weights/epoch_last.pt",
  "mpnn_params": {
    "num_neighbors": 48,
    "backbone_noise": 0.2,
    "temperature": 0.1
  },
  "protein_sequence": "M...",
  "mpnn_score": 0.73,
  "fold2_revision": "pinned-esmfold2-revision",
  "fold2_metrics": {
    "plddt_mean": 0.0,
    "ptm": 0.0,
    "iptm": 0.0
  },
  "mrna_objective": {
    "species": "human",
    "utr_policy": "fixed_or_designed",
    "cds_policy": "codon_optimized_candidate"
  }
}
```

Use `ProteinMPNN/design_manifest.schema.json` as the first machine-readable
version of this contract.

The mRNA side should consume this manifest and produce either:

- mRNABERT pretraining text, using the existing `mRNABERT/main.py preprocess`
  path when raw FASTA is available.
- Fine-tuning CSV rows for regression/classification tasks, with stable
  `design_id` linkage back to the ProteinMPNN row.

This keeps the four-step pipeline reproducible:

1. ESMFold2 determines the structural axis and backbone context.
2. ProteinMPNN broadens protein sequence candidates.
3. ESMFold2 refolds and filters candidates as a frozen evaluator.
4. mRNABERT optimizes or scores mRNA-level regulation for surviving protein
   sequences.

The trainable assets are ProteinMPNN-derived design and mRNABERT regulation.
ESMFold2 remains a pinned frozen evaluator unless a separate open-data folding
training stack is built.
