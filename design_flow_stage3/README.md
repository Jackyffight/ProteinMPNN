# Design-flow Stage 3 GPU Worker

This directory is deliberately separate from ProteinMPNN training and its fixed
engineering benchmarks. It consumes a checksum-bound vaccine-design job produced
by `mRNABERT/design-flow`, reuses the already deployed pinned ESMFold2-Fast
runtime, and writes resumable per-candidate structure results.

The worker does not generate candidates and does not modify ProteinMPNN training
runs. A job archive must contain exactly:

- `job-manifest.json`
- `sequences.fasta`

Run on the GPU server:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN/design_flow_stage3/run_stage3_esmfold2.sh \
  /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/transfer/stage3-job.tar.gz
```

The wrapper validates and safely unpacks the archive, loads the existing runtime
from `/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/structure_runtime/esmfold2-fast`,
folds each candidate sequentially, verifies PDB checksums and metrics, then writes
one result archive under `runs/design-flow-stage3/exports/`.

Rerunning the same command resumes valid completed records. A failed record is
retained and skipped until the same command is run with `RETRY_FAILED=1`.
