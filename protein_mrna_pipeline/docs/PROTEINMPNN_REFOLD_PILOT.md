# ProteinMPNN Paired Refold Pilot

## Purpose

This bounded engineering pilot compares the official ProteinMPNN `v_48_020`
checkpoint with the promoted 2026 Stage2a checkpoint. It asks whether sequences
sampled from the two models preserve backbone geometry when evaluated by the
same pinned ESMFold2-Fast oracle.

This is not another checkpoint-selection gate. The four source backbones come
from the fixed `valid` benchmark, and the test split remains unused. The result
may justify or reject a larger engineering benchmark, but it must not retroactively
select model weights or tune the declared sampling parameters.

## Fixed Inputs

| Input | Identity |
| --- | --- |
| Benchmark | `pdb-valid-7136a4ecae1956027aa6` |
| Official checkpoint | SHA256 `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd` |
| Stage2a checkpoint | SHA256 `08fc2549004d0e8a8b1ac1983dd4e94772f15445732926d8f7e677a4464ba6f7` |
| ProteinMPNN temperature | `0.1` |
| Paired sampling seeds | `11`, `23`, `42`, `67` |
| ESMFold2 parameters | loops 3, sampling steps 50, one diffusion sample, seed 42 |

Four distinct backbones are selected deterministically from the completed
native-agreement records:

1. lowest C-alpha lDDT;
2. lowest resolved-position TM-score among records with at least 95% native
   C-alpha coverage;
3. longest sequence not already selected;
4. highest C-alpha lDDT control not already selected.

For ProteinMPNN, the complete payload assembly is retained as structural and
sequence context. Only the target chain is designed. Target positions without a
complete N/CA/C/O backbone remain fixed to the native amino acid. This preserves
the v1 payload's exact polymer positions and allows Stage2a to use assembly
context learned during continuation.

The design matrix is fixed at:

```text
4 backbones x 2 checkpoints x 4 paired seeds = 32 target sequences
```

## Run

From the ProteinMPNN repository root on the GPU server:

```bash
git pull
scripts/run_proteinmpnn_refold_pilot.sh --dry-run
scripts/run_proteinmpnn_refold_pilot.sh
```

The default `all` mode runs three identity-bound phases:

1. `generate`: load each ProteinMPNN checkpoint once and generate 16 sequences;
2. `refold`: load ESMFold2-Fast once and fold all 32 target sequences;
3. `evaluate`: release the GPU and compute dual-reference structural metrics on
   one CPU thread.

Each phase can be resumed or run separately:

```bash
scripts/run_proteinmpnn_refold_pilot.sh generate
scripts/run_proteinmpnn_refold_pilot.sh refold
scripts/run_proteinmpnn_refold_pilot.sh evaluate
```

Explicit failed refolds or evaluations remain terminal until retried:

```bash
RETRY_FAILED=1 scripts/run_proteinmpnn_refold_pilot.sh all
```

The pilot exposes exactly one GPU. ProteinMPNN generation should finish quickly;
the 32 sequential ESMFold2 refolds are expected to dominate runtime. Based on
the fixed native benchmark, the expected wall time is roughly 10-25 minutes.

## Outputs

Default root:

```text
runs/benchmarks/
  proteinmpnn-refold-pilot-pdb-valid-7136a4ecae1956027aa6-t010-s4/
```

Important files:

```text
pilot-manifest.json
selected-backbones.json
designs.jsonl
generation-summary.json
refolds/esmfold2-fast-l3-s50-seed42/
  refold-manifest.json
  summary.json
  records/<design-id>/prediction.pdb
  records/<design-id>/result.json
evaluations/dual-reference-v1/
  evaluation-manifest.json
  summary.json
  records.jsonl
  records/<design-id>.json
```

Print the compact comparison again with:

```bash
scripts/report_proteinmpnn_refold_pilot.sh
```

## Metrics

The evaluation reports ProteinMPNN self-scored sampled NLL, sequence recovery,
mutation fraction, and ESMFold2 pLDDT/pTM. Geometry is compared against two
references:

- **experimental native:** lDDT, resolved/full TM-score, RMSD, and delta from the
  native-sequence ESMFold2 baseline on the same experimental chain;
- **native-sequence ESMFold2 prediction:** lDDT, TM-score, and RMSD between the
  designed-sequence refold and the already archived native-sequence prediction.

The second reference controls part of the shared structure-oracle bias. It does
not turn ESMFold2 agreement into experimental evidence. Sequence recovery is
descriptive and is not itself a quality objective.

The summary contains 16 same-backbone, same-seed Stage2a-minus-official pairs,
per-model distributions, per-backbone distributions, mean paired deltas, and
direction-aware structural win counts. Higher is better for lDDT and TM-score;
lower is better for RMSD. Sampled NLL is scored by the model that generated
each sequence, so it remains descriptive and is not counted as a cross-model
win. No automatic winner or release status is assigned.

The experimental chains may contain assembly-stabilized conformations, while
the ESMFold2 refolds are target-only. Possible overlap with ESMFold2 training
data has not been audited. These constraints apply even when all execution
records pass.
