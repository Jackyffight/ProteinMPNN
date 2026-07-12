# ProteinMPNN 2026 Stage2a Model Card

## Release Status

This is the promoted final checkpoint for the current 2026 continuation track.
No additional training is required for this release.

| Field | Value |
| --- | --- |
| Model ID | `proteinmpnn-2026-stage2a` |
| Model type | Full-backbone ProteinMPNN sequence designer |
| Stable artifact | `runs/promoted/proteinmpnn-2026-stage2a/model.pt` |
| SHA256 | `08fc2549004d0e8a8b1ac1983dd4e94772f15445732926d8f7e677a4464ba6f7` |
| Selected checkpoint | `epoch2_step688` |
| Source run | `proteinmpnn-2026-stage2a-v48-a100-20260713004156` |
| Training/evaluation code | Git commit `5fbf416` |
| Promotion code | Git commit `3653357` |
| Selection rule | Lowest stage2a validation NLL subject to the v1 NLL regression gate |
| Test status | Passed on both stage2a and v1 held-out populations |

The absolute path used on the training workspace was:

```text
/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN/
  runs/promoted/proteinmpnn-2026-stage2a/model.pt
```

## What The Release Contains

Keep the complete promoted directory, not only the checkpoint:

| File | Purpose |
| --- | --- |
| `model.pt` | Exact selected PyTorch training checkpoint, copied without modifying its bytes |
| `promotion.json` | Model ID, source run, selected checkpoint metadata, SHA256, gate deltas, and intended-use policy |
| `dual-valid-summary.json` | Complete paired validation evidence used to select the checkpoint |
| `dual-test-summary.json` | One-shot paired held-out test evidence |

`model.pt` is a full training checkpoint. It contains:

- `model_state_dict`: the weights used for inference
- `epoch`, `step`, `num_edges`, and `noise_level`
- `config`: architecture and training-time model configuration
- `optimizer_schedule` and `optimizer_state_dict`
- `best_validation_loss` and the sampled epoch `metrics`

The optimizer state is retained for provenance because promotion copies the
selected checkpoint exactly. It must not be restored for a new training stage.
The promotion manifest explicitly sets `restore_optimizer` to `false`.
`protein_mpnn_run.py` uses only the model weights and compatible model metadata
for inference.

The release does not embed the source code, Python environment, datasets, input
structures, or raw wwPDB files. Archive those separately when exact operational
reproduction is required.

## What The Model Does

ProteinMPNN designs or scores amino-acid sequences conditioned on protein
backbone geometry. This checkpoint is intended for full-backbone inputs with N,
CA, C, and O coordinates. It can design selected chains while holding other
chains or residue positions fixed, and supports the constraints exposed by
`repo/protein_mpnn_run.py`.

It is not a structure predictor, binding-affinity predictor, function
classifier, or mRNA model. In the larger pipeline its output must still be
refolded and evaluated before downstream mRNA design.

## Architecture

The architecture remains compatible with the official full-backbone
`v_48_020` model:

| Parameter | Value |
| --- | --- |
| Amino-acid vocabulary | 21 symbols: `ACDEFGHIKLMNPQRSTVWYX` |
| Hidden, node, and edge dimensions | 128 |
| Encoder layers | 3 |
| Decoder layers | 3 |
| Spatial neighbors | 48 |
| Training dropout | 0.1 |
| Training backbone noise | 0.2 Angstrom |
| Maximum context evaluated during continuation | 2,000 residues |

Inference normally uses `--backbone_noise 0.0`; the `noise_level=0.2` checkpoint
field records the augmentation used during training rather than requiring noisy
inference.

## Weight Lineage

This is a conservative continuation model, not a model trained from random
initialization.

1. Official ProteinMPNN `v_48_020.pt` provided the initial weights. The pinned
   upstream file has SHA256
   `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd`.
2. Stage 1 continued those weights on the validated post-2021 v1 dataset. The
   fixed validation gate selected epoch 1, step 494, SHA256
   `95de1d508d0778e3fb340486f2631ef5abd4d0ddc5898928790c15427093b95a`.
3. Stage2a started a new optimizer from the promoted stage-1 weights and trained
   on spatial crops of oversized assemblies. The pilot was used only as a gate;
   its weights are not part of this lineage.
4. The formal two-epoch Stage2a run selected epoch 2, step 688. Promotion copied
   that checkpoint to the stable artifact and verified the final SHA256 above.

### Stage 1 Data And Training

- Snapshot: `proteinmpnn_pdb_20260708`
- Deposition interval: 2021-08-03 through 2026-07-08
- Records: 46,619
- Unique split clusters: 7,743, including 7,589 train, 77 valid, and 77 test
- Context limit: 2,000 residues
- Formal schedule: 20 epochs, one sampled target per available train cluster on
  each two-epoch data reload, 10,000-token batches, seed 42
- Initialization: official `v_48_020` weights with a new optimizer
- Selected result: epoch 1, step 494; later epochs were retained but did not win
  the complete fixed validation ranking

Stage 1 improved the complete 461-record v1 test population over the official
checkpoint: NLL changed by `-0.006751`, perplexity by `-0.032250`, and accuracy
by `+0.001963`.

### Stage2a Data And Training

- Input: 10,166 oversized assemblies deferred by v1
- Published crops: 9,617 records, split into 9,585 train, 19 valid, and 13 test
- Unique split clusters: 1,734, including 1,720 train, 11 valid, and 3 test
- Crop policy: preserve the complete target and add nearest complete chains or
  one contiguous nearest-residue window under 2,000 residues and 62 chains
- Formal schedule: 2 epochs, one sampled target per train cluster per epoch,
  10,000-token batches, seed 42, mixed precision, one A100
- Optimizer: new Adam/Noam state, factor 0.25, 4,000 warmup steps, gradient norm 1.0
- Runtime result: 344 optimizer steps per epoch and 688 total steps

The pilot used 1,000 examples for one epoch and passed the dual validation gate.
The formal run then started again from the promoted Stage-1 checkpoint; it did
not resume from the pilot.

## Evaluation Evidence

All model comparisons below are paired: baseline and selected checkpoints saw
the same complete record population with seed 42. Checkpoint selection used
validation data only. The test populations were evaluated once after selection.

### Complete Validation Gate

The baseline is the promoted Stage-1 checkpoint.

| Dataset | Records | Baseline NLL | Selected NLL | Delta NLL | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Stage2a valid | 19 | 1.773673 | 1.759188 | -0.014485 | Improved |
| v1 valid | 426 | 1.544257 | 1.538968 | -0.005289 | Improved |

The v1 gate allowed at most `+0.001` NLL regression. The selected checkpoint
improved v1 instead, so both validation requirements passed.

### One-Shot Held-Out Test

| Dataset | Records | Metric | Baseline | Selected | Delta |
| --- | ---: | --- | ---: | ---: | ---: |
| Stage2a test | 13 | NLL | 1.558879 | 1.549649 | -0.009229 |
| Stage2a test | 13 | Perplexity | 4.753489 | 4.709819 | -0.043670 |
| Stage2a test | 13 | Accuracy | 0.505140 | 0.511308 | +0.006168 |
| v1 test | 461 | NLL | 1.560491 | 1.556187 | -0.004305 |
| v1 test | 461 | Perplexity | 4.761160 | 4.740709 | -0.020451 |
| v1 test | 461 | Accuracy | 0.513928 | 0.515329 | +0.001401 |

The final test status is `passed`. The positive result on 13 Stage2a test
records is directionally useful but statistically small; it must not be treated
as a broad long-complex benchmark by itself.

## Intended Use

Use this release for:

- internal ProteinMPNN sequence generation and sequence scoring
- comparison with the official and promoted Stage-1 checkpoints
- the ProteinMPNN stage of the ESMFold2 -> ProteinMPNN -> ESMFold2 -> mRNABERT
  pipeline
- initialization of a separately approved future experiment, without restoring
  this checkpoint's optimizer

Do not use the current evidence as proof of protein function, binding, folding,
expression, safety, or wet-lab success. Do not use it for clinical decisions.

## Inference

The promoted filename is `model.pt`, so use `--model_name model`:

```bash
MODEL_DIR=/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN/runs/promoted/proteinmpnn-2026-stage2a

python repo/protein_mpnn_run.py \
  --pdb_path /path/to/input.pdb \
  --pdb_path_chains "A" \
  --path_to_model_weights "$MODEL_DIR" \
  --model_name model \
  --out_folder /path/to/output \
  --num_seq_per_target 8 \
  --sampling_temp "0.1" \
  --backbone_noise 0.0 \
  --seed 42
```

Remove `--pdb_path_chains` to design all parsed chains. Preserve the seed,
sampling temperature, constraints, input structure checksum, code commit, and
model SHA256 with every reported design run.

## Packaging And Integrity

Create the transfer archive from the parent directory so the archive cannot
include itself:

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN/runs/promoted
tar -czf proteinmpnn-2026-stage2a-08fc2549.tar.gz proteinmpnn-2026-stage2a
sha256sum proteinmpnn-2026-stage2a-08fc2549.tar.gz \
  > proteinmpnn-2026-stage2a-08fc2549.tar.gz.sha256
tar -tzf proteinmpnn-2026-stage2a-08fc2549.tar.gz
```

After copying, verify both the archive sidecar and the model hash:

```bash
sha256sum -c proteinmpnn-2026-stage2a-08fc2549.tar.gz.sha256
tar -xzf proteinmpnn-2026-stage2a-08fc2549.tar.gz
sha256sum proteinmpnn-2026-stage2a/model.pt
```

The final command must print the release SHA256 recorded at the top of this
document.

## Known Limitations

- This is continued training from public weights, not a from-scratch retraining
  on a complete 2026 corpus.
- Stage2a test evidence contains only 13 records and one training seed.
- Training and checkpoint selection optimize native-sequence NLL/recovery, not
  refold quality, interface recovery, function, stability, or expression.
- Context was bounded to 2,000 residues. Stage2a crops can omit distant chains
  even when they belong to the original assembly.
- 532 complete target chains longer than 2,000 residues remain untrained.
- 101 compressed mmCIF inputs above 50 MiB were not parsed by v1 or Stage2a.
- 17 ambiguous records linking a v1 train cluster to a v1 valid cluster were
  quarantined and excluded.
- The model has not been qualified on predicted structures, nucleic-acid
  complexes, noncanonical residues, ligands, membrane-specific benchmarks, or
  structures deposited after the pinned snapshot.
- The current training loop is single-GPU and materializes sampled structures
  before each epoch. That affects scalability, not the promoted weights.
- PyTorch AMP and activation-checkpoint calls emit deprecation/no-grad warnings
  in the current environment. They did not invalidate the completed paired
  evaluations, but should be cleaned up before the next training campaign.

## Remaining Work

The current release is complete. The remaining items are release engineering
and broader product validation, not automatic additional epochs.

### P0: Preserve And Reproduce

1. Copy the promoted archive and checksum sidecar off the GPU workspace.
2. Register the archive SHA256, byte size, storage URI, repository commit, and
   environment lockfile in the model registry.
3. Keep the promoted directory and source run until two independently verified
   backups exist.
4. Run a fixed inference smoke suite covering a monomer, multimer, fixed-chain
   design, missing-coordinate input, and a near-2,000-residue crop. Assert finite
   scores, deterministic seeded output, and strict checkpoint loading.
5. Produce an inference-only derivative without optimizer state if transfer size
   matters. Give it a new SHA256 and retain this full checkpoint as the source of
   truth.

### P1: Validate Design Quality

1. Compare official, Stage 1, and Stage2a models on the same frozen design set
   across several sampling temperatures and seeds.
2. Refold generated sequences with the chosen structure predictor and report
   paired structure recovery, confidence, backbone deviation, interface quality,
   clashes, sequence diversity, and failure rates.
3. Add domain slices for monomers, homomers, heteromers, interfaces, missing
   residues, long contexts, and low-homology targets.
4. Exercise the full downstream pipeline through mRNABERT and preserve a design
   manifest linking structure, sequence, model, constraints, and random seed.
5. Add structure-level confidence intervals. Residues from the same structure
   must not be treated as independent samples.

### P2: Increase Evidence And Coverage

1. Train independent seeds before making a strong claim about the small Stage2a
   improvement; changing only evaluation seeds is not a substitute.
2. Build a larger external long-complex benchmark with a cutoff that prevents
   overlap with the 2026-07-08 training snapshot.
3. Define a separate policy for the 532 targets above 2,000 residues and the 101
   raw files above 50 MiB. Do not silently truncate complete targets.
4. Review the 17-record split-conflict component only if stronger homology or
   provenance evidence can resolve it without leakage.
5. Establish a versioned refresh procedure for structures deposited after the
   pinned snapshot.

### P3: Engineering Before Another Training Campaign

1. Add progress and memory telemetry to full dataset validation and epoch
   prefetch so long silent phases are observable.
2. Replace deprecated AMP calls and make activation-checkpoint behavior explicit
   for training versus evaluation.
3. Stream or chunk sampled structures to reduce host-memory duplication.
4. Add DDP and distributed metric reduction only if future full retraining or
   multi-seed throughput justifies the complexity.
5. Review upstream licensing and attribution requirements before any external
   redistribution of code or derived weights.

Further training should be triggered by a measured failure. If broader tests
show forgetting on the original distribution, the next experiment is a pinned
v1 plus Stage2a replay mixture with a new validation protocol. Simply extending
the current two-epoch Stage2a run is not the approved next step.

## Repository Evidence

- Dataset provenance and exclusions: `DATASET_VERSIONS.md`
- Stage2a procedure: `STAGE2A_RUNBOOK.md`
- Promotion gate: `repo/training/promote_stage2a_checkpoint.py`
- Training launcher: `scripts/run_2026_stage2a_a100.sh`
- Validation gate: `scripts/evaluate_2026_stage2a_checkpoints.sh`
- One-shot test gate: `scripts/evaluate_2026_stage2a_selected_test.sh`
- Promotion entry point: `scripts/promote_2026_stage2a.sh`
