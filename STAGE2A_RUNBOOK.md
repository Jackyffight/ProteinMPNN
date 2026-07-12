# ProteinMPNN Stage2a Runbook

Stage2a continues from the promoted 2026 v1 weights on bounded spatial crops of
oversized assemblies. Training remains single-GPU because the current loop does
not implement DDP.

## 1. Wait For The Local Build

Monitor without attaching:

```bash
tmux capture-pane -pt proteinmpnn-stage2a:1.1 -S -20
```

The build is ready only when this file exists and contains `"status": "ok"`:

```text
/data00/home/wangzhi.wit/models/datasets/proteinmpnn_custom/
  proteinmpnn_pdb_20260708/processed/proteinmpnn_tar_shards_stage2a_v1/validation.json
```

## 2. Sync To The GPU Workspace

Source:

```text
/data00/home/wangzhi.wit/models/datasets/proteinmpnn_custom/
  proteinmpnn_pdb_20260708/processed/proteinmpnn_tar_shards_stage2a_v1/
```

Destination:

```text
/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/
  proteinmpnn_custom/proteinmpnn_pdb_20260708/processed/
  proteinmpnn_tar_shards_stage2a_v1/
```

When the worker is reachable by SSH, use resumable rsync from the data host:

```bash
GPU_HOST=3998835.worker  # replace when the worker hostname changes
rsync -a --partial --info=progress2 \
  /data00/home/wangzhi.wit/models/datasets/proteinmpnn_custom/proteinmpnn_pdb_20260708/processed/proteinmpnn_tar_shards_stage2a_v1/ \
  "tiger@$GPU_HOST:/mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/datasets/proteinmpnn_custom/proteinmpnn_pdb_20260708/processed/proteinmpnn_tar_shards_stage2a_v1/"
```

## 3. Verify On The GPU Host

```bash
cd /mnt/bn/neptune/mlx/users/wangzhi.wit/playground/models/MPNN/ProteinMPNN
git pull
scripts/validate_2026_stage2a_dataset.sh
```

This rechecks shard hashes, every payload and crop interval, split leakage, and
the SHA256-pinned v1 reference dataset.

## 4. Run The Pilot

```bash
DEVICES=0 scripts/run_2026_stage2a_pilot_a100.sh --dry-run
DEVICES=0 scripts/run_2026_stage2a_pilot_a100.sh
scripts/evaluate_2026_stage2a_checkpoints.sh
```

The pilot uses 1,000 examples for one epoch. It initializes weights from the
promoted stage-1 model, resets optimizer state, and uses Noam factor `0.25`
instead of the original `2.0`.

The dual-valid gate requires both:

- lower NLL than stage 1 on complete stage2a validation records
- v1 validation NLL regression no greater than `0.001`

Do not run a test split after the pilot.

## 5. Run Formal Stage2a Training

Only continue if the pilot completes cleanly and its dual-valid gate passes:

```bash
DEVICES=0 scripts/run_2026_stage2a_a100.sh --dry-run
DEVICES=0 scripts/run_2026_stage2a_a100.sh
scripts/evaluate_2026_stage2a_checkpoints.sh
```

The formal launcher starts again from the promoted stage-1 weights and trains
for two epochs over all available stage2a training clusters. It saves every
epoch so the gate can choose epoch 1 even if epoch 2 regresses.

## 6. Use The Test Sets Once

After the formal dual-valid gate selects one checkpoint:

```bash
scripts/evaluate_2026_stage2a_selected_test.sh
```

This evaluates the selected checkpoint and stage-1 baseline on complete stage2a
and fixed v1 test populations. The script refuses to overwrite an existing test
summary.

If no checkpoint passes either dual gate, stop. The next experiment is a
stage2a plus v1 replay mixture, not more epochs on stage2a alone.
