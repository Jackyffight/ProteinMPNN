# Operational Scripts

These scripts wrap `ProteinMPNN/run_train.sh` with stable local paths and presets.
They are intentionally thin; the main launcher owns validation, manifests, and
runtime checks.

Recommended order:

```bash
scripts/run_baseline_from_scratch.sh --profile v100 --devices 0 --no-full
scripts/download_dataset_parts.sh --extract
scripts/validate_dataset.sh
scripts/smoke_train.sh
scripts/full_sanity.sh
scripts/full_train_v100.sh
scripts/benchmark_throughput.sh smoke
scripts/benchmark_throughput.sh quick
scripts/print_throughput_benchmark.sh runs/benchmarks/<benchmark-dir>
scripts/print_latest_metrics.sh runs/<run-name>
```

Latest PDB owned dataset track:

```bash
scripts/init_dataset_version.sh
scripts/sync_latest_pdb_assemblies.sh --dry-run
scripts/inspect_dataset_version.sh ../datasets/proteinmpnn_custom/proteinmpnn_pdb_latest_YYYYMMDD
```

For A100:

```bash
scripts/full_train_a100.sh
```

Current unmeasured presets mirror the mRNABERT launcher style: use `v100`
for conservative token budgets and `a100` for larger token budgets. Run the
throughput benchmark before long training on a new host, then keep the measured
winner in the run name and manifest.

Resume:

```bash
scripts/resume_train.sh runs/<run-name>/model_weights/epoch_last.pt
```

Expected training outputs:

```text
<run>/run_manifest.json
<run>/metrics.jsonl
<run>/eval_results.json
<run>/log.txt
<run>/model_weights/epoch_last.pt
<run>/model_weights/best.pt
```
