# Operational Scripts

These scripts wrap `ProteinMPNN/run_train.sh` with stable local paths and presets.
They are intentionally thin; the main launcher owns validation, manifests, and
runtime checks.

Recommended order:

```bash
scripts/smoke_train.sh
scripts/full_sanity.sh
scripts/full_train_v100.sh
scripts/print_latest_metrics.sh runs/<run-name>
```

For A100:

```bash
scripts/full_train_a100.sh
```

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
