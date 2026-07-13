# ESMFold2-Fast Engineering Runbook

## Decision

The structure benchmark uses the released Biohub **ESMFold2-Fast** model, not
the original Meta ESMFold model. Fast is the correct first runtime for the fixed
40-record single-chain benchmark because it is the inference-optimized,
single-sequence ESMFold2 variant. The full `biohub/ESMFold2` model adds optional
MSA and multi-biomolecule capabilities that this benchmark does not exercise.

The runtime is immutable at these identities:

| Component | Identity |
| --- | --- |
| Biohub transformers source | `ef32577f55da19a4989cd7b22e004dc43a4998cb` |
| `biohub/ESMFold2-Fast` | `b28d8ace5e05e61e5bec1e6820cfd3e221819d12` |
| `biohub/ESMC-6B` | `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a` |
| Inference | loops 3, sampling steps 50, diffusion samples 1, chunk size 64 |
| Benchmark seed | 42 |

The full `Biohub/esm` source package currently requires Python 3.12. This
single-chain runner intentionally does not install that package: it calls the
pinned Biohub `transformers` implementation directly, whose pinned source
supports Python 3.9 and newer. The project runner requires Python 3.10 and newer,
so the GPU server's existing Python 3.11 + CUDA PyTorch environment is valid.

Official sources:

- <https://huggingface.co/biohub/ESMFold2-Fast>
- <https://huggingface.co/biohub/ESMC-6B>
- <https://github.com/Biohub/transformers>

## Model Contents And Storage

`ESMFold2-Fast` is not a self-contained 721 MiB model. Its folding trunk loads
the separate `ESMC-6B` protein language model. The seven checksum-pinned weight
files total `26,163,565,812` bytes, or about 24.4 GiB:

- ESMFold2-Fast folding and confidence weights: 755,416,924 bytes;
- ESMC-6B language-model weights: six shards totaling 25,408,148,888 bytes.

The setup preflight requires the missing weight bytes plus 6 GiB of free-space
headroom. It creates an isolated venv under the structure runtime but reuses the
server's CUDA-enabled PyTorch through `--system-site-packages`; it does not
modify the ProteinMPNN training environment.

Default layout:

```text
$MPNN_WORKSPACE/structure_runtime/esmfold2-fast/
  venv/
  models/
    ESMFold2-Fast/
    ESMC-6B/
  hf-home/
  runtime-manifest.json
```

## Install

From the ProteinMPNN repository root on the GPU server:

```bash
scripts/setup_esmfold2_fast_runtime.sh --dry-run
scripts/setup_esmfold2_fast_runtime.sh
```

The downloads are resumable. Rerun the same setup command after a network or
shell interruption. Setup is complete only after all seven weight SHA256 values
pass and `runtime-manifest.json` is written.

## Four-Record Smoke

The smoke selects the longest record in each of the four existing length bins.
It loads the model once and folds the four records sequentially on one visible
GPU:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_esmfold2_fast.sh smoke
```

Default output:

```text
runs/benchmarks/
  esmfold2-fast-pdb-valid-7136a4ecae1956027aa6-smoke-l3-s50-seed42/
```

Check progress from another shell:

```bash
watch -n 2 nvidia-smi
cat runs/benchmarks/esmfold2-fast-pdb-valid-7136a4ecae1956027aa6-smoke-l3-s50-seed42/summary.json
```

The same launch command resumes the same output directory. A completed record
is skipped only when its result metadata, PDB byte size, and PDB SHA256 still
match. An interrupted in-flight record has no committed result and is rerun.
Explicit failed records remain visible; retry them with:

```bash
RETRY_FAILED=1 CUDA_VISIBLE_DEVICES=0 scripts/run_esmfold2_fast.sh smoke
```

## Full 40-Record Run

The full command refuses to start until the four-record summary is `passed`:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_esmfold2_fast.sh full
```

This remains deliberately single-GPU and sequential. It prevents accidental
batching of several long sequences into the same 80 GiB GPU and gives every
record an atomic recovery boundary. Four-worker scaling is a later benchmark;
it should partition records across four independent processes after this
one-GPU run measures actual memory and latency.

Each run contains:

```text
run-manifest.json
summary.json
records/<record-id>/prediction.pdb
records/<record-id>/result.json
```

The manifest pins the benchmark suite SHA256, model revisions, all weight
checksums, runtime identity, inference parameters, and record IDs. Each result
stores wall time, peak allocated/reserved GPU memory, pLDDT, pTM, PDB size, and
PDB SHA256. These are engineering measurements and computational predictions,
not experimental evidence.

The fixed full run completed on 2026-07-13 with 40/40 records, no failures,
857.57 seconds of record runtime, 21.44 mean seconds per record, and a peak of
23,187,596,800 allocated GPU bytes. Its run identity is
`3fdbe3d5df4233ce6debf5395d095e527032b9cc91ce42186f9936cbd361c1bc`.

## Native-Structure Agreement

The next step does not refold sequences and does not use a GPU. It compares the
40 predicted PDB files with the experimental target-chain coordinates already
stored in the checksum-bound v1 ProteinMPNN tar shards.

Install the separate pinned metrics environment, then run the resumable
evaluation:

```bash
scripts/setup_structure_metrics_runtime.sh --dry-run
scripts/setup_structure_metrics_runtime.sh
scripts/evaluate_esmfold2_native_agreement.sh
```

The metrics runtime pins Biotite 1.6.0, Biotraj 1.2.2, NumPy 2.4.6, and SciPy
1.17.1. It reuses the base interpreter's Torch only to read the existing payload
files. The launcher hides all GPUs and limits BLAS/OpenMP to one thread.

Default output:

```text
<full-esmfold2-run>/evaluations/native-structure-agreement-v1/
  evaluation-manifest.json
  summary.json
  records.jsonl
  records/<record-id>.json
```

The evaluator requires exact sequence-position correspondence. It rejects a
record if the benchmark sequence, tar index length, payload target, predicted
PDB sequence, residue numbering, run identity, or PDB SHA256 differs. It does
not perform a free sequence or chain alignment.

Reported metrics are:

- C-alpha lDDT with a 15 Angstrom inclusion radius and 0.5/1/2/4 Angstrom bins;
- C-alpha RMSD after one global Kabsch fit over experimentally resolved C-alpha
  positions;
- C-alpha TM-score after that same fit, normalized once by resolved native
  positions and once by full sequence length;
- native C-alpha and complete-backbone coverage;
- pLDDT-to-lDDT and pTM-to-full-length-TM Pearson correlations;
- overall and per-length-bin count, mean, median, minimum, and maximum.

The full-length TM-score can be capped by native coordinate coverage because an
unresolved native position contributes no matched C-alpha. Read it together
with the resolved-position score and coverage. No metric threshold is a release
gate here, and these 40 valid records must not be used to tune inference
parameters or select another checkpoint.

This is also not a strict ESMFold2 generalization benchmark: overlap between the
40 PDB records and the structure model's training corpus has not been audited.
The predictions are single-chain, while each experimental target chain was
extracted from a biological assembly and may include interface-stabilized
conformations. Those effects must remain visible when interpreting outliers.

Completed records are reused only when their evaluation identity and sequence
hash still match. Explicit failures remain visible. Retry them after correcting
the recorded cause with:

```bash
RETRY_FAILED=1 scripts/evaluate_esmfold2_native_agreement.sh
```
