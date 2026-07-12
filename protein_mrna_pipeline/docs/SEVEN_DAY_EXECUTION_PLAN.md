# Seven-Day Engineering Execution Plan

## Scope Decision

The current team has engineering ownership but no research owner supplying a
fusion target, domain boundaries, mutable residues, linker policy, or biological
objective. Those inputs must not be guessed by an engineer or inferred by the
pipeline.

This plan therefore separates two lanes:

- **Lane A, active:** benchmark the infrastructure on fixed native PDB sequences.
  This measures structure-model throughput, queue recovery, provenance, storage,
  and generic ProteinMPNN behavior. It performs no target-specific optimization.
- **Lane B, blocked:** run a fusion-protein-to-CDS design. This begins only after a
  research owner supplies an approved target package.

The promoted ProteinMPNN Stage2a checkpoint remains frozen. Another training run
is not the default response to unused GPU capacity.

## Engineering Inputs

Lane A uses the validated post-2021 ProteinMPNN v1 dataset already present on the
GPU server. The checked-in generator selects records with these fixed rules:

- source split: `valid`; `test` is never sampled;
- default count: 40;
- sequence length: 50-800 residues;
- deterministic seed: 42;
- at most one sequence per cluster;
- exact-sequence deduplication;
- canonical amino acids only;
- bounded length stratification.

The 40 `valid` records are calibration/evaluation inputs. They must not be used
to train a surrogate, tune structure-model parameters, or select another
ProteinMPNN checkpoint.

Generate it with:

```bash
scripts/prepare_2026_structure_benchmark.sh --dry-run
scripts/prepare_2026_structure_benchmark.sh
```

This command reads metadata only. It does not open tar shards or start a GPU.

## Resource Policy

- Do not allocate all four A100s before measuring one-GPU behavior.
- Keep GPU 0 for the first adapter smoke and 40-record benchmark.
- Use GPU 0-3 as independent workers only after lease recovery and output
  checksums pass on the actual mounted filesystem.
- Preserve at least 20% of the allocation for retries, debugging, recomputation,
  and archival.
- Do not manufacture a long run merely to consume a seven-day allocation.

## Day 0: Freeze The Benchmark And Runtime

1. Generate and validate the 40-record benchmark suite.
2. Record its benchmark ID, suite SHA256, source manifest hashes, and FASTA hash.
3. Inventory the structure predictor actually available on the GPU host.
4. Pin its code revision, weight checksum, environment identity, and inference
   parameters before writing the adapter.
5. Preflight output storage and the archive destination.

Gate 0: proceed without a research target, but not without a validated benchmark
suite and a reproducibly identifiable structure-model runtime.

## Day 1: One-GPU Structure Adapter

1. Implement the structure adapter behind `ToolAdapter` for the pinned runtime.
2. Fold one record from each length bucket on GPU 0.
3. Record wall time, peak GPU memory, output bytes, confidence fields, and errors.
4. Kill one attempt intentionally and verify lease expiry, retry, and attempt
   history.
5. Run all 40 records only after the four-record smoke succeeds.

Gate 1: every input has one terminal or explicitly retryable result, artifacts
match their declared SHA256 and size, and measured cost replaces guessed capacity.

## Days 2-3: Four-Worker Engineering Run

1. Partition the same immutable suite across four independent workers.
2. Verify that no work item is completed twice and no failed item disappears.
3. Compare single-worker and four-worker throughput and storage growth.
4. Where experimental coordinates are available, compute generic structural
   agreement metrics without introducing target-specific pass thresholds.
5. Run a small paired official-vs-Stage2a ProteinMPNN inference benchmark on fixed
   PDB backbones only after the structure workflow is stable.

Gate 2: scaling is accepted only if aggregate throughput improves, queue behavior
remains correct, and outputs stay provenance-complete.

## Days 4-5: Conditional Generic Label Expansion

Expand beyond 40 records only when Gate 2 shows that the labels are useful and the
storage budget is understood. A larger training suite must come from `train`, be a
new versioned artifact, retain cluster-aware separation, and keep both the fixed
40-record `valid` benchmark and held-out `test` records out of training.

Possible engineering outputs are native-sequence fold confidence, experimental
structure agreement, ProteinMPNN sequence recovery, bounded redesign/refold
comparisons, runtime, memory, and failure labels. They are generic model and
systems benchmarks, not fusion-target evidence.

Fit a Ridge/GBDT surrogate only on a train-derived suite with enough independent
clusters. Evaluate on untouched clusters or families, never by random candidate
row. A negative result is acceptable.

## Day 6: Translation-Preservation Plumbing

1. Generate a small synonymous CDS set for selected benchmark proteins.
2. Require exact translation back to the fixed source protein.
3. Compute GC, codon-frequency, repeat, and forbidden-motif rule features.
4. Exercise mRNABERT adapters only as separate recorded scores.
5. Make no claim that a benchmark CDS is a preferred biological construct.

This step validates contracts and software integration; it is not research target
selection or wet-lab prioritization.

## Day 7: Recompute And Archive

1. Stop launching jobs that cannot finish at least 12 hours before expiry.
2. Recompute a fixed subset to measure determinism.
3. Export queue tables, attempt history, benchmark suite, artifacts, checksums,
   runtime, memory, and failure summaries.
4. Verify the archive from a second location.
5. Write down unsupported model environments and negative results.

## Lane A Exit Criteria

- A deterministic, test-excluding benchmark suite is archived.
- Fold cost, latency, peak memory, failure rate, and bytes per record are measured.
- The queue survives an interrupted worker and preserves attempt history.
- Four-worker scaling is measured rather than assumed.
- Official and Stage2a checkpoints have a bounded paired engineering comparison,
  if the structure adapter is available.
- Every emitted CDS, if that plumbing is exercised, translates exactly to its
  source protein.
- No proxy metric is described as wet-lab efficacy evidence.

## Activating Lane B

Lane B is not an engineering TODO. It activates only when a named research owner
provides and approves:

- domain sequences and boundaries;
- immutable and explicitly mutable residues;
- allowed domain order and linker policy;
- oligomeric state, length, expression, and construct constraints;
- prohibited modifications and safety review.

Only then should the pipeline create fusion candidates, apply constrained
ProteinMPNN design, rank target-specific refolds, or produce a design shortlist.

## Conditional Training

Do not extend Stage2a automatically. The opened held-out test set cannot be reused
to select another release. A future replay mixture or full retraining campaign
requires a new predeclared holdout and a measured failure on the engineering or
research task that the training is intended to fix.
