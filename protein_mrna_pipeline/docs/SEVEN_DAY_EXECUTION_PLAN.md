# Seven-Day Execution Plan

## Objective

Use the GPU window to produce a versioned candidate-and-structure-label dataset
and one auditable target-to-CDS demonstration. The promoted ProteinMPNN Stage2a
checkpoint is frozen for this sprint; another checkpoint is not the primary
deliverable.

Assumption: four A100 GPUs are available. If fewer GPUs remain, preserve the gate
order and reduce candidate count rather than skipping provenance or hard checks.

## Resource Policy

- 70-75% of GPU-hours: structure fold/refold labels.
- At most 5%: paired ProteinMPNN inference comparisons.
- At most 5%: bounded mRNABERT/Evo 2 validation.
- 5-10%: surrogate work and high-confidence recomputation.
- At least 10%: recovery, failed shards, and final archival.

GPU 0-2 should become independent structure workers after the throughput gate.
GPU 3 handles bounded integration jobs and then joins the structure queue.

## Day 0: Contracts And Frozen Assets

1. Register official, Stage-1, and Stage2a ProteinMPNN model hashes.
2. Materialize at least one reviewed target package; three targets are preferred
   if a target-held-out surrogate result is expected.
3. Pin the structure oracle code revision, weight checksum, environment, and
   inference parameters.
4. Validate this project's schemas, run initialization, queue recovery, and
   export path.
5. Preflight storage and the continuous archive destination.

Gate 0: no GPU scale-up without an approved target, immutable/mutable residue
policy, stable IDs, and pinned expert identities.

## Day 1: Structure Adapter And Throughput

1. Implement the first structure adapter behind `ToolAdapter`.
2. Run 30-100 fold/refold smoke cases spanning representative sequence lengths.
3. Record wall time, peak GPU memory, failure rate, output bytes, and restart
   behavior.
4. Verify that an interrupted lease is reclaimed without losing attempt history.

Gate 1: the queue is resumable, every output has provenance, and measured
seconds-per-candidate determines the scale budget. Do not extrapolate candidate
count before this measurement.

## Day 2: Small Closed Loop

1. Enumerate 100-500 domain-order and linker backbone candidates on CPU.
2. Fold the initial architectures and retain a throughput-calibrated subset.
3. Run paired constrained design with official `v_48_020`, promoted Stage 1, and
   promoted Stage2a using identical backbones, constraints, seeds, and
   temperatures.
4. Reject every immutable-residue violation before refold.
5. Refold candidates and compute global, per-domain, junction, interface, clash,
   and compactness metrics where applicable.

Gate 2: immutable violations are zero, outputs are restartable, and structure
metrics differentiate candidates. The default ProteinMPNN expert is selected by
fusion-task refold outcomes, not validation NLL alone.

## Days 3-4: First Label Wave

1. Run independent structure workers over atomic candidate shards.
2. Retain scalar metrics and compressed structures for every candidate.
3. Retain large PAE arrays, logits, and embeddings only for top, uncertain, or
   diagnostic candidates until storage cost is measured.
4. Archive completed immutable shards continuously.
5. Record failures and bounded retries; never silently drop candidates.

The candidate budget is:

```text
floor(available worker-seconds / measured mean seconds per candidate)
```

Apply an explicit recovery reserve before creating the queue.

## Day 5: Surrogate And Active Selection

1. Fit honest Ridge/GBDT feature baselines before a neural surrogate.
2. Split by target or protein family, never by random candidate row.
3. Compare refold-pass enrichment against diverse random selection.
4. Send both high-scoring and high-uncertainty candidates into a second oracle
   wave.

Gate 3: retain a surrogate only if it enriches expensive passes on unseen
targets. With only one target, report an in-target heuristic and do not claim
generalization.

## Day 6: mRNA Layer

1. Generate synonymous CDS candidates for structurally retained proteins.
2. Require exact translated-protein equality.
3. Compute GC, CAI, codon-frequency, k-mer, repeat, and forbidden-motif features.
4. Record public and internal mRNABERT scores as separate components; the rule
   baseline remains primary until a learned scorer wins on held-out labels.
5. Produce a Pareto shortlist without collapsing structure and mRNA evidence into
   one opaque score.

## Day 7: Recompute And Archive

1. Stop launching jobs that cannot finish at least 12 hours before allocation
   expiry.
2. Independently recompute final candidates at the approved confidence level.
3. Export queue tables, candidate records, checksums, cost measurements, and
   explicit rejection reasons.
4. Write a decision report including negative results and unresolved failures.
5. Verify the archive from a second location.

## Exit Criteria

- At least one approved target reaches a ranked CDS shortlist.
- Immutable-residue and translation-preservation checks pass for every retained
  candidate.
- Fold/refold cost per 1,000 candidates is measured.
- A resumable, provenance-complete label dataset is archived.
- Official, Stage-1, and Stage2a ProteinMPNN experts receive a paired fusion-task
  comparison.
- A surrogate result or an explicit negative result is recorded.
- No structure proxy is described as wet-lab efficacy evidence.

## Conditional Training

Do not extend Stage2a automatically. Independent training seeds may use spare
V100 capacity for validation-only stability analysis, but the opened held-out
test set cannot be reused to select a replacement release. A future replay
mixture or full retraining campaign requires a new predeclared holdout and a
measured fusion-task failure.
