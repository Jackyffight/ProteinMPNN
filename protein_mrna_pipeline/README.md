# Protein-to-mRNA Pipeline

This directory is the integration project for the auditable design workflow:

```text
target package
  -> structure fold and geometry
  -> constrained ProteinMPNN design
  -> structure refold and preservation gates
  -> synonymous CDS generation
  -> rule-based and mRNABERT scoring
  -> Pareto-ranked candidates
```

ProteinMPNN, the structure oracle, and mRNABERT remain isolated expert tools.
This project owns their contracts, provenance, queue state, hard checks, and
candidate tables. It does not vendor their weights or hide their component
scores behind one scalar.

The project has two separate lanes:

- the active **engineering benchmark lane** uses fixed native sequences from the
  existing PDB validation split to test throughput, recovery, provenance, and
  storage without making biological design decisions;
- the conditional **research design lane** starts only when a research owner
  supplies and reviews a real target package. Engineers are not expected to
  invent domains, mutable residues, linkers, or biological objectives.

## Current Status

Implemented in the initial scaffold:

- strict JSON Schemas for target packages, work items, tool results, candidate
  records, and run manifests;
- semantic target checks for domain references, residue bounds, immutable versus
  mutable overlap, linker coverage, and maximum length;
- canonical JSON SHA256 identities and deterministic work IDs;
- deterministic, cluster-unique PDB validation benchmarks generated without
  reading structure tar shards or importing torch;
- atomic run initialization with a mandatory safety gate;
- a transactional SQLite queue with leases, retries, attempt history, status,
  and JSONL export;
- a typed adapter boundary for future structure, ProteinMPNN, and mRNA workers.

The installable JSON Schemas live beside the package source under
`src/protein_mrna_pipeline/schemas/`; they are included in wheels and are the
contract source of truth.

Not implemented yet:

- an ESMFold2 or fallback structure adapter;
- ProteinMPNN inference orchestration;
- structure comparison metrics;
- synonymous CDS generation and mRNABERT adapters;
- a real reviewed target package.

No large GPU job should start until the structure adapter passes the bounded
engineering smoke and throughput gate in `docs/SEVEN_DAY_EXECUTION_PLAN.md`.
An approved target is required for design work, not for the native-sequence
engineering benchmark.

## Install

Use an isolated environment for the orchestration layer. Expert tools should
keep their own environments.

```bash
python -m venv protein_mrna_pipeline/.venv
. protein_mrna_pipeline/.venv/bin/activate
pip install -e protein_mrna_pipeline
```

For repository-local development without installation:

```bash
PYTHONPATH=protein_mrna_pipeline/src \
  python -m protein_mrna_pipeline --help
```

## Engineering Benchmark

This is the current entry point and requires no research input. From the
ProteinMPNN repository root on the GPU server:

```bash
scripts/prepare_2026_structure_benchmark.sh --dry-run
scripts/prepare_2026_structure_benchmark.sh
```

The selected Python environment must contain the project dependency
`jsonschema>=4.18`; the script checks this before writing output.

The default selection is 40 records from `valid`, seed 42, lengths 50-800, one
record per sequence cluster, with exact sequences deduplicated. It verifies that
the source dataset passed split-leak validation. It never opens the large tar
shards and never consumes `test` records.
This fixed `valid` suite is for throughput calibration and evaluation only; it
must not train a surrogate or tune model weights.

Outputs:

```text
runs/benchmarks/structure-input-valid-.../
  benchmark-suite.json
  sequences.fasta
```

This command does not start a GPU process. The next engineering milestone is to
pin an actually available structure-model runtime and connect its adapter to
these records.

Inventory that runtime without installing packages or starting inference:

```bash
scripts/inspect_structure_runtime.sh
```

## Research Target Lane

The included target example is intentionally unreviewed and cannot initialize a
formal design run by default. It is not needed for the engineering benchmark:

```bash
protein-mrna-pipeline validate \
  --kind target \
  protein_mrna_pipeline/examples/target-package.example.json

protein-mrna-pipeline init-run \
  --target protein_mrna_pipeline/examples/target-package.example.json \
  --run-dir protein_mrna_pipeline/runs/example
```

The second command must fail until `safety.status` is `approved` with a review
record. `--allow-unreviewed` exists only for local engineering smoke tests and is
recorded in the run manifest.

## Queue Workflow

Create a run, enqueue a validated work item, claim it from a worker, then finish
it with a validated tool result:

```bash
protein-mrna-pipeline init-run \
  --target /path/to/approved-target.json \
  --run-dir runs/target-a

protein-mrna-pipeline enqueue \
  --run-dir runs/target-a \
  --item /path/to/work-item.json

protein-mrna-pipeline claim \
  --run-dir runs/target-a \
  --worker-id structure-gpu-0 \
  --lease-seconds 3600

protein-mrna-pipeline renew \
  --run-dir runs/target-a \
  --work-id initial_fold-... \
  --worker-id structure-gpu-0 \
  --attempt 1 \
  --lease-seconds 3600

protein-mrna-pipeline finish \
  --run-dir runs/target-a \
  --worker-id structure-gpu-0 \
  --result /path/to/tool-result.json

protein-mrna-pipeline status --run-dir runs/target-a
protein-mrna-pipeline export --run-dir runs/target-a
```

Workers must write output artifacts under their assigned work directory and
return relative paths plus SHA256 and byte size in the tool result. A
`retryable` result returns the item to the pending queue while retaining the
failed attempt, up to the work item's required `max_attempts`. Long-running
adapters must renew their lease before it expires.

## Run Layout

```text
<run-dir>/
  run-manifest.json
  inputs/
    target-package.json
  queue.sqlite3
  work/
    <work-id>/
  artifacts/
  tables/
    work-items.jsonl
    attempts.jsonl
```

The SQLite database is live mutable state. Exported JSONL tables and completed
artifacts are the archival interface. Copy immutable completed shards
continuously rather than waiting until the end of a GPU allocation.

SQLite is the initial single-controller backend for workers sharing one reliable
POSIX filesystem. Run a lock/lease smoke test on the actual mounted storage
before multi-process scale-up. Do not assume arbitrary NFS/object storage has
safe SQLite locking; replace the backend before distributing workers across an
unsupported filesystem.

## Tests

```bash
python -m unittest discover -s protein_mrna_pipeline/tests -v
```

The existing top-level `design_manifest.schema.json` remains the legacy
ProteinMPNN-to-mRNABERT bridge. The schemas in this project are the intended
source of truth for new end-to-end runs; migration of old rows should be
explicit rather than silently changing the old contract.
