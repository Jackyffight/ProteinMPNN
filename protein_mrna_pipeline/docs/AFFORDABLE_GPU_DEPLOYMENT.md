# Affordable GPU Deployment Decision

Date: 2026-07-15

Status: Accepted

## Decision

Use one 48 GB NVIDIA A40 or RTX A6000 cloud worker as the default affordable
runtime for the protein-to-mRNA pipeline. Load expert models sequentially rather
than keeping them resident together. Use an RTX 5090 or L40S only as a temporary
throughput worker when its hourly price is competitive.

Use Evo 2 7B in bfloat16 as the genomic-language-model baseline. Do not make Evo
2 20B or an H100/H200 dependency of this project. The current experiment only
needs frozen Evo 2 representations for comparison with mRNABERT; it has not
established a benefit that would justify Hopper-only infrastructure.

A Mac mini may remain the control, preprocessing, and result-archival node. It
is not the execution target for ESMFold2-Fast or Evo 2.

## Supported Workload

This decision covers:

- ProteinMPNN inference, continued training, and bounded evaluation;
- ESMFold2-Fast single-sequence fold and refold inference with the current
  three-loop, 50-step, one-sample configuration;
- Evo 2 7B frozen embedding extraction on ordinary mRNA/CDS inputs;
- mRNABERT inference and bounded fine-tuning;
- the current engineering benchmark and candidate-design queue.

It does not cover:

- Evo 2 20B;
- million-token Evo 2 inference merely because the checkpoint advertises a 1M
  maximum context;
- ESMFold2 or Evo 2 training;
- another full-corpus mRNABERT pretraining campaign on one consumer GPU;
- concurrent residency of ESMFold2-Fast and Evo 2 7B on one device.

Full mRNABERT pretraining remains a separately budgeted, temporary multi-GPU
job. It must not turn the normal pipeline worker into an always-on training
cluster.

## Capacity Audit

| Component | Current artifact | Observed or expected memory behavior | 48 GB worker |
| --- | --- | --- | --- |
| ProteinMPNN | `v_48_020.pt` and promoted Stage2a weights; official checkpoint is about 6.4 MiB | Small relative to every candidate GPU | Supported for inference and training |
| ESMFold2-Fast | Pinned ESMFold2-Fast plus ESMC-6B; about 24.4 GiB of files | About 17.9 GiB reserved for current design candidates and 22.35 GiB at length 791 in the recorded A100 runs | Supported with useful headroom |
| Evo 2 | Pinned `evo2_7b.pt`; about 12.8 GiB | Load in bfloat16 and process bounded sequences; memory still grows with sequence length | Supported for the current frozen-embedding workload |
| mRNABERT | Public checkpoint about 435 MiB; internal model has 12 layers and hidden size 768 | BERT-base-scale inference and fine-tuning | Supported |

A 24 GB RTX 4090 can run ProteinMPNN, mRNABERT, and Evo 2 7B, but the recorded
ESMFold2-Fast length-791 run leaves too little margin for allocator
fragmentation, CUDA context memory, or a more demanding input. Multiple 24 GB
cards do not automatically pool their memory, and the current ESMFold2 runner is
single-GPU. A 24 GB card is therefore a bounded short-sequence worker, not the
default all-pipeline worker.

An RTX 5090 has enough 32 GB memory for the tested ESMFold2 range and is much
faster than an A40. Its tradeoff is a newer CUDA stack and less memory margin.
An A40, RTX A6000, or L40S provides the safer 48 GB envelope.

## Cloud Options and Price Snapshot

Prices below are USD per GPU hour and were checked on 2026-07-14. They are a
planning snapshot, not a committed quote. Inventory, host RAM, storage, and
bandwidth charges must be checked when the instance is created.

| Provider and GPU | VRAM | Observed price | Use |
| --- | ---: | ---: | --- |
| RunPod A40 | 48 GB | $0.44/hour | Default stable worker |
| RunPod RTX A6000 | 48 GB | $0.49/hour | Equivalent fallback to A40 |
| Vast.ai RTX 5090 | 32 GB | approximately $0.32-$0.48/hour | Fast marketplace burst worker |
| Vast.ai L40S | 48 GB | approximately $0.53-$0.80/hour | Fastest preferred 48 GB burst worker when available |
| RunPod RTX 4090 | 24 GB | $0.69/hour | Poor value for this memory-constrained workload |
| RunPod RTX 5090 | 32 GB | $0.99/hour | Use only when availability or provider isolation matters |

Sources:

- RunPod pricing: <https://www.runpod.io/pricing>
- Vast.ai live pricing: <https://vast.ai/pricing>
- Vast.ai pricing model: <https://docs.vast.ai/documentation/instances/pricing>
- Evo 2 requirements: <https://github.com/ArcInstitute/evo2>
- ESMFold2-Fast model card: <https://huggingface.co/biohub/ESMFold2-Fast>

Vast.ai is a marketplace and its lowest offer may disappear. Use on-demand,
verified hosts with a strong reliability score for repeatable benchmark runs.
Do not place unpublished or otherwise sensitive biological sequences on a
community host unless the project's data-handling policy explicitly permits it.

## Budget Envelope

At the RunPod A40 snapshot price:

| Monthly GPU use | Compute cost |
| ---: | ---: |
| 100 hours | about $44 |
| 300 hours | about $132 |
| 720 hours (continuous) | about $317 |

Provision at least 150 GB of persistent storage and prefer 200 GB. The pinned
ESMFold2-Fast and Evo 2 7B checkpoints alone consume about 37.2 GiB. Separate
Python environments, package and compiler caches, ProteinMPNN and mRNABERT
weights, and run artifacts make a smaller volume unnecessarily fragile. A 200
GB persistent volume is expected to add roughly $10-$20 per month depending on
the provider and storage tier.

Cloud rental remains the default while use is intermittent. Reconsider a local
RTX 5090 or used RTX A6000 workstation only after measured utilization remains
above roughly 400-500 GPU hours per month for several months.

## Runtime Profiles

The deployment scripts must select the CUDA build and FlashAttention target from
the actual GPU rather than assuming the existing A100 profile.

| GPU family | Compute target | Runtime requirement |
| --- | --- | --- |
| A100 | `sm80` | Existing Torch 2.7 / CUDA 12.6 profile |
| A40 or RTX A6000 | `sm86` | Torch 2.7 / CUDA 12.6; rebuild Evo 2 FlashAttention for `86` |
| RTX 4090, L40, L40S, or RTX 6000 Ada | `sm89` | Torch 2.7 / CUDA 12.6; rebuild Evo 2 FlashAttention for `89` |
| RTX 5090 | `sm120` | Blackwell-compatible driver and Torch CUDA 12.8 build; rebuild extensions for `120` |

The current mRNABERT-side Evo 2 installer compiles FlashAttention for A100
`sm80` only. Before moving the pipeline, replace that constant with an explicit
GPU profile and fail when the detected GPU does not match the selected profile.
Do not reuse an A100-built extension on another architecture.

ESMFold2-Fast currently has a pure-PyTorch fallback for fused attention and
layer-normalization kernels. Preserve that fallback during the first migration;
fused-kernel installation is a throughput optimization, not a prerequisite for
the acceptance smoke.

## Deployment Shape

Use the following operational split:

```text
local controller or Mac mini
  -> prepare small input manifests and sequence batches
  -> submit to one ephemeral 48 GB GPU worker
       -> ProteinMPNN
       -> unload ProteinMPNN
       -> ESMFold2-Fast
       -> unload ESMFold2-Fast
       -> Evo 2 7B when required
       -> unload Evo 2 7B
       -> mRNABERT scoring or fine-tuning
  -> sync checksummed results back to durable storage
  -> stop or delete the GPU worker
```

Keep each expert model in its existing isolated environment. Share only input
contracts, immutable model caches, and checksummed output artifacts. Do not
merge ESMFold2, Evo 2, ProteinMPNN, and mRNABERT dependencies into one Python
environment.

## Acceptance Gate

Before selecting a provider for routine runs, execute the same bounded workload
on one candidate instance:

1. Record GPU model, driver, CUDA, Torch, host RAM, disk bandwidth, and free
   storage.
2. Run the four-record ESMFold2 smoke and verify every artifact checksum.
3. Run the length-791 ESMFold2 benchmark record and capture peak allocated and
   reserved GPU memory.
4. Extract Evo 2 7B embeddings for a small fixed mRNA/CDS set.
5. Run promoted ProteinMPNN inference and one bounded training step.
6. Run mRNABERT inference and a bounded fine-tuning step.
7. Record wall time, peak memory, failures, and total provider charge.

The worker passes only if all four components run from pinned artifacts without
OOM, outputs survive process restart, and the recorded cost agrees with the
provider bill. Compare cost per completed candidate rather than nominal GPU
TFLOPS or the lowest advertised hourly price.

## Revisit Conditions

Revisit this decision when any of the following occurs:

- ESMFold2 inputs routinely exceed the tested 800-residue range;
- Evo 2 workloads require substantially longer contexts than the current mRNA
  embedding benchmark;
- measured cloud use justifies ownership of a local worker;
- a controlled experiment shows Evo 2 20B materially improves a project metric;
- confidential-data policy prohibits the selected cloud tier;
- provider pricing or inventory changes enough to alter the cost ordering.
