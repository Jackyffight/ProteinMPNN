#!/usr/bin/env python3
"""Evaluate a ProteinMPNN checkpoint through the training data pipeline."""

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from checkpoint_utils import (
    checkpoint_metadata,
    load_checkpoint,
    load_model_weights,
    validate_num_edges,
)
from evaluation_utils import flatten_split_records
from model_utils import ProteinMPNN, featurize, loss_nll
from tar_shard_utils import loader_tar_pdb
from utils import (
    PDB_dataset,
    StructureDataset,
    StructureLoader,
    build_training_clusters,
    get_pdbs,
    loader_pdb,
    worker_init_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ProteinMPNN sequence NLL on a dataset split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset-format", choices=["auto", "pt", "tar"], default="auto")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--output", default="", help="optional JSON result path")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="maximum evaluation structures; 0 evaluates the complete selected split",
    )
    parser.add_argument(
        "--evaluation-unit",
        choices=["records", "clusters"],
        default="records",
        help="records evaluates every held-out chain; clusters samples one chain per cluster",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail if filtering or loading prevents evaluation of every requested structure",
    )
    parser.add_argument("--batch-tokens", type=int, default=3000)
    parser.add_argument("--max-protein-length", type=int, default=2000)
    parser.add_argument(
        "--load-chunk-size",
        type=int,
        default=16,
        help="structures converted and retained in memory at one time",
    )
    parser.add_argument("--rescut", type=float, default=3.5)
    parser.add_argument("--homology-cutoff", type=float, default=0.70)
    parser.add_argument("--date-cutoff", default="2030-Jan-01")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=0,
        help="0 reads num_edges from the checkpoint, falling back to 48",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--backbone-noise", type=float, default=0.0)
    parser.add_argument("--num-loader-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def detect_dataset_format(data_dir, requested_format):
    if requested_format != "auto":
        return requested_format
    if (data_dir / "manifest.json").is_file() and (data_dir / "shards").is_dir():
        return "tar"
    return "pt"


def resolve_device(requested_device):
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda:0")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FixedItemsDataset:
    """Load a deterministic list of chain records without cluster resampling."""

    def __init__(self, items, loader, params):
        self.items = list(items)
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.loader(self.items[index], self.params)


def main():
    args = parse_args()
    if args.max_examples < 0:
        raise ValueError("--max-examples must be >= 0")
    if args.batch_tokens <= 0:
        raise ValueError("--batch-tokens must be positive")
    if args.load_chunk_size <= 0:
        raise ValueError("--load-chunk-size must be positive")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {data_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_num_edges = checkpoint.get("num_edges")
    default_num_neighbors = int(checkpoint_num_edges) if checkpoint_num_edges is not None else 48
    num_neighbors = args.num_neighbors or default_num_neighbors
    validate_num_edges(checkpoint, num_neighbors)

    dataset_format = detect_dataset_format(data_dir, args.dataset_format)
    if dataset_format == "tar":
        pdb_loader = loader_tar_pdb
    else:
        pdb_loader = loader_pdb

    params = {
        "LIST": str(data_dir / "list.csv"),
        "VAL": str(data_dir / "valid_clusters.txt"),
        "TEST": str(data_dir / "test_clusters.txt"),
        "DIR": str(data_dir),
        "DATCUT": args.date_cutoff,
        "RESCUT": args.rescut,
        "HOMO": args.homology_cutoff,
    }

    started_at = time.time()
    print(f"Loading {args.split} split metadata from {data_dir}", file=sys.stderr, flush=True)
    train, valid, test = build_training_clusters(params, debug=False)
    split_clusters = {"train": train, "valid": valid, "test": test}[args.split]
    if not split_clusters:
        raise ValueError(f"dataset split is empty: {args.split}")

    loader_options = {
        "batch_size": 1,
        "shuffle": False,
        "pin_memory": device.type == "cuda",
        "num_workers": args.num_loader_workers,
        "worker_init_fn": worker_init_fn,
        "generator": torch.Generator().manual_seed(args.seed),
    }
    split_records = flatten_split_records(split_clusters)
    if args.evaluation_unit == "records":
        raw_dataset = FixedItemsDataset(split_records, pdb_loader, params)
    else:
        raw_dataset = PDB_dataset(list(split_clusters.keys()), pdb_loader, split_clusters, params)
    raw_loader = torch.utils.data.DataLoader(raw_dataset, **loader_options)
    available_evaluation_units = len(raw_dataset)
    target_structures = (
        available_evaluation_units
        if args.max_examples == 0
        else min(args.max_examples, available_evaluation_units)
    )

    model = ProteinMPNN(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        k_neighbors=num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
    ).to(device)
    load_model_weights(model, checkpoint, checkpoint_path)
    model.eval()

    print(
        f"Evaluating {target_structures}/{available_evaluation_units} "
        f"{args.evaluation_unit} on {device}",
        file=sys.stderr,
        flush=True,
    )
    raw_iterator = iter(raw_loader)
    data_loading_seconds = time.time() - started_at
    evaluation_seconds = 0.0
    loss_sum = 0.0
    accuracy_sum = 0.0
    token_count = 0.0
    batch_count = 0
    evaluated_structures = 0
    evaluated_structure_ids = []
    while evaluated_structures < target_structures:
        requested_chunk_size = min(
            args.load_chunk_size, target_structures - evaluated_structures
        )
        loading_started_at = time.time()
        structure_records = get_pdbs(
            raw_iterator,
            repeat=1,
            max_length=args.max_protein_length,
            num_units=requested_chunk_size,
        )
        data_loading_seconds += time.time() - loading_started_at
        if not structure_records:
            break

        structure_dataset = StructureDataset(
            structure_records,
            verbose=False,
            truncate=None,
            max_length=args.max_protein_length,
        )
        structure_loader = StructureLoader(structure_dataset, batch_size=args.batch_tokens)
        evaluated_structure_ids.extend(record["name"] for record in structure_dataset.data)
        chunk_evaluation_started_at = time.time()
        with torch.no_grad():
            for batch in structure_loader:
                X, S, mask, _, chain_M, residue_idx, _, chain_encoding_all = featurize(batch, device)
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                mask_for_loss = mask * chain_M
                loss, _, true_false = loss_nll(S, log_probs, mask_for_loss)
                batch_tokens = float(torch.sum(mask_for_loss).item())
                if batch_tokens == 0:
                    continue
                loss_sum += float(torch.sum(loss * mask_for_loss).item())
                accuracy_sum += float(torch.sum(true_false * mask_for_loss).item())
                token_count += batch_tokens
                batch_count += 1
        evaluation_seconds += time.time() - chunk_evaluation_started_at
        evaluated_structures += len(structure_dataset)
        print(
            f"Evaluated {evaluated_structures}/{target_structures} structures",
            file=sys.stderr,
            flush=True,
        )
        if len(structure_records) < requested_chunk_size:
            break

    if evaluated_structures == 0 or token_count == 0:
        raise RuntimeError("evaluation produced no structures with masked residues")
    if args.require_complete and evaluated_structures != target_structures:
        raise RuntimeError(
            "evaluation was incomplete: "
            f"evaluated={evaluated_structures} expected={target_structures}"
        )

    mean_loss = loss_sum / token_count
    result = {
        "schema": "proteinmpnn.checkpoint_evaluation.v2",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "metadata": checkpoint_metadata(checkpoint),
        },
        "data": {
            "path": str(data_dir),
            "dataset_format": dataset_format,
            "split": args.split,
            "cluster_count": len(split_clusters),
            "record_count": len(split_records),
            "evaluation_unit": args.evaluation_unit,
            "available_evaluation_units": available_evaluation_units,
            "requested_max_examples": args.max_examples,
            "expected_structures": target_structures,
            "evaluated_structures": evaluated_structures,
            "evaluated_structure_ids": evaluated_structure_ids,
            "evaluated_structure_ids_sha256": hashlib.sha256(
                "\n".join(evaluated_structure_ids).encode("utf-8")
            ).hexdigest(),
            "load_chunk_size": args.load_chunk_size,
            "max_protein_length": args.max_protein_length,
            "rescut": args.rescut,
            "homology_cutoff": args.homology_cutoff,
            "date_cutoff": args.date_cutoff,
            "require_complete": args.require_complete,
        },
        "model": {
            "hidden_dim": args.hidden_dim,
            "num_encoder_layers": args.encoder_layers,
            "num_decoder_layers": args.decoder_layers,
            "num_neighbors": num_neighbors,
            "dropout": args.dropout,
            "backbone_noise": args.backbone_noise,
        },
        "metrics": {
            "nll": mean_loss,
            "perplexity": math.exp(mean_loss),
            "accuracy": accuracy_sum / token_count,
            "masked_residues": int(token_count),
            "batches": batch_count,
        },
        "runtime": {
            "seed": args.seed,
            "device": str(device),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "data_loading_seconds": data_loading_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.time() - started_at,
        },
    }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {output_path}", file=sys.stderr, flush=True)
    print(rendered)


if __name__ == "__main__":
    main()
