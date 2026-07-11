#!/usr/bin/env python3
"""Build a ProteinMPNN training dataset from wwPDB assembly mmCIF files."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch

try:
    from Bio.Align import PairwiseAligner
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
except ImportError as exc:  # pragma: no cover - exercised on training host.
    raise SystemExit(
        "Missing dependency: biopython. Install requirements.txt before building "
        "the custom PDB dataset."
    ) from exc


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
ATOM_INDEX = {"N": 0, "CA": 1, "C": 2, "O": 3}
CHAIN_IDS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
CLUSTER_MAP: dict[tuple[str, str], int] = {}
ENTRY_METADATA: dict[str, dict] = {}
TAR_SHARD_FORMAT = "proteinmpnn.tar_shard.v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--cluster-file", default="")
    parser.add_argument("--entries-index", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=2,
        help="Maximum submitted parser jobs retained by the parent process.",
    )
    parser.add_argument("--assembly-id", default="all")
    parser.add_argument(
        "--assembly-policy",
        choices=["all", "first"],
        default="first",
        help="When assembly-id=all, keep every assembly or one canonical assembly per PDB.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-resolution", type=float, default=3.5)
    parser.add_argument("--min-date", default="")
    parser.add_argument("--max-date", default="2026-07-08")
    parser.add_argument("--min-chain-length", type=int, default=30)
    parser.add_argument("--max-chain-length", type=int, default=10000)
    parser.add_argument("--max-context-length", type=int, default=2000)
    parser.add_argument("--min-resolved-residues", type=int, default=30)
    parser.add_argument("--min-backbone-coverage", type=float, default=0.5)
    parser.add_argument("--max-chains", type=int, default=len(CHAIN_IDS))
    parser.add_argument(
        "--method-allow",
        default="X-RAY DIFFRACTION,ELECTRON MICROSCOPY",
        help="Comma-separated experimental methods. Empty string allows all.",
    )
    parser.add_argument("--valid-frac", type=float, default=0.01)
    parser.add_argument("--test-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def clean(value: str | None, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text in {"", ".", "?"}:
        return default
    return text


def first_value(cif: dict, keys: list[str], default: str = "") -> str:
    for key in keys:
        values = as_list(cif.get(key))
        for value in values:
            text = clean(value)
            if text:
                return text
    return default


def first_float(cif: dict, keys: list[str]) -> float:
    for key in keys:
        values = as_list(cif.get(key))
        for value in values:
            text = clean(value)
            if not text:
                continue
            try:
                parsed = float(text)
            except ValueError:
                continue
            if math.isfinite(parsed):
                return parsed
    return float("nan")


def read_mmcif(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return MMCIF2Dict(handle)
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        return MMCIF2Dict(handle)


def parse_file_name(path: Path) -> tuple[str, str] | None:
    match = re.match(
        r"(?P<pdb>[0-9A-Za-z]{4})-assembly(?P<assembly>[^.]+)\.cif(?:\.gz)?$",
        path.name,
    )
    if not match:
        return None
    assembly_id = re.sub(r"[^A-Za-z0-9]", "", match.group("assembly"))
    return match.group("pdb").lower(), assembly_id.lower()


def assembly_sort_key(assembly_id: str):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.findall(r"\d+|\D+", assembly_id)
    )


def discover_files(
    raw_dir: Path,
    assembly_id: str,
    limit: int,
    assembly_policy: str = "all",
) -> list[Path]:
    files = sorted(raw_dir.rglob("*.cif.gz")) + sorted(raw_dir.rglob("*.cif"))
    if assembly_id != "all":
        pattern = re.compile(rf"-assembly{re.escape(assembly_id)}\.cif(?:\.gz)?$")
        files = [path for path in files if pattern.search(path.name)]
    elif assembly_policy == "first":
        canonical = {}
        for path in files:
            ids = parse_file_name(path)
            if ids is None:
                continue
            pdb_id, candidate_assembly_id = ids
            candidate_key = (assembly_sort_key(candidate_assembly_id), path.name)
            current = canonical.get(pdb_id)
            if current is None or candidate_key < current[0]:
                canonical[pdb_id] = (candidate_key, path)
        files = [canonical[pdb_id][1] for pdb_id in sorted(canonical)]
    elif assembly_policy != "all":
        raise ValueError(f"unsupported assembly_policy: {assembly_policy}")
    if limit > 0:
        files = files[:limit]
    return files


def load_cluster_map(path: str) -> dict[tuple[str, str], int]:
    if not path or not os.path.isfile(path):
        return {}
    opener = gzip.open if path.endswith(".gz") else open
    cluster_map: dict[tuple[str, str], int] = {}
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for cluster_id, line in enumerate(handle, 1):
            for token in line.split():
                match = re.match(r"(?P<pdb>[0-9A-Za-z]{4})_(?P<entity>\S+)", token)
                if match:
                    cluster_map[(match.group("pdb").upper(), match.group("entity"))] = cluster_id
    return cluster_map


def parse_idx_date(value: str) -> str:
    match = re.match(r"(\d{2})/(\d{2})/(\d{2})$", value.strip())
    if not match:
        return ""
    month, day, year = [int(part) for part in match.groups()]
    full_year = 2000 + year if year <= 30 else 1900 + year
    return f"{full_year:04d}-{month:02d}-{day:02d}"


def load_entry_metadata(path: str) -> dict[str, dict]:
    if not path or not os.path.isfile(path):
        return {}
    metadata: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("IDCODE") or line.startswith("-"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            pdb_id = parts[0].strip().upper()
            date = parse_idx_date(parts[2])
            try:
                resolution = float(parts[6])
            except ValueError:
                resolution = float("nan")
            method = parts[7].strip()
            metadata[pdb_id] = {
                "date": date,
                "resolution": resolution,
                "method": method,
                "header": parts[1].strip(),
            }
    return metadata


def stable_hash_int(text: str, modulo: int) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


def cluster_for(pdb_id: str, entity_id: str, sequence: str) -> int:
    cluster_id = CLUSTER_MAP.get((pdb_id.upper(), entity_id))
    if cluster_id is not None:
        return cluster_id
    return 900_000_000 + stable_hash_int(sequence, 99_999_999)


def sequence_identity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.1
    alignment = aligner.align(left, right)[0]
    indices = np.asarray(alignment.indices)
    aligned = (indices[0] >= 0) & (indices[1] >= 0)
    matches = sum(
        left[left_index] == right[right_index]
        for left_index, right_index in zip(indices[0, aligned], indices[1, aligned])
    )
    return float(matches) / float(max(len(left), len(right)))


def value_at(values: list[str], index: int, default: str = "") -> str:
    if index >= len(values):
        return default
    return clean(values[index], default)


def parse_sequence_id(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def amino_acid(mon_id: str) -> str:
    return AA3_TO_1.get(clean(mon_id).upper(), "X")


def polymer_definitions(cif: dict):
    entity_ids = as_list(cif.get("_entity_poly.entity_id"))
    entity_types = as_list(cif.get("_entity_poly.type"))
    protein_entities = {
        value_at(entity_ids, index)
        for index in range(len(entity_ids))
        if value_at(entity_types, index).lower().startswith("polypeptide(l)")
    }

    entity_positions = defaultdict(dict)
    poly_entity_ids = as_list(cif.get("_entity_poly_seq.entity_id"))
    poly_sequence_ids = as_list(cif.get("_entity_poly_seq.num"))
    poly_mon_ids = as_list(cif.get("_entity_poly_seq.mon_id"))
    for index, entity_id in enumerate(poly_entity_ids):
        entity_id = clean(entity_id)
        sequence_id = parse_sequence_id(value_at(poly_sequence_ids, index))
        if not entity_id or sequence_id is None:
            continue
        entity_positions[entity_id][sequence_id] = amino_acid(value_at(poly_mon_ids, index))

    if not protein_entities:
        protein_entities = set(entity_positions)

    chain_positions = defaultdict(dict)
    chain_entities = defaultdict(Counter)
    scheme_chain_ids = as_list(cif.get("_pdbx_poly_seq_scheme.asym_id"))
    scheme_entity_ids = as_list(cif.get("_pdbx_poly_seq_scheme.entity_id"))
    scheme_sequence_ids = as_list(cif.get("_pdbx_poly_seq_scheme.seq_id"))
    scheme_mon_ids = as_list(cif.get("_pdbx_poly_seq_scheme.mon_id"))
    for index, chain_id in enumerate(scheme_chain_ids):
        chain_id = clean(chain_id)
        entity_id = value_at(scheme_entity_ids, index)
        sequence_id = parse_sequence_id(value_at(scheme_sequence_ids, index))
        if (
            not chain_id
            or not entity_id
            or sequence_id is None
            or entity_id not in protein_entities
        ):
            continue
        chain_positions[chain_id][sequence_id] = amino_acid(value_at(scheme_mon_ids, index))
        chain_entities[chain_id][entity_id] += 1

    return protein_entities, entity_positions, chain_positions, chain_entities


def extract_polymer_chains(cif: dict, config: dict) -> dict:
    protein_entities, entity_positions, chain_positions, scheme_chain_entities = (
        polymer_definitions(cif)
    )

    atom_names = as_list(cif.get("_atom_site.label_atom_id"))
    comp_ids = as_list(cif.get("_atom_site.label_comp_id"))
    chain_ids = as_list(cif.get("_atom_site.label_asym_id"))
    entity_ids = as_list(cif.get("_atom_site.label_entity_id"))
    seq_ids = as_list(cif.get("_atom_site.label_seq_id"))
    alt_ids = as_list(cif.get("_atom_site.label_alt_id"))
    groups = as_list(cif.get("_atom_site.group_PDB"))
    xs = as_list(cif.get("_atom_site.Cartn_x"))
    ys = as_list(cif.get("_atom_site.Cartn_y"))
    zs = as_list(cif.get("_atom_site.Cartn_z"))
    occs = as_list(cif.get("_atom_site.occupancy"))
    bfacs = as_list(cif.get("_atom_site.B_iso_or_equiv"))
    models = as_list(cif.get("_atom_site.pdbx_PDB_model_num"))

    residues = defaultdict(dict)
    atom_chain_entities = defaultdict(Counter)
    atom_chain_ids = set()
    for index, atom_name_raw in enumerate(atom_names):
        group = value_at(groups, index, "ATOM").upper()
        if group not in {"ATOM", "HETATM"}:
            continue
        model = value_at(models, index, "1")
        if model not in {"1", ""}:
            continue
        atom_index = ATOM_INDEX.get(clean(atom_name_raw).upper())
        if atom_index is None:
            continue
        alt_id = value_at(alt_ids, index, ".")
        if alt_id not in {".", "?", "A", "1"}:
            continue
        chain_id = value_at(chain_ids, index)
        entity_id = value_at(entity_ids, index, "0")
        sequence_id = parse_sequence_id(value_at(seq_ids, index))
        if not chain_id or sequence_id is None:
            continue
        if protein_entities and entity_id not in protein_entities:
            continue
        if not protein_entities and amino_acid(value_at(comp_ids, index)) == "X":
            continue
        try:
            coordinate = (
                float(value_at(xs, index)),
                float(value_at(ys, index)),
                float(value_at(zs, index)),
            )
        except ValueError:
            continue
        if not all(math.isfinite(component) for component in coordinate):
            continue
        try:
            occupancy = float(value_at(occs, index, "1.0"))
        except ValueError:
            occupancy = 1.0
        if not math.isfinite(occupancy):
            occupancy = 1.0
        try:
            b_factor = float(value_at(bfacs, index, "0.0"))
        except ValueError:
            b_factor = 0.0
        if not math.isfinite(b_factor):
            b_factor = 0.0

        atom_chain_ids.add(chain_id)
        atom_chain_entities[chain_id][entity_id] += 1
        residue = residues[(chain_id, sequence_id)]
        residue["aa"] = amino_acid(value_at(comp_ids, index))
        residue.setdefault("atoms", {})
        previous = residue["atoms"].get(atom_index)
        if previous is None or occupancy > previous["occ"]:
            residue["atoms"][atom_index] = {
                "coord": coordinate,
                "occ": occupancy,
                "bfac": b_factor,
            }

    chains = {}
    for chain_id in sorted(atom_chain_ids):
        entity_counts = scheme_chain_entities.get(chain_id, Counter()).copy()
        entity_counts.update(atom_chain_entities.get(chain_id, Counter()))
        if not entity_counts:
            continue
        entity_id = entity_counts.most_common(1)[0][0]
        positions = chain_positions.get(chain_id) or entity_positions.get(entity_id)
        if not positions:
            continue
        ordered_positions = sorted(positions)
        sequence = "".join(positions[sequence_id] for sequence_id in ordered_positions)
        if not (
            config["min_chain_length"]
            <= len(sequence)
            <= config["max_chain_length"]
        ):
            continue

        xyz = np.full((len(sequence), 14, 3), np.nan, dtype=np.float32)
        mask = np.zeros((len(sequence), 14), dtype=np.bool_)
        b_factors = np.zeros((len(sequence), 14), dtype=np.float32)
        occupancies = np.zeros((len(sequence), 14), dtype=np.float32)
        for offset, sequence_id in enumerate(ordered_positions):
            residue = residues.get((chain_id, sequence_id))
            if not residue:
                continue
            for atom_index, atom in residue["atoms"].items():
                xyz[offset, atom_index, :] = atom["coord"]
                mask[offset, atom_index] = True
                b_factors[offset, atom_index] = atom["bfac"]
                occupancies[offset, atom_index] = atom["occ"]

        resolved_residue_count = int(np.all(mask[:, :4], axis=1).sum())
        backbone_coverage = resolved_residue_count / float(len(sequence))
        if resolved_residue_count < config.get("min_resolved_residues", 0):
            continue
        if backbone_coverage < config.get("min_backbone_coverage", 0.0):
            continue
        chains[chain_id] = {
            "seq": sequence,
            "xyz": torch.tensor(xyz, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "bfac": torch.tensor(b_factors, dtype=torch.float32),
            "occ": torch.tensor(occupancies, dtype=torch.float32),
            "entity_id": entity_id,
            "source_chain_id": chain_id,
            "resolved_residue_count": resolved_residue_count,
            "backbone_coverage": backbone_coverage,
        }
    return chains


def select_target_chain(chains: dict) -> str:
    if not chains:
        raise ValueError("cannot select a target from an empty chain set")
    return min(
        chains,
        key=lambda chain_id: (
            -int(chains[chain_id]["resolved_residue_count"]),
            -float(chains[chain_id]["backbone_coverage"]),
            chain_id,
        ),
    )


def total_context_length(chains: dict) -> int:
    return sum(len(chain["seq"]) for chain in chains.values())


def reconcile_exact_sequence_clusters(rows: list[dict]) -> dict:
    parent = {}

    def find(cluster):
        parent.setdefault(cluster, cluster)
        if parent[cluster] != cluster:
            parent[cluster] = find(parent[cluster])
        return parent[cluster]

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    clusters_by_sequence = defaultdict(set)
    original_clusters = set()
    for row in rows:
        cluster = int(row["CLUSTER"])
        original_clusters.add(cluster)
        find(cluster)
        clusters_by_sequence[row["SEQUENCE"]].add(cluster)

    conflicting_sequences = 0
    for clusters in clusters_by_sequence.values():
        if len(clusters) > 1:
            conflicting_sequences += 1
        first, *rest = sorted(clusters)
        for cluster in rest:
            union(first, cluster)

    component_members = defaultdict(list)
    for cluster in original_clusters:
        component_members[find(cluster)].append(cluster)
    canonical_cluster = {
        member: min(members)
        for members in component_members.values()
        for member in members
    }
    for row in rows:
        row["CLUSTER"] = str(canonical_cluster[int(row["CLUSTER"])])

    final_clusters = {int(row["CLUSTER"]) for row in rows}
    return {
        "exact_sequence_conflicts_before": conflicting_sequences,
        "clusters_merged": len(original_clusters) - len(final_clusters),
    }


def date_in_range(date: str, min_date: str, max_date: str) -> bool:
    if not date:
        return False
    if min_date and date < min_date:
        return False
    if max_date and date > max_date:
        return False
    return True


def method_allowed(method: str, allow: set[str]) -> bool:
    if not allow:
        return True
    method_upper = method.upper()
    return any(item in method_upper for item in allow)


def parse_one(path_str: str, config: dict) -> dict:
    path = Path(path_str)
    ids = parse_file_name(path)
    if ids is None:
        return {"status": "skipped", "reason": "bad_file_name", "path": path_str}
    pdb_id, assembly_id = ids
    entry_id = f"{pdb_id}a{assembly_id}"
    entry_metadata = ENTRY_METADATA.get(pdb_id.upper(), {})

    metadata_method = entry_metadata.get("method", "")
    if metadata_method and not method_allowed(metadata_method, config["method_allow"]):
        return {"status": "skipped", "reason": "method", "path": path_str}
    metadata_date = entry_metadata.get("date", "")
    if metadata_date and not date_in_range(
        metadata_date, config["min_date"], config["max_date"]
    ):
        return {"status": "skipped", "reason": "date", "path": path_str}
    metadata_resolution = entry_metadata.get("resolution", float("nan"))
    if (
        math.isfinite(metadata_resolution)
        and metadata_resolution > 0
        and metadata_resolution > config["max_resolution"]
    ):
        return {"status": "skipped", "reason": "resolution", "path": path_str}

    try:
        cif = read_mmcif(path)
    except Exception as exc:
        return {"status": "failed", "reason": "mmcif_parse_error", "path": path_str, "error": str(exc)}

    method = entry_metadata.get("method") or first_value(cif, ["_exptl.method"], default="")
    if not method_allowed(method, config["method_allow"]):
        return {"status": "skipped", "reason": "method", "path": path_str}

    deposition_date = entry_metadata.get("date") or first_value(
        cif,
        [
            "_pdbx_database_status.recvd_initial_deposition_date",
            "_database_PDB_rev.date_original",
            "_database_PDB_rev.date",
        ],
    )
    if not date_in_range(deposition_date, config["min_date"], config["max_date"]):
        return {"status": "skipped", "reason": "date", "path": path_str}

    resolution = entry_metadata.get("resolution", float("nan"))
    if not math.isfinite(resolution) or resolution <= 0:
        resolution = first_float(
        cif,
        [
            "_refine.ls_d_res_high",
            "_em_3d_reconstruction.resolution",
            "_reflns.d_resolution_high",
        ],
        )
    if (
        not math.isfinite(resolution)
        or resolution <= 0
        or resolution > config["max_resolution"]
    ):
        return {"status": "skipped", "reason": "resolution", "path": path_str}

    if not as_list(cif.get("_atom_site.label_atom_id")):
        return {"status": "skipped", "reason": "no_atoms", "path": path_str}
    chains = extract_polymer_chains(cif, config)

    if not chains:
        return {"status": "skipped", "reason": "no_valid_chains", "path": path_str}
    if len(chains) > config["max_chains"]:
        return {
            "status": "skipped",
            "reason": "too_many_chains",
            "path": path_str,
            "entry_id": entry_id,
            "context_chains": len(chains),
            "context_length": total_context_length(chains),
        }
    context_length = total_context_length(chains)
    max_context_length = config.get("max_context_length", 0)
    if max_context_length > 0 and context_length > max_context_length:
        return {
            "status": "skipped",
            "reason": "context_too_long",
            "path": path_str,
            "entry_id": entry_id,
            "context_chains": len(chains),
            "context_length": context_length,
            "max_context_length": max_context_length,
        }

    target_source_chain_id = select_target_chain(chains)
    remap = {source: CHAIN_IDS[i] for i, source in enumerate(sorted(chains.keys()))}
    write_pt = config.get("write_pt", True)
    out_dir = Path(config["out_dir"]) / "pdb" / entry_id[1:3]
    if write_pt:
        out_dir.mkdir(parents=True, exist_ok=True)

    remapped_chain_ids = []
    chain_payloads = {}
    rows = []
    chain_clusters = []
    for source_chain_id, chain in sorted(chains.items()):
        chain_id = remap[source_chain_id]
        remapped_chain_ids.append(chain_id)
        chain_payload = {
            "seq": chain["seq"],
            "xyz": chain["xyz"],
            "mask": chain["mask"],
            "bfac": chain["bfac"],
            "occ": chain["occ"],
            "source_pdb_id": pdb_id,
            "source_assembly_id": assembly_id,
            "source_chain_id": source_chain_id,
            "source_entity_id": chain["entity_id"],
            "resolved_residue_count": chain["resolved_residue_count"],
            "backbone_coverage": chain["backbone_coverage"],
        }
        chain_payloads[chain_id] = chain_payload
        if write_pt:
            torch.save(chain_payload, out_dir / f"{entry_id}_{chain_id}.pt")
        cluster = cluster_for(pdb_id, chain["entity_id"], chain["seq"])
        if source_chain_id == target_source_chain_id:
            rows.append(
                {
                    "CHAINID": f"{entry_id}_{chain_id}",
                    "DEPOSITION": deposition_date,
                    "RESOLUTION": f"{resolution:.2f}",
                    "HASH": f"{stable_hash_int(chain['seq'], 1_000_000):06d}",
                    "CLUSTER": str(cluster),
                    "SEQUENCE": chain["seq"],
                }
            )
        chain_clusters.append(cluster)

    n = len(remapped_chain_ids)
    tm = torch.zeros((n, n, 3), dtype=torch.float32)
    for index in range(n):
        tm[index, index, 0] = 1.0
        tm[index, index, 1] = 1.0
    target_chain_id = remap[target_source_chain_id]
    target_index = remapped_chain_ids.index(target_chain_id)
    target_chain = chains[target_source_chain_id]
    for i in range(n):
        source_chain_id = sorted(chains)[i]
        other_chain = chains[source_chain_id]
        if (
            other_chain["entity_id"] == target_chain["entity_id"]
            or other_chain["seq"] == target_chain["seq"]
        ):
            identity = 1.0
        elif chain_clusters[i] == chain_clusters[target_index]:
            identity = sequence_identity(target_chain["seq"], other_chain["seq"])
        else:
            identity = 0.0
        tm[target_index, i, 0] = identity
        tm[target_index, i, 1] = identity
        tm[i, target_index, 0] = identity
        tm[i, target_index, 1] = identity

    meta = {
        "method": method,
        "date": deposition_date,
        "resolution": float(resolution),
        "chains": remapped_chain_ids,
        "source_pdb_id": pdb_id,
        "source_assembly_id": assembly_id,
        "source_chain_map": {remap[k]: k for k in remap},
        "target_chain": target_chain_id,
        "target_selection_policy": "max_resolved_backbone_then_coverage_then_source_id",
        "tm": tm,
        "asmb_ids": ["1"],
        "asmb_details": ["coordinates from wwPDB biological assembly mmCIF"],
        "asmb_method": ["identity"],
        "asmb_chains": [",".join(remapped_chain_ids)],
        "asmb_xform0": torch.eye(4, dtype=torch.float32).reshape(1, 4, 4),
    }
    if write_pt:
        torch.save(meta, out_dir / f"{entry_id}.pt")

    result = {
        "status": "ok",
        "path": path_str,
        "entry_id": entry_id,
        "chains": len(chain_payloads),
        "targets": len(rows),
        "rows": rows,
    }
    if config.get("return_payload", False):
        payload = {
            "format": TAR_SHARD_FORMAT,
            "entry_id": entry_id,
            # Cluster and split metadata can change during final reconciliation.
            # Keep it in list/index files so the structure payload stays immutable.
            "target_chain_ids": [row["CHAINID"] for row in rows],
            "meta": meta,
            "chains": chain_payloads,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        result["payload"] = buffer.getvalue()
    return result


def write_splits(rows: list[dict], out_dir: Path, valid_frac: float, test_frac: float, seed: int) -> dict:
    cluster_ids = sorted({int(row["CLUSTER"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(cluster_ids)
    n_total = len(cluster_ids)
    n_valid = max(1, int(n_total * valid_frac)) if n_total >= 3 else 0
    n_test = max(1, int(n_total * test_frac)) if n_total >= 3 else 0
    valid = sorted(cluster_ids[:n_valid])
    test = sorted(cluster_ids[n_valid : n_valid + n_test])

    with (out_dir / "valid_clusters.txt").open("w", encoding="utf-8") as handle:
        for cluster in valid:
            handle.write(f"{cluster}\n")
    with (out_dir / "test_clusters.txt").open("w", encoding="utf-8") as handle:
        for cluster in test:
            handle.write(f"{cluster}\n")

    valid_set = set(valid)
    test_set = set(test)
    sequence_splits = defaultdict(set)
    pdb_splits = defaultdict(set)
    pdb_target_counts = Counter()
    for row in rows:
        cluster = int(row["CLUSTER"])
        split = "valid" if cluster in valid_set else "test" if cluster in test_set else "train"
        sequence_splits[row["SEQUENCE"]].add(split)
        pdb_id = row["CHAINID"][:4].lower()
        pdb_splits[pdb_id].add(split)
        pdb_target_counts[pdb_id] += 1

    exact_sequence_split_leaks = sum(len(splits) > 1 for splits in sequence_splits.values())
    pdb_split_leaks = sum(len(splits) > 1 for splits in pdb_splits.values())
    duplicate_pdb_targets = sum(count > 1 for count in pdb_target_counts.values())
    if exact_sequence_split_leaks or pdb_split_leaks or duplicate_pdb_targets:
        raise RuntimeError(
            "split integrity failure: "
            f"exact_sequence_split_leaks={exact_sequence_split_leaks}, "
            f"pdb_split_leaks={pdb_split_leaks}, "
            f"duplicate_pdb_targets={duplicate_pdb_targets}"
        )

    return {
        "cluster_count": n_total,
        "valid_cluster_count": len(valid),
        "test_cluster_count": len(test),
        "exact_sequence_split_leaks": exact_sequence_split_leaks,
        "pdb_split_leaks": pdb_split_leaks,
        "duplicate_pdb_targets": duplicate_pdb_targets,
    }


def iter_parse_results(
    files: list[Path],
    config: dict,
    workers: int,
    max_in_flight: int,
):
    if workers <= 1:
        for path in files:
            yield parse_one(str(path), config)
        return

    in_flight_limit = max(workers, max_in_flight)
    file_iter = iter(files)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = set()

        def submit_until_full() -> None:
            while len(futures) < in_flight_limit:
                try:
                    path = next(file_iter)
                except StopIteration:
                    return
                futures.add(executor.submit(parse_one, str(path), config))

        submit_until_full()
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
            submit_until_full()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.max_in_flight < 1:
        raise SystemExit("--max-in-flight must be positive")
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pdb").mkdir(exist_ok=True)

    global CLUSTER_MAP
    CLUSTER_MAP = load_cluster_map(args.cluster_file)
    global ENTRY_METADATA
    ENTRY_METADATA = load_entry_metadata(args.entries_index)

    files = discover_files(
        raw_dir,
        args.assembly_id,
        args.limit,
        assembly_policy=args.assembly_policy,
    )
    if not files:
        raise SystemExit(f"No mmCIF files found under {raw_dir}")

    method_allow = {item.strip().upper() for item in args.method_allow.split(",") if item.strip()}
    config = {
        "out_dir": str(out_dir),
        "max_resolution": args.max_resolution,
        "min_date": args.min_date,
        "max_date": args.max_date,
        "min_chain_length": args.min_chain_length,
        "max_chain_length": args.max_chain_length,
        "max_context_length": args.max_context_length,
        "min_resolved_residues": args.min_resolved_residues,
        "min_backbone_coverage": args.min_backbone_coverage,
        "max_chains": args.max_chains,
        "method_allow": method_allow,
    }

    started = time.time()
    rows: list[dict] = []
    stats = Counter()
    failures = []
    deferred_oversized = []
    print(f"raw_dir: {raw_dir}")
    print(f"out_dir: {out_dir}")
    print(f"files: {len(files)}")
    print(f"workers: {args.workers}")
    print(f"max_in_flight: {max(args.workers, args.max_in_flight)}")
    print(f"cluster_map_entries: {len(CLUSTER_MAP)}")
    print(f"entry_metadata_records: {len(ENTRY_METADATA)}")

    iterator = iter_parse_results(
        files, config, args.workers, args.max_in_flight
    )
    for index, result in enumerate(iterator, 1):
        status_key = (
            result["status"]
            if result["status"] == "ok"
            else result.get("reason", "unknown")
        )
        stats[status_key] += 1
        if result["status"] == "ok":
            rows.extend(result["rows"])
        elif result["status"] == "failed":
            failures.append(result)
        elif result.get("reason") in {"context_too_long", "too_many_chains"}:
            deferred_oversized.append(result)
        if index % 1000 == 0:
            print(f"processed={index} ok_entries={stats['ok']} rows={len(rows)}")

    reconciliation_stats = reconcile_exact_sequence_clusters(rows)
    rows.sort(key=lambda row: row["CHAINID"])
    with (out_dir / "list.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["CHAINID", "DEPOSITION", "RESOLUTION", "HASH", "CLUSTER", "SEQUENCE"],
        )
        writer.writeheader()
        writer.writerows(rows)

    split_stats = write_splits(rows, out_dir, args.valid_frac, args.test_frac, args.seed)

    with (out_dir / "README").open("w", encoding="utf-8") as handle:
        handle.write(f"{args.version_id}\n")
        handle.write("ProteinMPNN-compatible dataset built from wwPDB biological assembly mmCIF files.\n")

    manifest = {
        "version_id": args.version_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "assembly_id": args.assembly_id,
        "assembly_policy": args.assembly_policy,
        "cluster_file": args.cluster_file,
        "entries_index": args.entries_index,
        "cluster_map_entries": len(CLUSTER_MAP),
        "entry_metadata_records": len(ENTRY_METADATA),
        "concurrency": {
            "workers": args.workers,
            "max_in_flight": max(args.workers, args.max_in_flight),
        },
        "filters": {
            "max_resolution": args.max_resolution,
            "min_date": args.min_date,
            "max_date": args.max_date,
            "min_chain_length": args.min_chain_length,
            "max_chain_length": args.max_chain_length,
            "max_context_length": args.max_context_length,
            "min_resolved_residues": args.min_resolved_residues,
            "min_backbone_coverage": args.min_backbone_coverage,
            "max_chains": args.max_chains,
            "method_allow": sorted(method_allow),
        },
        "counts": {
            "input_files": len(files),
            "list_rows": len(rows),
            "ok_entries": stats["ok"],
            "failures": len(failures),
            "deferred_oversized": len(deferred_oversized),
            **reconciliation_stats,
            **split_stats,
        },
        "skip_reasons": dict(stats),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with (out_dir / "build_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    if failures:
        with (out_dir / "build_failures.jsonl").open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, sort_keys=True) + "\n")
    if deferred_oversized:
        with (out_dir / "build_deferred_oversized.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for deferred in deferred_oversized:
                handle.write(json.dumps(deferred, sort_keys=True) + "\n")

    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    if not rows:
        raise SystemExit("No training rows were produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
