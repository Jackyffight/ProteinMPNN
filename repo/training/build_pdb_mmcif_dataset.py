#!/usr/bin/env python3
"""Build a ProteinMPNN training dataset from wwPDB assembly mmCIF files."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

try:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--cluster-file", default="")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--assembly-id", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-resolution", type=float, default=3.5)
    parser.add_argument("--min-date", default="")
    parser.add_argument("--max-date", default="2026-07-08")
    parser.add_argument("--min-chain-length", type=int, default=30)
    parser.add_argument("--max-chain-length", type=int, default=10000)
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


def discover_files(raw_dir: Path, assembly_id: str, limit: int) -> list[Path]:
    files = sorted(raw_dir.rglob("*.cif.gz")) + sorted(raw_dir.rglob("*.cif"))
    if assembly_id != "all":
        pattern = re.compile(rf"-assembly{re.escape(assembly_id)}\.cif(?:\.gz)?$")
        files = [path for path in files if pattern.search(path.name)]
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
    denom = max(len(left), len(right))
    matches = sum(a == b for a, b in zip(left, right))
    return float(matches) / float(denom)


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

    try:
        cif = read_mmcif(path)
    except Exception as exc:
        return {"status": "failed", "reason": "mmcif_parse_error", "path": path_str, "error": str(exc)}

    method = first_value(cif, ["_exptl.method"], default="")
    if not method_allowed(method, config["method_allow"]):
        return {"status": "skipped", "reason": "method", "path": path_str}

    deposition_date = first_value(
        cif,
        [
            "_pdbx_database_status.recvd_initial_deposition_date",
            "_database_PDB_rev.date_original",
            "_database_PDB_rev.date",
        ],
    )
    if not date_in_range(deposition_date, config["min_date"], config["max_date"]):
        return {"status": "skipped", "reason": "date", "path": path_str}

    resolution = first_float(
        cif,
        [
            "_refine.ls_d_res_high",
            "_em_3d_reconstruction.resolution",
            "_reflns.d_resolution_high",
        ],
    )
    if not math.isfinite(resolution) or resolution > config["max_resolution"]:
        return {"status": "skipped", "reason": "resolution", "path": path_str}

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

    n_atoms = len(atom_names)
    if n_atoms == 0:
        return {"status": "skipped", "reason": "no_atoms", "path": path_str}

    def value(values: list[str], index: int, default: str = "") -> str:
        if index >= len(values):
            return default
        return clean(values[index], default)

    residues = defaultdict(dict)
    chain_entity_counts = defaultdict(Counter)
    for i in range(n_atoms):
        if value(groups, i, "ATOM").upper() != "ATOM":
            continue
        model = value(models, i, "1")
        if model not in {"1", ""}:
            continue
        atom_name = value(atom_names, i).upper()
        atom_index = ATOM_INDEX.get(atom_name)
        if atom_index is None:
            continue
        alt_id = value(alt_ids, i, ".")
        if alt_id not in {".", "?", "A", "1"}:
            continue
        comp_id = value(comp_ids, i).upper()
        aa = AA3_TO_1.get(comp_id)
        if aa is None:
            continue
        chain_id = value(chain_ids, i)
        entity_id = value(entity_ids, i, "0")
        seq_id_raw = value(seq_ids, i)
        if not chain_id or not seq_id_raw:
            continue
        try:
            seq_id = int(float(seq_id_raw))
            x = float(value(xs, i))
            y = float(value(ys, i))
            z = float(value(zs, i))
            occupancy = float(value(occs, i, "1.0"))
            b_factor = float(value(bfacs, i, "0.0"))
        except ValueError:
            continue

        key = (chain_id, seq_id)
        chain_entity_counts[chain_id][entity_id] += 1
        residue = residues[key]
        residue["aa"] = aa
        residue["entity_id"] = entity_id
        residue.setdefault("atoms", {})
        previous = residue["atoms"].get(atom_index)
        if previous is None or occupancy > previous["occ"]:
            residue["atoms"][atom_index] = {
                "coord": (x, y, z),
                "occ": occupancy,
                "bfac": b_factor,
            }

    chains = {}
    for chain_id in sorted({chain for chain, _ in residues.keys()}):
        chain_residue_ids = sorted(seq_id for chain, seq_id in residues.keys() if chain == chain_id)
        seq = []
        coords = []
        masks = []
        b_factors = []
        occupancies = []
        for seq_id in chain_residue_ids:
            residue = residues[(chain_id, seq_id)]
            atoms = residue["atoms"]
            if any(index not in atoms for index in ATOM_INDEX.values()):
                continue
            xyz = np.full((14, 3), np.nan, dtype=np.float32)
            mask = np.zeros((14,), dtype=np.bool_)
            bfac = np.zeros((14,), dtype=np.float32)
            occ = np.zeros((14,), dtype=np.float32)
            for atom_index in ATOM_INDEX.values():
                atom = atoms[atom_index]
                xyz[atom_index, :] = atom["coord"]
                mask[atom_index] = True
                bfac[atom_index] = atom["bfac"]
                occ[atom_index] = atom["occ"]
            seq.append(residue["aa"])
            coords.append(xyz)
            masks.append(mask)
            b_factors.append(bfac)
            occupancies.append(occ)
        sequence = "".join(seq)
        if not (config["min_chain_length"] <= len(sequence) <= config["max_chain_length"]):
            continue
        entity_id = chain_entity_counts[chain_id].most_common(1)[0][0]
        chains[chain_id] = {
            "seq": sequence,
            "xyz": torch.tensor(np.stack(coords), dtype=torch.float32),
            "mask": torch.tensor(np.stack(masks), dtype=torch.bool),
            "bfac": torch.tensor(np.stack(b_factors), dtype=torch.float32),
            "occ": torch.tensor(np.stack(occupancies), dtype=torch.float32),
            "entity_id": entity_id,
            "source_chain_id": chain_id,
        }

    if not chains:
        return {"status": "skipped", "reason": "no_valid_chains", "path": path_str}
    if len(chains) > config["max_chains"]:
        return {"status": "skipped", "reason": "too_many_chains", "path": path_str}

    remap = {source: CHAIN_IDS[i] for i, source in enumerate(sorted(chains.keys()))}
    out_dir = Path(config["out_dir"]) / "pdb" / entry_id[1:3]
    out_dir.mkdir(parents=True, exist_ok=True)

    remapped_chain_ids = []
    rows = []
    sequences = []
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
        }
        torch.save(chain_payload, out_dir / f"{entry_id}_{chain_id}.pt")
        cluster = cluster_for(pdb_id, chain["entity_id"], chain["seq"])
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
        sequences.append(chain["seq"])

    n = len(remapped_chain_ids)
    tm = torch.zeros((n, n, 3), dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            seq_id_value = 1.0 if i == j else sequence_identity(sequences[i], sequences[j])
            tm[i, j, 0] = seq_id_value
            tm[i, j, 1] = seq_id_value
            tm[i, j, 2] = 0.0

    meta = {
        "method": method,
        "date": deposition_date,
        "resolution": float(resolution),
        "chains": remapped_chain_ids,
        "source_pdb_id": pdb_id,
        "source_assembly_id": assembly_id,
        "source_chain_map": {remap[k]: k for k in remap},
        "tm": tm,
        "asmb_ids": ["1"],
        "asmb_details": ["coordinates from wwPDB biological assembly mmCIF"],
        "asmb_method": ["identity"],
        "asmb_chains": [",".join(remapped_chain_ids)],
        "asmb_xform0": torch.eye(4, dtype=torch.float32).reshape(1, 4, 4),
    }
    torch.save(meta, out_dir / f"{entry_id}.pt")

    return {
        "status": "ok",
        "path": path_str,
        "entry_id": entry_id,
        "chains": len(rows),
        "rows": rows,
    }


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

    return {
        "cluster_count": n_total,
        "valid_cluster_count": len(valid),
        "test_cluster_count": len(test),
    }


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pdb").mkdir(exist_ok=True)

    global CLUSTER_MAP
    CLUSTER_MAP = load_cluster_map(args.cluster_file)

    files = discover_files(raw_dir, args.assembly_id, args.limit)
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
        "max_chains": args.max_chains,
        "method_allow": method_allow,
    }

    started = time.time()
    rows: list[dict] = []
    stats = Counter()
    failures = []
    print(f"raw_dir: {raw_dir}")
    print(f"out_dir: {out_dir}")
    print(f"files: {len(files)}")
    print(f"cluster_map_entries: {len(CLUSTER_MAP)}")

    if args.workers <= 1:
        iterator = (parse_one(str(path), config) for path in files)
        for index, result in enumerate(iterator, 1):
            stats[result["status"] if result["status"] == "ok" else result.get("reason", "unknown")] += 1
            if result["status"] == "ok":
                rows.extend(result["rows"])
            elif result["status"] == "failed":
                failures.append(result)
            if index % 1000 == 0:
                print(f"processed={index} ok_entries={stats['ok']} rows={len(rows)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(parse_one, str(path), config): path for path in files}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                stats[result["status"] if result["status"] == "ok" else result.get("reason", "unknown")] += 1
                if result["status"] == "ok":
                    rows.extend(result["rows"])
                elif result["status"] == "failed":
                    failures.append(result)
                if index % 1000 == 0:
                    print(f"processed={index} ok_entries={stats['ok']} rows={len(rows)}")

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
        "cluster_file": args.cluster_file,
        "cluster_map_entries": len(CLUSTER_MAP),
        "filters": {
            "max_resolution": args.max_resolution,
            "min_date": args.min_date,
            "max_date": args.max_date,
            "min_chain_length": args.min_chain_length,
            "max_chain_length": args.max_chain_length,
            "max_chains": args.max_chains,
            "method_allow": sorted(method_allow),
        },
        "counts": {
            "input_files": len(files),
            "list_rows": len(rows),
            "ok_entries": stats["ok"],
            "failures": len(failures),
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

    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    if not rows:
        raise SystemExit("No training rows were produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
