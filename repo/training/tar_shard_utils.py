"""Utilities for reading ProteinMPNN tar-shard datasets."""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import torch


_STORE_CACHE = {}


class TarShardStore:
    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.index_by_chain = {}
        with (self.dataset_dir / "index.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                self.index_by_chain[row["chain_id"]] = row

    def load_payload_for_chain(self, chain_id: str) -> dict:
        row = self.index_by_chain[chain_id]
        shard_path = self.dataset_dir / row["shard"]
        with shard_path.open("rb") as handle:
            handle.seek(row["offset"])
            data = handle.read(row["size"])
        return torch.load(io.BytesIO(data), map_location="cpu")


def get_store(dataset_dir: str | Path) -> TarShardStore:
    key = str(dataset_dir)
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = TarShardStore(dataset_dir)
    return _STORE_CACHE[key]


def loader_tar_pdb(item, params):
    chain_id = item[0]
    _, chid = chain_id.split("_")
    store = get_store(params["DIR"])
    payload = store.load_payload_for_chain(chain_id)
    meta = payload["meta"]
    all_chains = payload["chains"]
    asmb_ids = meta["asmb_ids"]
    asmb_chains = meta["asmb_chains"]
    chids = np.array(meta["chains"])

    asmb_candidates = {
        asmb_id
        for asmb_id, chain_list in zip(asmb_ids, asmb_chains)
        if chid in chain_list.split(",")
    }

    if len(asmb_candidates) < 1:
        chain = all_chains[chid]
        length = len(chain["seq"])
        return {
            "seq": chain["seq"],
            "xyz": chain["xyz"],
            "idx": torch.zeros(length).int(),
            "masked": torch.Tensor([0]).int(),
            "label": chain_id,
        }

    asmb_i = random.sample(list(asmb_candidates), 1)
    selected_transform_idx = np.where(np.array(asmb_ids) == asmb_i)[0]

    chains = {
        c: all_chains[c]
        for i in selected_transform_idx
        for c in asmb_chains[i].split(",")
        if c in meta["chains"]
    }

    asmb = {}
    for k in selected_transform_idx:
        xform = meta[f"asmb_xform{k}"]
        u = xform[:, :3, :3]
        r = xform[:, :3, 3]

        s1 = set(meta["chains"])
        s2 = set(asmb_chains[k].split(","))
        chains_k = s1 & s2

        for c in chains_k:
            try:
                xyz = chains[c]["xyz"]
                xyz_ru = torch.einsum("bij,raj->brai", u, xyz) + r[:, None, None, :]
                asmb.update({(c, k, i): xyz_i for i, xyz_i in enumerate(xyz_ru)})
            except KeyError:
                return {"seq": np.zeros(5)}

    seqid = meta["tm"][chids == chid][0, :, 1]
    homo = {
        ch_j
        for seqid_j, ch_j in zip(seqid, chids)
        if float(seqid_j) > params.get("HOMO", 0.70)
    }

    seq = ""
    xyz = []
    idx = []
    masked = []
    for counter, (assembly_key, transformed_xyz) in enumerate(asmb.items()):
        chain_key = assembly_key[0]
        seq += chains[chain_key]["seq"]
        xyz.append(transformed_xyz)
        idx.append(torch.full((transformed_xyz.shape[0],), counter))
        if chain_key in homo:
            masked.append(counter)

    return {
        "seq": seq,
        "xyz": torch.cat(xyz, dim=0),
        "idx": torch.cat(idx, dim=0),
        "masked": torch.tensor(masked).int(),
        "label": chain_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--chain-id", required=True)
    args = parser.parse_args()
    record = loader_tar_pdb(
        [args.chain_id, ""],
        {"DIR": args.dataset_dir, "HOMO": 0.70},
    )
    print(
        json.dumps(
            {
                "label": record["label"],
                "seq_len": len(record["seq"]),
                "xyz_shape": list(record["xyz"].shape),
                "idx_shape": list(record["idx"].shape),
                "masked": record["masked"].tolist(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
