"""Deterministic paired ProteinMPNN design pilot on fixed benchmark backbones."""

from __future__ import annotations

import gc
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable

from .benchmark import sha256_file, verify_benchmark_suite_files
from .contracts import ContractError, document_sha256, read_json, text_sha256
from .esmfold2_runner import verify_esmfold2_benchmark_run
from .run_store import write_json_atomic, write_jsonl_atomic
from .structure_agreement import (
    _load_payload,
    _selected_index_rows,
    _to_numpy,
    _validate_selected_index_rows,
    _validate_metrics_runtime_document,
    _verify_dataset,
    load_metrics_runtime_manifest,
    utc_now,
)


OFFICIAL_CHECKPOINT_SHA256 = (
    "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
)
STAGE2A_CHECKPOINT_SHA256 = (
    "08fc2549004d0e8a8b1ac1983dd4e94772f15445732926d8f7e677a4464ba6f7"
)
DEFAULT_SEEDS = (11, 23, 42, 67)
DEFAULT_TEMPERATURE = 0.1
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
CANONICAL_AMINO_ACIDS = frozenset(ALPHABET[:-1])


def _identity(document: dict, field: str) -> str:
    payload = dict(document)
    payload.pop(field, None)
    payload.pop("created_at_utc", None)
    return document_sha256(payload)


def _json_scalar(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _read_jsonl(path: Path) -> list[dict]:
    import json

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ContractError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _validate_native_agreement(
    evaluation_dir: Path,
    suite: dict,
) -> tuple[dict, list[dict]]:
    summary_path = evaluation_dir / "summary.json"
    records_path = evaluation_dir / "records.jsonl"
    summary = read_json(summary_path)
    if (
        summary.get("schema_version")
        != "protein-mrna.native-agreement-summary.v1"
        or summary.get("status") != "passed"
        or summary.get("records", {}).get("selected") != len(suite["records"])
        or summary.get("records", {}).get("succeeded") != len(suite["records"])
        or summary.get("records", {}).get("failed") != 0
        or summary.get("records", {}).get("pending") != 0
    ):
        raise ContractError("native agreement evaluation is not complete")
    rows = _read_jsonl(records_path)
    if len(rows) != len(suite["records"]):
        raise ContractError("native agreement records count differs from benchmark")
    expected_ids = {record["benchmark_record_id"] for record in suite["records"]}
    suite_by_id = {
        record["benchmark_record_id"]: record for record in suite["records"]
    }
    observed_ids = set()
    for row in rows:
        record_id = row.get("benchmark_record_id")
        if (
            row.get("schema_version") != "protein-mrna.native-agreement-record.v1"
            or row.get("result_identity") != _identity(row, "result_identity")
            or row.get("evaluation_identity") != summary["evaluation_identity"]
            or row.get("status") != "succeeded"
            or record_id in observed_ids
        ):
            raise ContractError(f"invalid native agreement record: {record_id}")
        benchmark_record = suite_by_id.get(record_id)
        if (
            benchmark_record is None
            or row.get("source_chain_id") != benchmark_record["source_chain_id"]
            or int(row.get("sequence_length", -1)) != benchmark_record["length"]
            or row.get("sequence_sha256") != benchmark_record["sequence_sha256"]
        ):
            raise ContractError(
                f"native agreement record differs from benchmark: {record_id}"
            )
        observed_ids.add(record_id)
    if observed_ids != expected_ids:
        raise ContractError("native agreement record IDs differ from benchmark")
    return summary, rows


def select_pilot_backbones(records: list[dict]) -> list[dict]:
    """Select four distinct records that exercise complementary failure modes."""

    if len(records) < 4:
        raise ContractError("paired pilot requires at least four native agreement records")
    by_id = {row["benchmark_record_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ContractError("native agreement records contain duplicate IDs")
    selected = []
    used = set()

    def pick(role: str, candidates: list[dict], key) -> None:
        available = [row for row in candidates if row["benchmark_record_id"] not in used]
        if not available:
            raise ContractError(f"no record available for pilot role: {role}")
        row = min(available, key=key)
        used.add(row["benchmark_record_id"])
        selected.append(
            {
                "selection_role": role,
                "benchmark_record_id": row["benchmark_record_id"],
                "source_chain_id": row["source_chain_id"],
                "sequence_length": int(row["sequence_length"]),
                "length_bin": row["length_bin"],
                "native_agreement": dict(row["metrics"]),
            }
        )

    pick(
        "lowest_ca_lddt",
        records,
        lambda row: (
            float(row["metrics"]["ca_lddt"]),
            row["benchmark_record_id"],
        ),
    )
    high_coverage = [
        row for row in records if float(row["metrics"]["native_ca_coverage"]) >= 0.95
    ]
    pick(
        "lowest_resolved_tm_high_coverage",
        high_coverage,
        lambda row: (
            float(row["metrics"]["ca_tm_score_resolved"]),
            row["benchmark_record_id"],
        ),
    )
    pick(
        "longest_sequence",
        records,
        lambda row: (-int(row["sequence_length"]), row["benchmark_record_id"]),
    )
    pick(
        "highest_ca_lddt_control",
        records,
        lambda row: (
            -float(row["metrics"]["ca_lddt"]),
            row["benchmark_record_id"],
        ),
    )
    return selected


def _load_checkpoint(path: Path) -> dict:
    try:
        import torch
    except Exception as error:
        raise ContractError(f"Torch is required for ProteinMPNN generation: {error}") from error
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state_dict"), dict
    ):
        raise ContractError(f"invalid ProteinMPNN checkpoint: {path}")
    return checkpoint


def _verify_checkpoint_file(path: Path, label: str, expected_sha256: str) -> str:
    if not path.is_file():
        raise ContractError(f"ProteinMPNN checkpoint not found: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ContractError(
            f"ProteinMPNN checkpoint SHA256 mismatch for {label}: "
            f"expected={expected_sha256} observed={observed_sha256}"
        )
    return observed_sha256


def _checkpoint_descriptor(
    label: str,
    path: Path,
    expected_sha256: str,
    checkpoint: dict,
) -> dict:
    observed_sha256 = _verify_checkpoint_file(path, label, expected_sha256)
    return {
        "label": label,
        "path": str(path),
        "sha256": observed_sha256,
        "metadata": {
            "num_edges": _json_scalar(checkpoint.get("num_edges")),
            "noise_level": _json_scalar(checkpoint.get("noise_level")),
            "epoch": _json_scalar(checkpoint.get("epoch")),
            "step": _json_scalar(checkpoint.get("step")),
        },
    }


def _validate_generated_sequence(
    generated: dict,
    expected_length: int,
    source_chain_id: str,
) -> None:
    sequence = generated.get("sequence")
    if (
        not isinstance(sequence, str)
        or len(sequence) != expected_length
        or not set(sequence) <= CANONICAL_AMINO_ACIDS
        or generated.get("sequence_sha256") != text_sha256(sequence)
    ):
        raise ContractError(f"invalid generated sequence: {source_chain_id}")
    try:
        designable = int(generated["designable_positions"])
        fixed = int(generated["fixed_missing_positions"])
        mutations = int(generated["mutation_count"])
        recovery = float(generated["sequence_recovery"])
        sampled_nll = float(generated["sampled_nll"])
        native_nll = float(generated["native_nll_same_order"])
        runtime_seconds = float(generated["runtime_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"invalid generation metrics: {source_chain_id}") from error
    if (
        designable <= 0
        or fixed < 0
        or designable + fixed != expected_length
        or not 0 <= mutations <= designable
        or not math.isclose(
            recovery,
            1.0 - mutations / float(designable),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not all(
            math.isfinite(value)
            for value in (recovery, sampled_nll, native_nll, runtime_seconds)
        )
        or runtime_seconds < 0.0
    ):
        raise ContractError(f"inconsistent generation metrics: {source_chain_id}")


def _protein_dict_from_payload(
    payload: dict,
    source_chain_id: str,
    expected_sequence: str,
) -> tuple[dict, dict, str]:
    import numpy as np

    if (
        payload.get("format") != "proteinmpnn.tar_shard.v2"
        or payload.get("target_chain_ids") != [source_chain_id]
    ):
        raise ContractError(f"ProteinMPNN payload target mismatch: {source_chain_id}")
    meta = payload.get("meta", {})
    chains = payload.get("chains", {})
    target_chain = meta.get("target_chain")
    chain_order = meta.get("chains")
    if (
        not isinstance(chain_order, list)
        or chain_order != list(chains)
        or target_chain not in chains
    ):
        raise ContractError(f"ProteinMPNN payload chain metadata mismatch: {source_chain_id}")
    if chains[target_chain].get("seq") != expected_sequence:
        raise ContractError(f"ProteinMPNN target sequence mismatch: {source_chain_id}")

    protein = {
        "name": source_chain_id,
        "num_of_chains": len(chain_order),
        "seq": "".join(chains[chain_id]["seq"] for chain_id in chain_order),
    }
    for chain_id in chain_order:
        chain = chains[chain_id]
        sequence = chain.get("seq")
        if not sequence or not set(sequence) <= CANONICAL_AMINO_ACIDS:
            raise ContractError(
                f"ProteinMPNN context sequence is noncanonical: {source_chain_id}/{chain_id}"
            )
        xyz = np.asarray(_to_numpy(chain["xyz"]), dtype=np.float32)
        mask = np.asarray(_to_numpy(chain["mask"]), dtype=bool)
        if xyz.shape != (len(sequence), 14, 3) or mask.shape != (len(sequence), 14):
            raise ContractError(
                f"ProteinMPNN context tensor shape mismatch: {source_chain_id}/{chain_id}"
            )
        backbone = xyz[:, :4, :].copy()
        expected_finite = mask[:, :4, None]
        if not np.array_equal(
            np.isfinite(backbone), np.broadcast_to(expected_finite, backbone.shape)
        ):
            raise ContractError(
                f"ProteinMPNN context coordinate mask mismatch: {source_chain_id}/{chain_id}"
            )
        protein[f"seq_chain_{chain_id}"] = sequence
        protein[f"coords_chain_{chain_id}"] = {
            f"N_chain_{chain_id}": backbone[:, 0, :],
            f"CA_chain_{chain_id}": backbone[:, 1, :],
            f"C_chain_{chain_id}": backbone[:, 2, :],
            f"O_chain_{chain_id}": backbone[:, 3, :],
        }
    chain_dict = {
        source_chain_id: (
            [target_chain],
            [chain_id for chain_id in chain_order if chain_id != target_chain],
        )
    }
    return protein, chain_dict, target_chain


class ProteinMPNNDesignBackend:
    """Thin binding to the pinned upstream ProteinMPNN inference implementation."""

    def __init__(
        self,
        repository_root: Path,
        checkpoint_path: Path,
        checkpoint: dict,
        device_name: str,
    ):
        try:
            import numpy as np
            import torch
        except Exception as error:
            raise ContractError(f"ProteinMPNN runtime dependency is missing: {error}") from error
        utility_root = repository_root / "repo"
        if not (utility_root / "protein_mpnn_utils.py").is_file():
            raise ContractError(f"ProteinMPNN utility source not found: {utility_root}")
        if str(utility_root) not in sys.path:
            sys.path.insert(0, str(utility_root))
        try:
            import protein_mpnn_utils
            from protein_mpnn_utils import ProteinMPNN, _scores, tied_featurize
        except Exception as error:
            raise ContractError(
                f"cannot import ProteinMPNN inference utilities: {error}"
            ) from error
        if Path(protein_mpnn_utils.__file__).resolve() != (
            utility_root / "protein_mpnn_utils.py"
        ).resolve():
            raise ContractError("loaded ProteinMPNN utilities from an unexpected repository")

        if device_name == "cuda":
            if not torch.cuda.is_available():
                raise ContractError("ProteinMPNN pilot requested CUDA but CUDA is unavailable")
            device = torch.device("cuda:0")
        elif device_name == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_edges = int(checkpoint.get("num_edges", 48))
        model = ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0,
            k_neighbors=num_edges,
        ).to(device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as error:
            raise ContractError(
                f"checkpoint is incompatible with upstream ProteinMPNN: {checkpoint_path}"
            ) from error
        self.model = model.eval()
        self.device = device
        self.np = np
        self.torch = torch
        self._scores = _scores
        self.tied_featurize = tied_featurize
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False

    def generate(
        self,
        payload: dict,
        source_chain_id: str,
        native_sequence: str,
        seed: int,
        temperature: float,
    ) -> dict:
        np = self.np
        torch = self.torch
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        protein, chain_dict, _ = _protein_dict_from_payload(
            payload, source_chain_id, native_sequence
        )
        features = self.tied_featurize(
            [protein],
            self.device,
            chain_dict,
            None,
            None,
            None,
            None,
            None,
            ca_only=False,
        )
        (
            X,
            S,
            mask,
            _,
            chain_M,
            chain_encoding_all,
            _,
            _,
            _,
            _,
            chain_M_pos,
            omit_AA_mask,
            residue_idx,
            _,
            _,
            pssm_coef,
            pssm_bias,
            pssm_log_odds_all,
            bias_by_res_all,
            _,
        ) = features
        omit_AAs_np = np.asarray([amino_acid == "X" for amino_acid in ALPHABET], dtype=np.float32)
        bias_AAs_np = np.zeros(len(ALPHABET), dtype=np.float32)
        pssm_log_odds_mask = (pssm_log_odds_all > 0.0).float()
        started = time.monotonic()
        with torch.inference_mode():
            decoding_noise = torch.randn(chain_M.shape, device=self.device)
            sample = self.model.sample(
                X,
                decoding_noise,
                S,
                chain_M,
                chain_encoding_all,
                residue_idx,
                mask=mask,
                temperature=temperature,
                omit_AAs_np=omit_AAs_np,
                bias_AAs_np=bias_AAs_np,
                chain_M_pos=chain_M_pos,
                omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef,
                pssm_bias=pssm_bias,
                pssm_multi=0.0,
                pssm_log_odds_flag=False,
                pssm_log_odds_mask=pssm_log_odds_mask,
                pssm_bias_flag=False,
                bias_by_res=bias_by_res_all,
            )
            sampled_S = sample["S"]
            design_mask = mask * chain_M * chain_M_pos
            sampled_log_probs = self.model(
                X,
                sampled_S,
                mask,
                design_mask,
                residue_idx,
                chain_encoding_all,
                decoding_noise,
                use_input_decoding_order=True,
                decoding_order=sample["decoding_order"],
            )
            native_log_probs = self.model(
                X,
                S,
                mask,
                design_mask,
                residue_idx,
                chain_encoding_all,
                decoding_noise,
                use_input_decoding_order=True,
                decoding_order=sample["decoding_order"],
            )
            sampled_nll = float(
                self._scores(sampled_S, sampled_log_probs, design_mask)[0].cpu()
            )
            native_nll = float(self._scores(S, native_log_probs, design_mask)[0].cpu())

        target_mask = chain_M[0] > 0.5
        target_design_mask = design_mask[0][target_mask] > 0.5
        sampled_target = sampled_S[0][target_mask].detach().cpu().tolist()
        native_target = S[0][target_mask].detach().cpu().tolist()
        designed_sequence = "".join(ALPHABET[index] for index in sampled_target)
        observed_native_sequence = "".join(ALPHABET[index] for index in native_target)
        if observed_native_sequence != native_sequence:
            raise ContractError(
                f"ProteinMPNN target extraction changed sequence: {source_chain_id}"
            )
        if len(designed_sequence) != len(native_sequence) or not set(
            designed_sequence
        ) <= CANONICAL_AMINO_ACIDS:
            raise ContractError(f"ProteinMPNN emitted an invalid sequence: {source_chain_id}")
        designable = target_design_mask.detach().cpu().tolist()
        mutation_count = sum(
            bool(is_designable) and native != designed
            for native, designed, is_designable in zip(
                native_sequence, designed_sequence, designable
            )
        )
        designable_positions = sum(bool(value) for value in designable)
        if designable_positions <= 0:
            raise ContractError(
                f"ProteinMPNN target has no designable positions: {source_chain_id}"
            )
        return {
            "sequence": designed_sequence,
            "sequence_sha256": text_sha256(designed_sequence),
            "designable_positions": designable_positions,
            "fixed_missing_positions": len(native_sequence) - designable_positions,
            "mutation_count": mutation_count,
            "sequence_recovery": 1.0 - mutation_count / float(designable_positions),
            "sampled_nll": sampled_nll,
            "native_nll_same_order": native_nll,
            "runtime_seconds": time.monotonic() - started,
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def _initialize_output(output_dir: Path, manifest: dict) -> None:
    manifest_path = output_dir / "pilot-manifest.json"
    if output_dir.exists():
        if not manifest_path.is_file():
            raise ContractError(f"existing ProteinMPNN pilot has no manifest: {output_dir}")
        existing = read_json(manifest_path)
        if (
            existing.get("pilot_identity") != manifest["pilot_identity"]
            or existing.get("pilot_identity") != _identity(existing, "pilot_identity")
        ):
            raise ContractError("existing ProteinMPNN pilot has different inputs")
        return
    output_dir.mkdir(parents=True)
    write_json_atomic(manifest_path, manifest)


def load_generated_pilot(pilot_dir: str | Path) -> tuple[dict, list[dict], dict]:
    root = Path(pilot_dir).expanduser().resolve()
    manifest = read_json(root / "pilot-manifest.json")
    if (
        manifest.get("schema_version") != "protein-mrna.proteinmpnn-refold-pilot.v1"
        or manifest.get("pilot_identity") != _identity(manifest, "pilot_identity")
    ):
        raise ContractError("ProteinMPNN pilot manifest identity mismatch")
    summary = read_json(root / "generation-summary.json")
    designs_path = root / "designs.jsonl"
    if (
        summary.get("schema_version")
        != "protein-mrna.proteinmpnn-generation-summary.v1"
        or summary.get("pilot_identity") != manifest["pilot_identity"]
        or summary.get("status") != "passed"
        or summary.get("designs_sha256") != sha256_file(designs_path)
    ):
        raise ContractError("ProteinMPNN generation summary is invalid")

    selection = manifest.get("selection", {}).get("records", [])
    models = manifest.get("models", [])
    seeds = manifest.get("sampling", {}).get("seeds", [])
    temperature = manifest.get("sampling", {}).get("temperature")
    if (
        not isinstance(selection, list)
        or len(selection) != 4
        or not all(isinstance(record, dict) for record in selection)
        or not isinstance(models, list)
        or not all(isinstance(model, dict) for model in models)
        or {model.get("label") for model in models}
        != {"official-v48-020", "stage2a"}
        or len(models) != 2
        or not isinstance(seeds, list)
        or not seeds
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0
            for seed in seeds
        )
        or len(seeds) != len(set(seeds))
        or not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ContractError("ProteinMPNN pilot design matrix is invalid")
    selection_by_id = {
        record.get("benchmark_record_id"): record for record in selection
    }
    model_by_label = {model.get("label"): model for model in models}
    if (
        None in selection_by_id
        or len(selection_by_id) != len(selection)
        or None in model_by_label
        or len(model_by_label) != len(models)
    ):
        raise ContractError("ProteinMPNN pilot matrix contains duplicate identities")

    designs = _read_jsonl(designs_path)
    expected_count = len(selection) * len(models) * len(seeds)
    try:
        declared_counts = (
            int(summary["designs"]),
            int(summary["backbones"]),
            int(summary["models"]),
            int(summary["seeds"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("ProteinMPNN generation counts are invalid") from error
    if (
        len(designs) != expected_count
        or declared_counts
        != (expected_count, len(selection), len(models), len(seeds))
    ):
        raise ContractError("ProteinMPNN designs count differs from summary")
    seen = set()
    observed_matrix = set()
    for design in designs:
        design_id = design.get("design_id")
        record_id = design.get("benchmark_record_id")
        model_label = design.get("model_label")
        seed = design.get("seed")
        selected = selection_by_id.get(record_id)
        model = model_by_label.get(model_label)
        expected_design_id = f"{record_id}--{model_label}--seed-{seed}"
        try:
            design_length = int(design["sequence_length"])
            selected_length = int(selected["sequence_length"])
            design_temperature = float(design["temperature"])
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(
                f"invalid ProteinMPNN design record: {design_id}"
            ) from error
        if (
            design.get("schema_version") != "protein-mrna.proteinmpnn-design.v1"
            or design.get("design_identity") != _identity(design, "design_identity")
            or design.get("pilot_identity") != manifest["pilot_identity"]
            or design_id in seen
            or selected is None
            or model is None
            or seed not in seeds
            or design_id != expected_design_id
            or design.get("selection_role") != selected.get("selection_role")
            or design.get("source_chain_id") != selected.get("source_chain_id")
            or design_length != selected_length
            or design.get("native_sequence_sha256")
            != selected.get("native_sequence_sha256")
            or design.get("checkpoint_sha256") != model.get("sha256")
            or design_temperature != float(temperature)
        ):
            raise ContractError(f"invalid ProteinMPNN design record: {design_id}")
        _validate_generated_sequence(
            design,
            selected_length,
            str(selected["source_chain_id"]),
        )
        seen.add(design_id)
        observed_matrix.add((record_id, model_label, seed))
    expected_matrix = {
        (record_id, model_label, seed)
        for record_id in selection_by_id
        for model_label in model_by_label
        for seed in seeds
    }
    if observed_matrix != expected_matrix:
        raise ContractError("ProteinMPNN designs do not cover the declared matrix")

    recovery_summary = summary.get("mean_sequence_recovery_by_model", {})
    for label in model_by_label:
        observed_mean = sum(
            float(row["sequence_recovery"])
            for row in designs
            if row["model_label"] == label
        ) / sum(row["model_label"] == label for row in designs)
        try:
            recorded_mean = float(recovery_summary[label])
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("ProteinMPNN recovery summary is invalid") from error
        if not math.isclose(
            observed_mean, recorded_mean, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ContractError("ProteinMPNN recovery summary differs from designs")
    return manifest, designs, summary


def generate_paired_design_pilot(
    suite_path: str | Path,
    native_agreement_dir: str | Path,
    native_prediction_run: str | Path,
    dataset_dir: str | Path,
    official_checkpoint: str | Path,
    stage2a_checkpoint: str | Path,
    output_dir: str | Path,
    metrics_runtime_root: str | Path,
    repository_root: str | Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    temperature: float = DEFAULT_TEMPERATURE,
    device: str = "auto",
    payload_loader: Callable[[Path, dict], dict] = _load_payload,
    backend_factory: Callable = ProteinMPNNDesignBackend,
    checkpoint_loader: Callable[[Path], dict] = _load_checkpoint,
    expected_checkpoint_sha256: dict[str, str] | None = None,
    metrics_runtime_document: dict | None = None,
) -> dict:
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(isinstance(seed, bool) or seed <= 0 for seed in seeds)
    ):
        raise ContractError("ProteinMPNN pilot seeds must be distinct positive integers")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ContractError("ProteinMPNN sampling temperature must be positive")
    suite_document_path = Path(suite_path).expanduser().resolve()
    verify_benchmark_suite_files(suite_document_path)
    suite = read_json(suite_document_path)
    if suite["source"]["split"] != "valid":
        raise ContractError("ProteinMPNN pilot requires the fixed valid benchmark")
    native_prediction_dir = Path(native_prediction_run).expanduser().resolve()
    verify_esmfold2_benchmark_run(
        suite_document_path, native_prediction_dir, mode="full"
    )
    agreement_dir = Path(native_agreement_dir).expanduser().resolve()
    agreement_summary, agreement_records = _validate_native_agreement(
        agreement_dir, suite
    )
    selected = select_pilot_backbones(agreement_records)
    selected_ids = {row["benchmark_record_id"] for row in selected}
    suite_by_id = {row["benchmark_record_id"]: row for row in suite["records"]}
    if not selected_ids <= set(suite_by_id):
        raise ContractError("selected ProteinMPNN pilot IDs are outside the benchmark")
    for selection in selected:
        benchmark_record = suite_by_id[selection["benchmark_record_id"]]
        selection["native_sequence_sha256"] = benchmark_record["sequence_sha256"]

    source_dir = Path(dataset_dir).expanduser().resolve()
    dataset = _verify_dataset(source_dir, suite)
    index_rows = _selected_index_rows(
        source_dir,
        {suite_by_id[record_id]["source_chain_id"] for record_id in selected_ids},
    )
    _validate_selected_index_rows(source_dir, index_rows, dataset)
    if metrics_runtime_document is None:
        runtime = load_metrics_runtime_manifest(metrics_runtime_root)
    else:
        runtime = metrics_runtime_document
        _validate_metrics_runtime_document(runtime)

    official_path = Path(official_checkpoint).expanduser().resolve()
    stage2a_path = Path(stage2a_checkpoint).expanduser().resolve()
    expected_hashes = expected_checkpoint_sha256 or {
        "official-v48-020": OFFICIAL_CHECKPOINT_SHA256,
        "stage2a": STAGE2A_CHECKPOINT_SHA256,
    }
    if set(expected_hashes) != {"official-v48-020", "stage2a"}:
        raise ContractError("ProteinMPNN pilot checkpoint hash map is incomplete")
    _verify_checkpoint_file(
        official_path, "official-v48-020", expected_hashes["official-v48-020"]
    )
    _verify_checkpoint_file(stage2a_path, "stage2a", expected_hashes["stage2a"])
    checkpoint_documents = {
        "official-v48-020": checkpoint_loader(official_path),
        "stage2a": checkpoint_loader(stage2a_path),
    }
    model_descriptors = [
        _checkpoint_descriptor(
            "official-v48-020",
            official_path,
            expected_hashes["official-v48-020"],
            checkpoint_documents["official-v48-020"],
        ),
        _checkpoint_descriptor(
            "stage2a",
            stage2a_path,
            expected_hashes["stage2a"],
            checkpoint_documents["stage2a"],
        ),
    ]
    repository = Path(repository_root).expanduser().resolve()
    manifest = {
        "schema_version": "protein-mrna.proteinmpnn-refold-pilot.v1",
        "pilot_identity": "pending",
        "created_at_utc": utc_now(),
        "benchmark": {
            "benchmark_id": suite["benchmark_id"],
            "suite_path": str(suite_document_path),
            "suite_sha256": sha256_file(suite_document_path),
            "source_split": "valid",
        },
        "native_agreement": {
            "directory": str(agreement_dir),
            "evaluation_identity": agreement_summary["evaluation_identity"],
            "summary_sha256": sha256_file(agreement_dir / "summary.json"),
            "records_sha256": sha256_file(agreement_dir / "records.jsonl"),
        },
        "native_prediction_run": str(native_prediction_dir),
        "native_dataset": dataset,
        "metrics_runtime_identity": runtime["runtime_identity"],
        "repository_root": str(repository),
        "models": model_descriptors,
        "selection": {
            "policy": "lowest_lddt__lowest_resolved_tm_cov095__longest__highest_lddt",
            "records": selected,
        },
        "sampling": {
            "temperature": temperature,
            "seeds": list(seeds),
            "samples_per_model_backbone": len(seeds),
            "backbone_noise": 0.0,
            "omit_amino_acids": ["X"],
            "paired_seed_policy": True,
            "context": "complete_payload_assembly_design_target_chain_only",
        },
        "implementation": {
            "module": "protein_mrna_pipeline.proteinmpnn_pilot",
            "module_sha256": sha256_file(Path(__file__).resolve()),
            "upstream_inference_source_sha256": sha256_file(
                repository / "repo/protein_mpnn_utils.py"
            ),
        },
        "limitations": [
            "Engineering valid-split pilot; no checkpoint or parameter selection is allowed.",
            "Only four deliberately stratified backbones are included.",
            "Missing target-backbone positions remain fixed to the native amino acid.",
        ],
    }
    manifest["pilot_identity"] = _identity(manifest, "pilot_identity")
    destination = Path(output_dir).expanduser().resolve()
    _initialize_output(destination, manifest)
    summary_path = destination / "generation-summary.json"
    designs_path = destination / "designs.jsonl"
    if summary_path.is_file() and designs_path.is_file():
        _, _, existing_summary = load_generated_pilot(destination)
        return existing_summary

    payloads = {}
    for selection in selected:
        benchmark_record = suite_by_id[selection["benchmark_record_id"]]
        source_chain_id = benchmark_record["source_chain_id"]
        index_row = index_rows[source_chain_id]
        if int(index_row.get("sequence_length", -1)) != benchmark_record["length"]:
            raise ContractError(f"pilot native index length mismatch: {source_chain_id}")
        payloads[selection["benchmark_record_id"]] = payload_loader(
            source_dir, index_row
        )

    designs = []
    for model_descriptor in model_descriptors:
        label = model_descriptor["label"]
        checkpoint_path = Path(model_descriptor["path"])
        backend = backend_factory(
            repository,
            checkpoint_path,
            checkpoint_documents[label],
            device,
        )
        try:
            for selection in selected:
                record_id = selection["benchmark_record_id"]
                benchmark_record = suite_by_id[record_id]
                for seed in seeds:
                    print(
                        f"generating {record_id} model={label} seed={seed} "
                        f"length={benchmark_record['length']}",
                        flush=True,
                    )
                    generated = backend.generate(
                        payloads[record_id],
                        benchmark_record["source_chain_id"],
                        benchmark_record["sequence"],
                        seed,
                        temperature,
                    )
                    _validate_generated_sequence(
                        generated,
                        benchmark_record["length"],
                        benchmark_record["source_chain_id"],
                    )
                    design_id = f"{record_id}--{label}--seed-{seed}"
                    design = {
                        "schema_version": "protein-mrna.proteinmpnn-design.v1",
                        "design_identity": "pending",
                        "pilot_identity": manifest["pilot_identity"],
                        "design_id": design_id,
                        "benchmark_record_id": record_id,
                        "selection_role": selection["selection_role"],
                        "source_chain_id": benchmark_record["source_chain_id"],
                        "model_label": label,
                        "checkpoint_sha256": model_descriptor["sha256"],
                        "seed": seed,
                        "temperature": temperature,
                        "native_sequence_sha256": benchmark_record["sequence_sha256"],
                        "sequence_length": benchmark_record["length"],
                        **generated,
                    }
                    design["design_identity"] = _identity(
                        design, "design_identity"
                    )
                    designs.append(design)
        finally:
            close = getattr(backend, "close", None)
            if close is not None:
                close()

    expected_count = len(selected) * len(model_descriptors) * len(seeds)
    if len(designs) != expected_count:
        raise ContractError(
            f"ProteinMPNN pilot design count mismatch: {len(designs)} != {expected_count}"
        )
    write_json_atomic(
        destination / "selected-backbones.json",
        {
            "schema_version": "protein-mrna.proteinmpnn-pilot-selection.v1",
            "pilot_identity": manifest["pilot_identity"],
            "records": selected,
        },
    )
    write_jsonl_atomic(designs_path, designs)
    summary = {
        "schema_version": "protein-mrna.proteinmpnn-generation-summary.v1",
        "pilot_identity": manifest["pilot_identity"],
        "status": "passed",
        "updated_at_utc": utc_now(),
        "backbones": len(selected),
        "models": len(model_descriptors),
        "seeds": len(seeds),
        "designs": len(designs),
        "designs_sha256": sha256_file(designs_path),
        "mean_sequence_recovery_by_model": {
            label: sum(
                float(row["sequence_recovery"])
                for row in designs
                if row["model_label"] == label
            )
            / sum(row["model_label"] == label for row in designs)
            for label in ("official-v48-020", "stage2a")
        },
    }
    write_json_atomic(summary_path, summary)
    return summary
