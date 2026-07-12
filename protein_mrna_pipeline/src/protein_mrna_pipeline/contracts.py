"""JSON contracts, canonical identities, and cross-field semantic checks."""

from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from jsonschema import Draft202012Validator, FormatChecker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
SCHEMA_FILES = {
    "target": "target-package.schema.json",
    "work-item": "work-item.schema.json",
    "tool-result": "tool-result.schema.json",
    "candidate": "candidate-record.schema.json",
    "run-manifest": "run-manifest.schema.json",
}

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class ContractError(ValueError):
    """Raised when a document violates its structural or semantic contract."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContractError(f"value is not canonical JSON: {error}") from error


def document_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON value: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def read_json(path: str | Path) -> dict:
    document_path = Path(path)
    try:
        value = json.loads(
            document_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot read JSON document {document_path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {document_path}")
    return value


def load_schema(kind: str) -> dict:
    try:
        schema_path = SCHEMA_DIR / SCHEMA_FILES[kind]
    except KeyError as error:
        raise ContractError(f"unknown contract kind: {kind}") from error
    schema = read_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        raise ContractError(f"invalid bundled schema {schema_path}: {error}") from error
    return schema


def validate_schema(document: dict, kind: str) -> None:
    canonical_json_bytes(document)
    validator = Draft202012Validator(
        load_schema(kind),
        format_checker=FormatChecker(),
    )
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if not errors:
        return
    messages = []
    for error in errors[:20]:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(f"{location}: {error.message}")
    if len(errors) > len(messages):
        messages.append(f"... {len(errors) - len(messages)} additional errors")
    raise ContractError(f"{kind} schema validation failed:\n" + "\n".join(messages))


def _unique_ids(records: Iterable[dict], field: str, label: str) -> set[str]:
    values = [str(record[field]) for record in records]
    if len(values) != len(set(values)):
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        raise ContractError(f"duplicate {label} IDs: {duplicates}")
    return set(values)


def validate_identifier(value: str, label: str) -> None:
    if ID_PATTERN.fullmatch(value) is None:
        raise ContractError(f"invalid {label}: {value}")


def _check_region(region: dict, domain_lengths: dict[str, int], label: str) -> None:
    domain_id = region["domain_id"]
    if domain_id not in domain_lengths:
        raise ContractError(f"{label} references unknown domain: {domain_id}")
    start = int(region["start"])
    end = int(region["end"])
    if start > end:
        raise ContractError(f"{label} has start after end: {domain_id}:{start}-{end}")
    if end > domain_lengths[domain_id]:
        raise ContractError(
            f"{label} exceeds {domain_id} length {domain_lengths[domain_id]}: "
            f"{start}-{end}"
        )


def _regions_overlap(left: dict, right: dict) -> bool:
    return (
        left["domain_id"] == right["domain_id"]
        and max(int(left["start"]), int(right["start"]))
        <= min(int(left["end"]), int(right["end"]))
    )


def validate_target(document: dict) -> None:
    validate_schema(document, "target")
    domains = document["domains"]
    domain_ids = _unique_ids(domains, "domain_id", "domain")
    domain_lengths = {
        domain["domain_id"]: len(domain["amino_acid_sequence"])
        for domain in domains
    }

    immutable_regions = []
    for domain in domains:
        for region in domain["immutable_regions"]:
            if region["domain_id"] != domain["domain_id"]:
                raise ContractError(
                    "domain immutable region must reference its containing domain: "
                    f"{domain['domain_id']} != {region['domain_id']}"
                )
            _check_region(region, domain_lengths, "immutable region")
            immutable_regions.append(region)

    constraints = document["design_constraints"]
    fixed_regions = constraints["fixed_regions"]
    mutable_regions = constraints["mutable_regions"]
    for label, regions in (
        ("fixed region", fixed_regions),
        ("mutable region", mutable_regions),
    ):
        for region in regions:
            _check_region(region, domain_lengths, label)

    protected_regions = immutable_regions + fixed_regions
    for mutable in mutable_regions:
        for protected in protected_regions:
            if _regions_overlap(mutable, protected):
                raise ContractError(
                    "mutable region overlaps fixed/immutable region: "
                    f"{mutable['domain_id']}:{mutable['start']}-{mutable['end']}"
                )

    architecture = document["architecture"]
    orders = architecture["allowed_domain_orders"]
    for order in orders:
        if len(order) != len(set(order)):
            raise ContractError(f"domain order contains duplicates: {order}")
        if set(order) != domain_ids:
            raise ContractError(
                f"domain order must contain every domain exactly once: {order}"
            )

    linkers = architecture["linker_options"]
    _unique_ids(linkers, "linker_id", "linker")
    adjacent_pairs = {
        (left, right)
        for order in orders
        for left, right in zip(order, order[1:])
    }
    linkers_by_pair: dict[tuple[str, str], list[dict]] = {}
    for linker in linkers:
        left, right = linker["between"]
        if left not in domain_ids or right not in domain_ids:
            raise ContractError(f"linker references unknown domain pair: {left}, {right}")
        if (left, right) not in adjacent_pairs:
            raise ContractError(
                f"linker pair is never adjacent in an allowed order: {left}, {right}"
            )
        minimum = int(linker["min_length"])
        maximum = int(linker["max_length"])
        if minimum > maximum:
            raise ContractError(
                f"linker min_length exceeds max_length: {linker['linker_id']}"
            )
        for sequence in linker.get("allowed_sequences", []):
            if not minimum <= len(sequence) <= maximum:
                raise ContractError(
                    f"allowed linker sequence length is outside bounds: {linker['linker_id']}"
                )
        linkers_by_pair.setdefault((left, right), []).append(linker)

    allow_direct = bool(architecture["allow_direct_fusion"])
    maximum_total_length = int(architecture["maximum_total_length"])
    for order in orders:
        minimum_total = sum(domain_lengths[domain_id] for domain_id in order)
        for pair in zip(order, order[1:]):
            options = linkers_by_pair.get(pair, [])
            if not options and not allow_direct:
                raise ContractError(
                    f"no linker option covers adjacent pair: {pair[0]}, {pair[1]}"
                )
            if options and not allow_direct:
                minimum_total += min(int(option["min_length"]) for option in options)
        if minimum_total > maximum_total_length:
            raise ContractError(
                f"minimum architecture length {minimum_total} exceeds maximum "
                f"{maximum_total_length} for order {order}"
            )


def target_sha256(document: dict) -> str:
    validate_target(document)
    return document_sha256(document)


def work_identity(document: dict) -> dict:
    return {
        key: value
        for key, value in document.items()
        if key not in {"$schema", "work_id", "created_at_utc"}
    }


def derive_work_id(document: dict) -> str:
    stage = str(document.get("stage", "work"))
    identity_document = work_identity(document)
    return f"{stage}-{document_sha256(identity_document)[:24]}"


def prepare_work_item(document: dict, created_at_utc: str) -> dict:
    prepared = copy.deepcopy(document)
    prepared.setdefault("created_at_utc", created_at_utc)
    expected_work_id = derive_work_id(prepared)
    existing_work_id = prepared.get("work_id")
    if existing_work_id is not None and existing_work_id != expected_work_id:
        raise ContractError(
            f"work_id does not match canonical payload: {existing_work_id} != {expected_work_id}"
        )
    prepared["work_id"] = expected_work_id
    validate_schema(prepared, "work-item")
    return prepared


def validate_work_item(document: dict) -> None:
    validate_schema(document, "work-item")
    if "work_id" not in document:
        raise ContractError("work item must have a derived work_id before execution")
    expected = derive_work_id(document)
    if document["work_id"] != expected:
        raise ContractError(
            f"work_id does not match canonical payload: {document['work_id']} != {expected}"
        )


def _parse_time(value: str, label: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(f"invalid {label}: {value}") from error


def validate_tool_result(document: dict) -> None:
    validate_schema(document, "tool-result")
    started = _parse_time(document["started_at_utc"], "started_at_utc")
    finished = _parse_time(document["finished_at_utc"], "finished_at_utc")
    if finished < started:
        raise ContractError("tool result finished before it started")


def translate_cds(cds_sequence: str) -> str:
    if len(cds_sequence) % 3 != 0:
        raise ContractError("candidate CDS length must be divisible by three")
    amino_acids = []
    for offset in range(0, len(cds_sequence), 3):
        codon = cds_sequence[offset : offset + 3]
        amino_acid = CODON_TABLE.get(codon, "X")
        if amino_acid == "*":
            if offset != len(cds_sequence) - 3:
                raise ContractError(
                    "candidate CDS contains an internal stop at codon "
                    f"{offset // 3 + 1}"
                )
            break
        amino_acids.append(amino_acid)
    return "".join(amino_acids)


def validate_candidate(document: dict) -> None:
    validate_schema(document, "candidate")
    mrna = document.get("mrna")
    if mrna is not None:
        translated = translate_cds(mrna["cds_sequence"])
        if (
            mrna["translation_check"]
            != document["hard_checks"]["translation_preservation"]
        ):
            raise ContractError("mRNA translation check disagrees with candidate hard checks")
        if mrna["translation_check"] == "passed":
            if translated != document["protein_sequence"]:
                raise ContractError("candidate CDS does not translate to the candidate protein")
            expected = text_sha256(document["protein_sequence"])
            if mrna["translated_protein_sha256"] != expected:
                raise ContractError(
                    "passed translation check does not match the candidate protein hash"
                )
    elif document["hard_checks"]["translation_preservation"] != "not_run":
        raise ContractError("candidate without mRNA must have translation_preservation=not_run")
    if document["decision"]["status"] == "retained":
        required_checks = ["immutable_residues", "maximum_length", "safety"]
        if mrna is not None:
            required_checks.append("translation_preservation")
        failed = [
            check
            for check in required_checks
            if document["hard_checks"][check] != "passed"
        ]
        if failed:
            raise ContractError(f"retained candidate has incomplete hard checks: {failed}")


def validate_run_manifest(document: dict) -> None:
    validate_schema(document, "run-manifest")


VALIDATORS = {
    "target": validate_target,
    "work-item": validate_work_item,
    "tool-result": validate_tool_result,
    "candidate": validate_candidate,
    "run-manifest": validate_run_manifest,
}


def validate_document(document: dict, kind: str) -> None:
    try:
        validator = VALIDATORS[kind]
    except KeyError as error:
        raise ContractError(f"unknown contract kind: {kind}") from error
    validator(document)
