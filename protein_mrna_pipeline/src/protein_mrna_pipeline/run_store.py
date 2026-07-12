"""Atomic run creation and a transactional, lease-based work queue."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .contracts import (
    ContractError,
    PROJECT_ROOT,
    canonical_json_bytes,
    prepare_work_item,
    read_json,
    target_sha256,
    validate_run_manifest,
    validate_identifier,
    validate_tool_result,
    validate_work_item,
    work_identity,
)


QUEUE_STATES = ("pending", "running", "completed", "failed")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(canonical_json_bytes(value) + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(row) + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_identity() -> tuple[str | None, bool]:
    configured_root = os.environ.get("PROTEIN_MRNA_PIPELINE_REPOSITORY")
    candidates = [Path(configured_root)] if configured_root else []
    candidates.extend((Path.cwd(), PROJECT_ROOT.parent))
    visited = set()
    for candidate in candidates:
        try:
            repository_root = Path(
                subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=candidate,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ).resolve()
            if repository_root in visited:
                continue
            visited.add(repository_root)
            if not (repository_root / "protein_mrna_pipeline/pyproject.toml").is_file():
                continue
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repository_root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
            return revision, dirty
        except (OSError, subprocess.CalledProcessError):
            continue
    return None, True


def initialize_run(
    target_path: str | Path,
    run_dir: str | Path,
    allow_unreviewed: bool = False,
) -> dict:
    target = read_json(target_path)
    target_hash = target_sha256(target)
    safety_status = target["safety"]["status"]
    if safety_status == "denied":
        raise ContractError("a denied target cannot initialize a run")
    override_used = safety_status != "approved"
    if override_used and not allow_unreviewed:
        raise ContractError(
            f"target safety status is {safety_status}; an approved review is required"
        )

    destination = Path(run_dir).expanduser().resolve()
    if destination.exists():
        raise ContractError(f"run directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ContractError(f"temporary run directory already exists: {temporary}")

    revision, dirty = _git_identity()
    run_id = (
        f"{target['target_id'][:80]}-{target_hash[:12]}-{uuid.uuid4().hex[:12]}"
    )
    manifest = {
        "schema_version": "protein-mrna.run-manifest.v1",
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "target": {
            "target_id": target["target_id"],
            "revision": target["revision"],
            "sha256": target_hash,
            "path": "inputs/target-package.json",
        },
        "pipeline": {
            "version": __version__,
            "git_commit": revision,
            "git_dirty": dirty,
        },
        "safety_gate": {
            "target_status": safety_status,
            "override_used": override_used,
        },
        "queue": {"backend": "sqlite", "path": "queue.sqlite3"},
    }
    validate_run_manifest(manifest)

    try:
        (temporary / "inputs").mkdir(parents=True)
        (temporary / "work").mkdir()
        (temporary / "artifacts").mkdir()
        (temporary / "tables").mkdir()
        write_json_atomic(temporary / "inputs/target-package.json", target)
        write_json_atomic(temporary / "run-manifest.json", manifest)
        RunStore(temporary).initialize_database()
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


class RunStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.database_path = self.run_dir / "queue.sqlite3"
        self.manifest_path = self.run_dir / "run-manifest.json"

    def _connect(self, *, allow_create: bool = False) -> sqlite3.Connection:
        if not allow_create and not self.database_path.is_file():
            raise ContractError(f"queue database not found: {self.database_path}")
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def initialize_database(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self._connect(allow_create=True) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS work_items (
                    work_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'completed', 'failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    lease_expires_at REAL,
                    result_json TEXT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS work_items_claim_idx
                    ON work_items(state, priority DESC, created_at_utc, work_id);
                CREATE TABLE IF NOT EXISTS attempts (
                    work_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    worker_id TEXT NOT NULL,
                    claimed_at_utc TEXT NOT NULL,
                    lease_expires_at REAL NOT NULL,
                    finished_at_utc TEXT,
                    status TEXT NOT NULL CHECK (
                        status IN ('running', 'succeeded', 'failed', 'retryable', 'lease_expired')
                    ),
                    result_json TEXT,
                    PRIMARY KEY (work_id, attempt),
                    FOREIGN KEY (work_id) REFERENCES work_items(work_id)
                );
                PRAGMA user_version=1;
                """
            )

    def manifest(self) -> dict:
        manifest = read_json(self.manifest_path)
        validate_run_manifest(manifest)
        return manifest

    def enqueue(self, document: dict) -> tuple[dict, bool]:
        prepared = prepare_work_item(document, utc_now())
        validate_work_item(prepared)
        manifest = self.manifest()
        if prepared["target_id"] != manifest["target"]["target_id"]:
            raise ContractError("work item target_id does not match the run manifest")
        if prepared["target_sha256"] != manifest["target"]["sha256"]:
            raise ContractError("work item target_sha256 does not match the run manifest")
        if (
            manifest["safety_gate"]["override_used"]
            and prepared["execution_class"] != "smoke"
        ):
            raise ContractError("an unreviewed target override can enqueue smoke work only")

        payload_json = canonical_json_bytes(prepared).decode("utf-8")
        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload_json FROM work_items WHERE work_id = ?",
                (prepared["work_id"],),
            ).fetchone()
            if existing is not None:
                existing_payload = json.loads(existing["payload_json"])
                existing_identity = canonical_json_bytes(work_identity(existing_payload))
                prepared_identity = canonical_json_bytes(work_identity(prepared))
                if existing_identity != prepared_identity:
                    connection.rollback()
                    raise ContractError(
                        f"work_id collision with a different payload: {prepared['work_id']}"
                    )
                connection.commit()
                return existing_payload, False
            connection.execute(
                """
                INSERT INTO work_items (
                    work_id, stage, priority, max_attempts, payload_json, state,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    prepared["work_id"],
                    prepared["stage"],
                    prepared["priority"],
                    prepared["max_attempts"],
                    payload_json,
                    now,
                    now,
                ),
            )
            connection.commit()
        return prepared, True

    def claim(self, worker_id: str, lease_seconds: int) -> dict | None:
        validate_identifier(worker_id, "worker_id")
        if lease_seconds < 1:
            raise ContractError("lease_seconds must be positive")
        now_epoch = time.time()
        now_utc = utc_now()
        lease_expires = now_epoch + lease_seconds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exhausted = connection.execute(
                """
                SELECT work_id, attempts FROM work_items
                WHERE state = 'running' AND lease_expires_at <= ?
                  AND attempts >= max_attempts
                """,
                (now_epoch,),
            ).fetchall()
            for expired in exhausted:
                connection.execute(
                    """
                    UPDATE attempts
                    SET status = 'lease_expired', finished_at_utc = ?
                    WHERE work_id = ? AND attempt = ? AND status = 'running'
                    """,
                    (now_utc, expired["work_id"], expired["attempts"]),
                )
                connection.execute(
                    """
                    UPDATE work_items
                    SET state = 'failed', worker_id = NULL, lease_expires_at = NULL,
                        updated_at_utc = ?
                    WHERE work_id = ?
                    """,
                    (now_utc, expired["work_id"]),
                )
            connection.execute(
                """
                UPDATE work_items
                SET state = 'failed', updated_at_utc = ?
                WHERE state = 'pending' AND attempts >= max_attempts
                """,
                (now_utc,),
            )
            row = connection.execute(
                """
                SELECT * FROM work_items
                WHERE attempts < max_attempts
                  AND (state = 'pending'
                       OR (state = 'running' AND lease_expires_at <= ?))
                ORDER BY priority DESC, created_at_utc, work_id
                LIMIT 1
                """,
                (now_epoch,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            if row["state"] == "running":
                connection.execute(
                    """
                    UPDATE attempts
                    SET status = 'lease_expired', finished_at_utc = ?
                    WHERE work_id = ? AND attempt = ? AND status = 'running'
                    """,
                    (now_utc, row["work_id"], row["attempts"]),
                )
            attempt = int(row["attempts"]) + 1
            connection.execute(
                """
                UPDATE work_items
                SET state = 'running', attempts = ?, worker_id = ?,
                    lease_expires_at = ?, updated_at_utc = ?
                WHERE work_id = ?
                """,
                (attempt, worker_id, lease_expires, now_utc, row["work_id"]),
            )
            connection.execute(
                """
                INSERT INTO attempts (
                    work_id, attempt, worker_id, claimed_at_utc,
                    lease_expires_at, status
                ) VALUES (?, ?, ?, ?, ?, 'running')
                """,
                (row["work_id"], attempt, worker_id, now_utc, lease_expires),
            )
            connection.commit()

        work_dir = self.run_dir / "work" / row["work_id"] / f"attempt-{attempt}"
        work_dir.mkdir(parents=True, exist_ok=True)
        return {
            "work_item": json.loads(row["payload_json"]),
            "queue": {
                "attempt": attempt,
                "worker_id": worker_id,
                "claimed_at_utc": now_utc,
                "lease_expires_at_epoch": lease_expires,
                "work_dir": str(work_dir.relative_to(self.run_dir)),
            },
        }

    def renew(
        self,
        work_id: str,
        worker_id: str,
        attempt: int,
        lease_seconds: int,
    ) -> dict:
        validate_identifier(work_id, "work_id")
        validate_identifier(worker_id, "worker_id")
        if lease_seconds < 1:
            raise ContractError("lease_seconds must be positive")
        now_epoch = time.time()
        lease_expires = now_epoch + lease_seconds
        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._active_lease(
                connection,
                work_id,
                worker_id,
                attempt,
                now_epoch,
            )
            connection.execute(
                """
                UPDATE work_items
                SET lease_expires_at = ?, updated_at_utc = ?
                WHERE work_id = ?
                """,
                (lease_expires, now, work_id),
            )
            connection.execute(
                """
                UPDATE attempts
                SET lease_expires_at = ?
                WHERE work_id = ? AND attempt = ? AND status = 'running'
                """,
                (lease_expires, work_id, attempt),
            )
            connection.commit()
        return {
            "work_id": work_id,
            "worker_id": worker_id,
            "attempt": attempt,
            "renewed_at_utc": now,
            "lease_expires_at_epoch": lease_expires,
        }

    @staticmethod
    def _active_lease(
        connection: sqlite3.Connection,
        work_id: str,
        worker_id: str,
        attempt: int,
        now_epoch: float,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM work_items WHERE work_id = ?",
            (work_id,),
        ).fetchone()
        if row is None:
            raise ContractError(f"unknown work item: {work_id}")
        if row["state"] != "running":
            raise ContractError(f"work item is not running: {work_id}")
        if row["worker_id"] != worker_id or int(row["attempts"]) != int(attempt):
            raise ContractError("lease is owned by another worker or attempt")
        if row["lease_expires_at"] is None or float(row["lease_expires_at"]) <= now_epoch:
            raise ContractError(f"work item lease has expired: {work_id}")
        return row

    def _verify_artifacts(self, result: dict) -> None:
        for artifact in result["artifacts"]:
            relative = Path(artifact["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ContractError(f"artifact path must be run-relative: {relative}")
            if not relative.parts or relative.parts[0] not in {"work", "artifacts"}:
                raise ContractError(
                    f"artifact must be stored under work/ or artifacts/: {relative}"
                )
            if relative.parts[0] == "work":
                expected_parent = (
                    Path("work")
                    / result["work_id"]
                    / f"attempt-{result['attempt']}"
                )
                try:
                    relative.relative_to(expected_parent)
                except ValueError as error:
                    raise ContractError(
                        "work artifact must belong to the active attempt: "
                        f"{relative}"
                    ) from error
            artifact_path = (self.run_dir / relative).resolve()
            if self.run_dir not in artifact_path.parents:
                raise ContractError(f"artifact escapes the run directory: {relative}")
            if not artifact_path.is_file():
                raise ContractError(f"artifact not found: {relative}")
            if artifact_path.stat().st_size != int(artifact["bytes"]):
                raise ContractError(f"artifact byte size mismatch: {relative}")
            digest = hashlib.sha256()
            with artifact_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != artifact["sha256"]:
                raise ContractError(f"artifact SHA256 mismatch: {relative}")

    def finish(self, result: dict, worker_id: str) -> str:
        validate_tool_result(result)
        validate_identifier(worker_id, "worker_id")
        if result["worker_id"] != worker_id:
            raise ContractError("tool result worker_id does not match the caller")

        with self._connect() as connection:
            self._active_lease(
                connection,
                result["work_id"],
                worker_id,
                result["attempt"],
                time.time(),
            )
        self._verify_artifacts(result)
        result_json = canonical_json_bytes(result).decode("utf-8")
        now = utc_now()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._active_lease(
                connection,
                result["work_id"],
                worker_id,
                result["attempt"],
                time.time(),
            )

            state_by_status = {
                "succeeded": "completed",
                "failed": "failed",
                "retryable": "pending",
            }
            new_state = state_by_status[result["status"]]
            if result["status"] == "retryable" and int(row["attempts"]) >= int(
                row["max_attempts"]
            ):
                new_state = "failed"
            connection.execute(
                """
                UPDATE attempts
                SET status = ?, finished_at_utc = ?, result_json = ?
                WHERE work_id = ? AND attempt = ? AND status = 'running'
                """,
                (
                    result["status"],
                    result["finished_at_utc"],
                    result_json,
                    result["work_id"],
                    result["attempt"],
                ),
            )
            connection.execute(
                """
                UPDATE work_items
                SET state = ?, worker_id = NULL, lease_expires_at = NULL,
                    result_json = ?, updated_at_utc = ?
                WHERE work_id = ?
                """,
                (new_state, result_json, now, result["work_id"]),
            )
            connection.commit()

        result_path = (
            self.run_dir
            / "work"
            / result["work_id"]
            / f"attempt-{result['attempt']}"
            / "result.json"
        )
        write_json_atomic(result_path, result)
        return new_state

    def status(self) -> dict:
        counts = {state: 0 for state in QUEUE_STATES}
        with self._connect() as connection:
            for row in connection.execute(
                "SELECT state, COUNT(*) AS count FROM work_items GROUP BY state"
            ):
                counts[row["state"]] = int(row["count"])
            attempts = int(
                connection.execute("SELECT COUNT(*) AS count FROM attempts").fetchone()[
                    "count"
                ]
            )
        return {
            "run_id": self.manifest()["run_id"],
            "counts": counts,
            "attempts": attempts,
            "total": sum(counts.values()),
        }

    def export(self) -> dict:
        with self._connect() as connection:
            work_rows = []
            for row in connection.execute(
                "SELECT * FROM work_items ORDER BY created_at_utc, work_id"
            ):
                work_rows.append(
                    {
                        "work_item": json.loads(row["payload_json"]),
                        "queue": {
                            "state": row["state"],
                            "attempts": row["attempts"],
                            "max_attempts": row["max_attempts"],
                            "worker_id": row["worker_id"],
                            "lease_expires_at_epoch": row["lease_expires_at"],
                            "created_at_utc": row["created_at_utc"],
                            "updated_at_utc": row["updated_at_utc"],
                        },
                        "latest_result": (
                            json.loads(row["result_json"])
                            if row["result_json"] is not None
                            else None
                        ),
                    }
                )
            attempt_rows = []
            for row in connection.execute(
                "SELECT * FROM attempts ORDER BY work_id, attempt"
            ):
                attempt_rows.append(
                    {
                        "work_id": row["work_id"],
                        "attempt": row["attempt"],
                        "worker_id": row["worker_id"],
                        "claimed_at_utc": row["claimed_at_utc"],
                        "lease_expires_at_epoch": row["lease_expires_at"],
                        "finished_at_utc": row["finished_at_utc"],
                        "status": row["status"],
                        "result": (
                            json.loads(row["result_json"])
                            if row["result_json"] is not None
                            else None
                        ),
                    }
                )

        work_path = self.run_dir / "tables/work-items.jsonl"
        attempt_path = self.run_dir / "tables/attempts.jsonl"
        write_jsonl_atomic(work_path, work_rows)
        write_jsonl_atomic(attempt_path, attempt_rows)
        return {
            "work_items": len(work_rows),
            "attempts": len(attempt_rows),
            "work_items_path": str(work_path),
            "attempts_path": str(attempt_path),
        }
