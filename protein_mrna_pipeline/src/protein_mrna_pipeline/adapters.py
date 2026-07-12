"""Boundary implemented by isolated expert-tool workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AdapterIdentity:
    name: str
    kind: str
    revision: str
    weights_sha256: str | None
    environment_id: str

    def as_record(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "revision": self.revision,
            "weights_sha256": self.weights_sha256,
            "environment_id": self.environment_id,
        }


class ToolAdapter(Protocol):
    """An expert adapter consumes one claimed work item and returns a tool result."""

    @property
    def identity(self) -> AdapterIdentity:
        ...

    def execute(self, work_item: dict, work_dir: Path) -> dict:
        """Run one item and return a protein-mrna.tool-result.v1 document."""
        ...
