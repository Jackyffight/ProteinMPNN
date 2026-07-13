#!/usr/bin/env python3
"""Expose the base interpreter's CUDA Torch installation to a runtime venv."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-site-packages", required=True)
    args = parser.parse_args()

    try:
        import torch
    except Exception as error:
        raise SystemExit(f"base Python cannot import torch: {error}") from error
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise SystemExit("base Python has no torch package metadata") from error

    module_file = getattr(torch, "__file__", None)
    if not module_file:
        raise SystemExit("base Python torch module has no filesystem path")
    source_site = Path(module_file).resolve().parent.parent
    if not source_site.is_dir():
        raise SystemExit(f"base Python torch site-packages not found: {source_site}")

    target_site = Path(args.target_site_packages).expanduser().resolve()
    target_site.mkdir(parents=True, exist_ok=True)
    link_path = target_site / "proteinmpnn-base-torch.pth"
    temporary = link_path.with_name(f".{link_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(f"{source_site}\n", encoding="utf-8")
        os.replace(temporary, link_path)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "base_torch_site": str(source_site),
                "link_path": str(link_path),
                "torch_version": version,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
