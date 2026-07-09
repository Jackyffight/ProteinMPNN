#!/usr/bin/env python3
"""Parallel HTTPS downloader for wwPDB biological assembly mmCIF files."""

from __future__ import annotations

import argparse
import html.parser
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from urllib.parse import urljoin


BASE_URL = "https://files.wwpdb.org/pub/pdb/data/assemblies/mmCIF/divided/"


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True, help="Destination raw/assemblies_mmcif directory.")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-in-flight", type=int, default=0)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--assembly-id", default="all", help="all or an assembly id such as 1.")
    parser.add_argument("--pdb-list", default="", help="Optional file with one PDB id per line.")
    parser.add_argument("--limit", type=int, default=0, help="Debug: download first n files only.")
    parser.add_argument("--manifest", default="", help="Path to write/read download manifest JSONL.")
    parser.add_argument("--list-only", action="store_true", help="Build manifest without downloading files.")
    parser.add_argument("--force-list", action="store_true", help="Rebuild manifest even if it exists.")
    return parser.parse_args()


def fetch_text(url: str, timeout: int = 60, retries: int = 10, retry_delay: float = 5.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ProteinMPNN-dataset-builder/1.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == retries:
                raise
            time.sleep(retry_delay)
    raise RuntimeError(f"failed to fetch {url}")


def parse_size(text: str) -> int:
    match = re.match(r"([0-9.]+)\s*([KMGT]?B)", text.strip())
    if not match:
        return 0
    value = float(match.group(1))
    unit = match.group(2)
    multiplier = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[unit]
    return int(value * multiplier)


def list_subdirs(base_url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(fetch_text(base_url))
    return sorted(href for href in parser.links if href.endswith("/") and href not in {"../", "./"})


def list_directory(base_url: str, subdir: str) -> list[dict]:
    url = urljoin(base_url, subdir)
    html = fetch_text(url)
    rows = re.findall(
        r'href="([^"]+\.cif\.gz)".*?<td>([^<]+)</td><td class="text-end">([^<]+)</td>',
        html,
    )
    records = []
    for filename, modified, size_text in rows:
        records.append(
            {
                "url": urljoin(url, filename),
                "relpath": f"{subdir}{filename}",
                "modified": modified.strip(),
                "size": parse_size(size_text),
            }
        )
    return records


def build_manifest(base_url: str, assembly_id: str, workers: int, limit: int) -> list[dict]:
    subdirs = list_subdirs(base_url)
    records: list[dict] = []
    pattern = None
    if assembly_id != "all":
        pattern = re.compile(rf"-assembly{re.escape(assembly_id)}\.cif\.gz$")

    if limit > 0:
        for index, subdir in enumerate(subdirs, 1):
            dir_records = list_directory(base_url, subdir)
            if pattern is not None:
                dir_records = [record for record in dir_records if pattern.search(record["relpath"])]
            records.extend(dir_records)
            if len(records) >= limit:
                return sorted(records, key=lambda item: item["relpath"])[:limit]
            if index % 100 == 0:
                total = sum(record["size"] for record in records)
                print(
                    f"listed_dirs={index}/{len(subdirs)} files={len(records)} "
                    f"gib={total / 1024**3:.1f}",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(list_directory, base_url, subdir): subdir for subdir in subdirs}
        for index, future in enumerate(as_completed(futures), 1):
            dir_records = future.result()
            if pattern is not None:
                dir_records = [record for record in dir_records if pattern.search(record["relpath"])]
            records.extend(dir_records)
            if index % 100 == 0:
                total = sum(record["size"] for record in records)
                print(
                    f"listed_dirs={index}/{len(subdirs)} files={len(records)} "
                    f"gib={total / 1024**3:.1f}",
                    flush=True,
                )

    records.sort(key=lambda item: item["relpath"])
    if limit > 0:
        records = records[:limit]
    return records


def build_manifest_from_pdb_list(base_url: str, pdb_list: str, assembly_id: str, limit: int) -> list[dict]:
    records: list[dict] = []
    with open(pdb_list, "r", encoding="utf-8") as handle:
        pdb_ids = [line.strip().lower() for line in handle if line.strip() and not line.startswith("#")]
    if limit > 0:
        pdb_ids = pdb_ids[:limit]
    assembly_ids = ["1"] if assembly_id == "all" else [assembly_id]
    for pdb_id in pdb_ids:
        if len(pdb_id) != 4:
            continue
        subdir = pdb_id[1:3]
        for current_assembly_id in assembly_ids:
            filename = f"{pdb_id}-assembly{current_assembly_id}.cif.gz"
            relpath = f"{subdir}/{filename}"
            records.append(
                {
                    "url": urljoin(base_url, relpath),
                    "relpath": relpath,
                    "modified": "",
                    "size": 0,
                }
            )
    return records


def load_manifest(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def local_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0

def size_matches(actual: int, expected: int) -> bool:
    if expected <= 0 or actual <= 0:
        return False
    tolerance = max(4096, int(expected * 0.002))
    return abs(actual - expected) <= tolerance


def download_one(record: dict, dest: Path, retries: int, retry_delay: float) -> tuple[str, int]:
    target = dest / record["relpath"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(record["size"])
    current = local_size(target)
    if expected_size == 0 and current > 0:
        return "skipped", current
    if expected_size > 0 and size_matches(current, expected_size):
        return "skipped", expected_size
    if expected_size > 0 and current > expected_size and not size_matches(current, expected_size):
        target.unlink()
        current = 0

    tmp = target.with_suffix(target.suffix + ".tmp")
    if current > 0 and not tmp.exists():
        target.rename(tmp)
    elif current == 0 and tmp.exists():
        current = local_size(tmp)

    for attempt in range(1, retries + 1):
        headers = {"User-Agent": "ProteinMPNN-dataset-builder/1.0"}
        mode = "wb"
        if current > 0:
            headers["Range"] = f"bytes={current}-"
            mode = "ab"
        request = urllib.request.Request(record["url"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if current > 0 and response.status == 200:
                    mode = "wb"
                    current = 0
                full_response_size = 0
                if current == 0:
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        try:
                            full_response_size = int(content_length)
                        except ValueError:
                            full_response_size = 0
                with tmp.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            got = local_size(tmp)
            if (
                expected_size == 0
                or (full_response_size > 0 and got == full_response_size)
                or size_matches(got, expected_size)
            ):
                tmp.replace(target)
                return "downloaded", got
            current = got
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and current > 0 and size_matches(current, expected_size):
                tmp.replace(target)
                return "downloaded", current
            if exc.code == 416 and tmp.exists():
                tmp.unlink()
                current = 0
            if attempt == retries:
                return f"failed:{type(exc).__name__}", 0
            time.sleep(retry_delay)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                return f"failed:{type(exc).__name__}", 0
            time.sleep(retry_delay)

    return "failed:unknown", 0


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    dest = Path(args.dest)
    manifest_path = Path(args.manifest) if args.manifest else dest.parent / "assembly_download_manifest.jsonl"

    if manifest_path.exists() and not args.force_list:
        records = load_manifest(manifest_path)
    elif args.pdb_list:
        records = build_manifest_from_pdb_list(args.base_url, args.pdb_list, args.assembly_id, args.limit)
        write_manifest(manifest_path, records)
    else:
        records = build_manifest(args.base_url, args.assembly_id, args.workers, args.limit)
        write_manifest(manifest_path, records)

    total_size = sum(int(record["size"]) for record in records)
    print(f"manifest: {manifest_path}")
    print(f"files: {len(records)}")
    print(f"bytes: {total_size}")
    print(f"gib: {total_size / 1024**3:.2f}")
    if args.list_only:
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    bytes_done = 0
    started = time.time()
    max_in_flight = args.max_in_flight if args.max_in_flight > 0 else args.workers * 4
    max_in_flight = max(args.workers, max_in_flight)
    record_iter = iter(records)
    submitted = 0
    completed = 0

    def submit_until_full(executor: ThreadPoolExecutor, futures: dict) -> None:
        nonlocal submitted
        while len(futures) < max_in_flight:
            try:
                record = next(record_iter)
            except StopIteration:
                break
            future = executor.submit(download_one, record, dest, args.retries, args.retry_delay)
            futures[future] = record
            submitted += 1

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict = {}
        submit_until_full(executor, futures)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                status, size = future.result()
                completed += 1
                counts[status] = counts.get(status, 0) + 1
                bytes_done += size
                if completed % 1000 == 0 or status.startswith("failed"):
                    print(
                        f"done={completed}/{len(records)} submitted={submitted} "
                        f"in_flight={len(futures)} bytes_done_gib={bytes_done / 1024**3:.1f} "
                        f"counts={counts}",
                        flush=True,
                    )
            submit_until_full(executor, futures)

    elapsed = time.time() - started
    failed = sum(count for status, count in counts.items() if status.startswith("failed"))
    print(json.dumps({"counts": counts, "elapsed_seconds": round(elapsed, 2)}, sort_keys=True))
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
