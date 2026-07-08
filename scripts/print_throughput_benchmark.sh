#!/usr/bin/env bash
# Print a ProteinMPNN throughput benchmark summary.
#
# Usage:
#   scripts/print_throughput_benchmark.sh runs/benchmarks/throughput-quick-YYYYmmddHHMMSS

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <benchmark_root>" >&2
  exit 1
fi

SUMMARY_FILE="$1/summary.tsv"
if [ ! -f "$SUMMARY_FILE" ]; then
  echo "summary.tsv not found: $SUMMARY_FILE" >&2
  exit 1
fi

python - "$SUMMARY_FILE" <<'PY'
import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1])
rows = list(csv.DictReader(summary.open(), delimiter="\t"))

def as_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0

rows.sort(key=lambda row: as_float(row.get("examples_per_second", "")), reverse=True)

columns = [
    ("case", 28),
    ("status", 10),
    ("preset", 8),
    ("batch", 8),
    ("loader", 8),
    ("prefetch", 9),
    ("examples/s", 12),
    ("train_ppl", 11),
    ("valid_ppl", 11),
]

print("".join(name.ljust(width) for name, width in columns))
for row in rows:
    line = [
        row.get("case", ""),
        row.get("status", ""),
        row.get("preset", ""),
        row.get("batch_tokens", ""),
        row.get("loader_workers", ""),
        row.get("prefetch_workers", ""),
        row.get("examples_per_second", ""),
        row.get("train_perplexity", ""),
        row.get("valid_perplexity", ""),
    ]
    print("".join(value[:width - 1].ljust(width) for value, (_, width) in zip(line, columns)))

print()
print(f"raw summary: {summary}")
PY
