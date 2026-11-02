"""Small file helpers: transparent .gz for JSONL data files."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator


def open_text(path, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_lines(path) -> list[str]:
    with open_text(path) as fh:
        return fh.read().splitlines()


def iter_jsonl(path) -> Iterator[dict]:
    with open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path, rows) -> int:
    n = 0
    with open_text(path, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
