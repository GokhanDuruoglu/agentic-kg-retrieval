"""Emit the retrieval corpus (one doc per Passage).

Reads the sampled records (records.jsonl) so the corpus passages exactly match the graph and the ground truth.
The retrieval target is the passage: doc_id = "Passage:<passage_id>" (== relevant_sets keys in the gold file).
"""

import argparse
import json
import re
import sys
from pathlib import Path

from .settings import DATA_DIR  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(*vals) -> list[str]:
    out: list[str] = []
    for v in vals:
        if v is None:
            continue
        out.extend(_TOKEN_RE.findall(str(v).lower()))
    # dedupe preserving order
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _unique_passages(records: list[dict]) -> dict[str, dict]:
    passages: dict[str, dict] = {}
    for rec in records:
        for p in rec["passages"]:
            passages.setdefault(p["passage_id"], {"id": p["passage_id"],
                                                  "title": p["title"], "text": p["text"]})
    return passages


def build(benchmark: str) -> Path:
    rec_path = DATA_DIR / benchmark / "records.jsonl"
    if not rec_path.exists():
        raise SystemExit(f"records.jsonl not found: {rec_path} (run build_kg.py first).")
    records = [json.loads(l) for l in rec_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    passages = _unique_passages(records)

    out = DATA_DIR / benchmark / f"{benchmark}_corpus.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for pid, p in passages.items():
            document = f"{p['title']}. {p['text']}".strip()
            record = {
                "doc_id":   f"Passage:{pid}",
                "id":       pid,
                "title":    p["title"],
                "document": document,
                "keywords": _tokenize(p["title"], p["text"]),
                "metadata": {"title": p["title"]},
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  corpus: {len(passages)} passages → {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Emit a passage retrieval corpus JSONL.")
    ap.add_argument("--benchmark", required=True, help="Benchmark key (<AKG_BUILD_DIR>/<key>/).")
    args = ap.parse_args(argv)
    build(args.benchmark)


if __name__ == "__main__":
    main()
