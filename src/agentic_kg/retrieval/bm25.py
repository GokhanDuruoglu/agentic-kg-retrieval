"""BM25 sparse retriever over the passage corpus.

Each passage is indexed over its ``document`` text plus its ``keywords`` token list, tokenized as lowercase
``[a-z0-9]+`` runs (no stemming, no stop words; non-ASCII letters split a word). Ties keep corpus order.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ..io import read_jsonl
from ..registry import register

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@register("retriever", "bm25")
class BM25Retriever:
    def __init__(self, corpus_path, k1: float = 1.5, b: float = 0.75):
        self.records: list[dict] = read_jsonl(corpus_path)
        tokenized = [_tokenize((rec.get("document") or "") + " " + " ".join(rec.get("keywords") or []))
                     for rec in self.records]
        self.bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [{"rank": rank, "score": round(float(scores[idx]), 4), "doc_id": self.records[idx].get("doc_id"),
                 "node_type": "Passage", "title": self.records[idx].get("title"),
                 "document": self.records[idx].get("document")}
                for rank, idx in enumerate(ranked[:top_k], start=1)]
