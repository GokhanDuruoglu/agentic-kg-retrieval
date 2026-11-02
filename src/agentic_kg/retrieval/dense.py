"""Dense retriever: multilingual E5 embeddings + exact cosine search (FAISS ``IndexFlatIP``).

Passages are encoded as ``passage: <document>``, queries as ``query: <question>``, both L2-normalized. Corpus
embeddings are cached under ``.cache/dense`` (keyed by model, corpus file and size).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import ROOT
from ..io import read_jsonl
from ..registry import register

_DEFAULT_CACHE = ROOT / ".cache" / "dense"


def _model_tag(model_name: str) -> str:
    base = Path(model_name).name or model_name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


@register("retriever", "dense")
class DenseRetriever:
    def __init__(self, corpus_path, model: str = "intfloat/multilingual-e5-large", *, revision: str | None = None,
                 device: str | None = None, batch_size: int = 64, max_seq_length: int = 512,
                 query_prefix: str = "query: ", passage_prefix: str = "passage: ",
                 cache_dir: str | Path | None = None, rebuild: bool = False):
        self.corpus_path = Path(corpus_path)
        self.model_name = model
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.records: list[dict] = read_jsonl(corpus_path)
        self.model = SentenceTransformer(model, device=device, revision=revision)
        self.model.max_seq_length = max_seq_length

        embeddings = self._corpus_embeddings(cache_dir, rebuild)
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    def _cache_path(self, cache_dir) -> Path:
        base = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        key = "|".join([self.model_name, self.corpus_path.name, str(len(self.records)), self.passage_prefix])
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return base / f"emb_{_model_tag(self.model_name)}_{digest}.npy"

    def _corpus_embeddings(self, cache_dir, rebuild: bool) -> np.ndarray:
        cache_path = self._cache_path(cache_dir)
        if cache_path.exists() and not rebuild:
            print(f"  loading cached embeddings: {cache_path}")
            return np.load(cache_path)
        print(f"  encoding {len(self.records)} documents with {self.model_name} ...")
        passages = [self.passage_prefix + (rec.get("document") or "") for rec in self.records]
        emb = self.model.encode(passages, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        return emb

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
        scores, indices = self.index.search(q, min(top_k, len(self.records)))
        hits = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            rec = self.records[int(idx)]
            hits.append({"rank": rank, "score": round(float(score), 4), "doc_id": rec.get("doc_id"),
                         "node_type": "Passage", "title": rec.get("title"), "document": rec.get("document")})
        return hits
