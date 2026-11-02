"""Entity descriptions + the embedding index used for Microsoft-style seed selection.

Microsoft GraphRAG seeds local search by embedding the query and ranking entities by entity-description
embedding similarity. The 2Wiki KG has no per-entity descriptions, and one LLM description per entity (~42k
calls) is prohibitive, so a description is SYNTHESIZED from the KG (name + type + aliases + relation phrases + one
mention snippet) and embedded.

Embeddings use the vanilla E5 model (``intfloat/multilingual-e5-large``) with the standard ``query:`` /
``passage:`` prefixes and L2 normalization, indexed with FAISS ``IndexFlatIP`` (cosine). The mention snippet and
local-search tie-breaking iterate Python sets of strings, so results depend on ``PYTHONHASHSEED`` (the CLI pins it
to 0).
"""

import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import CACHE_DIR, DEFAULT_EMBED_MODEL
from .loader import GraphArtifacts

_MAX_REL_PHRASES = 30      # cap relation phrases folded into a synthesized description
_SNIPPET_CHARS = 240       # mention-snippet length


def _model_tag(model_name: str) -> str:
    base = Path(model_name).name or model_name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def synthesize_description(entity_id: str, art: GraphArtifacts) -> str:
    """Build an embeddable description from the KG (no LLM)."""
    ent = art.entities[entity_id]
    parts = [ent["name"]]
    if ent.get("type"):
        parts[0] = f"{ent['name']} ({ent['type']})"
    if ent.get("aliases"):
        parts.append("Also known as: " + ", ".join(ent["aliases"][:6]) + ".")

    rels = []
    for nb in art.adjacency.get(entity_id, [])[:_MAX_REL_PHRASES]:
        nb_ent = art.entities.get(nb["neighbor"])
        if not nb_ent:
            continue
        phrase = nb.get("name") or (nb.get("rel_type") or "related to")
        rels.append(f"{phrase} {nb_ent['name']}")
    if rels:
        parts.append("Relations: " + "; ".join(rels) + ".")

    # One mention snippet for extra lexical/semantic signal.
    for doc_id in list(art.entity_passages.get(entity_id, []))[:1]:
        rec = art.text_units.get(doc_id)
        if rec and rec.get("document"):
            parts.append(rec["document"][:_SNIPPET_CHARS])
        break

    return " ".join(p for p in parts if p).strip()


class EntityIndex:
    """Embed entity descriptions and answer ``search_entities(query, k)``."""

    def __init__(
        self,
        art: GraphArtifacts,
        *,
        model_name: str = DEFAULT_EMBED_MODEL,
        cache_dir: Path = CACHE_DIR,
        rebuild: bool = False,
        device: str | None = None,
        batch_size: int = 64,
        max_seq_length: int = 512,
    ):
        self.art = art
        self.model_name = model_name
        self.batch_size = batch_size
        self.entity_ids: list[str] = list(art.entities.keys())

        self.descriptions = [synthesize_description(eid, art) for eid in self.entity_ids]

        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length

        embeddings = self._embeddings(Path(cache_dir), rebuild)
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    # -- embedding cache ----------------------------------------------------

    def _cache_path(self, cache_dir: Path) -> Path:
        key = "|".join([self.model_name, self.art.source.name,
                        str(len(self.entity_ids)), "synth"])
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return cache_dir / f"entemb_{self.art.source.name}_{_model_tag(self.model_name)}_{digest}.npy"

    def _embeddings(self, cache_dir: Path, rebuild: bool) -> np.ndarray:
        cache_path = self._cache_path(cache_dir)
        if cache_path.exists() and not rebuild:
            print(f"  loading cached entity embeddings: {cache_path}")
            return np.load(cache_path)

        print(f"  embedding {len(self.entity_ids)} entity descriptions with {self.model_name} …")
        passages = ["passage: " + d for d in self.descriptions]
        emb = self.model.encode(
            passages, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
        ).astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        print(f"  cached entity embeddings → {cache_path}")
        return emb

    # -- search -------------------------------------------------------------

    def search_entities(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return [(entity_id, cosine_score)] for the top-k entities by description similarity."""
        q = self.model.encode(["query: " + query], normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
        k = min(k, len(self.entity_ids))
        scores, idx = self.index.search(q, k)
        out: list[tuple[str, float]] = []
        for i, sc in zip(idx[0], scores[0]):
            if i < 0:
                continue
            out.append((self.entity_ids[int(i)], float(sc)))
        return out
