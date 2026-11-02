"""GraphRAG-local configuration: where the graph artifacts and caches live, and the local-search settings."""

from __future__ import annotations

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path

from ...config import ROOT

DATA = ROOT / "data" / "2wiki"
CACHE_DIR = ROOT / ".cache" / "graphrag"

# Vanilla E5 for entity seed-selection embeddings.
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-large"


@dataclass(frozen=True)
class Source:
    """Where a graph's artifacts live: the knowledge graph (kg.jsonl[.gz]) and the passage corpus (text units)."""

    name: str
    kg_path: Path
    corpus_path: Path


SOURCES: dict[str, Source] = {
    "2wiki": Source(name="2wiki", kg_path=DATA / "graph" / "kg.jsonl.gz", corpus_path=DATA / "corpus.jsonl.gz"),
}

# Index data shipped with the repository (compressed): the 9,828 LLM community reports and the community side-map.
SHIPPED = {
    "communities_{name}.json": DATA / "graphrag" / "communities.json.gz",
    "reports_{name}.jsonl": DATA / "graphrag" / "reports.jsonl.gz",
}


def get_source(name: str) -> Source:
    if name not in SOURCES:
        raise SystemExit(f"unknown source {name!r}; choose one of {list(SOURCES)}")
    return SOURCES[name]


def ensure_shipped_caches(source: Source) -> None:
    """Unpack the shipped community reports / side-map into the cache dir unless already there."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for pattern, gz in SHIPPED.items():
        target = CACHE_DIR / pattern.format(name=source.name)
        if not target.exists() and gz.exists():
            with gzip.open(gz, "rb") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


@dataclass
class LocalSearchConfig:
    """Microsoft local-search context-builder hyperparameters (defaults match GraphRAG)."""

    top_k_mapped_entities: int = 10     # seed entities selected by description embedding
    oversample_scaler: int = 2          # fetch 2x seeds to tolerate dropouts (Microsoft default)
    top_k_relationships: int = 10       # out-of-network relationships pulled into context
    text_unit_prop: float = 0.5         # context-window share for text units (passages)
    community_prop: float = 0.1         # context-window share for community reports
    max_context_tokens: int = 0         # total context budget; 0 = derived so that all top-k passages fit
    max_cluster_size: int = 10          # graspologic hierarchical_leiden cap
