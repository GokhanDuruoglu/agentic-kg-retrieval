"""GraphRAG local search over the 2Wiki knowledge graph (Microsoft-style): entity seeds -> one-hop expansion ->
ranked passages, plus an optional local-search context of community reports, entities, relationships and sources."""

from .local_search import GraphRAGRetriever

__all__ = ["GraphRAGRetriever"]


from ...registry import register  # noqa: E402
from .config import DEFAULT_EMBED_MODEL, LocalSearchConfig  # noqa: E402


@register("retriever", "graphrag")
def build_graphrag_retriever(corpus_path=None, source: str = "2wiki", embed_model: str = DEFAULT_EMBED_MODEL,
                             top_k_mapped_entities: int = 10, top_k_relationships: int = 10,
                             max_context_tokens: int = 0, rebuild: bool = False) -> GraphRAGRetriever:
    """GraphRAG retriever; the corpus is taken from the source (``corpus_path`` is accepted for uniformity)."""
    cfg = LocalSearchConfig(top_k_mapped_entities=top_k_mapped_entities, top_k_relationships=top_k_relationships,
                            max_context_tokens=max_context_tokens)
    return GraphRAGRetriever.from_source(source, cfg=cfg, model_name=embed_model, rebuild=rebuild)
