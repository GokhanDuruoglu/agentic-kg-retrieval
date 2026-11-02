"""One-time GraphRAG indexing: entity embeddings + hierarchical Leiden communities + LLM community reports.

The repository ships the 9,828 community reports of the paper runs (``data/2wiki/graphrag/reports.jsonl.gz``,
keyed by member set and the full id of the model that wrote them), so indexing normally makes no LLM call. Reports
for communities not in the cache, or written by another model, are generated with the ``answer`` role model;
``--no-reports`` only re-applies cached reports.

    akg index graphrag [--no-reports] [--report-limit N] [--rebuild]
"""

from __future__ import annotations

from ...llm import LLM
from .communities import detect_communities, generate_reports, save_cache
from .config import DEFAULT_EMBED_MODEL, ensure_shipped_caches, get_source
from .entities import EntityIndex
from .loader import load_artifacts


def build_index(source_name: str = "2wiki", *, answer_llm: LLM | None, model_id: str,
                embed_model: str = DEFAULT_EMBED_MODEL, max_cluster_size: int = 10, no_reports: bool = False,
                report_limit: int | None = None, rebuild: bool = False, skip_embed: bool = False) -> None:
    src = get_source(source_name)
    ensure_shipped_caches(src)
    print(f"Loading artifacts for {src.name} ...")
    art = load_artifacts(src)
    print(f"  {art.stats()}")

    if not skip_embed:
        print("Building entity embedding index ...")
        EntityIndex(art, model_name=embed_model, rebuild=rebuild)

    print(f"Detecting hierarchical Leiden communities (max_cluster_size={max_cluster_size}) ...")
    communities, node_communities = detect_communities(art, max_cluster_size=max_cluster_size)
    n_levels = len({c["level"] for c in communities.values()}) if communities else 0
    print(f"  {len(communities)} communities across {n_levels} levels")

    if no_reports or answer_llm is None:
        generate_reports(art, communities, model=model_id, reuse_only=True)
    else:
        print("Generating missing LLM community reports (bottom-up per level, parallel, resumable) ...")
        generate_reports(art, communities, model=model_id, llm=answer_llm, limit=report_limit)

    save_cache(art, communities, node_communities)
    print("Done.")
