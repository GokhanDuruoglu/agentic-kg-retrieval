"""GraphRAGRetriever — Microsoft GraphRAG *local search* over offline graph artifacts.

``search(query, top_k)`` returns ranked **passage** hits in the same dict shape as the BM25 and dense retrievers;
``build_context`` renders the Microsoft local-search context (reports + entities + relationships + sources).

Pipeline (Microsoft local search):
  1. **Seed entities** — top ``top_k_mapped_entities`` entities by description-embedding similarity
     (fetched with ``oversample_scaler``×).
  2. **One-hop expansion** — neighbor entities via relationships, split in-network (both endpoints
     are seeds) vs out-network (seed↔non-seed). Out-network neighbors are ranked by #links-to-seeds
     then combined degree; the first ``top_k_relationships`` get full weight, the rest a decayed
     weight (kept so deep recall@k stays meaningful for the benchmark).
  3. **Text units** — passages mapped from the expanded entity set via the mention map, scored by
     the best mentioning-entity weight then mention count → this ranked passage list is returned.
  4. **Community reports** — best-effort community context (title/report) for the seed entities is
     attached to hits and exposed via ``community_context`` for the answer stage.
"""

import json

from ...tokens import count_tokens
from .communities import community_cache_path
from .config import (
    DEFAULT_EMBED_MODEL,
    LocalSearchConfig,
    Source,
    ensure_shipped_caches,
    get_source,
)
from .entities import EntityIndex
from .loader import GraphArtifacts, load_artifacts

_HEAD_DECAY = 0.5     # weight factor for the top_k_relationships out-network neighbors
_TAIL_DECAY = 0.25    # weight factor for the remaining out-network neighbors


class GraphRAGRetriever:
    def __init__(
        self,
        art: GraphArtifacts,
        entity_index: EntityIndex,
        *,
        cfg: LocalSearchConfig | None = None,
        community_meta: dict | None = None,
        entity_to_communities: dict | None = None,
    ):
        self.art = art
        self.entities = entity_index
        self.cfg = cfg or LocalSearchConfig()
        self.community_meta = community_meta or {}
        self.entity_to_communities = entity_to_communities or {}

    # -- construction -------------------------------------------------------

    @classmethod
    def from_source(
        cls,
        source: str | Source,
        *,
        cfg: LocalSearchConfig | None = None,
        model_name: str = DEFAULT_EMBED_MODEL,
        rebuild: bool = False,
    ) -> "GraphRAGRetriever":
        src = source if isinstance(source, Source) else get_source(source)
        ensure_shipped_caches(src)
        art = load_artifacts(src)
        print(f"  [graphrag] {src.name}: {art.stats()}")
        entity_index = EntityIndex(art, model_name=model_name, rebuild=rebuild)
        community_meta, e2c = _load_community_cache(src.name)
        return cls(art, entity_index, cfg=cfg,
                   community_meta=community_meta, entity_to_communities=e2c)

    # -- helpers ------------------------------------------------------------

    def _combined_degree(self, a: str, b: str) -> int:
        return self.art.degree.get(a, 0) + self.art.degree.get(b, 0)

    def _expand(self, seed_ids: list[str], seed_score: dict[str, float]) -> dict[str, float]:
        """Return entity_id -> weight for the expanded set (seeds + ranked out-network neighbors)."""
        S = set(seed_ids)
        weight: dict[str, float] = {sid: seed_score[sid] for sid in seed_ids}

        out_links: dict[str, int] = {}
        out_best_seed: dict[str, float] = {}
        for sid in seed_ids:
            for nb in self.art.adjacency.get(sid, []):
                n = nb["neighbor"]
                if n in S or n not in self.art.entities:
                    continue
                out_links[n] = out_links.get(n, 0) + 1
                out_best_seed[n] = max(out_best_seed.get(n, 0.0), seed_score[sid])

        ranked_out = sorted(
            out_links,
            key=lambda n: (-out_links[n], -self.art.degree.get(n, 0)),
        )
        top_k = self.cfg.top_k_relationships
        for rank_i, n in enumerate(ranked_out):
            decay = _HEAD_DECAY if rank_i < top_k else _TAIL_DECAY
            weight[n] = out_best_seed[n] * decay + 1e-3 * out_links[n] - 1e-6 * rank_i
        return weight

    def _seed_community(self, seed_ids: list[str]):
        """Pick the highest-rank leaf community among the seeds (for hit annotation)."""
        best = None
        for sid in seed_ids:
            for cid in self.entity_to_communities.get(sid, []):
                meta = self.community_meta.get(str(cid))
                if not meta:
                    continue
                if best is None or (meta.get("rank") or 0) > (best[1].get("rank") or 0):
                    best = (cid, meta)
        return best  # (cid, meta) or None

    # -- public API ---------------------------------------------------------

    def community_context(self, seed_ids: list[str], max_reports: int = 5) -> list[dict]:
        """Reports of the communities the seed entities belong to, ranked by #seed matches then rank."""
        matches: dict[int, int] = {}
        for sid in seed_ids:
            for cid in self.entity_to_communities.get(sid, []):
                matches[cid] = matches.get(cid, 0) + 1
        ranked = sorted(
            matches,
            key=lambda cid: (-matches[cid], -(self.community_meta.get(str(cid), {}).get("rank") or 0)),
        )
        out = []
        for cid in ranked[:max_reports]:
            meta = self.community_meta.get(str(cid), {})
            if meta.get("report"):
                out.append({"community": cid, "title": meta.get("title"),
                            "report": meta.get("report"), "rank": meta.get("rank"),
                            "matches": matches[cid]})
        return out

    # -- Microsoft-style local-search context --------------------------------

    def _entity_description(self, eid: str) -> str:
        """Short description from the KG: aliases + a few relation phrases (no LLM; the graph carries no descriptions)."""
        ent = self.art.entities[eid]
        parts = []
        if ent.get("aliases"):
            parts.append("aka " + ", ".join(ent["aliases"][:4]))
        rels = []
        for nb in self.art.adjacency.get(eid, [])[:5]:
            other = self.art.entities.get(nb["neighbor"])
            if other:
                rels.append(f"{nb.get('name') or nb.get('rel_type') or 'related to'} {other['name']}")
        if rels:
            parts.append("; ".join(rels))
        return " — ".join(parts)

    def build_context(self, query: str, hits: list[dict]) -> tuple[str, int, dict]:
        """Assemble the answer context the way Microsoft GraphRAG local search does.

        Four tables, each under its share of the token budget:
          community reports   (community_prop, default 0.1)  — reports of the communities holding the seed entities
          entities + relationships (the remainder, 0.4)      — the seed entities and their ranked neighbours
          sources / text units (text_unit_prop, default 0.5) — the ranked passages (the same 20 as the passages-only variant)
        With cfg.max_context_tokens <= 0 the total is derived so that all retrieved passages fit in their 0.5 share
        (keeps the evidence of the other retrieval baselines: the same top-k passages, plus the added sections).
        Returns (context_text, n_sources_used, meta).
        """

        st = getattr(self, "_last", None)
        if not st or not hits:
            return "", 0, {}
        cfg = self.cfg
        src_rows = [f"{h['rank']}|{(h.get('document') or '').strip()}" for h in hits]
        src_tokens = [count_tokens(r) for r in src_rows]
        total = cfg.max_context_tokens if cfg.max_context_tokens > 0 else int(sum(src_tokens) / max(cfg.text_unit_prop, 0.05))
        b_src = int(total * cfg.text_unit_prop) if cfg.max_context_tokens > 0 else sum(src_tokens)
        b_rep = int(total * cfg.community_prop)
        b_loc = max(total - b_src - b_rep, 0)

        # sources
        used, spent = [], 0
        for row, n in zip(src_rows, src_tokens):
            if used and spent + n > b_src:
                break
            used.append(row); spent += n

        # community reports
        rep_rows, spent_rep = [], 0
        for r in self.community_context(st["seed_ids"], max_reports=30):
            row = f"{r['community']}|{r.get('title') or ''}|{(r.get('report') or '').strip()}"
            n = count_tokens(row)
            if spent_rep + n > b_rep:
                break
            rep_rows.append(row); spent_rep += n

        # entities (seeds first, then ranked neighbours) + their relationships, until the local budget is used
        order = sorted(st["weight"], key=lambda e: -st["weight"][e])
        selected, ent_rows, rel_rows, spent_loc = [], [], [], 0
        seen_edges: set = set()
        for eid in order[:60]:
            ent = self.art.entities.get(eid)
            if not ent:
                continue
            row = f"{len(ent_rows) + 1}|{ent['name']}|{ent.get('type') or ''}|{self._entity_description(eid)}"
            n = count_tokens(row)
            if spent_loc + n > b_loc:
                break
            ent_rows.append(row); selected.append(eid); spent_loc += n
        sel = set(selected)
        edges = []
        for eid in selected:
            for nb in self.art.adjacency.get(eid, []):
                other = nb["neighbor"]
                key = tuple(sorted((eid, other))) + (nb.get("rel_type"),)
                if key in seen_edges or other not in self.art.entities:
                    continue
                seen_edges.add(key)
                in_net = other in sel
                edges.append((0 if in_net else 1, -self._combined_degree(eid, other), eid, other, nb))
        edges.sort(key=lambda t: (t[0], t[1]))
        n_out = 0
        for in_out, _deg, a, b, nb in edges:
            if in_out == 1:
                if n_out >= cfg.top_k_relationships:
                    continue
            row = (f"{len(rel_rows) + 1}|{self.art.entities[a]['name']}|{self.art.entities[b]['name']}|"
                   f"{nb.get('name') or nb.get('rel_type') or 'related to'}")
            n = count_tokens(row)
            if spent_loc + n > b_loc:
                break
            rel_rows.append(row); spent_loc += n
            n_out += in_out

        sections = []
        if rep_rows:
            sections.append("-----Community reports-----\nid|title|content\n" + "\n".join(rep_rows))
        if ent_rows:
            sections.append("-----Entities-----\nid|entity|type|description\n" + "\n".join(ent_rows))
        if rel_rows:
            sections.append("-----Relationships-----\nid|source|target|description\n" + "\n".join(rel_rows))
        sections.append("-----Sources-----\nid|text\n" + "\n".join(used))
        meta = {"n_reports": len(rep_rows), "n_entities": len(ent_rows), "n_relationships": len(rel_rows),
                "n_sources": len(used), "tokens": {"reports": spent_rep, "local": spent_loc, "sources": spent},
                "budget_total": total}
        return "\n\n".join(sections), len(used), meta

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        n_seed = self.cfg.top_k_mapped_entities
        seeds = self.entities.search_entities(query, k=n_seed * self.cfg.oversample_scaler)
        seeds = seeds[:n_seed]
        if not seeds:
            return []
        seed_ids = [eid for eid, _ in seeds]
        seed_score = {eid: sc for eid, sc in seeds}

        weight = self._expand(seed_ids, seed_score)

        # Text-unit (passage) ranking.
        passage_score: dict[str, float] = {}
        passage_count: dict[str, int] = {}
        for eid, w in weight.items():
            for doc_id in self.art.entity_passages.get(eid, ()):  # passage doc_ids
                if doc_id not in self.art.text_units:
                    continue
                if w > passage_score.get(doc_id, -1.0):
                    passage_score[doc_id] = w
                passage_count[doc_id] = passage_count.get(doc_id, 0) + 1

        ranked = sorted(
            passage_score,
            key=lambda d: (-passage_score[d], -passage_count[d]),
        )[:top_k]

        # kept for build_context(): the seeds and the expanded entity weights of THIS query
        # (the retriever is used sequentially, one query at a time).
        self._last = {"seed_ids": seed_ids, "seed_score": seed_score, "weight": weight}

        seed_comm = self._seed_community(seed_ids)
        hits: list[dict] = []
        for rank, doc_id in enumerate(ranked, start=1):
            rec = self.art.text_units[doc_id]
            hit = {
                "rank":        rank,
                "score":       round(float(passage_score[doc_id]), 4),
                "doc_id":      rec.get("doc_id"),
                "node_type":   "Passage",
                "title":       rec.get("title"),
                "document":    rec.get("document"),
                "n_entities":  passage_count[doc_id],
            }
            if seed_comm:
                hit["community"] = seed_comm[0]
                hit["community_title"] = seed_comm[1].get("title")
            hits.append(hit)
        return hits


def _load_community_cache(source_name: str) -> tuple[dict, dict]:
    path = community_cache_path(source_name)
    if not path.exists():
        print(f"  [graphrag] no community side-map at {path} "
              f"(run `index` to build communities/reports); proceeding without community context")
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("community_meta", {}), payload.get("entity_to_communities", {})
