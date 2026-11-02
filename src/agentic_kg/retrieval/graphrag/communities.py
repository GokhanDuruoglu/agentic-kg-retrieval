"""Hierarchical Leiden communities + LLM community reports.

Mirrors Microsoft GraphRAG indexing:
  * build a weighted entity graph and run graspologic ``hierarchical_leiden`` (the algorithm GraphRAG uses) to get
    multi-level communities;
  * generate an LLM community report per community, bottom-up (a parent's context includes its child communities'
    reports), cached and resumable;
  * cache a JSON side-map (community meta + entity -> communities) that the retriever loads at query time.

Disconnected graphs are handled by running Leiden per connected component (isolated nodes get no community), with
community ids offset so they stay globally unique.
"""

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
from graspologic.partition import hierarchical_leiden

from ...io import read_lines
from .config import CACHE_DIR
from .loader import GraphArtifacts

_RANDOM_SEED = 0xC0FFEE
_REPORT_MAX_ENTITIES = 50
_REPORT_MAX_RELS = 50
_REPORT_MAX_SUBREPORTS = 10
_REPORT_CTX_CHARS = 8000


def community_cache_path(source_name: str) -> Path:
    return CACHE_DIR / f"communities_{source_name}.json"


# ---------------------------------------------------------------------------
# Leiden
# ---------------------------------------------------------------------------

def detect_communities(art: GraphArtifacts, max_cluster_size: int = 10):
    """Run hierarchical Leiden per connected component.

    Returns ``(communities, node_communities)`` where
      communities[cid] = {id, level, members:set[str], parent:int|None, size}
      node_communities[entity_id] = [cid, ...]   (one per level the node appears in)
    """
    G = nx.Graph()
    G.add_nodes_from(art.entities.keys())
    for e in art.edges:
        s, t = e["source"], e["target"]
        if s == t:
            continue
        if G.has_edge(s, t):
            G[s][t]["weight"] += 1.0
        else:
            G.add_edge(s, t, weight=1.0)

    communities: dict[int, dict] = {}
    node_communities: dict[str, list[int]] = defaultdict(list)
    offset = 0

    for comp in nx.connected_components(G):
        if len(comp) < 2:
            continue
        sub = G.subgraph(comp).copy()
        clusters = hierarchical_leiden(sub, max_cluster_size=max_cluster_size,
                                       random_seed=_RANDOM_SEED)
        local_max = 0
        for hc in clusters:
            cid = hc.cluster + offset
            pid = (hc.parent_cluster + offset) if hc.parent_cluster is not None else None
            c = communities.setdefault(
                cid, {"id": cid, "level": hc.level, "members": set(), "parent": pid, "size": 0})
            c["members"].add(hc.node)
            node_communities[hc.node].append(cid)
            local_max = max(local_max, hc.cluster)
        offset += local_max + 1

    for c in communities.values():
        c["size"] = len(c["members"])
    return communities, dict(node_communities)


# ---------------------------------------------------------------------------
# LLM reports (bottom-up)
# ---------------------------------------------------------------------------

_REPORT_SYSTEM = (
    "You are summarizing a community of related entities from a knowledge graph. "
    "Given the entities, their relationships, and any sub-community reports, write a concise "
    "report. Respond with STRICT JSON: "
    '{"title": "<short title>", "summary": "<2-4 sentence summary>", "rating": <0-10 importance>}'
)


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {}


def _report_context(cid: int, art: GraphArtifacts, communities: dict[int, dict],
                    children: dict[int, list[int]]) -> str:
    c = communities[cid]
    members = c["members"]
    lines: list[str] = []

    ent_names = [art.entities[m]["name"] for m in list(members)[:_REPORT_MAX_ENTITIES]
                 if m in art.entities]
    lines.append("Entities: " + ", ".join(ent_names))

    rels: list[str] = []
    seen = set()
    for m in members:
        for nb in art.adjacency.get(m, []):
            if nb["neighbor"] in members:
                key = tuple(sorted((m, nb["neighbor"]))) + (nb.get("rel_type"),)
                if key in seen:
                    continue
                seen.add(key)
                a = art.entities[m]["name"]
                b = art.entities.get(nb["neighbor"], {}).get("name", nb["neighbor"])
                rels.append(f"{a} —{nb.get('name') or nb.get('rel_type')}→ {b}")
                if len(rels) >= _REPORT_MAX_RELS:
                    break
        if len(rels) >= _REPORT_MAX_RELS:
            break
    if rels:
        lines.append("Relationships: " + "; ".join(rels))

    sub = [communities[ch].get("report") for ch in children.get(cid, [])
           if communities[ch].get("report")][:_REPORT_MAX_SUBREPORTS]
    if sub:
        lines.append("Sub-community reports:\n" + "\n".join(f"- {s}" for s in sub))

    return "\n".join(lines)[:_REPORT_CTX_CHARS]


def _member_key(members: set[str]) -> str:
    """Stable cache key from a community's member set — survives Leiden ID reassignment."""
    import hashlib
    return hashlib.sha1("|".join(sorted(members)).encode()).hexdigest()[:16]


def generate_reports(art: GraphArtifacts, communities: dict[int, dict], *, model: str, llm=None,
                     cache_path: Path | None = None, limit: int | None = None,
                     reuse_only: bool = False, workers: int = 8) -> None:
    """Fill title/report/rank into each community with LLM community reports (Microsoft GraphRAG step).

      * the cache is keyed by (member set, MODEL) with the FULL model id: reports written by another model are never
        reused, so swapping the answer model cannot silently leak old reports into the answer prompt;
      * communities of the same hierarchy level are independent, so each level runs on `workers` threads
        (deepest level first, so a parent still sees its children's reports);
      * `reuse_only=True` applies cached reports and makes no LLM call.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    children: dict[int, list[int]] = defaultdict(list)
    for cid, c in communities.items():
        if c["parent"] is not None:
            children[c["parent"]].append(cid)

    cache_path = cache_path or (CACHE_DIR / f"reports_{art.source.name}.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if cache_path.exists():
        for line in read_lines(cache_path):
            if line.strip():
                obj = json.loads(line)
                if obj.get("member_key") and obj.get("model") == model:   # other models' reports are ignored
                    done[obj["member_key"]] = obj

    by_level: dict[int, list[int]] = defaultdict(list)
    ids = sorted(communities, key=lambda cid: -communities[cid]["level"])
    if limit is not None:
        ids = ids[:limit]
    for cid in ids:
        by_level[communities[cid]["level"]].append(cid)

    lock = threading.Lock()
    stats = {"reused": 0, "generated": 0, "skipped": 0}
    fh = cache_path.open("a", encoding="utf-8")

    def apply(cid: int, entry: dict) -> None:
        communities[cid].update(title=entry.get("title"), report=entry.get("report"), rank=entry.get("rank"))

    def work(cid: int) -> None:
        mk = _member_key(communities[cid]["members"])
        with lock:
            cached = done.get(mk)
        if cached:
            apply(cid, cached)
            with lock:
                stats["reused"] += 1
            return
        if reuse_only:
            with lock:
                stats["skipped"] += 1
            return
        ctx = _report_context(cid, art, communities, children)
        try:
            out = _extract_json(llm.chat(_REPORT_SYSTEM, ctx)["content"])
        except Exception:                                   # a failed call falls back to the raw context
            out = {}
        title = out.get("title") or f"Community {cid}"
        report = out.get("summary") or ctx[:500]
        try:
            rank = float(out.get("rating"))
        except (TypeError, ValueError):
            rank = 0.0
        entry = {"member_key": mk, "model": model, "title": title, "report": report, "rank": rank}
        apply(cid, entry)
        with lock:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()                                      # at most the in-flight calls are lost on an abrupt stop
            done[mk] = entry
            stats["generated"] += 1
            if stats["generated"] % 100 == 0:
                print(f"    community reports generated: {stats['generated']} "
                      f"(reused {stats['reused']}) of {len(ids)}")

    try:
        for level in sorted(by_level, reverse=True):         # deepest first
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                list(pool.map(work, by_level[level]))
    finally:
        fh.close()
    print(f"  community reports ({model}): {stats['generated']} generated, {stats['reused']} reused"
          + (f", {stats['skipped']} without report (reuse-only)" if stats["skipped"] else ""))


# ---------------------------------------------------------------------------
# Persistence: JSON side-map + `_leiden` Neo4j database
# ---------------------------------------------------------------------------

def save_cache(art: GraphArtifacts, communities: dict[int, dict],
               node_communities: dict[str, list[int]]) -> Path:
    """Write the query-time side-map: community meta + entity→communities."""
    path = community_cache_path(art.source.name)
    meta = {str(cid): {"level": c["level"], "parent": c["parent"], "size": c["size"],
                       "title": c.get("title"), "report": c.get("report"),
                       "rank": c.get("rank", 0.0)}
            for cid, c in communities.items()}
    payload = {"community_meta": meta,
               "entity_to_communities": {eid: cids for eid, cids in node_communities.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"  community side-map → {path}")
    return path
