"""Load graph artifacts (entities / edges / mentions / text units) for GraphRAG local search.

``from_kg_jsonl`` reads the knowledge graph from ``kg.jsonl`` (entities, dynamically typed edges and
entity->passage mentions); the text units (passages) come from the retrieval corpus. ``doc_id`` is the join key
(``Passage:<id>``).

It applies three inline passes that mirror ``graph_build/consolidate.py`` + ``graph_build/resolve.py``, so the
loader is self-contained regardless of whether consolidation has been run on the source ``kg.jsonl``:

  1. Relation canonicalization — applies ``relation_map.json`` (raw phrase -> canonical type/name). On the shipped,
     already consolidated graph this is a second pass; it turns 84 PRODUCTION_COMPANY edges into RELATED_TO in the
     GraphRAG graph only. It is kept because the paper's GraphRAG numbers were produced with it.
  2. Self-loop filtering — drops any edge whose subject == object.
  3. Alias/merge pass with three-veto gate — detects alias-signalling edges (also_known_as, birth_name, ...),
     cross-aliases their endpoints, proposes merges, and runs them through the same three-veto gate as
     ``resolve.safe_merge()``:
       a. already-connected veto: two candidates with a direct non-identity edge are different entities;
       b. contradiction veto: conflicting birth/death date, place, nationality, start/end date;
       c. shared-answer veto: for any relation type BOTH sides have facts about, they must share a common
          object value; zero overlap = different entities.
     Survivors are union-found with a bounded group cap (max_group=4) and remapped.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ...io import read_lines
from .config import Source

_REL_TOKEN_RE = re.compile(r"[^\w]+")


def _rel_type_token(s: str) -> str:
    return (_REL_TOKEN_RE.sub("_", (s or "").strip()).strip("_").upper()) or "RELATED_TO"


def _load_relation_map(kg_path: Path) -> dict[str, tuple[str, str]]:
    """Load relation_map.json → {raw_phrase: (UPPER_SNAKE_type, canonical_name_lower)}.

    Returns {} if the map file does not exist (graceful degradation).
    """
    map_path = kg_path.parent / "relation_map.json"
    if not map_path.exists():
        return {}
    raw: dict[str, str] = json.loads(map_path.read_text(encoding="utf-8"))
    result: dict[str, tuple[str, str]] = {}
    for phrase, canonical in raw.items():
        c = (canonical or "").strip()
        if not c or c.lower() == "other":
            result[phrase] = ("RELATED_TO", "related to")
        else:
            result[phrase] = (_rel_type_token(c), c.lower())
    return result


# ---------------------------------------------------------------------------
# Veto-gated alias/merge pass
# (inlined from bench/build/resolve.py + consolidate.py so the loader is
#  self-contained and does not import build-pipeline code)
# ---------------------------------------------------------------------------

_ALIAS_ONLY_KEYWORDS = ("abbreviation", "acronym")
_ALIAS_MERGE_KEYWORDS = (
    "also_known_as", "alias", "aka", "pseudonym", "nickname", "pen_name", "stage_name",
    "birth_name", "maiden_name", "née", "nee_", "full_name", "real_name", "legal_name",
    "formal_name",
)

# Relations a real entity can have at most ONE value of.  Each bucket is
# (tokens ALL required in phrase, tokens ANY required in phrase).
_EXCLUSIVE_PHRASE_BUCKETS: dict[str, tuple[tuple, tuple]] = {
    "birth_date":  (("date",),  ("birth", "born")),
    "death_date":  (("date",),  ("death", "died")),
    "birth_place": (("place",), ("birth", "born")),
    "death_place": (("place",), ("death", "died")),
    "nationality": ((),         ("nationality", "citizenship")),
    "start_date":  (("date",),  ("start",)),
    "end_date":    (("date",),  ("end",)),
}

_VAGUE_RELATION_PHRASES = {"related to"}


def _kw_match(name: str, keywords: tuple) -> bool:
    n = (name or "").lower()
    return any(k in n for k in keywords)


def _canon_rel(name: str, relation_canon: dict[str, str] | None) -> str:
    """Lower-case phrase, optionally canonicalized via relation_map."""
    phrase = (name or "").strip().lower()
    return relation_canon.get(phrase, phrase) if relation_canon else phrase


def _excl_facts(by_subj: dict[str, list[tuple]], eid: str,
                relation_canon: dict[str, str] | None) -> dict[str, set[str]]:
    """Exclusive-bucket fact sets for one entity (birth_date, death_date, …)."""
    out: dict[str, set[str]] = {}
    for rname, obj in by_subj.get(eid, []):
        phrase = _canon_rel(rname, relation_canon)
        for bk, (req, any_of) in _EXCLUSIVE_PHRASE_BUCKETS.items():
            if all(r in phrase for r in req) and any(a in phrase for a in any_of):
                out.setdefault(bk, set()).add((obj or "").strip().lower())
    return out


def _all_facts(by_subj: dict[str, list[tuple]], eid: str,
               relation_canon: dict[str, str] | None) -> dict[str, set[str]]:
    """All relation→value facts for one entity (vague phrases excluded)."""
    out: dict[str, set[str]] = {}
    for rname, obj in by_subj.get(eid, []):
        phrase = _canon_rel(rname, relation_canon)
        val = (obj or "").strip().lower()
        if phrase and val and phrase not in _VAGUE_RELATION_PHRASES:
            out.setdefault(phrase, set()).add(val)
    return out


def _has_exclusive_conflict(fa: dict[str, set[str]], fb: dict[str, set[str]]) -> bool:
    for bk, va in fa.items():
        vb = fb.get(bk)
        if va and vb and not (va & vb):
            return True
    return False


def _no_shared_answer(fa: dict[str, set[str]], fb: dict[str, set[str]]) -> bool:
    for rel, va in fa.items():
        vb = fb.get(rel)
        if va and vb and not (va & vb):
            return True
    return False


def _cross_alias(entity_by_id: dict[str, dict], id_a: str, id_b: str) -> None:
    """Add each entity's name+aliases to the OTHER's alias list (no id change)."""
    ea, eb = entity_by_id.get(id_a), entity_by_id.get(id_b)
    if not ea or not eb:
        return
    names_a = {ea["name"], *(ea.get("aliases") or [])}
    names_b = {eb["name"], *(eb.get("aliases") or [])}
    ea["aliases"] = sorted((set(ea.get("aliases") or []) | names_b) - {ea["name"]})
    eb["aliases"] = sorted((set(eb.get("aliases") or []) | names_a) - {eb["name"]})


def _vetoed_merge(
    entities_list: list[dict],
    edges_tuples: set[tuple],
    candidate_pairs: list[tuple[str, str]],
    relation_canon: dict[str, str] | None = None,
    identity_relations: set[str] | None = None,
    max_group: int = 4,
) -> dict[str, str]:
    """Union-find merge gated by three vetoes.  Mirrors resolve.safe_merge() faithfully.

    Three vetoes (in order):
      1. Already-connected — a direct non-identity edge between candidates proves they
         are DIFFERENT entities (a real entity never has a relation pointing at itself).
      2. Contradiction — conflicting exclusive-bucket facts (birth/death date/place, …).
      3. Shared-answer — any relation type BOTH sides have facts about must share ≥1
         common object value; zero overlap = different entities.

    Checks are group-aware: before unioning, EVERY pair across the two prospective groups
    is tested, not just the originally proposed (id_a, id_b).  This prevents a thin
    intermediate entity from silently chaining two genuinely conflicting groups.
    """
    if not candidate_pairs:
        return {}

    non_proof = _VAGUE_RELATION_PHRASES | (identity_relations or set())
    connected = {
        frozenset((s, o)) for (s, o, _rt, rn, _p) in edges_tuples
        if s != o and (rn or "").strip().lower() not in non_proof
    }
    by_subj: dict[str, list[tuple]] = {}
    for (s, o, _rt, rn, _p) in edges_tuples:
        by_subj.setdefault(s, []).append((rn or "", o))

    idx = {e["id"]: i for i, e in enumerate(entities_list)}
    parent = list(range(len(entities_list)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def pair_veto(ia: str, ib: str) -> str | None:
        if frozenset((ia, ib)) in connected:
            return "connected"
        if _has_exclusive_conflict(_excl_facts(by_subj, ia, relation_canon),
                                   _excl_facts(by_subj, ib, relation_canon)):
            return "contradiction"
        if _no_shared_answer(_all_facts(by_subj, ia, relation_canon),
                             _all_facts(by_subj, ib, relation_canon)):
            return "no_shared"
        return None

    vc = vd = vs = vg = 0
    for id_a, id_b in candidate_pairs:
        ia, ib = idx.get(id_a), idx.get(id_b)
        if ia is None or ib is None:
            continue
        ra, rb = find(ia), find(ib)
        if ra == rb:
            continue

        grp_a = [entities_list[x]["id"] for x in range(len(entities_list)) if find(x) == ra]
        grp_b = [entities_list[x]["id"] for x in range(len(entities_list)) if find(x) == rb]
        reason = None
        for ga in grp_a:
            for gb in grp_b:
                reason = pair_veto(ga, gb)
                if reason:
                    break
            if reason:
                break

        if reason == "connected":
            vc += 1; continue
        if reason == "contradiction":
            vd += 1; continue
        if reason == "no_shared":
            vs += 1; continue
        if len(grp_a) + len(grp_b) > max_group:
            vg += 1; continue

        parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(len(entities_list)):
        groups.setdefault(find(i), []).append(i)

    remap: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canon_i = max(members,
                      key=lambda i: (len(entities_list[i].get("aliases") or []),
                                     entities_list[i]["id"]))
        canon_id = entities_list[canon_i]["id"]
        for i in members:
            if entities_list[i]["id"] != canon_id:
                remap[entities_list[i]["id"]] = canon_id

    if vc: print(f"  [loader] already-connected veto: blocked {vc} merge(s)")
    if vd: print(f"  [loader] contradiction veto: blocked {vd} merge(s)")
    if vs: print(f"  [loader] shared-answer veto: blocked {vs} merge(s)")
    if vg: print(f"  [loader] group-size cap: blocked {vg} merge(s)")
    return remap


def _apply_merge_remap(
    entities: dict[str, dict],
    edges: list[dict],
    entity_passages: dict[str, set[str]],
    passage_entities: dict[str, set[str]],
    id_remap: dict[str, str],
) -> tuple[dict[str, dict], list[dict], dict[str, set[str]], dict[str, set[str]]]:
    """Fold committed merges into all four artifact structures.

    Mirrors resolve.apply_id_remap() for the loader's in-memory structures.
    """
    def final(eid: str) -> str:
        seen: set[str] = set()
        while eid in id_remap and eid not in seen:
            seen.add(eid)
            eid = id_remap[eid]
        return eid

    # Merge entities: canonical survivor absorbs aliases from deleted ids.
    merged: dict[str, dict] = {}
    for eid, e in entities.items():
        canon = final(eid)
        base = entities.get(canon, e)
        if canon not in merged:
            merged[canon] = {"id": canon, "name": base["name"], "type": base.get("type"),
                             "aliases": set(base.get("aliases") or [])}
        if eid != canon:
            merged[canon]["aliases"].add(e["name"])
            merged[canon]["aliases"].update(e.get("aliases") or [])
    new_entities = {
        eid: {**e, "aliases": sorted(a for a in e["aliases"] if a != e["name"])}
        for eid, e in merged.items()
    }

    # Remap + dedup edges; drop self-loops produced by the merge.
    seen_edges: set[tuple] = set()
    new_edges: list[dict] = []
    for e in edges:
        s, t = final(e["source"]), final(e["target"])
        if s == t:
            continue
        key = (s, t, e.get("rel_type"), e.get("name"), e.get("passage_id"))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        new_edges.append({**e, "source": s, "target": t})

    # Remap entity_passages and passage_entities.
    new_ep: dict[str, set[str]] = defaultdict(set)
    for eid, pids in entity_passages.items():
        new_ep[final(eid)].update(pids)

    new_pe: dict[str, set[str]] = defaultdict(set)
    for pid, eids in passage_entities.items():
        for eid in eids:
            new_pe[pid].add(final(eid))

    return new_entities, new_edges, dict(new_ep), dict(new_pe)


# ---------------------------------------------------------------------------
# GraphArtifacts container
# ---------------------------------------------------------------------------

@dataclass
class GraphArtifacts:
    entities: dict[str, dict]                       # entity_id -> {id, name, type, aliases}
    edges: list[dict]                               # {source, target, rel_type, name, passage_id}
    adjacency: dict[str, list[dict]]                # entity_id -> [{neighbor, rel_type, name, passage_id}]
    degree: dict[str, int]                          # entity_id -> incident-edge count
    entity_passages: dict[str, set[str]]            # entity_id -> {passage doc_id}
    passage_entities: dict[str, set[str]]           # passage doc_id -> {entity_id}
    text_units: dict[str, dict]                     # passage doc_id -> corpus record
    source: Source

    def stats(self) -> str:
        n_rel_ent = sum(1 for e in self.entities if self.degree.get(e, 0) > 0)
        return (f"{len(self.entities)} entities ({n_rel_ent} with edges), "
                f"{len(self.edges)} edges, {len(self.text_units)} text units, "
                f"{sum(len(v) for v in self.entity_passages.values())} mentions")


def _load_corpus(corpus_path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for line in read_lines(corpus_path):
        if line.strip():
            rec = json.loads(line)
            index[rec["doc_id"]] = rec
    return index


def _build_adjacency(edges: list[dict]) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Undirected adjacency + combined-degree (incident-edge count) per entity."""
    adjacency: dict[str, list[dict]] = defaultdict(list)
    degree: dict[str, int] = defaultdict(int)
    for e in edges:
        s, t = e["source"], e["target"]
        adjacency[s].append({"neighbor": t, "rel_type": e["rel_type"],
                             "name": e.get("name"), "passage_id": e.get("passage_id")})
        adjacency[t].append({"neighbor": s, "rel_type": e["rel_type"],
                             "name": e.get("name"), "passage_id": e.get("passage_id")})
        degree[s] += 1
        degree[t] += 1
    return dict(adjacency), dict(degree)


# ---------------------------------------------------------------------------
# Bench adapter: kg.jsonl + corpus
# ---------------------------------------------------------------------------

def from_kg_jsonl(source: Source) -> GraphArtifacts:
    entities: dict[str, dict] = {}
    edges: list[dict] = []
    entity_passages: dict[str, set[str]] = defaultdict(set)
    passage_entities: dict[str, set[str]] = defaultdict(set)

    kg_path = Path(source.kg_path)
    relation_map = _load_relation_map(kg_path)
    n_self_loops = 0
    n_canonical = 0

    def _pid(raw: str | None) -> str | None:
        return f"Passage:{raw}" if raw else None

    for line in read_lines(kg_path):
        if not line.strip():
            continue
        obj = json.loads(line)
        kind = obj.get("kind")
        if kind == "entity":
            entities[obj["id"]] = {
                "id": obj["id"],
                "name": obj.get("name") or obj["id"],
                "type": obj.get("type"),
                "aliases": list(obj.get("aliases") or []),
            }
        elif kind == "edge":
            s, t = obj["subject_id"], obj["object_id"]
            if s == t:
                n_self_loops += 1
                continue
            name = obj.get("name")
            if relation_map and name in relation_map:
                rel_type, name = relation_map[name]
                n_canonical += 1
            else:
                rel_type = obj.get("rel_type") or "RELATED_TO"
            edges.append({
                "source": s,
                "target": t,
                "rel_type": rel_type,
                "name": name,
                "passage_id": _pid(obj.get("passage_id")),
            })
        elif kind == "mention":
            doc_id = _pid(obj.get("passage_id"))
            if doc_id:
                entity_passages[obj["entity_id"]].add(doc_id)
                passage_entities[doc_id].add(obj["entity_id"])

    if n_self_loops:
        print(f"  [loader] dropped {n_self_loops} self-loop edge(s)")
    if relation_map:
        print(f"  [loader] canonicalized {n_canonical}/{len(edges) + n_self_loops} "
              f"edge(s) via relation_map.json ({len(relation_map)} phrase mappings)")

    # --- Alias/merge pass with three-veto gate ---
    # Mirrors consolidate.py _apply_relation_signals() + resolve.safe_merge().
    # relation_canon for the veto checks: {raw_phrase_lower -> canonical_name_lower}
    relation_canon: dict[str, str] | None = (
        {ph.lower(): nm for ph, (_, nm) in relation_map.items()} if relation_map else None
    )
    edges_tuples = {
        (e["source"], e["target"], e["rel_type"], e.get("name"), e.get("passage_id"))
        for e in edges
    }

    merge_candidates: list[tuple[str, str]] = []
    identity_rels: set[str] = set()
    for e in edges:
        name = e.get("name") or ""
        s, t = e["source"], e["target"]
        if _kw_match(name, _ALIAS_ONLY_KEYWORDS):
            _cross_alias(entities, s, t)
        if _kw_match(name, _ALIAS_MERGE_KEYWORDS):
            _cross_alias(entities, s, t)
            merge_candidates.append((s, t))
            identity_rels.add(name.strip().lower())

    if merge_candidates:
        entities_list = list(entities.values())
        id_remap = _vetoed_merge(
            entities_list, edges_tuples, merge_candidates,
            relation_canon=relation_canon, identity_relations=identity_rels,
        )
        if id_remap:
            print(f"  [loader] {len(id_remap)} entity id(s) merged via alias/identity signals")
            entities, edges, entity_passages, passage_entities = _apply_merge_remap(
                entities, edges, dict(entity_passages), dict(passage_entities), id_remap,
            )

    # Final endpoint filter: drop edges whose endpoints are not in the declared entity set.
    edges = [e for e in edges if e["source"] in entities and e["target"] in entities]
    adjacency, degree = _build_adjacency(edges)
    text_units = _load_corpus(source.corpus_path)

    return GraphArtifacts(
        entities=entities, edges=edges, adjacency=adjacency, degree=degree,
        entity_passages=dict(entity_passages), passage_entities=dict(passage_entities),
        text_units=text_units, source=source,
    )


def load_artifacts(source: Source) -> GraphArtifacts:
    return from_kg_jsonl(source)
