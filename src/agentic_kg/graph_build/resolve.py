"""Entity resolution / canonicalization.

Precision-first: the goal is to merge obvious surface-form variants of the SAME entity
(so multi-hop chains connect) WITHOUT over-merging distinct entities (which silently
corrupts answers — e.g. fusing a father and son who share a name with a patronymic suffix,
"Stefan Nemanja" vs "Stefan Nemanjić", into one node with contradictory birth/death dates).

Pipeline, cheapest/most-precise first:

  1. Exact-normalized-string match (Resolver.add) — entities sharing a normalized name.
  2. CANDIDATE generation — two independent sources of "these might be the same entity":
       - title_merge_candidates(): a passage title that differs from the entity most central
         to that passage's own extracted triples (e.g. title "Anastasia of Serbia", lede "Ana").
       - find_embedding_candidates(): cross-passage near-duplicates via E5 cosine similarity
         (mutual top-1 nearest neighbor, same-type-only) — catches transliteration/honorific
         splits exact matching cannot, e.g. "Yelizaveta Mavrikievna" vs "Elizabeth Mavrikievna".
     Both are heuristics and may be WRONG; neither merges anything by itself.
  3. safe_merge() — the single gate every candidate pair must pass before being committed: a
     CONTRADICTION VETO checks each pair's existing facts (from the edges already extracted)
     for mutually-exclusive relations (a real person has at most one birth date and one death
     date); a pair whose dates disagree is refused regardless of name/embedding similarity.
     Survivors are unioned (bounded group size, so no merge can transitively chain into an
     unrelated "supercluster" — a failure mode observed in testing: two unrelated names each
     absorbing thousands of others via naive single-link clustering).

The public surface used by the KG builders:
  - normalize_name(name)  -> normalized key (lowercase, NFKC, strip determiners/punct)
  - Resolver.add(name, type) -> canonical_id   (accumulates aliases/type)
  - Resolver.entities() -> list of {id, name, type, aliases}
  - title_merge_candidates(passages, cache) -> list of (title_id, primary_id) candidate pairs
  - find_embedding_candidates(entities, threshold) -> list of (id_a, id_b) candidate pairs
  - safe_merge(entities, edges, candidate_pairs) -> {old_id: canonical_id} sparse, vetted remap
  - apply_id_remap(entities, edges, mentions, id_remap) -> folds a safe_merge() remap in place
"""

import re
import unicodedata

_ARTICLE_PREFIX_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_name(name: str) -> str:
    """Normalized comparison key for an entity surface form."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", str(name)).strip().lower()
    s = _ARTICLE_PREFIX_RE.sub("", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _slug(norm: str) -> str:
    return _WS_RE.sub("_", norm).strip("_") or "_blank"


_REL_TOKEN_RE = re.compile(r"[^0-9a-zA-Z]+")


def rel_type_token(relation: str) -> str:
    """Turn a relation phrase into a valid Neo4j relationship type (UPPER_SNAKE).

    e.g. "date of birth" -> DATE_OF_BIRTH, "country of citizenship" -> COUNTRY_OF_CITIZENSHIP.
    Used so edges carry the real relation name AS the edge type (queried via the type itself,
    not a generic [:REL {type}] property).
    """
    s = _REL_TOKEN_RE.sub("_", (relation or "").strip()).strip("_").upper()
    return s or "RELATED_TO"


class Resolver:
    """Accumulates surface forms into canonical entities keyed by EXACT normalized name only.

    Pure exact-match — no cross-string aliasing happens here (that is the job of the
    candidate-generation + safe_merge() pipeline below, which can fact-check a merge before
    committing it). `title_hint` only affects which exact-matching surface form is preferred
    as the display name; it does not create cross-string links.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, dict] = {}

    def add(self, name: str, type: str | None = None, title_hint: bool = False) -> str:
        key = normalize_name(name)
        if not key:
            key = normalize_name("unknown entity")
        ent = self._by_key.get(key)
        if ent is None:
            ent = {"id": _slug(key), "name": name.strip(), "type": type or "Misc",
                   "aliases": set(), "_title_named": title_hint}
            self._by_key[key] = ent
        else:
            ent["aliases"].add(name.strip())
            # Upgrade display name to a title-sourced surface form if we get one later.
            if title_hint and not ent["_title_named"]:
                ent["aliases"].add(ent["name"])
                ent["name"] = name.strip()
                ent["_title_named"] = True
            # Fill in a type if we previously only had the default.
            if type and ent["type"] in (None, "Misc"):
                ent["type"] = type
        ent["aliases"].discard(ent["name"])
        return ent["id"]

    def id_for(self, name: str) -> str | None:
        ent = self._by_key.get(normalize_name(name))
        return ent["id"] if ent else None

    def entities(self) -> list[dict]:
        return [
            {"id": e["id"], "name": e["name"], "type": e["type"],
             "aliases": sorted(e["aliases"])}
            for e in self._by_key.values()
        ]


# ---------------------------------------------------------------------------
# Candidate generation — heuristics that PROPOSE merges (never commit one directly)
# ---------------------------------------------------------------------------

def title_merge_candidates(passages: dict, cache: dict, resolver: "Resolver") -> list[tuple[str, str]]:
    """Propose (title_id, primary_id) pairs: a passage's title vs. the entity most central
    to that SAME passage's own extracted triples (highest subject/object degree).

    A Wikipedia-style lede often names its subject differently from the article title (e.g.
    title "Anastasia of Serbia", lede "Ana") — the most-connected entity in the passage's own
    extraction is almost always the real subject. Registers the title as a (possibly bare)
    entity via resolver.add() so it has an id to propose a merge for; does NOT alias it.
    Must be called AFTER all passages' entities/triples have been resolver.add()-ed, so
    primary_id lookups succeed and any pre-existing facts under the title string itself are
    already in the graph for safe_merge()'s contradiction check to see.
    """
    candidates: list[tuple[str, str]] = []
    for pid, p in passages.items():
        ext = cache.get(pid)
        triples = (ext or {}).get("triples") or []
        ents = (ext or {}).get("entities") or []
        title_id = resolver.add(p["title"], type=None, title_hint=True)
        if not triples and not ents:
            continue  # nothing extracted from this passage — bare title anchor only
        degree: dict[str, int] = {}
        for tr in triples:
            for k in ("subject", "object"):
                nm = (tr.get(k) or "").strip()
                if nm:
                    degree[nm] = degree.get(nm, 0) + 1
        primary_name = max(degree, key=degree.get) if degree else \
            ((ents[0].get("name") or "").strip() or p["title"])
        primary_id = resolver.id_for(primary_name)
        if primary_id and primary_id != title_id:
            candidates.append((title_id, primary_id))
    return candidates


_EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"  # same "vanilla" E5 alias as methods/dense


def find_embedding_candidates(entities: list[dict], edges: set[tuple], threshold: float,
                              top_k: int = 8, model_name: str = _EMBED_MODEL_NAME,
                              min_subject_edges: int = 1) -> list[tuple[str, str]]:
    """Propose cross-passage near-duplicate pairs via E5 embedding similarity.

    Catches transliteration / honorific-wrapper splits exact-string matching cannot, e.g.
    "Yelizaveta Mavrikievna" vs "Elizabeth Mavrikievna" vs "Grand Duchess Elizabeth
    Mavrikievna of Russia" — same person, three nodes under normalize_name() alone.

    Criterion is intentionally strict, by construction NOT just "above threshold":
      - MUTUAL top-1 nearest neighbor only — i and j are proposed ONLY if i's best non-self
        match is j AND j's best non-self match is i (both above `threshold`, same exact
        `type`, no cross-type wildcard). A naive "pair up whenever any top-k match exceeds
        threshold" performs single-link clustering, which can chain transitively through weak
        intermediate matches into a runaway "supercluster" (observed in testing: an unrelated
        name absorbing thousands of others this way). Mutual-top-1 can only ever propose PAIRS.
      - MINIMUM OWN EVIDENCE — both entities must have at least `min_subject_edges` of their
        OWN extracted triples as subject. An entity that exists only as an OBJECT mention in
        someone else's facts (e.g. "Stefan Nemanja abdicated in favor of his son Stefan
        Nemanjić" — Nemanjić never appears as a subject of his own facts in the sample) has
        nothing to fact-check a merge against, so name/embedding similarity alone is
        insufficient evidence — exactly the case (father vs. son sharing a patronymic-suffixed
        name) where the contradiction veto below has nothing to compare and would wrongly let
        the merge through.

    This function only PROPOSES pairs — every candidate must still pass safe_merge()'s
    fact-contradiction check before committing.
    """
    if len(entities) < 2:
        return []

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    subject_degree: dict[str, int] = {}
    for (s, _o, _rt, _rn, _p) in edges:
        subject_degree[s] = subject_degree.get(s, 0) + 1

    ids = [e["id"] for e in entities]
    names = [e["name"] for e in entities]
    types = [e.get("type") or "Misc" for e in entities]
    has_evidence = [subject_degree.get(eid, 0) >= min_subject_edges for eid in ids]

    model = SentenceTransformer(model_name)
    emb = model.encode([f"query: {n}" for n in names], batch_size=128,
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False).astype(np.float32)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    k = min(top_k + 1, len(entities))  # +1 — each vector's nearest neighbor is itself
    scores, idxs = index.search(emb, k)

    def best_match(i: int) -> int | None:
        if not has_evidence[i]:
            return None
        for score, j in zip(scores[i], idxs[i]):
            j = int(j)
            if j < 0 or j == i:
                continue
            if score < threshold:
                break  # scores are sorted descending — no further candidate can qualify
            if types[i] == types[j] and has_evidence[j]:
                return j
        return None

    best = [best_match(i) for i in range(len(entities))]
    return [(ids[i], ids[j]) for i, j in enumerate(best)
            if j is not None and best[j] == i and j > i]


# ---------------------------------------------------------------------------
# safe_merge — the single fact-checked gate every candidate pair must pass
# ---------------------------------------------------------------------------

# Relation-PHRASE substrings (not the post-consolidation edge type) that denote a relation a
# real person/entity can have AT MOST ONE value of (in this dataset's historical-figure
# context). Checked against the raw `name` (human phrase) carried on each edge, since this
# runs before relation consolidation. Each bucket = (tokens ALL required, tokens ANY required).
# Deliberately excludes multi-valued relations like SPOUSE (remarriage is real) — a person
# having several different recorded spouses is not by itself contradictory; it just means
# multiple unrelated people sharing a common bare given name (e.g. five different historical
# "Ana"s) got merged on name similarity with nothing yet to individually contradict. Dates
# alone proved too narrow for that case (the merged entities often lack independent
# birth/death dates of their own); place-of-birth/death and nationality catch more of it.
_EXCLUSIVE_PHRASE_BUCKETS = {
    "birth_date":  (("date",),  ("birth", "born")),
    "death_date":  (("date",),  ("death", "died")),
    "birth_place": (("place",), ("birth", "born")),
    "death_place": (("place",), ("death", "died")),
    "nationality": ((),         ("nationality", "citizenship")),
    "start_date":  (("date",),  ("start",)),
    "end_date":    (("date",),  ("end",)),
}


def load_relation_map(path) -> dict[str, str]:
    """Load a consolidate.py relation_map.json (raw phrase -> canonical phrase), if present.

    Reusing the REAL, LLM-derived canonicalization (instead of a hand-written heuristic like
    stripping prepositions) is what correctly recognizes "spouse" and "spouse of" as the same
    relation for the veto checks below — consolidation already knows this; the only issue was
    that it previously ran AFTER entity resolution, too late for safe_merge() to benefit from
    it. Returns {} if the file doesn't exist yet (e.g. the very first build of a benchmark,
    before any consolidation has run) — callers degrade gracefully to raw-phrase matching.
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import json
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {str(k).strip().lower(): str(v).strip().lower() for k, v in raw.items()}
    except Exception:
        return {}


def _canon_phrase(rel_name: str, relation_canon: dict[str, str] | None) -> str:
    phrase = (rel_name or "").strip().lower()
    if relation_canon:
        phrase = relation_canon.get(phrase, phrase)
    return phrase


def _exclusive_facts(edges_by_subject: dict[str, list[tuple[str, str]]],
                     entity_id: str, relation_canon: dict[str, str] | None = None) -> dict[str, set[str]]:
    """{'birth_date': {...}, 'death_date': {...}} object-value sets for one entity's edges."""
    buckets: dict[str, set[str]] = {}
    for rel_name, obj in edges_by_subject.get(entity_id, []):
        phrase = _canon_phrase(rel_name, relation_canon)
        for bucket, (required, any_of) in _EXCLUSIVE_PHRASE_BUCKETS.items():
            if all(r in phrase for r in required) and any(a in phrase for a in any_of):
                buckets.setdefault(bucket, set()).add(obj.strip().lower())
    return buckets


def _conflicts(facts_a: dict[str, set[str]], facts_b: dict[str, set[str]]) -> bool:
    """True if any shared exclusive-fact bucket has non-overlapping values on both sides."""
    for bucket, va in facts_a.items():
        vb = facts_b.get(bucket)
        if va and vb and not (va & vb):
            return True
    return False


# Vague catch-all relation phrases the LLM emits directly when it can't pin down a specific
# relation (e.g. "X related to Y"). Too noisy to be reliable evidence either for or against a
# merge — two genuinely identical entities can easily have non-overlapping "related to" facts
# (different passages noted different loose associations), so it is excluded from the veto
# checks below entirely (both as a basis for "already connected" and for "shared answer").
_VAGUE_RELATION_PHRASES = {"related to"}


def _all_facts_by_relation(edges_by_subject: dict[str, list[tuple[str, str]]],
                           entity_id: str,
                           relation_canon: dict[str, str] | None = None) -> dict[str, set[str]]:
    """{relation_phrase_lower: {object_value_lower, ...}} for ALL of entity_id's own
    subject-edges — every relation type (spouse, occupation, member_of, ...), not just the
    curated exclusive-bucket list above. Excludes _VAGUE_RELATION_PHRASES. When `relation_canon`
    is given (see load_relation_map()), phrases are canonicalized first so variants like
    "spouse"/"spouse of" bucket together instead of silently never being compared.
    """
    buckets: dict[str, set[str]] = {}
    for rel_name, obj in edges_by_subject.get(entity_id, []):
        phrase = _canon_phrase(rel_name, relation_canon)
        val = (obj or "").strip().lower()
        if phrase and val and phrase not in _VAGUE_RELATION_PHRASES:
            buckets.setdefault(phrase, set()).add(val)
    return buckets


def _any_relation_has_no_shared_answer(facts_a: dict[str, set[str]],
                                       facts_b: dict[str, set[str]]) -> bool:
    """Check EVERY relation type independently: if BOTH candidates have at least one value
    recorded for the SAME relation (e.g. both have a "spouse"), they must share AT LEAST ONE
    common value for it. A relation present on only ONE side is neglected (no information to
    compare, not a conflict). If even a single shared relation type has zero overlap, that is
    treated as evidence the two candidates are different real-world entities — e.g. one
    "spouse" value with nothing in common, or one "occupation" value with nothing in common.
    """
    for relation, va in facts_a.items():
        vb = facts_b.get(relation)
        if not va or not vb:
            continue  # relation absent on one side — nothing to compare, not a conflict
        if not (va & vb):
            return True
    return False


def safe_merge(entities: list[dict], edges: set[tuple],
               candidate_pairs: list[tuple[str, str]],
               max_group_size: int = 4,
               relation_canon: dict[str, str] | None = None,
               identity_relations: set[str] | None = None) -> dict[str, str]:
    """The single gate every proposed (id_a, id_b) merge must pass before being committed.

    `relation_canon` (see load_relation_map()): when given, relation phrases are canonicalized
    before bucketing in the contradiction/shared-answer checks below, so wording variants like
    "spouse"/"spouse of" are recognized as the SAME relation instead of silently never being
    compared against each other.

    `identity_relations`: relation phrases (already-canonical, lowercase) that themselves ASSERT
    "these two are the same referent" — e.g. "birth_name"/"full_name" — as opposed to ordinary
    relations between two DIFFERENT entities. Exempts those edges from the already-connected veto
    below (see its docstring for why an ordinary edge is normally proof of distinctness) so the
    very edge that proposed the candidate pair doesn't automatically veto it. Does not affect the
    contradiction/shared-answer checks — those still see identity-relation facts normally.

    Checks run GROUP-AWARE, not just pairwise on the original candidate: before unioning, the
    candidate is checked against EVERY member of both prospective groups (not only id_a/id_b
    themselves). This is necessary because union-find can otherwise transitively fuse two
    genuinely conflicting entities through a shared WEAK intermediate that was never itself in
    conflict with either side (e.g. a thin "Anna2" with no spouse fact merges safely with both
    "Ana" [spouse: Nemanja] and "Anna" [spouse: Enno] individually, yet "Ana" and "Anna"
    themselves were never directly compared and do conflict). The unsafe addition is refused
    while the already-validated parts of the group stay merged.

    1. ALREADY-CONNECTED VETO: if any pair across the two groups is already linked by a direct,
       non-identity edge (in either direction), refuse — a real entity is (in our extraction
       style) never the subject of a relation pointing at itself, so an existing edge between two
       candidates is direct relational proof they are DIFFERENT entities. Catches a father/son
       pair sharing a patronymic-suffix name (e.g. "Stefan Nemanja"/"Stefan Nemanjić", linked
       by an extracted "abdicated in favor of" edge) even with no independently contradicting
       fact to compare.
    2. CONTRADICTION VETO: if any pair has a birth/death-date fact and the values disagree,
       refuse.
    3. SHARED-ANSWER VETO: for EVERY OTHER relation type independently (spouse, occupation,
       member_of, ...) — if both sides of any pair have at least one recorded value for the
       SAME relation, they must share at least one common value; a relation present on only
       one side is neglected (nothing to compare). Even a single shared relation type with
       zero overlap (e.g. two completely different "spouse" values) refuses the union.
    4. Survivors are unioned with a bounded group size (defense in depth — costs nothing and
       guards against any future candidate source producing larger fan-out).

    Returns a sparse {old_id: canonical_id} remap; ids absent are unchanged.
    """
    if not candidate_pairs:
        return {}

    non_proof_phrases = _VAGUE_RELATION_PHRASES | (identity_relations or set())
    connected_pairs = {frozenset((s, o)) for (s, o, _rt, rn, _p) in edges
                      if s != o and (rn or "").strip().lower() not in non_proof_phrases}
    edges_by_subject: dict[str, list[tuple[str, str]]] = {}
    for (s, o, _rt, rn, _p) in edges:
        edges_by_subject.setdefault(s, []).append((rn, o))

    index_of = {e["id"]: i for i, e in enumerate(entities)}
    parent = list(range(len(entities)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _pair_conflicts(id_x: str, id_y: str) -> str | None:
        """Returns which veto fires for (id_x, id_y), or None if the pair is compatible."""
        if frozenset((id_x, id_y)) in connected_pairs:
            return "connected"
        if _conflicts(_exclusive_facts(edges_by_subject, id_x, relation_canon),
                      _exclusive_facts(edges_by_subject, id_y, relation_canon)):
            return "contradiction"
        if _any_relation_has_no_shared_answer(
                _all_facts_by_relation(edges_by_subject, id_x, relation_canon),
                _all_facts_by_relation(edges_by_subject, id_y, relation_canon)):
            return "no_shared"
        return None

    vetoed_connected = vetoed_contradiction = vetoed_no_shared = rejected_size = 0
    for id_a, id_b in candidate_pairs:
        ia, ib = index_of.get(id_a), index_of.get(id_b)
        if ia is None or ib is None:
            continue
        ra, rb = find(ia), find(ib)
        if ra == rb:
            continue

        # Check the candidate against EVERY member of BOTH prospective groups, not just the
        # originally-proposed (id_a, id_b) pair. Union-find can otherwise transitively chain
        # two never-directly-compared, genuinely conflicting entities through a shared WEAK
        # intermediate (e.g. a thin "Anna2" with no spouse fact of its own merges safely with
        # both "Ana" [spouse: Nemanja] and "Anna" [spouse: Enno] individually — neither
        # pairwise check alone sees a conflict — yet the union-find still fuses all three,
        # even though "Ana" and "Anna" themselves were never checked and do conflict).
        group_a = [entities[x]["id"] for x in range(len(entities)) if find(x) == ra]
        group_b = [entities[x]["id"] for x in range(len(entities)) if find(x) == rb]
        reason = None
        for ga in group_a:
            for gb in group_b:
                reason = _pair_conflicts(ga, gb)
                if reason:
                    break
            if reason:
                break
        if reason == "connected":
            vetoed_connected += 1
            continue
        if reason == "contradiction":
            vetoed_contradiction += 1
            continue
        if reason == "no_shared":
            vetoed_no_shared += 1
            continue

        group_size = len(group_a) + len(group_b)
        if group_size > max_group_size:
            rejected_size += 1
            continue
        parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(len(entities)):
        groups.setdefault(find(i), []).append(i)

    remap: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        # Canonical = richest entity (most aliases already attached), tie-broken deterministically.
        canon_idx = max(members, key=lambda i: (len(entities[i].get("aliases") or []), entities[i]["id"]))
        canon_id = entities[canon_idx]["id"]
        for i in members:
            if entities[i]["id"] != canon_id:
                remap[entities[i]["id"]] = canon_id

    if vetoed_connected:
        print(f"    (already-connected veto: blocked {vetoed_connected} merge(s) — the two "
              f"candidates already have a direct edge to each other, so they're different "
              f"entities by definition)")
    if vetoed_contradiction:
        print(f"    (contradiction veto: blocked {vetoed_contradiction} merge(s) with "
              f"conflicting birth/death-date facts — likely distinct entities despite "
              f"similar names)")
    if vetoed_no_shared:
        print(f"    (shared-answer veto: blocked {vetoed_no_shared} merge(s) — a relation "
              f"type present on both sides (e.g. spouse, occupation) had zero overlapping "
              f"values)")
    if rejected_size:
        print(f"    (safety net: rejected {rejected_size} merge(s) that would exceed "
              f"max_group_size={max_group_size})")
    return remap


def apply_id_remap(entities: list[dict], edges: set[tuple], mentions: set[tuple],
                   id_remap: dict[str, str]) -> tuple[list[dict], set[tuple], set[tuple]]:
    """Fold merged entity ids (from safe_merge()) into their canonical survivor across
    entities/edges/mentions. Shared by every caller of safe_merge() (build_openie's own
    title/embedding merges, and consolidate.py's relation-signalled merges)."""
    if not id_remap:
        return entities, edges, mentions

    def final(eid: str) -> str:
        seen = set()
        while eid in id_remap and eid not in seen:
            seen.add(eid)
            eid = id_remap[eid]
        return eid

    by_id = {e["id"]: e for e in entities}
    merged: dict[str, dict] = {}
    for e in entities:
        canon = final(e["id"])
        base = by_id.get(canon)
        if base is None:
            continue
        if canon not in merged:
            merged[canon] = {"id": canon, "name": base["name"], "type": base.get("type"),
                             "aliases": set(base.get("aliases") or [])}
        if e["id"] != canon:
            merged[canon]["aliases"].add(e["name"])
            merged[canon]["aliases"].update(e.get("aliases") or [])

    new_entities = [
        {"id": v["id"], "name": v["name"], "type": v["type"],
         "aliases": sorted(a for a in v["aliases"] if a != v["name"])}
        for v in merged.values()
    ]
    new_edges = {(final(s), final(o), rt, rn, p) for (s, o, rt, rn, p) in edges}
    new_mentions = {(final(e), p) for (e, p) in mentions}
    return new_entities, new_edges, new_mentions
