"""LLM OpenIE: extract (entity, relation, entity) triples per passage and assemble a KG.

One LLM call per unique passage (stage "openie" of configs/graph_build.yaml), parallelised and cached by
passage_id so each deduped passage is extracted once (and runs are resumable). Extracted
surface entities are canonicalized with build.resolve.Resolver (title-anchored), triples become
typed edges (the relation name is the edge type), and every entity links to the passages it was
extracted from (MENTIONED_IN) for provenance.

Output matches build_kg.build_gold's KG dict shape, so the loader/corpus/gt steps are identical.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import settings as S
from .settings import DATA_DIR
from .resolve import (
    Resolver,
    apply_id_remap,
    find_embedding_candidates,
    load_relation_map,
    rel_type_token,
    safe_merge,
    title_merge_candidates,
)
from .schema_induction import induce

_SYSTEM = """You extract a knowledge graph from a single encyclopedic passage.

Return STRICT JSON with two keys:
  "entities": a list of {{"name": <surface string as it appears>, "type": <one of: {entity_types}>}}
  "triples":  a list of {{"subject": <entity name>, "relation": <short lowercase relation phrase>,
                          "object": <entity name>}}

Rules:
- Extract only facts STATED IN THE PASSAGE. Do not use outside knowledge.
- subject and object MUST be entity names that also appear in "entities".
- Prefer these relation phrases when they fit (but you may introduce new ones if needed):
    {relation_vocab}
- Keep relation phrases short and canonical (e.g. "director", "place of birth", "spouse",
  "country of citizenship", "publication date", "located in").
- Use the most specific named entities (people, places, organizations, works, dates, numbers).
- Output ONLY the JSON object, no commentary, no markdown fences."""

_USER = """Title: {title}

Passage:
{text}

Extract the entities and triples as JSON."""


def _parse_json(content: str) -> dict:
    """Best-effort strict-JSON parse: strip fences, then take the first {...} block."""
    s = content.strip()
    s = re.sub(r"^```[a-z]*\n?", "", s, flags=re.IGNORECASE).strip().rstrip("`").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start, depth = s.find("{"), 0
    if start == -1:
        return {"entities": [], "triples": []}
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except Exception:
                    break
    return {"entities": [], "triples": []}


def extract_passage(passage: dict, entity_types: str, relation_vocab: str) -> dict:
    """Extract {entities, triples} from one passage via the LLM."""
    system = _SYSTEM.format(entity_types=entity_types, relation_vocab=relation_vocab)
    user = _USER.format(title=passage["title"], text=passage["text"])
    out = S.llm("openie").chat(system, user)
    parsed = _parse_json(out["content"])
    return {
        "passage_id": passage["id"],
        "entities":   parsed.get("entities") or [],
        "triples":    parsed.get("triples") or [],
    }


# ── Batched extraction (several passages per LLM call) ───────────────────────────

_SYSTEM_BATCH = """You extract a knowledge graph from SEVERAL short encyclopedic passages.

Each passage is tagged with an id like [P1], [P2], .... Return STRICT JSON:
  {{"results": [
     {{"id": "P1",
       "entities": [{{"name": <surface string>, "type": <one of: {entity_types}>}}],
       "triples":  [{{"subject": <entity name>, "relation": <short lowercase relation phrase>,
                      "object": <entity name>}}]}},
     ...
  ]}}
with EXACTLY ONE object per input passage, reusing the SAME ids.

Rules:
- Extract only facts STATED IN THAT passage. Do not use outside knowledge or mix passages.
- subject and object MUST be entity names that also appear in that passage's "entities".
- Prefer these relation phrases when they fit (you may introduce new ones if needed):
    {relation_vocab}
- Keep relation phrases short and canonical (e.g. "director", "place of birth", "spouse").
- Output ONLY the JSON object, no commentary, no markdown fences."""

_USER_BATCH_HEAD = "Extract entities and triples for EACH passage below.\n\n"


def extract_batch(passages: list[dict], entity_types: str, relation_vocab: str) -> list[dict]:
    """Extract for several passages in one LLM call; falls back to per-passage on failure.

    Returns one {passage_id, entities, triples} per input passage (order-aligned).
    """
    if len(passages) == 1:
        return [extract_passage(passages[0], entity_types, relation_vocab)]

    system = _SYSTEM_BATCH.format(entity_types=entity_types, relation_vocab=relation_vocab)
    blocks = [f"[P{i + 1}] Title: {p['title']}\n{p['text']}" for i, p in enumerate(passages)]
    user = _USER_BATCH_HEAD + "\n\n".join(blocks)

    out = S.llm("openie").chat(system, user)
    parsed = _parse_json(out["content"])
    results = parsed.get("results")

    if not isinstance(results, list) or not results:
        # Couldn't parse the batch — degrade to reliable per-passage calls.
        return [extract_passage(p, entity_types, relation_vocab) for p in passages]

    by_id = {str(r.get("id", "")).strip().upper(): r for r in results if isinstance(r, dict)}
    out_list: list[dict] = []
    missing: list[int] = []
    for i, p in enumerate(passages):
        r = by_id.get(f"P{i + 1}")
        if r is None:
            missing.append(i)
            out_list.append({"passage_id": p["id"], "entities": [], "triples": []})
        else:
            out_list.append({
                "passage_id": p["id"],
                "entities":   r.get("entities") or [],
                "triples":    r.get("triples") or [],
            })
    # Repair any passages the batch dropped with individual calls (keeps recall).
    for i in missing:
        out_list[i] = extract_passage(passages[i], entity_types, relation_vocab)
    return out_list


# ---------------------------------------------------------------------------
# Cache (resumable): one JSON line per extracted passage, keyed by passage_id.
# ---------------------------------------------------------------------------

def _load_cache(path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            cache[obj["passage_id"]] = obj
        except Exception:
            continue
    return cache


def _unique_passages(records: list[dict]) -> dict[str, dict]:
    passages: dict[str, dict] = {}
    for rec in records:
        for p in rec["passages"]:
            passages.setdefault(p["passage_id"], {"id": p["passage_id"],
                                                  "title": p["title"], "text": p["text"]})
    return passages


def build_openie(records: list[dict], benchmark: str, out_key: str | None = None) -> dict:
    """Induce ontology -> extract per passage (cached, parallel) -> resolve -> KG dict."""
    out_key = out_key or benchmark
    ontology = induce(records, benchmark)
    entity_types_str = ", ".join(ontology["entity_types"])
    relation_vocab = ", ".join(r["name"] for r in ontology["relations"]) or "(none — infer freely)"

    passages = _unique_passages(records)
    cache_path = DATA_DIR / out_key / "openie_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)

    todo = [p for pid, p in passages.items() if pid not in cache]
    batch_size = max(1, S.cfg()["openie"]["batch_size"])
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"  OpenIE: {len(passages)} unique passages, {len(cache)} cached, {len(todo)} to "
          f"extract in {len(batches)} batch(es) of up to {batch_size}")

    if todo:
        with cache_path.open("a", encoding="utf-8") as fh:
            with ThreadPoolExecutor(max_workers=S.cfg()["openie"]["workers"]) as pool:
                futures = {pool.submit(extract_batch, b, entity_types_str, relation_vocab): b
                           for b in batches}
                done = 0
                for fut in as_completed(futures):
                    for res in fut.result():
                        cache[res["passage_id"]] = res
                        fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                        done += 1
                    fh.flush()
                    if done % 50 == 0 or done >= len(todo):
                        print(f"    extracted {min(done, len(todo))}/{len(todo)}")

    # ── Assemble KG from cached extractions ──────────────────────────────────
    # Pass 1: pure exact-match resolution (entities/triples + a bare title node per passage).
    # No aliasing/merging happens yet — that is decided afterward by safe_merge() once all
    # facts (including, for any title string that independently appears as a real subject/
    # object, its own facts) are visible to the contradiction check.
    resolver = Resolver()
    edges: set[tuple] = set()
    mentions: set[tuple] = set()
    relations: dict[str, str] = {}

    for pid, ext in cache.items():
        for ent in ext["entities"]:
            name = (ent.get("name") or "").strip()
            if not name:
                continue
            eid = resolver.add(name, type=ent.get("type"))
            mentions.add((eid, pid))
        for tr in ext["triples"]:
            s = (tr.get("subject") or "").strip()
            o = (tr.get("object") or "").strip()
            r = (tr.get("relation") or "").strip()
            if not (s and o and r):
                continue
            sid = resolver.add(s)
            oid = resolver.add(o)
            rel_tok = rel_type_token(r)
            relations[rel_tok] = r
            edges.add((sid, oid, rel_tok, r, pid))
            mentions.add((sid, pid))
            mentions.add((oid, pid))

    # Pass 2: propose merge candidates from two independent sources (title-vs-lede-entity
    # within a passage, and cross-passage embedding near-duplicates), then commit only the
    # ones that pass safe_merge()'s fact-contradiction check in ONE unified, audited gate.
    title_pairs = title_merge_candidates(passages, cache, resolver)
    entities = resolver.entities()  # re-fetch: title_merge_candidates may have added title nodes

    embed_pairs: list[tuple[str, str]] = []
    if S.cfg()["resolve"]["use_embeddings"]:
        threshold = S.cfg()["resolve"]["cosine_threshold"]
        print(f"  finding near-duplicate entity candidates via embeddings (threshold={threshold}) ...")
        embed_pairs = find_embedding_candidates(entities, edges, threshold)

    candidates = title_pairs + embed_pairs
    if candidates:
        # Reuse a PRIOR consolidate.py run's relation_map.json (if this benchmark has been
        # consolidated before) so the veto checks compare relation phrases by their REAL,
        # LLM-derived canonical form ("spouse" == "spouse of") instead of raw exact strings.
        # Absent on a from-scratch build; the checks degrade gracefully to raw-phrase matching.
        relation_canon = load_relation_map(DATA_DIR / out_key / "relation_map.json")
        if relation_canon:
            print(f"  reusing relation_map.json ({len(relation_canon)} phrases) for veto checks")
        print(f"  resolving {len(title_pairs)} title-merge + {len(embed_pairs)} embedding "
              f"candidate(s) ...")
        id_remap = safe_merge(entities, edges, candidates, relation_canon=relation_canon)
        print(f"    committed {len(id_remap)} merge(s)")
        entities, edges, mentions = apply_id_remap(entities, edges, mentions, id_remap)

    # Drop self-referential edges (subject == object) — never meaningful for any relation in
    # this graph (an entity is never its own director/spouse/parent/etc.). Catches BOTH raw
    # OpenIE extraction noise (the LLM occasionally outputs subject==object directly) and any
    # residual merge-driven self-loop as a final, unconditional safety net.
    before = len(edges)
    edges = {e for e in edges if e[0] != e[1]}
    dropped = before - len(edges)
    if dropped:
        print(f"  dropped {dropped} self-referential edge(s) (subject == object)")

    return {
        "passages": [{"id": p["id"], "title": p["title"], "text": p["text"]}
                     for p in passages.values()],
        "entities": entities,
        "edges":    [{"subject_id": s, "object_id": o, "rel_type": rt, "name": rn,
                      "passage_id": p}
                     for (s, o, rt, rn, p) in edges],
        "mentions": [{"entity_id": e, "passage_id": p} for (e, p) in mentions],
        "relations": _relations_with_counts(edges, relations),
        "entity_types": ontology["entity_types"],
    }


def _relations_with_counts(edges, name_by_tok: dict) -> list[dict]:
    """[{type, name, count}] sorted by edge frequency desc (then alpha)."""
    from collections import Counter
    counts = Counter(rt for (_s, _o, rt, _rn, _p) in edges)
    return [{"type": rt, "name": name_by_tok[rt], "count": counts[rt]}
            for rt in sorted(name_by_tok, key=lambda t: (-counts[t], t))]
