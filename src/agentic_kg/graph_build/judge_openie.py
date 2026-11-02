"""Judge the OpenIE cache with a strong model and emit a corrected v2 cache.

The OpenIE cache (openie_cache.jsonl) is produced by a cheap extractor, so it carries noise:
triples not actually stated in the passage, and wrong relation/subject/object. This pass gives a
STRONG model each passage's text plus its extracted triples and asks it to audit them — keeping,
DROPping (not grounded), or FIXing (wrong relation/subject/object) each one. It does NOT invent
new triples (prune-&-fix only), so KG recall is unchanged and precision goes up.

The model returns a small per-passage DIFF ({"drop": [i...], "fix": [{"i", ...}]}) rather than a
re-serialized triple list, so output tokens stay tiny and every change is auditable. Applying the
diffs to the original cache yields a v2 cache in the EXACT same schema, which the unchanged build
pipeline consumes (``akg graph build --step rebuild``). The audit log openie_judge.jsonl is resumable: with a
complete log (shipped in data/2wiki/build_cache/) no LLM call is made.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import settings as S  # noqa: E402
from .openie import _load_cache, _parse_json, _unique_passages  # noqa: E402
from .settings import DATA_DIR  # noqa: E402


_SYSTEM = """You audit a knowledge-graph extraction against its SOURCE passage.

You are given one encyclopedic passage and a numbered list of (subject, relation, object) triples
that a weaker model extracted from it. For EACH triple, decide:
  - KEEP it (the fact is clearly stated in the passage) — do nothing,
  - DROP it (the fact is NOT stated in the passage, or is wrong / hallucinated),
  - FIX it (the fact is real but the relation, subject, or object is wrong or imprecise).

Return STRICT JSON with two keys:
  "drop": a list of triple indices (integers) to remove,
  "fix":  a list of {{"i": <triple index>, "subject": <new>, "relation": <new>, "object": <new>}}
          where you include ONLY the field(s) that change (omit the ones that stay the same).

Rules:
- Judge ONLY against facts STATED IN THE PASSAGE. Do not use outside knowledge.
- Do NOT add brand-new triples. Every index you reference must be one of the given triples.
- Only FIX a relation when it states the WRONG relationship (e.g. "daughter of" where the passage
  says the person is the father, or "director" for someone who is actually an actor). Do NOT change
  a relation merely to reword, restyle, or canonicalize it ("date of birth" -> "date_of_birth",
  "director of" -> "director", "member of" -> "member_of") — a later step normalizes vocabulary.
- If a triple is already correct, do not list it in either "drop" or "fix".
- Output ONLY the JSON object, no commentary, no markdown fences."""

_SYSTEM_BATCH = """You audit knowledge-graph extractions against their SOURCE passages.

You are given SEVERAL passages, each tagged with an id like [P1], [P2], .... Each has a numbered
list of (subject, relation, object) triples a weaker model extracted from THAT passage. For EACH
triple decide KEEP (stated in the passage), DROP (not stated / wrong), or FIX (real fact but wrong
relation/subject/object).

Return STRICT JSON:
  {{"results": [
     {{"id": "P1",
       "drop": [<triple indices to remove>],
       "fix":  [{{"i": <triple index>, "subject": <new>, "relation": <new>, "object": <new>}}]}},
     ...
  ]}}
with EXACTLY ONE object per input passage, reusing the SAME ids. In each "fix" include ONLY the
field(s) that change.

Rules:
- Judge each triple ONLY against facts STATED IN ITS OWN passage. Do not use outside knowledge or
  mix passages.
- Do NOT add brand-new triples. Every index must be one of that passage's given triples.
- Only FIX a relation when it states the WRONG relationship (e.g. "daughter of" where the passage
  says the person is the father). Do NOT change a relation merely to reword, restyle, or
  canonicalize it ("date of birth" -> "date_of_birth", "member of" -> "member_of") — a later step
  normalizes vocabulary.
- If a triple is already correct, list it in neither "drop" nor "fix".
- Output ONLY the JSON object, no commentary, no markdown fences."""

_USER_BATCH_HEAD = "Audit the triples for EACH passage below.\n\n"


def _render_block(tag: str, item: dict) -> str:
    """One passage rendered for the prompt: title, text, and numbered triples."""
    lines = [f"{tag} Title: {item['title']}", f"Passage: {item['text']}", "Triples:"]
    for i, tr in enumerate(item["triples"]):
        s = (tr.get("subject") or "").strip()
        r = (tr.get("relation") or "").strip()
        o = (tr.get("object") or "").strip()
        lines.append(f"  {i}) {s} | {r} | {o}")
    return "\n".join(lines)


def _clean_diff(raw: dict, n_triples: int) -> dict:
    """Sanitize a model diff: keep only in-range indices and non-empty fix fields."""
    drop = sorted({int(i) for i in (raw.get("drop") or [])
                   if isinstance(i, (int, float)) and 0 <= int(i) < n_triples})
    fix = []
    for f in raw.get("fix") or []:
        if not isinstance(f, dict) or "i" not in f:
            continue
        try:
            i = int(f["i"])
        except (TypeError, ValueError):
            continue
        if not (0 <= i < n_triples) or i in drop:
            continue
        entry = {"i": i}
        for k in ("subject", "relation", "object"):
            v = (f.get(k) or "").strip() if isinstance(f.get(k), str) else ""
            if v:
                entry[k] = v
        if len(entry) > 1:  # at least one real change besides "i"
            fix.append(entry)
    return {"drop": drop, "fix": fix}


def judge_passage(item: dict) -> dict:
    """Audit one passage's triples; returns {passage_id, drop, fix}."""
    n = len(item["triples"])
    if n == 0:
        return {"passage_id": item["id"], "drop": [], "fix": []}
    user = _render_block("Passage", item) + "\n\nAudit the triples as JSON."
    out = _judge().chat(_SYSTEM, user)
    diff = _clean_diff(_parse_json(out["content"]), n)
    return {"passage_id": item["id"], **diff}


def judge_batch(items: list[dict]) -> list[dict]:
    """Audit several passages in one call; falls back to per-passage on failure.

    Returns one {passage_id, drop, fix} per input passage (order-aligned).
    """
    if len(items) == 1:
        return [judge_passage(items[0])]

    blocks = [_render_block(f"[P{i + 1}]", it) for i, it in enumerate(items)]
    user = _USER_BATCH_HEAD + "\n\n".join(blocks)

    out = _judge().chat(_SYSTEM_BATCH, user)
    results = _parse_json(out["content"]).get("results")
    if not isinstance(results, list) or not results:
        # Couldn't parse the batch — degrade to reliable per-passage calls.
        return [judge_passage(it) for it in items]

    by_id = {str(r.get("id", "")).strip().upper(): r for r in results if isinstance(r, dict)}
    out_list: list[dict] = []
    missing: list[int] = []
    for i, it in enumerate(items):
        r = by_id.get(f"P{i + 1}")
        if r is None:
            missing.append(i)
            out_list.append({"passage_id": it["id"], "drop": [], "fix": []})
        else:
            out_list.append({"passage_id": it["id"], **_clean_diff(r, len(it["triples"]))})
    # Repair any passages the batch dropped with individual calls.
    for i in missing:
        out_list[i] = judge_passage(items[i])
    return out_list


# ── Apply a diff to a cache entry → corrected {entities, triples} ─────────────────

def apply_diff(entry: dict, diff: dict) -> tuple[dict, int, int]:
    """Return (new_entry, n_dropped, n_fixed) after applying drop/fix to one cache entry."""
    triples = entry.get("triples") or []
    drop_set = set(diff.get("drop") or [])
    fix_map = {f["i"]: f for f in (diff.get("fix") or [])}

    new_triples: list[dict] = []
    n_fixed = 0
    for i, tr in enumerate(triples):
        if i in drop_set:
            continue
        f = fix_map.get(i)
        if f:
            tr = {"subject":  f.get("subject",  tr.get("subject")),
                  "relation": f.get("relation", tr.get("relation")),
                  "object":   f.get("object",   tr.get("object"))}
            n_fixed += 1
        new_triples.append(tr)

    # Keep original entities; ensure every surviving subject/object still appears as an entity
    # (a fix may introduce a corrected name) so the downstream invariant holds.
    entities = [dict(e) for e in (entry.get("entities") or [])]
    have = {(e.get("name") or "").strip().lower() for e in entities}
    for tr in new_triples:
        for name in ((tr.get("subject") or "").strip(), (tr.get("object") or "").strip()):
            if name and name.lower() not in have:
                entities.append({"name": name, "type": "Misc"})
                have.add(name.lower())

    n_dropped = len(drop_set & set(range(len(triples))))
    return {"passage_id": entry["passage_id"], "entities": entities,
            "triples": new_triples}, n_dropped, n_fixed


def _load_judged(path: Path) -> dict[str, dict]:
    """Resumable audit log: {passage_id: {passage_id, drop, fix}}."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            out[obj["passage_id"]] = obj
        except Exception:
            continue
    return out


_model_override: str | None = None


def _judge():
    """The audit LLM (stage "triple_judge" of configs/graph_build.yaml; --model overrides the model)."""
    return S.llm("triple_judge", _model_override)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Judge the OpenIE cache and write a v2 cache.")
    ap.add_argument("--in-benchmark", default="2wiki",
                    help="Source build key (<AKG_BUILD_DIR>/<in>/).")
    ap.add_argument("--out-benchmark", default=None,
                    help="Output benchmark key for the v2 cache. Default: <in>-judged.")
    ap.add_argument("--model", default=None, help="Fireworks model name (default: configs/graph_build.yaml).")
    ap.add_argument("--batch-size", type=int, default=S.cfg()["triple_judge"]["batch_size"])
    ap.add_argument("--workers", type=int, default=S.cfg()["triple_judge"]["workers"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Judge only the first N (unjudged) passages — for a cheap sample check.")
    args = ap.parse_args(argv)

    in_dir = DATA_DIR / args.in_benchmark
    out_key = args.out_benchmark or f"{args.in_benchmark}-judged"
    out_dir = DATA_DIR / out_key

    cache = _load_cache(in_dir / "openie_cache.jsonl")
    if not cache:
        raise SystemExit(f"no openie_cache.jsonl found in {in_dir}")
    passages = _unique_passages(
        [json.loads(l) for l in (in_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
         if l.strip()])

    global _model_override
    _model_override = args.model

    judge_path = in_dir / "openie_judge.jsonl"
    judged = _load_judged(judge_path)

    # Build the work list: passages that have triples, a known passage text, and are not yet judged.
    todo_items = []
    for pid, ext in cache.items():
        if pid in judged or pid not in passages:
            continue
        if not (ext.get("triples") or []):
            continue
        p = passages[pid]
        todo_items.append({"id": pid, "title": p["title"], "text": p["text"],
                           "triples": ext["triples"]})
    if args.limit is not None:
        todo_items = todo_items[:args.limit]

    batches = [todo_items[i:i + args.batch_size]
               for i in range(0, len(todo_items), args.batch_size)]
    print(f"Judging {args.in_benchmark} with {_judge().model}")
    print(f"  {len(cache)} cached passages, {len(judged)} already judged, "
          f"{len(todo_items)} to judge in {len(batches)} batch(es) of up to {args.batch_size}")

    if batches:
        with judge_path.open("a", encoding="utf-8") as fh:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(judge_batch, b): b for b in batches}
                done = 0
                for fut in as_completed(futs):
                    for res in fut.result():
                        judged[res["passage_id"]] = res
                        fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                        done += 1
                    fh.flush()
                    if done % 50 == 0 or done >= len(todo_items):
                        print(f"    judged {min(done, len(todo_items))}/{len(todo_items)}")

    # ── Build v2 cache: apply every available diff to the original cache (unjudged pass through). ──
    out_dir.mkdir(parents=True, exist_ok=True)
    total_drop = total_fix = changed = 0
    with (out_dir / "openie_cache.jsonl").open("w", encoding="utf-8") as fh:
        for pid, entry in cache.items():
            diff = judged.get(pid)
            if diff:
                new_entry, nd, nf = apply_diff(entry, diff)
                total_drop += nd
                total_fix += nf
                if nd or nf:
                    changed += 1
            else:
                new_entry = {"passage_id": pid, "entities": entry.get("entities") or [],
                             "triples": entry.get("triples") or []}
            fh.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

    # Copy sample.json so build_kg's loader reproduces identical passages/passage_ids for the v2 key.
    sample_src = in_dir / "sample.json"
    if sample_src.exists():
        (out_dir / "sample.json").write_text(sample_src.read_text(encoding="utf-8"),
                                             encoding="utf-8")

    print(f"  audit log : {judge_path}")
    print(f"  v2 cache  : {out_dir / 'openie_cache.jsonl'}")
    print(f"  changes   : {changed} passage(s) changed, {total_drop} triple(s) dropped, "
          f"{total_fix} triple(s) fixed")
    print(f"Done → rebuild with:  akg graph build --step rebuild")


if __name__ == "__main__":
    main()
