"""LLM-judge every record of a run with ONE judge model (configs/models.yaml, role ``judge``).

    akg judge --methods validated-tool bm25 --repeats 1-3

Output: judgments/<method>/r<k>.jsonl, one record per question
  {id, answer_sha, judge_recall, judge_precision, judge_f1, judge_correct, rationale, judge: {model, system_sha,
   user_sha, prompt_tokens, completion_tokens, elapsed_sec, attempts}}   or {id, judge_error, ...}.
Resumable and safe: a verdict is reused only if it was produced for exactly this prompt, so a regenerated answer
never keeps an old verdict.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config import ROOT, load_dataset, load_settings
from ..io import read_jsonl
from ..llm import LLM, build_llms
from . import judge_prompts as JP

JUDGMENTS = ROOT / "judgments"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_RETRY = ("\n\nYour previous reply was not valid JSON. Reply again with ONLY the JSON object "
          "described above — no markdown fences, no commentary.")


def _parse(content: str):
    text = _FENCE.sub("", (content or "").strip()).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    for cand in (text, m[0] if m else None):
        if not cand:
            continue
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    return None


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def judge_one(llm: LLM, qid: str, system: str, user: str, answer_sha: str) -> dict:
    usage = {"model": llm.model, "system_sha": sha12(system), "user_sha": sha12(user),
             "prompt_tokens": 0, "completion_tokens": 0, "elapsed_sec": 0.0, "attempts": 0}
    parsed = None
    for suffix in ("", _RETRY):
        resp = llm.chat(system, user + suffix)
        usage["attempts"] += 1
        usage["prompt_tokens"] += resp.get("prompt_tokens") or 0
        usage["completion_tokens"] += resp.get("completion_tokens") or 0
        usage["elapsed_sec"] = round(usage["elapsed_sec"] + (resp.get("elapsed_sec") or 0), 3)
        parsed = _parse(resp["content"])
        if parsed is not None:
            break
    if parsed is None:
        return {"id": qid, "answer_sha": answer_sha, "judge_error": "invalid_json", "judge": usage}
    try:
        r = max(0.0, min(1.0, float(parsed.get("recall", 0.0))))
        p = max(0.0, min(1.0, float(parsed.get("precision", 0.0))))
    except (TypeError, ValueError):
        return {"id": qid, "answer_sha": answer_sha, "judge_error": "bad_score_types", "judge": usage}
    return {"id": qid, "answer_sha": answer_sha, "judge_recall": r, "judge_precision": p,
            "judge_f1": (2 * p * r / (p + r)) if (p + r) else 0.0,
            "judge_correct": 1.0 if parsed.get("correct") else 0.0,
            "rationale": str(parsed.get("rationale", "")), "judge": usage}


def golds_of(gold_row: dict) -> list[str]:
    return [gold_row.get("answer_value")] + list(gold_row.get("answer_aliases") or [])


def judge_run(llm: LLM, gold: dict[str, dict], recs: list[dict], out: Path, workers: int = 8) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"]: r for r in read_jsonl(out)} if out.exists() else {}
    built = {}
    for rec in recs:
        qid = str(rec["id"])
        if qid in gold:
            ans = (rec.get("answer") or "").strip()
            built[qid] = (JP.SYSTEM_PROMPT, JP.build_user_prompt(rec["question"], golds_of(gold[qid]), ans),
                          sha12(ans))
    todo = [q for q, (_s, u, _a) in built.items()
            if "judge_error" in done.get(q, {"judge_error": 1}) or done[q].get("judge", {}).get("user_sha") != sha12(u)]
    print(f"  {out.parent.name}/{out.stem}: {len(recs)} answers, {len(built) - len(todo)} up-to-date verdicts, "
          f"{len(todo)} to do")
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as ex, open(out, "a", encoding="utf-8") as fh:
        futs = {ex.submit(judge_one, llm, q, *built[q][:2], built[q][2]): q for q in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as exc:                       # noqa: BLE001
                r = {"id": futs[f], "judge_error": repr(exc)}
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
    final = {r["id"]: r for r in read_jsonl(out)}          # keep the last record per id
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final.values()), encoding="utf-8")
    n_err = sum("judge_error" in r for r in final.values())
    if n_err:
        print(f"    ! {n_err} judge errors remain (re-run to retry)")


def judge_methods(methods: list[str], repeats: list[int], *, runs_root: Path | None = None,
                  workers: int = 8, dataset: str = "2wiki") -> None:
    from ..runner import RUNS
    runs_root = runs_root or RUNS
    settings = load_settings()
    ds = load_dataset(dataset)
    gold = {str(r["id"]): r for r in read_jsonl(ds["gold"])}
    llm = build_llms(settings, 1)["judge"]
    print(f"judge model: {llm.model}")
    for m in methods:
        for rep in repeats:
            rec_path = runs_root / m / f"r{rep}" / "records.jsonl"
            if rec_path.exists():
                judge_run(llm, gold, read_jsonl(rec_path), JUDGMENTS / m / f"r{rep}.jsonl", workers)
