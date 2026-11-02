"""Run records: one JSON object per question, with the full LLM-call accounting.

Cost and latency are computed offline from these records (``eval/efficiency.py``), never from provider
dashboards. Agentic and retrieval methods write the same envelope (``id, question, answer, method, run,
llm_calls, elapsed_sec``) plus their own fields.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

_USAGE_KEYS = ("model", "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "elapsed_sec")
_STEP_EXTRAS = ("label", "chain_pos", "chain_len", "dependency_failed")


def _call(stage: str, llm: dict | None, **extra) -> dict | None:
    if not llm:
        return None
    rec = {"stage": stage, **{k: llm.get(k) for k in _USAGE_KEYS}, **extra}
    if llm.get("raw_token_usage"):
        rec["raw_token_usage"] = llm["raw_token_usage"]
    return rec


def llm_calls(result: dict) -> list[dict]:
    """Every LLM call made while answering one question, in pipeline order."""
    calls = [_call("decision", result["_decision"].llm)]
    for step in result["steps"]:
        for att in step.get("_attempts") or []:
            calls.append(_call("query", (att.get("qg") or {}).get("llm"), step=step["index"], attempt=att["attempt"]))
    partials = result.get("_answer_partials")
    if partials:
        for i, p in enumerate(partials):
            calls.append(_call("answer_partial", (p or {}).get("llm"), chunk=i))
        calls.append(_call("combiner", (result.get("_combiner") or {}).get("llm")))
    else:
        calls.append(_call("answer", (result.get("_answer") or {}).get("llm")))
    return [c for c in calls if c]


def agentic_record(result: dict, source: dict, elapsed: float, run: dict, method: str) -> dict:
    dec = result["_decision"]
    ans = result["_answer"]["llm"] or {}
    record = {
        "id":                 source.get("id"),
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "method":             method,
        "kind":               "agentic",
        "database":           result["database"],
        "components":         result["components"],
        "question":           result["question"],
        "source":             source,
        "answer":             result["answer"],
        "sub_questions":      result["sub_questions"],
        **{k: result[k] for k in ("sub_question_labels", "labeled_sub_questions", "decision_groups") if k in result},
        "sub_question_count": result["sub_question_count"],
        "tool_routed_count":  result["tool_routed_count"],
        "answer_chunk_count": result["answer_chunk_count"],
        "combiner_used":      result["combiner_used"],
        "context_tokens":     result["context_tokens"],
        "context_truncated":  result["context_truncated"],
        "total_sec":          result["total_sec"],
        "total_tokens":       result["total_tokens"],
        "steps": [
            {
                "index":         s["index"],
                "sub_question":  s["sub_question"],
                "tool":          s.get("tool"),
                "tool_args":     s.get("tool_args"),
                "cypher":        s["cypher"],
                "params":        s["params"],
                "rows":          s["rows"],
                "row_count":     s["row_count"],
                "attempt_count": s.get("attempt_count", 1),
                "validator_ok":  s.get("validator_ok"),
                "note":          s.get("note"),
                "error":         s["error"],
                "query_gen_sec": s["query_gen_sec"],
                "neo4j_sec":     s["neo4j_sec"],
                "total_sec":     s["total_sec"],
                **{k: s[k] for k in _STEP_EXTRAS if k in s},
            }
            for s in result["steps"]
        ],
        "decision_agent": {
            "decomposed":        dec.decomposed,
            "raw_llm_output":    dec.raw,
            "prompt_tokens":     (dec.llm or {}).get("prompt_tokens"),
            "completion_tokens": (dec.llm or {}).get("completion_tokens"),
            "elapsed_sec":       (dec.llm or {}).get("elapsed_sec"),
        },
        "answer_agent": {
            "prompt_tokens":     ans.get("prompt_tokens"),
            "completion_tokens": ans.get("completion_tokens"),
            "elapsed_sec":       ans.get("elapsed_sec"),
        },
        "run":       run,
        "llm_calls": llm_calls(result),
        "elapsed_sec": elapsed,
    }
    return record


def append_record(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def record_error(run_dir: Path, source: dict | None, question: str, exc: BaseException, run: dict) -> None:
    """Append a failed question to <run_dir>/errors.jsonl; the batch carries on."""
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "id": (source or {}).get("id"),
           "question": question, "error": f"{type(exc).__name__}: {exc}",
           "traceback": traceback.format_exc(), "run": run}
    append_record(Path(run_dir) / "errors.jsonl", rec)
    print(f"! question {rec['id']} failed: {rec['error']} (logged to errors.jsonl)")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def done_ids(path: Path) -> set[str]:
    return {str(r.get("id")) for r in read_jsonl(path)}
