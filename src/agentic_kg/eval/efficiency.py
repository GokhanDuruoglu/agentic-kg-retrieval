"""Efficiency from the run's OWN records: latency, LLM calls, tokens and USD.

Everything comes from the record that holds the answer being scored, so cost, latency and effectiveness cannot be
cross-wired. USD = tokens x list price from configs/models.yaml (judge, embedding and index-time calls are not
included):

  usd_list     prompt*input + completion*output                      (cache-agnostic, the headline)
  usd_cache    cached prompt tokens at the cached-input price (if the API reported them and the price is set)

A model without a price yields None (never a silent 0).
"""


def _short(model: str | None) -> str:
    return (model or "").split("/")[-1]


def _usd(call: dict, pricing: dict) -> tuple[float | None, float | None]:
    p = pricing.get(_short(call.get("model"))) or {}
    pin, pout, pc = p.get("input"), p.get("output"), p.get("cached_input")
    pt, ct = call.get("prompt_tokens") or 0, call.get("completion_tokens") or 0
    if pin is None or pout is None:
        return None, None
    listed = (pt * pin + ct * pout) / 1e6
    cached = call.get("cached_tokens")
    if pc is not None and cached:
        return listed, ((pt - cached) * pin + cached * pc + ct * pout) / 1e6
    return listed, listed


def question_efficiency(rec: dict, pricing: dict) -> dict:
    calls = rec.get("llm_calls") or []
    if rec.get("kind") == "agentic":
        steps = rec.get("steps", [])
        out = {"latency_sec": rec.get("total_sec"), "n_subquestions": rec.get("sub_question_count"),
               "n_tool_routed": rec.get("tool_routed_count"), "combiner_used": float(bool(rec.get("combiner_used"))),
               "n_query_attempts": sum(s.get("attempt_count", 1) for s in steps),
               "neo4j_sec": sum(s.get("neo4j_sec") or 0 for s in steps)}
    else:
        out = {"latency_sec": rec.get("elapsed_sec"), "retrieval_sec": rec.get("retrieval_sec")}
    usd_l, usd_c, missing = 0.0, 0.0, False
    for c in calls:
        l, ca = _usd(c, pricing)
        if l is None:
            missing = True
        else:
            usd_l += l
            usd_c += ca
    out.update({
        "n_llm_calls": len(calls),
        "tokens_prompt": sum(c.get("prompt_tokens") or 0 for c in calls),
        "tokens_completion": sum(c.get("completion_tokens") or 0 for c in calls),
        "usd_list": None if missing else usd_l,
        "usd_cache": None if missing else usd_c,
    })
    out["tokens_total"] = out["tokens_prompt"] + out["tokens_completion"]
    return out
