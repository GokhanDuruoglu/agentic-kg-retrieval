"""Retrieve-then-read pipeline: retriever -> context builder -> ONE answer call.

    retriever   registry kind "retriever"  (bm25, dense, graphrag):  search(query, top_k) -> hits
    context     registry kind "context"    (passages, graphrag_local): build(retriever, query, hits) -> (text, n_used, meta)

A hit is ``{rank, score, doc_id, node_type, title, document}``. Every method answers with the same prompt.
"""

from __future__ import annotations

import time

from ..agentic.runtime import Runtime
from ..registry import get, register
from ..tokens import count_tokens
from . import prompts as P

DEFAULT_CONTEXT_TOKENS = 128_000


def format_context(hits: list[dict], max_tokens: int | None = DEFAULT_CONTEXT_TOKENS) -> tuple[str, int]:
    """Render hits as ``[rank] (Passage) document`` blocks in rank order, within the token budget.

    Whole documents are kept until the budget would be exceeded; lower-ranked ones that do not fit are dropped.
    ``max_tokens`` of None/<=0 keeps everything.
    """
    blocks, used, total = [], 0, 0
    for h in hits:
        block = f"[{h['rank']}] ({h['node_type']}) {h['document']}"
        n = count_tokens(block)
        if max_tokens and max_tokens > 0 and blocks and total + n > max_tokens:
            break
        blocks.append(block)
        total += n
        used += 1
    return "\n\n".join(blocks), used


@register("context", "passages")
class PassagesContext:
    """The ranked passages only."""

    def __init__(self, max_tokens: int = DEFAULT_CONTEXT_TOKENS):
        self.max_tokens = max_tokens

    def build(self, retriever, query: str, hits: list[dict]):
        text, used = format_context(hits, self.max_tokens)
        return text, used, None


@register("context", "graphrag_local")
class GraphRAGLocalContext:
    """Microsoft local-search context: community reports + entities + relationships + sources."""

    def build(self, retriever, query: str, hits: list[dict]):
        if not hasattr(retriever, "build_context"):
            raise TypeError("context 'graphrag_local' needs the 'graphrag' retriever")
        return retriever.build_context(query, hits)


class RetrievalPipeline:
    def __init__(self, cfg: dict, rt: Runtime, corpus_path):
        self.cfg = cfg
        self.rt = rt
        r = dict(cfg["retriever"])
        self.retriever_type = r.pop("type")
        self.retriever = get("retriever", self.retriever_type)(corpus_path, **r)
        c = dict(cfg.get("context") or {"type": "passages"})
        self.context_type = c.pop("type")
        self.context = get("context", self.context_type)(**c)
        self.top_k = int(cfg.get("top_k", 20))
        if self.context_type == "graphrag_local" and not any(
                m.get("report") for m in getattr(self.retriever, "community_meta", {}).values()):
            raise SystemExit("context 'graphrag_local' needs community reports for the current answer model; "
                             "run `akg index graphrag` first")

    def run(self, question: str) -> dict:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, top_k=self.top_k)
        retrieval_sec = round(time.perf_counter() - t0, 3)
        context, n_used, meta = self.context.build(self.retriever, question, hits)
        user = P.answer_user_prompt.format(question=question, context=context)
        llm = self.rt.llms["answer"].chat(P.answer_system_prompt, user)
        return {"answer": llm["content"], "hits": hits, "llm": llm, "retrieval_sec": retrieval_sec,
                "n_hits_in_context": n_used, "context_meta": meta,
                # the exact text the answer model saw, stored only for the local-search context
                "context_text": context if meta is not None else None,
                "elapsed_sec": round(time.perf_counter() - t0, 3)}


def retrieval_record(result: dict, source: dict, run: dict, method: str, pipe: RetrievalPipeline) -> dict:
    from datetime import datetime, timezone
    llm = result["llm"]
    return {
        "id": source.get("id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "kind": "retrieval",
        "topic": source.get("topic"),
        "level": source.get("level"),
        "operation_type": source.get("operation_type"),
        "question": source.get("question"),
        "answer": result["answer"],
        "retrieved": result["hits"],
        "retriever": {"type": pipe.retriever_type, "context": pipe.context_type},
        "top_k": pipe.top_k,
        "llm_model": llm.get("model"),
        "prompt_tokens": llm.get("prompt_tokens"),
        "completion_tokens": llm.get("completion_tokens"),
        "total_tokens": llm.get("total_tokens"),
        "cached_tokens": llm.get("cached_tokens"),
        "llm_elapsed_sec": llm.get("elapsed_sec"),
        "retrieval_sec": result["retrieval_sec"],
        "n_hits_in_context": result["n_hits_in_context"],
        "context_meta": result["context_meta"],
        "context_text": result["context_text"],
        "run": run,
        "llm_calls": [{"stage": "answer", **{k: llm.get(k) for k in (
            "model", "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "elapsed_sec")},
            **({"raw_token_usage": llm["raw_token_usage"]} if llm.get("raw_token_usage") else {})}],
        "elapsed_sec": result["elapsed_sec"],
    }
