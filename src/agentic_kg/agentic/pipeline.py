"""The agentic text-to-Cypher pipeline, composed from registered components.

    question -> decomposer -> [query_generator -> Neo4j -> validator]* per sub-question -> answerer

Which implementation fills each slot comes from the method config (see configs/methods/*.yaml); the pipeline
itself only knows the interfaces in ``components/__init__.py``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..registry import get
from .components import evidence  # noqa: F401  (also registers the built-in components)
from .runtime import Runtime


def _build(kind: str, section: dict | None, default: str, rt: Runtime, **kwargs):
    section = dict(section or {"type": default})
    typ = section.pop("type")
    return get(kind, typ)(rt, **kwargs, **section), typ


class AgenticPipeline:
    def __init__(self, cfg: dict, rt: Runtime):
        self.cfg = cfg
        self.rt = rt
        self.decomposer, d_type = _build("decomposer", cfg.get("decomposer"), "flat", rt)
        qg_section = dict(cfg.get("query_generator") or {"type": "cypher"})
        qg_type = qg_section.pop("type")
        self.query_generator = get("query_generator", qg_type)(rt, tools=cfg.get("tools"), **qg_section)
        self.validator, v_type = _build("validator", cfg.get("validator"), "none", rt)
        self.answerer, a_type = _build("answerer", cfg.get("answerer"), "single_cut", rt,
                                       system_prompt=self.decomposer.answer_system_prompt)
        from .tools_registry import tool_names
        self.components = {"decomposer": d_type, "query_generator": qg_type, "validator": v_type,
                           "answerer": a_type, "tools": sorted(tool_names(cfg.get("tools")))}

    # ── one sub-question ─────────────────────────────────────────────────────────────────────────
    def _process_sub_question(self, index: int, sub_question: str, dependency_context: str | None = None) -> dict:
        t0 = time.perf_counter()
        rt = self.rt
        attempts = self.validator.run(
            lambda feedback: self.query_generator.generate(sub_question, feedback, dependency_context),
            lambda qg: rt.neo4j.run_cypher(qg["cypher"], rt.database, params=qg["params"]),
        )
        qg = attempts[-1]["qg"]
        neo = attempts[-1]["neo4j"]
        return {
            "index":         index,
            "sub_question":  sub_question,
            "tool":          qg["tool"],
            "tool_args":     qg.get("tool_args"),
            "cypher":        neo["cypher"],
            "params":        qg["params"],
            "rows":          neo["rows"],
            "row_count":     neo["row_count"],
            "error":         neo["error"],
            "note":          neo.get("note"),
            "original_row_count": neo.get("original_row_count"),
            "attempt_count": len(attempts),
            "validator_ok":  (neo["error"] is None and neo["row_count"] > 0),
            "query_gen_sec": round(sum(a["qg"]["llm"]["elapsed_sec"] for a in attempts), 3),
            "neo4j_sec":     round(sum(a["neo4j"]["elapsed_sec"] for a in attempts), 3),
            "total_sec":     round(time.perf_counter() - t0, 3),
            "_attempts":     attempts,
        }

    def _run_unit(self, unit: dict, chained: bool) -> list[dict]:
        """An independent sub-question, or a dependency chain executed in order."""
        if unit["kind"] == "dependent":
            prior: list[dict] = []
            steps: list[dict] = []
            chain_len = len(unit["items"])
            for pos, (idx, sq) in enumerate(unit["items"], 1):
                step = self._process_sub_question(idx, sq, evidence.format_dependency_context(prior))
                step["label"] = unit["label"]
                step["chain_pos"] = pos
                step["chain_len"] = chain_len
                # an earlier step that ended without a value means this one was built on a missing input
                step["dependency_failed"] = any(
                    (p.get("error") is not None) or not (p.get("rows") or []) for p in prior)
                steps.append(step)
                prior.append(step)
            return steps
        idx, sq = unit["items"][0]
        step = self._process_sub_question(idx, sq)
        if chained:
            step["label"] = unit["label"]
            step["chain_pos"] = step["chain_len"] = 1
        return [step]

    # ── whole question ───────────────────────────────────────────────────────────────────────────
    def run(self, question: str) -> dict:
        t_start = time.perf_counter()
        dec = self.decomposer.decompose(question)
        ordered: list[dict] = [None] * len(dec.sub_questions)
        with ThreadPoolExecutor(max_workers=self.rt.max_workers) as pool:
            futures = {pool.submit(self._run_unit, u, dec.chained): i for i, u in enumerate(dec.groups)}
            for future in as_completed(futures):
                for step in future.result():
                    ordered[step["index"]] = step

        final_answer, partials, combiner_record = self.answerer.answer(question, ordered)

        result = {
            "question":           question,
            "database":           self.rt.database,
            "components":         self.components,
            "answer":             final_answer["answer"],
            "sub_question_count": len(dec.sub_questions),
            "tool_routed_count":  sum(1 for s in ordered if s.get("tool")),
            "answer_chunk_count": 1 if partials is None else len(partials),
            "combiner_used":      combiner_record is not None,
            "context_tokens":     final_answer.get("context_tokens"),
            "context_truncated":  final_answer.get("context_truncated", False),
            "total_sec":          round(time.perf_counter() - t_start, 3),
            "sub_questions":      dec.sub_questions,
            "steps":              ordered,
            "_decision":          dec,
            "_answer":            final_answer,
            "_answer_partials":   partials,
            "_combiner":          combiner_record,
        }
        if dec.chained:
            result["sub_question_labels"] = dec.labels
            result["labeled_sub_questions"] = [f"[{l}] {sq}" for sq, l in zip(dec.sub_questions, dec.labels)]
            result["decision_groups"] = dec.groups
        result["total_tokens"] = _total_tokens(result)
        return result


def _total_tokens(result: dict) -> int:
    def t(llm):
        return (llm or {}).get("total_tokens") or 0
    total = t(result["_decision"].llm)
    total += sum(t(a["qg"]["llm"]) for s in result["steps"] for a in s["_attempts"])
    if result["_answer_partials"] is None:
        total += t(result["_answer"].get("llm"))
    else:
        total += sum(t(p.get("llm")) for p in result["_answer_partials"])
        if result["_combiner"]:
            total += t(result["_combiner"].get("llm"))
    return total
