"""Answerers turn the evidence of all steps into the final answer.

single_cut   ONE answer call over all evidence; the tail beyond ``max_context_tokens`` is cut.
chunked      evidence over ``max_prompt_tokens`` is split (long row lists are sliced) into chunks; each chunk is
             answered in parallel and a combiner merges the partial answers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from ...domains.twowiki import prompts as P
from ...registry import get, register
from ...tokens import truncate_to_tokens
from .evidence import chunk_steps_by_size, expand_large_steps, payload_of


class _Base:
    def __init__(self, rt, system_prompt: str):
        self.rt = rt
        self.system_prompt = system_prompt

    def extract(self, user_question: str, payload: str) -> dict:
        user = P.answer_extraction_user_prompt.format(user_question=user_question,
                                                      subquestions_and_answers=payload)
        llm = self.rt.llms["answer"].chat(self.system_prompt, user)
        return {"answer": llm["content"], "input_chars": len(payload), "llm": llm}


@register("answerer", "single_cut")
class SingleCutAnswerer(_Base):
    def __init__(self, rt, system_prompt: str, max_context_tokens: int = 1_000_000):
        super().__init__(rt, system_prompt)
        self.max_context_tokens = int(max_context_tokens)

    def answer(self, question: str, steps: list[dict]):
        payload, n_tokens, truncated = truncate_to_tokens(payload_of(steps), self.max_context_tokens)
        record = self.extract(question, payload)
        record["context_tokens"] = n_tokens
        record["context_truncated"] = truncated
        return record, None, None


@register("answerer", "chunked")
class ChunkedAnswerer(_Base):
    def __init__(self, rt, system_prompt: str, max_prompt_tokens: int = 128_000, combiner: dict | None = None):
        super().__init__(rt, system_prompt)
        self.max_prompt_tokens = int(max_prompt_tokens)
        comb = dict(combiner or {"type": "llm"})
        self.combiner = get("combiner", comb.pop("type"))(rt, **comb)

    def _on_chunk(self, question: str, chunk: list[dict]) -> dict:
        return self.extract(question, payload_of(chunk))

    def answer(self, question: str, steps: list[dict]):
        expanded = expand_large_steps(steps, self.max_prompt_tokens)
        chunks = chunk_steps_by_size(expanded, self.max_prompt_tokens)
        if len(chunks) <= 1:
            return self._on_chunk(question, chunks[0] if chunks else []), None, None
        partials: list[dict] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=self.rt.max_workers) as pool:
            futures = {pool.submit(self._on_chunk, question, ch): i for i, ch in enumerate(chunks)}
            for future in as_completed(futures):
                partials[futures[future]] = future.result()
        combined = self.combiner.combine(question, [p["answer"] for p in partials])
        return combined, partials, combined
