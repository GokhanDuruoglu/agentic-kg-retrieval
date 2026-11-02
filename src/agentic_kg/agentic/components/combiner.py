"""Combiners merge the partial answers of an evidence set that did not fit one answer call.

llm      one extra LLM call that consolidates the partial answers.
concat   no LLM call; the partial answers are joined with newlines.
"""

from __future__ import annotations

from ...domains.twowiki import prompts as P
from ...registry import register


@register("combiner", "llm")
class LLMCombiner:
    def __init__(self, rt):
        self.rt = rt

    def combine(self, user_question: str, partial_answers: list[str]) -> dict:
        joined = "\n\n---\n\n".join(f"Partial answer {i + 1}:\n{ans}" for i, ans in enumerate(partial_answers))
        user = P.combiner_user_prompt.format(user_question=user_question, partial_answers=joined)
        llm = self.rt.llms["answer"].chat(P.combiner_system_prompt, user)
        return {"answer": llm["content"], "partial_count": len(partial_answers),
                "input_chars": len(joined), "llm": llm}


@register("combiner", "concat")
class ConcatCombiner:
    def __init__(self, rt):
        self.rt = rt

    def combine(self, user_question: str, partial_answers: list[str]) -> dict:
        return {"answer": "\n".join(partial_answers), "partial_count": len(partial_answers),
                "input_chars": sum(len(a) for a in partial_answers), "llm": None}
