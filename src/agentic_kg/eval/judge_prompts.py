"""Answer-only judge prompts for 2WikiMultihopQA.

The judge sees only the question, the expected answer (+ accepted aliases; none for this dataset) and the
method's final answer text. It returns JSON ``{"recall", "precision", "correct", "rationale"}``; ``correct`` is the
headline metric (``judge_correct``).
"""

_ANSWER_CHARS = 4000


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"... (truncated, {len(s)} chars total)"


SYSTEM_PROMPT = """You are an expert evaluator for 2WikiMultiHopQA, a \
multi-hop open-domain question-answering benchmark built from Wikipedia. Questions require \
combining facts from multiple Wikipedia passages (e.g. comparing two people's birth dates, \
or chaining "who is X's father" -> "where was he born"). You are judging ONLY a method's \
final natural-language answer against the expected short answer -- you do NOT see the \
retrieved passages or any intermediate reasoning: just the question, the expected answer \
(+ accepted aliases), and the method's answer text.

Judge how well the method's answer matches the expected answer, focusing on substance -- the \
correct entity, date, place, or yes/no verdict -- rather than wording or formatting. Give \
credit for paraphrases, reordering, and equivalent formatting (e.g. date format differences, \
full name vs surname, "yes"/"no" vs the full sentence) -- judge meaning, not exact string \
match. Always respond with a single strict JSON object as specified below, and nothing else \
(no markdown fences, no commentary)."""

OUTPUT_INSTRUCTIONS = """## Your task
Compare the method's final answer against the expected answer above. Respond with ONLY a \
JSON object (no markdown fences, no extra text):

{
  "recall": <float 0-1>,
  "precision": <float 0-1>,
  "correct": <true|false>,
  "rationale": "<= 2 sentences explaining the scores"
}

- "recall": does the method's answer contain the correct fact (the entity/date/place/yes-or-no \
that the expected answer names)? 0 if absent, 1 if present (this benchmark's gold is a single \
short fact, not a set of items).
- "precision": is the method's answer free of incorrect or irrelevant/hallucinated content?
- "correct": does the method's answer correctly answer the question -- i.e. does it match the \
expected answer or one of its accepted aliases?"""


def build_user_prompt(question: str, golds: list[str], answer: str) -> str:
    gold_str = " / ".join(g for g in (golds or []) if g) or "n/a"
    parts = [
        f"## Question\n{question}",
        f"## Expected answer\n{gold_str}",
        f"## Method's answer\n{_truncate((answer or '').strip(), _ANSWER_CHARS)}",
        OUTPUT_INSTRUCTIONS,
    ]
    return "\n\n".join(parts)
