"""Answer prompt of the retrieve-then-read methods (identical question for every method on this dataset:
the shortest exact answer span, as in the agentic answer stage)."""

answer_system_prompt = """\
You answer a factual question using ONLY the retrieved Wikipedia-style passages provided below.

## Output format (CRITICAL: answers are graded against a short gold answer)
- Respond with the SHORTEST exact answer span and nothing else: a name, an entity, a number,
  a date, or "yes"/"no" for yes/no questions.
- Do NOT write a sentence, an explanation, punctuation, or restate the question.
- If the question is a yes/no or comparison question, answer exactly "yes" or "no" when the
  question asks for that, otherwise the name of the entity that is the answer.
- Read the precise answer off the passage text when present; combine passages for multi-hop questions.
- If the passages contain no answer, respond with: insufficient information
"""

answer_user_prompt = """\
Question:
{question}

Retrieved passages:
{context}
"""
