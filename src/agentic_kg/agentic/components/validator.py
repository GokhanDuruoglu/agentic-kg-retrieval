"""Validators: decide whether a query attempt is accepted or regenerated.

none    exactly one attempt.
retry   execution feedback loop: an attempt that errors or returns 0 rows is regenerated, with all failed
        attempts (Cypher + outcome) in the prompt, up to ``max_attempts``. The LAST attempt is kept.
"""

from __future__ import annotations

from typing import Callable

from ...registry import register
from .evidence import build_attempts_feedback


@register("validator", "none")
class NoValidator:
    max_attempts = 1

    def __init__(self, rt):
        self.rt = rt

    def run(self, generate: Callable[[str | None], dict], execute: Callable[[dict], dict]) -> list[dict]:
        qg = generate(None)
        return [{"attempt": 1, "qg": qg, "neo4j": execute(qg)}]


@register("validator", "retry")
class RetryValidator:
    def __init__(self, rt, max_attempts: int = 3):
        self.rt = rt
        self.max_attempts = int(max_attempts)

    @staticmethod
    def failed(neo: dict) -> bool:
        return (neo["error"] is not None) or (neo["row_count"] == 0)

    def run(self, generate: Callable[[str | None], dict], execute: Callable[[dict], dict]) -> list[dict]:
        attempts: list[dict] = []
        feedback: str | None = None
        for attempt_no in range(1, self.max_attempts + 1):
            qg = generate(feedback)
            neo = execute(qg)
            attempts.append({"attempt": attempt_no, "qg": qg, "neo4j": neo})
            if not self.failed(neo) or attempt_no == self.max_attempts:
                break
            feedback = build_attempts_feedback(attempts)
        return attempts
