"""Decomposers: split a question into sub-questions and decide how they are scheduled.

flat   all sub-questions are independent and run in parallel (one per output line).
chain  labeled output ``[independent]`` / ``[dependentN]``; steps with the same dependent label form an ordered
       chain, each step sees the results of the earlier ones; chains and independent questions run in parallel.
none   no decomposition, no LLM call: the question is the single sub-question.

A decomposer also fixes the prompts that depend on it: the schema it receives, and the answer system prompt
(the chain variant explains tagged steps).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ...domains.twowiki import prompts as P
from ...registry import register

logger = logging.getLogger(__name__)


@dataclass
class Decomposition:
    sub_questions: list[str]
    groups: list[dict]                      # execution units: {"kind", "label", "items": [(index, sub_question)]}
    decomposed: bool
    raw: str = ""
    llm: dict | None = None
    labels: list[str] | None = None         # chain only
    chained: bool = False
    extra: dict = field(default_factory=dict)


def _independent_groups(sub_questions: list[str]) -> list[dict]:
    return [{"kind": "independent", "label": None, "items": [(i, sq)]} for i, sq in enumerate(sub_questions)]


@register("decomposer", "none")
class NoDecomposer:
    chained = False
    answer_system_prompt = P.answer_extraction_system_prompt

    def __init__(self, rt):
        self.rt = rt

    def decompose(self, question: str) -> Decomposition:
        return Decomposition([question], _independent_groups([question]), False)


@register("decomposer", "flat")
class FlatDecomposer:
    """One sub-question per line. Reproduces the paper runs, including their parser quirk: when the model
    answers with a JSON list, the ``[``/``]`` lines and the quoted, comma-terminated items become
    sub-questions. ``strip_json_list: true`` parses such a reply as JSON instead."""

    chained = False
    answer_system_prompt = P.answer_extraction_system_prompt

    def __init__(self, rt, strip_json_list: bool = False):
        self.rt = rt
        self.strip_json_list = strip_json_list

    def parse(self, raw: str, question: str) -> tuple[list[str], bool]:
        if self.strip_json_list:
            try:
                data = json.loads(re.sub(r"^```[a-z]*\n?|```$", "", raw.strip(), flags=re.I | re.M))
                if isinstance(data, list) and data and all(isinstance(x, str) for x in data):
                    return (data, True) if len(data) > 1 else ([question], False)
            except ValueError:
                pass
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if len(lines) == 1:
            return [question], False
        sub_questions = []
        for line in lines:
            cleaned = re.sub(r'^[\d]+[.)]\s*|^[-*•]\s*', '', line).strip()
            if cleaned:
                sub_questions.append(cleaned)
        if not sub_questions:
            sub_questions = [question]
        return sub_questions, len(sub_questions) > 1

    def decompose(self, question: str) -> Decomposition:
        system = P.decision_system_prompt.format(schema=self.rt.schema)
        user = P.decision_user_prompt.format(question=question)
        llm = self.rt.llms["decision"].chat(system, user)
        sub_questions, decomposed = self.parse(llm["content"], question)
        return Decomposition(sub_questions, _independent_groups(sub_questions), decomposed, llm["content"], llm)


# ── chain ──────────────────────────────────────────────────────────────────────────────────────────
_LABEL_RE = re.compile(r'^\[\s*(independent|dependent\d+)\s*\]\s*(.+)$', re.IGNORECASE)
_MARKER_RE = re.compile(r'^[\d]+[.)]\s*|^[-*•]\s*')
_THINK_RE = re.compile(r'<think>.*?</think>', re.IGNORECASE | re.DOTALL)
# A parse with far more sub-questions than this is a runaway output, not a decomposition.
MAX_SUB_QUESTIONS = 30


def parse_labeled(raw: str, question: str) -> tuple[list[str], list[str]]:
    """Raw LLM output -> parallel lists (sub_questions, labels).

    ``<think>`` blocks are stripped. Lines ``[label] text`` are used directly; if at least one line is
    labeled, unlabeled lines are dropped as stray prose. Without any label, lines become ``independent``.
    A single sub-question, or more than MAX_SUB_QUESTIONS, falls back to the original question.
    """
    raw = _THINK_RE.sub('', raw)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    has_any_label = any(_LABEL_RE.match(l) for l in lines)

    sub_questions: list[str] = []
    labels: list[str] = []
    for line in lines:
        m = _LABEL_RE.match(line)
        if m:
            label = m.group(1).lower()
            text = m.group(2).strip()
        elif has_any_label:
            continue
        else:
            label = "independent"
            text = _MARKER_RE.sub('', line).strip()
        if text:
            sub_questions.append(text)
            labels.append(label)

    if len(sub_questions) <= 1 or len(sub_questions) > MAX_SUB_QUESTIONS:
        if len(sub_questions) > MAX_SUB_QUESTIONS:
            logger.warning("decomposer parsed %d sub-questions (> %d); using the original question. question=%r",
                           len(sub_questions), MAX_SUB_QUESTIONS, question)
        return [question], ["independent"]
    return sub_questions, labels


def build_groups(sub_questions: list[str], labels: list[str]) -> list[dict]:
    """Every ``independent`` sub-question is its own unit; each ``dependentN`` label is one ordered unit."""
    units: list[dict] = []
    dependent_units: dict[str, dict] = {}
    for idx, (sq, label) in enumerate(zip(sub_questions, labels)):
        if label == "independent":
            units.append({"kind": "independent", "label": label, "items": [(idx, sq)]})
        else:
            unit = dependent_units.get(label)
            if unit is None:
                unit = {"kind": "dependent", "label": label, "items": []}
                dependent_units[label] = unit
                units.append(unit)
            unit["items"].append((idx, sq))
    return units


@register("decomposer", "chain")
class ChainDecomposer:
    chained = True
    answer_system_prompt = P.answer_extraction_system_prompt_chain

    def __init__(self, rt):
        self.rt = rt

    def decompose(self, question: str) -> Decomposition:
        system = P.decision_system_prompt_chain.format(schema=self.rt.decision_schema)
        user = P.decision_user_prompt.format(question=question)
        llm = self.rt.llms["decision"].chat(system, user)
        sub_questions, labels = parse_labeled(llm["content"], question)
        return Decomposition(sub_questions, build_groups(sub_questions, labels), len(sub_questions) > 1,
                             llm["content"], llm, labels=labels, chained=True)
