"""Formatting of per-step evidence for the answer stage, row-slicing and chunk packing, retry feedback."""

from __future__ import annotations

import json

from ...tokens import count_tokens

# Char budget for the dependency context handed to a dependent sub-question's query generator.
DEPENDENCY_CONTEXT_MAX_CHARS = 4000


def build_attempts_feedback(attempts: list[dict]) -> str:
    """Render the history of failed attempts into feedback for the next query-generation call."""
    blocks: list[str] = []
    for a in attempts:
        n = a["attempt"]
        cypher = a["qg"]["cypher"]
        neo = a["neo4j"]
        if neo["error"] is not None:
            outcome = f"Outcome: ERROR — {neo['error']}"
        else:
            outcome = (
                "Outcome: executed successfully but returned 0 rows. "
                "The entity name, relation type, or syntax is likely wrong."
            )
        blocks.append(f"Attempt {n} Cypher:\n{cypher}\n{outcome}")
    return "\n\n".join(blocks)


def format_dependency_context(prior_steps: list[dict]) -> str | None:
    """Render already-completed steps of a dependency chain as context for the next step.

    A prior step that errored OR returned 0 rows (after its own retries) is reported as an explicit FAILURE
    that produced no value, so the next step does not read a bare "[]" as an answer.
    """
    if not prior_steps:
        return None

    blocks: list[str] = []
    for s in prior_steps:
        head = f"Earlier sub-question: {s['sub_question']}"
        attempts = s.get("attempt_count", 1)
        rows = s.get("rows") or []
        if s.get("error"):
            body = (f"Result: FAILED after {attempts} attempt(s) — {s['error']}\n"
                    f"This step produced NO value.")
        elif not rows:
            body = (f"Cypher: {s['cypher']}\n"
                    f"Result: FAILED after {attempts} attempt(s) — the query ran but returned "
                    f"NO rows, so this step produced NO value.")
        else:
            rows_json = json.dumps(rows, ensure_ascii=False, default=str)
            if len(rows_json) > DEPENDENCY_CONTEXT_MAX_CHARS:
                kept: list[dict] = []
                size = 2  # opening "[]"
                for row in rows:
                    rj = json.dumps(row, ensure_ascii=False, default=str)
                    if size + len(rj) + 2 > DEPENDENCY_CONTEXT_MAX_CHARS:
                        break
                    kept.append(row)
                    size += len(rj) + 2
                rows_json = (
                    json.dumps(kept, ensure_ascii=False, default=str)
                    + f"  … ({len(rows) - len(kept)} more rows omitted)"
                )
            body = f"Cypher: {s['cypher']}\nResult: {rows_json}"
        blocks.append(f"{head}\n{body}")

    return "\n\n".join(blocks)


def format_step(s: dict) -> str:
    """One step as the answer model sees it."""
    # chain decomposition: tag each step of a chain with its position and whether it is the FINAL step
    label = s.get("label")
    if label and label != "independent" and s.get("chain_len", 1) > 1:
        pos, ln = s["chain_pos"], s["chain_len"]
        role = ("FINAL step — carries this chain's answer" if pos == ln
                else "intermediate step — resolved a value used by later steps")
        head = (f"Sub-question {s['index'] + 1} [{label} · step {pos} of {ln}, {role}]: "
                f"{s['sub_question']}")
    else:
        head = f"Sub-question {s['index'] + 1}: {s['sub_question']}"
    if s.get("tool"):
        args_str = ", ".join(f"{k}={v!r}" for k, v in (s.get("tool_args") or {}).items())
        source = f"Tool: {s['tool']}({args_str})"
    else:
        source = "Source: LLM-generated Cypher"
    note_line = f"\nNote: {s['note']}" if s.get("note") else ""
    if s.get("dependency_failed"):
        note_line += ("\nWarning: an earlier step in this chain produced no value, so this "
                      "result may be unreliable — do not state it as fact.")
    rows_str = json.dumps(s["rows"], ensure_ascii=False, default=str)
    return f"{head}\n{source}\nCypher: {s['cypher']}\nResult: {rows_str}{note_line}"


def payload_of(steps: list[dict]) -> str:
    return "\n\n".join(format_step(s) for s in steps)


def expand_large_steps(steps: list[dict], max_tokens: int) -> list[dict]:
    """Split any step whose formatted text exceeds ``max_tokens`` into virtual sub-steps.

    Each virtual sub-step carries a row-slice of the original step and an annotated sub_question
    ("... [rows X–Y of Z]") so the answer model knows it sees a partial result and the combiner knows to
    merge across slices. Steps that already fit are returned unchanged.
    """
    if max_tokens <= 0:
        return steps

    result: list[dict] = []
    for s in steps:
        if count_tokens(format_step(s)) <= max_tokens:
            result.append(s)
            continue

        rows = s.get("rows") or []
        total = len(rows)

        # per-step overhead (everything except the rows JSON) -> tokens left for rows in each virtual step
        overhead = count_tokens(format_step({**s, "rows": [], "sub_question": s["sub_question"] + f" [rows 0–0 of {total}]"}))
        row_budget = max(max_tokens - overhead - 8, 200)  # ~8-token safety margin

        batches: list[list[dict]] = []
        current_batch: list[dict] = []
        current_size = 1  # opening "[]"
        for row in rows:
            row_json = json.dumps(row, ensure_ascii=False, default=str)
            row_tokens = count_tokens(row_json)
            added = row_tokens + (1 if current_batch else 0)  # ", " separator
            if current_batch and current_size + added > row_budget:
                batches.append(current_batch)
                current_batch = [row]
                current_size = 1 + row_tokens
            else:
                current_batch.append(row)
                current_size += added
        if current_batch:
            batches.append(current_batch)

        offset = 0
        for batch in batches:
            start = offset + 1
            end = offset + len(batch)
            result.append({
                **s,
                "rows":         batch,
                "row_count":    len(batch),
                "sub_question": f"{s['sub_question']} [rows {start}–{end} of {total}]",
            })
            offset = end

    return result


def chunk_steps_by_size(steps: list[dict], max_tokens: int) -> list[list[dict]]:
    """Group steps into contiguous chunks whose formatted text fits ``max_tokens``.

    Call ``expand_large_steps`` first so every step already fits; a step that still exceeds the budget
    (a single enormous row) gets its own chunk as a last resort.
    """
    if max_tokens <= 0 or not steps:
        return [steps]

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_size = 0
    sep_size = 1  # the "\n\n" between formatted steps

    for s in steps:
        text_size = count_tokens(format_step(s))
        added = text_size + (sep_size if current else 0)
        if current and current_size + added > max_tokens:
            chunks.append(current)
            current = [s]
            current_size = text_size
        else:
            current.append(s)
            current_size += added

    if current:
        chunks.append(current)

    return chunks
