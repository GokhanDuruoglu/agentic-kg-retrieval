from agentic_kg.agentic.components import evidence as E
from agentic_kg.tokens import count_tokens, truncate_to_tokens

from conftest import golden


def test_logged_evidence_token_counts_and_chunks():
    for c in golden("evidence.json"):
        if c["is_baseline"]:
            assert count_tokens(E.payload_of(c["steps"])) == c["context_tokens"]
        chunks = E.chunk_steps_by_size(E.expand_large_steps(c["steps"], 128000), 128000)
        assert max(len(chunks), 1) == c["chunks"]


def _step(i, rows, **kw):
    return {"index": i, "sub_question": f"sq{i}", "tool": None, "tool_args": None, "cypher": "MATCH (n) RETURN n",
            "params": {}, "rows": rows, "row_count": len(rows), "note": None, "error": None, **kw}


def test_format_step_variants():
    s = _step(0, [{"a": 1}])
    assert E.format_step(s).startswith("Sub-question 1: sq0\nSource: LLM-generated Cypher\nCypher:")
    t = _step(1, [], tool="neighbors", tool_args={"name": "X"})
    assert "Tool: neighbors(name='X')" in E.format_step(t)
    c = _step(2, [], label="dependent1", chain_pos=2, chain_len=2, dependency_failed=True)
    out = E.format_step(c)
    assert "[dependent1 · step 2 of 2, FINAL step" in out and "Warning: an earlier step" in out


def test_large_steps_are_sliced_and_chunked():
    rows = [{"v": "x" * 40 + str(i)} for i in range(400)]
    steps = [_step(0, rows)]
    expanded = E.expand_large_steps(steps, 1500)
    assert len(expanded) > 1 and sum(len(s["rows"]) for s in expanded) == 400
    assert expanded[0]["sub_question"].endswith("[rows 1–%d of 400]" % len(expanded[0]["rows"]))
    chunks = E.chunk_steps_by_size(expanded, 1500)
    assert len(chunks) > 1 and all(count_tokens(E.payload_of(ch)) <= 1500 + 50 for ch in chunks)


def test_truncate_to_tokens():
    text = "alpha beta gamma " * 100
    cut, n, truncated = truncate_to_tokens(text, 50)
    assert truncated and n > 50 and count_tokens(cut) <= 50
    assert truncate_to_tokens(text, 0) == (text, n, False)


def test_dependency_context_failure_notice():
    ok = _step(0, [{"a": 1}], attempt_count=1)
    failed = _step(1, [], attempt_count=3)
    out = E.format_dependency_context([ok, failed])
    assert "Result: [{" in out and "FAILED after 3 attempt(s)" in out and E.format_dependency_context([]) is None
