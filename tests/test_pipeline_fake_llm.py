"""Every agentic method config end to end with a scripted LLM and a fake Neo4j (no network)."""
import pytest

from agentic_kg import records as R
from agentic_kg.agentic.pipeline import AgenticPipeline
from agentic_kg.config import list_methods, load_method

from conftest import FakeNeo4j, make_runtime

AGENTIC = [n for n in list_methods()["methods"] + list_methods()["ablations"]
           if load_method(n)["kind"] == "agentic"]


def handlers(first_query_empty=False):
    def decision(system, user, tools):
        if "[independent]" in system:
            return "[dependent1] Who directed Alpha?\n[dependent1] Where did that person die?\n[independent] Beta?"
        return "Who directed Alpha?\nWhere did that person die?"

    def query(system, user, tools):
        if first_query_empty and "Where did" in user and "Attempt 1 Cypher" not in user:
            return "MATCH (e:Entity) WHERE e.name = 'EMPTY' RETURN e"
        if tools and "Who directed" in user:
            return {"content": "", "tool_calls": [{"name": "neighbors", "args": {"name": "Alpha"}, "id": "1"}]}
        return "MATCH (e:Entity) RETURN e.name"
    return {"decision": decision, "query": query}


@pytest.mark.parametrize("name", AGENTIC)
def test_method_runs_end_to_end(name):
    cfg = load_method(name)
    rt = make_runtime(handlers(first_query_empty=True), FakeNeo4j())
    pipe = AgenticPipeline(cfg, rt)
    result = pipe.run("Where did the director of Alpha die?")
    assert result["answer"] == "ANSWER"
    steps = result["steps"]
    assert len(steps) == result["sub_question_count"] >= 2
    chained = cfg["decomposer"]["type"] == "chain"
    assert all(("label" in s) == chained for s in steps)
    if cfg["validator"]["type"] == "retry":
        assert any(s["attempt_count"] == 2 for s in steps)        # the empty first query was regenerated
    else:
        assert all(s["attempt_count"] == 1 for s in steps)
    tools_on = bool(cfg.get("tools"))
    assert (result["tool_routed_count"] > 0) == tools_on
    assert (rt.llms["query"].calls[0][2] is not None) == tools_on   # tool schemas offered only when configured
    # the system prompt follows the toolset
    assert ("function-call" in rt.llms["query"].calls[0][0]) == tools_on
    assert result["combiner_used"] is False and result["answer_chunk_count"] == 1

    rec = R.agentic_record(result, {"id": "q1", "question": "Where?"}, 1.0, {"version": name}, name)
    stages = [c["stage"] for c in rec["llm_calls"]]
    assert stages[0] == "decision" and stages[-1] == "answer" and "query" in stages
    assert sum(c["total_tokens"] for c in rec["llm_calls"]) == rec["total_tokens"]
    assert rec["components"]["answerer"] == cfg["answerer"]["type"]


def test_chain_step_gets_dependency_context_and_flat_does_not():
    rt = make_runtime(handlers(), FakeNeo4j())
    AgenticPipeline(load_method("cost-validated"), rt).run("Q?")
    users = [c[1] for c in rt.llms["query"].calls]
    assert any("Results from earlier sub-questions in this chain" in u for u in users)
    rt2 = make_runtime(handlers(), FakeNeo4j())
    AgenticPipeline(load_method("validated"), rt2).run("Q?")
    assert not any("earlier sub-questions" in c[1] for c in rt2.llms["query"].calls)


def test_validator_gives_feedback_after_error_and_keeps_last_attempt():
    seen = []

    def query(system, user, tools):
        seen.append(user)
        return "MATCH BOOM" if len(seen) == 1 else "MATCH (e:Entity) RETURN e"
    rt = make_runtime({"decision": lambda s, u, t: "single", "query": query})
    result = AgenticPipeline(load_method("validated"), rt).run("Q?")
    assert "Outcome: ERROR — SyntaxError: boom" in seen[1] and result["steps"][0]["attempt_count"] == 2
    assert result["steps"][0]["error"] is None


def test_chunked_answerer_calls_combiner_for_large_evidence():
    big = {"rows": [{"v": "y" * 50 + str(i)} for i in range(3000)]}

    class BigNeo(FakeNeo4j):
        def run_cypher(self, cypher, database, params=None):
            out = super().run_cypher(cypher, database, params)
            out.update(rows=big["rows"], row_count=len(big["rows"]))
            return out
    rt = make_runtime({"decision": lambda s, u, t: "one"}, BigNeo())
    cfg = load_method("non-validated")
    cfg["answerer"] = {"type": "chunked", "max_prompt_tokens": 20000, "combiner": {"type": "llm"}}
    result = AgenticPipeline(cfg, rt).run("Q?")
    assert result["combiner_used"] and result["answer_chunk_count"] > 1
    rec = R.agentic_record(result, {"id": "q"}, 1.0, {}, "x")
    stages = [c["stage"] for c in rec["llm_calls"]]
    assert stages.count("answer_partial") == result["answer_chunk_count"] and stages[-1] == "combiner"
    # concat combiner: no LLM call for the merge
    rt2 = make_runtime({"decision": lambda s, u, t: "one"}, BigNeo())
    cfg["answerer"]["combiner"] = {"type": "concat"}
    r2 = AgenticPipeline(cfg, rt2).run("Q?")
    assert r2["combiner_used"] and "combiner" not in [c["stage"] for c in R.agentic_record(r2, {"id": "q"}, 1, {}, "x")["llm_calls"]]


def test_single_cut_truncates_at_the_token_budget():
    class BigNeo(FakeNeo4j):
        def run_cypher(self, cypher, database, params=None):
            out = super().run_cypher(cypher, database, params)
            out.update(rows=[{"v": "z" * 80} for _ in range(2000)], row_count=2000)
            return out
    rt = make_runtime({"decision": lambda s, u, t: "one"}, BigNeo())
    cfg = load_method("graph-dt-gpt")
    cfg["answerer"] = {"type": "single_cut", "max_context_tokens": 5000}
    result = AgenticPipeline(cfg, rt).run("Q?")
    assert result["context_truncated"] is True and result["context_tokens"] > 5000
