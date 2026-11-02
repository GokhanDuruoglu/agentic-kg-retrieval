"""run_method end to end with scripted LLMs: records, manifest, resume, errors."""
import json

from agentic_kg import runner
from agentic_kg.config import load_settings

from conftest import FakeLLM, FakeNeo4j


def fake_build_llms(settings, repeat=1, fail_on=None):
    def answer(s, u, t):
        if fail_on and fail_on in u:
            raise RuntimeError("endpoint down")
        return "ANSWER"
    handlers = {"decision": lambda s, u, t: "x", "query": lambda s, u, t: "MATCH (e:Entity) RETURN e", "answer": answer}
    return {r: FakeLLM(r, handlers.get(r, lambda s, u, t: "{}")) for r in ("decision", "query", "answer", "judge")}


def test_agentic_run_writes_records_manifest_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "build_llms", fake_build_llms)
    monkeypatch.setattr(runner, "Neo4jClient", lambda **kw: FakeNeo4j())
    m = runner.run_method("validated", [1], limit=4, graph_check=False, out_root=tmp_path)[0]
    rdir = tmp_path / "validated" / "r1"
    recs = [json.loads(l) for l in (rdir / "records.jsonl").read_text().splitlines()]
    assert m["status"] == "complete" and len(recs) == 4 and m["n_done"] == 4
    assert recs[0]["method"] == "validated" and recs[0]["run"]["seed"] == 1001
    assert m["config"]["validator"] == {"type": "retry", "max_attempts": 3} and m["models"]["judge"].endswith("glm-5p3")
    # a complete run is skipped; a forced one starts over
    assert runner.run_method("validated", [1], limit=4, graph_check=False, out_root=tmp_path) == []
    assert len(runner.run_method("validated", [1], limit=4, graph_check=False, out_root=tmp_path, force=True)) == 1


def test_failed_question_is_logged_and_retried_on_resume(tmp_path, monkeypatch):
    qs = runner.pick_questions(runner.load_dataset("2wiki")["questions"], 3)
    bad = qs[1]["question"]
    monkeypatch.setattr(runner, "Neo4jClient", lambda **kw: FakeNeo4j())
    monkeypatch.setattr(runner, "build_llms", lambda s, r=1: fake_build_llms(s, r, fail_on=bad))
    m = runner.run_method("graph-dt-gpt", [1], limit=3, graph_check=False, out_root=tmp_path)[0]
    assert m["status"] == "incomplete" and m["n_done"] == 2 and m["n_error_lines"] == 1
    err = json.loads((tmp_path / "graph-dt-gpt" / "r1" / "errors.jsonl").read_text().splitlines()[0])
    assert "endpoint down" in err["error"] and err["id"] == qs[1]["id"]
    monkeypatch.setattr(runner, "build_llms", fake_build_llms)                  # the endpoint is back
    m2 = runner.run_method("graph-dt-gpt", [1], limit=3, graph_check=False, out_root=tmp_path)[0]
    assert m2["status"] == "complete" and m2["n_done"] == 3


def test_retrieval_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "build_llms", fake_build_llms)
    m = runner.run_method("sparse-bm25", [1, 2], limit=3, out_root=tmp_path)
    assert [x["status"] for x in m] == ["complete", "complete"]
    rec = json.loads((tmp_path / "sparse-bm25" / "r2" / "records.jsonl").read_text().splitlines()[0])
    assert len(rec["retrieved"]) == 20 and rec["kind"] == "retrieval" and rec["run"]["seed"] == 1002
