import pytest

from agentic_kg import config as C
from agentic_kg.agentic.pipeline import AgenticPipeline
from agentic_kg.registry import get, load_plugins, names, register

from conftest import make_runtime


def test_deep_merge_semantics():
    base = {"a": {"type": "x", "p": 1}, "b": {"q": 1, "r": 2}, "c": 5}
    assert C.deep_merge(base, {"b": {"q": 9}})["b"] == {"q": 9, "r": 2}
    assert "c" not in C.deep_merge(base, {"c": None})
    assert C.deep_merge(base, {"a": {"type": "y", "z": 1}})["a"] == {"type": "y", "z": 1}   # type change replaces
    assert C.deep_merge(base, {"a": {"p": 2}})["a"] == {"type": "x", "p": 2}


def test_ladder_changes_one_component_per_rung():
    m = {n: C.load_method(n) for n in ("graph-dt-gpt", "non-validated", "validated", "validated-tool",
                                       "cost-validated", "cost-validated-tool")}
    diff = lambda a, b: sorted(k for k in set(m[a]) | set(m[b]) if k != "name" and m[a].get(k) != m[b].get(k))
    assert diff("graph-dt-gpt", "non-validated") == ["answerer"]
    assert diff("non-validated", "validated") == ["validator"]
    assert diff("validated", "validated-tool") == ["tools"]
    assert diff("validated", "cost-validated") == ["decomposer"]
    assert diff("validated-tool", "cost-validated-tool") == ["decomposer"]
    assert diff("cost-validated", "cost-validated-tool") == ["tools"]


def test_every_method_config_loads_and_agentic_ones_build():
    listing = C.list_methods()
    for name in listing["methods"] + listing["ablations"]:
        cfg = C.load_method(name)
        assert cfg["kind"] in ("agentic", "retrieval")
        if cfg["kind"] == "agentic":
            AgenticPipeline(cfg, make_runtime())


def test_extends_cycle_is_detected(tmp_path, monkeypatch):
    d = tmp_path / "methods"
    d.mkdir()
    (d / "a.yaml").write_text("extends: b\n")
    (d / "b.yaml").write_text("extends: a\n")
    monkeypatch.setattr(C, "CONFIGS", tmp_path)
    with pytest.raises(ValueError):
        C.load_method("a")


def test_registry_plugin_component_runs_in_pipeline():
    @register("validator", "test-always-twice")
    class Twice:
        max_attempts = 2

        def __init__(self, rt):
            self.rt = rt

        def run(self, generate, execute):
            out = []
            for i in (1, 2):
                qg = generate(None)
                out.append({"attempt": i, "qg": qg, "neo4j": execute(qg)})
            return out

    rt = make_runtime()
    cfg = C.load_method("graph-dt-gpt")
    cfg["validator"] = {"type": "test-always-twice"}
    result = AgenticPipeline(cfg, rt).run("Q?")
    assert all(s["attempt_count"] == 2 for s in result["steps"])
    assert "test-always-twice" in names("validator")
    with pytest.raises(KeyError):
        get("validator", "nope")
    load_plugins([])


def test_settings_seeds_and_judge_independence():
    s = C.load_settings()
    assert s.seed_for("answer", 2) == 1002 and s.seed_for("judge", 2) == 1000
    assert s.request_params("decision", 1)["top_k"] == 5 and "top_k" not in s.request_params("query", 1)
    assert s.price("accounts/fireworks/models/deepseek-v4p1-flash")["input"] == 0.30
