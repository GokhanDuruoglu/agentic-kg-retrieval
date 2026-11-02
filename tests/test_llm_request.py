import json

from agentic_kg.config import load_settings
from agentic_kg.llm import LLM, _parse_tool_calls

from conftest import golden


def _llm(role, repeat):
    s = load_settings()
    return LLM(role, s.model_id(role), s.request_params(role, repeat), api_key="x", base_url="http://localhost")


def test_request_bodies_equal_the_old_client():
    g = golden("request_bodies.json")
    tools = golden("tool_schemas_wiki-validated-tool.json")
    for role, repeat in (("decision", 1), ("query", 1), ("answer", 1), ("judge", 1)):
        llm = _llm(role, repeat)
        want = g[role]["body"]
        got = llm.request_body("SYS", "USER")
        assert got == want, role
    q = _llm("query", 1)
    assert q.request_body("SYS", "USER", tools=tools) == g["query"]["body_tools"]


def test_tool_call_parsing_matches_langchain_rules():
    raw = [
        {"id": "1", "function": {"name": "a", "arguments": "{bad json"}},
        {"id": "2", "function": {"name": "neighbors", "arguments": '{"name": "X"}'}},
        {"id": "3", "function": {"name": "noargs", "arguments": ""}},
        {"id": "4", "function": {"name": "empty", "arguments": "{}"}},
    ]
    valid, invalid = _parse_tool_calls(raw)
    assert [v["name"] for v in valid] == ["neighbors", "noargs", "empty"]
    assert valid[0]["args"] == {"name": "X"} and valid[1]["args"] == {} and valid[2]["args"] == {}
    assert [i["name"] for i in invalid] == ["a"]


def test_chat_with_stub_client(monkeypatch):
    class Msg:
        content = None
        tool_calls = []
    class Resp:
        choices = [type("C", (), {"message": Msg})]
        usage = type("U", (), {"model_dump": lambda self: {"prompt_tokens": 3, "completion_tokens": 2,
                                                          "total_tokens": 5,
                                                          "prompt_tokens_details": {"cached_tokens": 1}}})()
    llm = _llm("answer", 1)
    monkeypatch.setattr(llm, "_create", lambda body: Resp)
    out = llm.chat("s", "u")
    assert out["content"] == "" and out["prompt_tokens"] == 3 and out["cached_tokens"] == 1
    assert out["total_tokens"] == 5 and out["tool_calls"] == []
