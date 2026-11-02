import json
from pathlib import Path

import pytest

from agentic_kg.agentic.runtime import Runtime
from agentic_kg.config import load_settings
from agentic_kg.domains.twowiki.schema import DECISION_SCHEMA, STATIC_SCHEMA

GOLDEN = Path(__file__).parent / "golden"


def golden(name: str):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


class FakeLLM:
    """Scripted LLM: ``handler(system, user, tools) -> content or {"content", "tool_calls"}``; counts calls."""

    def __init__(self, role, handler):
        self.role, self.handler, self.calls, self.model = role, handler, [], f"fake/{role}"

    def chat(self, system, user, tools=None):
        out = self.handler(system, user, tools)
        if isinstance(out, str):
            out = {"content": out}
        self.calls.append((system, user, tools))
        return {"content": out["content"], "model": self.model, "elapsed_sec": 0.01, "prompt_tokens": 10,
                "completion_tokens": 5, "total_tokens": 15, "cached_tokens": None, "raw_token_usage": {},
                "system_chars": len(system), "user_chars": len(user), "response_chars": len(out["content"]),
                "tool_calls": out.get("tool_calls", []), "invalid_tool_calls": []}


class FakeNeo4j:
    """Returns one row per query, except queries containing 'EMPTY' (0 rows) and 'BOOM' (an error)."""

    def __init__(self):
        self.queries = []

    def close(self):
        pass

    def run_cypher(self, cypher, database, params=None):
        self.queries.append(cypher)
        if "BOOM" in cypher:
            return {"rows": [], "row_count": 0, "error": "SyntaxError: boom", "elapsed_sec": 0.0,
                    "database": database, "cypher": cypher, "params": params or {}, "note": None}
        rows = [] if "EMPTY" in cypher else [{"answer": "x"}]
        return {"rows": rows, "row_count": len(rows), "error": None, "elapsed_sec": 0.0, "database": database,
                "cypher": cypher, "params": params or {}, "note": None}


def make_runtime(handlers: dict | None = None, neo=None) -> Runtime:
    handlers = handlers or {}
    default = {
        "decision": lambda s, u, t: "Sub question one?\nSub question two?",
        "query": lambda s, u, t: "MATCH (e:Entity) RETURN e.name",
        "answer": lambda s, u, t: "ANSWER",
    }
    llms = {r: FakeLLM(r, handlers.get(r, default.get(r, lambda s, u, t: "{}"))) for r in
            ("decision", "query", "answer", "judge")}
    return Runtime(settings=load_settings(), llms=llms, neo4j=neo or FakeNeo4j(), database="paper-2wiki",
                   max_workers=2, schema=STATIC_SCHEMA, decision_schema=DECISION_SCHEMA)


@pytest.fixture
def rt():
    return make_runtime()
