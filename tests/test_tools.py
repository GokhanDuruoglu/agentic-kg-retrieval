import pytest

from agentic_kg.agentic import tools_registry as TR
from agentic_kg.agentic.components.query_generator import call_builder

from conftest import golden

CASES = {
    "wiki-validated-tool": {"preset": "generic5", "descriptions": "concise"},
    "wiki-validated-10tools": {"preset": "extended10", "descriptions": "concise"},
    "wiki-validated-10tools-guided": {"preset": "extended10", "descriptions": "guided"},
}


@pytest.mark.parametrize("name", CASES)
def test_tool_schemas_match_golden(name):
    assert TR.build_tool_schemas(CASES[name]) == golden(f"tool_schemas_{name}.json")


def test_no_tools():
    assert TR.build_tool_schemas(None) == [] and TR.tool_names(None) == []


def test_include_exclude_and_names():
    cfg = {"preset": "generic5", "exclude": ["neighbors"], "include": ["kin_of"]}
    assert "neighbors" not in TR.tool_names(cfg) and "kin_of" in TR.tool_names(cfg)
    assert TR.tool_names({"names": ["neighbors"]}) == ["neighbors"]
    with pytest.raises(KeyError):
        TR.tool_names({"names": ["no_such_tool"]})


def test_tool_cypher_matches_logged():
    cases = golden("tool_cypher.json")
    assert len(cases) > 50
    for c in cases:
        cypher, params = call_builder(c["tool"], c["args"] or {}, "paper-2wiki")
        assert cypher == c["cypher"] and params == c["params"], c["tool"]


def test_register_custom_tool():
    @TR.register
    def my_tool(name: str, database: str = "x"):
        return "MATCH (e:Entity) RETURN e", {"name": name}
    try:
        schemas = TR.build_tool_schemas({"names": ["my_tool"], "extra_descriptions": {"my_tool": "mine"}})
        assert schemas[0]["function"]["description"] == "mine"
        assert schemas[0]["function"]["parameters"]["required"] == ["name"]
    finally:
        TR.TOOLS_BY_NAME.pop("my_tool")
