"""The code examples of docs/components.md must work (so the docs cannot rot)."""
import re
from pathlib import Path

from agentic_kg.agentic import tools_registry as TR
from agentic_kg.agentic.pipeline import AgenticPipeline
from agentic_kg.config import deep_merge, load_method

from conftest import FakeNeo4j, make_runtime

DOC = (Path(__file__).parent.parent / "docs" / "components.md").read_text(encoding="utf-8")
BLOCKS = re.findall(r"```python\n(.*?)```", DOC, re.S)


def test_custom_validator_example():
    ns: dict = {}
    exec(BLOCKS[0], ns)                                       # registers "retry-once-then-give-up"
    cfg = deep_merge(load_method("validated"), {"validator": {"type": "retry-once-then-give-up", "only_on_error": True}})
    rt = make_runtime(neo=FakeNeo4j())
    result = AgenticPipeline(cfg, rt).run("Q?")
    assert all(s["attempt_count"] == 1 for s in result["steps"])      # rows came back, so no second attempt


def test_custom_tool_example():
    exec(BLOCKS[1], {})
    try:
        cfg = {"preset": "generic5", "include": ["films_by_director"],
               "extra_descriptions": {"films_by_director": "Films directed by ONE named person."}}
        schemas = TR.build_tool_schemas(cfg)
        assert len(schemas) == 6 and any(s["function"]["name"] == "films_by_director" for s in schemas)
    finally:
        TR.TOOLS_BY_NAME.pop("films_by_director", None)
