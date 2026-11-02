"""Registry of Cypher-builder tools and the OpenAI function schemas offered to the query LLM.

A tool is a function ``f(arg, ..., database=...) -> (cypher, params)`` registered with ``@register``.
``tools:`` in a method config selects which of them the query generator is offered:

    tools: {preset: generic5, descriptions: concise}      # the five original generic tools
    tools: {preset: extended10, descriptions: guided}     # + five 2Wiki-specific builders
    tools: {names: [neighbors, kin_of], descriptions: concise}
    tools: null                                           # no tools: the model writes Cypher only

Add a tool: decorate a builder with ``@register`` in a module you import (``--plugin``), give it a one-line
entry in a descriptions dict, and list it under ``names``.
"""

from __future__ import annotations

import inspect
import types
from typing import Any, Callable, Union, get_args, get_origin

TOOLS_BY_NAME: dict[str, Callable[..., Any]] = {}

GENERIC5 = ("passages_for_entity", "neighbors", "relation_between", "shortest_path_between", "triples_with_relation")
EXTENDED10 = GENERIC5 + ("attribute_of_entity", "relation_chain", "compare_attribute",
                         "entities_with_relation_to", "kin_of")
PRESETS = {"generic5": GENERIC5, "extended10": EXTENDED10}


def register(func: Callable[..., Any]) -> Callable[..., Any]:
    TOOLS_BY_NAME[func.__name__] = func
    return func


def _ensure_domain_tools() -> None:
    from ..domains.twowiki import tools  # noqa: F401  (registers on import)


def _py_type_to_json_type(annotation: Any) -> str:
    """Python type annotation -> JSON-schema primitive type name."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _py_type_to_json_type(non_none[0])
        return "string"
    if annotation is str:    return "string"
    if annotation is int:    return "integer"
    if annotation is float:  return "number"
    if annotation is bool:   return "boolean"
    if annotation is list or origin is list:
        return "array"
    return "string"


def tool_names(tools_cfg: dict | None) -> list[str]:
    """Names selected by a ``tools`` config section (empty for ``None``)."""
    if not tools_cfg:
        return []
    _ensure_domain_tools()
    if "names" in tools_cfg:
        names = list(tools_cfg["names"])
    else:
        names = list(PRESETS[tools_cfg.get("preset", "generic5")])
    names = [n for n in names if n not in set(tools_cfg.get("exclude") or [])]
    names += [n for n in (tools_cfg.get("include") or []) if n not in names]
    missing = [n for n in names if n not in TOOLS_BY_NAME]
    if missing:
        raise KeyError(f"unknown tool(s): {missing}")
    return names


def descriptions_for(style: str) -> dict[str, str]:
    from ..domains.twowiki.tools import descriptions
    if style == "concise":
        return descriptions.CONCISE
    if style == "guided":
        return descriptions.GUIDED
    raise ValueError(f"unknown description style '{style}' (concise | guided)")


def build_tool_schemas(tools_cfg: dict | None) -> list[dict]:
    """OpenAI function-calling schemas, sorted by tool name; ``database`` is stripped (the caller injects it)."""
    names = sorted(tool_names(tools_cfg))
    if not names:
        return []
    desc = {**descriptions_for(tools_cfg.get("descriptions", "concise")), **(tools_cfg.get("extra_descriptions") or {})}
    schemas: list[dict] = []
    for name in names:
        func = TOOLS_BY_NAME[name]
        properties: dict = {}
        required: list[str] = []
        for p_name, p in inspect.signature(func).parameters.items():
            if p_name == "database":
                continue
            properties[p_name] = {"type": _py_type_to_json_type(p.annotation)}
            if p.default is inspect.Parameter.empty:
                required.append(p_name)
        schemas.append({"type": "function", "function": {
            "name": name,
            "description": desc[name],
            "parameters": {"type": "object", "properties": properties, "required": required},
        }})
    return schemas
