"""Query generator: one sub-question -> a Cypher string, either written by the model or built by a tool.

With tools configured, their OpenAI function schemas are offered to the model and the first valid tool call
is turned into Cypher by the registered builder. Without tools the model can only write Cypher. If a tool call
fails to build (or no tool is called), the reply text is used as Cypher (markdown fences stripped).
"""

from __future__ import annotations

import inspect
import re

from ...domains.twowiki import prompts as P
from ...registry import register
from ..tools_registry import TOOLS_BY_NAME, build_tool_schemas


def strip_fences(text: str) -> tuple[str, bool]:
    cleaned = re.sub(r'^```[a-z]*\n?', '', text, flags=re.IGNORECASE)
    cleaned = cleaned.strip().rstrip('`').strip()
    return cleaned, cleaned != text


def call_builder(name: str, args: dict, database: str) -> tuple[str, dict]:
    """Look up the registered builder by name and call it with ``database`` injected."""
    builder = TOOLS_BY_NAME[name]
    sig = inspect.signature(builder)
    call_kwargs = {k: v for k, v in args.items() if k in sig.parameters}
    if "database" in sig.parameters:
        call_kwargs["database"] = database
    return builder(**call_kwargs)


@register("query_generator", "cypher")
class CypherQueryGenerator:
    def __init__(self, rt, tools: dict | None = None):
        self.rt = rt
        self.tools_cfg = tools
        self.schemas = build_tool_schemas(tools)
        self.has_tools = bool(self.schemas)
        self.system = P.get_query_generation_system_prompt(self.has_tools).format(schema=rt.schema)
        self.retry_suffix = P.get_retry_suffix(self.has_tools)

    def generate(self, sub_question: str, attempts_feedback: str | None = None,
                 dependency_context: str | None = None) -> dict:
        user = P.query_generation_user_prompt.format(sub_question=sub_question)
        if dependency_context:      # chain decomposition only: results of the earlier steps of this chain
            user += P.query_generation_dependency_suffix.format(dependency_context=dependency_context)
        if attempts_feedback:
            user += self.retry_suffix.format(attempts_feedback=attempts_feedback)

        llm = self.rt.llms["query"]
        resp = llm.chat(self.system, user, tools=self.schemas or None)
        llm_info = {k: resp[k] for k in ("model", "elapsed_sec", "prompt_tokens", "completion_tokens",
                                         "total_tokens", "cached_tokens", "raw_token_usage")}

        fallback_error = None
        if resp["tool_calls"]:
            tc = resp["tool_calls"][0]
            tool_name, args = tc["name"], tc.get("args") or {}
            try:
                cypher, params = call_builder(tool_name, args, self.rt.database)
                return {"cypher": cypher, "params": params, "tool": tool_name, "tool_args": args,
                        "raw_output": f"tool_call:{tool_name}({args})", "llm": llm_info}
            except Exception as exc:
                fallback_error = f"{type(exc).__name__}: {exc}"

        raw_cypher = resp["content"].strip()
        cypher, stripped = strip_fences(raw_cypher)
        return {"cypher": cypher, "params": {}, "tool": None, "raw_output": raw_cypher,
                "fences_stripped": stripped, "llm": llm_info,
                **({"builder_error": fallback_error} if fallback_error else {})}
