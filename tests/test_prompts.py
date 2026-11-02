from agentic_kg.domains.twowiki import prompts as P
from agentic_kg.eval import judge_prompts as JP
from agentic_kg.retrieval import prompts as RP

from conftest import golden


def test_prompts_match_golden():
    g = golden("prompts.json")
    got = {
        "decision_system_prompt": P.decision_system_prompt, "decision_user_prompt": P.decision_user_prompt,
        "query_generation_user_prompt": P.query_generation_user_prompt,
        "answer_extraction_user_prompt": P.answer_extraction_user_prompt,
        "combiner_system_prompt": P.combiner_system_prompt, "combiner_user_prompt": P.combiner_user_prompt,
        "query_system_tools": P.get_query_generation_system_prompt(True),
        "query_system_none": P.get_query_generation_system_prompt(False),
        "retry_tools": P.get_retry_suffix(True), "retry_none": P.get_retry_suffix(False),
        "decision_chain": P.decision_system_prompt_chain, "dependency_suffix": P.query_generation_dependency_suffix,
        "answer_flat": P.answer_extraction_system_prompt, "answer_chain": P.answer_extraction_system_prompt_chain,
        "retrieval_answer_system": RP.answer_system_prompt, "retrieval_answer_user": RP.answer_user_prompt,
        "judge_system": JP.SYSTEM_PROMPT,
        "judge_user_example": JP.build_user_prompt("Who directed Alpha?", ["Jane Doe"], "Jane Doe " * 3),
    }
    assert set(got) == set(g)
    for k in g:
        assert got[k] == g[k], k


def test_no_tool_prompt_mentions_no_tools():
    assert "tool" not in P.get_query_generation_system_prompt(False).lower().replace("toolset", "")
    assert "function-call" in P.get_query_generation_system_prompt(True)
