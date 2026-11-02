from agentic_kg.agentic.components.decomposer import ChainDecomposer, FlatDecomposer, build_groups, parse_labeled

from conftest import golden, make_runtime


def _groups(g):
    return [(u["kind"], u["label"], [list(i) for i in u["items"]]) for u in g]


def test_decisions_match_logged():
    flat = FlatDecomposer(None)
    cases = golden("decisions.json")
    assert any(c["chain"] for c in cases) and any(not c["chain"] for c in cases)
    for c in cases:
        if c["chain"]:
            sq, labels = parse_labeled(c["raw"], c["question"])
            assert sq == c["sub_questions"] and labels == c["labels"]
            assert _groups(build_groups(sq, labels)) == _groups(c["groups"])
        else:
            assert flat.parse(c["raw"], c["question"])[0] == c["sub_questions"]


def test_flat_keeps_json_list_quirk_and_fix():
    raw = '[\n  "Who directed Alpha?",\n  "Where did he die?"\n]'
    q = "Where did the director of Alpha die?"
    quirk, _ = FlatDecomposer(None).parse(raw, q)
    assert quirk == ["[", '"Who directed Alpha?",', '"Where did he die?"', "]"]
    fixed, decomposed = FlatDecomposer(None, strip_json_list=True).parse(raw, q)
    assert fixed == ["Who directed Alpha?", "Where did he die?"] and decomposed


def test_chain_falls_back_to_original_question():
    assert parse_labeled("[independent] only one", "orig?") == (["orig?"], ["independent"])
    assert parse_labeled("<think>x</think>\n[independent] a\n[dependent1] b\nstray prose", "q")[0] == ["a", "b"]


def test_none_decomposer_makes_no_llm_call():
    from agentic_kg.agentic.components.decomposer import NoDecomposer
    rt = make_runtime()
    d = NoDecomposer(rt).decompose("Q?")
    assert d.sub_questions == ["Q?"] and d.llm is None and not rt.llms["decision"].calls


def test_chain_decomposer_uses_short_schema_and_chain_answer_prompt():
    rt = make_runtime({"decision": lambda s, u, t: "[dependent1] a\n[dependent1] b"})
    d = ChainDecomposer(rt).decompose("Q?")
    assert d.chained and d.labels == ["dependent1", "dependent1"]
    assert rt.decision_schema in rt.llms["decision"].calls[0][0]
    assert "Dependency chains" in ChainDecomposer.answer_system_prompt
