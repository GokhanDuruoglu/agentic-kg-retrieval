import json

import numpy as np

from agentic_kg.eval import efficiency as EFF
from agentic_kg.eval import judge as J
from agentic_kg.eval import stats as ST

from conftest import FakeLLM

PRICING = {"deepseek-v4p1-flash": {"input": 0.30, "cached_input": 0.006, "output": 1.20}}


def test_usd_from_llm_calls():
    rec = {"kind": "retrieval", "elapsed_sec": 2.5, "llm_calls": [
        {"model": "accounts/fireworks/models/deepseek-v4p1-flash", "prompt_tokens": 3209, "completion_tokens": 358}]}
    out = EFF.question_efficiency(rec, PRICING)
    assert abs(out["usd_list"] - (3209 * 0.30 + 358 * 1.20) / 1e6) < 1e-12 and out["latency_sec"] == 2.5
    rec["llm_calls"].append({"model": "other/unpriced", "prompt_tokens": 1, "completion_tokens": 1})
    assert EFF.question_efficiency(rec, PRICING)["usd_list"] is None        # never a silent 0


def test_bootstrap_and_mcnemar():
    a = np.array([0, 0, 1, 1, 0, 1, 0, 0] * 5, float)
    b = np.array([1, 1, 1, 1, 0, 1, 1, 0] * 5, float)
    r = ST.paired_bootstrap(a, b)
    assert r["mean_diff"] > 0 and r["ci_lo"] > 0 and r["p_boot"] < 0.05 and r == ST.paired_bootstrap(a, b)
    m = ST.mcnemar(a > .5, b > .5)
    assert m["a_only"] == 0 and m["b_only"] == 15 and m["p_mcnemar"] < 0.001
    assert ST.mean_std([0.5, 0.7]) == (0.6, float(np.std([0.5, 0.7], ddof=1)))


def test_judge_parses_json_retries_and_flags_errors():
    replies = iter(["not json", '```json\n{"recall": 1, "precision": 0.5, "correct": true, "rationale": "ok"}\n```'])
    llm = FakeLLM("judge", lambda s, u, t: next(replies))
    r = J.judge_one(llm, "q1", "S", "U", "sha")
    assert r["judge_correct"] == 1.0 and r["judge"]["attempts"] == 2 and abs(r["judge_f1"] - 2 * .5 / 1.5) < 1e-9
    bad = J.judge_one(FakeLLM("judge", lambda s, u, t: "nope"), "q1", "S", "U", "sha")
    assert bad["judge_error"] == "invalid_json"
    wrong = J.judge_one(FakeLLM("judge", lambda s, u, t: json.dumps({"recall": 0, "precision": 0, "correct": False})), "q", "S", "U", "s")
    assert wrong["judge_correct"] == 0.0


def test_judge_run_skips_up_to_date_verdicts_and_redoes_changed_answers(tmp_path):
    calls = []
    def h(s, u, t):
        calls.append(u)
        return '{"recall": 1, "precision": 1, "correct": true, "rationale": ""}'
    llm = FakeLLM("judge", h)
    gold = {"1": {"id": "1", "answer_value": "Paris", "answer_aliases": []}}
    out = tmp_path / "r1.jsonl"
    recs = [{"id": "1", "question": "Where?", "answer": "Paris"}]
    J.judge_run(llm, gold, recs, out, workers=1)
    J.judge_run(llm, gold, recs, out, workers=1)
    assert len(calls) == 1                                                # second pass reused the verdict
    recs[0]["answer"] = "Lyon"
    J.judge_run(llm, gold, recs, out, workers=1)
    assert len(calls) == 2 and "Lyon" in calls[1]                         # a regenerated answer is re-judged


def test_report_end_to_end_on_synthetic_runs(tmp_path, monkeypatch):
    import json

    from agentic_kg.eval import report as RP
    from agentic_kg.eval import judge as J

    runs, judg = tmp_path / "runs", tmp_path / "judgments"
    monkeypatch.setattr(J, "JUDGMENTS", judg)
    monkeypatch.setattr(RP, "JUDGMENTS", judg)
    monkeypatch.setattr(RP, "RESULTS", tmp_path / "results")
    questions = [json.loads(l) for l in open(RP.load_dataset("2wiki")["questions"])][:6]

    def make(method, kind, accs, usd_tokens):
        for rep in (1, 2):
            rdir = runs / method / f"r{rep}"
            rdir.mkdir(parents=True)
            recs, jud = [], []
            for q, ok in zip(questions, accs):
                ans = "a"
                calls = [{"stage": "answer", "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                          "prompt_tokens": usd_tokens, "completion_tokens": 10}]
                recs.append({"id": q["id"], "question": q["question"], "answer": ans, "kind": kind,
                             "llm_calls": calls, "elapsed_sec": 2.0 + rep, "total_sec": 2.0 + rep,
                             "steps": [], "sub_question_count": 1})
                import hashlib
                jud.append({"id": q["id"], "answer_sha": hashlib.sha256(ans.encode()).hexdigest()[:12],
                            "judge_correct": float(ok), "judge_recall": float(ok), "judge_precision": float(ok)})
            (rdir / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
            (rdir / "manifest.json").write_text(json.dumps({"status": "complete", "n_expected": 6,
                                                            "config": {"kind": kind}}))
            (judg / method).mkdir(parents=True, exist_ok=True)
            (judg / method / f"r{rep}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in jud))

    make("graph-dt-gpt", "agentic", [1, 0, 0, 0, 1, 0], 1000)
    make("validated", "agentic", [1, 1, 1, 0, 1, 1], 2000)
    out = RP.make_report(runs_root=runs)
    text = out.read_text()
    assert "Graph-DT-GPT" in text and "validated" in text and "Paired comparisons" in text
    t1 = RP.table1(RP.build_rows(None, runs, "2wiki", {"deepseek-v4p1-flash": {"input": 0.3, "output": 1.2}}), {})
    assert t1.loc["validated", "Acc"].startswith("0.833") and t1.loc["Graph-DT-GPT", "Acc"].startswith("0.333")
    assert t1.loc["validated", "USD/q"] == f"{(2000 * 0.3 + 10 * 1.2) / 1e6:.4f}"
    assert t1.loc["Graph-DT-GPT", "med s"] == "3.5"
    assert t1.loc["validated", "Tok (k)"] == "2.0" and t1.loc["Graph-DT-GPT", "Tok (k)"] == "1.0"
