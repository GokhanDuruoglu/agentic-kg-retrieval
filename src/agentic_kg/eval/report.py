"""Aggregate runs + judgments into the paper's Table 1 columns and the paired comparisons along the ladder.

    akg report [--methods ...]      ->  results/report.md, results/per_question.csv

* accuracy = answer-only ``judge_correct``; mean ± std over repeats of the per-repeat mean over questions. A
  question without an answer counts as 0; a missing verdict for an existing answer stays NaN (see the QC table).
* USD / question = tokens x list price (judge and index-time calls excluded); Tok = prompt + completion tokens per
  question, in thousands.
* "med s" = mean over repeats of the per-repeat MEDIAN latency, "mean s" = mean over repeats of the per-repeat
  MEAN latency (agentic: ``total_sec``; retrieval: retrieval + context + answer call).
* paired comparisons: bootstrap over questions (per-question accuracy averaged over repeats, 95% CI; the test of the
  paper), plus an exact McNemar test on the majority vote as an extra (``eval/stats.py``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, load_dataset, load_settings
from ..io import read_jsonl
from . import efficiency as EFF
from . import stats as ST
from .judge import JUDGMENTS

RESULTS = ROOT / "results"

# Table 1 order and display names (configs/methods/*.yaml names are the paper names).
DISPLAY = {
    "sparse-bm25": "Sparse (BM25)", "dense-e5": "Dense (E5)", "graphrag-local": "GraphRAG (local search)",
    "graph-dt-gpt": "Graph-DT-GPT", "non-validated": "non-validated", "validated": "validated",
    "validated-tool": "validated-tool", "cost-validated": "cost-validated", "cost-validated-tool": "cost-validated-tool",
}
ORDER = list(DISPLAY)

# paired steps of the ladder (A -> B)
PAIRS = [("sparse-bm25", "graph-dt-gpt"), ("graphrag-local", "cost-validated-tool"),
         ("graph-dt-gpt", "non-validated"), ("non-validated", "validated"), ("validated", "validated-tool"),
         ("validated", "cost-validated"), ("validated-tool", "cost-validated-tool"),
         ("cost-validated", "cost-validated-tool")]


def load_judgments(method: str, rep: int) -> dict[str, dict]:
    p = JUDGMENTS / method / f"r{rep}.jsonl"
    return {str(r["id"]): r for r in read_jsonl(p)} if p.exists() else {}


def build_rows(methods: list[str] | None, runs_root: Path, dataset: str, pricing: dict) -> pd.DataFrame:
    ds = load_dataset(dataset)
    questions = {str(q["id"]): q for q in read_jsonl(ds["questions"])}
    rows = []
    for mdir in sorted(p for p in runs_root.iterdir() if p.is_dir()) if runs_root.exists() else []:
        if methods and mdir.name not in methods:
            continue
        for rdir in sorted(mdir.glob("r*")):
            mp = rdir / "manifest.json"
            if not mp.exists() or json.loads(mp.read_text()).get("status") != "complete":
                continue
            rep = int(rdir.name[1:])
            recs = {str(r["id"]): r for r in read_jsonl(rdir / "records.jsonl")}
            ja = load_judgments(mdir.name, rep)
            n_expected = json.loads(mp.read_text()).get("n_expected", len(questions))
            # a smoke run (--limit) is scored on the questions it answered; a full run on all of them,
            # where a missing answer counts as incorrect
            ids = [q for q in questions if q in recs] if n_expected < len(questions) else list(questions)
            for qid in ids:
                q = questions[qid]
                rec = recs.get(qid)
                row = {"method": mdir.name, "repeat": rep, "id": qid, "topic": q.get("topic"), "answered": float(rec is not None)}
                ans = (rec.get("answer") or "").strip() if rec else ""
                j = ja.get(qid)
                stale = bool(j and rec is not None and j.get("answer_sha") != hashlib.sha256(ans.encode()).hexdigest()[:12])
                row["judgment_stale"] = float(stale)
                for k in ("judge_correct", "judge_recall", "judge_precision"):
                    row[k] = (j.get(k) if j and "judge_error" not in j and not stale else (0.0 if rec is None else np.nan))
                if rec is not None:
                    row.update(EFF.question_efficiency(rec, pricing))
                rows.append(row)
    return pd.DataFrame(rows)


def _ms(vals):
    m, s = ST.mean_std(vals)
    return m, s


def table1(df: pd.DataFrame, tool_counts: dict[str, int]) -> pd.DataFrame:
    out = {}
    for m, d in df.groupby("method", sort=False):
        per = d.groupby("repeat")
        acc_m, acc_s = _ms(per["judge_correct"].mean().values)
        lat = d.dropna(subset=["latency_sec"]) if "latency_sec" in d else d
        out[m] = {
            "Tools": tool_counts.get(m, "--"),
            "Acc": f"{acc_m:.3f} ± {acc_s:.3f}" if len(per) > 1 else f"{acc_m:.3f}",
            "USD/q": f"{d['usd_list'].mean():.4f}" if "usd_list" in d and d["usd_list"].notna().any() else "n/a",
            "Tok (k)": f"{d['tokens_total'].mean() / 1000:.1f}" if "tokens_total" in d and d["tokens_total"].notna().any() else "n/a",
            "med s": f"{ST.mean_std(lat.groupby('repeat')['latency_sec'].median().values)[0]:.1f}" if len(lat) else "n/a",
            "mean s": f"{ST.mean_std(lat.groupby('repeat')['latency_sec'].mean().values)[0]:.1f}" if len(lat) else "n/a",
            "repeats": len(per),
        }
    t = pd.DataFrame(out).T
    order = [m for m in ORDER if m in t.index] + sorted(m for m in t.index if m not in ORDER)
    t = t.reindex(order)
    t.index = [DISPLAY.get(m, m) for m in t.index]
    return t


def type_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["method", "repeat", "topic"])["judge_correct"].mean().groupby(["method", "topic"]).mean().unstack()
    order = [m for m in ORDER if m in t.index] + sorted(m for m in t.index if m not in ORDER)
    t = t.reindex(order).round(3)
    t.index = [DISPLAY.get(m, m) for m in t.index]
    return t


def stats_table(df: pd.DataFrame, pairs=PAIRS) -> pd.DataFrame:
    rows = []
    wide = df.pivot_table(index="id", columns=["method", "repeat"], values="judge_correct")
    def per_q(m):
        cols = [c for c in wide.columns if c[0] == m]
        return wide[cols] if cols else None
    for a_name, b_name in pairs:
        A, B = per_q(a_name), per_q(b_name)
        if A is None or B is None:
            continue
        ids = A.dropna().index.intersection(B.dropna().index)
        if len(ids) == 0:
            continue
        a_mean, b_mean = A.loc[ids].mean(axis=1).values, B.loc[ids].mean(axis=1).values
        boot = ST.paired_bootstrap(a_mean, b_mean)
        mc = ST.mcnemar(a_mean >= .5, b_mean >= .5)
        rows.append({"A": a_name, "B": b_name, "n": boot["n"], "mean A": round(float(a_mean.mean()), 3),
                     "mean B": round(float(b_mean.mean()), 3), "B-A": round(boot["mean_diff"], 3),
                     "95% CI": f"[{boot['ci_lo']:+.3f}, {boot['ci_hi']:+.3f}]", "p (bootstrap)": round(boot["p_boot"], 4),
                     "A only": mc["a_only"], "B only": mc["b_only"], "p (McNemar)": round(mc["p_mcnemar"], 4)})
    return pd.DataFrame(rows)


def qc_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "repeat"])
    return pd.DataFrame({"questions": g.size(), "answered": g["answered"].sum(),
                         "judged": g["judge_correct"].apply(lambda s: s.notna().sum()),
                         "stale verdicts": g["judgment_stale"].sum()}).reset_index()


def tool_counts_from_manifests(runs_root: Path) -> dict[str, int]:
    from ..agentic.tools_registry import tool_names
    out = {}
    for mp in runs_root.glob("*/r*/manifest.json"):
        cfg = json.loads(mp.read_text()).get("config", {})
        if cfg.get("kind", "agentic") == "agentic":
            out[mp.parts[-3]] = len(tool_names(cfg.get("tools")))
    return out


def md(df: pd.DataFrame, index=True) -> str:
    return df.to_markdown(index=index)


def make_report(methods: list[str] | None = None, runs_root: Path | None = None, dataset: str = "2wiki") -> Path:
    from ..runner import RUNS
    runs_root = runs_root or RUNS
    settings = load_settings()
    df = build_rows(methods, runs_root, dataset, settings.models.get("pricing") or {})
    if df.empty:
        raise SystemExit("no complete runs found under " + str(runs_root))
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "per_question.csv", index=False)
    parts = ["# Results", "",
             f"Dataset: {dataset}; models: { {r: settings.models['roles'][r]['model'] for r in settings.models['roles']} }.", "",
             "## Table 1: accuracy, cost and latency", "",
             "`Acc` = answer-only judge accuracy (mean ± std over repeats). `USD/q` = list price of the pipeline's LLM calls "
             "(judge excluded). `med s` / `mean s` = per-repeat median / mean latency per question, averaged over repeats.", "",
             md(table1(df, tool_counts_from_manifests(runs_root))), "",
             "## Accuracy by question type", "", md(type_table(df)), "",
             "## Paired comparisons", "", "Paired bootstrap over questions (per-question accuracy averaged over repeats; significant at p < 0.05 when the "
             "95% CI excludes 0), plus an exact McNemar test on the majority vote.", "", md(stats_table(df), index=False), "",
             "## QC", "", md(qc_table(df), index=False), ""]
    out = RESULTS / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts[:12]))
    print("wrote", out)
    return out
