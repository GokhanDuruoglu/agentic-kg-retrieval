"""Run a method (agentic or retrieval) over the dataset questions, one repeat at a time.

    runs/<method>/r<k>/records.jsonl   one record per question (answer + full LLM-call accounting)
    runs/<method>/r<k>/errors.jsonl    questions that raised (the batch carries on; ``--resume`` retries them)
    runs/<method>/r<k>/manifest.json   resolved config, models, request params, hashes, graph check, timings
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import records as R
from .agentic.pipeline import AgenticPipeline
from .agentic.runtime import Runtime
from .config import ROOT, Settings, load_dataset, load_method, load_settings, sha_of
from .graph_build import fingerprint as fp
from .io import read_jsonl
from .llm import build_llms
from .neo4j_client import Neo4jClient
from .retrieval.common import RetrievalPipeline, retrieval_record

RUNS = ROOT / "runs"


def code_hash() -> str:
    h = hashlib.sha256()
    base = ROOT / "src" / "agentic_kg"
    for p in sorted(base.rglob("*.py")):
        h.update(str(p.relative_to(base)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def pick_questions(path: Path, limit: int | None) -> list[dict]:
    """All questions, or ``limit`` evenly spaced ones (smoke runs)."""
    items = [q for q in read_jsonl(path) if q.get("question")]
    if limit and limit < len(items):
        idx = sorted({round(i * (len(items) - 1) / max(limit - 1, 1)) for i in range(limit)})
        items = [items[i] for i in idx]
    return items


def build_runtime(settings: Settings, ds: dict, repeat: int, *, needs_graph: bool) -> Runtime:
    from .domains.twowiki.schema import DECISION_SCHEMA, STATIC_SCHEMA
    p = settings.params
    neo = Neo4jClient(query_timeout_sec=p["neo4j"]["query_timeout_sec"],
                      max_rows_per_query=p["pipeline"]["max_rows_per_query"]) if needs_graph else None
    return Runtime(settings=settings, llms=build_llms(settings, repeat), neo4j=neo, database=ds["neo4j_database"],
                   max_workers=int(p["pipeline"]["max_workers"]), schema=STATIC_SCHEMA,
                   decision_schema=DECISION_SCHEMA)


def run_dir(method: str, repeat: int, out_root: Path | None = None) -> Path:
    return (out_root or RUNS) / method / f"r{repeat}"


def run_method(method: str, repeats: list[int], *, limit: int | None = None, resume: bool = True,
               force: bool = False, graph_check: bool = True, out_root: Path | None = None,
               dataset: str = "2wiki") -> list[dict]:
    cfg = load_method(method)
    settings = load_settings()
    ds = load_dataset(dataset)
    kind = cfg.get("kind", "agentic")
    questions = pick_questions(ds["questions"], limit)
    manifests = []

    for rep in repeats:
        rdir = run_dir(method, rep, out_root)
        mpath = rdir / "manifest.json"
        if mpath.exists() and not force and json.loads(mpath.read_text()).get("status") == "complete":
            print(f"skip {method}/r{rep}: already complete")
            continue
        rdir.mkdir(parents=True, exist_ok=True)
        rec_path = rdir / "records.jsonl"
        if force and rec_path.exists():
            rec_path.unlink()

        rt = build_runtime(settings, ds, rep, needs_graph=(kind == "agentic"))
        seed = settings.seed_for("decision", rep)
        run_id = f"{method}__r{rep}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}__{uuid.uuid4().hex[:6]}"
        run_ctx = {"version": method, "run_id": run_id, "repeat": rep, "seed": seed,
                   "models": {r: settings.model_id(r) for r in ("decision", "query", "answer", "judge")}}

        graph_ok = None
        if kind == "agentic" and graph_check and ds["fingerprint"].exists():
            graph_ok, _ = fp.check(ds["neo4j_database"], ds["fingerprint"], rt.neo4j)
            if not graph_ok:
                raise SystemExit(f"graph fingerprint mismatch for database '{ds['neo4j_database']}': it does not "
                                 f"hold the frozen 2Wiki graph (see `akg graph load`, or pass --no-graph-check)")

        pipe = (AgenticPipeline(cfg, rt) if kind == "agentic" else RetrievalPipeline(cfg, rt, ds["corpus"]))
        manifest = {
            "run_id": run_id, "method": method, "kind": kind, "repeat": rep, "seed": seed, "dataset": dataset,
            "status": "running", "config": cfg, "config_sha": sha_of(cfg),
            "models": run_ctx["models"],
            "request_params": {r: settings.request_params(r, rep) for r in ("decision", "query", "answer")},
            "params_sha": sha_of(settings.params), "models_sha": sha_of(settings.models),
            "code_sha": code_hash(), "data": {"questions": file_hash(ds["questions"]), "gold": file_hash(ds["gold"])},
            "graph_database": ds["neo4j_database"] if kind == "agentic" else None,
            "graph_fingerprint_before": None if graph_ok is None else ("ok" if graph_ok else "MISMATCH"),
            "n_expected": len(questions), "started": datetime.now(timezone.utc).isoformat(),
        }
        mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        done = R.done_ids(rec_path) if resume else set()
        print(f"\n=== {method}/r{rep} (seed {seed}): {len(questions)} questions, {len(done)} already done")
        for src in questions:
            if str(src.get("id")) in done:
                continue
            q = src["question"]
            try:
                if kind == "agentic":
                    t0 = time.perf_counter()
                    result = pipe.run(q)
                    record = R.agentic_record(result, src, round(time.perf_counter() - t0, 2), run_ctx, method)
                else:
                    result = pipe.run(q)
                    record = retrieval_record(result, {**src, "question": q}, run_ctx, method, pipe)
                R.append_record(rec_path, record)
                print(f"  {str(src.get('id'))[:10]:<10} {record['answer'][:60]!r:<64} {record['elapsed_sec']}s")
            except Exception as exc:                       # one bad question must not stop the batch
                R.record_error(rdir, src, q, exc, run_ctx)

        n_done = len(R.read_jsonl(rec_path))
        n_err = len(R.read_jsonl(rdir / "errors.jsonl"))
        graph_after = None
        if kind == "agentic" and graph_check and ds["fingerprint"].exists():
            graph_after, _ = fp.check(ds["neo4j_database"], ds["fingerprint"], rt.neo4j)
        manifest.update(n_done=n_done, n_error_lines=n_err, finished=datetime.now(timezone.utc).isoformat(),
                        graph_fingerprint_after=None if graph_after is None else ("ok" if graph_after else "MISMATCH"),
                        status="complete" if (n_done >= len(questions) and graph_after is not False) else "incomplete")
        mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        print(f"    {manifest['status']}: {n_done}/{len(questions)} done, {n_err} error lines")
        manifests.append(manifest)
        if rt.neo4j:
            rt.neo4j.close()
    return manifests
