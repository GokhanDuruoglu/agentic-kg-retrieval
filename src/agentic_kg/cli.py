"""Command line interface: ``akg <command>``.

    akg methods                                   list the method configs (paper methods + ablations)
    akg show METHOD                               print a method's resolved config (after ``extends``)
    akg run METHOD [METHOD ...] [--repeats 1-3] [--limit N] [--force] [--no-resume] [--no-graph-check]
    akg judge [--methods M ...] [--repeats 1-3]   answer-only LLM judge over the records of the runs
    akg report [--methods M ...]                  Table 1 columns, per-type accuracy, paired comparisons
    akg index graphrag [--no-reports]             entity embeddings + communities (+ reports) for GraphRAG
    akg graph build|load|fingerprint              knowledge-graph construction / Neo4j loading / graph check
    akg verify                                    offline checks against the golden files (no LLM, no Neo4j)

``--plugin pkg.module`` (any command) imports a module whose top-level code registers extra components.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import hashseed


def parse_repeats(spec: str) -> list[int]:
    reps: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        reps |= set(range(int(a), int(b or a) + 1))
    return sorted(reps)


def main(argv: list[str] | None = None) -> None:
    hashseed.ensure()          # GraphRAG tie-breaking depends on PYTHONHASHSEED (re-executes once)
    ap = argparse.ArgumentParser(prog="akg", description="Agentic retrieval over knowledge graphs (2WikiMultihopQA).")
    ap.add_argument("--plugin", action="append", default=[], help="module that registers extra components")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("methods", help="list method configs")
    p = sub.add_parser("show", help="print a resolved method config")
    p.add_argument("method")

    p = sub.add_parser("run", help="run methods over the questions")
    p.add_argument("methods", nargs="+")
    p.add_argument("--repeats", default="1-3")
    p.add_argument("--limit", type=int, default=None, help="evenly spaced subset of N questions (smoke run)")
    p.add_argument("--force", action="store_true", help="redo complete runs")
    p.add_argument("--no-resume", action="store_true", help="do not skip questions already answered")
    p.add_argument("--no-graph-check", action="store_true", help="skip the graph fingerprint check")

    p = sub.add_parser("judge", help="answer-only LLM judge")
    p.add_argument("--methods", nargs="*")
    p.add_argument("--repeats", default="1-3")
    p.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("report", help="aggregate runs + judgments")
    p.add_argument("--methods", nargs="*")

    p = sub.add_parser("index", help="build retrieval indexes")
    p.add_argument("what", choices=["graphrag"])
    p.add_argument("--no-reports", action="store_true", help="only re-apply cached community reports")
    p.add_argument("--report-limit", type=int, default=None)
    p.add_argument("--rebuild", action="store_true", help="re-embed entity descriptions")

    p = sub.add_parser("graph", help="knowledge-graph construction and loading")
    gsub = p.add_subparsers(dest="gcmd", required=True)
    b = gsub.add_parser("build", help="build the graph from the dataset (see graph_build/pipeline.py)")
    b.add_argument("--step", action="append", help="run only these steps (repeatable)")
    b.add_argument("--from-cache", action="store_true", help="use the shipped OpenIE/audit caches: no LLM, no download")
    b.add_argument("--from-raw", action="store_true",
                   help="start from the shipped pre-consolidation graph; consolidate + corpus + gold only")
    b.add_argument("--reuse-map", action="store_true", help="reuse the saved relation map in consolidation")
    b.add_argument("--dataset-dir", default=None, help="local datasets.save_to_disk copy of the dataset")
    b.add_argument("--install", action="store_true", help="copy the outputs over data/2wiki/")
    l = gsub.add_parser("load", help="load a kg.jsonl[.gz] into Neo4j", add_help=False)
    l.add_argument("rest", nargs=argparse.REMAINDER)
    f = gsub.add_parser("fingerprint", help="compare the Neo4j graph with the frozen fingerprint")
    f.add_argument("--database", default=None)
    f.add_argument("--write", default=None, help="write the live fingerprint to this file instead of comparing")

    sub.add_parser("verify", help="offline golden checks")

    args = ap.parse_args(argv)
    from .registry import load_plugins
    load_plugins(args.plugin)

    if args.cmd == "methods":
        from .config import list_methods
        for kind, names in list_methods().items():
            print(f"{kind}:\n  " + "\n  ".join(names))
    elif args.cmd == "show":
        from .config import load_method
        import yaml
        print(yaml.safe_dump(load_method(args.method), sort_keys=False))
    elif args.cmd == "run":
        from .runner import run_method
        for m in args.methods:
            run_method(m, parse_repeats(args.repeats), limit=args.limit, resume=not args.no_resume,
                       force=args.force, graph_check=not args.no_graph_check)
    elif args.cmd == "judge":
        from .eval.judge import judge_methods
        from .runner import RUNS
        methods = args.methods or sorted(p.name for p in RUNS.iterdir() if p.is_dir())
        judge_methods(methods, parse_repeats(args.repeats), workers=args.workers)
    elif args.cmd == "report":
        from .eval.report import make_report
        make_report(args.methods)
    elif args.cmd == "index":
        from .config import load_settings
        from .llm import build_llms
        from .retrieval.graphrag.index import build_index
        st = load_settings()
        llms = build_llms(st, 1)
        build_index("2wiki", answer_llm=llms["answer"], model_id=st.model_id("answer"),
                    no_reports=args.no_reports, report_limit=args.report_limit, rebuild=args.rebuild)
    elif args.cmd == "graph":
        if args.gcmd == "build":
            from .graph_build.pipeline import run
            run(args.step, from_cache=args.from_cache, from_raw=args.from_raw, reuse_map=args.reuse_map,
                dataset_dir=args.dataset_dir, install=args.install)
        elif args.gcmd == "load":
            from .graph_build.load_neo4j import main as load_main
            load_main(args.rest)
        else:
            from pathlib import Path

            from .config import load_dataset
            from .graph_build import fingerprint as fp
            ds = load_dataset("2wiki")
            db = args.database or ds["neo4j_database"]
            if args.write:
                live = fp.fingerprint(db)
                Path(args.write).write_text(json.dumps(live, indent=2), encoding="utf-8")
                print(f"fingerprint of {db} written: {live['labels']}")
            else:
                ok, live = fp.check(db, ds["fingerprint"])
                print(f"fingerprint {'OK' if ok else 'MISMATCH'} for {db}: {live['labels']}")
                sys.exit(0 if ok else 1)
    elif args.cmd == "verify":
        from .eval.verify import main as verify_main
        sys.exit(verify_main())


if __name__ == "__main__":
    main()
