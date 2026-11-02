"""Emit the ground truth + question metadata from the sampled records. The gold answer ships with the dataset; there
are no answer aliases.

Produces two JSONL files :
  <key>_benchmark.jsonl : {id, question, topic, level, operation_type, level_name}
  <key>_relevant.jsonl  : {id, question, relevant_sets, answer_value, answer_aliases}

relevant_sets is the OR-of-AND-sets convention; here a single AND-set of the supporting
passage doc ids ("Passage:<passage_id>", == corpus doc_ids). answer_value is the gold short
answer; answer_aliases carries extra accepted forms for EM/F1.

"""

import argparse
import json
import sys
from pathlib import Path

from .settings import DATA_DIR  # noqa: E402


def build(benchmark: str) -> tuple[Path, Path]:
    rec_path = DATA_DIR / benchmark / "records.jsonl"
    if not rec_path.exists():
        raise SystemExit(f"records.jsonl not found: {rec_path} (run build_kg.py first).")
    records = [json.loads(l) for l in rec_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    bench_path = DATA_DIR / benchmark / f"{benchmark}_benchmark.jsonl"
    rel_path = DATA_DIR / benchmark / f"{benchmark}_relevant.jsonl"

    with bench_path.open("w", encoding="utf-8") as bf, rel_path.open("w", encoding="utf-8") as rf:
        for rec in records:
            qid = rec["qid"]
            question = rec["question"]
            bf.write(json.dumps({
                "id":             qid,
                "question":       question,
                "topic":          rec.get("topic"),
                "level":          rec.get("level"),
                "operation_type": rec.get("topic"),
                "level_name":     f"hop{rec.get('level')}",
            }, ensure_ascii=False) + "\n")

            supporting = sorted(set(rec.get("supporting_passage_ids") or []))
            relevant_sets = [[f"Passage:{pid}" for pid in supporting]] if supporting else []
            rf.write(json.dumps({
                "id":             qid,
                "question":       question,
                "relevant_sets":  relevant_sets,
                "answer_value":   rec.get("answer"),
                "answer_aliases": rec.get("answer_aliases") or [],
            }, ensure_ascii=False) + "\n")

    print(f"  benchmark meta: {len(records)} → {bench_path}")
    print(f"  relevant/gt   : {len(records)} → {rel_path}")
    return bench_path, rel_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Emit harness benchmark + relevant JSONL.")
    ap.add_argument("--benchmark", required=True, help="Benchmark key (<AKG_BUILD_DIR>/<key>/).")
    args = ap.parse_args(argv)
    build(args.benchmark)


if __name__ == "__main__":
    main()
