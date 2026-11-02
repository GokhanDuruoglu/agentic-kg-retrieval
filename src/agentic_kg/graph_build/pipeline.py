"""The graph build as an ordered sequence of steps.

    sample       load the 2WikiMultihopQA validation split, take the seeded sample           -> 2wiki/records.jsonl
    openie       LLM OpenIE per passage (cached) + entity resolution                        -> 2wiki/kg.jsonl
    judge        LLM audit of the extracted triples (drop / fix), resumable                  -> 2wiki-judged/openie_cache.jsonl
    rebuild      rebuild the graph from the audited cache                                    -> 2wiki-judged/kg.jsonl
    consolidate  LLM relation vocabulary -> canonical relation types, alias merges           -> 2wiki-judged/kg.jsonl (final)
    corpus       passage corpus for the retrieval baselines                                  -> 2wiki/2wiki_corpus.jsonl
    gold         question metadata + gold answers                                            -> 2wiki/2wiki_benchmark.jsonl, ..._relevant.jsonl

Intermediate files live in $AKG_BUILD_DIR (default ./build). ``--from-cache`` first copies the shipped caches
(data/2wiki/build_cache) and the records, so no step needs an LLM or the dataset download: the OpenIE cache and
the triple-audit log are complete, and consolidation reuses the shipped relation map. The result is close to, but
not byte-identical with, the shipped graph (consolidation iterates a set whose order depends on the original
hash seed); the shipped data/2wiki/graph/kg.jsonl.gz is the canonical graph of the paper.

    akg graph build --from-cache                 # everything, without LLM calls
    akg graph build --from-raw                   # consolidate the shipped pre-consolidation graph (closest to the paper graph)
    akg graph build --step judge --step rebuild  # selected steps
    akg graph build --install                    # copy the final graph / corpus / gold into data/2wiki/
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

from ..config import ROOT
from . import build_corpus, build_gt, build_kg, consolidate, judge_openie
from .settings import DATA_DIR

STEPS = ["sample", "openie", "judge", "rebuild", "consolidate", "corpus", "gold"]
SHIPPED = ROOT / "data" / "2wiki"


def _gunzip(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)


def seed_from_cache() -> None:
    """Unpack the shipped build caches into the build directory (never overwrites an existing file)."""
    raw, judged = DATA_DIR / "2wiki", DATA_DIR / "2wiki-judged"
    plan = [
        (SHIPPED / "records.jsonl.gz", raw / "records.jsonl", True),
        (SHIPPED / "build_cache" / "openie_cache.jsonl.gz", raw / "openie_cache.jsonl", True),
        (SHIPPED / "build_cache" / "openie_judge.jsonl.gz", raw / "openie_judge.jsonl", True),
        (SHIPPED / "graph" / "relation_map.json", judged / "relation_map.json", False),
        (SHIPPED / "sample.json", raw / "sample.json", False),
        (SHIPPED / "sample.json", judged / "sample.json", False),
        (SHIPPED / "records.jsonl.gz", judged / "records.jsonl", True),
    ]
    for src, dst, gz in plan:
        if dst.exists():
            continue
        if gz:
            _gunzip(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    print(f"  shipped caches unpacked into {DATA_DIR}")


def seed_from_raw() -> None:
    """Start consolidation from the shipped pre-consolidation graph (kg_raw.jsonl) instead of re-running the
    OpenIE / audit / rebuild steps: the closest reproduction of the shipped graph."""
    seed_from_cache()
    judged = DATA_DIR / "2wiki-judged"
    _gunzip(SHIPPED / "build_cache" / "kg_raw.jsonl.gz", judged / "kg.jsonl")
    shutil.copy(SHIPPED / "graph" / "schema.json", judged / "schema.json")


def run(steps: list[str] | None = None, *, from_cache: bool = False, from_raw: bool = False,
        reuse_map: bool = False, dataset_dir: str | None = None, install: bool = False) -> None:
    steps = steps or STEPS
    if from_raw:
        seed_from_raw()
        from_cache = True
        steps = [st for st in steps if st in ("consolidate", "corpus", "gold")]
    elif from_cache:
        seed_from_cache()
    if from_cache:
        reuse_map = True
    records_arg = ["--records", str(DATA_DIR / "2wiki" / "records.jsonl")] if from_cache else \
        (["--dataset-dir", dataset_dir] if dataset_dir else [])

    for step in steps:
        print(f"\n=== graph build: {step}")
        if step == "sample":
            if not from_cache:
                from . import records as R
                import json
                from . import settings as S
                c = S.cfg()["sample"]
                recs = R.load(c["size"], c["seed"], c["split"], dataset_dir)
                out = DATA_DIR / "2wiki"
                out.mkdir(parents=True, exist_ok=True)
                build_kg._write_records(recs, out)
                (out / "sample.json").write_text(json.dumps({"benchmark": "2wiki", "split": c["split"],
                                                             "seed": c["seed"], "sample": c["size"]}, indent=2))
        elif step == "openie":
            build_kg.main(["--out", "2wiki", *records_arg])
        elif step == "judge":
            judge_openie.main(["--in-benchmark", "2wiki", "--out-benchmark", "2wiki-judged"])
        elif step == "rebuild":
            build_kg.main(["--out", "2wiki-judged", "--records", str(DATA_DIR / "2wiki" / "records.jsonl")])
        elif step == "consolidate":
            consolidate.main(["--benchmark", "2wiki-judged", *(["--reuse-map"] if reuse_map else [])])
        elif step == "corpus":
            build_corpus.main(["--benchmark", "2wiki"])
        elif step == "gold":
            build_gt.main(["--benchmark", "2wiki"])
        else:
            raise SystemExit(f"unknown step '{step}' (choose from {STEPS})")

    if install:
        install_outputs()


def install_outputs() -> None:
    """Copy the build outputs over the shipped data (kept in git, so review the diff before committing)."""
    judged, raw = DATA_DIR / "2wiki-judged", DATA_DIR / "2wiki"
    for src, dst in ((judged / "kg.jsonl", SHIPPED / "graph" / "kg.jsonl.gz"),
                     (raw / "2wiki_corpus.jsonl", SHIPPED / "corpus.jsonl.gz")):
        if src.exists():
            with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            print(f"  installed {dst}")
    for src, dst in ((judged / "relation_map.json", SHIPPED / "graph" / "relation_map.json"),
                     (judged / "schema.json", SHIPPED / "graph" / "schema.json"),
                     (raw / "2wiki_relevant.jsonl", SHIPPED / "gold.jsonl")):
        if src.exists():
            shutil.copy(src, dst)
            print(f"  installed {dst}")
