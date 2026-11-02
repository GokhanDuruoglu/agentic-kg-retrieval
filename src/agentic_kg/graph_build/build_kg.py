"""Build the knowledge graph (kg.jsonl) + ontology (schema.json) from the sampled question records.

OpenIE extraction (cached per passage) -> entity resolution -> typed edges. Also writes the sampled records
(records.jsonl), the single sampling source of truth for the corpus and gold builders.

kg.jsonl is a typed-line JSONL:
  {"kind":"passage","id","title","text"}
  {"kind":"entity","id","name","type","aliases"}
  {"kind":"edge","subject_id","object_id","rel_type","name","passage_id"}
  {"kind":"mention","entity_id","passage_id"}

    akg graph build --step openie|rebuild ...      (see graph_build/pipeline.py for the whole sequence)
"""

import argparse
import json
import random
from pathlib import Path

from . import settings as S


def _write_kg(kg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kg_path = out_dir / "kg.jsonl"
    with kg_path.open("w", encoding="utf-8") as fh:
        for p in kg["passages"]:
            fh.write(json.dumps({"kind": "passage", **p}, ensure_ascii=False) + "\n")
        for e in kg["entities"]:
            fh.write(json.dumps({"kind": "entity", **e}, ensure_ascii=False) + "\n")
        for ed in kg["edges"]:
            fh.write(json.dumps({"kind": "edge", **ed}, ensure_ascii=False) + "\n")
        for m in kg["mentions"]:
            fh.write(json.dumps({"kind": "mention", **m}, ensure_ascii=False) + "\n")

    schema = {"entity_types": kg["entity_types"], "relations": kg["relations"]}
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"  kg.jsonl     : {len(kg['passages'])} passages, {len(kg['entities'])} entities, "
          f"{len(kg['edges'])} edges, {len(kg['mentions'])} mentions")
    print(f"  schema.json  : {len(kg['relations'])} relation types")


def _write_records(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  records.jsonl: {len(records)} questions")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the KG (kg.jsonl) + ontology from sampled records via OpenIE.")
    ap.add_argument("--out", default="2wiki", help="Output build key (<AKG_BUILD_DIR>/<out>/).")
    ap.add_argument("--sample", type=int, default=S.cfg()["sample"]["size"])
    ap.add_argument("--seed", type=int, default=None,
                    help="Sampling seed. Default: the seed of configs/graph_build.yaml, persisted to sample.json "
                         "and reused on resume (so the OpenIE cache stays valid).")
    ap.add_argument("--split", default=S.cfg()["sample"]["split"])
    ap.add_argument("--dataset-dir", default=None, help="Local datasets.save_to_disk copy (default: the Hub).")
    ap.add_argument("--records", default=None,
                    help="Use this records.jsonl[.gz] (e.g. the shipped data/2wiki/records.jsonl.gz) instead of "
                         "loading the dataset.")
    args = ap.parse_args(argv)

    out_dir = S.DATA_DIR / args.out
    meta_path = out_dir / "sample.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        seed, sample = meta["seed"], meta["sample"]
        print(f"Resuming build in {out_dir.name}: reusing seed={seed} sample={sample} (from sample.json)")
    else:
        seed = args.seed if args.seed is not None else S.cfg()["sample"]["seed"]
        sample = args.sample
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"benchmark": "2wiki", "split": args.split, "seed": seed, "sample": sample},
                                        indent=2), encoding="utf-8")
        print(f"New build in {out_dir.name}: seed={seed} sample={sample} (saved to sample.json)")

    if args.records:
        from ..io import read_jsonl
        records = read_jsonl(args.records)
        print(f"  using {len(records)} records from {args.records}")
    else:
        from . import records as R
        print(f"Loading 2wiki [{args.split}] sample={sample} seed={seed} ...")
        records = R.load(sample, seed, args.split, args.dataset_dir)
        print(f"  loaded {len(records)} questions")

    from .openie import build_openie
    kg = build_openie(records, benchmark="2wiki", out_key=args.out)
    _write_records(records, out_dir)
    _write_kg(kg, out_dir)
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
