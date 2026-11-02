"""2WikiMultihopQA question records: load the dataset split and take the seeded sample.

Every record (a plain dict) has the shape

    {"qid", "question", "answer", "answer_aliases": [], "topic" (dataset ``type``), "level" (number of gold
     triples, a hop proxy), "gold_triples": [[s, r, o], ...], "passages": [{passage_id, title, text,
     is_supporting}, ...], "supporting_passage_ids"}

``passage_id`` = first 16 hex of sha1("<title>|<text>"): derived from content only, so a passage shared by several
questions is one node everywhere downstream (graph, corpus, gold ``relevant_sets``).

The dataset is ``xanhho/2WikiMultihopQA`` (the original 2WikiMultihopQA release, Ho et al., COLING 2020). Its
fields ``context``, ``supporting_facts`` and ``evidences`` are JSON-encoded strings.
"""

from __future__ import annotations

import hashlib
import json
import random

from . import settings as S


def passage_id(title: str, text: str) -> str:
    return hashlib.sha1(f"{title or ''}|{text or ''}".encode("utf-8")).hexdigest()[:16]


def sample_indices(n_total: int, sample_size: int, seed: int) -> list[int]:
    """Deterministic sample of row indices (sorted). sample_size<=0 means 'all'."""
    if sample_size <= 0 or sample_size >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), sample_size))


def make_passage(title: str, text: str, is_supporting: bool) -> dict:
    return {"passage_id": passage_id(title, text), "title": title, "text": text, "is_supporting": bool(is_supporting)}


def _supporting_titles(supporting_facts_json: str) -> set[str]:
    try:
        return {t for t, _sent in json.loads(supporting_facts_json)}
    except Exception:
        return set()


def record_from_row(row: dict) -> dict:
    context = json.loads(row["context"])
    supporting_titles = _supporting_titles(row["supporting_facts"])
    try:
        evidences = json.loads(row["evidences"])
    except Exception:
        evidences = []

    passages, supporting_ids = [], []
    for title, sentences in context:
        text = " ".join(sentences)
        is_sup = title in supporting_titles
        p = make_passage(title, text, is_sup)
        passages.append(p)
        if is_sup:
            supporting_ids.append(p["passage_id"])

    gold_triples = [[s, r, o] for s, r, o in evidences] if evidences else []
    return {
        "qid": row["_id"],
        "question": row["question"],
        "answer": row["answer"],
        "answer_aliases": [],
        "topic": row.get("type") or "unknown",
        "level": max(len(gold_triples), 1),
        "gold_triples": gold_triples,
        "passages": passages,
        "supporting_passage_ids": sorted(set(supporting_ids)),
    }


def load_split(split: str = "validation", dataset_dir: str | None = None):
    """The dataset split: from a local ``save_to_disk`` copy if ``dataset_dir`` is given, else from the Hub."""
    if dataset_dir:
        from datasets import load_from_disk
        ds = load_from_disk(str(dataset_dir))
        return ds[split]
    from datasets import load_dataset
    c = S.cfg()["sample"]
    return load_dataset(c["hf_dataset"], split=split, revision=c.get("hf_revision"))


def load(sample_size: int, seed: int, split: str = "validation", dataset_dir: str | None = None) -> list[dict]:
    ds = load_split(split, dataset_dir)
    return [record_from_row(ds[i]) for i in sample_indices(len(ds), sample_size, seed)]
