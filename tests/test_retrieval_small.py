import gzip
import json

from agentic_kg.agentic.runtime import Runtime
from agentic_kg.config import load_method
from agentic_kg.retrieval.bm25 import BM25Retriever
from agentic_kg.retrieval.common import RetrievalPipeline, format_context, retrieval_record

from conftest import make_runtime

DOCS = [
    {"doc_id": "Passage:a", "title": "Alpha", "document": "Alpha. Alpha is a film directed by Jane Doe.", "keywords": ["alpha", "film"]},
    {"doc_id": "Passage:b", "title": "Beta", "document": "Beta. Beta is a river in Spain.", "keywords": ["beta", "river"]},
    {"doc_id": "Passage:c", "title": "Gamma", "document": "Gamma. Gamma was born in Paris.", "keywords": ["gamma", "paris"]},
]


def corpus(tmp_path):
    p = tmp_path / "corpus.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for d in DOCS:
            fh.write(json.dumps(d) + "\n")
    return p


def test_bm25_ranks_and_reads_gz(tmp_path):
    r = BM25Retriever(corpus(tmp_path))
    hits = r.search("who directed the film Alpha", 2)
    assert hits[0]["doc_id"] == "Passage:a" and hits[0]["rank"] == 1 and len(hits) == 2
    assert hits[0]["node_type"] == "Passage" and hits[0]["title"] == "Alpha"


def test_format_context_respects_budget():
    hits = [{"rank": i, "node_type": "Passage", "document": "word " * 50} for i in (1, 2, 3)]
    text, used = format_context(hits, max_tokens=80)
    assert used == 1 and text.startswith("[1] (Passage)")
    assert format_context(hits, max_tokens=0)[1] == 3


def test_retrieval_pipeline_end_to_end(tmp_path):
    cfg = load_method("sparse-bm25")
    rt = make_runtime({"answer": lambda s, u, t: "Jane Doe"})
    pipe = RetrievalPipeline(cfg, rt, corpus(tmp_path))
    res = pipe.run("Who directed Alpha?")
    system, user, _ = rt.llms["answer"].calls[0]
    assert "[1] (Passage) Alpha. Alpha is a film" in user and "Question:\nWho directed Alpha?" in user
    rec = retrieval_record(res, {"id": "q1", "question": "Who directed Alpha?", "topic": "t"}, {"version": "x"}, "sparse-bm25", pipe)
    assert rec["kind"] == "retrieval" and rec["answer"] == "Jane Doe" and rec["llm_calls"][0]["stage"] == "answer"
    assert rec["retrieved"][0]["doc_id"] == "Passage:a" and rec["context_text"] is None
