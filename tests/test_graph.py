import gzip
import json
import os

import pytest

from agentic_kg.config import load_dataset
from agentic_kg.graph_build import fingerprint as fp
from agentic_kg.graph_build.load_neo4j import read_kg
from agentic_kg.graph_build.resolve import rel_type_token

DS = load_dataset("2wiki")


def test_shipped_graph_matches_frozen_fingerprint_counts():
    kg = read_kg(DS["kg"])
    ref = json.loads(DS["fingerprint"].read_text())
    assert len(kg["entity"]) == ref["labels"]["Entity"] == 42622 and len(kg["passage"]) == ref["labels"]["Passage"] == 6350
    types = {e["rel_type"] for e in kg["edge"]} | {"MENTIONED_IN"}
    assert types == set(ref["relationships"])
    assert len(types) == 67
    typed_edges = sum(v for k, v in ref["relationships"].items() if k != "MENTIONED_IN")
    assert typed_edges == len(kg["edge"]) - 1 + 0 or typed_edges == len(kg["edge"]) - 1


def test_data_files_are_consistent():
    q = [json.loads(l) for l in open(DS["questions"])]
    gold = {json.loads(l)["id"] for l in open(DS["gold"])}
    assert len(q) == 200 and {x["id"] for x in q} <= gold and len(gold) == 1000
    with gzip.open(DS["corpus"], "rt") as fh:
        corpus = [json.loads(l) for l in fh]
    assert len(corpus) == 6350 and set(corpus[0]) >= {"doc_id", "title", "document", "keywords"}
    assert "vendor_name" not in corpus[0]


def test_rel_type_token():
    assert rel_type_token("date of birth") == "DATE_OF_BIRTH"


@pytest.mark.skipif(not os.environ.get("NEO4J_PASSWORD"), reason="needs a Neo4j server (NEO4J_* in the environment)")
def test_live_graph_fingerprint():
    ok, live = fp.check(DS["neo4j_database"], DS["fingerprint"])
    assert ok, live["labels"]
