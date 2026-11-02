"""Content fingerprint of a Neo4j database, so a run can prove it saw the frozen graph.

Fingerprint = per-label node counts + md5 over all sorted node-property JSON + per-type relationship counts +
md5 over sorted (type, start key, end key) triples. The ``database`` field is informational and is ignored when
comparing, so a copy of the graph under another database name still matches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..neo4j_client import Neo4jClient


def _md5(lines) -> str:
    h = hashlib.md5()
    for ln in lines:
        h.update(ln.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def fingerprint(db: str, client: Neo4jClient | None = None) -> dict:
    client = client or Neo4jClient()
    out = {"database": db, "labels": {}, "relationships": {}}
    with client.driver.session(database=db) as s:
        node_lines = []
        for rec in s.run("MATCH (n) RETURN labels(n) AS l, properties(n) AS p"):
            lab = ":".join(sorted(rec["l"]))
            out["labels"][lab] = out["labels"].get(lab, 0) + 1
            node_lines.append(lab + "|" + json.dumps(rec["p"], sort_keys=True, default=str, ensure_ascii=False))
        out["nodes_md5"] = _md5(sorted(node_lines))
        rel_lines = []
        for rec in s.run("MATCH (a)-[r]->(b) RETURN type(r) AS t, "
                         "coalesce(toString(a.id), a.name) AS a, coalesce(toString(b.id), b.name) AS b"):
            out["relationships"][rec["t"]] = out["relationships"].get(rec["t"], 0) + 1
            rel_lines.append(f'{rec["t"]}|{rec["a"]}|{rec["b"]}')
        out["relationships_md5"] = _md5(sorted(rel_lines))
    return out


def same(a: dict, b: dict) -> bool:
    keys = ("labels", "relationships", "nodes_md5", "relationships_md5")
    return all(a.get(k) == b.get(k) for k in keys)


def check(db: str, reference_file: Path, client: Neo4jClient | None = None) -> tuple[bool, dict]:
    ref = json.loads(Path(reference_file).read_text(encoding="utf-8"))
    fp = fingerprint(db, client)
    return same(fp, ref), fp
