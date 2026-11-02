"""Load a knowledge graph (kg.jsonl[.gz]) into a Neo4j database.

Graph model: ``(:Entity {id, name, type, aliases})``, ``(:Passage {id, title, text})``,
``(:Entity)-[:MENTIONED_IN]->(:Passage)`` and one dynamically typed edge per fact between entities, e.g.
``(:Entity)-[:DIRECTOR {passage_id, name}]->(:Entity)`` where ``name`` is the original relation phrase.

Order matters for fidelity with the paper graph: passages, entities, mentions, then edges, each in file order and
in batches of 500. Edges are created with ``apoc.merge.relationship`` (APOC must be installed; the docker-compose
file enables it). Without APOC (``--no-apoc``), one MERGE statement per relation type is used instead; that
loads the same graph, but the store order of relationships (and so the row order of queries without ORDER BY)
can differ.

    akg graph load [--kg PATH] [--database NAME] [--clean] [--no-apoc]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from ..io import open_text
from ..neo4j_client import Neo4jClient

_BATCH = 500


def read_kg(path: Path) -> dict:
    groups: dict[str, list[dict]] = {"passage": [], "entity": [], "edge": [], "mention": []}
    with open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.pop("kind")
            groups[kind].append(obj)
    return groups


def _batched(rows, size=_BATCH):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def _ensure_database(driver, database: str) -> None:
    """Create ``database`` on a multi-database server; the default database always exists."""
    try:
        with driver.session(database="system") as s:
            s.run(f"CREATE DATABASE `{database}` IF NOT EXISTS").consume()
        for _ in range(60):
            with driver.session(database="system") as s:
                rec = s.run("SHOW DATABASE $name YIELD currentStatus", name=database).single()
            if rec and rec["currentStatus"] == "online":
                return
            time.sleep(0.5)
        raise RuntimeError(f"Database {database} did not come online in time.")
    except Exception as exc:                                  # single-database (Community) server
        if database != "neo4j":
            raise SystemExit(f"cannot create database '{database}' ({type(exc).__name__}: {exc}); this server "
                             f"probably has a single database: use NEO4J_DATABASE=neo4j") from exc


def _run(driver, database: str, cypher: str, **params) -> None:
    with driver.session(database=database) as s:
        s.run(cypher, **params).consume()


def create_indexes(driver, database: str) -> None:
    for st in (
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (p:Passage) REQUIRE p.id IS UNIQUE",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
    ):
        _run(driver, database, st)


def load(database: str, kg: dict, clean: bool, client: Neo4jClient | None = None, use_apoc: bool = True) -> dict:
    client = client or Neo4jClient()
    driver = client.driver
    _ensure_database(driver, database)

    if clean:
        print(f"  cleaning {database} ...")
        _run(driver, database, "MATCH (n) DETACH DELETE n")

    create_indexes(driver, database)

    print(f"  loading {len(kg['passage'])} passages ...")
    for batch in _batched(kg["passage"]):
        _run(driver, database, "UNWIND $rows AS row MERGE (p:Passage {id: row.id}) "
                               "SET p.title = row.title, p.text = row.text", rows=batch)

    print(f"  loading {len(kg['entity'])} entities ...")
    for batch in _batched(kg["entity"]):
        _run(driver, database, "UNWIND $rows AS row MERGE (e:Entity {id: row.id}) "
                               "SET e.name = row.name, e.type = row.type, e.aliases = row.aliases", rows=batch)

    print(f"  loading {len(kg['mention'])} mentions ...")
    for batch in _batched(kg["mention"]):
        _run(driver, database, "UNWIND $rows AS row "
                               "MATCH (e:Entity {id: row.entity_id}) MATCH (p:Passage {id: row.passage_id}) "
                               "MERGE (e)-[:MENTIONED_IN]->(p)", rows=batch)

    print(f"  loading {len(kg['edge'])} edges ({'apoc.merge.relationship' if use_apoc else 'one MERGE per type'}) ...")
    if use_apoc:
        for batch in _batched(kg["edge"]):
            _run(driver, database,
                 "UNWIND $rows AS row "
                 "MATCH (a:Entity {id: row.subject_id}) MATCH (b:Entity {id: row.object_id}) "
                 "CALL apoc.merge.relationship(a, row.rel_type, "
                 "{passage_id: row.passage_id}, {name: row.name}, b) YIELD rel "
                 "RETURN count(*)", rows=batch)
    else:
        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in kg["edge"]:
            by_type[e["rel_type"]].append(e)
        for rel_type, rows in by_type.items():
            if not rel_type.replace("_", "").isalnum():
                raise SystemExit(f"unsafe relationship type {rel_type!r}")
            for batch in _batched(rows):
                _run(driver, database,
                     "UNWIND $rows AS row "
                     "MATCH (a:Entity {id: row.subject_id}) MATCH (b:Entity {id: row.object_id}) "
                     f"MERGE (a)-[r:`{rel_type}` {{passage_id: row.passage_id}}]->(b) ON CREATE SET r.name = row.name",
                     rows=batch)

    with driver.session(database=database) as s:
        counts = s.run(
            "MATCH (e:Entity) WITH count(e) AS ents "
            "MATCH (p:Passage) WITH ents, count(p) AS pas "
            "MATCH (:Entity)-[r]->(:Entity) WITH ents, pas, count(r) AS rels "
            "MATCH (:Entity)-[m:MENTIONED_IN]->(:Passage) "
            "RETURN ents, pas, rels, count(m) AS mentions").single()
    print(f"  in Neo4j[{database}]: {counts['ents']} entities, {counts['pas']} passages, "
          f"{counts['rels']} relation edges, {counts['mentions']} MENTIONED_IN")
    return dict(counts)


def main(argv: list[str] | None = None) -> None:
    from ..config import load_dataset
    ds = load_dataset("2wiki")
    ap = argparse.ArgumentParser(description="Load kg.jsonl into a Neo4j database.")
    ap.add_argument("--kg", default=str(ds["kg"]), help="kg.jsonl[.gz] (default: the shipped graph).")
    ap.add_argument("--database", default=ds["neo4j_database"], help="Target database (default: NEO4J_DATABASE).")
    ap.add_argument("--clean", action="store_true", help="DETACH DELETE the database before loading.")
    ap.add_argument("--no-apoc", action="store_true", help="Do not use APOC for the edges.")
    args = ap.parse_args(argv)

    kg_path = Path(args.kg)
    if not kg_path.exists():
        raise SystemExit(f"kg file not found: {kg_path}")
    print(f"Loading {kg_path} -> Neo4j database '{args.database}'")
    load(args.database, read_kg(kg_path), clean=args.clean, use_apoc=not args.no_apoc)
    print("Done.")


if __name__ == "__main__":
    main()
