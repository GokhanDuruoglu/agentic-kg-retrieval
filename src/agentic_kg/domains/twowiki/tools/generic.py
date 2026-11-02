"""The five generic graph tools (Cypher builders) for the Entity/Passage graph.

passages_for_entity, neighbors, relation_between, shortest_path_between, triples_with_relation.
Each builder returns ``(cypher, params)``; execution is uniform in the pipeline. Entity names are matched
case-insensitively as substrings against BOTH ``name`` and ``aliases``: entity resolution can fold a passage
title into the aliases, so ``.name`` alone would silently return zero rows.
"""

from ....agentic.tools_registry import register

# Shared name-match predicate: substring match against the canonical name OR any alias.
_NAME_MATCH = (
    "(toLower(e.name) CONTAINS toLower($name) "
    "OR ANY(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($name)))"
)


@register
def passages_for_entity(name: str, database: str = "bench-2wiki") -> tuple[str, dict]:
    """Raw source passage text mentioning ONE named entity (free-text grounding).

    USE: a single entity's free-text detail not captured as a structured edge — a quote, a
    description, a number/date not extracted as its own fact — or to verify original wording.
    AVOID: multi-hop "the R1 of the R2 of X" (one entity's passages lack facts about a second,
    related entity — write chained Cypher); a detail that IS a structured relation (match it).
    """
    cypher = f"""
    MATCH (e:Entity)-[:MENTIONED_IN]->(p:Passage)
    WHERE {_NAME_MATCH}
    RETURN DISTINCT p.title AS title, p.text AS text
    LIMIT 25
    """
    return cypher, {"name": name}


_SEED_MATCH = (
    "(toLower(e.name) CONTAINS toLower($name) "
    "OR ANY(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($name)))"
)
_SUBJECT_MATCH = (
    "(toLower(a.name) CONTAINS toLower($subject) "
    "OR ANY(alias IN a.aliases WHERE toLower(alias) CONTAINS toLower($subject)))"
)
_OBJECT_MATCH = (
    "(toLower(b.name) CONTAINS toLower($object) "
    "OR ANY(alias IN b.aliases WHERE toLower(alias) CONTAINS toLower($object)))"
)


@register
def neighbors(name: str, hops: int = 1, database: str = "bench-2wiki") -> tuple[str, dict]:
    """Entities within `hops` (1-3) of ONE named entity, undirected — open-ended exploration.

    USE: "what is X connected to", or finding candidate bridge entities when you do NOT yet
    know which relation(s) to chain. Returns each 1-hop neighbor WITH its relation type, plus
    names reachable within `hops`.
    AVOID: a named single-hop fact (match it directly: `MATCH (x)-[:REL]-(y)`); multi-hop
    "the R1 of the R2 of X" (write chained Cypher) — this dumps an undifferentiated neighbor
    list the answer model must then guess from.
    """
    h = max(1, min(int(hops or 1), 3))
    cypher = f"""
    MATCH (e:Entity)
    WHERE {_SEED_MATCH}
    MATCH (e)-[r]-(n1:Entity)
    OPTIONAL MATCH path = (e)-[*1..{h}]-(reached:Entity)
    WHERE all(n IN nodes(path) WHERE n:Entity)
    WITH e, collect(DISTINCT {{relation: type(r), neighbor: n1.name}}) AS direct,
         collect(DISTINCT reached.name) AS reachable
    RETURN e.name AS seed, e.type AS type, direct, reachable[..100] AS reachable
    LIMIT 10
    """
    return cypher, {"name": name}


@register
def relation_between(subject: str, object: str,
                     database: str = "bench-2wiki") -> tuple[str, dict]:
    """Direct (1-hop) relation(s) between TWO already-named entities, either direction.

    USE: both entity names are known and you want how/whether they are directly linked —
    verifying a specific link, or a comparison after you already have each entity's facts.
    AVOID: only ONE name known (match its relation in Cypher); entities linked only through an
    unnamed intermediate (use shortest_path_between, or chained Cypher if you know the relations).

    Returns the relation type(s) and provenance passage_id for each connecting edge.
    """
    cypher = f"""
    MATCH (a:Entity), (b:Entity)
    WHERE {_SUBJECT_MATCH}
      AND {_OBJECT_MATCH}
    MATCH (a)-[r]-(b)
    RETURN DISTINCT a.name AS subject, type(r) AS relation, b.name AS object,
           r.passage_id AS passage_id
    LIMIT 50
    """
    return cypher, {"subject": subject, "object": object}


@register
def shortest_path_between(subject: str, object: str,
                          database: str = "bench-2wiki") -> tuple[str, dict]:
    """Shortest relation path between TWO already-named entities, via any intermediates.

    USE: both endpoint names known, you want how they connect but do NOT know the relation
    names to chain — e.g. "how is X connected to Y?". Discovers the bridge AND the relation
    types in one query.
    AVOID: you know the relations to chain from the wording ("the R1 of the R2 of X") — write
    that Cypher directly (more reliable); only ONE name known (use neighbors or write Cypher).

    Returns the ordered entity names along the shortest path and each edge's relation type.
    """
    cypher = f"""
    MATCH (a:Entity), (b:Entity)
    WHERE {_SUBJECT_MATCH}
      AND {_OBJECT_MATCH}
    MATCH p = shortestPath((a)-[*1..6]-(b))
    WHERE all(n IN nodes(p) WHERE n:Entity)
    RETURN [n IN nodes(p) | n.name] AS entities,
           [r IN relationships(p) | type(r)] AS relations,
           length(p) AS hops
    ORDER BY hops
    LIMIT 5
    """
    return cypher, {"subject": subject, "object": object}


@register
def triples_with_relation(relation: str, subject: str | None = None,
                          database: str = "bench-2wiki") -> tuple[str, dict]:
    """All (subject, relation, object) triples for ONE relation type (optionally by subject).

    USE: the question is anchored on a RELATION across many entities, e.g. "list all films
    directed by X" (subject=X, relation="director"), or browsing every fact of one relation type.
    AVOID: all facts of a single entity (match `(e)-[r]-(nb)` for it); multi-hop "the R1 of the
    R2 of X" (returns only ONE hop — write chained Cypher).

    `relation` matches the edge type case-/underscore-insensitively; `subject`, if given,
    restricts by case-insensitive substring (on name or alias).
    """
    cypher = """
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(replace(type(r), '_', ' ')) CONTAINS toLower(replace($relation, '_', ' '))
      AND ($subject IS NULL
           OR toLower(a.name) CONTAINS toLower($subject)
           OR ANY(alias IN a.aliases WHERE toLower(alias) CONTAINS toLower($subject)))
    RETURN a.name AS subject, type(r) AS relation, b.name AS object,
           r.passage_id AS passage_id
    LIMIT 100
    """
    return cypher, {"relation": relation, "subject": subject}
