"""Extended 2Wiki catalog: attribute / chain / comparison / reverse-lookup / kinship builders.

These five are what the extended10 preset adds on top of the original five generic graph tools.
They target the four 2WikiMultihopQA question shapes the original catalog could not serve with a
single deterministic call:

  attribute_of_entity     "when was X born"                         (single hop, ONE entity)
  relation_chain          "where did the director of film X die"    (two hops, ONE named entity)
  compare_attribute       "which of A / B was released first"       (same attribute, TWO entities,
                          optionally through a bridge: "which film's director died earlier")
  entities_with_relation_to  "which films did X direct"            (reverse lookup)
  kin_of                  "who is X's mother / grandfather"         (kinship with inverse edges)

Design notes (all from the induced KG, not assumptions):
  * Edge direction is NOT reliable in this OpenIE graph (a DIRECTOR edge can run film->person or
    person->organization), so every builder matches undirected and reports `entity_is_subject`
    so the answer model can see which way the stored edge ran.
  * Surface forms often live only in `aliases`, so every name filter matches BOTH `name` and
    `aliases` (same rule as the original tools).
  * Relation arguments match the edge type case- and underscore-insensitively, as a substring
    ("date of death" -> DATE_OF_DEATH, "birth" -> DATE_OF_BIRTH + PLACE_OF_BIRTH + BIRTH_NAME).
"""

from ....agentic.tools_registry import register


def _name_match(var: str, param: str) -> str:
    return (f"(toLower({var}.name) CONTAINS toLower(${param}) "
            f"OR ANY(alias IN {var}.aliases WHERE toLower(alias) CONTAINS toLower(${param})))")


def _rel_match(var: str, param: str) -> str:
    return (f"toLower(replace(type({var}), '_', ' ')) "
            f"CONTAINS toLower(replace(${param}, '_', ' '))")


# RELATED_TO is the graph's catch-all edge and carries country / year style attributes that have no
# typed edge. It is also where most noise lives (census years, counts), so it is used ONLY when the
# entity has NO typed edge matching the relation, and ONLY for values whose entity type fits what
# the relation asks for. Anything else returns no rows, so the validator's retry loop gets a chance.
_FALLBACK_TYPE_RULES = (
    (("country", "nationality", "location", "place", "origin", "residence", "born in", "citizen"), ["Location"]),
    (("date", "year", "born", "died", "death", "birth", "release", "publication", "published",
      "founded", "when"), ["Date"]),
)


def _fallback_types(relation: str) -> list[str]:
    r = (relation or "").lower().replace("_", " ")
    for keywords, types in _FALLBACK_TYPE_RULES:
        if any(k in r for k in keywords):
            return types
    return []


@register
def attribute_of_entity(name: str, relation: str, database: str = "bench-2wiki") -> tuple[str, dict]:
    """Value(s) of ONE relation for ONE named entity (a single-hop fact lookup).

    USE: the question asks one attribute of an entity named in the question — "when was X born"
    (relation="date of birth"), "where did X die" ("place of death"), "who directed film X"
    ("director"), "what is X's nationality".
    AVOID: the entity is NOT named but must first be found through another relation ("the
    director of film X ..." -> relation_chain); the same attribute for TWO entities
    (compare_attribute); kinship (kin_of, which also follows inverse edges).

    Returns one row per matching edge: the entity, the relation type, the value, the value's
    entity type, and `entity_is_subject` (true when the stored edge runs entity -> value).
    Returns NO rows when the entity has no such typed edge and no generic RELATED_TO value of a
    fitting type (country -> a place, date -> a date); the fact may then live only in the passage
    text, so write Cypher instead.
    """
    cypher = f"""
    CALL {{
        MATCH (e:Entity)
        WHERE {_name_match('e', 'name')}
        MATCH (e)-[r]-(v:Entity)
        WHERE {_rel_match('r', 'relation')} AND v <> e
        RETURN e.name AS entity, type(r) AS relation, v.name AS value, v.type AS value_type,
               startNode(r) = e AS entity_is_subject, r.passage_id AS passage_id
        UNION
        MATCH (e:Entity)
        WHERE {_name_match('e', 'name')}
          AND NOT EXISTS {{ MATCH (e)-[r0]-(:Entity) WHERE {_rel_match('r0', 'relation')} }}
        MATCH (e)-[r]-(v:Entity)
        WHERE type(r) = 'RELATED_TO' AND v.type IN $fallback_types AND v <> e
        RETURN e.name AS entity, type(r) AS relation, v.name AS value, v.type AS value_type,
               startNode(r) = e AS entity_is_subject, r.passage_id AS passage_id
    }}
    RETURN entity, relation, value, value_type, entity_is_subject, passage_id
    LIMIT 25
    """
    return cypher, {"name": name, "relation": relation, "fallback_types": _fallback_types(relation)}


@register
def relation_chain(name: str, relation1: str, relation2: str,
                   database: str = "bench-2wiki") -> tuple[str, dict]:
    """Two-hop chain from ONE named entity: entity -[relation1]- via -[relation2]- answer.

    USE: "the R2 of the R1 of X" where only X is named — "where did the director of film X die"
    (relation1="director", relation2="place of death"), "when did X's father die"
    (relation1="father", relation2="date of death"), "where was the spouse of X born".
    AVOID: a one-hop question (attribute_of_entity); two named entities (compare_attribute);
    chains longer than two hops (write chained Cypher); kinship wording where the parent may be
    stored only through the other parent's spouse (kin_of, then attribute_of_entity).

    Returns the intermediate (`via`) entity AND the final answer for every path, so the
    bridge can be verified.
    """
    cypher = f"""
    MATCH (x:Entity)
    WHERE {_name_match('x', 'name')}
    MATCH (x)-[r1]-(mid:Entity)
    WHERE {_rel_match('r1', 'relation1')} AND mid <> x
    MATCH (mid)-[r2]-(ans:Entity)
    WHERE {_rel_match('r2', 'relation2')} AND ans <> mid AND ans <> x
    RETURN DISTINCT x.name AS entity, type(r1) AS relation1, mid.name AS via,
           type(r2) AS relation2, ans.name AS answer, ans.type AS answer_type
    LIMIT 50
    """
    return cypher, {"name": name, "relation1": relation1, "relation2": relation2}


@register
def compare_attribute(entity1: str, entity2: str, relation: str,
                      via_relation: str | None = None,
                      database: str = "bench-2wiki") -> tuple[str, dict]:
    """The SAME attribute for TWO named entities, side by side — for comparison questions.

    USE: "which was released first, A or B", "were A and B born in the same country", "who died
    earlier". With `via_relation` the attribute is read from a bridge entity instead: "which
    film's director died earlier, A or B" -> relation="date of death", via_relation="director".
    AVOID: one entity only (attribute_of_entity / relation_chain); a bridge longer than one hop.

    Returns, for each entity (`asked` = the name you passed), the value(s) of the attribute
    (and the bridge entity when `via_relation` is set). The comparison itself (earlier / older /
    same) is left to the answer step; dates come back as stored text.
    Returns NO rows unless BOTH entities have a value (a one-sided result cannot be compared),
    so a retry or hand-written Cypher gets a chance.
    """
    entities = [entity1, entity2]
    n_needed = len(set(entities))
    if via_relation:
        cypher = f"""
        UNWIND $entities AS asked
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower(asked)
           OR ANY(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower(asked))
        MATCH (e)-[r0]-(mid:Entity)
        WHERE {_rel_match('r0', 'via_relation')} AND mid <> e
        CALL {{
            WITH mid, e
            MATCH (mid)-[r]-(v:Entity)
            WHERE {_rel_match('r', 'relation')} AND v <> mid AND v <> e
            RETURN type(r) AS relation, v.name AS value, v.type AS value_type
            UNION
            WITH mid, e
            MATCH (mid)-[r]-(v:Entity)
            WHERE NOT EXISTS {{ MATCH (mid)-[r2]-(:Entity) WHERE {_rel_match('r2', 'relation')} }}
              AND type(r) = 'RELATED_TO' AND v.type IN $fallback_types AND v <> mid AND v <> e
            RETURN type(r) AS relation, v.name AS value, v.type AS value_type
        }}
        WITH asked, collect({{entity: e.name, via: mid.name, relation: relation, value: value,
                              value_type: value_type}}) AS vals
        WITH collect({{asked: asked, vals: vals}}) AS groups
        WHERE size(groups) = $n_needed
        UNWIND groups AS g
        UNWIND g.vals AS x
        RETURN g.asked AS asked, x.entity AS entity, x.via AS via, x.relation AS relation,
               x.value AS value, x.value_type AS value_type
        LIMIT 60
        """
    else:
        cypher = f"""
        UNWIND $entities AS asked
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower(asked)
           OR ANY(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower(asked))
        CALL {{
            WITH e
            MATCH (e)-[r]-(v:Entity)
            WHERE {_rel_match('r', 'relation')} AND v <> e
            RETURN type(r) AS relation, v.name AS value, v.type AS value_type
            UNION
            WITH e
            MATCH (e)-[r]-(v:Entity)
            WHERE NOT EXISTS {{ MATCH (e)-[r2]-(:Entity) WHERE {_rel_match('r2', 'relation')} }}
              AND type(r) = 'RELATED_TO' AND v.type IN $fallback_types AND v <> e
            RETURN type(r) AS relation, v.name AS value, v.type AS value_type
        }}
        WITH asked, collect({{entity: e.name, relation: relation, value: value,
                              value_type: value_type}}) AS vals
        WITH collect({{asked: asked, vals: vals}}) AS groups
        WHERE size(groups) = $n_needed
        UNWIND groups AS g
        UNWIND g.vals AS x
        RETURN g.asked AS asked, x.entity AS entity, x.relation AS relation, x.value AS value,
               x.value_type AS value_type
        LIMIT 60
        """
    return cypher, {"entities": entities, "relation": relation, "via_relation": via_relation,
                    "fallback_types": _fallback_types(relation), "n_needed": n_needed}


@register
def entities_with_relation_to(object_name: str, relation: str,
                              database: str = "bench-2wiki") -> tuple[str, dict]:
    """Entities that hold ONE relation pointing at a named entity — a reverse lookup.

    USE: "which films did X direct / write / produce" (object_name=X, relation="director"),
    "who are the members of organization Y", "which works are based on Z", "who was born in
    place P" (relation="place of birth").
    AVOID: you have the SUBJECT and want its value (attribute_of_entity); the edge type is
    unknown (neighbors lists every relation).

    Returns each matching entity with the relation type and `entity_is_subject` (true when the
    stored edge runs entity -> object, i.e. the entity "has" that relation to the object).
    """
    cypher = f"""
    MATCH (o:Entity)
    WHERE {_name_match('o', 'object_name')}
    MATCH (s:Entity)-[r]-(o)
    WHERE {_rel_match('r', 'relation')} AND s <> o
    RETURN o.name AS object, type(r) AS relation, s.name AS entity, s.type AS entity_type,
           startNode(r) = s AS entity_is_subject
    LIMIT 50
    """
    return cypher, {"object_name": object_name, "relation": relation}


_KINSHIP = ("father", "mother", "parent", "child", "spouse", "sibling",
            "grandparent", "grandchild", "uncle")

# Each kinship is a list of (via label, pattern binding x and k); each becomes one UNION arm.
_KIN_BRANCHES: dict[str, list[tuple[str, str]]] = {
    "father": [
        ("FATHER",               "(x)-[:FATHER]->(k:Entity)"),
        ("CHILD (inverse)",      "(k:Entity)-[:CHILD]->(x)"),
        ("via mother's spouse",      "(x)-[:MOTHER]->(:Entity)-[:SPOUSE]-(k:Entity)"),
    ],
    "mother": [
        ("MOTHER",               "(x)-[:MOTHER]->(k:Entity)"),
        ("CHILD (inverse)",      "(k:Entity)-[:CHILD]->(x)"),
        ("via father's spouse",      "(x)-[:FATHER]->(:Entity)-[:SPOUSE]-(k:Entity)"),
    ],
    "parent": [
        ("FATHER|MOTHER|PARENT", "(x)-[:FATHER|MOTHER|PARENT]->(k:Entity)"),
        ("CHILD (inverse)",      "(k:Entity)-[:CHILD]->(x)"),
        ("via other parent's spouse", "(x)-[:FATHER|MOTHER|PARENT]->(:Entity)-[:SPOUSE]-(k:Entity)"),
    ],
    "child": [
        ("CHILD",                "(x)-[:CHILD]->(k:Entity)"),
        ("FATHER|MOTHER|PARENT (inverse)", "(k:Entity)-[:FATHER|MOTHER|PARENT]->(x)"),
    ],
    "spouse": [
        ("SPOUSE",               "(x)-[:SPOUSE]-(k:Entity)"),
    ],
    "sibling": [
        ("SIBLING",              "(x)-[:SIBLING]-(k:Entity)"),
        ("shared parent",        "(x)-[:FATHER|MOTHER|PARENT]->(:Entity)<-[:FATHER|MOTHER|PARENT]-(k:Entity)"),
    ],
    "grandparent": [
        ("GRANDPARENT",          "(x)-[:GRANDPARENT]->(k:Entity)"),
        ("GRANDCHILD (inverse)", "(k:Entity)-[:GRANDCHILD]->(x)"),
        ("parent of parent",     "(x)-[:FATHER|MOTHER|PARENT]->(:Entity)-[:FATHER|MOTHER|PARENT]->(k:Entity)"),
    ],
    "grandchild": [
        ("GRANDCHILD",           "(x)-[:GRANDCHILD]->(k:Entity)"),
        ("GRANDPARENT (inverse)", "(k:Entity)-[:GRANDPARENT]->(x)"),
        ("child of child",       "(x)-[:CHILD]->(:Entity)-[:CHILD]->(k:Entity)"),
    ],
    "uncle": [
        ("UNCLE",                "(x)-[:UNCLE]-(k:Entity)"),
    ],
}


@register
def kin_of(name: str, kinship: str, database: str = "bench-2wiki") -> tuple[str, dict]:
    """A named person's relatives of ONE kinship type, following inverse and via-spouse edges.

    `kinship` is one of: father, mother, parent, child, spouse, sibling, grandparent,
    grandchild, uncle.
    USE: "who is X's mother", "who are X's children", "X's husband / wife" (spouse),
    "X's grandfather" (grandparent). The same fact is often stored from the OTHER side
    (a CHILD edge instead of FATHER) or only through the other parent's SPOUSE; this tool
    unions all of those and labels each hit with `via`.
    AVOID: a non-kinship attribute (attribute_of_entity); a fact about the relative — chain it
    (relation_chain, e.g. father then date of death).

    Note CHILD / GRANDCHILD inverse hits carry no gender, so for "father" / "mother" prefer the
    rows whose `via` is the gendered edge when both appear.
    """
    key = (kinship or "").strip().lower()
    if key not in _KIN_BRANCHES:
        raise ValueError(f"kinship must be one of {', '.join(_KINSHIP)}; got {kinship!r}")
    seed = f"MATCH (x:Entity) WHERE {_name_match('x', 'name')}"
    arms = [
        f"{seed}\n    MATCH {pattern}\n    WHERE k <> x\n"
        f"    RETURN x.name AS entity, '{via.replace(chr(39), '')}' AS via, k.name AS kin, k.type AS kin_type"
        for via, pattern in _KIN_BRANCHES[key]
    ]
    cypher = "\n    UNION\n    ".join(arms)
    return cypher, {"name": name}
