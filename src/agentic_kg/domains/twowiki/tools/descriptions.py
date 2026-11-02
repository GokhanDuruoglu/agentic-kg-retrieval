"""Tool descriptions sent to the query LLM.

CONCISE   one line per tool: the purpose only.
GUIDED    returns / does NOT return / "Do NOT call this tool - write Cypher instead - when ..." / arguments.
          Every statement is checked against the builder's Cypher (limits, columns, null and empty behaviour).
"""

CONCISE = {
    "passages_for_entity":   "Raw source passage text mentioning ONE named entity.",
    "neighbors":             "Entities within `hops` (1-3) of ONE named entity, with the relation type of each direct neighbor.",
    "relation_between":      "Direct (1-hop) relation(s) between TWO named entities, either direction.",
    "shortest_path_between": "Shortest relation path between TWO named entities, via any intermediates.",
    "triples_with_relation": "All (subject, relation, object) triples for ONE relation type, optionally restricted to a subject.",
    "attribute_of_entity":   "Value(s) of ONE relation for ONE named entity (a single-hop fact lookup).",
    "relation_chain":        "Two-hop chain from ONE named entity: entity -[relation1]- via -[relation2]- answer.",
    "compare_attribute":     "The same attribute for TWO named entities, side by side; optionally via a bridge relation.",
    "entities_with_relation_to": "Entities that hold ONE relation pointing at a named entity (reverse lookup).",
    "kin_of":                "A named person's relatives of ONE kinship type (father, mother, parent, child, spouse, sibling, grandparent, grandchild, uncle).",
}

_REL_VOCAB = ("DATE_OF_BIRTH, DATE_OF_DEATH, PLACE_OF_BIRTH, PLACE_OF_DEATH, CAUSE_OF_DEATH, NATIONALITY, "
              "OCCUPATION, POSITION, EDUCATED_AT, EMPLOYER, MEMBER_OF, AWARD, DIRECTOR, WRITER, PRODUCER, "
              "COMPOSER, CAST_MEMBER, PUBLICATION_DATE, COUNTRY_OF_ORIGIN, LANGUAGE, GENRE, LOCATION, "
              "PART_OF, FOUNDED")

GUIDED = {
    "attribute_of_entity": f"""One entity's value(s) for ONE relation: a date of birth or death, a place of birth or death, a director, a nationality, an occupation, an award.

Returns one row per matching edge: entity, relation, value, value_type, entity_is_subject (true when the stored edge runs entity -> value) and passage_id. When the entity has no typed edge of that relation, generic RELATED_TO values of a fitting type (a place for country / location, a date for date / year) are returned instead. It does NOT follow a second relation and does NOT compare entities.

Do NOT call this tool — write Cypher instead — when the sub-question:
  • Describes the entity only through another entity ("the director of film X ...", "X's husband ...") — use relation_chain (two hops) or kin_of (kinship).
  • Names TWO entities and compares the same attribute — use compare_attribute.
  • Needs a fact that is stored only in the passage text — this tool then returns NO rows; read Passage.text through MENTIONED_IN.

Use this tool only for ONE already-named entity and ONE relation.
name = the entity as written (matched on name and aliases, case-insensitive substring). relation = an edge type or its words, matched case- and underscore-insensitively as a substring: {_REL_VOCAB}. A short word such as "birth" matches several edge types at once.""",

    "relation_chain": """Two-hop chain from ONE named entity: entity -[relation1]- via -[relation2]- answer. Finds a fact about an entity that the question does not name (the bridge).

Returns, for every path: entity, relation1, via (the bridge entity), relation2, answer and answer_type, at most 50 rows. Check that `via` is the intended bridge before trusting `answer`; several bridge candidates can appear. It does NOT handle a third hop and does NOT compare two entities.

Do NOT call this tool — write Cypher instead — when the sub-question:
  • Needs only one hop ("when was X born") — use attribute_of_entity.
  • Names TWO entities to compare ("which of A or B ...") — use compare_attribute, with via_relation for a bridge.
  • Asks for a relative that the graph may store from the other side or through the other parent's spouse ("X's mother", "X's grandfather") — use kin_of first, then attribute_of_entity on the result.
  • Needs three or more hops — write one chained Cypher query.

Use this tool only for a two-hop question anchored on ONE named entity: "where did the director of film X die" (relation1="director", relation2="place of death"), "when did X's father die" ("father", "date of death"), "what is the nationality of X's spouse" ("spouse", "nationality").
name = the single named entity (matched on name and aliases). relation1 = edge from the named entity to the bridge, relation2 = edge from the bridge to the answer; both are case- and underscore-insensitive substrings, so use full edge names (DIRECTOR, FATHER, MOTHER, SPOUSE, ... then DATE_OF_DEATH, PLACE_OF_BIRTH, NATIONALITY, ...). No rows means a relation name did not match.""",

    "compare_attribute": """The SAME attribute for TWO named entities, side by side, so the answer step can compare them. With via_relation the attribute is read from a bridge entity one hop away instead.

Returns rows tagged `asked` (the name you passed) with entity, relation, value, value_type (and via when via_relation is set), at most 60 rows. Dates come back as stored text in mixed formats — compare them yourself. If EITHER entity has no value, NO rows are returned (a one-sided result cannot be compared). It does NOT decide the comparison.

Do NOT call this tool — write Cypher instead — when the sub-question:
  • Involves only one entity — use attribute_of_entity or relation_chain.
  • Needs a bridge more than one hop away, or asks for a list rather than a comparison.
  • Got NO rows from this tool — fetch each entity separately with attribute_of_entity, or read the passage text.

Use this tool only for: "which film came out first, A or B" (relation="publication date"), "were A and B from the same country" (relation="country of origin"), "who died earlier, A or B" ("date of death"), and with a bridge "which film has the older director, A or B" (relation="date of birth", via_relation="director").
entity1, entity2 = the two names as written. relation = attribute edge (full names such as DATE_OF_BIRTH, PUBLICATION_DATE, COUNTRY_OF_ORIGIN, NATIONALITY). via_relation = optional edge from each entity to its bridge (DIRECTOR, FATHER, MOTHER, SPOUSE, PRODUCER).""",

    "entities_with_relation_to": """Reverse lookup: the entities that hold ONE relation pointing AT a named entity.

Returns up to 50 rows: object, relation, entity, entity_type and entity_is_subject (true when the stored edge runs entity -> object). Edge direction in this graph is not reliable, so treat entity_is_subject as a hint. It does NOT return the subject's own attributes.

Do NOT call this tool — write Cypher instead — when the sub-question:
  • Has the SUBJECT and wants its value ("who directed film X") — use attribute_of_entity.
  • Does not know the relation type — use neighbors to list every relation of the entity.
  • Needs a second hop from the results — write chained Cypher.

Use this tool only for: "which films did X direct" (object_name=X, relation="director"), "who are the cast members of film Y", "who was born in place P" ("place of birth"), "which works are based on Z" ("based on"), "who is a member of organization O" ("member of").
object_name = the entity the relation points at (name and alias match). relation = edge type or its words (DIRECTOR, WRITER, PRODUCER, CAST_MEMBER, PLACE_OF_BIRTH, MEMBER_OF, BASED_ON, ...).""",

    "kin_of": """A named person's relatives of ONE kinship type, following inverse edges and the other parent's spouse.

Returns rows of entity, via (which edge pattern produced the hit), kin and kin_type. The same fact is often stored from the other side (a CHILD edge instead of FATHER) or only through the other parent's SPOUSE; this tool unions all of those. CHILD / GRANDCHILD (inverse) hits carry no gender, so prefer hits whose via is the gendered edge (FATHER, MOTHER). It does NOT return any attribute of the relative.

Do NOT call this tool — write Cypher instead — when the sub-question:
  • Asks a non-kinship attribute of the person — use attribute_of_entity.
  • Wants a fact ABOUT the relative ("when did X's father die") — use relation_chain, or kin_of and then attribute_of_entity on the returned name.
  • Got NO rows — no kin edge of any pattern exists; read the passage text.

Use this tool only for: "who is X's mother / father", "who are X's children", "X's husband / wife" (spouse), "X's brother or sister" (sibling), "X's grandfather" (grandparent), "X's uncle".
name = the person as written (name and alias match). kinship = one of father, mother, parent, child, spouse, sibling, grandparent, grandchild, uncle.""",

    "neighbors": """What ONE named entity is directly connected to, up to 1-3 hops, undirected — open-ended exploration when the relation is not yet known.

Returns per matching entity: seed, type, direct (every 1-hop neighbor WITH its relation type) and reachable (names within `hops`, at most 100). The names in `direct` are bridge candidates for the next step. It does NOT answer a specific attribute and does NOT rank anything.

Do NOT call this tool — write Cypher instead — when the sub-question:
  • Asks a specific attribute or a two-hop fact ("the R1 of the R2 of X") — use attribute_of_entity or relation_chain, or write chained Cypher; this returns an undifferentiated neighbor list you would have to guess from.
  • Names TWO entities — use relation_between or compare_attribute.

Use this tool only for "what is X connected to" style questions, or to discover candidate bridge entities and relation types.
name = the entity (name and alias match). hops = 1 (default) to 3.""",

    "relation_between": """The direct (1-hop) relation(s) between TWO already-named entities, in either direction.

Returns up to 50 rows: subject, relation, object and passage_id. No rows means there is no DIRECT edge, not that the entities are unrelated. It does NOT search through intermediate entities.

Do NOT call this tool — write Cypher instead — when the sub-question:
  • Names only ONE entity — use attribute_of_entity or neighbors.
  • Links the entities through an unnamed intermediate — use shortest_path_between, or relation_chain when the relations are known.

Use this tool only to verify or read how two named entities are directly linked: "is X the spouse of Y", "did X work with Y".
subject, object = the two entity names (name and alias match, case-insensitive substring).""",

    "shortest_path_between": """The shortest chain of relations between TWO named entities through any intermediate entities, up to 6 hops.

Returns up to 5 paths: entities (ordered names along the path), relations (relation type of each edge) and hops. It discovers the bridge AND the relation types in one query. Paths of 4+ hops usually go through generic hub entities and are not the intended relation. It does NOT return attributes.

Do NOT call this tool — write Cypher instead — when the sub-question:
  • Tells you the relations ("the R1 of the R2 of X") — use relation_chain or chained Cypher, which is more reliable.
  • Names only ONE entity — use neighbors.
  • Needs only the direct link — use relation_between.

Use this tool only for "how is X connected to Y" with both endpoints named and the relations unknown.
subject, object = the two entity names (name and alias match).""",

    "triples_with_relation": """(subject, relation, object) triples for ONE relation type across the graph, optionally restricted to a subject.

Returns up to 100 rows: subject, relation, object and passage_id. Without a subject the result is an arbitrary slice of that relation, not a ranking. It returns ONE hop only.

Do NOT call this tool — write Cypher instead — when the sub-question:
  • Wants every fact of a single named entity — use neighbors.
  • Is a reverse lookup of what points at a named entity — use entities_with_relation_to.
  • Is multi-hop ("the R1 of the R2 of X") — use relation_chain or chained Cypher.

Use this tool only when the question is anchored on a RELATION across many entities, or to browse every instance of one relation type.
relation = edge type, matched case- and underscore-insensitively as a substring. subject = optional entity name restricting the source side (name and alias match).""",

    "passages_for_entity": """The raw source passages that mention ONE named entity — the free text the graph was extracted from.

Returns up to 25 distinct passages: title and text. Read `text` to extract the answer; passages of different entities that share a name can both appear, so check the title. It does NOT return structured relations.

Do NOT call this tool — write Cypher instead — when the sub-question:
  • Is multi-hop — one entity's passages rarely contain facts about a second, unnamed entity; chain the relations instead.
  • Asks for a detail that IS a typed relation — match the edge (attribute_of_entity, relation_chain).

Use this tool only for a detail that is not stored as an edge (a description, a quote, a number or date not extracted as its own fact, a country that only appears in prose), to verify original wording, or as the fallback when attribute_of_entity or kin_of return no rows.
name = the entity (name and alias match, case-insensitive substring).""",
}
