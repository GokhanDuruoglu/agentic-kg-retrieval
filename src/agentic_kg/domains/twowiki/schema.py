"""Schema texts of the 2WikiMultihopQA graph that go into the prompts.

STATIC_SCHEMA   the hand-written catalogue of every edge type (grouped, with meaning, inverse pairs and
                "how to query" rules); the flat decomposer and the query generator receive it.
DECISION_SCHEMA a short entity summary for the labeled (chain) decomposer: what kinds of facts a
                sub-question may refer to, nothing about how to query.
"""

_STATIC_2WIKI_HEADER = """\
This is a knowledge graph extracted from Wikipedia-style passages. It has ONE entity label
(`Entity`, with a `type` property) and a `Passage` label that holds the raw source text.

Node properties:
  Entity
    id:      STRING    canonical entity id
    name:    STRING    canonical surface name (match case-insensitively)
    type:    STRING    one of: Person, Location, Organization, Work, Event, Date, Number, Misc
    aliases: STRING[]  OTHER surface forms for this entity. The exact string the question uses
                       may live HERE, not in `name` — always match both (see "How to query").
  Passage
    id:      STRING    passage id
    title:   STRING    source passage title
    text:    STRING    raw passage text — READ THIS to extract the final answer

Relationships (the relation name IS the edge type; each edge carries a `passage_id` provenance
property). Direction is subject-[:REL]->object; grouped by domain:"""

# (edge type, meaning + optional use case). Order = readability, NOT frequency.
_STATIC_2WIKI_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Kinship / family (see INVERSE PAIRS — match BOTH directions)", [
        ("FATHER",      "father of a person; (child)-[:FATHER]->(father). 'who is X's father'"),
        ("MOTHER",      "mother of a person; (child)-[:MOTHER]->(mother). 'X's mother'"),
        ("PARENT",      "parent, gender unstated; (child)-[:PARENT]->(parent)"),
        ("CHILD",       "child of a person; (parent)-[:CHILD]->(child). INVERSE of father/mother/parent"),
        ("SIBLING",     "brother/sister — symmetric, match undirected -[:SIBLING]-"),
        ("SPOUSE",      "husband/wife — symmetric, match undirected -[:SPOUSE]-"),
        ("GRANDPARENT", "grandparent; (grandchild)-[:GRANDPARENT]->(grandparent)"),
        ("GRANDCHILD",  "grandchild; (grandparent)-[:GRANDCHILD]->(grandchild). INVERSE of grandparent"),
        ("UNCLE",       "uncle of a person"),
    ]),
    ("Biographical (facts about a person)", [
        ("DATE_OF_BIRTH",     "when a person was born"),
        ("DATE_OF_DEATH",     "when a person died. 'when did X die'"),
        ("PLACE_OF_BIRTH",    "where a person was born"),
        ("PLACE_OF_DEATH",    "where a person died. 'where did X die'"),
        ("CAUSE_OF_DEATH",    "how a person died"),
        ("NATIONALITY",       "a person's country of citizenship"),
        ("OCCUPATION",        "a person's profession"),
        ("POSITION",          "office/role held (president, bishop, captain, …)"),
        ("EDUCATED_AT",       "school/university a person attended"),
        ("EMPLOYER",          "organization a person worked for"),
        ("MEMBER_OF",         "group/party/team/body a person or org belongs to"),
        ("AWARD",             "award won by a person or work"),
        ("NOMINATION",        "award a person or work was nominated for"),
        ("INFLUENCED_BY",     "who/what influenced the subject"),
        ("COLLABORATED_WITH", "people the subject worked with"),
        ("PARTICIPATED_IN",   "event the subject took part in"),
        ("DYNASTY",           "royal/noble house a person belongs to"),
        ("BIRTH_NAME",        "name given at birth"),
        ("FULL_NAME",         "full/formal name"),
        ("ALSO_KNOWN_AS",     "alternative name / alias"),
        ("ABBREVIATION",      "short form of a name"),
        ("LIFESPAN",          "birth–death span (text)"),
        ("TITLE",             "title/honorific of a person, or the title of a work"),
    ]),
    ("Works & film (books, films, songs, albums)", [
        ("DIRECTOR",           "director of a film/work. 'director of film X'"),
        ("WRITER",             "writer/screenwriter of a work"),
        ("PRODUCER",           "producer of a film/work"),
        ("COMPOSER",           "composer of a work's music"),
        ("CAST_MEMBER",        "actor appearing in a film/show"),
        ("ART_DIRECTOR",       "art director of a film"),
        ("PRODUCTION_COMPANY", "company that produced a work"),
        ("DISTRIBUTOR",        "company that distributed a film"),
        ("PUBLISHER",          "publisher of a book/work"),
        ("PUBLICATION_DATE",   "release/publication date of a work"),
        ("GENRE",              "genre of a work"),
        ("LANGUAGE",           "language of a work"),
        ("BASED_ON",           "source material a work is based on"),
        ("SEQUEL_TO",          "the earlier work this is a sequel to"),
        ("SUBJECT",            "topic/subject of a work"),
        ("SETTING",            "setting/locale of a work"),
        ("CHART_POSITION",     "chart ranking of a song/album"),
        ("RECORDED",           "recording date/place of a song"),
        ("CRITICAL_RECEPTION", "critical-reception note for a work"),
        ("NAMED_AFTER",        "what an entity was named after"),
        ("COUNTRY_OF_ORIGIN",  "country a work/thing originates from"),
    ]),
    ("Organization & geography", [
        ("LOCATION", "where an entity is located"),
        ("PART_OF",  "larger entity/whole this belongs to"),
        ("FOUNDED",  "organization/place a person founded (or founding fact)"),
        ("OWNER",    "owner of an entity"),
        ("LEAGUE",   "sports league a team plays in"),
    ]),
    ("Temporal, ordering & misc", [
        ("START_DATE",    "start date of a term/event/relationship"),
        ("END_DATE",      "end date of a term/event/relationship"),
        ("MARRIAGE_DATE", "date a marriage took place"),
        ("DATE_ADDED",    "date an entity was added/registered"),
        ("PREDECESSOR",   "who/what came immediately before (role/series). INVERSE of successor"),
        ("SUCCESSOR",     "who/what came immediately after (role/series). INVERSE of predecessor"),
        ("VALUE",         "a numeric / measured value"),
        ("RELATED_TO",    "generic catch-all — OFTEN holds attributes (country, year, place, …) not captured by a specific edge; check it whenever a typed attribute edge is empty"),
    ]),
]

_STATIC_2WIKI_FOOTER = """\
  (:Entity)-[:MENTIONED_IN]->(:Passage)        # links an entity to passages that mention it

INVERSE PAIRS — the SAME fact may be recorded from either side under a DIFFERENT edge type,
so match the forward edge AND its inverse (undirected -[:REL]- is NOT enough — it keeps the
same type). Combine with UNION or a multi-type match:
    FATHER / MOTHER / PARENT   <->  CHILD
        X's parent:  MATCH (x)-[:FATHER|MOTHER|PARENT]->(p)   UNION   MATCH (p)-[:CHILD]->(x)
        Also add a spouse hop — a parent may only be recorded as the OTHER parent's spouse
        (X's mother = X's father's SPOUSE). Include this branch too, e.g.
            MATCH (x)-[:FATHER|MOTHER|PARENT]-(q:Entity)-[:SPOUSE]-(p:Entity)
        so a missing direct MOTHER/FATHER/CHILD edge still resolves the parent.
    GRANDPARENT                <->  GRANDCHILD
        X's grandparent:  (x)-[:GRANDPARENT]->(g)  OR  (g)-[:GRANDCHILD]->(x)  OR  2-hop parent-of-parent
    PREDECESSOR                <->  SUCCESSOR
    SPOUSE, SIBLING             =   symmetric — match undirected -[:SPOUSE]- / -[:SIBLING]-

How to query (CRITICAL):
  - Match entity names case-insensitively, as substrings, AND against aliases — NEVER `.name`
    alone (the question's surface form may live in `.aliases`):
      WHERE toLower(e.name) CONTAINS toLower($name)
         OR ANY(a IN e.aliases WHERE toLower(a) CONTAINS toLower($name))
    Use the question's phrase as ONE value; do NOT invent alternate spellings/transliterations
    (e.g. John/Ioann) — `aliases` already holds those variants.
  - Use the relation name directly as the edge type: MATCH (a:Entity)-[:DIRECTOR]-(b:Entity).
    Treat non-kinship edges as undirected (-[:REL]-) unless the question implies a direction;
    for kinship/ordered edges also match the inverse (see INVERSE PAIRS).
  - Multi-hop "the R1 of the R2 of X": chain BOTH edges in ONE query, e.g.
      MATCH (x:Entity)-[:DIRECTOR]-(mid:Entity)-[:PLACE_OF_DEATH]-(ans:Entity)
      WHERE toLower(x.name) CONTAINS toLower($name)
         OR ANY(a IN x.aliases WHERE toLower(a) CONTAINS toLower($name))
      RETURN mid.name AS via, ans.name AS answer
  - If unsure which relation applies: MATCH (a:Entity)-[r]-(b:Entity) RETURN type(r), b.name.
  - A wanted attribute may have NO typed edge — for an attribute / comparison lookup, ALSO
    match the generic `-[:RELATED_TO]-` edge AND return the passage text so the value can be
    read off the raw text even when the specific edge is empty:
      MATCH (e:Entity)-[:MENTIONED_IN]->(p:Passage) RETURN p.text"""


def _render_static_2wiki() -> str:
    """Assemble the static 2wiki schema string (header + grouped edges + footer)."""
    all_types = [t for _, edges in _STATIC_2WIKI_GROUPS for t, _ in edges]
    width = max(len(t) for t in all_types)
    parts = [_STATIC_2WIKI_HEADER, ""]
    for title, edges in _STATIC_2WIKI_GROUPS:
        parts.append(f"  # {title}")
        for rel_type, meaning in edges:
            edge = f"  (:Entity)-[:{rel_type}]->(:Entity)"
            parts.append(edge.ljust(width + 27) + f"# {meaning}")
        parts.append("")
    parts.append(_STATIC_2WIKI_FOOTER)
    return "\n".join(parts)


STATIC_SCHEMA = _render_static_2wiki()


DECISION_SCHEMA = """
A knowledge graph extracted from Wikipedia passages (the 2WikiMultihopQA domain).

Entities (all stored as `Entity`, with a type):
  Person, Location, Organization, Work (film, book, song, album), Event, Date, Number, Award, Misc.
Facts are relations between two entities:
  kinship        — father, mother, parent, child, spouse, sibling, grandparent, grandchild, uncle
  biography      — date / place of birth, date / place of death, cause of death, nationality,
                   occupation, position, education, employer, member of, award
  works          — director, writer, producer, composer, cast member, publication date,
                   country of origin, language, genre, based on, sequel to
  organizations  — location, part of, founded, owner
  generic        — a catch-all related-to edge that often holds attributes such as country or year
Each fact comes from a source Passage whose raw text is also stored.
"""
