"""Induce a per-benchmark ontology (entity types + relation vocabulary) used to guide OpenIE.

All benchmarks are treated identically: the relation vocabulary is left OPEN and discovered
during extraction (no gold/test-label peeking, so cross-benchmark comparisons stay fair). A
future Phase can cluster the observed relations into a closed vocab via an LLM pass; the
deterministic gold-triple KG (`build_kg --mode gold`) remains a separate validation-only path
and is never used here. The induced ontology is returned to openie.build_openie, which
persists the final schema.json after extraction.
"""

# A small, domain-general entity-type set for Wikipedia-style passages.
DEFAULT_ENTITY_TYPES = [
    "Person", "Location", "Organization", "Work", "Event", "Date", "Number", "Misc",
]


def induce(records: list[dict], benchmark: str) -> dict:
    """Return {entity_types, relations:[{type,name}]} to guide extraction.

    `relations` is a *preferred* vocabulary, not a hard constraint. It is intentionally left
    empty (open extraction) so every benchmark is built the same way without leaking gold
    relations; openie discovers relations freely and they become real edge types.
    """
    return {"entity_types": DEFAULT_ENTITY_TYPES, "relations": []}
