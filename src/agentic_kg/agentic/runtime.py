"""Shared handles every component receives: LLMs, Neo4j, database name, settings."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from ..llm import LLM
from ..neo4j_client import Neo4jClient


@dataclass
class Runtime:
    settings: Settings
    llms: dict[str, LLM]                  # decision / query / answer / judge
    neo4j: Neo4jClient | None
    database: str
    max_workers: int = 4
    schema: str = ""                      # full schema text (flat decomposer, query generator)
    decision_schema: str = ""             # short entity summary (chain decomposer)
    extras: dict = field(default_factory=dict)
