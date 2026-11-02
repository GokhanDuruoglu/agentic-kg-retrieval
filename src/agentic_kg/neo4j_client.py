"""Read-only Neo4j access for LLM-written Cypher.

Two layers keep the shared graph safe: ``read_only_violation`` rejects write clauses and non-metadata
procedure calls before anything is sent (the message is fed back to the LLM), and every session is opened
with ``READ_ACCESS``, which the server enforces. Queries carry a server-side timeout.
"""

from __future__ import annotations

import atexit
import os
import re
import time

from neo4j import READ_ACCESS, GraphDatabase, Query

_HAS_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+|\$\w+)", re.IGNORECASE)

_STRIP_RE = re.compile(r"""//[^\n]*|/\*.*?\*/|'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`[^`]*`""", re.S)
_WRITE_RE = re.compile(r"(?<![\w.:$])(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b", re.I)
_CALL_RE = re.compile(r"(?<![\w.])CALL\s+([A-Za-z_][\w.]*)", re.I)        # `CALL { subquery }` has no name -> allowed
_SAFE_CALL_RE = re.compile(r"^(db\.(labels|relationshipTypes|propertyKeys|schema\.\w+|indexes|constraints)"
                           r"|dbms\.components)$", re.I)


def read_only_violation(cypher: str) -> str | None:
    """None if ``cypher`` is a plain read query, else an error message that is fed back to the LLM."""
    stripped = _STRIP_RE.sub(" ", cypher or "")          # ignore keywords inside strings, comments, `identifiers`
    m = _WRITE_RE.search(stripped)
    if m:
        what = re.sub(r"\s+", " ", m.group(1)).upper()
    else:
        c = next((x for x in _CALL_RE.findall(stripped) if not _SAFE_CALL_RE.match(x)), None)
        if c is None:
            return None
        what = f"CALL {c}"
    return (f"READ-ONLY VIOLATION: `{what}` is not allowed. This database is read-only and shared: only read queries "
            f"(MATCH, OPTIONAL MATCH, WHERE, WITH, UNWIND, RETURN, ORDER BY, LIMIT) may be run. Do not create, merge, set, "
            f"delete or remove nodes, relationships or properties, and do not store intermediate results (no scratch "
            f"properties, no procedure calls). Compute the answer inside one read query instead.")


class Neo4jClient:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None,
                 *, query_timeout_sec: float = 120, max_rows_per_query: int = 0):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password if password is not None else os.getenv("NEO4J_PASSWORD", "")
        self.timeout = query_timeout_sec
        self.max_rows = max_rows_per_query
        self._driver = None
        atexit.register(self.close)

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None

    def _execute(self, cypher: str, database: str, params: dict | None) -> dict:
        t0 = time.perf_counter()
        error: str | None = None
        rows: list[dict] = []

        violation = read_only_violation(cypher)
        if violation:        # never sent to the server
            return {"rows": [], "row_count": 0, "error": violation, "elapsed_sec": 0.0, "database": database,
                    "cypher": cypher, "params": params or {}, "note": None, "blocked_write": True}
        try:
            with self.driver.session(database=database, default_access_mode=READ_ACCESS) as session:
                rows = session.run(Query(cypher, timeout=self.timeout), params or {}).data()
        except Exception as exc:
            error = str(exc)
        return {"rows": rows, "row_count": len(rows), "error": error,
                "elapsed_sec": round(time.perf_counter() - t0, 3), "database": database,
                "cypher": cypher, "params": params or {}, "note": None}

    def run_cypher(self, cypher: str, database: str, params: dict | None = None) -> dict:
        """Execute ``cypher``; if a row cap is configured and the query has no LIMIT, re-run with one."""
        result = self._execute(cypher, database, params)
        if (self.max_rows > 0 and result["error"] is None and result["row_count"] > self.max_rows
                and not _HAS_LIMIT_RE.search(cypher)):
            original = result["row_count"]
            limited = self._execute(cypher.rstrip().rstrip(";") + f"\nLIMIT {self.max_rows}", database, params)
            if limited["error"] is None:
                limited["original_row_count"] = original
                limited["limit_applied"] = self.max_rows
                limited["note"] = (f"Result truncated: original query returned {original} rows; "
                                   f"re-executed with LIMIT {self.max_rows} to keep the answer concise.")
                return limited
        return result
