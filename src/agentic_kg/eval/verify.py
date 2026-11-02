"""``akg verify``: offline fidelity checks against the golden files in tests/golden (no LLM, no Neo4j).

The goldens were produced from the code and run logs of the paper experiments: prompt texts, tool schemas and
tool Cypher, the decomposition of 120 logged model replies, evidence formatting / token counts, the request body
of each LLM role, and the shipped graph's counts. With a Neo4j server configured (NEO4J_PASSWORD) it also
compares the live graph's fingerprint with the frozen one.
"""

from __future__ import annotations

import sys

from ..config import ROOT

OFFLINE = ["test_prompts.py", "test_tools.py", "test_decomposer.py", "test_evidence.py",
           "test_llm_request.py", "test_graph.py"]


def main() -> int:
    tests = ROOT / "tests"
    if not tests.exists():
        print("tests/ not found: run `akg verify` from a source checkout.")
        return 2
    import pytest
    return int(pytest.main(["-q", "-p", "no:cacheprovider", *[str(tests / t) for t in OFFLINE]]))


if __name__ == "__main__":
    sys.exit(main())
