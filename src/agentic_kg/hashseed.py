"""GraphRAG tie-breaking iterates Python sets of strings, so its retrieval depends on PYTHONHASHSEED.

The paper runs used PYTHONHASHSEED=0. ``ensure()`` re-executes the interpreter once with that value.
"""

import os
import sys

WANTED = "0"


def ensure() -> None:
    if os.environ.get("PYTHONHASHSEED") == WANTED or os.environ.get("AKG_NO_REEXEC") == "1":
        return
    env = dict(os.environ, PYTHONHASHSEED=WANTED)
    os.execve(sys.executable, [sys.executable, *sys.argv], env)
