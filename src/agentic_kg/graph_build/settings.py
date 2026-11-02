"""Settings of the graph build: where intermediate files live and which LLM runs which stage."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ..config import CONFIGS, ROOT, load_yaml
from ..llm import LLM

# Intermediate build outputs (one folder per build key: "2wiki" = raw OpenIE build, "2wiki-judged" = after the
# triple audit). Shipped, final data lives in data/2wiki/ and is not touched unless you ask for it.
DATA_DIR = Path(os.environ.get("AKG_BUILD_DIR") or ROOT / "build")


@lru_cache(maxsize=1)
def cfg() -> dict:
    return load_yaml(CONFIGS / "graph_build.yaml")


@lru_cache(maxsize=None)
def llm(stage: str, model: str | None = None) -> LLM:
    """The LLM of a build stage (openie | triple_judge | consolidate)."""
    models = load_yaml(CONFIGS / "models.yaml")
    sec = cfg()[stage]
    name = model or sec["model"]
    api_key = os.environ.get(models.get("api_key_env", "FIREWORKS_API_KEY"), "")
    base_url = os.environ.get("LLM_BASE_URL") or models["base_url"]
    return LLM(stage, models.get("model_prefix", "") + name, dict(sec["params"]), api_key=api_key, base_url=base_url)

