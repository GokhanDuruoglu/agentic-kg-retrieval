"""Configuration: repo paths, .env, models/params YAML, method configs with ``extends``.

A method config is a YAML file under configs/methods/ (or configs/methods/ablations/). ``extends: <name>``
starts from another method config; mappings are merged recursively and a ``null`` value removes the key.
That is how the ladder is expressed: every rung changes one component of the rung before it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(os.environ.get("AKG_HOME") or Path(__file__).resolve().parents[2])
load_dotenv(ROOT / ".env")

CONFIGS = ROOT / "configs"
ROLES = ("decision", "query", "answer", "judge")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def deep_merge(base: dict, over: dict) -> dict:
    """``over`` on top of ``base``; nested dicts merge, ``None`` deletes, anything else replaces.

    A component section whose ``type`` changes is replaced as a whole (the parameters of the old type do not
    apply to the new one).
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif (isinstance(v, dict) and isinstance(out.get(k), dict)
              and not ("type" in v and v["type"] != out[k].get("type"))):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _method_file(name: str) -> Path:
    for d in (CONFIGS / "methods", CONFIGS / "methods" / "ablations"):
        p = d / f"{name}.yaml"
        if p.exists():
            return p
    raise FileNotFoundError(f"no method config '{name}' under configs/methods/")


def list_methods() -> dict[str, list[str]]:
    main = sorted(p.stem for p in (CONFIGS / "methods").glob("*.yaml"))
    abl = sorted(p.stem for p in (CONFIGS / "methods" / "ablations").glob("*.yaml"))
    return {"methods": main, "ablations": abl}


def load_method(name: str, _seen: tuple = ()) -> dict:
    """Resolved method config (``extends`` chain applied). ``name`` is stored under ``name``."""
    if name in _seen:
        raise ValueError(f"circular extends: {' -> '.join(_seen + (name,))}")
    raw = load_yaml(_method_file(name))
    parent = raw.pop("extends", None)
    cfg = deep_merge(load_method(parent, _seen + (name,)), raw) if parent else raw
    cfg["name"] = name
    return cfg


@dataclass
class Settings:
    models: dict
    params: dict

    @property
    def base_seed(self) -> int:
        return int(self.params["base_seed"])

    def seed_for(self, role: str, repeat: int) -> int:
        return self.base_seed if role == "judge" else self.base_seed + int(repeat)

    def model_id(self, role: str) -> str:
        return self.models.get("model_prefix", "") + self.models["roles"][role]["model"]

    def request_params(self, role: str, repeat: int) -> dict:
        """params.yaml section + the role's extra_kwargs + the seed -> the keys of one request."""
        p = copy.deepcopy(self.params[f"{role}_llm"])
        p.update(copy.deepcopy(self.models["roles"][role].get("extra_kwargs") or {}))
        p["seed"] = self.seed_for(role, repeat)
        return p

    def check(self) -> None:
        roles = self.models["roles"]
        for r in ("decision", "query", "answer"):
            if roles[r]["model"] == roles["judge"]["model"]:
                raise SystemExit(f"config error: the judge model equals the {r} model; it must be independent")

    def price(self, model: str) -> dict | None:
        short = model.split("/")[-1]
        return (self.models.get("pricing") or {}).get(short)


def load_settings() -> Settings:
    s = Settings(models=load_yaml(CONFIGS / "models.yaml"), params=load_yaml(CONFIGS / "params.yaml"))
    s.check()
    return s


def load_dataset(name: str = "2wiki") -> dict:
    d = load_yaml(CONFIGS / "datasets" / f"{name}.yaml")
    d["neo4j_database"] = os.environ.get("NEO4J_DATABASE") or d["neo4j_database"]
    for k in ("questions", "gold", "corpus", "kg", "fingerprint"):
        d[k] = ROOT / d[k]
    return d


def sha_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
