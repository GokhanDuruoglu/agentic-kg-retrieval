"""Name -> implementation registry for pipeline components.

    @register("validator", "retry")
    class RetryValidator: ...

    cls = get("validator", "retry")

A plain dict keyed by (kind, name). Anything imported before the pipeline is built can register itself, so a
new component is one decorated class in your own module plus ``--plugin your_pkg.your_module`` on the CLI.
"""

from __future__ import annotations

import importlib

_REGISTRY: dict[tuple[str, str], object] = {}


def register(kind: str, name: str):
    def deco(obj):
        key = (kind, name)
        if key in _REGISTRY and _REGISTRY[key] is not obj:
            raise ValueError(f"{kind} '{name}' is already registered")
        _REGISTRY[key] = obj
        return obj
    return deco


def get(kind: str, name: str):
    try:
        return _REGISTRY[(kind, name)]
    except KeyError:
        known = ", ".join(sorted(n for k, n in _REGISTRY if k == kind)) or "none"
        raise KeyError(f"unknown {kind} '{name}' (registered: {known})") from None


def names(kind: str) -> list[str]:
    return sorted(n for k, n in _REGISTRY if k == kind)


def load_plugins(modules: list[str] | None) -> None:
    """Import user modules whose top-level code registers extra components."""
    for m in modules or []:
        importlib.import_module(m)
