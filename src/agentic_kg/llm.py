"""OpenAI-compatible chat client (Fireworks by default) with the request/response behaviour of the paper runs.

* One ``LLM`` per role (decision / query / answer / judge); the request keys come from
  ``Settings.request_params``. OpenAI-standard keys are passed as arguments, provider extras
  (``top_k``, ``prompt_cache_max_len``, ``thinking``) go into the body via ``extra_body``.
* Every call is a single system + user turn. ``n=1``, ``stop=None``, ``stream=False``, no ``tool_choice``.
* Retries: 3 attempts, exponential wait 4-10 s, on timeouts / connection errors / 429 / 5xx.
* ``content`` is ``(message.content or "").strip()``. With tools, only syntactically valid calls are returned
  (``arguments`` parsed with ``json.loads(strict=False)``, empty arguments -> ``{}``); invalid ones are listed
  separately, so ``tool_calls[0]`` is the first valid call.
"""

from __future__ import annotations

import json
import os
import time

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings

_RETRYABLE = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)

# request keys the OpenAI SDK accepts as arguments; everything else is sent through extra_body
_SDK_KEYS = {"temperature", "max_tokens", "seed", "user", "reasoning_effort"}
_CLIENT_KEYS = {"max_retries", "request_timeout"}


def _cached_tokens(raw_usage: dict) -> int | None:
    """Prompt-cache reuse as reported by the provider, or None."""
    if not isinstance(raw_usage, dict):
        return None
    details = raw_usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return details["cached_tokens"]
    return raw_usage.get("cached_tokens")


def _parse_tool_calls(raw_calls) -> tuple[list[dict], list[dict]]:
    valid, invalid = [], []
    for rc in raw_calls or []:
        fn = rc.get("function") or {}
        args = fn.get("arguments")
        try:
            parsed = {} if not args else json.loads(args, strict=False)
        except json.JSONDecodeError as exc:
            invalid.append({"name": fn.get("name"), "args": args, "id": rc.get("id"), "error": str(exc)})
            continue
        valid.append({"name": fn.get("name") or "", "args": parsed or {}, "id": rc.get("id")})
    return valid, invalid


class LLM:
    def __init__(self, role: str, model: str, params: dict, *, api_key: str, base_url: str):
        self.role = role
        self.model = model
        self.params = dict(params)
        self.max_retries = int(self.params.pop("max_retries", 2) or 0)
        timeout = self.params.pop("request_timeout", None)
        self._client = openai.OpenAI(api_key=api_key or "missing", base_url=base_url,
                                     timeout=timeout, max_retries=0)

    def _body(self, system: str, user: str, tools: list[dict] | None) -> dict:
        p = self.params
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "n": 1,
            "stop": None,
        }
        extra = {}
        for k, v in p.items():
            if k in _SDK_KEYS:
                body[k] = v
            elif k not in _CLIENT_KEYS:
                extra[k] = v
        if tools:
            body["tools"] = tools
        if extra:
            body["extra_body"] = extra
        return body

    def request_body(self, system: str, user: str, tools: list[dict] | None = None) -> dict:
        """The JSON object that goes on the wire (for tests)."""
        body = self._body(system, user, tools)
        extra = body.pop("extra_body", {})
        return {**body, **extra}

    def _create(self, body: dict):
        @retry(retry=retry_if_exception_type(_RETRYABLE), stop=stop_after_attempt(self.max_retries + 1),
               wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
        def _call():
            return self._client.chat.completions.create(**body)
        return _call()

    def chat(self, system: str, user: str, tools: list[dict] | None = None) -> dict:
        t0 = time.perf_counter()
        resp = self._create(self._body(system, user, tools))
        elapsed = round(time.perf_counter() - t0, 3)

        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        raw_calls = [tc.model_dump() for tc in (msg.tool_calls or [])]
        tool_calls, invalid_tool_calls = _parse_tool_calls(raw_calls)

        raw_usage = resp.usage.model_dump() if resp.usage else {}
        usage = {}
        if raw_usage:
            usage = {"input_tokens": raw_usage.get("prompt_tokens", 0),
                     "output_tokens": raw_usage.get("completion_tokens", 0),
                     "total_tokens": raw_usage.get("total_tokens", 0)}
        usage = usage or raw_usage
        return {
            "content": content,
            "model": self.model,
            "elapsed_sec": elapsed,
            "prompt_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
            "completion_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cached_tokens": _cached_tokens(raw_usage),
            "raw_token_usage": raw_usage,
            "system_chars": len(system),
            "user_chars": len(user),
            "response_chars": len(content),
            "tool_calls": tool_calls,
            "invalid_tool_calls": invalid_tool_calls,
        }


def build_llms(settings: Settings, repeat: int = 1) -> dict[str, LLM]:
    """One LLM per role for this repeat (seed = base_seed + repeat; the judge keeps base_seed)."""
    base_url = os.environ.get("LLM_BASE_URL") or settings.models["base_url"]
    api_key = os.environ.get(settings.models.get("api_key_env", "FIREWORKS_API_KEY"), "")
    return {r: LLM(r, settings.model_id(r), settings.request_params(r, repeat), api_key=api_key, base_url=base_url)
            for r in ("decision", "query", "answer", "judge")}
