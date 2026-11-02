"""Token counting for evidence budgets (tiktoken ``o200k_base`` as a stand-in for the model tokenizer)."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return None


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, int, bool]:
    """Cut ``text`` to at most ``max_tokens`` tokens -> (text, original_token_count, truncated).

    max_tokens <= 0 disables the cut.
    """
    n = count_tokens(text)
    if max_tokens <= 0 or n <= max_tokens:
        return text, n, False
    enc = _encoder()
    if enc is None:  # chars/3 fallback, the same ratio count_tokens uses
        return text[: max_tokens * 3], n, True
    return enc.decode(enc.encode(text, disallowed_special=())[:max_tokens]), n, True


def count_tokens(text: str) -> int:
    """Approximate number of tokens in ``text``."""
    if not text:
        return 0
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // 3)
    return len(enc.encode(text, disallowed_special=()))
