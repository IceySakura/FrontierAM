"""Shared token counting helpers for BrowserGym modules."""

from __future__ import annotations

import logging

from functools import lru_cache
from typing import Any


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cl100k_encoding() -> Any | None:
    """Lazily initialize the cl100k_base encoding if it is available."""
    try:
        import tiktoken
    except Exception as exc:
        logger.debug('[browser_gym] tiktoken unavailable, falling back to heuristic token counting: %s', exc)
        return None

    try:
        return tiktoken.get_encoding('cl100k_base')
    except Exception as exc:
        logger.warning(
            '[browser_gym] Failed to initialize cl100k_base, falling back to heuristic token counting: %s',
            exc,
        )
        return None


def count_tokens_with_fallback(text: str, chars_per_token: int) -> int:
    """Count tokens with tiktoken when available, otherwise use a heuristic."""
    encoding = get_cl100k_encoding()
    if encoding is not None:
        try:
            return len(encoding.encode(text))
        except Exception as exc:
            logger.warning(
                '[browser_gym] Failed to encode text with cl100k_base, falling back to heuristic token counting: %s',
                exc,
            )

    return len(text) // chars_per_token
