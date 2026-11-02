"""Shared CJK language detection used by the General rollout."""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)

Lang = str  # "en" | "cn"

# Fraction of CJK characters (among all alphanumeric/CJK characters) above
# which a question is considered Chinese. Punctuation / digits / whitespace
# are ignored so that a mostly-Chinese question with embedded English terms
# or numbers still routes to ``cn``.
_CJK_THRESHOLD = 0.15


def _is_cjk(ch: str) -> bool:
    # CJK Unified Ideographs + common extension / compatibility blocks.
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # Extension A
        or 0xF900 <= code <= 0xFAFF  # Compatibility Ideographs
        or 0x3000 <= code <= 0x303F  # CJK punctuation
        or 0xFF00 <= code <= 0xFFEF  # Fullwidth forms
    )


def detect_lang(text: str | None) -> Lang:
    """Return ``"cn"`` if *text* is predominantly Chinese, else ``"en"``.

    Heuristic: among "letter-like" characters (ASCII letters + CJK), if the
    CJK ratio exceeds :data:`_CJK_THRESHOLD`, classify as Chinese. Empty or
    non-letter input defaults to English.
    """
    if not text:
        return 'en'
    cjk = 0
    ascii_alpha = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        elif ch.isascii() and ch.isalpha():
            ascii_alpha += 1
    total = cjk + ascii_alpha
    if total == 0:
        return 'en'
    return 'cn' if (cjk / total) >= _CJK_THRESHOLD else 'en'


def resolve_lang(explicit: str | None, question: str | None) -> Lang:
    """Resolve the rollout language.

    Prefer an explicit, pre-tagged ``language`` (``"en"`` / ``"cn"``);
    otherwise infer it from the question text via :func:`detect_lang`.
    """
    if isinstance(explicit, str):
        norm = explicit.strip().lower()
        if norm in ('cn', 'zh', 'zh-cn', 'chinese', '中文'):
            return 'cn'
        if norm in ('en', 'eng', 'english'):
            return 'en'
    return detect_lang(question)
