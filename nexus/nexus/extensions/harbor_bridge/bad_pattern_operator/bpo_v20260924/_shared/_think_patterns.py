"""Shared think-tag patterns for the FORMAT think_tag_* detectors.

Extracted from the former merged think_tag detectors so the parked channel-split
detectors (think_tag_content / __reasoning / __tool_call) and think_protocol_leak_hy
share one anchored regex set.

``ANY_TAG`` / ``OPEN_TAG`` / ``CLOSE_TAG`` / ``HASHED_TAG`` remain the original
hy3/taiji ``<think>`` family: the literal ``think`` keyword with an optional
``:hex`` / ``_hex`` suffix — deliberately NOT loose angle-bracket matching, so
code-heavy text with ``<-chan``, ``<= x``, ``<T>`` does NOT false-fire.

``PROTOCOL_THINK_TAG`` is the all-family lexicon used by
think_protocol_leak_all_family. It keeps bare ``<think>`` / ``</think>`` plus
the other chat-template think delimiters (Seed, LongCat, Kimi K3). Hashed
``<think:hex>`` / ``<think_hex>`` stay on think_protocol_leak_hy. Bare
``/think`` is not matched; only ``/think_on`` and ``/think_off``. Parked
channel-split think_tag_* keep ``ANY_TAG``.
"""
from __future__ import annotations

import re

OPEN_TAG = re.compile(r"<think(?:[:_][0-9a-fA-F]+)?>")
CLOSE_TAG = re.compile(r"</think(?:[:_][0-9a-fA-F]+)?>")
ANY_TAG = re.compile(r"</?think(?:[:_][0-9a-fA-F]+)?>")
HASHED_TAG = re.compile(r"</?think[:_][0-9a-fA-F]+>")

# Extended think delimiters. Kept as a separate variable so ANY_TAG stays the
# original <think> predicate (bare + hashed). PROTOCOL_THINK_TAG is bare
# <think> plus these; hashed :hex / _hex stays on HASHED_TAG.
_EXTENDED_THINK_TAG = (
    r"</?seed:think>"
    r"|</?longcat_think>"
    r"|(?:^|(?<=\s))/think_on\b"
    r"|(?:^|(?<=\s))/think_off\b"
    r"|<\|open\|>think<\|sep\|>"
    r"|<\|close\|>think<\|sep\|>"
)
PROTOCOL_THINK_TAG = re.compile(
    r"(?:</?think>|" + _EXTENDED_THINK_TAG + r")"
)


def is_plain_think_tag(token: str) -> bool:
    """True for unsuffixed ``<think>`` / ``</think>`` (not hashed, not extended)."""
    return bool(ANY_TAG.fullmatch(token) and not HASHED_TAG.fullmatch(token))


def format_seen(matches: list[str]) -> str:
    """Classify matched tokens as 'plain', 'hashed', 'protocol', or 'both'."""
    has_hashed = any(HASHED_TAG.fullmatch(t) for t in matches)
    has_plain = any(is_plain_think_tag(t) for t in matches)
    has_protocol = any(not ANY_TAG.fullmatch(t) for t in matches)
    n_kinds = int(has_plain) + int(has_hashed) + int(has_protocol)
    if n_kinds > 1:
        return "both"
    if has_hashed:
        return "hashed"
    if has_protocol:
        return "protocol"
    return "plain"


def snippet_around(text: str, start: int, end: int, ctx: int = 80) -> str:
    s = max(0, start - ctx)
    e = min(len(text), end + ctx)
    out = text[s:e].replace("\n", "\\n")
    return ("… " if s > 0 else "") + out + (" …" if e < len(text) else "")


__all__ = [
    "OPEN_TAG",
    "CLOSE_TAG",
    "ANY_TAG",
    "HASHED_TAG",
    "PROTOCOL_THINK_TAG",
    "is_plain_think_tag",
    "format_seen",
    "snippet_around",
]
