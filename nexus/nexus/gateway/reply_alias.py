"""Whether a chain's first resend of a sampled reply may stand for it."""

from __future__ import annotations

import json
import os
import unicodedata

from dataclasses import dataclass
from typing import Any

from nexus.gateway.scaffold_policy import _canonical_arguments


REPLY_ALIAS_ENV = 'GATEWAY_REPLY_ALIAS'
REPLY_ALIAS_CONTENT_SIMILARITY_ENV = 'GATEWAY_REPLY_ALIAS_CONTENT_SIMILARITY'
LEDGER_VIEW_MAX_CHARS = 2000


@dataclass(frozen=True)
class ReplyAlias:
    enabled: bool = True
    content_similarity: float = 1.0

    @classmethod
    def from_env(cls) -> ReplyAlias:
        raw = os.environ.get(REPLY_ALIAS_CONTENT_SIMILARITY_ENV, '1')
        try:
            content_similarity = float(raw)
        except ValueError:
            content_similarity = float('nan')
        if not 0.0 <= content_similarity <= 1.0:
            raise ValueError(f'{REPLY_ALIAS_CONTENT_SIMILARITY_ENV} must be a number in [0, 1], got {raw!r}')
        return cls(enabled=os.environ.get(REPLY_ALIAS_ENV, '') != '0', content_similarity=content_similarity)

    def rejection(self, sampled: dict, resent: dict) -> str | None:
        # Arguments are not compared: scaffolds rewrite them, while the sample is what the model produced.
        sampled_calls, resent_calls = _call_names(sampled), _call_names(resent)
        if len(resent.get('tool_calls') or []) != len(sampled.get('tool_calls') or []):
            return 'tool_call_count_mismatch'
        if resent_calls.keys() != sampled_calls.keys():
            return 'tool_call_id_mismatch'
        if resent_calls != sampled_calls:
            return 'tool_name_mismatch'
        if content_similarity(sampled, resent) < self.content_similarity:
            return 'content_below_similarity'
        return None


def resend_difference(sampled: dict, resent: dict) -> str:
    a, b = _comparable_view(sampled), _comparable_view(resent)
    start = max(0, len(os.path.commonprefix([a, b])) - 24)
    lead = '…' if start else ''
    return f'resent {lead + b[start : start + 104]!r} where the sample has {lead + a[start : start + 104]!r}'


def ledger_view(message: dict) -> str:
    """One-line comparable form of a message for the reply alias ledger, capped at LEDGER_VIEW_MAX_CHARS."""
    view = _comparable_view(message)
    return view if len(view) <= LEDGER_VIEW_MAX_CHARS else view[: LEDGER_VIEW_MAX_CHARS - 1] + '…'


def content_similarity(sampled: dict, resent: dict) -> float:
    a, b = _normalized_text(sampled.get('content')), _normalized_text(resent.get('content'))
    if a == b:
        return 1.0
    if not (isinstance(a, str) and isinstance(b, str)):
        return 0.0
    prefix = len(os.path.commonprefix([a, b]))
    suffix = len(os.path.commonprefix([a[prefix:][::-1], b[prefix:][::-1]]))
    longest = max(len(a), len(b))
    return 1.0 - (longest - prefix - suffix) / longest


def _call_names(message: dict) -> dict[Any, Any]:
    return {call.get('id'): (call.get('function') or {}).get('name') for call in message.get('tool_calls') or []}


def _normalized_text(content: Any) -> Any:
    if content is None:
        return ''
    if isinstance(content, str):
        return ' '.join(unicodedata.normalize('NFC', content).split())
    return content


def _comparable_view(message: dict) -> str:
    parts = [f'content {_normalized_text(message.get("content"))!r}']
    for call in sorted(message.get('tool_calls') or [], key=lambda call: str(call.get('id'))):
        function = call.get('function') or {}
        arguments = json.dumps(_canonical_arguments(function.get('arguments')), ensure_ascii=False)
        parts.append(f'{call.get("id")} {function.get("name")}{arguments}')
    return ' | '.join(parts)
