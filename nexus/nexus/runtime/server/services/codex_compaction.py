"""Shared, version-neutral detection for Codex local compaction requests."""

from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Any, Literal


DEFAULT_CODEX_COMPACTION_PROMPT_PREFIX = 'You are performing a CONTEXT CHECKPOINT COMPACTION.'
CODEX_TURN_METADATA_HEADER = 'x-codex-turn-metadata'


@dataclass(frozen=True)
class CodexCompactionEvidence:
    """Evidence that a Codex request is a local summary inference."""

    detector: Literal[
        'turn_metadata',
        'client_metadata',
        'configured_prompt',
        'default_prompt_prefix',
    ]


def _metadata_request_kind(metadata: Any) -> str | None:
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, ValueError):
            return None
    if not isinstance(metadata, dict):
        return None
    request_kind = metadata.get('request_kind')
    return request_kind if isinstance(request_kind, str) else None


def final_user_message_text(
    input_items: Any,
    *,
    require_responses_message_type: bool,
) -> str | None:
    """Extract text from the final user item in Responses or rendered Chat form."""
    if not isinstance(input_items, (list, tuple)) or not input_items:
        return None

    final_item = input_items[-1]
    if not isinstance(final_item, dict) or final_item.get('role') != 'user':
        return None
    if require_responses_message_type and final_item.get('type') != 'message':
        return None

    content = final_item.get('content')
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get('text') or block.get('input_text')
            if isinstance(text, str):
                parts.append(text)
    return ''.join(parts) if parts else None


def classify_codex_compaction(
    request_body: dict[str, Any],
    compact_prompt: str | None = None,
    *,
    turn_metadata: Any = None,
    require_responses_message_type: bool = True,
) -> CodexCompactionEvidence | None:
    """Classify native metadata first, then a final-user prompt fallback."""
    turn_request_kind = _metadata_request_kind(turn_metadata)
    if turn_request_kind is not None:
        return CodexCompactionEvidence('turn_metadata') if turn_request_kind == 'compaction' else None

    client_metadata = request_body.get('client_metadata')
    if isinstance(client_metadata, dict):
        client_turn_metadata = client_metadata.get(CODEX_TURN_METADATA_HEADER)
        client_request_kind = _metadata_request_kind(client_turn_metadata)
        if client_request_kind is not None:
            return CodexCompactionEvidence('client_metadata') if client_request_kind == 'compaction' else None

    text = final_user_message_text(
        request_body.get('input'),
        require_responses_message_type=require_responses_message_type,
    )
    if text is None:
        return None

    normalized = text.strip()
    if compact_prompt is not None:
        return CodexCompactionEvidence('configured_prompt') if normalized == compact_prompt.strip() else None
    if normalized.startswith(DEFAULT_CODEX_COMPACTION_PROMPT_PREFIX):
        return CodexCompactionEvidence('default_prompt_prefix')
    return None


__all__ = [
    'CODEX_TURN_METADATA_HEADER',
    'DEFAULT_CODEX_COMPACTION_PROMPT_PREFIX',
    'CodexCompactionEvidence',
    'classify_codex_compaction',
    'final_user_message_text',
]
