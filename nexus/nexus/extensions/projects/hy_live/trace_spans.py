"""Trace marks that let a tracker draw one hy-live episode as its layers.

Every model call already says who made it (``request_meta.invocation_kind``), but not what its
tool calls did nor which request a background agent is serving. These marks add exactly that,
without opening spans that background work would inherit as an ended parent:

- ``record_tool_call`` writes one finished ``hy_live_tool`` span per executed tool call, after
  the tool returned, so nothing it spawned runs under it.
- ``midbrain_request_attributes`` and ``brain_turn_attributes`` are for
  ``trace_attributes_context``: every span of one midbrain request, or of one CodeBuddy turn,
  says which request or operation it serves.
"""

from __future__ import annotations

import json
import time
import uuid

from typing import Any

from nexus.trace import start_span


CEREBELLUM_INVOCATION = 'hy_live_cerebellum'
MIDBRAIN_INVOCATION = 'hy_live_midbrain'
USER_SIMULATOR_INVOCATION = 'hy_live_user_simulator'
"""``request_meta.invocation_kind`` of the cerebellum, midbrain and LLM user simulator calls."""

TOOL_SPAN_NAME = 'hy_live_tool'
LAYER_ATTRIBUTE = 'hy_live.layer'
CEREBELLUM_LAYER = 'cerebellum'
MIDBRAIN_LAYER = 'midbrain'
BRAIN_LAYER = 'brain'
REQUEST_ATTRIBUTE = 'hy_live.request_id'
REQUEST_SOURCE_ATTRIBUTE = 'hy_live.request_source'
REQUEST_TEXT_ATTRIBUTE = 'hy_live.request_text'
REQUEST_PATH_ATTRIBUTE = 'hy_live.request_path'
OPERATION_ATTRIBUTE = 'hy_live.operation_id'
PROMPT_ATTRIBUTE = 'hy_live.prompt'
_TEXT_CHARS = 2000


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def record_tool_call(
    *,
    layer: str,
    call: dict[str, Any],
    result: str,
    started_at: float,
    request_id: str | None = None,
) -> None:
    """Mark one executed tool call of ``layer`` with its arguments and result.

    Args:
        layer: ``CEREBELLUM_LAYER`` or ``MIDBRAIN_LAYER``.
        call: The tool call as the model emitted it.
        result: The tool result the model reads next.
        started_at: ``time.time()`` when the tool started.
        request_id: The midbrain request a cerebellum call submitted; None keeps the request the
            surrounding ``midbrain_request_attributes`` names.
    """
    function = call.get('function') or {}
    arguments = function.get('arguments')
    attributes: dict[str, Any] = {
        LAYER_ATTRIBUTE: layer,
        'hy_live.tool_call_id': str(call.get('id') or ''),
        'hy_live.tool_name': str(function.get('name') or ''),
        'hy_live.arguments': arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
        'hy_live.result': result,
        'hy_live.started_at': started_at,
        'hy_live.ended_at': time.time(),
    }
    if request_id is not None:
        attributes[REQUEST_ATTRIBUTE] = request_id
    start_span(TOOL_SPAN_NAME, attributes).finish()


def midbrain_request_attributes(*, request_id: str, source: str, text: str = '', path: str = '') -> dict[str, Any]:
    """Attributes for ``trace_attributes_context`` around one midbrain request."""
    return {
        LAYER_ATTRIBUTE: MIDBRAIN_LAYER,
        REQUEST_ATTRIBUTE: request_id,
        REQUEST_SOURCE_ATTRIBUTE: source,
        REQUEST_TEXT_ATTRIBUTE: text[:_TEXT_CHARS],
        REQUEST_PATH_ATTRIBUTE: path,
    }


def brain_turn_attributes(*, operation_id: str, session_id: str, prompt: str) -> dict[str, Any]:
    """Attributes for ``trace_attributes_context`` around one CodeBuddy turn."""
    return {
        LAYER_ATTRIBUTE: BRAIN_LAYER,
        OPERATION_ATTRIBUTE: operation_id,
        'hy_live.session_id': session_id,
        PROMPT_ATTRIBUTE: prompt[:_TEXT_CHARS],
    }


__all__ = [
    'BRAIN_LAYER',
    'CEREBELLUM_INVOCATION',
    'CEREBELLUM_LAYER',
    'LAYER_ATTRIBUTE',
    'MIDBRAIN_INVOCATION',
    'MIDBRAIN_LAYER',
    'USER_SIMULATOR_INVOCATION',
    'OPERATION_ATTRIBUTE',
    'PROMPT_ATTRIBUTE',
    'REQUEST_ATTRIBUTE',
    'REQUEST_PATH_ATTRIBUTE',
    'REQUEST_SOURCE_ATTRIBUTE',
    'REQUEST_TEXT_ATTRIBUTE',
    'TOOL_SPAN_NAME',
    'brain_turn_attributes',
    'midbrain_request_attributes',
    'new_request_id',
    'record_tool_call',
]
