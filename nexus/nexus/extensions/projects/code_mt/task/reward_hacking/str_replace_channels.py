"""code_mt-owned str_replace no-op detector, split by call CHANNEL.

Distinguishes:
  * ``tool_call``     — the edit arrived as a STRUCTURED tool-call object (the
                        assistant message's ``tool_calls`` entries, i.e. the
                        native tool_call protocol).
  * ``function_call`` — the edit was recovered from TEXT-PARSED content (XML
                        such as Hunyuan ``<arg_key>`` / Qwen ``<function=>``
                        embedded in the assistant text — what the text parser
                        reconstructs).

The SWE task's low-level format parsers are reused verbatim as the single source
of truth for the JSON/XML tool-call shapes (SWE is a reference; it is NOT
modified). Only the per-channel split — the code_mt-specific requirement — lives
here. Flag-only: this never changes the score.
"""

from __future__ import annotations

import logging

from typing import Any

from nexus.extensions.projects.code_mt.task.reward_hacking._vendored_swe.monitors import (
    _extract_named_str_replace_args,
    _extract_xml_str_replace_args,
    _has_old_eq_new,
)


logger = logging.getLogger(__name__)


def _empty_channel() -> dict[str, int]:
    return {'count': 0, 'old_eq_new_count': 0}


def detect_str_replace_noop_by_channel(messages: list[dict] | None) -> dict[str, Any]:
    """Count str_replace calls (and old==new no-ops) per channel.

    Returns the combined ``str_replace_count`` / ``old_eq_new_count`` (the keys
    the penalty code consumes, so the penalty math is unchanged) plus a
    ``tool_call`` / ``function_call`` breakdown.
    """
    tool_call = _empty_channel()
    function_call = _empty_channel()

    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get('role') != 'assistant':
            continue

        # --- tool_call channel: structured tool_calls entries ---
        for tc in msg.get('tool_calls') or []:
            if not isinstance(tc, dict):
                continue
            args = _extract_named_str_replace_args(tc)
            if args is None:
                fn = tc.get('function')  # OpenAI shape nests name/arguments here
                if isinstance(fn, dict):
                    args = _extract_named_str_replace_args(fn)
            if args is None:
                continue
            tool_call['count'] += 1
            if _has_old_eq_new(args):
                tool_call['old_eq_new_count'] += 1

        # --- function_call channel: XML tool calls embedded in the text ---
        content = msg.get('content')
        if isinstance(content, str) and content:
            for args in _extract_xml_str_replace_args(content):
                function_call['count'] += 1
                if _has_old_eq_new(args):
                    function_call['old_eq_new_count'] += 1

    total = tool_call['count'] + function_call['count']
    old_eq_new = tool_call['old_eq_new_count'] + function_call['old_eq_new_count']
    return {
        'str_replace_count': total,
        'old_eq_new_count': old_eq_new,
        'tool_call': tool_call,
        'function_call': function_call,
    }


__all__ = ['detect_str_replace_noop_by_channel']
