"""Baseline-eval monitors for code_mt (FLAG-ONLY).

Scaffold-agnostic quality signals used for BASELINE evaluation of a rollout.
Per req.md (0725) baseline eval is a design deliverable — these monitors only
FLAG, they never change the reward. All functions fail open to a neutral flag
dict on any error, so enabling them can never break a rollout.

  L1 empty_final_answer    — no non-empty final assistant text was produced.
  L2 noop_edit             — str_replace calls whose old==new (no-op edits).
  L3 trailing_tool_failure — the run's LAST tool observation is an error.
  L4 submit_loop           — more than one submit/finish tool call in the run.

Detection is protocol-tolerant: it reads OpenAI-chat messages (tool_calls list)
which the code_mt loops stash under ctx.tool_state['_code_mt_messages'].
"""

from __future__ import annotations

import logging

from typing import Any

from nexus.extensions.projects.code_mt.task.reward_hacking._vendored_swe.monitors import detect_str_replace_old_eq_new


logger = logging.getLogger(__name__)

_SUBMIT_TOOL_NAMES = {'submit', 'finish', 'submit_answer', 'complete', 'done'}
_ERROR_MARKERS = ('error', 'traceback', 'exception', 'command not found', 'no such file', 'exit code 1', 'failed')


def _msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                t = c.get('text') or c.get('content')
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(c, str):
                parts.append(c)
        return '\n'.join(parts)
    return ''


def detect_empty_final_answer(messages: list[dict] | None, answer: str = '') -> dict[str, Any]:
    """L1 — flag when the run produced no non-empty final assistant text."""
    final = answer or ''
    if not final:
        for m in reversed(messages or []):
            if isinstance(m, dict) and m.get('role') == 'assistant':
                final = _msg_text(m.get('content'))
                if final.strip():
                    break
    return {'is_empty_final_answer': not bool(final.strip())}


def detect_noop_edit(messages: list[dict] | None) -> dict[str, Any]:
    """L2 — flag no-op str_replace edits (old==new). Reuses the swe detector."""
    det = detect_str_replace_old_eq_new({'messages': messages or []})
    noop = int(det.get('old_eq_new_count', 0) or 0)
    total = int(det.get('str_replace_count', 0) or 0)
    ratio = (noop / total) if total > 0 else 0.0
    return {
        'noop_edit_count': noop,
        'str_replace_count': total,
        'noop_edit_ratio': ratio,
        'is_noop_edit': noop > 0,
    }


def detect_trailing_tool_failure(messages: list[dict] | None) -> dict[str, Any]:
    """L3 — flag when the LAST tool observation looks like an error."""
    last_tool_text = ''
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get('role') == 'tool':
            last_tool_text = _msg_text(m.get('content')).lower()
            break
    hit = bool(last_tool_text) and any(marker in last_tool_text for marker in _ERROR_MARKERS)
    return {'is_trailing_tool_failure': hit}


def detect_submit_loop(messages: list[dict] | None) -> dict[str, Any]:
    """L4 — flag more than one submit/finish tool call across the run."""
    submit_calls = 0
    for m in messages or []:
        if not isinstance(m, dict) or m.get('role') != 'assistant':
            continue
        for tc in m.get('tool_calls') or []:
            fn = tc.get('function', {}) if isinstance(tc, dict) else {}
            name = (fn.get('name', '') if isinstance(fn, dict) else '') or ''
            if name.rsplit('/', 1)[-1].strip().lower() in _SUBMIT_TOOL_NAMES:
                submit_calls += 1
    return {'submit_call_count': submit_calls, 'is_submit_loop': submit_calls > 1}


def run_baseline_monitors(
    messages: list[dict] | None,
    answer: str = '',
    struct_messages: list[dict] | None = None,
) -> dict[str, Any]:
    """Run all baseline (flag-only) monitors; fail open per-monitor.

    ``messages`` is the raw trajectory (assistant NL text) — the text-reading
    monitors (empty_final_answer, noop_edit) read it. ``struct_messages`` is the
    tool_call-shaped view (native tool_calls / role='tool'); the structured-only
    monitors (trailing_tool_failure, submit_loop) read it so they also fire on
    text_parse trajectories. Defaults to ``messages`` for backward compatibility.
    """
    struct = struct_messages if struct_messages is not None else messages
    out: dict[str, Any] = {}
    baseline: dict[str, Any] = {}
    for name, fn in (
        ('empty_final_answer', lambda: detect_empty_final_answer(messages, answer)),
        ('noop_edit', lambda: detect_noop_edit(messages)),
        ('trailing_tool_failure', lambda: detect_trailing_tool_failure(struct)),
        ('submit_loop', lambda: detect_submit_loop(struct)),
    ):
        try:
            baseline[name] = fn()
        except Exception:
            logger.exception('baseline monitor %s failed', name)
            baseline[name] = {}
    out['baseline'] = baseline
    return out


__all__ = [
    'detect_empty_final_answer',
    'detect_noop_edit',
    'detect_trailing_tool_failure',
    'detect_submit_loop',
    'run_baseline_monitors',
]
