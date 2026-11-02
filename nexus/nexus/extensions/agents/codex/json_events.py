"""Parse the Codex ``exec --json`` event stream and rebuild HyTuner trace data.

Codex v0.146.1 ``exec --json`` prints one JSON
object per line to **stdout** — a curated "thread event" schema (NOT the raw
internal protocol). Confirmed against the codex ``rust-v0.146.1`` source
(``codex-rs/exec/src/exec_events.rs`` / ``event_processor_with_jsonl_output.rs``):

Top-level events (tagged by ``type``):
  * ``thread.started``   -> ``{thread_id}``
  * ``turn.started``     -> ``{}``
  * ``turn.completed``   -> ``{usage: {input_tokens, cached_input_tokens, output_tokens}}``
  * ``turn.failed``      -> ``{error: {message}}``
  * ``item.started`` / ``item.updated`` / ``item.completed`` -> ``{item: {...}}``
  * ``error``            -> ``{message}``

Item objects carry ``id`` (``"item_0"``, ...), an ``item_type`` discriminator
and a payload. Relevant ``item_type`` values:
  * ``command_execution`` -> ``{command: str (shell-joined), aggregated_output, exit_code|null, status}``
  * ``file_change``       -> ``{changes: [{path, kind}], status}``  (apply_patch surfaces here)
  * ``mcp_tool_call``     -> ``{...}``
  * ``collab_tool_call``  -> ``{tool, sender_thread_id, receiver_thread_ids, prompt, status}``
  * ``agent_message`` / ``reasoning`` / ``web_search`` / ``todo_list`` / ``error``

IMPORTANT — physical limits of this stream (do not fabricate around them):
  * Events carry **no timestamps and no per-item durations**.
  * There is **no compaction event** in the ``--json`` stream at all.
So accurate per-tool wall-clock timing and ``agent_compaction`` spans are
impossible from this source. The module rebuilds fine-grained termination and
ordered tool observations. Timeline segment timestamps are synthetic (evenly
spread across the real host-side wall-clock window) and are ordered-but-approximate
by construction — they are for visualization only and never feed reward/advantage.
"""

from __future__ import annotations

import json
import logging

from typing import Any

from nexus.core.openai_client import CONTEXT_LENGTH_EXCEEDED_CODE


logger = logging.getLogger(__name__)


# item_type values that represent a tool observation worth a timeline segment.
# agent_message / reasoning are model output (already covered by host-side
# openai_completion spans) and are intentionally excluded.
_OBSERVATION_ITEM_TYPES = frozenset({
    'command_execution',
    'file_change',
    'mcp_tool_call',
    'web_search',
})

# The in-sandbox proxy returns a call refused on token budget as a 400 whose message carries this
# marker (see rsio_llm_bridge.bridge_error_payload). Codex records a 4xx body verbatim in the
# failing event, so the marker names the exact call it died on. A 5xx would not survive: codex
# replaces its text with a canned "high demand" message.
_ERROR_CODE_MARKER = f'[{CONTEXT_LENGTH_EXCEEDED_CODE}]'


def parse_codex_jsonl(stdout: str) -> list[dict[str, Any]]:
    """Parse the codex ``--json`` stdout into a list of event dicts.

    Robust to interleaved non-JSON lines (progress text, warnings): any line
    that is not a JSON object is skipped. Never raises — returns whatever it
    could parse.
    """
    events: list[dict[str, Any]] = []
    if not stdout:
        return events
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith('{'):
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and obj.get('type'):
            events.append(obj)
    return events


def classify_termination(
    events: list[dict[str, Any]],
    exit_code: int,
    timed_out: bool,
) -> str:
    """Derive a fine-grained termination reason from the event stream.

    Only reasons that are *unambiguously* signalled by the stream are emitted;
    otherwise falls back to the exact same coarse mapping the non-JSON path uses
    (TASK_TIMEOUT / ENV_DONE / UNHANDLED_ERROR), so enabling ``--json`` never
    changes the classification of a run whose events are absent or unclear.

    The refined reasons (``CODEX_TURN_FAILED`` / ``CODEX_ERROR``) are distinct,
    codex-specific strings that are NOT in the default trainer
    ``additional_terminate_reason`` list, so they only add SwanLab termination
    buckets and never silently trigger sample maskout.

    ``TRUNCATION`` is emitted only when the failure that ended the run names an exhausted token
    budget. Reading it off the last failure event is what keeps the attribution honest: a run
    that is refused on budget early, compacts, and later dies of something unrelated is
    classified by the failure it actually died of.
    """
    if timed_out:
        return 'TASK_TIMEOUT'

    has_turn_failed = any(e.get('type') == 'turn.failed' for e in events)
    has_error_event = any(e.get('type') == 'error' for e in events)

    # A non-zero exit with an explicit failure signal -> refined reason.
    if exit_code != 0:
        if _ERROR_CODE_MARKER in _terminal_error_message(events):
            return 'TRUNCATION'
        if has_turn_failed:
            return 'CODEX_TURN_FAILED'
        if has_error_event:
            return 'CODEX_ERROR'
        return 'UNHANDLED_ERROR'

    # exit_code == 0: normal completion. A trailing error event on an otherwise
    # successful exit is unusual; keep ENV_DONE (success is authoritative).
    return 'ENV_DONE'


def _terminal_error_message(events: list[dict[str, Any]]) -> str:
    """Message of the last failure event in the stream — the one codex exited on."""
    for event in reversed(events):
        event_type = event.get('type')
        if event_type == 'turn.failed':
            error = event.get('error')
            return str((error or {}).get('message') or '') if isinstance(error, dict) else ''
        if event_type == 'error':
            return str(event.get('message') or '')
    return ''


def _item_of(event: dict[str, Any]) -> dict[str, Any]:
    item = event.get('item')
    return item if isinstance(item, dict) else {}


def _item_type(item: dict[str, Any]) -> str:
    # The public schema flattens the discriminator as ``item_type``; tolerate a
    # nested ``type`` too, defensively.
    return str(item.get('item_type') or item.get('type') or '')


def _segment_name_and_args(item: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Map a completed item to a (timeline segment name, tool_args) pair."""
    it = _item_type(item)
    if it == 'command_execution':
        command = item.get('command')
        args: dict[str, Any] = {}
        if command is not None:
            args['command'] = command
        if item.get('exit_code') is not None:
            args['exit_code'] = item.get('exit_code')
        if item.get('status'):
            args['status'] = item.get('status')
        return 'shell', (args or None)
    if it == 'file_change':
        changes = item.get('changes')
        args = {'changes': changes} if changes is not None else {}
        return 'apply_patch', (args or None)
    if it == 'mcp_tool_call':
        name = item.get('tool') or item.get('name') or 'mcp_tool_call'
        return str(name), None
    if it == 'web_search':
        return 'web_search', None
    return it or 'observation', None


def build_timeline_segments(
    events: list[dict[str, Any]],
    start_time: float,
    end_time: float,
    *,
    include_tool_args: bool = False,
) -> list[dict[str, Any]]:
    """Build an ordered list of timeline segment descriptors from the events.

    Each descriptor is ``{name, tool_args, start_time, end_time}`` with synthetic
    timestamps (seconds) evenly spread across ``[start_time, end_time]`` — the
    real host-side wall-clock window of the codex run. The ordering is faithful;
    the per-segment durations are approximate (equal slices), NOT measured, since
    the codex stream carries no timings. ``tool_args`` is only attached when
    ``include_tool_args`` is set (mirrors ``timeline_collect_tool_args`` gating).

    Only ``item.completed`` observation items produce segments (a started-but-not-
    completed item, e.g. on timeout, has no end and is skipped).
    """
    completed = [
        _item_of(e)
        for e in events
        if e.get('type') == 'item.completed' and _item_type(_item_of(e)) in _OBSERVATION_ITEM_TYPES
    ]
    n = len(completed)
    if n == 0:
        return []

    # Guard against a bad window (clock skew / zero span): fall back to a tiny
    # nominal 1ms-per-segment spread so timestamps stay strictly monotonic.
    span = end_time - start_time
    if not (span > 0):
        span = n * 0.001
        end_time = start_time + span
    slice_dt = span / n

    segments: list[dict[str, Any]] = []
    for i, item in enumerate(completed):
        name, tool_args = _segment_name_and_args(item)
        seg: dict[str, Any] = {
            'name': name,
            'start_time': start_time + i * slice_dt,
            'end_time': start_time + (i + 1) * slice_dt,
        }
        if include_tool_args and tool_args is not None:
            seg['tool_args'] = tool_args
        segments.append(seg)
    return segments


__all__ = [
    'parse_codex_jsonl',
    'classify_termination',
    'build_timeline_segments',
]


_TEXT_ITEMS = frozenset({'agent_message', 'assistant_message'})

_REASONING_ITEMS = frozenset({'reasoning'})

_META_ITEMS = frozenset({'todo_list', 'token_count', 'web_search', 'user_message'})


def _tool_name(item: dict[str, Any]) -> str:
    """Derive a CC-style tool name from a codex item."""
    item_type = str(item.get('type') or 'unknown')
    if item_type == 'command_execution':
        return 'shell'
    if item_type in {'file_change', 'patch_apply'}:
        return 'apply_patch'
    if item_type == 'mcp_tool_call':
        server = str(item.get('server') or item.get('server_name') or '')
        tool = str(item.get('tool') or item.get('tool_name') or item.get('name') or '')
        if server and tool:
            return f'mcp__{server}__{tool}'
        return tool or 'mcp_tool_call'
    return item_type


def _tool_input(item: dict[str, Any]) -> Any:
    """Best-effort tool input, mirroring CC's ``tool_use.input``."""
    for key in ('command', 'arguments', 'args', 'input', 'changes', 'path'):
        if key in item and item[key] is not None:
            return item[key]
    return None


def parse_codex_stream(raw: str) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Parse ``codex exec --json`` stdout.

    Returns ``(trajectory, final_text, terminal_status)`` where ``terminal_status``
    is the last turn outcome seen (``'completed'`` / ``'failed'``) or an
    ``'error'`` marker, used by :func:`decide_codex_exit` the way CC's
    ``result_subtype`` is used. Never raises: non-JSON lines are skipped.
    """
    trajectory: list[dict[str, Any]] = []
    final_text: str | None = None
    terminal_status: str | None = None

    for line in (raw or '').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        etype = str(event.get('type') or '')
        if etype == 'error':
            terminal_status = 'error'
            trajectory.append({'kind': 'error', 'text': str(event.get('message') or '')})
            continue
        if etype == 'turn.failed':
            terminal_status = 'failed'
            error = event.get('error')
            if isinstance(error, dict) and error.get('message'):
                trajectory.append({'kind': 'error', 'text': str(error['message'])})
            continue
        if etype == 'turn.completed':
            terminal_status = 'completed'
            continue
        if etype != 'item.completed':
            continue

        item = event.get('item')
        if not isinstance(item, dict):
            continue
        item_type = str(item.get('type') or '')
        if item_type in _META_ITEMS:
            continue
        if item_type in _TEXT_ITEMS:
            text = str(item.get('text') or '')
            trajectory.append({'kind': 'text', 'text': text})
            # Last agent message wins — that is codex's final answer.
            final_text = text or final_text
        elif item_type in _REASONING_ITEMS:
            trajectory.append({'kind': 'reasoning', 'text': str(item.get('text') or '')})
        elif item_type == 'error':
            message = str(item.get('message') or '')
            trajectory.append({'kind': 'error', 'text': message})
            # A stream-level error item is not necessarily terminal (codex emits one
            # for benign things like unknown model metadata), so do not set
            # terminal_status here; turn.failed / a non-zero rc decide that.
        else:
            trajectory.append({
                'kind': 'tool_use',
                'name': _tool_name(item),
                'input': _tool_input(item),
            })

    return trajectory, final_text, terminal_status


def decide_codex_exit(
    rc: int | None,
    final_text: str | None,
    terminal_status: str | None,
    done_marker: str,
    infeasible_marker: str = 'TASK_INFEASIBLE',
    stderr: str | None = None,
    timed_out: bool = False,
) -> str:
    """Map codex outcome → the CC exit_status vocabulary.

    Returns one of ``finished`` / ``infeasible`` / ``max_iterations`` / ``tool_error``
    so downstream pipelines keyed on CC's strings keep working.
    """
    text = final_text or ''
    if done_marker and done_marker in text:
        return 'finished'
    if infeasible_marker and infeasible_marker in text:
        return 'infeasible'
    # Codex has no turn cap of its own; a wall-clock kill is the analogue of CC's
    # error_max_turns — the agent ran and produced work, so it is NOT a tool_error.
    if timed_out or 'timed out' in str(stderr or '').lower():
        return 'max_iterations'
    if terminal_status == 'failed' or rc != 0:
        return 'tool_error'
    return 'finished'
