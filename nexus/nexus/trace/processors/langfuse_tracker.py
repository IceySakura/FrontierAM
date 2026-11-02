# /// script
# dependencies = ["langfuse>=4.0.0"]
#
# [project]
# name = "langfuse_tracker"
# authors = [
#     { name = "kechizhang" },
# ]
#
# [tool.nexus]
# span_processors = { langfuse_tracker = "nexus.trace.processors.langfuse_tracker" }
# ///

"""Realtime Langfuse tracker as a SpanProcessor.

Listens for ``openai_completion`` spans and uploads each LLM call to Langfuse
*immediately* on ``on_end``, matching the structure used by upload_nexus_swe.py:

  Trace  (with tags, session_id, input=task, output=result)
    └── Span "Agent"
          ├── Span "System Prompt"
          ├── Span "User Prompt"  (with context_blocks)
          ├── Generation "Step 1: Agent"
          │     └── Tool: Agent
          │           └── Subagent: {desc}
          │                 ├── System Prompt
          │                 ├── User Prompt
          │                 ├── Step 1: Grep, Read
          │                 └── Final Response
          ├── Generation "Step 2: Read"
          │     ...
          └── Generation "Final Response"
"""

from __future__ import annotations

import base64
import contextlib
import copy
import json
import logging
import math
import os
import re
import threading
import time as _time
import urllib.request

from typing import Any

from langfuse import Langfuse, propagate_attributes
from langfuse.types import TraceContext

from nexus.core.types.usage import billed_prompt_usage
from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import SpanProcessor, span_processor_registry
from nexus.trace.types import Span
from nexus.utils.common import serialize_object


logger = logging.getLogger(__name__)


class _SuppressOtelDetachMismatch(logging.Filter):
    """Drop the benign OpenTelemetry "Failed to detach context" ERROR record.

    ``propagate_attributes`` (langfuse) attaches an OpenTelemetry context on
    ``__enter__`` and detaches it on ``__exit__``. In the RL agent loop those two
    calls run in DIFFERENT asyncio Tasks: ``__enter__`` fires from the inline
    ``on_end`` span hook (on the run's Task, on the first ``openai_completion``
    span), while ``__exit__`` runs from ``finalize`` (invoked by
    ``processor_context.__exit__`` in a different contextvars.Context). Detaching a
    token created in another context raises ``ValueError`` INSIDE
    ``opentelemetry.context.detach``, which catches it itself and logs it via
    ``logger.exception`` before returning normally -- so a ``try/except`` around
    ``__exit__`` cannot suppress it. This filter, attached only for the duration of
    that single ``__exit__`` call, drops exactly that record and nothing else. The
    attach/detach still happen; only the cosmetic ERROR log is silenced.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return 'Failed to detach context' not in record.getMessage()


@contextlib.contextmanager
def _suppressed_otel_detach_log():
    """Temporarily silence the benign OTel detach-mismatch ERROR (see filter above).

    Safe because the guarded ``__exit__`` is synchronous (no ``await``), so no other
    coroutine can interleave between add/remove; and a stray record from another
    thread would at worst be dropped once. The filter is always removed on exit.
    """
    otel_logger = logging.getLogger('opentelemetry.context')
    log_filter = _SuppressOtelDetachMismatch()
    otel_logger.addFilter(log_filter)
    try:
        yield
    finally:
        otel_logger.removeFilter(log_filter)


_AGENT_TOOL_NAMES = frozenset({'Agent', 'Task', 'Explore', 'Plan'})
_CCH_PATTERN = re.compile(r'cch=[^;]*;')
_PROMPT_OFFSET_NS = 200_000_000  # 200ms gap to separate prompt spans from first generation
_TAIJI_CONTRACT = 'episode_trace_generation_turn'
_TAIJI_METADATA_ENV = {
    'instance_id': 'TJ_INSTANCE_ID',
    'task_id': 'TJ_TASK_ID',
    'task_name': 'TJ_TASK_NAME',
    'workspace_id': 'TJ_WORKSPACE_ID',
}
# (api_host, public_key) -> project id. /trace/{id} is only a 307; the UI path needs project id.
_PROJECT_ID_CACHE: dict[tuple[str, str], str] = {}
_PROJECT_ID_LOCK = threading.Lock()
# Lookups already handed to a background thread. on_end must not wait on them.
_PROJECT_ID_STARTED: set[tuple[str, str]] = set()
_TAIJI_API_HOST_MARKERS = ('langfuse-taiji-api.woa.com',)
_TAIJI_UI_HOST = 'https://langfuse-taiji.woa.com'


def _normalize_langfuse_host(host: str) -> str:
    return host.rstrip('/')


def _langfuse_ui_host(host: str) -> str:
    """API host is not clickable; Taiji UI lives on langfuse-taiji.woa.com."""
    override = (os.getenv('LANGFUSE_UI_HOST') or '').strip().rstrip('/')
    if override:
        return override
    base = _normalize_langfuse_host(host)
    if any(marker in base for marker in _TAIJI_API_HOST_MARKERS):
        return _TAIJI_UI_HOST
    return base


def _langfuse_trace_url(host: str, project_id: str | None, trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    base = _langfuse_ui_host(host)
    if project_id:
        return f'{base}/project/{project_id}/traces/{trace_id}'
    return f'{base}/trace/{trace_id}'


def _configured_project_id() -> str | None:
    pid = (os.getenv('LANGFUSE_PROJECT_ID') or '').strip()
    return pid or None


def _cached_project_id(cache_key: tuple[str, str]) -> str | None:
    with _PROJECT_ID_LOCK:
        return _PROJECT_ID_CACHE.get(cache_key)


def _known_project_id(host: str, public_key: str, current: str | None = None) -> str | None:
    if current:
        return current
    configured = _configured_project_id()
    if configured:
        return configured
    return _cached_project_id((_normalize_langfuse_host(host), public_key))


def _fetch_langfuse_project_id(host: str, public_key: str, secret_key: str) -> str | None:
    """Blocking project-id lookup. Callers on the inline ``on_end`` path use the scheduler below."""
    configured = _configured_project_id()
    if configured:
        return configured
    cache_key = (_normalize_langfuse_host(host), public_key)
    cached = _cached_project_id(cache_key)
    if cached:
        return cached
    url = f'{cache_key[0]}/api/public/projects'
    token = base64.b64encode(f'{public_key}:{secret_key}'.encode()).decode()
    request = urllib.request.Request(
        url,
        headers={'Authorization': f'Basic {token}', 'Accept': 'application/json'},
    )
    try:
        # Synchronous on purpose: this runs on a daemon thread, not the agent event loop.
        with urllib.request.urlopen(request, timeout=5) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        logger.warning('Failed to resolve Langfuse project id for trace_url', exc_info=True)
        return None
    projects = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(projects, list) or not projects or not isinstance(projects[0], dict):
        logger.warning('Langfuse /api/public/projects returned no projects')
        return None
    project_id = projects[0].get('id')
    if not isinstance(project_id, str) or not project_id:
        logger.warning('Langfuse /api/public/projects missing project id')
        return None
    if len(projects) > 1:
        logger.debug(
            'Langfuse key has %d projects; using %s for trace_url',
            len(projects),
            project_id,
        )
    with _PROJECT_ID_LOCK:
        _PROJECT_ID_CACHE[cache_key] = project_id
    return project_id


def _schedule_langfuse_project_id(
    host: str,
    public_key: str,
    secret_key: str,
    trace_id: str | None,
) -> str | None:
    """Return a project id without blocking. One HTTP lookup runs in the background."""
    known = _known_project_id(host, public_key)
    if known:
        return known
    cache_key = (_normalize_langfuse_host(host), public_key)
    with _PROJECT_ID_LOCK:
        if cache_key in _PROJECT_ID_STARTED or cache_key in _PROJECT_ID_CACHE:
            return _PROJECT_ID_CACHE.get(cache_key)
        _PROJECT_ID_STARTED.add(cache_key)

    def _resolve() -> None:
        project_id = _fetch_langfuse_project_id(host, public_key, secret_key)
        if not project_id:
            return
        url = _langfuse_trace_url(host, project_id, trace_id)
        if url:
            logger.debug('Langfuse trace url: %s', url)

    threading.Thread(target=_resolve, name='langfuse-project-id', daemon=True).start()
    return None


def _set_span_start_time(span: Any, ns: int) -> None:
    """Patch OTel span _start_time to guarantee ordering in Langfuse UI."""
    try:
        otel = getattr(span, '_otel_span', None)
        if otel is not None and hasattr(otel, '_start_time'):
            otel._start_time = ns
    except Exception:
        pass


def _trunc(s: Any, limit: int) -> str:
    if not s:
        return ''
    s = str(s)
    return s if (limit <= 0 or len(s) <= limit) else s[:limit] + f'...[{len(s)} chars]'


def _trunc_json(value: Any, limit: int) -> Any:
    """Keep a JSON payload as-is unless it exceeds ``limit`` serialized chars."""
    if limit <= 0 or value is None:
        return value
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return _trunc(value, limit)
    if len(encoded) <= limit:
        return value
    return {'truncated': True, 'chars': len(encoded), 'payload': encoded[:limit]}


def _dedupe_nonempty(items: list[Any]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item is None:
            continue
        value = str(item)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _build_taiji_metadata(trace_metadata: dict[str, Any] | None) -> dict[str, Any]:
    env_metadata = {key: value for key, env_name in _TAIJI_METADATA_ENV.items() if (value := os.environ.get(env_name))}
    metadata = {**env_metadata, **(trace_metadata or {})}
    metadata['contract'] = _TAIJI_CONTRACT
    metadata.setdefault('episode', 0)

    instance_id = metadata.get('instance_id')
    if instance_id and not metadata.get('episode_id'):
        metadata['episode_id'] = f'{instance_id}-ep{metadata["episode"]}'

    return metadata


def _default_trace_name(trace_metadata: dict[str, Any]) -> str:
    return str(trace_metadata.get('instance_id') or trace_metadata.get('task_name') or '')


def _extract_numeric_score(task_output: dict[str, Any] | None) -> float | None:
    if not task_output or 'score' not in task_output:
        return None
    try:
        score = float(task_output['score'])
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                t = block.get('type', '')
                if t == 'text':
                    parts.append(block.get('text', ''))
                elif t == 'thinking':
                    parts.append(block.get('thinking', ''))
            elif isinstance(block, str):
                parts.append(block)
        return '\n'.join(p for p in parts if p)
    return str(content)


def _normalize_system(content: Any) -> str:
    """Normalize system prompt for comparison (strip volatile cch= tokens)."""
    text = _extract_text_content(content)
    return _CCH_PATTERN.sub('', text)


def _extract_user_message(content: Any) -> dict[str, Any]:
    """Extract task_text, full_text, and context_blocks from user message."""
    if isinstance(content, str):
        return {'task_text': content, 'full_text': content, 'blocks': [], 'context_blocks': []}
    if not isinstance(content, list):
        s = str(content)
        return {'task_text': s, 'full_text': s, 'blocks': [], 'context_blocks': []}
    task_text = ''
    parts = []
    blocks = []
    context_blocks = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get('text', '')
        btype = block.get('type', 'text')
        blocks.append({'type': btype, 'text': text})
        if text:
            parts.append(text)
        if btype == 'text' and text.strip().startswith('<system-reminder>'):
            context_blocks.append({'type': btype, 'text': _trunc(text, 500)})
        elif btype == 'text' and not task_text:
            task_text = text
    full_text = '\n'.join(parts)
    return {
        'task_text': task_text or full_text,
        'full_text': full_text,
        'blocks': blocks,
        'context_blocks': context_blocks,
    }


def _collect_input_signatures(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Locate thinking signatures in the input history with their owning message.

    Returns a list of ``{message_index, tool_call_id, signature_index, signature}``
    so the langfuse trace shows precisely which assistant message a signature
    belongs to. This lets a reader diff against the prior turn's
    output_signatures and pinpoint mis-mapping (Bedrock validates each turn
    independently and reports the exact ``messages[N]`` index on error, so the
    trace must mirror that granularity).
    """
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get('role') != 'assistant':
            continue
        tc_id = None
        for tc in msg.get('tool_calls') or []:
            if isinstance(tc, dict) and tc.get('id'):
                tc_id = str(tc['id'])
                break
        sigs_for_msg: list[str] = []
        content = msg.get('content')
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'thinking' and block.get('signature'):
                    sigs_for_msg.append(str(block['signature']))
        for tb in msg.get('thinking_blocks') or []:
            if isinstance(tb, dict) and tb.get('signature'):
                sigs_for_msg.append(str(tb['signature']))
        for j, sig in enumerate(sigs_for_msg):
            out.append({
                'message_index': i,
                'tool_call_id': tc_id,
                'signature_index': j,
                'signature': sig,
            })
    return out


def _message_tool_call_ids(msg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for tc in msg.get('tool_calls') or []:
        if isinstance(tc, dict) and tc.get('id'):
            out.append(str(tc['id']))
    return out


def _normalize_reasoning_item_for_trace(item: Any) -> dict[str, Any] | None:
    """Copy a Responses reasoning item for trace upload, including raw encrypted_content."""
    if not isinstance(item, dict):
        item = serialize_object(item)
    if not isinstance(item, dict):
        return None

    normalized = copy.deepcopy(item)
    if not normalized.get('type'):
        normalized['type'] = 'reasoning'
    if normalized.get('encrypted_content') is not None:
        normalized['encrypted_content'] = str(normalized['encrypted_content'])
    if normalized.get('id') is not None:
        normalized['id'] = str(normalized['id'])
    return normalized


def _reasoning_trace_entry(
    item: Any,
    *,
    reasoning_index: int,
    tool_call_ids: list[str],
    message_index: int | None = None,
    choice_index: int | None = None,
) -> dict[str, Any] | None:
    normalized = _normalize_reasoning_item_for_trace(item)
    if normalized is None:
        return None

    entry = dict(normalized)
    if message_index is not None:
        entry['message_index'] = message_index
    if choice_index is not None:
        entry['choice_index'] = choice_index
    entry['tool_call_ids'] = list(tool_call_ids)
    entry['reasoning_index'] = reasoning_index
    return entry


def _collect_input_reasoning_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Locate OpenAI Responses reasoning_items in the input history."""
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get('role') != 'assistant':
            continue
        reasoning_items = msg.get('reasoning_items') or []
        if not isinstance(reasoning_items, list):
            continue
        tool_call_ids = _message_tool_call_ids(msg)
        for j, item in enumerate(reasoning_items):
            entry = _reasoning_trace_entry(
                item,
                message_index=i,
                reasoning_index=j,
                tool_call_ids=tool_call_ids,
            )
            if entry is not None:
                out.append(entry)
    return out


def _reasoning_match_key(entry: dict[str, Any]) -> tuple[str, str] | None:
    encrypted_content = entry.get('encrypted_content')
    if encrypted_content:
        return ('encrypted_content', str(encrypted_content))
    item_id = entry.get('id')
    if item_id:
        return ('id', str(item_id))
    return None


def _parse_response(
    output: Any,
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse LLM response into text, thinking, tools, usage, signatures, reasoning_items.

    thinking_signatures: list of ``{tool_call_id, signature_index, signature}``
    entries, one per signed thinking block, so the langfuse trace records both
    the signature value and the assistant turn it belongs to (matched on the
    next request via tool_call_id).
    """
    payload = serialize_object(output) if not isinstance(output, dict) else output
    if not isinstance(payload, dict):
        return str(output) if output else '', '', [], None, [], []

    choices = payload.get('choices') or []
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    reasoning_items: list[dict[str, Any]] = []

    for choice_index, c in enumerate(choices):
        msg = c.get('message') or c.get('delta') or {}
        if msg.get('content'):
            text_parts.append(str(msg['content']))
        # litellm mirrors the SAME reasoning onto BOTH message.reasoning_content AND
        # message.provider_specific_fields.reasoning_content (convert_dict_to_response sets the
        # field on the Message and copies it into provider_specific_fields). Appending both
        # double-counts the thinking in the langfuse trace (e.g. 412 -> 824) for any litellm
        # openai-chat agent (codex, opencode, kilo...). Take exactly ONE via `or`: prefer the
        # top-level field (always set on the common path), fall back to the provider_specific copy
        # only when the top-level is empty (some providers, e.g. deepseek/moonshot, populate psf
        # first and promote later) -- so we never double-count and never drop reasoning.
        reasoning_content = msg.get('reasoning_content') or (msg.get('provider_specific_fields') or {}).get(
            'reasoning_content'
        )
        if reasoning_content:
            thinking_parts.append(str(reasoning_content))
        first_tc_id: str | None = None
        for tc in msg.get('tool_calls') or []:
            if isinstance(tc, dict) and tc.get('id'):
                first_tc_id = str(tc['id'])
                break
        for j, tb in enumerate(msg.get('thinking_blocks') or []):
            if isinstance(tb, dict) and tb.get('signature'):
                # Record the tool_call_id the signature should be re-anchored
                # to, so the input collector on the next turn can match it.
                signatures.append({
                    'tool_call_id': first_tc_id,
                    'signature_index': j,
                    'signature': str(tb['signature']),
                })
        msg_tool_call_ids = _message_tool_call_ids(msg)
        raw_reasoning_items = msg.get('reasoning_items') or []
        if isinstance(raw_reasoning_items, list):
            for j, item in enumerate(raw_reasoning_items):
                entry = _reasoning_trace_entry(
                    item,
                    choice_index=choice_index,
                    reasoning_index=j,
                    tool_call_ids=msg_tool_call_ids,
                )
                if entry is not None:
                    reasoning_items.append(entry)
        for tc in msg.get('tool_calls') or []:
            fn = tc.get('function', {})
            if fn.get('name'):
                args_raw = fn.get('arguments', '{}')
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except (json.JSONDecodeError, TypeError):
                    args = {'raw': str(args_raw)[:500]}
                tool_calls.append({
                    'id': tc.get('id', ''),
                    'name': fn['name'],
                    'arguments': args,
                })

    usage = payload.get('usage')
    billed = billed_prompt_usage(payload)
    if billed is not None and isinstance(usage, dict):
        # Several provider requests behind one completion: report billed input, not the context size.
        prompt_tokens, cached_tokens = billed
        usage = {
            **usage,
            'prompt_tokens': prompt_tokens,
            'total_tokens': prompt_tokens + (usage.get('completion_tokens') or 0),
            'prompt_tokens_details': {**(usage.get('prompt_tokens_details') or {}), 'cached_tokens': cached_tokens},
        }
    return ''.join(text_parts), ''.join(thinking_parts), tool_calls, usage, signatures, reasoning_items


def _convert_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    if not usage or not isinstance(usage, dict):
        return {'input': 0, 'output': 0, 'total': 0}
    cached = 0
    ptd = usage.get('prompt_tokens_details') or {}
    if isinstance(ptd, dict):
        cached = ptd.get('gpu_cached_tokens', 0) or ptd.get('cached_tokens', 0) or 0
    return {
        'input': usage.get('prompt_tokens', 0) or 0,
        'output': usage.get('completion_tokens', 0) or 0,
        'total': usage.get('total_tokens', 0) or 0,
        'cached_tokens': cached,
    }


def _extract_model_parameters(inputs: dict[str, Any]) -> dict[str, Any]:
    """Extract LLM call parameters from span inputs."""
    params: dict[str, Any] = {}
    kwargs = inputs.get('kwargs', {})
    if not isinstance(kwargs, dict):
        return params
    if kwargs.get('temperature') is not None:
        params['temperature'] = kwargs['temperature']
    if kwargs.get('max_tokens') is not None:
        params['max_tokens'] = kwargs['max_tokens']
    extra_body = kwargs.get('extra_body', {})
    if isinstance(extra_body, dict):
        chat_kwargs = extra_body.get('chat_template_kwargs', {})
        if isinstance(chat_kwargs, dict):
            if chat_kwargs.get('reasoning_effort'):
                params['reasoning_effort'] = chat_kwargs['reasoning_effort']
            if chat_kwargs.get('interleaved_thinking') is not None:
                params['interleaved_thinking'] = chat_kwargs['interleaved_thinking']
    extra_headers = kwargs.get('extra_headers', {})
    if isinstance(extra_headers, dict) and extra_headers.get('StrategyType'):
        params['StrategyType'] = extra_headers['StrategyType']
    return params


def _extract_tools_summary(inputs: dict[str, Any]) -> list[dict[str, str]]:
    """Extract tools definition summary from span inputs."""
    tools = inputs.get('kwargs', {}).get('tools') or []
    if not isinstance(tools, list):
        return []
    summary = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get('function', {})
        name = fn.get('name') or t.get('name', '')
        desc = fn.get('description') or t.get('description', '')
        if name:
            summary.append({'name': name, 'description': _trunc(desc, 120)})
    return summary


def _get_system_prompt(messages: list[dict[str, Any]]) -> str:
    """Extract normalized system prompt from message list."""
    for m in messages:
        if m.get('role') == 'system':
            return _normalize_system(m.get('content', ''))
    return ''


def _truncate_message(msg: dict[str, Any], max_content: int, max_tool: int) -> dict[str, Any]:
    """Truncate a single message's content while preserving full structure."""
    out: dict[str, Any] = {'role': msg.get('role', '')}

    content = msg.get('content')
    if isinstance(content, str):
        out['content'] = _trunc(content, max_tool if msg.get('role') == 'tool' else max_content)
    elif isinstance(content, list):
        truncated_blocks = []
        for block in content:
            if isinstance(block, dict):
                tb = dict(block)
                if 'text' in tb:
                    tb['text'] = _trunc(tb['text'], max_content)
                if 'thinking' in tb:
                    tb['thinking'] = _trunc(tb['thinking'], max_content)
                truncated_blocks.append(tb)
            else:
                truncated_blocks.append(block)
        out['content'] = truncated_blocks
    elif content is not None:
        out['content'] = content

    if msg.get('tool_calls'):
        out['tool_calls'] = []
        for tc in msg['tool_calls']:
            fn = tc.get('function', {})
            out['tool_calls'].append({
                'id': tc.get('id', ''),
                'type': tc.get('type', 'function'),
                'function': {
                    'name': fn.get('name', ''),
                    'arguments': _trunc(fn.get('arguments', ''), 500),
                },
            })

    if msg.get('tool_call_id'):
        out['tool_call_id'] = msg['tool_call_id']
    if msg.get('name'):
        out['name'] = msg['name']
    if msg.get('reasoning_content'):
        out['reasoning_content'] = _trunc(msg['reasoning_content'], max_content)
    psf = msg.get('provider_specific_fields')
    if psf and isinstance(psf, dict) and psf.get('reasoning_content'):
        out['reasoning_content'] = _trunc(psf['reasoning_content'], max_content)
    if msg.get('reasoning_items'):
        reasoning_items = []
        for item in msg['reasoning_items']:
            normalized = _normalize_reasoning_item_for_trace(item)
            if normalized is not None:
                reasoning_items.append(normalized)
        if reasoning_items:
            out['reasoning_items'] = reasoning_items

    return out


# ---- Agent context: tracks per-agent state (main or subagent) ----


class _AgentContext:
    """State for a single agent level (main or subagent)."""

    __slots__ = (
        'parent_span',
        'step',
        'prev_generation',
        'prev_tool_calls',
        'prev_turn_metadata',
        'prev_message_count',
        'prev_output_signatures',
        'prev_output_reasoning_items',
        'system_prompt_created',
        'user_prompt_created',
        'tools_definition_created',
        'task_text',
        'last_output',
        'system_prompt_norm',
    )

    def __init__(self, parent_span: Any, system_prompt_norm: str = ''):
        self.parent_span = parent_span
        self.step = 0
        self.prev_generation: Any = None
        self.prev_tool_calls: list[dict[str, Any]] = []
        self.prev_turn_metadata: dict[str, Any] = {}
        self.prev_message_count = 0
        self.prev_output_signatures: list[dict[str, Any]] = []
        self.prev_output_reasoning_items: list[dict[str, Any]] = []
        self.system_prompt_created = False
        self.user_prompt_created = False
        self.tools_definition_created = False
        self.task_text = ''
        self.last_output = ''
        self.system_prompt_norm = system_prompt_norm


@span_processor_registry.register('langfuse_tracker')
class LangfuseTrackerProcessor(SpanProcessor):
    """Realtime Langfuse tracker that uploads each LLM call as it completes."""

    def __init__(
        self,
        public_key: str = '',
        secret_key: str = '',
        host: str = '',
        session_id: str = '',
        trace_name: str = '',
        tags: list[str] | None = None,
        trace_metadata: dict[str, Any] | None = None,
        max_tool_chars: int = 8000,
        max_content_chars: int = 0,
        record_provider_call: bool = False,
        max_provider_call_chars: int = 0,
        flush_on_finalize: bool = True,
    ):
        self._public_key = public_key or os.environ.get('LANGFUSE_PUBLIC_KEY', '')
        self._secret_key = secret_key or os.environ.get('LANGFUSE_SECRET_KEY', '')
        self._host = (
            host
            or os.environ.get('LANGFUSE_HOST', '')
            or os.environ.get('LANGFUSE_BASE_URL', '')
            or 'https://cloud.langfuse.com'
        )
        self._trace_metadata = _build_taiji_metadata(trace_metadata)
        self._trace_name = trace_name or _default_trace_name(self._trace_metadata)
        self._session_id = session_id or self._trace_metadata.get('episode_id', '')
        self._extra_tags = _dedupe_nonempty(
            list(tags or [])
            + [
                self._trace_metadata.get('instance_id'),
                self._trace_metadata.get('task_id'),
                self._trace_metadata.get('task_name'),
                self._trace_metadata.get('workspace_id'),
            ]
        )
        self._max_tool_chars = max_tool_chars
        self._max_content_chars = max_content_chars
        # The generation input/output above is Nexus' normalized OpenAI shape.
        # With this on, the provider-level call recorded on the span is uploaded
        # too, so the trace also holds the exact wire payload (Responses ``input``
        # with its encrypted reasoning, ``store``/``include``/cache key, and the
        # per-attempt error bodies). ``request``/``response`` are only present
        # when the client runs with ``capture_raw_io``.
        self._record_provider_call = record_provider_call
        self._max_provider_call_chars = max_provider_call_chars
        # Hooks run inline on the caller's event loop, and ``flush()`` blocks on network I/O while
        # draining the process-wide Langfuse queue. Long-lived hosts that run many concurrent
        # agents (RL rollout workers) pass False and flush once at process shutdown instead.
        self._flush_on_finalize = flush_on_finalize
        self._task_output: dict[str, Any] | None = None

        # Langfuse objects
        self._langfuse: Any = None
        self._trace: Any = None
        self._trace_id: str | None = None
        self._project_id: str | None = None
        self._agent_span: Any = None
        self._propagate_ctx: Any = None

        # Agent context stack: [main, subagent1, ...]
        self._ctx_stack: list[_AgentContext] = []
        self._ctx: _AgentContext | None = None  # current context (top of stack)

        # Subagent tracking
        self._subagent_tool_span: Any = None  # open Tool: Agent span
        self._subagent_wrapper_span: Any = None  # open Subagent: {desc} span

        # Global state
        self._model_name = ''
        self._model_parameters: dict[str, Any] = {}
        self._tools_summary: list[dict[str, str]] = []
        self._total_usage = {'input': 0, 'output': 0, 'total': 0, 'cached_tokens': 0}
        self._global_step = 0
        self._initialized = False

    def _ensure_init(self) -> bool:
        if self._initialized:
            return self._langfuse is not None
        self._initialized = True

        if not self._public_key or not self._secret_key:
            logger.warning('Langfuse credentials not configured, tracker disabled')
            return False

        try:
            self._langfuse = Langfuse(
                public_key=self._public_key,
                secret_key=self._secret_key,
                host=self._host,
            )

            experiment = self._trace_metadata.get('experiment', '')
            ts = str(int(_time.time() * 1000))
            seed_parts = [p for p in [self._trace_name, experiment, ts] if p]
            seed = '::'.join(seed_parts) if seed_parts else None
            trace_id = self._langfuse.create_trace_id(seed=seed)
            self._trace_id = trace_id
            # The inline on_end hook is on the agent task. The project id is only for the
            # clickable URL, so the HTTP lookup must not stall that task.
            self._project_id = _schedule_langfuse_project_id(
                self._host,
                self._public_key,
                self._secret_key,
                trace_id,
            )
            trace_ctx = TraceContext(trace_id=trace_id, parent_span_id='')

            self._propagate_ctx = propagate_attributes(
                tags=self._extra_tags,
                session_id=self._session_id or None,
                trace_name=self._trace_name or None,
            )
            self._propagate_ctx.__enter__()

            self._trace = self._langfuse.start_observation(
                trace_context=trace_ctx,
                name=self._trace_name or 'Agent Run',
                as_type='span',
                input={'note': 'SWE-bench agent run'},
                metadata={**self._trace_metadata, 'source': 'langfuse_tracker'},
            )

            self._agent_span = self._trace.start_observation(
                name='Agent',
                as_type='span',
                metadata={'source': 'langfuse_tracker'},
            )

            # Initialize main agent context
            self._ctx = _AgentContext(parent_span=self._agent_span)
            self._ctx_stack = [self._ctx]

            logger.debug(
                'Langfuse tracker initialized: trace_id=%s url=%s',
                trace_id,
                _langfuse_trace_url(self._host, self._project_id, trace_id),
            )
            return True
        except Exception as e:
            logger.warning('Failed to initialize Langfuse: %s', e)
            self._langfuse = None
            return False

    def _trunc_tool(self, s: Any) -> str:
        return _trunc(s, self._max_tool_chars)

    def _trunc_content(self, s: Any) -> str:
        return _trunc(s, self._max_content_chars)

    def _episode_metadata_fields(self) -> dict[str, Any]:
        fields = {
            'contract': _TAIJI_CONTRACT,
            'episode': self._trace_metadata.get('episode', 0),
        }
        if self._trace_metadata.get('episode_id'):
            fields['episode_id'] = self._trace_metadata['episode_id']
        return fields

    def _turn_metadata_fields(self, step: int, global_step: int) -> dict[str, Any]:
        return {
            **self._episode_metadata_fields(),
            'turn': step,
            'step': step,
            'global_step': global_step,
        }

    def _score_comment(self) -> str:
        parts = []
        if self._trace_metadata.get('instance_id'):
            parts.append(f'instance={self._trace_metadata["instance_id"]}')
        parts.append(f'episode={self._trace_metadata.get("episode", 0)}')
        if self._trace_metadata.get('task_name'):
            parts.append(f'task={self._trace_metadata["task_name"]}')
        return ', '.join(parts)

    def _create_system_user_prompts(
        self,
        ctx: _AgentContext,
        messages: list[dict[str, Any]],
    ) -> None:
        """Extract system/user prompts from messages and create spans."""
        if ctx.system_prompt_created:
            return
        ctx.system_prompt_created = True

        # Use timestamps slightly before now so prompts sort before Step 1 in UI
        now_ns = _time.time_ns()
        sys_start_ns = now_ns - 2 * _PROMPT_OFFSET_NS
        user_start_ns = now_ns - _PROMPT_OFFSET_NS

        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')

            if role == 'system' and not ctx.user_prompt_created:
                text = _extract_text_content(content)
                if text:
                    sp = ctx.parent_span.start_observation(
                        name='System Prompt',
                        as_type='span',
                        input={'content': self._trunc_content(text)},
                        metadata={'length': len(text)},
                    )
                    _set_span_start_time(sp, sys_start_ns)
                    sp.end()

            elif role == 'user' and not ctx.user_prompt_created:
                ctx.user_prompt_created = True
                user_info = _extract_user_message(content)
                ctx.task_text = user_info['task_text']
                context_blocks = user_info.get('context_blocks', [])

                user_input: dict[str, Any] = {
                    'task': self._trunc_content(ctx.task_text),
                }
                if context_blocks:
                    user_input['context_blocks'] = context_blocks

                sp = ctx.parent_span.start_observation(
                    name='User Prompt',
                    as_type='span',
                    input=user_input,
                    metadata={
                        'length': len(ctx.task_text),
                        'block_count': len(user_info.get('blocks', [])),
                    },
                )
                _set_span_start_time(sp, user_start_ns)
                sp.end()

                # Update trace input only for main agent
                if ctx is self._ctx_stack[0]:
                    trace_input: dict[str, Any] = {
                        'task': self._trunc_content(ctx.task_text),
                    }
                    if context_blocks:
                        trace_input['context_blocks'] = context_blocks
                    self._trace.update(input=trace_input)
                break

    # ---- Subagent transition logic ----

    def _enter_subagent(
        self,
        messages: list[dict[str, Any]],
        sys_norm: str,
    ) -> None:
        """Transition into a subagent: create Tool:Agent + Subagent spans."""
        main_ctx = self._ctx
        if not main_ctx:
            return

        # Find the Agent tool_call from prev_tool_calls
        agent_tc = None
        non_agent_tcs = []
        for tc in main_ctx.prev_tool_calls:
            if tc['name'] in _AGENT_TOOL_NAMES:
                agent_tc = tc
            else:
                non_agent_tcs.append(tc)

        if not agent_tc or not main_ctx.prev_generation:
            logger.warning('Subagent detected but no Agent tool_call found in prev step')
            return

        # Create non-Agent tool spans first (if any parallel tools)
        for tc in non_agent_tcs:
            tool_input = (
                {k: self._trunc_tool(v) for k, v in tc['arguments'].items()}
                if isinstance(tc['arguments'], dict)
                else {'raw': self._trunc_tool(tc['arguments'])}
            )
            ts = main_ctx.prev_generation.start_observation(
                name=f'Tool: {tc["name"]}',
                as_type='tool',
                input=tool_input,
                output='(concurrent with Agent call)',
                metadata={
                    **main_ctx.prev_turn_metadata,
                    'tool_name': tc['name'],
                    'source': 'realtime',
                },
            )
            ts.end()

        # Create Tool: Agent span (keep open until subagent exits)
        args = agent_tc.get('arguments', {})
        desc = args.get('description', 'subagent') if isinstance(args, dict) else 'subagent'
        prompt = args.get('prompt', '') if isinstance(args, dict) else ''
        tool_input = (
            {k: self._trunc_tool(v) for k, v in args.items()}
            if isinstance(args, dict)
            else {'raw': self._trunc_tool(args)}
        )

        self._subagent_tool_span = main_ctx.prev_generation.start_observation(
            name=f'Tool: {agent_tc["name"]}',
            as_type='tool',
            input=tool_input,
            metadata={
                **main_ctx.prev_turn_metadata,
                'tool_call_id': agent_tc.get('id', ''),
                'tool_name': agent_tc['name'],
                'is_agent_call': True,
                'source': 'realtime',
            },
        )

        # Create Subagent wrapper span under Tool: Agent
        self._subagent_wrapper_span = self._subagent_tool_span.start_observation(
            name=f'Subagent: {desc}',
            as_type='span',
            input={'prompt': self._trunc_content(prompt or desc)},
            metadata={
                **self._episode_metadata_fields(),
                'parent_turn': main_ctx.prev_turn_metadata.get('turn'),
                'parent_step': main_ctx.prev_turn_metadata.get('step'),
                'parent_global_step': main_ctx.prev_turn_metadata.get('global_step'),
                'description': desc,
                'subagent_type': args.get('subagent_type', '') if isinstance(args, dict) else '',
                'source': 'realtime',
            },
        )

        # Push new context for subagent
        sub_ctx = _AgentContext(
            parent_span=self._subagent_wrapper_span,
            system_prompt_norm=sys_norm,
        )
        self._ctx_stack.append(sub_ctx)
        self._ctx = sub_ctx

        # Clear main context's prev state (already consumed)
        main_ctx.prev_generation = None
        main_ctx.prev_tool_calls = []
        main_ctx.prev_turn_metadata = {}

        logger.debug('Entered subagent: %s', desc)

    def _exit_subagent(self, messages: list[dict[str, Any]]) -> None:
        """Exit subagent: close subagent spans, restore main context."""
        if len(self._ctx_stack) < 2:
            return

        sub_ctx = self._ctx_stack.pop()

        # Attach any remaining tool results from subagent
        if sub_ctx.prev_generation and sub_ctx.prev_tool_calls:
            self._attach_tool_results_for(sub_ctx, messages)

        # Close Subagent wrapper span
        if self._subagent_wrapper_span:
            self._subagent_wrapper_span.update(
                output={'result': sub_ctx.last_output[:2000] if sub_ctx.last_output else ''},
                metadata={
                    **self._episode_metadata_fields(),
                    'total_steps': sub_ctx.step,
                    'source': 'realtime',
                },
            )
            self._subagent_wrapper_span.end()
            self._subagent_wrapper_span = None

        # Close Tool: Agent span with subagent result
        if self._subagent_tool_span:
            self._subagent_tool_span.update(
                output=self._trunc_tool(sub_ctx.last_output),
            )
            self._subagent_tool_span.end()
            self._subagent_tool_span = None

        # Restore main context
        self._ctx = self._ctx_stack[-1]
        # Main ctx prev state was cleared on enter, clear again to avoid double-attach
        self._ctx.prev_generation = None
        self._ctx.prev_tool_calls = []
        self._ctx.prev_turn_metadata = {}

        logger.info('Exited subagent after %d steps', sub_ctx.step)

    # ---- Core on_end ----

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        # Capture task output from task span
        if span.attributes.get('type') == 'task':
            outputs = span.attributes.get('outputs')
            if outputs and isinstance(outputs, dict):
                self._task_output = outputs
            return

        if span.name != 'openai_completion':
            return
        if not self._ensure_init():
            return
        if not self._ctx:
            return

        inputs = span.attributes.get('inputs', {})
        outputs = span.attributes.get('outputs')
        if not inputs:
            return

        messages = inputs.get('messages', [])
        if not messages:
            return

        # Capture tools and model parameters on first call
        if not self._tools_summary:
            self._tools_summary = _extract_tools_summary(inputs)
        if not self._model_parameters:
            self._model_parameters = _extract_model_parameters(inputs)

        # Detect subagent transitions via system prompt comparison. Skip this for
        # tool-less utility sub-calls (e.g. Claude Code's "generate a title" call):
        # they carry a different, tiny system prompt but are NOT a subagent, so they
        # must not set the main system prompt baseline or trigger a false transition.
        call_has_tools = bool(inputs.get('kwargs', {}).get('tools'))
        current_sys = _get_system_prompt(messages)
        main_ctx = self._ctx_stack[0]

        if call_has_tools:
            if not main_ctx.system_prompt_norm:
                main_ctx.system_prompt_norm = current_sys

            is_main_agent = current_sys == main_ctx.system_prompt_norm
            in_subagent = len(self._ctx_stack) > 1

            if not is_main_agent and not in_subagent:
                self._enter_subagent(messages, current_sys)
            elif is_main_agent and in_subagent:
                self._exit_subagent(messages)

        ctx = self._ctx
        if not ctx:
            return

        # The "System Prompt" / "Tools Definition" summary spans describe the MAIN
        # agent. Claude Code (interactive) fires small utility sub-calls — e.g.
        # "generate a session title" — that carry their own tiny system prompt and
        # NO tools. Locking the summary spans onto whichever call happens to be
        # first would record the title call's system prompt as the agent's. So we
        # only create these summary spans from a call that actually carries tools
        # (the real agent request). Each generation's own input still faithfully
        # records that specific call's full system prompt + tools regardless.

        # Create System/User Prompt spans on first tool-bearing call per context
        if call_has_tools:
            self._create_system_user_prompts(ctx, messages)

        # Record full tools definition once per agent context
        if call_has_tools and not ctx.tools_definition_created:
            ctx.tools_definition_created = True
            raw_tools = inputs.get('kwargs', {}).get('tools') or []
            if raw_tools:
                tools_span = ctx.parent_span.start_observation(
                    name='Tools Definition',
                    as_type='span',
                    input={'tools': raw_tools},
                    metadata={'tool_count': len(raw_tools)},
                )
                _set_span_start_time(tools_span, _time.time_ns() - _PROMPT_OFFSET_NS // 2)
                tools_span.end()

        ctx.step += 1
        self._global_step += 1
        (
            assistant_text,
            thinking_text,
            tool_calls,
            usage,
            output_signatures,
            output_reasoning_items,
        ) = _parse_response(outputs)
        input_signatures = _collect_input_signatures(messages)
        input_reasoning_items = _collect_input_reasoning_items(messages)

        if assistant_text:
            ctx.last_output = assistant_text
            if ctx is main_ctx:
                pass  # tracked via main_ctx.last_output

        model = ''
        if isinstance(outputs, dict):
            model = outputs.get('model', '') or ''
        if not model:
            model = inputs.get('model', '') or ''
        if model:
            self._model_name = model

        # Retroactively attach tool results to previous generation
        self._attach_tool_results_for(ctx, messages)

        # Build generation input: faithfully record the COMPLETE request that was
        # sent to the LLM for this call — the full message list (system + all
        # history + new messages) and the tools definition — not just the delta.
        # Each generation therefore stands on its own: e.g. Claude Code's internal
        # "generate a title" sub-call records ITS own (title) system prompt and its
        # (empty) tools, while main-agent calls record the main system prompt and
        # the full tool set. This avoids one call's system prompt/tools being
        # mistaken for the whole trace's.
        full_messages = [_truncate_message(m, self._max_content_chars, self._max_tool_chars) for m in messages]
        delta_count = len(messages) - ctx.prev_message_count
        ctx.prev_message_count = len(messages)

        gen_input: Any = {
            'messages': full_messages,
            'total_messages': len(messages),
            'new_messages': delta_count,
        }
        raw_tools_input = inputs.get('kwargs', {}).get('tools') or []
        if raw_tools_input:
            # Record the full tool definitions (names + descriptions + schemas) as-is.
            gen_input['tools'] = raw_tools_input
        # Faithfully record signatures present in the input history so users can
        # diff against the previous turn's output_signatures and detect whether
        # thinking signatures round-trip.
        if input_signatures:
            gen_input['thinking_signatures'] = input_signatures
        # Upload the full input-history reasoning_items (MR !1267). message_index is the
        # absolute index in this request's messages[]; round-trip verification below uses
        # the same list.
        if input_reasoning_items:
            gen_input['input_reasoning_items'] = input_reasoning_items

        # Build generation output
        gen_output: dict[str, Any] = {}
        if thinking_text:
            gen_output['thinking'] = self._trunc_content(thinking_text)
        if assistant_text:
            gen_output['text'] = self._trunc_content(assistant_text)
        if output_signatures:
            gen_output['thinking_signatures'] = output_signatures
        if output_reasoning_items:
            gen_output['reasoning_items'] = output_reasoning_items
        provider_call = span.attributes.get('provider_call')
        if self._record_provider_call and isinstance(provider_call, dict):
            request = provider_call.get('request')
            if request is not None:
                gen_input['provider_request'] = _trunc_json(request, self._max_provider_call_chars)
            response = provider_call.get('response')
            if response is not None:
                gen_output['provider_response'] = _trunc_json(response, self._max_provider_call_chars)
        if tool_calls:
            gen_output['tool_calls'] = [
                {
                    'name': tc['name'],
                    'args': {
                        k: _trunc(str(v), 300)
                        for k, v in (tc['arguments'] if isinstance(tc['arguments'], dict) else {}).items()
                    },
                }
                for tc in tool_calls
            ]

        tool_names = [tc['name'] for tc in tool_calls]
        if tool_names:
            step_label = f'Step {ctx.step}: {", ".join(tool_names[:4])}{"..." if len(tool_names) > 4 else ""}'
        else:
            step_label = f'Step {ctx.step}'

        lf_usage = _convert_usage(usage)
        if lf_usage.get('total', 0) > 0:
            self._total_usage['input'] += lf_usage.get('input', 0)
            self._total_usage['output'] += lf_usage.get('output', 0)
            self._total_usage['total'] += lf_usage.get('total', 0)
            self._total_usage['cached_tokens'] += lf_usage.get('cached_tokens', 0)

        gen_metadata: dict[str, Any] = {
            **self._turn_metadata_fields(ctx.step, self._global_step),
            'step': ctx.step,
            'global_step': self._global_step,
            'tool_count': len(tool_calls),
            'tool_names': tool_names,
            'parallel': len(tool_calls) > 1,
            'source': 'realtime',
        }
        if lf_usage.get('cached_tokens', 0):
            gen_metadata['cached_tokens'] = lf_usage['cached_tokens']
        if span.start_time_unix_nano and span.end_time_unix_nano:
            duration_s = (span.end_time_unix_nano - span.start_time_unix_nano) / 1e9
            gen_metadata['duration_s'] = round(duration_s, 2)

        # Compare prev turn's output signatures against this turn's input
        # signatures and surface the result for the langfuse reader. Each
        # entry says whether the prior signature was found and at which
        # message_index it landed (so a mis-mapping is visible).
        if ctx.prev_output_signatures:
            input_sig_to_msg = {entry['signature']: entry for entry in input_signatures}
            roundtrip = []
            for prev in ctx.prev_output_signatures:
                hit = input_sig_to_msg.get(prev.get('signature'))
                roundtrip.append({
                    'tool_call_id': prev.get('tool_call_id'),
                    'found': hit is not None,
                    'landed_at_message_index': hit.get('message_index') if hit else None,
                })
            gen_metadata['signature_round_trip'] = roundtrip
            gen_metadata['signature_round_trip_rate'] = f'{sum(1 for r in roundtrip if r["found"])}/{len(roundtrip)}'

        if ctx.prev_output_reasoning_items:
            input_reasoning_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for entry in input_reasoning_items:
                key = _reasoning_match_key(entry)
                if key is not None:
                    input_reasoning_by_key.setdefault(key, []).append(entry)

            roundtrip = []
            for prev in ctx.prev_output_reasoning_items:
                key = _reasoning_match_key(prev)
                hits = input_reasoning_by_key.get(key, []) if key is not None else []
                hit = hits.pop(0) if hits else None
                roundtrip.append({
                    'id': prev.get('id'),
                    'match_on': key[0] if key else None,
                    'found': hit is not None,
                    'tool_call_ids': prev.get('tool_call_ids') or [],
                    'landed_tool_call_ids': hit.get('tool_call_ids') if hit else None,
                    'landed_at_message_index': hit.get('message_index') if hit else None,
                    'landed_at_reasoning_index': hit.get('reasoning_index') if hit else None,
                })
            gen_metadata['reasoning_round_trip'] = roundtrip
            gen_metadata['reasoning_round_trip_rate'] = f'{sum(1 for r in roundtrip if r["found"])}/{len(roundtrip)}'

        if self._record_provider_call and isinstance(provider_call, dict):
            # Endpoint, per-attempt status and the provider's error body, i.e. the
            # part that explains retries and failures.
            gen_metadata['provider_call'] = {
                key: value for key, value in provider_call.items() if key not in ('request', 'response')
            }

        gen = ctx.parent_span.start_observation(
            name=step_label,
            as_type='generation',
            model=self._model_name,
            model_parameters=self._model_parameters if self._model_parameters else None,
            input=gen_input,
            output=gen_output,
            usage_details=lf_usage if lf_usage.get('total', 0) > 0 else None,
            metadata=gen_metadata,
        )
        gen.end()

        ctx.prev_generation = gen
        ctx.prev_tool_calls = tool_calls
        ctx.prev_turn_metadata = self._turn_metadata_fields(ctx.step, self._global_step)
        ctx.prev_output_signatures = output_signatures
        ctx.prev_output_reasoning_items = output_reasoning_items

        tc_info = f', tools: {tool_names}' if tool_names else ''
        level = 'sub' if ctx is not main_ctx else 'main'
        logger.debug('Langfuse [%s] %s: %d chars%s', level, step_label, len(assistant_text), tc_info)

        if self._flush_on_finalize and self._global_step % 5 == 0 and self._langfuse:
            with contextlib.suppress(Exception):
                self._langfuse.flush()

    def _attach_tool_results_for(
        self,
        ctx: _AgentContext,
        messages: list[dict[str, Any]],
    ) -> None:
        """Attach tool results from messages to previous generation's tool spans."""
        if not ctx.prev_generation or not ctx.prev_tool_calls:
            return

        result_map: dict[str, str] = {}
        for m in messages:
            if m.get('role') == 'tool':
                result_map[m.get('tool_call_id', '')] = _extract_text_content(m.get('content', ''))

        for tc in ctx.prev_tool_calls:
            tc_id = tc.get('id', '')
            result = result_map.get(tc_id, '')
            args = tc.get('arguments', {})
            tool_input = (
                {k: self._trunc_tool(v) for k, v in args.items()}
                if isinstance(args, dict)
                else {'raw': self._trunc_tool(args)}
            )

            tool_span = ctx.prev_generation.start_observation(
                name=f'Tool: {tc["name"]}',
                as_type='tool',
                input=tool_input,
                output=self._trunc_tool(result),
                metadata={
                    **ctx.prev_turn_metadata,
                    'tool_call_id': tc_id,
                    'tool_name': tc['name'],
                    'is_agent_call': tc['name'] in _AGENT_TOOL_NAMES,
                    'is_parallel': len(ctx.prev_tool_calls) > 1,
                    'source': 'realtime',
                },
            )
            tool_span.end()
        ctx.prev_generation = None
        ctx.prev_tool_calls = []
        ctx.prev_turn_metadata = {}

    def finalize(self) -> None:
        if not self._langfuse or not self._trace:
            return

        try:
            # Close any open subagent
            while len(self._ctx_stack) > 1:
                self._exit_subagent([])

            main_ctx = self._ctx_stack[0] if self._ctx_stack else None
            if not main_ctx:
                return

            # Attach remaining tool results
            if main_ctx.prev_generation and main_ctx.prev_tool_calls:
                for tc in main_ctx.prev_tool_calls:
                    tool_input = (
                        {k: self._trunc_tool(v) for k, v in tc['arguments'].items()}
                        if isinstance(tc['arguments'], dict)
                        else {'raw': self._trunc_tool(tc['arguments'])}
                    )
                    tool_span = main_ctx.prev_generation.start_observation(
                        name=f'Tool: {tc["name"]}',
                        as_type='tool',
                        input=tool_input,
                        output='(agent finished before tool result)',
                        metadata={
                            **main_ctx.prev_turn_metadata,
                            'tool_call_id': tc.get('id', ''),
                            'tool_name': tc['name'],
                            'is_final': True,
                            'source': 'realtime',
                        },
                    )
                    tool_span.end()

            # If last step was pure text, create Final Response gen
            if main_ctx.last_output and not main_ctx.prev_tool_calls:
                final_input: Any = {'note': 'Final assistant response (no tool calls)'}
                final_gen = self._agent_span.start_observation(
                    name='Final Response',
                    as_type='generation',
                    model=self._model_name,
                    model_parameters=self._model_parameters if self._model_parameters else None,
                    input=final_input,
                    output={'role': 'assistant', 'content': self._trunc_content(main_ctx.last_output)},
                    metadata={
                        **self._turn_metadata_fields(main_ctx.step, self._global_step),
                        'is_final': True,
                        'total_steps': main_ctx.step,
                        'source': 'realtime',
                    },
                )
                final_gen.end()

            # Dynamically add model_name to tags
            if self._model_name:
                clean_model = self._model_name.removeprefix('openai/')
                if clean_model not in self._extra_tags:
                    self._extra_tags.append(clean_model)

            cache_hit_rate = (
                round(self._total_usage['cached_tokens'] / self._total_usage['input'], 3)
                if self._total_usage.get('input', 0) > 0
                else 0
            )
            score_value = _extract_numeric_score(self._task_output)

            # Close agent span
            self._agent_span.update(
                output={'total_steps': main_ctx.step, 'model': self._model_name},
                metadata={
                    'total_steps': main_ctx.step,
                    'model': self._model_name,
                    'model_parameters': self._model_parameters,
                    'total_usage': self._total_usage,
                    'cache_hit_rate': cache_hit_rate,
                    'tools': self._tools_summary,
                    'source': 'langfuse_tracker',
                },
            )
            self._agent_span.end()

            # Close trace root span
            task_metadata = {'task_result': self._task_output} if self._task_output else {}
            if score_value is not None:
                task_metadata['score'] = score_value

            self._trace.update(
                output={
                    'total_steps': main_ctx.step,
                    'last_output': main_ctx.last_output[:2000] if main_ctx.last_output else '',
                    'total_usage': self._total_usage,
                },
                metadata={
                    **self._trace_metadata,
                    'total_steps': main_ctx.step,
                    'model': self._model_name,
                    'model_full': self._model_name,
                    'num_rounds': main_ctx.step,
                    'total_usage': self._total_usage,
                    'cache_hit_rate': cache_hit_rate,
                    **task_metadata,
                },
            )
            self._trace.end()

            if score_value is not None:
                try:
                    self._langfuse.create_score(
                        trace_id=self._trace_id,
                        name='episode_reward',
                        value=score_value,
                        comment=self._score_comment(),
                    )
                except Exception:
                    logger.warning('Failed to upload Langfuse score', exc_info=True)
            elif self._task_output and 'score' in self._task_output:
                logger.warning('Skipping non-numeric Langfuse score: %r', self._task_output.get('score'))

            if self._propagate_ctx:
                # __exit__ detaches an OTel context that was attached in a different
                # asyncio Task (see _suppressed_otel_detach_log). OpenTelemetry logs
                # the resulting cross-context ValueError itself and returns normally,
                # so contextlib.suppress cannot silence it -- filter the log instead.
                with contextlib.suppress(Exception), _suppressed_otel_detach_log():
                    self._propagate_ctx.__exit__(None, None, None)

            if self._flush_on_finalize:
                self._langfuse.flush()
            logger.info(
                'Langfuse tracker finalized: trace_id=%s, steps=%d, tokens=%s',
                self._trace_id,
                main_ctx.step,
                self._total_usage,
            )
        except Exception:
            logger.warning('Langfuse finalize failed', exc_info=True)

    def get_result(self) -> dict[str, Any]:
        trace_id = getattr(self, '_trace_id', None)
        project_id = _known_project_id(self._host, self._public_key, self._project_id)
        main_ctx = self._ctx_stack[0] if self._ctx_stack else None
        return {
            'trace_id': trace_id,
            'trace_url': _langfuse_trace_url(self._host, project_id, trace_id),
            'total_steps': main_ctx.step if main_ctx else 0,
            'model': self._model_name,
            'total_usage': self._total_usage,
        }
