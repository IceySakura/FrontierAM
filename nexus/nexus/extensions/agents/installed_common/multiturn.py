"""Shared multi-turn helpers for installed stream-json CLI agents."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import shlex
import subprocess
import time
import uuid

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import aiofiles
import aiofiles.os

from nexus.core import UserInputRequest
from nexus.extensions.agents.installed_common import (
    DEFAULT_CLI_STREAM_READER_LIMIT,
    DEFAULT_PROXY_API_KEY,
    build_runtime_path,
    call_runtime_agent,
    coerce_user_prompt,
    find_free_port,
    find_runtime_binary,
)
from nexus.extensions.agents.installed_common.stream_trace import (
    flush_stream_trace_events,
    record_stream_trace_event,
)
from nexus.utils.common import format_exception, get_process_returncode, terminate_process


logger = logging.getLogger(__name__)


def _short_hash(value: Any) -> str:
    """Return a short stable digest for correlating turn inputs/outputs in logs.

    Diagnoses multi-turn "off-by-one" artifact issues: if a turn's ``output_hash``
    equals the *previous* turn's, the response is stale (this turn returned the
    artifact that belonged to the earlier prompt).
    """
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if not text:
        return 'empty'
    return hashlib.sha1(text.encode('utf-8', errors='replace')).hexdigest()[:10]


InstallCli = Callable[..., Awaitable[Path]]
CommandBuilder = Callable[[str | Path, str, list[str] | None, bool, bool], list[str]]

_SESSIONS: dict[tuple[str, str], _StreamJsonSession] = {}
_ASK_USER_HOOK_CALLBACK_ID = 'nexus_ask_user_question'
_EXIT_PLAN_HOOK_CALLBACK_ID = 'nexus_exit_plan_mode'

# Host-side fallback wait for a Hook input provider, used when the agent input has
# no ``hook_input_timeout`` attribute and no per-Hook timeout is available.
_DEFAULT_HOST_INPUT_TIMEOUT = 240.0

# Turn-boundary stabilization (off-by-one guard). Before sending a new prompt into the
# long-lived stream-json session, we drain any events still queued from the previous turn
# so this turn cannot immediately consume a stale ``result`` and return the previous turn's
# artifact. We clear the queue and require it to stay empty for ``_TURN_DRAIN_STABLE_WINDOW``
# seconds (to also catch trailing events still in flight), bounded by ``_TURN_DRAIN_MAX_WAIT``.
_TURN_DRAIN_STABLE_WINDOW = 0.2
_TURN_DRAIN_POLL_INTERVAL = 0.05
_TURN_DRAIN_MAX_WAIT = 5.0

# Post-write turn-boundary window. ``stdin.drain()`` yields the event loop, and during that
# yield the stdout reader task can enqueue trailing events from the previous turn that were
# still in the pipe / parser at pre-write drain time. Any event arriving within this window
# after the prompt bytes were flushed is by definition still a leftover -- a stream-json CLI
# cannot produce a fresh reply for the just-written prompt in tens of milliseconds (a real
# LLM round-trip is far longer). Kept short so a genuinely fast turn cannot be misclassified.
_TURN_POST_WRITE_QUIET_WINDOW = 0.05

# Reader-injected failure sentinels. These are NOT ordinary turn artifacts and must never be
# silently discarded by the turn-boundary drain -- they signal the subprocess/reader died and
# must reach the read loop so it can raise.
_CONTROL_SENTINEL_TYPES = frozenset({
    '_eof',
    '_protocol_error',
    '_stdout_read_error',
    '_stderr_read_error',
    '_control_write_error',
})


def encode_user_message(prompt: str, *, session_id: str, parent_tool_use_id: str | None = None) -> bytes:
    """Return one JSONL user message for stream-json stdin."""
    content: str | list[dict[str, Any]]
    if parent_tool_use_id:
        content = [
            {
                'type': 'tool_result',
                'tool_use_id': parent_tool_use_id,
                'content': prompt,
                'is_error': False,
            }
        ]
    else:
        content = prompt
    payload = {
        'type': 'user',
        'message': {
            'role': 'user',
            'content': content,
        },
        'parent_tool_use_id': parent_tool_use_id,
        'session_id': session_id,
    }
    return (json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')


def parse_stream_line(line: bytes) -> dict[str, Any]:
    """Decode one stdout JSONL line from a stream-json CLI."""
    return json.loads(line.decode('utf-8', errors='replace'))


def extract_text_from_message(message: dict[str, Any]) -> str:
    """Extract human-readable text from a stream-json message event."""
    raw_message = message.get('message')
    if not isinstance(raw_message, dict):
        return ''

    content = raw_message.get('content')
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''

    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get('type') == 'text' and isinstance(item.get('text'), str):
            chunks.append(item['text'])
    return ''.join(chunks)


def summarize_turn_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact turn summary from raw stream-json events."""
    assistant_parts: list[str] = []
    result_event: dict[str, Any] | None = None
    session_id: str | None = None

    for event in events:
        event_type = event.get('type')
        if isinstance(event.get('session_id'), str):
            session_id = event['session_id']
        if event_type == 'assistant':
            text = extract_text_from_message(event)
            if text:
                assistant_parts.append(text)
        elif event_type == 'result':
            result_event = event
            if isinstance(event.get('session_id'), str):
                session_id = event['session_id']

    result_text = ''
    is_error = False
    if result_event:
        result_text = str(result_event.get('result') or '')
        is_error = bool(result_event.get('is_error'))

    assistant_text = ''.join(assistant_parts) or result_text
    return {
        'session_id': session_id,
        'assistant_text': assistant_text,
        'result_text': result_text,
        'is_error': is_error,
        'result_event': result_event or {},
    }


def normalize_stream_json_session_id(session_id: str | None, *, namespace: str) -> str:
    """Return a UUID session id accepted by OpenCC/Claude-style CLIs."""
    if not session_id:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(session_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f'nexus:{namespace}:{session_id}'))


def coerce_multiturn_prompt(prompt: Any) -> str | None:
    if prompt is None:
        return None
    text = coerce_user_prompt(prompt)
    return text if text.strip() else None


def build_deferred_tool_hint(agent_label: str, deferred_tool_use: dict[str, Any]) -> str | None:
    tool_name = deferred_tool_use.get('name')
    tool_input = deferred_tool_use.get('input')
    if not isinstance(tool_input, dict):
        tool_input = {}
    if tool_name == 'AskUserQuestion':
        questions = tool_input.get('questions')
        if not isinstance(questions, list):
            return f'{agent_label} asks for user input.'
        lines = [f'{agent_label} asks for user input.']
        for index, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                continue
            prompt = question.get('question')
            header = question.get('header')
            lines.append(f'{index}. {header + ": " if header else ""}{prompt or ""}')
        return '\n'.join(lines)
    if tool_name == 'ExitPlanMode':
        return f'{agent_label} wants to exit plan mode with this plan:\n' + str(tool_input.get('plan') or '')
    return None


async def build_user_input_request(
    *,
    agent_label: str,
    output: str,
    result: dict[str, Any],
    session_id: str,
    turn: int,
) -> UserInputRequest:
    result_event = result.get('result_event') if isinstance(result.get('result_event'), dict) else {}
    deferred_tool_use = result_event.get('deferred_tool_use') if isinstance(result_event, dict) else None
    interaction = result.get('interaction')
    if isinstance(interaction, dict):
        deferred_tool_use = {
            'id': interaction.get('interaction_id'),
            'name': interaction.get('tool_name'),
            'input': interaction.get('tool_input') if isinstance(interaction.get('tool_input'), dict) else {},
        }
        result_event = interaction
    raw_events = result.get('events')
    events: list[dict[str, Any]] = (
        [event for event in raw_events if isinstance(event, dict)] if isinstance(raw_events, list) else []
    )
    hint = output or None
    if isinstance(deferred_tool_use, dict):
        hint = build_deferred_tool_hint(agent_label, deferred_tool_use) or hint
    latest_screen = result.get('latest_screen')
    if not isinstance(latest_screen, str) or not latest_screen:
        latest_screen = None
    raw_recent_events = result.get('recent_events')
    recent_events = (
        [event for event in raw_recent_events if isinstance(event, dict)] if isinstance(raw_recent_events, list) else []
    )
    metadata: dict[str, Any] = {
        'turn': turn,
        'agent': agent_label,
        'latest_screen': latest_screen,
        'recent_events': recent_events,
    }
    if isinstance(interaction, dict):
        hook_timeout = interaction.get('timeout_seconds')
        if isinstance(hook_timeout, (int, float)) and math.isfinite(hook_timeout) and hook_timeout > 0:
            metadata['hook_timeout_seconds'] = float(hook_timeout)
    event_cursor = result.get('event_cursor')
    if isinstance(event_cursor, dict):
        metadata['event_cursor'] = event_cursor
    if latest_screen or recent_events:
        if os.environ.get('NEXUS_PTY_CONTEXT_DEBUG'):
            logger.debug(
                '[pty-user-context] session_id=%s turn=%s latest_screen_chars=%s recent_events=%s',
                session_id,
                turn,
                len(latest_screen) if latest_screen else 0,
                len(recent_events),
            )
        dump_path = os.environ.get('NEXUS_PTY_CONTEXT_DUMP_PATH')
        if dump_path:
            payload = {
                'session_id': session_id,
                'turn': turn,
                'hook': deferred_tool_use if isinstance(deferred_tool_use, dict) else None,
                'latest_screen': latest_screen,
                'recent_events': recent_events,
            }
            if isinstance(event_cursor, dict):
                payload['event_cursor'] = event_cursor
            dump_file = Path(dump_path).expanduser()
            await aiofiles.os.makedirs(dump_file.parent, exist_ok=True)
            async with aiofiles.open(dump_file, 'a', encoding='utf-8') as file:
                await file.write(json.dumps(payload, ensure_ascii=False) + '\n')
    return UserInputRequest(
        source='hook' if isinstance(deferred_tool_use, dict) else 'turn',
        hint=hint,
        session_id=session_id,
        message=result_event,
        events=events,
        hook=deferred_tool_use if isinstance(deferred_tool_use, dict) else None,
        metadata=metadata,
    )


def get_deferred_tool_use_id(result: dict[str, Any]) -> str | None:
    result_event = result.get('result_event')
    if not isinstance(result_event, dict):
        return None
    deferred_tool_use = result_event.get('deferred_tool_use')
    if not isinstance(deferred_tool_use, dict):
        return None
    tool_use_id = deferred_tool_use.get('id')
    return tool_use_id if isinstance(tool_use_id, str) else None


async def run_installed_multiturn(
    input_data: Any,
    *,
    agent_key: str,
    agent_label: str,
    session_namespace: str,
    session_turn_target: str | Callable[..., Any],
    close_session_target: str | Callable[..., Any],
    runtime_kwargs: dict[str, Any],
    respond_session_target: str | Callable[..., Any] | None = None,
    llm_channel: bool = False,
    openai_client_name: str = 'main',
    disable_invoke_wait_timeout: bool = False,
    close_failure_affects_result: bool = True,
) -> dict[str, Any]:
    """Run a Host-side loop, including optional input requests inside a turn."""
    from nexus.trace import RemoteTraceContext, Status, span_context

    session_id = normalize_stream_json_session_id(input_data.session_id, namespace=session_namespace)
    llm_channel_id = session_id if llm_channel else None
    prompt = coerce_multiturn_prompt(input_data.user_prompt)

    start_time = time.time()
    turns = 0
    success = True
    exit_code = 0
    ended_by_user = False
    reached_max_turns = False
    outputs: list[str] = []
    stderr_parts: list[str] = []
    events: list[dict[str, Any]] = []
    relay_frames: int | None = None
    status: str | None = None
    parent_tool_use_id: str | None = None
    session_started = False
    no_thinking_request_args = getattr(input_data, 'no_thinking_request_args', None)
    no_thinking_request_args_mode = cast(
        Literal['replace', 'inherit'],
        getattr(input_data, 'no_thinking_request_args_mode', 'replace'),
    )
    haiku_model = getattr(input_data, 'haiku_model', None)

    try:
        while prompt is not None and turns < input_data.max_turns:
            turn_index = turns + 1
            turn_started_at = time.time()
            logger.debug(
                '[multiturn] turn begin agent=%s session=%s turn=%s prompt_hash=%s parent_tool_use_id=%s',
                agent_key,
                session_id,
                turn_index,
                _short_hash(prompt),
                parent_tool_use_id,
            )
            outer_trace_context = RemoteTraceContext.from_current_span()
            with span_context(
                'agent.turn',
                {
                    'type': 'agent_turn',
                    'agent': agent_key,
                    'session_id': session_id,
                    'turn_index': turn_index,
                },
            ) as turn_span:
                result: dict[str, Any] = await call_runtime_agent(
                    session_turn_target,
                    no_thinking_request_args=no_thinking_request_args,
                    no_thinking_request_args_mode=no_thinking_request_args_mode,
                    haiku_model=haiku_model,
                    session_id=session_id,
                    prompt=prompt,
                    parent_tool_use_id=parent_tool_use_id,
                    runtime_trace_context=outer_trace_context,
                    channel_id=llm_channel_id,
                    persistent_channel=llm_channel,
                    openai_client_name=openai_client_name,
                    disable_invoke_wait_timeout=disable_invoke_wait_timeout,
                    **runtime_kwargs,
                )
                session_started = True
                if result.get('relay_frames') is not None:
                    relay_frames = (relay_frames or 0) + int(result['relay_frames'])
                turn_events = list(result.get('events') or [])
                while result.get('status') == 'needs_user_input':
                    if respond_session_target is None:
                        result = {
                            'success': False,
                            'status': 'interaction_error',
                            'stderr': f'{agent_label} requested input but no response target is configured',
                            'exit_code': -1,
                            'events': [],
                        }
                        break
                    interaction = result.get('interaction')
                    if not isinstance(interaction, dict) or not isinstance(interaction.get('interaction_id'), str):
                        result = {
                            'success': False,
                            'status': 'interaction_error',
                            'stderr': f'{agent_label} returned an invalid interaction request',
                            'exit_code': -1,
                            'events': [],
                        }
                        break

                    request = await build_user_input_request(
                        agent_label=agent_label,
                        output='',
                        result=result,
                        session_id=session_id,
                        turn=turn_index,
                    )
                    answer: str | None = None
                    try:
                        configured_input_timeout = getattr(
                            input_data, 'hook_input_timeout', _DEFAULT_HOST_INPUT_TIMEOUT
                        )
                        if configured_input_timeout is None:
                            configured_input_timeout = request.metadata.get(
                                'hook_timeout_seconds',
                                runtime_kwargs.get('hook_wait_timeout', _DEFAULT_HOST_INPUT_TIMEOUT),
                            )
                        input_timeout = float(configured_input_timeout)
                        answer = await asyncio.wait_for(input_data.get_user_input(request), timeout=input_timeout)
                    except asyncio.TimeoutError:
                        logger.warning(
                            '%s input provider timed out for session=%s turn=%s interaction=%s; using unattended fallback',
                            agent_label,
                            session_id,
                            turn_index,
                            interaction['interaction_id'],
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            '%s input provider failed for session=%s turn=%s interaction=%s: %s; '
                            'using unattended fallback',
                            agent_label,
                            session_id,
                            turn_index,
                            interaction['interaction_id'],
                            exc,
                        )

                    result = await call_runtime_agent(
                        respond_session_target,
                        no_thinking_request_args=no_thinking_request_args,
                        no_thinking_request_args_mode=no_thinking_request_args_mode,
                        haiku_model=haiku_model,
                        session_id=session_id,
                        interaction_id=interaction['interaction_id'],
                        answer=answer,
                        timeout=runtime_kwargs.get('timeout', 3600),
                        runtime_trace_context=outer_trace_context,
                        channel_id=llm_channel_id,
                        persistent_channel=llm_channel,
                        openai_client_name=openai_client_name,
                        disable_invoke_wait_timeout=disable_invoke_wait_timeout,
                    )
                    turn_events.extend(result.get('events') or [])
                if not result.get('success', False):
                    message = str(result.get('stderr') or f'{agent_label} turn failed')
                    turn_span.update(status=Status(code='ERROR', message=message))

            turns += 1
            success = success and bool(result.get('success', False))
            exit_code = int(result.get('exit_code') or 0)
            output = str(result.get('output') or '')
            stderr = str(result.get('stderr') or '')
            status = str(result.get('status')) if result.get('status') is not None else None
            if output:
                outputs.append(output)
            if stderr:
                stderr_parts.append(stderr)
            events.extend(turn_events)

            logger.info(
                '[multiturn] turn end agent=%s session=%s turn=%s prompt_hash=%s output_hash=%s success=%s '
                'status=%s turn_elapsed=%.2fs result_exec_time=%.2fs turn_events=%s',
                agent_key,
                session_id,
                turns,
                _short_hash(prompt),
                _short_hash(output),
                result.get('success', False),
                status,
                time.time() - turn_started_at,
                float(result.get('execution_time') or 0.0),
                len(turn_events),
            )

            if not result.get('success', False):
                break
            if turns >= input_data.max_turns:
                reached_max_turns = True
                break

            next_parent_tool_use_id = get_deferred_tool_use_id(result)
            prompt = coerce_multiturn_prompt(
                await input_data.get_user_input(
                    await build_user_input_request(
                        agent_label=agent_label,
                        output=output,
                        result=result,
                        session_id=session_id,
                        turn=turns,
                    )
                )
            )
            parent_tool_use_id = next_parent_tool_use_id if prompt is not None else None
            ended_by_user = prompt is None
    finally:
        if input_data.close_on_finish and session_started:
            try:
                close_result = await call_runtime_agent(
                    close_session_target,
                    no_thinking_request_args=no_thinking_request_args,
                    no_thinking_request_args_mode=no_thinking_request_args_mode,
                    haiku_model=haiku_model,
                    session_id=session_id,
                )
                close_stderr = close_result.get('stderr') or ''
                if close_stderr:
                    stderr_parts.append(close_stderr)
            except Exception as exc:
                if close_failure_affects_result:
                    success = False
                stderr_parts.append(f'Failed to close {agent_label} session {session_id}: {exc}')
        if input_data.close_on_finish and llm_channel_id:
            from nexus.core import require_context
            from nexus.extensions.agents.installed_common import unbind_llm_channel

            runtime = await require_context().get_runtime()
            unbind_llm_channel(runtime, llm_channel_id)

    return {
        'success': success,
        'output': '\n'.join(outputs),
        'stderr': '\n'.join(part for part in stderr_parts if part),
        'exit_code': exit_code,
        'execution_time': time.time() - start_time,
        'session_id': session_id,
        'turns': turns,
        'events': events,
        'ended_by_user': ended_by_user,
        'reached_max_turns': reached_max_turns,
        'status': status,
        **({'relay_frames': relay_frames} if relay_frames is not None else {}),
    }


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _resolve_config_path(raw: str) -> Path:
    return Path(os.path.expanduser(raw)).resolve()


def _apply_json_patch(raw_path: str, patch: dict[str, Any]) -> None:
    path = _resolve_config_path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text()) or {}
        except Exception as exc:
            logger.warning('Failed to parse existing %s: %s; overwriting.', path, exc)
            existing = {}
    path.write_text(json.dumps(_deep_merge(existing, patch), indent=2, ensure_ascii=False))
    logger.debug('Patched %s', path)


async def _run_pre_start_script(
    script: dict[str, Any],
    working_dir: str,
    base_env: dict[str, str],
) -> None:
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    interpreter = script.get('interpreter', 'bash')
    code = script.get('code', '')
    timeout = float(script.get('timeout', 300.0))
    cwd = script.get('cwd') or working_dir
    fail_on_error = bool(script.get('fail_on_error', True))

    if interpreter == 'bash':
        shell = find_runtime_binary('bash') or find_runtime_binary('sh') or 'bash'
        cmd = f'{shlex.quote(shell)} -c {shlex.quote(code)}'
    elif interpreter == 'python':
        py = find_runtime_binary('python') or find_runtime_binary('python3') or 'python3'
        cmd = f'{shlex.quote(py)} -c {shlex.quote(code)}'
    else:
        raise ValueError(f'Unsupported pre-start interpreter: {interpreter}')

    script_env = dict(base_env)
    if script.get('env'):
        script_env.update(script['env'])

    logger.debug('Running pre-start (%s) in %s', interpreter, cwd)
    result = await run_command(RunCommandRequest(command=cmd, cwd=cwd, timeout=timeout, extra_env=script_env))
    if result.stdout:
        logger.debug('[pre-start stdout]\n%s', result.stdout)
    if result.stderr:
        logger.debug('[pre-start stderr]\n%s', result.stderr)

    ok = result.status == CommandRunStatus.Finished and (result.return_code or 0) == 0
    if not ok and fail_on_error:
        raise RuntimeError(
            f'Pre-start script failed (status={result.status}, rc={result.return_code}): '
            f'{result.stderr or result.stdout or ""}'
        )


@dataclass
class _StreamJsonSession:
    agent_key: str
    agent_label: str
    session_id: str
    proc: asyncio.subprocess.Process
    proxy: Any
    cwd: str
    started_at: float
    stdout_task: asyncio.Task | None = None
    stderr_task: asyncio.Task | None = None
    events: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    control_responses: dict[str, asyncio.Future[dict[str, Any]]] = field(default_factory=dict)
    stderr_lines: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    record_trace_events: bool = True

    @property
    def is_running(self) -> bool:
        return self.proc.returncode is None

    async def _drain_stale_events(self) -> tuple[int, bool]:
        """Drain leftover events queued from a previous turn before a new prompt is sent.

        The stream-json subprocess feeds every stdout event into ``self.events`` via a
        background reader. When a turn returns on its ``result`` event, the subprocess may
        still be flushing trailing events for that turn. Left in the queue, the *next*
        ``send_user_message`` would immediately consume them and return the previous turn's
        output (the observed multi-turn off-by-one). Because this runs *before* the new
        prompt is written, anything currently queued is necessarily a leftover, so it is safe
        to discard. We repeatedly clear the queue and require it to stay empty for
        ``_TURN_DRAIN_STABLE_WINDOW`` seconds (to also catch trailing events still in flight),
        bounded by ``_TURN_DRAIN_MAX_WAIT``.

        Returns ``(drained_count, saw_result)`` for diagnostics.
        """
        drained = 0
        saw_result = False
        deadline = time.time() + _TURN_DRAIN_MAX_WAIT
        quiet_since: float | None = None
        while True:
            got_any = False
            while True:
                try:
                    event = self.events.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if event.get('type') in _CONTROL_SENTINEL_TYPES:
                    # A failure sentinel must not be dropped: re-queue it so the read loop
                    # surfaces it, and stop draining immediately.
                    self.events.put_nowait(event)
                    return drained, saw_result
                drained += 1
                got_any = True
                if event.get('type') == 'result':
                    saw_result = True
            now = time.time()
            if got_any:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= _TURN_DRAIN_STABLE_WINDOW:
                break
            if now >= deadline:
                break
            await asyncio.sleep(_TURN_DRAIN_POLL_INTERVAL)
        return drained, saw_result

    async def _drain_post_write_stale_events(self) -> tuple[int, bool]:
        """Drop events that reach the queue *right after* the new prompt was flushed.

        Closes the write→drain race described in the multi-turn CR: at pre-write time
        ``self.events`` may look empty, yet ``await stdin.drain()`` immediately yields the
        event loop, giving the stdout reader task time to enqueue a trailing event from the
        *previous* turn (which was sitting in the OS pipe or the parser). Without this
        second drain, the read loop below would consume that stale event and — if it is a
        ``result`` — return the previous turn's artifact as this turn's fresh reply.

        A stream-json CLI cannot produce a genuine reply for the just-written prompt within
        this short window (LLM round-trip is orders of magnitude longer), so any event we
        see here is by definition stale and can be dropped. Reader-failure sentinels are
        never dropped: we re-queue them so the read loop can raise.

        Implementation: sleep one short quiet window (``_TURN_POST_WRITE_QUIET_WINDOW``) to
        let the stdout reader flush anything that was in flight during ``drain()``, then
        drain non-blockingly. The window is bounded and unconditional (single sleep, no
        loop) so it can never mask a fresh reply — the CLI cannot beat this window.

        Returns ``(drained_count, saw_result)`` for diagnostics.
        """
        # Yield long enough for the reader task to push any event that was being parsed while
        # stdin.drain() ran. Kept intentionally small — orders of magnitude below the fastest
        # plausible fresh LLM reply — so healthy turns pay only this constant.
        await asyncio.sleep(_TURN_POST_WRITE_QUIET_WINDOW)

        drained = 0
        saw_result = False
        while True:
            try:
                event = self.events.get_nowait()
            except asyncio.QueueEmpty:
                break
            if event.get('type') in _CONTROL_SENTINEL_TYPES:
                # Never drop a sentinel: re-queue and stop.
                self.events.put_nowait(event)
                break
            drained += 1
            if event.get('type') == 'result':
                saw_result = True
        return drained, saw_result

    async def send_user_message(
        self,
        prompt: str,
        *,
        timeout: float,
        parent_tool_use_id: str | None = None,
    ) -> dict[str, Any]:
        async with self.lock:
            if not self.is_running:
                raise RuntimeError(f'{self.agent_label} session {self.session_id} is not running')
            if self.proc.stdin is None:
                raise RuntimeError(f'{self.agent_label} session {self.session_id} has no stdin')

            # Off-by-one diagnostics: events still queued *before* we send this prompt are
            # stale leftovers from the previous turn. If this is > 0 we may immediately
            # consume a stale `result` below and return the previous turn's artifact.
            stale_events = self.events.qsize()
            prompt_hash = _short_hash(prompt)
            logger.debug(
                '[multiturn-turn-boundary] send session=%s prompt_hash=%s parent_tool_use_id=%s '
                'stale_events_in_queue=%s',
                self.session_id,
                prompt_hash,
                parent_tool_use_id,
                stale_events,
            )
            if stale_events:
                # Turn-boundary stabilization (off-by-one fix): drain leftover events from the
                # previous turn BEFORE writing this prompt, so this turn cannot immediately
                # consume a stale ``result`` and return the previous turn's artifact.
                drained, saw_result = await self._drain_stale_events()
                logger.warning(
                    '[multiturn-turn-boundary] DRAINED session=%s prompt_hash=%s drained=%s saw_result=%s '
                    'leftover event(s) from a previous turn before sending new prompt (off-by-one guard)',
                    self.session_id,
                    prompt_hash,
                    drained,
                    saw_result,
                )

            self.proc.stdin.write(
                encode_user_message(prompt, session_id=self.session_id, parent_tool_use_id=parent_tool_use_id)
            )
            await self.proc.stdin.drain()

            # Post-write turn-boundary drain. ``stdin.drain()`` yields the event loop, and a
            # trailing event from the previous turn may only reach ``self.events`` during
            # that yield -- exactly the race where the pre-write ``qsize() == 0`` check
            # would falsely conclude the queue was clean. Anything that shows up in this
            # short window before the CLI could plausibly generate a fresh reply is stale by
            # definition; drop it. See ``_drain_post_write_stale_events`` for the argument.
            post_drained, post_saw_result = await self._drain_post_write_stale_events()
            if post_drained:
                logger.warning(
                    '[multiturn-turn-boundary] POST-WRITE-DRAINED session=%s prompt_hash=%s drained=%s '
                    'saw_result=%s leftover event(s) arrived during stdin.drain() after new prompt was '
                    'flushed (off-by-one guard, write→drain race)',
                    self.session_id,
                    prompt_hash,
                    post_drained,
                    post_saw_result,
                )

            start = time.time()
            first_event_at: float | None = None
            events: list[dict[str, Any]] = []
            while True:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    raise TimeoutError(f'{self.agent_label} session {self.session_id} timed out waiting for result')

                event = await asyncio.wait_for(self.events.get(), timeout=remaining)
                event_type = event.get('type')
                if event_type == '_eof':
                    stderr = '\n'.join(self.stderr_lines)
                    raise RuntimeError(
                        f'{self.agent_label} session {self.session_id} exited before result '
                        f'(return_code={event.get("return_code")}): {stderr}'
                    )
                if event_type == '_protocol_error':
                    stderr = '\n'.join(self.stderr_lines)
                    raise RuntimeError(
                        f'{self.agent_label} session {self.session_id} emitted invalid stream-json: '
                        f'{event.get("error")}: {stderr}'
                    )
                if event_type in {'_stdout_read_error', '_stderr_read_error', '_control_write_error'}:
                    stderr = '\n'.join(self.stderr_lines)
                    detail = str(event.get('error') or 'unknown error')
                    error_name = {
                        '_stdout_read_error': 'stdout reader',
                        '_stderr_read_error': 'stderr reader',
                        '_control_write_error': 'control response write',
                    }[event_type]
                    raise RuntimeError(
                        f'{self.agent_label} session {self.session_id} {error_name} failed: {detail}'
                        + (f': {stderr}' if stderr else '')
                    )

                if first_event_at is None:
                    first_event_at = time.time()
                events.append(event)
                if event_type == 'result':
                    summary = summarize_turn_events(events)
                    elapsed = time.time() - start
                    first_event_latency = (first_event_at if first_event_at is not None else start) - start
                    output_hash = _short_hash(summary['assistant_text'])
                    logger.info(
                        '[multiturn-turn-boundary] result session=%s prompt_hash=%s result_session_id=%s '
                        'output_hash=%s events_consumed=%s elapsed=%.2fs first_event_latency=%.3fs is_error=%s',
                        self.session_id,
                        prompt_hash,
                        summary['session_id'],
                        output_hash,
                        len(events),
                        elapsed,
                        first_event_latency,
                        summary['is_error'],
                    )
                    if first_event_latency < 0.5:
                        logger.warning(
                            '[multiturn-turn-boundary] FAST-RETURN session=%s prompt_hash=%s output_hash=%s '
                            'first event arrived in %.3fs (<0.5s) — likely a stale leftover result rather than a '
                            'fresh generation for this prompt (off-by-one suspect)',
                            self.session_id,
                            prompt_hash,
                            output_hash,
                            first_event_latency,
                        )
                    return {
                        'success': not summary['is_error'],
                        'session_id': summary['session_id'] or self.session_id,
                        'output': summary['assistant_text'],
                        'stderr': '\n'.join(self.stderr_lines),
                        'exit_code': self.proc.returncode or 0,
                        'execution_time': elapsed,
                        'events': events,
                        'result_event': summary['result_event'],
                    }

    async def send_control_request(self, request: dict[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
        if self.proc.stdin is None:
            raise RuntimeError(f'{self.agent_label} session {self.session_id} has no stdin')

        request_id = f'req_{uuid.uuid4().hex}'
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self.control_responses[request_id] = future
        payload = {
            'type': 'control_request',
            'request_id': request_id,
            'request': request,
        }
        self.proc.stdin.write((json_dumps(payload) + '\n').encode('utf-8'))
        await self.proc.stdin.drain()

        try:
            event = await asyncio.wait_for(future, timeout=timeout)
        finally:
            self.control_responses.pop(request_id, None)

        response = event.get('response') or {}
        if response.get('subtype') == 'error':
            raise RuntimeError(str(response.get('error') or f'{self.agent_label} control request failed'))
        data = response.get('response')
        return data if isinstance(data, dict) else {}

    async def initialize(self, *, hooks: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
        return await self.send_control_request({'subtype': 'initialize', 'hooks': hooks}, timeout=timeout)

    async def close(self) -> dict[str, Any]:
        if self.proc.stdin:
            with contextlib.suppress(Exception):
                self.proc.stdin.close()
                await self.proc.stdin.wait_closed()

        if self.proc.returncode is None:
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self.proc.kill()
                    with contextlib.suppress(Exception):
                        await self.proc.wait()

        for task in (self.stdout_task, self.stderr_task):
            if task is None:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        with contextlib.suppress(Exception):
            await self.proxy.stop()

        return {
            'success': True,
            'session_id': self.session_id,
            'exit_code': self.proc.returncode if self.proc.returncode is not None else 0,
            'stderr': '\n'.join(self.stderr_lines),
        }


async def _read_stdout(session: _StreamJsonSession) -> None:
    assert session.proc.stdout is not None
    try:
        while True:
            line = await session.proc.stdout.readline()
            if not line:
                break
            text = line.decode('utf-8', errors='replace').rstrip('\n')
            logger.debug('[%s:%s] %s', session.agent_key, session.session_id, text)
            try:
                event = parse_stream_line(line)
            except Exception as exc:
                await session.events.put({'type': '_protocol_error', 'error': str(exc), 'line': text})
                continue

            event_type = event.get('type')
            if event_type == 'control_response':
                response = event.get('response') or {}
                request_id = response.get('request_id')
                if not isinstance(request_id, str):
                    continue
                future = session.control_responses.get(request_id)
                if future is not None and not future.done():
                    future.set_result(event)
                continue
            if event_type == 'control_request':
                asyncio.create_task(_handle_control_request(session, event))
                continue
            if event_type in {'control_cancel_request', 'transcript_mirror'}:
                continue

            _record_trace_event(session, event, enabled=session.record_trace_events)
            await session.events.put(event)
    except Exception as exc:
        error = format_exception(exc)
        session.stderr_lines.append(f'stdout reader failed: {error}')
        logger.exception('[%s:%s] stdout reader failed', session.agent_key, session.session_id)
        await session.events.put({'type': '_stdout_read_error', 'error': error})
        await terminate_process(session.proc)
    finally:
        flush_stream_trace_events(session, agent_key=session.agent_key)
        return_code = await get_process_returncode(session.proc)
        await session.events.put({'type': '_eof', 'return_code': return_code})


async def _read_stderr(session: _StreamJsonSession) -> None:
    assert session.proc.stderr is not None
    try:
        while True:
            line = await session.proc.stderr.readline()
            if not line:
                break
            text = line.decode('utf-8', errors='replace').rstrip('\n')
            session.stderr_lines.append(text)
            logger.warning('[%s:%s:err] %s', session.agent_key, session.session_id, text)
    except Exception as exc:
        error = format_exception(exc)
        session.stderr_lines.append(f'stderr reader failed: {error}')
        logger.exception('[%s:%s:err] stderr reader failed', session.agent_key, session.session_id)
        await session.events.put({'type': '_stderr_read_error', 'error': error})
        await terminate_process(session.proc)


def _record_trace_event(session: _StreamJsonSession, event: dict[str, Any], *, enabled: bool = True) -> None:
    record_stream_trace_event(session, event, agent_key=session.agent_key, enabled=enabled)


async def _handle_control_request(session: _StreamJsonSession, event: dict[str, Any]) -> None:
    if session.proc.stdin is None:
        return

    request_id = event.get('request_id')
    request = event.get('request') or {}
    subtype = request.get('subtype')
    try:
        if subtype == 'can_use_tool':
            response_data = await _handle_can_use_tool(request)
            response = _control_success(request_id, response_data)
        elif subtype == 'hook_callback':
            response = _control_success(request_id, await _handle_hook_callback(session, request))
        else:
            response = _control_error(request_id, f'Unsupported control request subtype: {subtype}')
    except Exception as exc:
        logger.exception('%s control request failed: %s', session.agent_label, exc)
        response = _control_error(request_id, str(exc))
    try:
        session.proc.stdin.write((json_dumps(response) + '\n').encode('utf-8'))
        await session.proc.stdin.drain()
    except Exception as exc:
        error = format_exception(exc)
        session.stderr_lines.append(f'control response write failed: {error}')
        logger.exception('%s control response write failed: %s', session.agent_label, exc)
        await session.events.put({'type': '_control_write_error', 'error': error})
        await terminate_process(session.proc)


def _control_success(request_id: Any, response_data: dict[str, Any]) -> dict[str, Any]:
    return {
        'type': 'control_response',
        'response': {
            'subtype': 'success',
            'request_id': request_id,
            'response': response_data,
        },
    }


def _control_error(request_id: Any, error: str) -> dict[str, Any]:
    return {
        'type': 'control_response',
        'response': {
            'subtype': 'error',
            'request_id': request_id,
            'error': error,
        },
    }


async def _handle_can_use_tool(request: dict[str, Any]) -> dict[str, Any]:
    raw_tool_input = request.get('input')
    tool_input: dict[str, Any] = raw_tool_input if isinstance(raw_tool_input, dict) else {}
    return {
        'behavior': 'allow',
        'updatedInput': tool_input,
    }


def _build_hook_output() -> dict[str, Any]:
    return {
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'defer',
            'permissionDecisionReason': 'Nexus will collect the user response and continue the session.',
        }
    }


def _build_user_input_hooks() -> dict[str, Any]:
    return {
        'PreToolUse': [
            {
                'matcher': 'ExitPlanMode',
                'hookCallbackIds': [_EXIT_PLAN_HOOK_CALLBACK_ID],
            },
            {
                'matcher': 'AskUserQuestion',
                'hookCallbackIds': [_ASK_USER_HOOK_CALLBACK_ID],
            },
        ],
    }


async def _handle_hook_callback(_session: _StreamJsonSession, _request: dict[str, Any]) -> dict[str, Any]:
    return _build_hook_output()


def _strip_stream_json_conflicts(args: list[str] | None) -> list[str]:
    result: list[str] = []
    raw = list(args or [])
    value_options = {'--input-format', '--output-format', '--session-id'}
    flags = {
        '--dangerously-skip-permissions',
        '--include-hook-events',
        '--include-partial-messages',
        '--print',
        '--verbose',
    }

    i = 0
    while i < len(raw):
        arg = raw[i]
        if arg in value_options:
            i += 2
            continue
        if any(arg.startswith(option + '=') for option in value_options):
            i += 1
            continue
        if arg in flags or any(arg.startswith(flag + '=') for flag in flags):
            i += 1
            continue
        if arg == '-p':
            if i + 1 < len(raw) and not raw[i + 1].startswith('-'):
                i += 2
            else:
                i += 1
            continue
        result.append(arg)
        i += 1
    return result


def build_stream_json_command(
    binary: str | Path,
    session_id: str,
    extra_cli_args: list[str] | None,
    include_partial_messages: bool,
    include_hook_events: bool,
) -> list[str]:
    extra_args = _strip_stream_json_conflicts(extra_cli_args)
    resume_or_continue = any(
        arg in {'--resume', '-r', '--continue', '-c'} or arg.startswith('--resume=') for arg in extra_args
    )
    fork_session = any(arg == '--fork-session' for arg in extra_args)

    cmd = [
        str(binary),
        '--print',
        '--output-format',
        'stream-json',
        '--verbose',
    ]
    if not resume_or_continue or fork_session:
        cmd.extend(['--session-id', session_id])
    cmd.append('--dangerously-skip-permissions')
    if include_partial_messages:
        cmd.append('--include-partial-messages')
    if include_hook_events:
        cmd.append('--include-hook-events')
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(['--input-format', 'stream-json'])
    return cmd


async def _start_session(
    *,
    agent_key: str,
    agent_label: str,
    session_id: str,
    working_dir: str,
    timeout: float,
    install_cli: InstallCli,
    install_dir: str,
    install_kwargs: dict[str, Any] | None,
    path_prepend: list[str] | None,
    extra_env: dict[str, str] | None,
    extra_cli_args: list[str] | None,
    config_file_patches: list[dict[str, Any]] | None,
    pre_start_scripts: list[dict[str, Any]] | None,
    conversion_log_level: str,
    thinking_preset: str,
    include_partial_messages: bool,
    include_hook_events: bool,
    enable_user_input_hooks: bool,
    record_trace_events: bool,
    command_builder: CommandBuilder,
) -> _StreamJsonSession:
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy
    from nexus.utils.shell import get_os_env_without_ld_library_path

    rt = require_runtime_control()
    binary = await install_cli(install_dir=install_dir, **(install_kwargs or {}))

    proxy_port = find_free_port()
    proxy = AnthropicProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        conversion_log_level=conversion_log_level,
        thinking_config=thinking_preset,
    )
    await proxy.start()

    session: _StreamJsonSession | None = None
    try:
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        home = os.environ.get('HOME', '/root')
        launch_env: dict[str, str] = {
            'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{proxy_port}',
            'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
            'HOME': home,
            'PATH': build_runtime_path(*(path_prepend or [install_dir])),
            'IS_SANDBOX': '1',
            'CLAUDE_CODE_ENTRYPOINT': 'nexus-stream-json',
            'PWD': str(work_path),
        }
        if extra_env:
            launch_env.update(extra_env)

        for patch in config_file_patches or []:
            _apply_json_patch(patch['path'], patch.get('content') or {})

        for script in pre_start_scripts or []:
            await _run_pre_start_script(script, working_dir, launch_env)

        base_env = get_os_env_without_ld_library_path()
        base_env['BASH_ENV'] = '~/.bashrc'
        merged_env = {**base_env, **launch_env}
        cmd = command_builder(
            binary,
            session_id,
            extra_cli_args,
            include_partial_messages,
            include_hook_events,
        )
        logger.debug('Starting %s session %s: %s', agent_label, session_id, ' '.join(shlex.quote(c) for c in cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_dir,
            env=merged_env,
            limit=DEFAULT_CLI_STREAM_READER_LIMIT,
        )
        session = _StreamJsonSession(
            agent_key=agent_key,
            agent_label=agent_label,
            session_id=session_id,
            proc=proc,
            proxy=proxy,
            cwd=working_dir,
            started_at=time.time(),
            record_trace_events=record_trace_events,
        )
        session.stdout_task = asyncio.create_task(_read_stdout(session))
        session.stderr_task = asyncio.create_task(_read_stderr(session))
        if enable_user_input_hooks:
            await session.initialize(
                hooks=_build_user_input_hooks(),
                timeout=min(max(timeout, 1.0), 60.0),
            )
        return session
    except Exception:
        if session is not None:
            with contextlib.suppress(Exception):
                await session.close()
        else:
            with contextlib.suppress(Exception):
                await proxy.stop()
        raise


async def run_installed_cli_session_turn(
    *,
    agent_key: str,
    agent_label: str,
    session_id: str,
    prompt: str,
    parent_tool_use_id: str | None = None,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    install_cli: InstallCli,
    install_dir: str,
    install_kwargs: dict[str, Any] | None = None,
    path_prepend: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    config_file_patches: list[dict[str, Any]] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    conversion_log_level: str = 'INFO',
    thinking_preset: str = 'hunyuan',
    include_partial_messages: bool = False,
    include_hook_events: bool = False,
    enable_user_input_hooks: bool = False,
    record_trace_events: bool = True,
    command_builder: CommandBuilder = build_stream_json_command,
) -> dict[str, Any]:
    """Start or reuse a stream-json CLI session and run one user turn."""
    start_time = time.time()
    session_key = (agent_key, session_id)
    session = _SESSIONS.get(session_key)
    if session is not None and not session.is_running:
        await close_installed_cli_session(agent_key=agent_key, session_id=session_id)
        return {
            'success': False,
            'session_id': session_id,
            'output': '',
            'stderr': f'{agent_label} session {session_id} exited before the next turn',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'events': [],
        }

    try:
        if session is None:
            session = await _start_session(
                agent_key=agent_key,
                agent_label=agent_label,
                session_id=session_id,
                working_dir=working_dir,
                timeout=timeout,
                install_cli=install_cli,
                install_dir=install_dir,
                install_kwargs=install_kwargs,
                path_prepend=path_prepend,
                extra_env=extra_env,
                extra_cli_args=extra_cli_args,
                config_file_patches=config_file_patches,
                pre_start_scripts=pre_start_scripts,
                conversion_log_level=conversion_log_level,
                thinking_preset=thinking_preset,
                include_partial_messages=include_partial_messages,
                include_hook_events=include_hook_events,
                enable_user_input_hooks=enable_user_input_hooks,
                record_trace_events=record_trace_events,
                command_builder=command_builder,
            )
            _SESSIONS[session_key] = session
        return await session.send_user_message(prompt, timeout=timeout, parent_tool_use_id=parent_tool_use_id)
    except Exception as exc:
        logger.exception('%s session turn failed: %s', agent_label, exc)
        stderr = ''
        if session is not None:
            stderr = '\n'.join(session.stderr_lines)
        with contextlib.suppress(Exception):
            await close_installed_cli_session(agent_key=agent_key, session_id=session_id)
        return {
            'success': False,
            'session_id': session_id,
            'output': '',
            'stderr': '\n'.join(part for part in (str(exc), stderr) if part),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'events': [],
        }
    finally:
        if session is not None:
            flush_stream_trace_events(session, agent_key=session.agent_key)


async def close_installed_cli_session(*, agent_key: str, session_id: str) -> dict[str, Any]:
    """Close one runtime-side stream-json CLI session."""
    session = _SESSIONS.pop((agent_key, session_id), None)
    if session is None:
        return {'success': True, 'session_id': session_id, 'exit_code': 0, 'stderr': ''}
    return await session.close()


async def close_all_installed_cli_sessions(*, agent_key: str | None = None) -> dict[str, Any]:
    """Close stream-json CLI sessions, optionally restricted to one agent key."""
    keys = [key for key in _SESSIONS if agent_key is None or key[0] == agent_key]
    results = [await close_installed_cli_session(agent_key=key[0], session_id=key[1]) for key in keys]
    return {'success': True, 'closed': results}


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


__all__ = [
    'build_deferred_tool_hint',
    'build_stream_json_command',
    'build_user_input_request',
    'close_all_installed_cli_sessions',
    'close_installed_cli_session',
    'coerce_multiturn_prompt',
    'encode_user_message',
    'extract_text_from_message',
    'get_deferred_tool_use_id',
    'normalize_stream_json_session_id',
    'parse_stream_line',
    'run_installed_cli_session_turn',
    'run_installed_multiturn',
    'summarize_turn_events',
]
