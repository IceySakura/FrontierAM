#!/usr/bin/env python3
"""Exercise a live-runner Claude Code state download/restore flow.

The script talks only to the live server API:

1. Create a run that asks Claude Code to write /workspace/index.html.
2. Stream and print important events.
3. Submit another user turn.
4. Download /workspace and Claude Code state from the runtime via live proxy.
5. Submit an empty turn to end the first run.
6. Create a second run, upload the saved state, and resume the Claude session.
7. Submit a new task, download state again, and end the run.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import shlex
import sys
import tarfile
import textwrap
import time
import uuid

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


JsonObject = dict[str, Any]
JsonList = list[JsonObject]

STATE_ARCHIVE_REMOTE = '/tmp/nexus-live-claude-state.tgz'
RESTORE_ARCHIVE_REMOTE = '/tmp/nexus-live-restore-state.tgz'
RESTORE_ARCHIVE_UPLOAD_REMOTE = f'{RESTORE_ARCHIVE_REMOTE}.uploading'
RESTORE_PRE_START_TIMEOUT_SECONDS = 660
RUNTIME_NAME = 'main'
LOG_WIDTH = 96


def truncate(value: str, limit: int = 500) -> str:
    value = ' '.join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + '...'


def truncate_multiline(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + '...'


def dumps_compact(value: Any, limit: int = 700) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except TypeError:
        text = repr(value)
    return truncate(text, limit)


def dumps_pretty(value: Any, limit: int = 3000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        text = repr(value)
    return truncate_multiline(text, limit)


def log(message: str = '', *, indent: int = 0) -> None:
    prefix = time.strftime('[%H:%M:%S]')
    print(f'{prefix} {" " * indent}{message}', flush=True)


def log_section(title: str) -> None:
    print('', flush=True)
    print('=' * LOG_WIDTH, flush=True)
    log(title)
    print('=' * LOG_WIDTH, flush=True)


def log_step(title: str) -> None:
    print('', flush=True)
    log(f'-- {title}')


def log_kv(label: str, value: Any, *, indent: int = 2) -> None:
    log(f'{label:<22} {value}', indent=indent)


def log_block(title: str, text: Any, *, indent: int = 2, limit: int = 4000) -> None:
    log(f'{title}:', indent=indent)
    value = truncate_multiline(str(text), limit)
    lines = value.splitlines() or ['']
    width = max(40, LOG_WIDTH - indent - 6)
    for line in lines:
        if not line:
            log('', indent=indent + 2)
            continue
        wrapped = textwrap.wrap(line, width=width, replace_whitespace=False, drop_whitespace=False)
        for part in wrapped or ['']:
            log(part, indent=indent + 2)


def log_json_block(title: str, value: Any, *, indent: int = 2, limit: int = 3000) -> None:
    log_block(title, dumps_pretty(value, limit=limit), indent=indent, limit=limit)


def log_run(run_id: str, message: str, *, indent: int = 0) -> None:
    log(f'[{run_id}] {message}', indent=indent)


def format_bytes(size: int) -> str:
    units = ['B', 'KiB', 'MiB', 'GiB']
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == 'B':
                return f'{int(value)} {unit}'
            return f'{value:.1f} {unit}'
        value /= 1024
    return f'{size} B'


def read_archive_text(tf: tarfile.TarFile, name: str) -> str | None:
    try:
        member = tf.extractfile(name)
    except KeyError:
        return None
    if member is None:
        return None
    return member.read().decode('utf-8', errors='replace')


def read_archive_json(tf: tarfile.TarFile, name: str) -> JsonObject | None:
    text = read_archive_text(tf, name)
    if text is None:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def read_archive_jsonl(tf: tarfile.TarFile, name: str) -> JsonList:
    text = read_archive_text(tf, name)
    if text is None:
        return []
    rows: JsonList = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def inspect_claude_state_archive(path: Path) -> JsonObject:
    with tarfile.open(path, 'r:gz') as tf:
        names = sorted(tf.getnames())
        session_file_names = [
            name for name in names if name.startswith('root/.claude/sessions/') and name.endswith('.json')
        ]
        session_files = {name: read_archive_json(tf, name) or {} for name in session_file_names}
        transcript_names = [
            name for name in names if name.startswith('root/.claude/projects/') and name.endswith('.jsonl')
        ]
        transcripts = {name: read_archive_jsonl(tf, name) for name in transcript_names}
        return {
            'path': str(path),
            'size': path.stat().st_size,
            'session_files': session_files,
            'transcripts': transcripts,
        }


def compare_state_archives(first_state: Path, second_state: Path) -> None:
    log_section('Resume state validation')
    first = inspect_claude_state_archive(first_state)
    second = inspect_claude_state_archive(second_state)
    first_sessions = first['session_files']
    second_sessions = second['session_files']
    first_session_ids = sorted({data.get('sessionId') for data in first_sessions.values() if data.get('sessionId')})
    second_session_ids = sorted({data.get('sessionId') for data in second_sessions.values() if data.get('sessionId')})
    first_session_file_set = set(first_sessions)
    second_session_file_set = set(second_sessions)
    added_session_files = sorted(second_session_file_set - first_session_file_set)

    log_kv('first state', f'{first_state} ({format_bytes(first["size"])})')
    log_kv('second state', f'{second_state} ({format_bytes(second["size"])})')
    log_kv('first session ids', ', '.join(first_session_ids) or '(none)')
    log_kv('second session ids', ', '.join(second_session_ids) or '(none)')
    log_kv('same session id set', first_session_ids == second_session_ids)
    if added_session_files:
        log_block(
            'new process metadata files',
            '\n'.join(added_session_files),
            indent=2,
            limit=1200,
        )
        for name in added_session_files:
            log_json_block(f'{name}', second_sessions.get(name), indent=4, limit=1000)

    first_transcripts = first['transcripts']
    second_transcripts = second['transcripts']
    common_transcripts = sorted(set(first_transcripts) & set(second_transcripts))
    for name in common_transcripts:
        first_rows = first_transcripts[name]
        second_rows = second_transcripts[name]
        prefix_match = second_rows[: len(first_rows)] == first_rows
        log_kv('transcript', name)
        log_kv('first rows', len(first_rows), indent=4)
        log_kv('second rows', len(second_rows), indent=4)
        log_kv('appended rows', len(second_rows) - len(first_rows), indent=4)
        log_kv('second has first as prefix', prefix_match, indent=4)
        if len(second_rows) > len(first_rows):
            appended = second_rows[len(first_rows)]
            log_json_block('first appended row', appended, indent=4, limit=1400)

    if first_session_ids == second_session_ids and common_transcripts:
        log_block(
            'interpretation',
            (
                'Resume is considered valid here: Claude Code kept the same sessionId and appended to the same '
                'project transcript jsonl. Extra files under root/.claude/sessions/<pid>.json are per-process '
                'metadata created by the new Claude Code process in the second container; they are not separate '
                'conversation transcripts.'
            ),
            indent=2,
            limit=1800,
        )


def unwrap_executor_response(payload: JsonObject) -> JsonObject:
    if 'result' in payload and isinstance(payload['result'], dict):
        return payload['result']
    return payload


@dataclass
class DemoConfig:
    live_url: str
    api_key: str
    api_key_query: str
    base_url: str
    model: str
    output_dir: Path
    session_id: str
    claude_code_version: str
    timeout_seconds: float
    runtime_provider: str
    sandbox_image: str
    worker_venv: bool
    gongfeng_api_base_url: str
    gongfeng_token: str | None
    gongfeng_runtime_image: str
    gongfeng_runtime_provider: str
    gongfeng_queue_wait_max: float

    @property
    def effective_api_key(self) -> str:
        if not self.api_key_query or '?' in self.api_key:
            return self.api_key
        return f'{self.api_key}?{self.api_key_query}'


class LiveClient:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        self.live_url = config.live_url.rstrip('/')

    async def close(self) -> None:
        await self.session.close()

    async def request_json(self, method: str, path: str, *, json_body: JsonObject | None = None) -> JsonObject:
        url = f'{self.live_url}{path}'
        async with self.session.request(method, url, json=json_body) as response:
            text = await response.text()
            if response.status >= 400:
                raise RuntimeError(f'{method} {path} failed with HTTP {response.status}: {truncate(text, 1000)}')
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'{method} {path} returned non-JSON response: {truncate(text, 1000)}') from exc

    async def create_run(self, payload: JsonObject) -> JsonObject:
        return await self.request_json('POST', '/v1/runs', json_body=payload)

    async def get_run(self, run_id: str) -> JsonObject:
        return await self.request_json('GET', f'/v1/runs/{run_id}')

    async def submit_turn(self, run_id: str, text: str) -> JsonObject:
        return await self.request_json('POST', f'/v1/runs/{run_id}/turns', json_body={'text': text})

    async def call_executor(self, run_id: str, executor_name: str, arg: JsonObject) -> JsonObject:
        path = f'/v1/runs/{run_id}/runtimes/{RUNTIME_NAME}/proxy/api/v1/call_executor'
        payload = {'executor_name': executor_name, 'arg': arg}
        result = await self.request_json('POST', path, json_body=payload)
        return unwrap_executor_response(result)

    async def wait_for_status(self, run_id: str, wanted: set[str], *, timeout: float, label: str) -> JsonObject:
        started = time.monotonic()
        last_status = ''
        while time.monotonic() - started < timeout:
            run = await self.get_run(run_id)
            status = str(run.get('status', ''))
            if status != last_status:
                log_run(run_id, f'observed status: {status}')
                last_status = status
            if status in wanted:
                return run
            if status in {'failed', 'killed'} and status not in wanted:
                raise RuntimeError(f'{run_id} reached terminal status={status}: {dumps_compact(run)}')
            if status == 'closed' and 'closed' not in wanted:
                raise RuntimeError(f'{run_id} closed while waiting for {label}: {dumps_compact(run)}')
            await asyncio.sleep(1.0)
        raise TimeoutError(f'Timed out waiting for {run_id} to reach {label}')

    async def wait_for_runtime_proxy(self, run_id: str, *, timeout: float) -> JsonObject:
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            run = await self.get_run(run_id)
            runtimes = run.get('runtimes') or {}
            runtime = runtimes.get(RUNTIME_NAME) if isinstance(runtimes, dict) else None
            if isinstance(runtime, dict):
                status = runtime.get('status') or runtime.get('state')
                if runtime.get('proxy_url') or runtime.get('base_url') or status in {'ready', 'running'}:
                    log_run(run_id, f'runtime "{RUNTIME_NAME}" is ready for proxy calls')
                    log_json_block('runtime endpoint', runtime, indent=4, limit=1200)
                    return run
            if str(run.get('status')) in {'failed', 'killed', 'closed'}:
                raise RuntimeError(f'{run_id} stopped before runtime proxy became ready: {dumps_compact(run)}')
            await asyncio.sleep(0.5)
        raise TimeoutError(f'Timed out waiting for {run_id} runtime proxy')


class EventPrinter:
    def __init__(self, client: LiveClient, run_id: str) -> None:
        self.client = client
        self.run_id = run_id
        self.cursor: int | None = None
        self._stopped = asyncio.Event()

    def stop(self) -> None:
        self._stopped.set()

    async def run(self) -> None:
        while not self._stopped.is_set():
            path = f'/v1/runs/{self.run_id}/events'
            if self.cursor is not None:
                path += f'?cursor={self.cursor}'
            try:
                await self._stream_once(path)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stopped.is_set():
                    return
                log_run(self.run_id, f'event stream reconnecting after error: {exc}')
                await asyncio.sleep(1.0)

    async def _stream_once(self, path: str) -> None:
        url = f'{self.client.live_url}{path}'
        async with self.client.session.get(url) as response:
            if response.status >= 400:
                text = await response.text()
                raise RuntimeError(f'GET {path} failed with HTTP {response.status}: {truncate(text, 500)}')
            buffer = ''
            async for chunk in response.content.iter_chunked(65536):
                if self._stopped.is_set():
                    return
                buffer += chunk.decode('utf-8', errors='replace')
                buffer = buffer.replace('\r\n', '\n').replace('\r', '\n')
                while '\n\n' in buffer:
                    block, buffer = buffer.split('\n\n', 1)
                    data_lines = [line[5:].lstrip() for line in block.split('\n') if line.startswith('data:')]
                    if data_lines:
                        self._handle_sse_payload('\n'.join(data_lines))
                        if self._stopped.is_set():
                            return

    def _handle_sse_payload(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict) and payload.get('type') == 'ping':
            return
        if not isinstance(payload, dict):
            return
        event_id = payload.get('id')
        if isinstance(event_id, int):
            self.cursor = max(self.cursor or 0, event_id + 1)
        self._print_event(payload)

    def _print_event(self, payload: JsonObject) -> None:
        event_type = str(payload.get('type') or '')
        data = payload.get('data') if isinstance(payload.get('data'), dict) else {}

        if event_type == 'run.status':
            status = data.get('status')
            turn_id = data.get('turn_id')
            suffix = f' turn={turn_id}' if turn_id else ''
            log_run(self.run_id, f'RUN STATUS -> {status}{suffix}')
            if event_type == 'run.status' and data.get('status') in {'closed', 'failed', 'killed'}:
                self.stop()
            return

        if event_type == 'turn.started':
            log_run(self.run_id, f'TURN STARTED {data.get("turn_id")}')
            return

        if event_type == 'turn.completed':
            log_run(self.run_id, f'TURN COMPLETED {data.get("turn_id")}')
            log_json_block('turn result', data, indent=4, limit=1200)
            return

        if event_type == 'input.required':
            request = data.get('request') if isinstance(data.get('request'), dict) else {}
            log_run(self.run_id, f'INPUT REQUIRED source={request.get("source")} turn={data.get("turn_id")}')
            hint = request.get('hint')
            if hint:
                log_block('agent output / prompt for next turn', hint, indent=4, limit=2500)
            message = request.get('message')
            if isinstance(message, dict):
                log_json_block('raw input request message', message, indent=4, limit=1400)
            return

        if event_type == 'run.output':
            log_run(self.run_id, f'RUN OUTPUT success={data.get("success")}')
            output = data.get('output') if isinstance(data.get('output'), dict) else {}
            agent_output = output.get('agent_output') if isinstance(output.get('agent_output'), dict) else {}
            task_output = output.get('task_output')
            if agent_output:
                log_kv('agent success', agent_output.get('success'), indent=4)
                if agent_output.get('output'):
                    log_block('agent final output', agent_output.get('output'), indent=4, limit=2200)
            if task_output not in (None, {}):
                log_json_block('task output', task_output, indent=4, limit=1200)
            return

        if event_type == 'run.error':
            log_run(self.run_id, 'RUN ERROR')
            log_json_block('error payload', data, indent=4, limit=2400)
            return

        if event_type == 'turn.submitted':
            text = str(data.get('text') or data.get('input') or '')
            log_run(self.run_id, f'USER TURN submitted turn={data.get("turn_id")}')
            log_block('prompt', text if text else '(empty input)', indent=4, limit=2500)
            return

        if event_type == 'runtime.ready':
            log_run(self.run_id, 'RUNTIME READY')
            runtimes = data.get('runtimes')
            if runtimes:
                log_json_block('runtimes', runtimes, indent=4, limit=1600)
            return

        if event_type != 'trace.event':
            return

        trace_event = data.get('event') if isinstance(data.get('event'), dict) else {}
        direct_span_event = trace_event.get('event')
        if isinstance(direct_span_event, dict):
            self._print_trace_span_event(
                str(trace_event.get('span_name') or ''),
                str(trace_event.get('span_id') or ''),
                direct_span_event,
            )

        items = trace_event.get('items')
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                span_name = str(item.get('span_name') or trace_event.get('span_name') or '')
                span_id = str(item.get('span_id') or trace_event.get('span_id') or '')
                for item_event in item.get('events') or []:
                    if isinstance(item_event, dict):
                        self._print_trace_span_event(span_name, span_id, item_event)

    def _print_trace_span_event(self, span_name: str, span_id: str, span_event: JsonObject) -> None:
        span_event_name = str(span_event.get('name') or '')
        span_event_type = str(span_event.get('type') or '')
        attrs = span_event.get('attributes') if isinstance(span_event.get('attributes'), dict) else {}

        if span_event_name == 'openai_completion' and span_event_type in {'START', 'END'}:
            inputs = attrs.get('inputs') if isinstance(attrs.get('inputs'), dict) else {}
            messages = inputs.get('messages') if isinstance(inputs.get('messages'), list) else []
            model = inputs.get('model') or attrs.get('model') or '(default)'
            client_key = attrs.get('_client_key') or '(unknown client)'
            log_run(
                self.run_id,
                (
                    f'MODEL REQUEST {span_event_type.lower()} '
                    f'span={span_id or span_name} client={client_key} model={model} messages={len(messages)}'
                ),
            )
            if span_event_type == 'START' and messages:
                last_message = messages[-1] if isinstance(messages[-1], dict) else {}
                last_role = last_message.get('role')
                last_content = last_message.get('content')
                if isinstance(last_content, str) and last_content.strip():
                    log_block(f'last {last_role} message', last_content, indent=4, limit=1400)
            return

        if span_event_name == 'agent.stream':
            cc_event = attrs.get('event')
            if isinstance(cc_event, dict):
                self._print_claude_code_event(cc_event)

    def _print_claude_code_event(self, event: JsonObject) -> None:
        event_type = str(event.get('type') or '')
        subtype = str(event.get('subtype') or '')

        if event_type == 'system':
            if subtype == 'init':
                log_run(
                    self.run_id,
                    (
                        'CLAUDE INIT '
                        f'session={event.get("session_id")} cwd={event.get("cwd")} '
                        f'model={event.get("model")} version={event.get("claude_code_version")}'
                    ),
                )
                tools = event.get('tools') if isinstance(event.get('tools'), list) else []
                mcp_servers = event.get('mcp_servers') if isinstance(event.get('mcp_servers'), list) else []
                log_kv('tools', len(tools), indent=4)
                log_kv('mcp servers', len(mcp_servers), indent=4)
            else:
                detail = (
                    event.get('subtype') or event.get('status') or event.get('message') or dumps_compact(event, 300)
                )
                log_run(self.run_id, f'CLAUDE SYSTEM {detail}')
            return

        if event_type == 'assistant':
            message = event.get('message') if isinstance(event.get('message'), dict) else {}
            self._print_claude_content(message.get('content'), 'CLAUDE MESSAGE')
            return

        if event_type == 'user':
            message = event.get('message') if isinstance(event.get('message'), dict) else {}
            self._print_claude_content(message.get('content'), 'CLAUDE USER CONTEXT')
            return

        if event_type == 'result':
            result = event.get('result') or event.get('subtype') or event.get('is_error')
            log_run(self.run_id, f'CLAUDE RESULT subtype={event.get("subtype")} error={event.get("is_error")}')
            if result:
                log_block('result', result, indent=4, limit=2200)
            return

        if event_type == 'stream_event':
            self._print_claude_stream_event(event)
            return

        if event_type:
            log_run(self.run_id, f'CLAUDE EVENT {event_type}{f"/{subtype}" if subtype else ""}')
            log_json_block('event payload', event, indent=4, limit=1200)

    def _print_claude_stream_event(self, event: JsonObject) -> None:
        stream_event = event.get('event') if isinstance(event.get('event'), dict) else {}
        stream_type = str(stream_event.get('type') or '')
        if stream_type == 'content_block_delta':
            delta = stream_event.get('delta') if isinstance(stream_event.get('delta'), dict) else {}
            if delta.get('type') in {'thinking_delta', 'signature_delta'}:
                text = str(delta.get('thinking') or delta.get('text') or '')
                if text.strip():
                    log_run(self.run_id, 'CLAUDE THINKING')
                    log_block('thinking', text, indent=4, limit=1400)

    def _print_claude_content(self, content: Any, title: str) -> None:
        if isinstance(content, str):
            if content.strip():
                log_run(self.run_id, title)
                log_block('content', content, indent=4, limit=2500)
            return
        if not isinstance(content, list):
            return
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get('type') or '')
            if item_type == 'text':
                text = str(item.get('text') or '')
                if text.strip():
                    log_run(self.run_id, title)
                    log_block('content', text, indent=4, limit=2500)
            elif item_type == 'thinking':
                text = str(item.get('thinking') or item.get('text') or '')
                if text.strip():
                    log_run(self.run_id, 'CLAUDE THINKING')
                    log_block('thinking', text, indent=4, limit=1800)
            elif item_type == 'tool_use':
                name = item.get('name') or item.get('tool_name') or 'tool'
                tool_input = item.get('input') or item.get('parameters') or {}
                log_run(self.run_id, f'TOOL CALL {name}')
                log_json_block('input', tool_input, indent=4, limit=2200)
            elif item_type == 'tool_result':
                content_value = item.get('content') or item.get('result') or ''
                log_run(self.run_id, 'TOOL RESULT')
                log_block('output', content_value, indent=4, limit=2200)


def make_common_run_input(
    config: DemoConfig, *, user_prompt: str | None, pre_start_scripts: list[JsonObject]
) -> JsonObject:
    agent_config: JsonObject = {
        'session_id': config.session_id,
        'working_dir': '/workspace',
        'version': config.claude_code_version,
        'npm_registry': 'https://registry.npmmirror.com/',
        'max_turns': 10000,
        'timeout': 86400,
        'enable_user_input_hooks': True,
        'include_hook_events': True,
        'include_partial_messages': True,
        'record_trace_events': True,
        'pre_start_scripts': pre_start_scripts,
    }
    if user_prompt is not None:
        agent_config['user_prompt'] = user_prompt

    runtime_config: JsonObject = {
        'image': config.sandbox_image,
        'worker_venv': config.worker_venv,
    }
    runtime_provider_config: JsonObject = {}
    if config.runtime_provider == 'docker':
        runtime_config['extra_args'] = [
            '--tmpfs',
            '/run',
            '--tmpfs',
            '/run/lock',
            '-e',
            'GTK_A11Y=none',
            '-e',
            'NEXUS_GUI_REQUIRE_SYSTEM_DBUS=1',
        ]
    elif config.runtime_provider == 'gongfeng':
        runtime_provider_config = {
            'api_base_url': config.gongfeng_api_base_url,
            'runtime_image': config.gongfeng_runtime_image,
            'runtime_provider': config.gongfeng_runtime_provider,
            'queue_wait_max': config.gongfeng_queue_wait_max,
            'print_port_mappings': True,
        }
        if config.gongfeng_token:
            runtime_provider_config['token'] = config.gongfeng_token
    else:
        raise ValueError(f'Unsupported runtime provider for this demo: {config.runtime_provider}')

    return {
        'imports': [
            'nexus.extensions',
            'nexus.extensions.openai_clients.litellm',
        ],
        'task_name': 'identity',
        'agent_name': 'claude_code_multiturn',
        'agent_config': agent_config,
        'runtime_provider_name': config.runtime_provider,
        'runtime_provider_config': runtime_provider_config,
        'runtime_configs': {
            RUNTIME_NAME: runtime_config,
        },
        'openai_clients': {
            'main': {
                'name': 'litellm',
                'config': {
                    'client_args': {
                        'api_key': config.effective_api_key,
                        'base_url': config.base_url,
                    },
                    'request_args': {
                        'model': config.model,
                        'custom_llm_provider': 'bedrock',
                    },
                },
            }
        },
        'runner': {
            'timeout': 86400,
        },
    }


def make_first_run_payload(config: DemoConfig) -> JsonObject:
    prompt = (
        'In /workspace, create a self-contained index.html for a small product landing page. '
        'Use only HTML, CSS, and JavaScript in that file. Include a clear header, a few content cards, '
        'and one simple interactive control. Keep it concise, then briefly summarize the file you created.'
    )
    return make_common_run_input(
        config,
        user_prompt=prompt,
        pre_start_scripts=[
            {
                'code': (
                    "set -e\nmkdir -p /workspace\nprintf 'Nexus live runner resume demo\\n' > /workspace/README.md\n"
                )
            }
        ],
    )


def make_restore_run_payload(config: DemoConfig) -> JsonObject:
    prompt = (
        'Resume the previous Claude Code session and workspace. Update /workspace/index.html: '
        "add a visible footer that says 'Restored session verified' and add one more small interactive control. "
        'Do not recreate the project from scratch.'
    )
    restore_script = (
        'set -e\n'
        'for i in $(seq 1 600); do\n'
        f'  if [ -f {RESTORE_ARCHIVE_REMOTE} ]; then break; fi\n'
        '  sleep 1\n'
        'done\n'
        f'test -f {RESTORE_ARCHIVE_REMOTE} || '
        "{ echo 'restore archive was not uploaded before timeout' >&2; exit 1; }\n"
        f'tar -C / -xzf {RESTORE_ARCHIVE_REMOTE}\n'
        'mkdir -p /workspace\n'
    )
    payload = make_common_run_input(
        config,
        user_prompt=prompt,
        pre_start_scripts=[{'code': restore_script, 'timeout': RESTORE_PRE_START_TIMEOUT_SECONDS}],
    )
    payload['agent_config']['extra_cli_args'] = ['--resume', config.session_id]
    return payload


async def run_command(
    client: LiveClient, run_id: str, command: str, *, timeout: int = 300, cwd: str = '/workspace'
) -> JsonObject:
    log_run(run_id, 'EXECUTOR run_command')
    log_block('command', command, indent=4, limit=1200)
    result = await client.call_executor(
        run_id,
        'run_command',
        {
            'command': command,
            'cwd': cwd,
            'timeout': timeout,
        },
    )
    return_code = result.get('return_code')
    if return_code not in (0, None):
        raise RuntimeError(
            f'Command failed in {run_id} with return_code={return_code}: '
            f'stdout={truncate(str(result.get("stdout") or ""), 1000)} '
            f'stderr={truncate(str(result.get("stderr") or ""), 1000)}'
        )
    stdout = str(result.get('stdout') or '').strip()
    stderr = str(result.get('stderr') or '').strip()
    if stdout:
        log_block('stdout', stdout, indent=4, limit=1200)
    if stderr:
        log_block('stderr', stderr, indent=4, limit=1200)
    return result


async def download_state(client: LiveClient, run_id: str, local_path: Path) -> None:
    log_step(f'Download state from {run_id}')
    log_run(run_id, 'packaging /workspace and Claude Code state inside runtime')
    log_kv('remote archive', STATE_ARCHIVE_REMOTE, indent=4)
    log_kv('local archive', local_path, indent=4)
    command = (
        f'rm -f {STATE_ARCHIVE_REMOTE} && '
        f'tar --ignore-failed-read -C / -czf {STATE_ARCHIVE_REMOTE} '
        'workspace root/.claude root/.claude.json root/.claude.json.backup root/.config/claude && '
        f'ls -lh {STATE_ARCHIVE_REMOTE}'
    )
    await run_command(client, run_id, command, timeout=300)

    log_run(run_id, 'EXECUTOR download_file')
    result = await client.call_executor(
        run_id,
        'download_file',
        {
            'files': [
                {
                    'path': STATE_ARCHIVE_REMOTE,
                    'encoding': 'base64',
                }
            ]
        },
    )
    files = result.get('files')
    if not isinstance(files, list) or not files:
        raise RuntimeError(f'download_file returned no files: {dumps_compact(result)}')
    file_result = files[0]
    if not isinstance(file_result, dict):
        raise RuntimeError(f'Unexpected download_file response: {dumps_compact(result)}')
    content = file_result.get('content') or file_result.get('data')
    if not isinstance(content, str):
        raise RuntimeError(f'download_file response has no base64 content: {dumps_compact(file_result)}')
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(base64.b64decode(content))
    log_run(run_id, 'state archive saved')
    log_kv('path', local_path, indent=4)
    log_kv('size', format_bytes(local_path.stat().st_size), indent=4)


async def upload_state(client: LiveClient, run_id: str, local_path: Path) -> None:
    log_step(f'Upload state into {run_id}')
    log_run(run_id, 'EXECUTOR upload_file')
    log_kv('local archive', local_path, indent=4)
    log_kv('remote archive', RESTORE_ARCHIVE_REMOTE, indent=4)
    log_kv('upload temp archive', RESTORE_ARCHIVE_UPLOAD_REMOTE, indent=4)
    log_kv('size', format_bytes(local_path.stat().st_size), indent=4)
    encoded = base64.b64encode(local_path.read_bytes()).decode('ascii')
    result = await client.call_executor(
        run_id,
        'upload_file',
        {
            'files': [
                {
                    'path': RESTORE_ARCHIVE_UPLOAD_REMOTE,
                    'content': encoded,
                    'encoding': 'base64',
                }
            ],
            'overwrite': True,
        },
    )
    log_json_block('upload result', result, indent=4, limit=1200)
    results = result.get('results')
    if (
        not isinstance(results, list)
        or not results
        or not all(isinstance(item, dict) and item.get('success') for item in results)
    ):
        raise RuntimeError(f'upload_file failed: {dumps_compact(result)}')

    log_run(run_id, 'publish uploaded state archive')
    wait_for_upload = (
        f'for i in $(seq 1 120); do '
        f'test -f {shlex.quote(RESTORE_ARCHIVE_UPLOAD_REMOTE)} && break; '
        'sleep 1; '
        'done; '
        f'test -f {shlex.quote(RESTORE_ARCHIVE_UPLOAD_REMOTE)} || '
        f'{{ echo "uploaded state archive not visible: {RESTORE_ARCHIVE_UPLOAD_REMOTE}" >&2; ls -la /tmp >&2; exit 1; }}; '
    )
    publish_result = await run_command(
        client,
        run_id,
        wait_for_upload
        + f'mv -f {shlex.quote(RESTORE_ARCHIVE_UPLOAD_REMOTE)} {shlex.quote(RESTORE_ARCHIVE_REMOTE)} && '
        f'ls -lh {shlex.quote(RESTORE_ARCHIVE_REMOTE)}',
        timeout=180,
        cwd='/',
    )
    if publish_result.get('status') != 'Finished' or publish_result.get('return_code') not in (0, None):
        raise RuntimeError(f'failed to publish uploaded state archive: {dumps_compact(publish_result)}')


async def end_run_with_empty_turn(client: LiveClient, run_id: str, timeout: float) -> None:
    log_step(f'End {run_id} with empty input')
    log_run(run_id, 'submitting empty user turn; Claude Code should treat this as normal shutdown request')
    log_block('prompt', '(empty input)', indent=4)
    await client.submit_turn(run_id, '')
    await client.wait_for_status(run_id, {'closed'}, timeout=timeout, label='closed')


async def stop_printer(printer: EventPrinter, task: asyncio.Task[None]) -> None:
    printer.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def run_demo(config: DemoConfig) -> None:
    client = LiveClient(config)
    watchers: list[tuple[EventPrinter, asyncio.Task[None]]] = []
    try:
        log_section('Nexus live runner Claude Code resume demo')
        log_kv('live server', config.live_url)
        log_kv('model', config.model)
        log_kv('base url', config.base_url)
        log_kv('claude code version', config.claude_code_version)
        log_kv('session id', config.session_id)
        log_kv('output dir', config.output_dir)
        log_kv('runtime provider', config.runtime_provider)
        log_kv('sandbox image', config.sandbox_image)
        if config.runtime_provider == 'gongfeng':
            log_kv('gongfeng api', config.gongfeng_api_base_url)
            log_kv('gongfeng runtime image', config.gongfeng_runtime_image)
            log_kv('gongfeng runtime provider', config.gongfeng_runtime_provider or '(auto)')
        log_kv('task / agent', 'identity / claude_code_multiturn')

        first_run_id = f'run_demo_{uuid.uuid4().hex[:12]}'
        first_payload = make_first_run_payload(config)
        first_payload['run_id'] = first_run_id
        first_prompt = str(first_payload['agent_config']['user_prompt'])
        log_section('Stage 1: create first run')
        log_kv('run id', first_run_id)
        log_block('initial user prompt', first_prompt, indent=2, limit=3000)
        log_json_block(
            'run input summary',
            {
                'task_name': first_payload.get('task_name'),
                'agent_name': first_payload.get('agent_name'),
                'working_dir': first_payload['agent_config'].get('working_dir'),
                'session_id': first_payload['agent_config'].get('session_id'),
                'runtime_provider_name': first_payload.get('runtime_provider_name'),
                'runtime_provider_config': {
                    key: ('***' if key == 'token' else value)
                    for key, value in (first_payload.get('runtime_provider_config') or {}).items()
                },
                'runtime_name': RUNTIME_NAME,
                'sandbox_image': first_payload['runtime_configs'][RUNTIME_NAME].get('image'),
                'worker_venv': first_payload['runtime_configs'][RUNTIME_NAME].get('worker_venv'),
            },
            indent=2,
            limit=1600,
        )
        await client.create_run(first_payload)
        first_printer = EventPrinter(client, first_run_id)
        watchers.append((first_printer, asyncio.create_task(first_printer.run())))

        log_step('Wait until first Claude Code turn reaches idle')
        await client.wait_for_status(
            first_run_id, {'idle'}, timeout=config.timeout_seconds, label='idle after first turn'
        )

        tweak_prompt = (
            'Please update /workspace/index.html: make the visual theme green/teal, add a compact status badge, '
            'and keep the page self-contained. Then summarize the change.'
        )
        log_section('Stage 2: submit follow-up turn in the same run')
        log_kv('run id', first_run_id)
        log_block('follow-up user prompt', tweak_prompt, indent=2, limit=3000)
        await client.submit_turn(first_run_id, tweak_prompt)
        log_step('Wait until follow-up turn reaches idle')
        await client.wait_for_status(
            first_run_id, {'idle'}, timeout=config.timeout_seconds, label='idle after tweak turn'
        )

        first_state = config.output_dir / f'{first_run_id}-state.tgz'
        await download_state(client, first_run_id, first_state)
        await end_run_with_empty_turn(client, first_run_id, config.timeout_seconds)
        await stop_printer(first_printer, watchers.pop()[1])

        second_run_id = f'run_demo_{uuid.uuid4().hex[:12]}'
        second_payload = make_restore_run_payload(config)
        second_payload['run_id'] = second_run_id
        restore_prompt = str(second_payload['agent_config']['user_prompt'])
        log_section('Stage 3: create restore run')
        log_kv('run id', second_run_id)
        log_kv('resume session id', config.session_id)
        log_kv('restore archive source', first_state)
        log_block('restore user prompt', restore_prompt, indent=2, limit=3000)
        log_block(
            'restore pre-start behavior',
            (
                f'Claude Code pre-start waits for {RESTORE_ARCHIVE_REMOTE}, extracts it into /, '
                'then starts Claude Code with --resume using the same session id.'
            ),
            indent=2,
            limit=1200,
        )
        await client.create_run(second_payload)
        second_printer = EventPrinter(client, second_run_id)
        watchers.append((second_printer, asyncio.create_task(second_printer.run())))

        log_step('Wait for restore runtime proxy, then upload saved state')
        await client.wait_for_runtime_proxy(second_run_id, timeout=config.timeout_seconds)
        await upload_state(client, second_run_id, first_state)
        log_step('Wait until restored Claude Code turn reaches idle')
        await client.wait_for_status(
            second_run_id, {'idle'}, timeout=config.timeout_seconds, label='idle after restore turn'
        )

        second_state = config.output_dir / f'{second_run_id}-state.tgz'
        await download_state(client, second_run_id, second_state)
        await end_run_with_empty_turn(client, second_run_id, config.timeout_seconds)

        log_section('Demo completed successfully')
        log_kv('Claude session id', config.session_id)
        log_kv('first run', first_run_id)
        log_kv('restore run', second_run_id)
        log_kv('first state archive', first_state)
        log_kv('first state size', format_bytes(first_state.stat().st_size))
        log_kv('second state archive', second_state)
        log_kv('second state size', format_bytes(second_state.stat().st_size))
        compare_state_archives(first_state, second_state)
    finally:
        for printer, task in watchers:
            printer.stop()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await client.close()


def parse_args(argv: list[str]) -> DemoConfig:
    parser = argparse.ArgumentParser(description='Run the live-runner Claude Code resume demo.')
    parser.add_argument('--live-url', default=os.environ.get('NEXUS_LIVE_URL', 'http://127.0.0.1:8765'))
    parser.add_argument('--api-key', default=os.environ.get('NEXUS_LIVE_DEMO_API_KEY'))
    parser.add_argument(
        '--api-key-query',
        default=os.environ.get('NEXUS_LIVE_DEMO_API_KEY_QUERY', 'provider=aws_third&timeout=3600'),
        help='Query string appended to a bare API key before passing it to LiteLLM. Use empty string to disable.',
    )
    parser.add_argument('--base-url', default='http://trpc-gpt-eval.production.polaris:8080')
    parser.add_argument('--model', default='anthropic.claude-sonnet-4-6')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/live_runner_resume_demo'))
    parser.add_argument('--session-id', default=str(uuid.uuid4()))
    parser.add_argument('--claude-code-version', default='2.1.153')
    parser.add_argument('--timeout-seconds', type=float, default=1800.0)
    parser.add_argument(
        '--runtime-provider',
        choices=['gongfeng', 'docker'],
        default=os.environ.get('NEXUS_LIVE_DEMO_RUNTIME_PROVIDER', 'gongfeng'),
    )
    parser.add_argument(
        '--sandbox-image',
        default=os.environ.get(
            'NEXUS_LIVE_DEMO_SANDBOX_IMAGE',
            'mirrors.tencent.com/hunyuan_yanguan/nexus-ubuntu-session:latest',
        ),
    )
    parser.add_argument(
        '--no-worker-venv',
        action='store_true',
        help='Disable worker_venv in runtime_configs.main. Enabled by default so source changes are uploaded.',
    )
    parser.add_argument(
        '--gongfeng-api-base-url',
        default=os.environ.get('GONGFENG_API_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052'),
    )
    parser.add_argument('--gongfeng-token', default=os.environ.get('GONGFENG_TOKEN'))
    parser.add_argument(
        '--gongfeng-runtime-image',
        default=os.environ.get('NEXUS_GONGFENG_RUNTIME_IMAGE', 'current'),
        help='Use "current" to resolve to the installed Nexus version, e.g. nexus-runtime:1.4.0.',
    )
    parser.add_argument(
        '--gongfeng-runtime-provider',
        default=os.environ.get('GONGFENG_RUNTIME_PROVIDER', 'gongfeng'),
        help='Gongfeng scheduling cluster, for example gongfeng or suanli. Empty string lets backend auto-select.',
    )
    parser.add_argument(
        '--gongfeng-queue-wait-max',
        type=float,
        default=float(os.environ.get('GONGFENG_QUEUE_WAIT_MAX', '1800')),
    )
    args = parser.parse_args(argv)

    if not args.api_key:
        parser.error('Provide --api-key or set NEXUS_LIVE_DEMO_API_KEY.')
    if args.runtime_provider == 'gongfeng' and not args.gongfeng_token:
        parser.error('Provide --gongfeng-token or set GONGFENG_TOKEN when using --runtime-provider gongfeng.')

    return DemoConfig(
        live_url=args.live_url,
        api_key=args.api_key,
        api_key_query=args.api_key_query,
        base_url=args.base_url,
        model=args.model,
        output_dir=args.output_dir,
        session_id=args.session_id,
        claude_code_version=args.claude_code_version,
        timeout_seconds=args.timeout_seconds,
        runtime_provider=args.runtime_provider,
        sandbox_image=args.sandbox_image,
        worker_venv=not args.no_worker_venv,
        gongfeng_api_base_url=args.gongfeng_api_base_url,
        gongfeng_token=args.gongfeng_token,
        gongfeng_runtime_image=args.gongfeng_runtime_image,
        gongfeng_runtime_provider=args.gongfeng_runtime_provider,
        gongfeng_queue_wait_max=args.gongfeng_queue_wait_max,
    )


def main(argv: list[str]) -> int:
    config = parse_args(argv)
    try:
        asyncio.run(run_demo(config))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
