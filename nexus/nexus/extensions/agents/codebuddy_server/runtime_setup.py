"""Runtime-side setup and execution for CodeBuddy agent-server.

Executed on the runtime side via ``runtime.invoke``.  Responsible for:
  1. Installing ``@tencent-ai/agent-server`` (npm),
  2. Starting an OpenAI-compatible proxy on a free port,
  3. Starting the agent-server with ``BASE_URL`` pointing at the proxy,
  4. Sending an HTTP request to ``/api/v1/swe/completions`` or ``/api/v1/craft/completions``,
  5. Collecting the streamed response and returning the result.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
import time

from pathlib import Path
from typing import Any

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    build_runtime_path,
    find_free_port,
    find_nexus_runtime_binary,
    find_runtime_binary,
)


logger = logging.getLogger(__name__)


DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/codebuddy_server'
DEFAULT_MODEL_ID = 'nexus-proxy-model'


async def _wait_for_server(url: str, timeout: float = 60, interval: float = 1.0) -> bool:
    """Poll *url* until the server accepts connections or *timeout* elapses."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=5):
                return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            pass
        await asyncio.sleep(interval)
    return False


async def _pipe_to_logger(
    stream: asyncio.StreamReader,
    prefix: str,
    level: int,
    buffer: list[str],
) -> None:
    """Read lines from *stream*, log each, and append to *buffer*."""
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode('utf-8', errors='replace').rstrip('\n')
        buffer.append(text)
        logger.log(level, '%s%s', prefix, text)


def _parse_sse_events(raw: str) -> dict[str, Any]:
    """Parse SSE stream into structured output: assistant text, tool calls, and tool results."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    active_tools: dict[str, dict[str, Any]] = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith('data: '):
            continue
        data_str = line[6:]
        if data_str == '[done]' or data_str == '[DONE]':
            continue
        try:
            event = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            continue

        event_type = event.get('type', '')
        if event_type == 'text-delta':
            text_parts.append(event.get('textDelta', ''))
        elif event_type == 'tool-call-streaming-start':
            tool_id = event.get('toolCallId', '')
            active_tools[tool_id] = {
                'id': tool_id,
                'name': event.get('toolName', ''),
                'args_text': '',
            }
        elif event_type == 'tool-call-delta':
            tool_id = event.get('toolCallId', '')
            if tool_id in active_tools:
                active_tools[tool_id]['args_text'] += event.get('argsTextDelta', '')
        elif event_type == 'tool-call':
            tool_id = event.get('toolCallId', '')
            tool_calls.append({
                'id': tool_id,
                'name': event.get('toolName', ''),
                'args': event.get('args', {}),
            })
            active_tools.pop(tool_id, None)
        elif event_type == 'tool-result':
            tool_results.append({
                'id': event.get('toolCallId', ''),
                'name': event.get('toolName', ''),
                'result': event.get('result', ''),
            })

    return {
        'assistant_text': ''.join(text_parts),
        'tool_calls': tool_calls,
        'tool_results': tool_results,
    }


def _find_stream_error(raw: str) -> str:
    """The error frame agent-server writes into the SSE body, or '' if there is none.

    Every event the stream is meant to carry is tagged with ``type``; a frame carrying ``code`` instead is the server reporting failure. Its codes mix HTTP-style values (500) with internal ones (10101), so the missing ``type`` is what identifies an error, not the numeric range.
    """
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith('data: '):
            continue
        data_str = line[6:]
        if data_str in ('[done]', '[DONE]'):
            continue
        try:
            event = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(event, dict) and 'type' not in event and 'code' in event:
            return data_str
    return ''


def _blocking_sse_request(
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[bool, str, str]:
    """Perform a blocking HTTP POST and read the SSE stream. Returns (ok, raw, error).

    Handles ``IncompleteRead`` gracefully — the data already collected is
    still usable because agent-server flushes SSE events line-by-line.
    """
    import http.client
    import urllib.request

    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    chunks: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                try:
                    chunk = resp.read(4096)
                except http.client.IncompleteRead as ir:
                    if ir.partial:
                        chunks.append(ir.partial.decode('utf-8', errors='replace'))
                    logger.warning('IncompleteRead after %d chunks — treating collected data as final', len(chunks))
                    break
                if not chunk:
                    break
                text = chunk.decode('utf-8', errors='replace')
                chunks.append(text)
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped:
                        logger.debug('[agent-server] %s', stripped[:500])
    except http.client.IncompleteRead as ir:
        if ir.partial:
            chunks.append(ir.partial.decode('utf-8', errors='replace'))
        logger.warning('IncompleteRead (outer) — using partial data (%d chunks)', len(chunks))
        return bool(chunks), ''.join(chunks), ''
    except Exception as exc:
        return False, ''.join(chunks), str(exc)

    return True, ''.join(chunks), ''


async def _send_completion_request(
    server_url: str,
    mode: str,
    messages: list[dict[str, Any]],
    reasoning_params: dict[str, str] | None = None,
    timeout: float = 3600,
) -> dict[str, Any]:
    """Send a completion request to the agent-server and collect the streamed response."""
    endpoint = f'{server_url}/api/v1/{mode}/completions'
    payload: dict[str, Any] = {'messages': messages}
    if reasoning_params:
        payload['reasoningParams'] = reasoning_params

    logger.debug(f'Sending {mode} completion request to {endpoint}')

    ok, raw, error = await asyncio.to_thread(_blocking_sse_request, endpoint, payload, timeout)

    if not ok:
        logger.error(f'Error reading agent-server response: {error}')

    # Reading the stream to completion says nothing about what it carried: agent-server reports its own failures as an in-band frame and still closes cleanly with [done].
    stream_error = _find_stream_error(raw)
    if stream_error:
        logger.error('agent-server reported a failure in the stream: %s', stream_error)
        ok = False
        error = error or stream_error

    parsed = _parse_sse_events(raw)
    logger.debug(
        'SSE parsed: %d chars text, %d tool calls, %d tool results',
        len(parsed['assistant_text']),
        len(parsed['tool_calls']),
        len(parsed['tool_results']),
    )

    return {
        'success': ok,
        'raw_response': raw,
        'assistant_text': parsed['assistant_text'],
        'tool_calls': parsed['tool_calls'],
        'tool_results': parsed['tool_results'],
        'error': error,
    }


async def setup_and_run_agent_server(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    mode: str = 'swe',
    version: str = 'latest',
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    model_id: str = DEFAULT_MODEL_ID,
    extra_env: dict[str, str] | None = None,
    reasoning_params: dict[str, str] | None = None,
    server_startup_timeout: float = 120,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    max_step_limit: int = 0,
) -> dict[str, Any]:
    """Set up and run the CodeBuddy agent-server on the runtime side."""
    from nexus.extensions.agents.codebuddy_server.installer import install_agent_server
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy
    from nexus.utils.shell import get_os_env_without_ld_library_path

    rt = require_runtime_control()
    start_time = time.time()

    # 1) Install agent-server and patch all hard limits.
    try:
        server_bin = await install_agent_server(
            install_dir=DEFAULT_INSTALL_DIR,
            version=version,
            npm_registry=npm_registry,
        )
        logger.debug(f'agent-server binary at: {server_bin}')

        from nexus.extensions.agents.codebuddy_server.installer import patch_agent_limits

        patch_agent_limits(DEFAULT_INSTALL_DIR, max_step_limit=max_step_limit)
    except Exception as e:
        logger.error(f'Failed to install agent-server: {e}')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install agent-server: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    # 2) Start the OpenAI proxy (bridges LLM requests to the host).
    proxy_port = find_free_port()
    logger.debug(f'Starting OpenAI Proxy on port {proxy_port}...')
    proxy = OpenAIProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        models=[model_id],
    )

    try:
        await proxy.start()
        logger.debug('OpenAI Proxy started')
    except Exception as e:
        logger.error(f'Failed to start OpenAI Proxy: {e}')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to start OpenAI Proxy: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    server_proc = None
    try:
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        # 3) Pre-start scripts.
        base_env = get_os_env_without_ld_library_path()
        base_env['BASH_ENV'] = '~/.bashrc'
        for script in pre_start_scripts or []:
            await _run_pre_start_script(script, working_dir, base_env)

        # 4) Start agent-server with env vars pointing at the OpenAI proxy.
        server_port = find_free_port()
        proxy_base_url = f'http://127.0.0.1:{proxy_port}/v1'

        server_env = {
            **base_env,
            'MODEL_ID': model_id,
            'MODEL_NAME': model_id,
            'BASE_URL': proxy_base_url,
            'API_KEY': DEFAULT_PROXY_API_KEY,
            'TOKEN': DEFAULT_PROXY_API_KEY,
            'PROVIDER_TYPE': 'openai',
            'WORKSPACE_DIR': str(work_path),
            'SERVER_PORT': str(server_port),
            'PATH': build_runtime_path(DEFAULT_INSTALL_DIR),
            'NO_COLOR': '1',
            'MAX_RETRIES': '10',
        }
        if extra_env:
            server_env.update(extra_env)

        node_path = find_nexus_runtime_binary('node')
        if not node_path:
            raise RuntimeError('Node.js is required but not found in runtime')

        cmd = [node_path, str(server_bin)]
        logger.debug(f'Starting agent-server on port {server_port}: {" ".join(cmd)}')

        server_stdout: list[str] = []
        server_stderr: list[str] = []

        server_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(work_path),
            env=server_env,
        )
        assert server_proc.stdout is not None
        assert server_proc.stderr is not None

        asyncio.create_task(_pipe_to_logger(server_proc.stdout, '[agent-server] ', logging.DEBUG, server_stdout))
        asyncio.create_task(_pipe_to_logger(server_proc.stderr, '[agent-server:err] ', logging.WARNING, server_stderr))

        # 5) Wait for the server to become ready.
        server_url = f'http://127.0.0.1:{server_port}'
        ready = await _wait_for_server(server_url, timeout=server_startup_timeout)
        if not ready:
            raise RuntimeError(
                f'agent-server did not become ready within {server_startup_timeout}s. '
                f'stderr: {"".join(server_stderr[-20:])}'
            )

        logger.debug(f'agent-server is ready at {server_url}')

        # 6) Send the task to the SWE or craft endpoint.
        messages = [{'role': 'user', 'content': task}]
        result = await _send_completion_request(
            server_url=server_url,
            mode=mode,
            messages=messages,
            reasoning_params=reasoning_params,
            timeout=timeout,
        )

        execution_time = time.time() - start_time
        logger.info(f'agent-server task completed in {execution_time:.2f}s, success={result["success"]}')

        # 7) Read the last tool results from .agent_history/ on disk.
        # The SSE stream never contains tool-result events, and when
        # agent-server ends without a final LLM call the host-side
        # tracker has no way to capture them.  Reading the conversation
        # log is the only reliable source.
        last_tool_results = _read_last_tool_results_from_history(working_dir)

        output = result.get('assistant_text') or result.get('raw_response', '')
        return {
            'success': result['success'],
            'output': output,
            'stderr': result.get('error', ''),
            'exit_code': 0 if result['success'] else 1,
            'execution_time': execution_time,
            'tool_calls': result.get('tool_calls', []),
            'tool_results': result.get('tool_results', []),
            'last_tool_results': last_tool_results,
            'raw_response': result.get('raw_response', ''),
        }

    except Exception as e:
        logger.error(f'agent-server execution error: {e}')
        return {
            'success': False,
            'output': '',
            'stderr': str(e),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    finally:
        if server_proc is not None:
            logger.debug('Stopping agent-server process...')
            server_proc.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(server_proc.wait(), timeout=10)
            if server_proc.returncode is None:
                server_proc.kill()
                with contextlib.suppress(Exception):
                    await server_proc.wait()

        logger.debug('Stopping OpenAI Proxy...')
        await proxy.stop()


def _read_last_tool_results_from_history(working_dir: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Read the last tool results from agent-server's ``.agent_history/`` conversation log.

    Agent-server stores every message (user, assistant, tool) as individual JSON
    files.  Tool messages contain the full execution result including stdout,
    stderr, and exit code.  This is the **only** reliable way to capture the
    results of the very last tool calls, which are never sent back through the
    OpenAI proxy when the agent ends without a final LLM call.
    """
    history_root = Path(working_dir) / '.agent_history'
    if not history_root.is_dir():
        return []

    try:
        index_files = sorted(history_root.rglob('index.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not index_files:
            return []

        with open(index_files[0]) as f:
            index_data = json.load(f)

        messages_meta = index_data.get('messages', [])
        msgs_dir = index_files[0].parent / 'messages'
        if not msgs_dir.is_dir():
            return []

        trailing_tool_ids: list[str] = []
        for m in reversed(messages_meta):
            if m.get('role') == 'tool':
                trailing_tool_ids.append(m['id'])
            else:
                break
        trailing_tool_ids.reverse()

        if not trailing_tool_ids:
            return []

        results: list[dict[str, Any]] = []
        for tid in trailing_tool_ids[-max_results:]:
            msg_file = msgs_dir / f'{tid}.json'
            if not msg_file.is_file():
                continue
            with open(msg_file) as f:
                msg = json.load(f)
            content_raw = msg.get('message', '')
            try:
                content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
            except (json.JSONDecodeError, ValueError):
                content = content_raw

            tool_results_in_msg = []
            if isinstance(content, dict):
                inner = content.get('content', [])
                tool_results_in_msg = inner if isinstance(inner, list) else [content]
            elif isinstance(content, list):
                tool_results_in_msg = content

            for tr in tool_results_in_msg:
                if not isinstance(tr, dict):
                    continue
                tc_id = tr.get('toolCallId', '')
                tool_name = tr.get('toolName', '')
                result_data = tr.get('result', '')
                result_str = (
                    json.dumps(result_data, ensure_ascii=False) if isinstance(result_data, dict) else str(result_data)
                )
                results.append({
                    'id': tc_id,
                    'name': tool_name,
                    'result': result_str[:4000],
                })

        logger.debug('Read %d trailing tool results from .agent_history', len(results))
        return results
    except Exception:
        logger.warning('Failed to read .agent_history', exc_info=True)
        return []


async def _run_pre_start_script(
    script: dict[str, Any],
    working_dir: str,
    base_env: dict[str, str],
) -> None:
    """Execute a single pre-start script.  Raises if fail_on_error and exit != 0."""
    import shlex

    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    interpreter = script.get('interpreter', 'bash')
    code = script.get('code', '')
    timeout = float(script.get('timeout', 300.0))
    cwd = script.get('cwd') or working_dir
    fail_on_error = bool(script.get('fail_on_error', True))

    if interpreter == 'bash':
        cmd = f'bash -c {shlex.quote(code)}'
    elif interpreter == 'python':
        py = find_runtime_binary('python') or find_runtime_binary('python3') or 'python3'
        cmd = f'{shlex.quote(py)} -c {shlex.quote(code)}'
    else:
        raise ValueError(f'Unsupported pre-start interpreter: {interpreter}')

    script_env = dict(base_env)
    if script.get('env'):
        script_env.update(script['env'])

    logger.debug(f'Running pre-start ({interpreter}) in {cwd}')
    result = await run_command(RunCommandRequest(command=cmd, cwd=cwd, timeout=timeout, extra_env=script_env))
    if result.stdout:
        logger.debug(f'[pre-start stdout]\n{result.stdout}')
    if result.stderr:
        logger.debug(f'[pre-start stderr]\n{result.stderr}')

    ok = result.status == CommandRunStatus.Finished and (result.return_code or 0) == 0
    if not ok and fail_on_error:
        raise RuntimeError(
            f'Pre-start script failed (status={result.status}, rc={result.return_code}): '
            f'{result.stderr or result.stdout or ""}'
        )


__all__ = ['setup_and_run_agent_server']
