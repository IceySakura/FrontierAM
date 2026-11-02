"""Runtime-side installation and execution for the pi coding agent CLI.

pi is driven headless (``pi -p``) and its model calls are routed through a
local OpenAI-compatible proxy back to the host-side Nexus client over
Socket.IO.  pi has no ``ANTHROPIC_BASE_URL``-style environment override, so the
routing is expressed as a ``models.json`` provider entry pointing at the proxy.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time

from pathlib import Path
from typing import Any

from nexus.extensions.agents.installed_common import (
    DEFAULT_CLI_STREAM_READER_LIMIT,
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    DEFAULT_RUNTIME_BIN_DIR,
    build_runtime_path,
    find_free_port,
)

from .installer import DEFAULT_INSTALL_DIR, DEFAULT_PI_VERSION


logger = logging.getLogger(__name__)

PI_PROVIDER_ID = 'nexus-proxy'
DEFAULT_MODEL_ID = 'nexus-proxy-model'
DEFAULT_CONTEXT_WINDOW = 200_000
DEFAULT_MAX_TOKENS = 16_384
DEFAULT_TOOLS = ('read', 'bash', 'edit', 'write', 'grep', 'find', 'ls')
"""pi leaves grep/find/ls off by default; coding tasks generally want them."""

AGENT_DIR_ENV = 'PI_CODING_AGENT_DIR'
SESSION_DIR_ENV = 'PI_CODING_AGENT_SESSION_DIR'

FAILED_STOP_REASONS = frozenset({'error', 'aborted'})
"""Terminal ``stopReason`` values pi reports for a turn that did not succeed."""

_PROTECTED_ENV = frozenset({
    AGENT_DIR_ENV,
    SESSION_DIR_ENV,
    'HOME',
    'PATH',
    'PI_OFFLINE',
    'PI_SKIP_VERSION_CHECK',
    'PWD',
})


def build_models_config(
    *,
    proxy_port: int,
    model_id: str = DEFAULT_MODEL_ID,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning: bool = False,
) -> dict[str, Any]:
    """Build the ``models.json`` that points pi at the local Nexus proxy.

    The ``compat`` block is required, not cosmetic.  pi's ``detectCompat`` keys
    off the provider name and base URL; an unknown ``127.0.0.1`` host is treated
    as stock OpenAI, so pi would otherwise send ``developer``-role messages,
    ``max_completion_tokens``, ``store`` and ``reasoning_effort``.  The Nexus
    proxy forwards request bodies to the host client untouched, so anything
    wrong here reaches the upstream model as-is.
    """
    return {
        'providers': {
            PI_PROVIDER_ID: {
                'name': 'Nexus Proxy',
                'baseUrl': f'http://127.0.0.1:{proxy_port}/v1',
                'api': 'openai-completions',
                # A literal (non-`$`/`!`) value marks the provider authenticated
                # without a `/login` round-trip.
                'apiKey': DEFAULT_PROXY_API_KEY,
                'compat': {
                    'supportsStore': False,
                    'supportsDeveloperRole': False,
                    'supportsReasoningEffort': False,
                    'maxTokensField': 'max_tokens',
                },
                'models': [
                    {
                        'id': model_id,
                        'name': model_id,
                        'contextWindow': context_window,
                        'maxTokens': max_tokens,
                        'reasoning': reasoning,
                        'input': ['text'],
                        'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
                    },
                ],
            },
        },
    }


def build_pi_argv(
    binary: str,
    task: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    output_format: str = 'json',
    tools: tuple[str, ...] | list[str] | None = DEFAULT_TOOLS,
    load_context_files: bool = False,
    trust_project_files: bool = False,
    load_extensions: bool = False,
    save_session: bool = False,
    thinking: str | None = None,
    extra_cli_args: list[str] | None = None,
) -> list[str]:
    """Assemble the pi command line.

    ``tools`` is authoritative: any empty value (``[]`` or ``None``) means "no
    tools" and emits ``--no-tools``.  Callers wanting pi's built-ins pass them
    explicitly -- ``setup_and_run_pi`` and ``run_pi`` both normalise their own
    ``None`` to :data:`DEFAULT_TOOLS` before reaching here.

    ``load_context_files`` and ``trust_project_files`` are independent in pi:
    ``--no-approve`` gates executable project resources (``.pi/`` settings,
    extensions, skills) while ``AGENTS.md``/``CLAUDE.md`` loading is gated only
    by ``--no-context-files``.  Both default off so a run depends on the task
    prompt alone -- pi walks every ancestor directory of the working dir looking
    for context files, which would otherwise leak host repo instructions into
    the prompt and make benchmark results unreproducible.
    """
    argv = [binary, '--model', f'{PI_PROVIDER_ID}/{model_id}']

    if output_format != 'text':
        argv += ['--mode', output_format]
    if tools:
        argv += ['--tools', ','.join(tools)]
    else:
        # An empty allowlist has to be stated explicitly: given neither --tools
        # nor --no-tools, pi falls back to read/bash/edit/write (its
        # defaultActiveToolNames), so a caller asking for no tools would still
        # get shell and write access.  Verified against pi 0.84.1 on the wire.
        argv.append('--no-tools')
    if thinking:
        argv += ['--thinking', thinking]

    argv.append('--approve' if trust_project_files else '--no-approve')
    if not load_context_files:
        argv.append('--no-context-files')
    if not load_extensions:
        argv.append('--no-extensions')
    if not save_session:
        argv.append('--no-session')

    if extra_cli_args:
        argv += list(extra_cli_args)

    # `-p` must stay adjacent to the prompt, and the prompt must come last so a
    # task starting with `-` is not parsed as a flag.
    argv += ['-p', task]
    return argv


def _build_pi_env(
    *,
    working_dir: str,
    session_root: Path,
    extra_env: dict[str, str] | None,
) -> dict[str, str]:
    """Merge user configuration while keeping Nexus routing authoritative."""
    supplied = dict(extra_env or {})
    ignored = sorted(set(supplied) & _PROTECTED_ENV)
    if ignored:
        logger.warning('Ignoring protected pi environment overrides: %s', ', '.join(ignored))
    allowed = {key: value for key, value in supplied.items() if key not in _PROTECTED_ENV}

    home = session_root / 'home'
    agent_dir = session_root / 'agent'
    sessions = session_root / 'sessions'
    for path in (home, agent_dir, sessions):
        path.mkdir(parents=True, exist_ok=True)

    enforced = {
        'HOME': str(home),
        # Redirect every pi config path (models.json, auth.json, settings.json)
        # away from the real ~/.pi so concurrent runs cannot see each other.
        AGENT_DIR_ENV: str(agent_dir),
        SESSION_DIR_ENV: str(sessions),
        'PATH': build_runtime_path(DEFAULT_INSTALL_DIR, DEFAULT_RUNTIME_BIN_DIR),
        'PWD': working_dir,
        # No registry/version chatter from a sandbox with no outbound access.
        'PI_OFFLINE': '1',
        'PI_SKIP_VERSION_CHECK': '1',
        'CI': '1',
        'NO_COLOR': '1',
    }
    return {**allowed, **enforced}


async def _pipe_to_logger(
    stream: asyncio.StreamReader,
    prefix: str,
    level: int,
    buffer: list[str],
) -> None:
    while True:
        try:
            line = await stream.readline()
        except (asyncio.LimitOverrunError, ValueError):
            # A single JSONL event can exceed the stream limit; keep draining
            # rather than losing the rest of the run's output.
            logger.warning('%sdropped an oversized output line', prefix)
            continue
        if not line:
            break
        text = line.decode('utf-8', errors='replace').rstrip('\n')
        buffer.append(text)
        logger.log(level, '%s%s', prefix, text)


async def _terminate_process_tree(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    if os.name == 'posix':
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
    else:  # pragma: no cover - Nexus runtimes are Linux, but keep local development safe.
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
        return
    except asyncio.TimeoutError:
        pass
    if os.name == 'posix':
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    else:  # pragma: no cover
        proc.kill()
    with contextlib.suppress(Exception):
        await proc.wait()


async def _run_pi_streaming(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
) -> tuple[int, str, str]:
    from nexus.utils.shell import get_os_env_without_ld_library_path

    base_env = get_os_env_without_ld_library_path()
    base_env['BASH_ENV'] = '~/.bashrc'
    merged_env = {**base_env, **env}
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=merged_env,
        limit=DEFAULT_CLI_STREAM_READER_LIMIT,
        start_new_session=os.name == 'posix',
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_task = asyncio.create_task(_pipe_to_logger(proc.stdout, '[pi] ', logging.DEBUG, stdout_lines))
    stderr_task = asyncio.create_task(_pipe_to_logger(proc.stderr, '[pi:err] ', logging.WARNING, stderr_lines))
    timed_out = False
    try:
        exit_code = await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        await _terminate_process_tree(proc)
        exit_code = -1
    except BaseException:
        await _terminate_process_tree(proc)
        raise
    finally:
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    stderr = '\n'.join(stderr_lines)
    if timed_out:
        stderr = '\n'.join(part for part in (stderr, f'[timeout after {timeout:g}s]') if part)
    return exit_code or 0, '\n'.join(stdout_lines), stderr


def summarize_json_trace(stdout: str) -> dict[str, Any]:
    """Summarize a ``--mode json`` JSONL stream.

    Event names are taken from the runtime stream, not the exported types:
    tool activity arrives as ``tool_execution_start``/``tool_execution_end``,
    and a turn can span several ``message_start``/``message_end`` pairs.

    Also reports ``stop_reason``/``error_message`` from the final assistant
    message.  pi only converts a failed turn into a non-zero exit in text mode
    (``print-mode.ts`` guards that branch with ``mode === 'text'``), so in JSON
    mode a provider or model error still exits 0 -- the trace is the only signal
    that the run actually failed.
    """
    counts: dict[str, int] = {}
    final_text: list[str] = []
    stop_reason: str | None = None
    error_message: str | None = None
    malformed = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(event, dict):
            malformed += 1
            continue
        event_type = event.get('type')
        if not isinstance(event_type, str):
            malformed += 1
            continue
        counts[event_type] = counts.get(event_type, 0) + 1
        if event_type == 'agent_end':
            # A retried turn emits several agent_end events; the last one wins.
            for message in event.get('messages') or []:
                if not isinstance(message, dict) or message.get('role') != 'assistant':
                    continue
                reason = message.get('stopReason')
                stop_reason = reason if isinstance(reason, str) else None
                detail = message.get('errorMessage')
                error_message = detail if isinstance(detail, str) else None
                for part in message.get('content') or []:
                    if isinstance(part, dict) and part.get('type') == 'text' and isinstance(part.get('text'), str):
                        final_text.append(part['text'])

    return {
        'event_counts': counts,
        'turns': counts.get('turn_start', 0),
        'tool_calls': counts.get('tool_execution_start', 0),
        'malformed_lines': malformed,
        'final_text': '\n'.join(final_text),
        'stop_reason': stop_reason,
        'error_message': error_message,
    }


async def setup_and_run_pi(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str = DEFAULT_PI_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    model_id: str = DEFAULT_MODEL_ID,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning: bool = False,
    thinking: str | None = None,
    output_format: str = 'json',
    tools: list[str] | None = None,
    load_context_files: bool = False,
    trust_project_files: bool = False,
    load_extensions: bool = False,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
) -> dict[str, Any]:
    """Install pi, route its model calls through Nexus, and run one task."""
    from nexus.extensions.agents.pi_agent.installer import install_pi_via_npm
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy

    start_time = time.time()
    try:
        binary = await install_pi_via_npm(
            install_dir=DEFAULT_INSTALL_DIR,
            version=version,
            npm_registry=npm_registry,
        )
    except Exception as exc:
        logger.exception('Failed to install pi')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install pi: {exc}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    rt = require_runtime_control()
    proxy_port = find_free_port()
    proxy = OpenAIProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        models=[model_id],
    )
    try:
        await proxy.start()
    except BaseException as exc:
        # start() spawns a detached uvicorn task before its trailing sleep, so a
        # failure (or a cancellation during that sleep) can leave the listener
        # holding the port. Always tear it down before giving up.
        await proxy.stop()
        if isinstance(exc, asyncio.CancelledError):
            raise
        logger.exception('Failed to start the pi OpenAI proxy')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to start OpenAI Proxy: {exc}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    session_root = Path(tempfile.mkdtemp(prefix='nexus-pi-'))
    try:
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        launch_env = _build_pi_env(
            working_dir=str(work_path),
            session_root=session_root,
            extra_env=extra_env,
        )
        models_config = build_models_config(
            proxy_port=proxy_port,
            model_id=model_id,
            context_window=context_window,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )
        models_path = Path(launch_env[AGENT_DIR_ENV]) / 'models.json'
        models_path.write_text(json.dumps(models_config, ensure_ascii=False, indent=2), encoding='utf-8')

        cmd = build_pi_argv(
            str(binary),
            task,
            model_id=model_id,
            output_format=output_format,
            tools=tuple(tools) if tools is not None else DEFAULT_TOOLS,
            load_context_files=load_context_files,
            trust_project_files=trust_project_files,
            load_extensions=load_extensions,
            thinking=thinking,
            extra_cli_args=extra_cli_args,
        )
        logger.debug('Running pi in %s', work_path)
        exit_code, stdout, stderr = await _run_pi_streaming(
            cmd,
            cwd=str(work_path),
            env=launch_env,
            timeout=timeout,
        )

        result: dict[str, Any] = {
            'success': exit_code == 0,
            'output': stdout,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': time.time() - start_time,
        }
        if output_format == 'json':
            trace = summarize_json_trace(stdout)
            result['trace'] = trace
            # pi exits 0 in JSON mode even when the turn failed, so a provider or
            # model error would otherwise be reported as a successful run.
            if trace['stop_reason'] in FAILED_STOP_REASONS:
                detail = trace['error_message'] or f'pi turn ended with stopReason={trace["stop_reason"]}'
                logger.warning('pi reported a failed turn despite exit=%d: %s', exit_code, detail)
                result['success'] = False
                result['stderr'] = '\n'.join(part for part in (stderr, detail) if part)
            logger.info(
                'pi finished: %d turn(s), %d tool call(s), exit=%d, stop_reason=%s',
                trace['turns'],
                trace['tool_calls'],
                exit_code,
                trace['stop_reason'],
            )
        return result
    finally:
        await proxy.stop()
        shutil.rmtree(session_root, ignore_errors=True)


__all__ = [
    'AGENT_DIR_ENV',
    'DEFAULT_MODEL_ID',
    'DEFAULT_TOOLS',
    'FAILED_STOP_REASONS',
    'PI_PROVIDER_ID',
    'SESSION_DIR_ENV',
    'build_models_config',
    'build_pi_argv',
    'setup_and_run_pi',
    'summarize_json_trace',
]
