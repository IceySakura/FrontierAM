"""Runtime-side setup and execution for the WorkBuddy ``wb`` CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import signal
import subprocess
import time

from collections.abc import Callable
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from nexus.extensions.agents.installed_common import (
    DEFAULT_CLI_STREAM_READER_LIMIT,
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    build_runtime_path,
    find_free_port,
    find_nexus_runtime_binary,
    find_runtime_binary,
)

from .acp_output import (
    assistant_text,
    classify_wb_result,
    conversation_id,
    failure_fields,
    final_turn_result,
    parse_wb_json_lines,
    task_outcome,
    transcript_path,
)
from .installer import (
    DEFAULT_INSTALL_DIR,
    DEFAULT_WORKBUDDY_VERSION,
    REMOTE_TARBALL,
    get_installed_version,
    install_workbuddy,
)


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = 'general-model'
_CUSTOM_MODEL_PREFIX = 'custom-local:'
DEFAULT_PRESET = 'code'
DEFAULT_MODE = 'craft'
# The inner CodeBuddy CLI reads thinking switches from its user settings, not from models.json.
THINKING_SETTINGS_PATH = '~/.codebuddy/settings.json'

# Nexus owns model routing and the sandbox layout; extra_env must not redirect wb
# away from the local proxy or relocate its data dir.
_PROTECTED_ENV = frozenset({
    'CODEBUDDY_API_KEY',
    'CODEBUDDY_BASE_URL',
    'CODEBUDDY_IS_SANDBOX',
    'HOME',
    'IS_SANDBOX',
    'PATH',
    'WORKBUDDY_CONFIG_DIR',
    'WORKBUDDY_USER_DATA_DIR',
})

# The runtime's Nix-built Node bakes in an OpenSSL cert dir that does not exist in
# the image, so any HTTPS wb makes (marketplace, remote MCP) fails without these.
# Not protected: a different image may keep its CA bundle elsewhere.
_CA_ENV_DEFAULTS = {
    'SSL_CERT_FILE': '/etc/ssl/certs/ca-certificates.crt',
    'SSL_CERT_DIR': '/etc/ssl/certs',
    'NODE_EXTRA_CA_CERTS': '/etc/ssl/certs/ca-certificates.crt',
}

# wb logs this once the conversation is over and its sidecar has killed the agent session. The main
# process can still linger for hours afterwards, so this is the cue to stop waiting for it.
_SHUTDOWN_MARKER = '[wb:daemon-app-server] stopped'
_SHUTDOWN_GRACE_SECONDS = 20.0
_TERMINATE_GRACE_SECONDS = 10.0
_DRAIN_TIMEOUT_SECONDS = 10.0
_POLL_INTERVAL_SECONDS = 1.0


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


def build_models_config(
    proxy_url: str,
    model_id: str = DEFAULT_MODEL_ID,
    api_key: str = DEFAULT_PROXY_API_KEY,
    supports_reasoning: bool = False,
) -> dict[str, Any]:
    """Build ``~/.workbuddy/models.json`` so ``wb`` routes to the Nexus proxy.

    wb drops every reasoning setting, ``--effort`` included, for a model that does not
    declare ``supportsReasoning``.
    """
    return {
        'models': [
            {
                'id': model_id,
                'name': model_id,
                'vendor': 'nexus',
                'apiKey': api_key,
                'maxInputTokens': 200000,
                'maxOutputTokens': 16384,
                'url': proxy_url,
                'supportsToolCall': True,
                'supportsImages': True,
                'supportsReasoning': supports_reasoning,
            }
        ],
        'availableModels': [model_id],
    }


def build_thinking_settings(effort: str | None) -> dict[str, Any]:
    """Settings that make wb send ``reasoning_effort`` and read back ``reasoning_content``."""
    settings: dict[str, Any] = {'alwaysThinkingEnabled': True}
    if effort:
        settings['reasoningEffort'] = effort
    return settings


def cli_model_flag(model_id: str, version: str) -> str:
    """Id passed to ``wb --model``.

    5.9 copies ``models.json`` into the published product-config list and
    rewrites a bare id to ``custom-local:<id>``. ``validatePromptModelRoute``
    only accepts that published id, so ``--model general-model`` fails before
    any completion request. The request body still uses the bare id. 5.7.0
    takes the bare id, so the prefix stays off for that release.
    """
    if model_id.startswith((_CUSTOM_MODEL_PREFIX, 'custom:')) or not _uses_published_model_id(version):
        return model_id
    return f'{_CUSTOM_MODEL_PREFIX}{model_id}'


def _uses_published_model_id(version: str) -> bool:
    # npm prereleases like ``5.11.0-hotfix.202610272104`` are not PEP 440; compare the release part only.
    try:
        return version == 'latest' or Version(version.split('-', 1)[0]).release >= (5, 9)
    except InvalidVersion:
        return False


def write_models_json(
    workbuddy_home: Path,
    proxy_url: str,
    model_id: str,
    api_key: str,
    supports_reasoning: bool = False,
) -> Path:
    models_json = workbuddy_home / 'models.json'
    config = build_models_config(proxy_url, model_id, api_key, supports_reasoning=supports_reasoning)
    models_json.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    logger.debug('Wrote models.json at %s', models_json)
    return models_json


def build_workbuddy_argv(
    node_path: str,
    wb_js: str,
    task: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    effort: str | None = 'xhigh',
    preset: str | None = DEFAULT_PRESET,
    mode: str | None = None,
    json_output: bool = True,
    rollout_mode: bool = False,
    extra_cli_args: list[str] | None = None,
) -> list[str]:
    """Assemble ``node wb.js ... prompt``. The prompt stays last."""
    cmd = [node_path, wb_js, '--model', model_id]
    if preset:
        cmd.extend(['--preset', preset])
    if mode and mode != DEFAULT_MODE:
        cmd.extend(['--mode', mode])
    if effort:
        cmd.extend(['--effort', effort])
    if json_output:
        cmd.append('--json')
    if rollout_mode:
        if any(arg == '--features' or arg.startswith('--features=') for arg in extra_cli_args or []):
            raise ValueError('rollout_mode owns --features; do not override it in extra_cli_args')
        cmd.extend(['--features', '{"rollout":true}'])
    if extra_cli_args:
        cmd.extend(extra_cli_args)
    cmd.append(task)
    return cmd


def build_workbuddy_env(
    *,
    home: str,
    install_dir: str,
    proxy_base_url: str,
    api_key: str,
    config_dir: str,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    supplied = dict(extra_env or {})
    ignored = sorted(set(supplied) & _PROTECTED_ENV)
    if ignored:
        logger.warning('Ignoring protected WorkBuddy environment overrides: %s', ', '.join(ignored))
    allowed = {key: value for key, value in supplied.items() if key not in _PROTECTED_ENV}

    enforced = {
        'HOME': home,
        'PATH': build_runtime_path(
            str(Path(install_dir) / 'node_modules' / '.bin'),
            str(Path(install_dir) / 'package' / 'bin'),
            install_dir,
        ),
        'IS_SANDBOX': '1',
        # The isolated-sandbox full pass that makes bypassPermissions skip HIGH/CRITICAL prompts
        # too (wb never reads the bare IS_SANDBOX above). Only reaches the agent when wb.js is
        # bypassed: the wrapper rebuilds the worker env from an allowlist that drops every
        # CODEBUDDY_* key, which is why headless relies on --permission-mode fullAccess instead.
        'CODEBUDDY_IS_SANDBOX': '1',
        'CODEBUDDY_BASE_URL': proxy_base_url,
        'CODEBUDDY_API_KEY': api_key,
        'WORKBUDDY_CONFIG_DIR': config_dir,
        # AppConfigService reads app-config.json from here, and this is the only resolver
        # input we control: the fallback chain ends at an internal runtime context we cannot
        # see from outside. Pinning it keeps the path config patches write to and the path wb
        # reads from the same one.
        'WORKBUDDY_USER_DATA_DIR': str(Path(config_dir) / 'app'),
    }
    return {'WORKBUDDY_LOG_LEVEL': 'info', **_CA_ENV_DEFAULTS, **allowed, **enforced}


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
        cmd = f'bash -c {shlex.quote(code)}'
    elif interpreter == 'python':
        # The runtime's own interpreter first: a testbed image is free to ship anything as
        # `python`, and some ship one too old to parse the scripts we send (a csharp image
        # failed here on an f-string).
        py = find_nexus_runtime_binary('python3') or find_runtime_binary('python') or 'python3'
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


async def _pipe_to_logger(
    stream: asyncio.StreamReader,
    prefix: str,
    level: int,
    buffer: list[str],
    on_line: Callable[[str], None] | None = None,
) -> None:
    while True:
        try:
            line = await stream.readline()
        except (asyncio.LimitOverrunError, ValueError):
            # A single --json ACP frame (e.g. a large tool_call_update) can exceed the
            # reader limit. Drop that line loudly instead of killing the whole pump.
            logger.warning('%sdropped a line longer than %d bytes', prefix, DEFAULT_CLI_STREAM_READER_LIMIT)
            continue
        if not line:
            break
        text = line.decode('utf-8', errors='replace').rstrip('\n')
        buffer.append(text)
        logger.log(level, '%s%s', prefix, text)
        if on_line is not None:
            on_line(text)


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _stop_process_group(pgid: int) -> None:
    """SIGTERM wb and everything it spawned, then SIGKILL whatever is left."""
    loop = asyncio.get_running_loop()
    for sig, grace in ((signal.SIGTERM, _TERMINATE_GRACE_SECONDS), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = loop.time() + grace
        while loop.time() < deadline and _process_group_alive(pgid):
            await asyncio.sleep(0.2)


async def _run_workbuddy_streaming(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: float,
    should_abort: Callable[[], bool] | None = None,
) -> tuple[int, str, str]:
    from nexus.utils.shell import get_os_env_without_ld_library_path

    base_env = get_os_env_without_ld_library_path()
    base_env['BASH_ENV'] = '~/.bashrc'
    merged_env = {**base_env, **env}

    # A new session keeps wb's sidecar, MCP servers and any background shells in one process group.
    # They inherit our pipes, so they must be killed together with wb or the pipes never close.
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        # Without a prompt wb reads prompts from stdin; an inherited stdin would
        # make a mis-assembled argv hang until the timeout instead of failing.
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

    shutdown_seen = asyncio.Event()

    def watch_stderr(text: str) -> None:
        if _SHUTDOWN_MARKER in text:
            shutdown_seen.set()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_task = asyncio.create_task(_pipe_to_logger(proc.stdout, '[workbuddy] ', logging.DEBUG, stdout_lines))
    stderr_task = asyncio.create_task(
        _pipe_to_logger(proc.stderr, '[workbuddy:err] ', logging.WARNING, stderr_lines, on_line=watch_stderr)
    )

    # proc.wait() only resolves after every holder of the pipes has exited, so poll the main process.
    # Cancellation lands on the poll sleep. Stop and drain in finally so a cancelled run cannot
    # leave wb, or the children that inherited its pipes, running.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    shutdown_at: float | None = None
    outcome = 'exited'
    try:
        while proc.returncode is None:
            if should_abort is not None and should_abort():
                outcome = 'prompt_identity'
                break
            now = loop.time()
            if now >= deadline:
                outcome = 'timeout'
                break
            if shutdown_seen.is_set():
                if shutdown_at is None:
                    shutdown_at = now
                elif now - shutdown_at >= _SHUTDOWN_GRACE_SECONDS:
                    outcome = 'finished'
                    break
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        if outcome == 'finished':
            logger.warning('WorkBuddy finished but its process did not exit; stopping it')
    finally:
        await _stop_process_group(proc.pid)
        readers = (stdout_task, stderr_task)
        _, pending = await asyncio.wait(readers, timeout=_DRAIN_TIMEOUT_SECONDS)
        for task in pending:
            task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)

    stdout = '\n'.join(stdout_lines)
    stderr = '\n'.join(stderr_lines)
    if outcome == 'timeout':
        return -1, stdout, stderr + '\n[timeout]'
    if outcome == 'prompt_identity':
        return -1, stdout, stderr + '\n[WorkBuddy prompt identity verification failed]'
    if outcome == 'finished':
        return 0, stdout, stderr
    return proc.returncode or 0, stdout, stderr


async def setup_and_run_workbuddy(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    npm_token: str | None = None,
    npm_username: str | None = None,
    version: str = DEFAULT_WORKBUDDY_VERSION,
    package_url: str | None = None,
    package_sha256: str | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    effort: str | None = 'xhigh',
    thinking: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
    preset: str | None = DEFAULT_PRESET,
    mode: str | None = None,
    json_output: bool = True,
    rollout_mode: bool = False,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    config_file_patches: list[dict[str, Any]] | None = None,
    request_capture_dir: str | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    remote_part_paths: list[str] | None = None,
    assembled_tar: str = REMOTE_TARBALL,
) -> dict[str, Any]:
    """Install WorkBuddy, start the OpenAI proxy, and run ``wb``."""
    from nexus.runtime.context import require_runtime_control

    from .prompt_identity import prepare_builtin_prompt_plugins
    from .proxy import WorkBuddyProxy

    rt = require_runtime_control()
    start_time = time.time()

    try:
        wb_js = await install_workbuddy(
            DEFAULT_INSTALL_DIR,
            version=version,
            package_url=package_url,
            package_sha256=package_sha256,
            remote_part_paths=remote_part_paths,
            assembled_tar=assembled_tar,
            npm_registry=npm_registry,
            npm_token=npm_token,
            npm_username=npm_username,
        )
        logger.debug('WorkBuddy entrypoint: %s', wb_js)
    except Exception as exc:
        logger.error('Failed to install WorkBuddy: %s', exc)
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install WorkBuddy: {exc}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'status': 'error',
            'failure_kind': 'infrastructure',
            'failure_stage': 'install',
        }

    proxy_port = find_free_port()
    logger.debug('Starting OpenAI Proxy on port %s...', proxy_port)
    proxy_class: type[WorkBuddyProxy] = WorkBuddyProxy
    proxy_kwargs: dict[str, Any] = {'chat_template_kwargs': chat_template_kwargs, 'verify_prompt': rollout_mode}
    if request_capture_dir:
        # NOTE(luxyhliu): metadata capture is opt-in because request bodies contain full task context.
        from .proxy import WorkBuddyMetadataCaptureProxy

        proxy_class = WorkBuddyMetadataCaptureProxy
        proxy_kwargs['capture_dir'] = request_capture_dir
    proxy = proxy_class(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        models=[model_id],
        **proxy_kwargs,
    )
    try:
        await proxy.start()
        logger.debug('OpenAI Proxy started')
    except Exception as exc:
        logger.error('Failed to start OpenAI Proxy: %s', exc)
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to start OpenAI Proxy: {exc}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'status': 'error',
            'failure_kind': 'infrastructure',
            'failure_stage': 'proxy',
        }

    try:
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        home = os.environ.get('HOME', '/root')
        config_dir = str(Path(home) / '.workbuddy')
        Path(config_dir).mkdir(parents=True, exist_ok=True)

        proxy_base = f'http://127.0.0.1:{proxy_port}/v1'
        proxy_url = f'{proxy_base}/chat/completions'
        write_models_json(Path(config_dir), proxy_url, model_id, DEFAULT_PROXY_API_KEY, supports_reasoning=thinking)

        wb_env = build_workbuddy_env(
            home=home,
            install_dir=DEFAULT_INSTALL_DIR,
            proxy_base_url=proxy_base,
            api_key=DEFAULT_PROXY_API_KEY,
            config_dir=config_dir,
            extra_env=extra_env,
        )

        if thinking:
            _apply_json_patch(THINKING_SETTINGS_PATH, build_thinking_settings(effort))
        for patch in config_file_patches or []:
            _apply_json_patch(patch['path'], patch.get('content') or {})

        for script in pre_start_scripts or []:
            try:
                await _run_pre_start_script(script, working_dir, wb_env)
            except Exception as exc:
                return {
                    'success': False,
                    'output': '',
                    'stderr': f'Pre-start script failed: {exc}',
                    'exit_code': -1,
                    'execution_time': time.time() - start_time,
                    'status': 'error',
                    'failure_kind': 'infrastructure',
                    'failure_stage': 'pre_start',
                }

        prompt_plugins = None
        if rollout_mode:
            try:
                if not preset:
                    raise ValueError('rollout_mode requires an explicit WorkBuddy preset')
                prompt_plugins = prepare_builtin_prompt_plugins(wb_js, Path(config_dir), preset, mode or DEFAULT_MODE)
                logger.info('Verified WorkBuddy prompt plugins: %s', prompt_plugins['enabled_plugins'])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return {
                    'success': False,
                    'output': '',
                    'stderr': f'WorkBuddy prompt preparation failed: {exc}',
                    'exit_code': -1,
                    'execution_time': time.time() - start_time,
                    'status': 'error',
                    'failure_kind': 'infrastructure',
                    'failure_stage': 'prompt_identity',
                }

        node_path = find_nexus_runtime_binary('node')
        if not node_path:
            raise RuntimeError('Node.js is required but not found in runtime')
        logger.debug('Using Node.js binary for WorkBuddy CLI: %s', node_path)

        cmd = build_workbuddy_argv(
            node_path,
            str(wb_js),
            task,
            # A dist-tag such as ``latest`` only says which release once installed.
            model_id=cli_model_flag(model_id, get_installed_version(DEFAULT_INSTALL_DIR) or version),
            effort=effort,
            preset=preset,
            mode=mode,
            json_output=json_output,
            rollout_mode=rollout_mode,
            extra_cli_args=extra_cli_args,
        )
        logger.debug('Running WorkBuddy: %s', ' '.join(shlex.quote(part) for part in cmd))

        exit_code, stdout, stderr = await _run_workbuddy_streaming(
            cmd=cmd,
            cwd=working_dir,
            env=wb_env,
            timeout=timeout,
            should_abort=(lambda: bool(proxy.prompt_guard and proxy.prompt_guard.failure_reason))
            if rollout_mode
            else None,
        )
        execution_time = time.time() - start_time

        updates, results = parse_wb_json_lines(stdout) if json_output else ([], [])
        result_line = final_turn_result(results)
        outcome = task_outcome(result_line)
        status, detail = classify_wb_result(result_line, exit_code, json_output=json_output)
        failure_kind, failure_stage = failure_fields(status, outcome)
        prompt_identity = proxy.prompt_guard.report() if proxy.prompt_guard is not None else None
        if prompt_identity is not None and not prompt_identity['verified']:
            status, detail = 'error', prompt_identity['failure_reason'] or 'No verified WorkBuddy model request'
            failure_kind, failure_stage = 'infrastructure', 'prompt_identity'

        logger.info(
            'WorkBuddy finished with exit code %s in %.2fs (status=%s, taskOutcome=%s)',
            exit_code,
            execution_time,
            status,
            outcome,
        )
        if detail:
            stderr = '\n'.join(part for part in (stderr, f'[wb] {detail}') if part)

        payload: dict[str, Any] = {
            'success': status == 'completed',
            # Keep the raw JSONL: it is the only copy of the tool calls the agent made.
            'output': stdout,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': execution_time,
            'status': status,
        }
        if prompt_identity is not None:
            payload['prompt_identity'] = {**prompt_identity, 'plugins': prompt_plugins}
            logger.info('WorkBuddy prompt identity: %s', json.dumps(payload['prompt_identity']))
        if json_output:
            payload['assistant_text'] = assistant_text(updates)
            payload['task_outcome'] = outcome
            payload['stop_reason'] = (result_line or {}).get('stopReason')
            payload['transcript_path'] = transcript_path(result_line)
            payload['conversation_id'] = conversation_id(result_line)
        if failure_kind is not None:
            payload['failure_kind'] = failure_kind
            payload['failure_stage'] = failure_stage
        return payload
    finally:
        logger.debug('Stopping OpenAI Proxy...')
        await proxy.stop()


__all__ = [
    'DEFAULT_MODE',
    'DEFAULT_MODEL_ID',
    'DEFAULT_PRESET',
    'build_models_config',
    'build_workbuddy_argv',
    'cli_model_flag',
    'build_workbuddy_env',
    'setup_and_run_workbuddy',
    'write_models_json',
]
