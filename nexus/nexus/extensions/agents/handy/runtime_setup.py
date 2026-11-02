"""Runtime-side installation and execution for the packaged Handy CLI."""

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
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    DEFAULT_RUNTIME_BIN_DIR,
    build_runtime_path,
    find_free_port,
)

from .installer import DEFAULT_HANDY_VERSION, DEFAULT_INSTALL_DIR, NPM_PACKAGE


logger = logging.getLogger(__name__)

DEFAULT_HANDY_MODEL = 'claude-opus-5'
DEFAULT_HANDY_MAX_OUTPUT_TOKENS = 64_000

_PROTECTED_ENV = frozenset({
    'ANTHROPIC_API_KEY',
    'ANTHROPIC_AUTH_TOKEN',
    'ANTHROPIC_BASE_URL',
    'ANTHROPIC_DEFAULT_OPUS_MODEL',
    'ANTHROPIC_MODEL',
    'CLAUDE_CODE_ENTRYPOINT',
    'CLAUDE_CODE_MAX_OUTPUT_TOKENS',
    'CLAUDE_CODE_USE_BEDROCK',
    'CLAUDE_CODE_USE_FOUNDRY',
    'CLAUDE_CODE_USE_VERTEX',
    'CLAUDE_CONFIG_DIR',
    'HOME',
    'HYNESS_APP_CONFIG',
    'HYNESS_DISTRIBUTION_NAME',
    'HYNESS_DISTRIBUTION_PACKAGE',
    'HYNESS_DISTRIBUTION_VERSION',
    'HYNESS_PI_TUI_YOLO',
    'IS_SANDBOX',
    'PATH',
    'PWD',
})

ARTIFACT_OUTPUT_DIR_ENV = 'HANDY_ARTIFACT_OUTPUT_DIR'
"""Handy writes ``submit_file_artifact`` deliveries here instead of uploading them."""

_ARTIFACT_METADATA_NAME = 'artifact.json'
_ARTIFACT_DIR_PREFIX = 'art_'


def _collect_artifact_manifest(artifact_dir: str | None) -> list[dict[str, Any]]:
    """List the artifacts Handy finished writing, newest last.

    Handy stages each delivery in ``.staging-<id>/`` and renames it to ``art_<id>/``
    only once the copy and its ``artifact.json`` are both complete, so a directory
    matching ``art_*`` is by construction finished — no completion marker needed.
    """
    if not artifact_dir:
        return []
    root = Path(artifact_dir)
    if not root.is_dir():
        return []

    manifest: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith(_ARTIFACT_DIR_PREFIX):
            continue
        metadata_path = entry / _ARTIFACT_METADATA_NAME
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except Exception as exc:
            logger.warning('Skipping unreadable Handy artifact metadata %s: %s', metadata_path, exc)
            continue
        relative_path = metadata.get('relativePath') or f'{entry.name}/files/{metadata.get("filename")}'
        file_path = root / relative_path
        if not file_path.is_file():
            logger.warning('Handy artifact %s names a missing file %s', entry.name, file_path)
            continue
        metadata['sandbox_path'] = str(file_path)
        metadata['size'] = file_path.stat().st_size
        manifest.append(metadata)
    return manifest


def _resolve_plugin_dirs(plugin_dirs: list[str] | None) -> list[str]:
    """Resolve requested plugin directories against the installed package.

    A relative path is taken against the installed ``@tencent/handy-agent``
    directory, so callers can name a plugin that ships inside the package —
    ``plugins/ui-ux-pro-max`` — without knowing the sandbox install prefix.

    A missing directory is dropped with a warning rather than passed through:
    Handy logs those at debug level and carries on without the plugin, which
    surfaces much later as "the subagent silently did not exist".
    """
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in plugin_dirs or []:
        entry = str(raw).strip()
        if not entry or entry == '.':
            # Both resolve to the package root, which would mount the whole install
            # as one plugin instead of failing.
            logger.warning('Ignoring empty Handy plugin dir entry: %r', raw)
            continue
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = Path(DEFAULT_INSTALL_DIR) / 'node_modules' / NPM_PACKAGE / candidate
        if not candidate.is_dir():
            logger.warning('Skipping Handy plugin dir that does not exist in the runtime: %s', candidate)
            continue
        path = str(candidate)
        # Handy rejects two plugins sharing a name, and dedupes identical paths
        # itself; collapsing repeats here keeps the command line honest either way.
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    if resolved:
        logger.debug('Loading %d Handy session plugin(s): %s', len(resolved), ', '.join(resolved))
    return resolved


def _build_handy_env(
    *,
    proxy_port: int,
    working_dir: str,
    session_root: Path,
    extra_env: dict[str, str] | None,
    model: str = DEFAULT_HANDY_MODEL,
    max_output_tokens: int = DEFAULT_HANDY_MAX_OUTPUT_TOKENS,
    artifact_dir: str | None = None,
) -> dict[str, str]:
    """Merge user tool configuration while keeping Nexus routing authoritative."""
    supplied = dict(extra_env or {})
    ignored = sorted(set(supplied) & _PROTECTED_ENV)
    if ignored:
        logger.warning('Ignoring protected Handy environment overrides: %s', ', '.join(ignored))
    allowed = {key: value for key, value in supplied.items() if key not in _PROTECTED_ENV}

    home = session_root / 'home'
    config = session_root / 'config'
    home.mkdir(parents=True, exist_ok=True)
    config.mkdir(parents=True, exist_ok=True)
    enforced = {
        'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{proxy_port}',
        'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
        'ANTHROPIC_MODEL': model,
        'ANTHROPIC_DEFAULT_OPUS_MODEL': model,
        'CLAUDE_CODE_MAX_OUTPUT_TOKENS': str(max_output_tokens),
        'HOME': str(home),
        'CLAUDE_CONFIG_DIR': str(config),
        'PATH': build_runtime_path(DEFAULT_INSTALL_DIR, DEFAULT_RUNTIME_BIN_DIR),
        'PWD': working_dir,
        'IS_SANDBOX': '1',
        'CLAUDE_CODE_ENTRYPOINT': 'nexus',
        'HYNESS_PI_TUI_YOLO': '1',
        'CI': '1',
        'NO_COLOR': '1',
    }
    if artifact_dir:
        # Keep deliveries on local disk instead of Handy's object-storage upload.
        # This must live outside session_root, which is deleted when the run ends.
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        enforced[ARTIFACT_OUTPUT_DIR_ENV] = artifact_dir
    return {**allowed, **enforced}


async def _pipe_to_logger(
    stream: asyncio.StreamReader,
    prefix: str,
    level: int,
    buffer: list[str],
) -> None:
    while True:
        line = await stream.readline()
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


async def _run_handy_streaming(
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
        start_new_session=os.name == 'posix',
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_task = asyncio.create_task(_pipe_to_logger(proc.stdout, '[handy] ', logging.DEBUG, stdout_lines))
    stderr_task = asyncio.create_task(_pipe_to_logger(proc.stderr, '[handy:err] ', logging.WARNING, stderr_lines))
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


async def setup_and_run_handy(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str = DEFAULT_HANDY_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    extra_env: dict[str, str] | None = None,
    model: str = DEFAULT_HANDY_MODEL,
    max_output_tokens: int = DEFAULT_HANDY_MAX_OUTPUT_TOKENS,
    conversion_log_level: str = 'INFO',
    thinking_preset: str = 'hunyuan',
    artifact_dir: str | None = None,
    plugin_dirs: list[str] | None = None,
) -> dict[str, object]:
    """Install Handy, route its model calls through Nexus, and run one task.

    When *artifact_dir* is set, Handy stores ``submit_file_artifact`` deliveries
    there and the returned dict carries an ``artifacts`` manifest describing them,
    so the host can fetch the files before the sandbox goes away.
    """
    from nexus.extensions.agents.handy.installer import install_handy_via_npm
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy

    start_time = time.time()
    try:
        binary = await install_handy_via_npm(
            install_dir=DEFAULT_INSTALL_DIR,
            version=version,
            npm_registry=npm_registry,
        )
    except Exception as exc:
        logger.exception('Failed to install Handy')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install Handy: {exc}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    rt = require_runtime_control()
    proxy_port = find_free_port()
    proxy = AnthropicProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        conversion_log_level=conversion_log_level,
        thinking_config=thinking_preset,
    )
    try:
        await proxy.start()
    except Exception as exc:
        logger.exception('Failed to start the Handy Anthropic proxy')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to start Anthropic Proxy: {exc}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    session_root = Path(tempfile.mkdtemp(prefix='nexus-handy-'))
    try:
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        launch_env = _build_handy_env(
            proxy_port=proxy_port,
            working_dir=str(work_path),
            session_root=session_root,
            extra_env=extra_env,
            model=model,
            max_output_tokens=max_output_tokens,
            artifact_dir=artifact_dir,
        )
        cmd = [str(binary), '--dangerously-skip-permissions']
        for plugin_dir in _resolve_plugin_dirs(plugin_dirs):
            cmd += ['--plugin-dir', plugin_dir]
        cmd += ['-p', task]
        logger.debug('Running Handy in %s', work_path)
        exit_code, stdout, stderr = await _run_handy_streaming(
            cmd,
            cwd=str(work_path),
            env=launch_env,
            timeout=timeout,
        )
        artifacts = _collect_artifact_manifest(artifact_dir)
        if artifacts:
            logger.debug('Handy delivered %d artifact(s) to %s', len(artifacts), artifact_dir)
        return {
            'success': exit_code == 0,
            'output': stdout,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': time.time() - start_time,
            'artifacts': artifacts,
        }
    finally:
        await proxy.stop()
        shutil.rmtree(session_root, ignore_errors=True)


__all__ = [
    'ARTIFACT_OUTPUT_DIR_ENV',
    'DEFAULT_HANDY_MAX_OUTPUT_TOKENS',
    'DEFAULT_HANDY_MODEL',
    'setup_and_run_handy',
]
