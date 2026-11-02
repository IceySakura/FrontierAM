"""Runtime-side setup and execution for Hermes Agent."""

from __future__ import annotations

import json
import logging
import math
import shlex
import shutil
import tempfile
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_PROXY_API_KEY,
    build_runtime_path,
    find_free_port,
    unpack_command_result,
)

from .installer import DEFAULT_HERMES_VERSION, DEFAULT_INSTALL_ROOT, install_hermes_agent


logger = logging.getLogger(__name__)


DEFAULT_PROVIDER = 'custom'
DEFAULT_TOOLSETS = 'hermes-cli'
DEFAULT_HOME_ROOT = f'{DEFAULT_AGENT_INSTALL_ROOT}/hermes-home'


@dataclass
class HermesResult:
    """Result of running Hermes Agent in the runtime."""

    success: bool
    output: str
    stderr: str
    exit_code: int
    execution_time: float


def build_hermes_config(
    *,
    model: str,
    working_dir: str,
    proxy_port: int,
    toolsets: str = DEFAULT_TOOLSETS,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Build a minimal Hermes config bound to the local OpenAI proxy."""
    config: dict[str, Any] = {
        'model': {
            'provider': DEFAULT_PROVIDER,
            'default': model,
            'base_url': f'http://127.0.0.1:{proxy_port}/v1',
            'api_key': DEFAULT_PROXY_API_KEY,
            'api_mode': 'chat_completions',
        },
        'terminal': {
            'backend': 'local',
            'cwd': working_dir,
            'timeout': 180,
            'lifetime_seconds': 300,
        },
        'display': {
            'streaming': False,
            'tool_progress': 'off',
        },
        'toolsets': [item.strip() for item in toolsets.split(',') if item.strip()],
    }
    if max_turns is not None:
        config['agent'] = {'max_turns': max_turns}
    return config


def build_hermes_command(
    *,
    hermes_path: str | Path,
    task: str,
    model: str,
    toolsets: str = DEFAULT_TOOLSETS,
) -> list[str]:
    """Build the Hermes one-shot command for a single agent turn."""
    return [
        str(hermes_path),
        '--model',
        model,
        '--provider',
        DEFAULT_PROVIDER,
        '--toolsets',
        toolsets,
        '-z',
        task,
    ]


def _write_config(config_path: Path, config: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML, and avoids depending on PyYAML inside the runtime.
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')


def _append_artifact_path(stderr: str, artifact_dir: Path) -> str:
    note = f'Hermes runtime artifacts preserved at {artifact_dir}'
    return f'{stderr}\n{note}'.strip() if stderr else note


def _read_jsonl_files(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for path in sorted(root.glob('**/*.jsonl')):
        try:
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    records.append({'raw': line, 'parse_error': str(exc), 'path': str(path)})
        except Exception:
            continue
    return records


async def setup_and_run_hermes(
    task: str,
    model: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str = DEFAULT_HERMES_VERSION,
    toolsets: str = DEFAULT_TOOLSETS,
    install_root: str | None = None,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Install Hermes, configure it against the local proxy, then run one agent turn."""
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.executors.command import RunCommandRequest, run_command
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy

    rt = require_runtime_control()

    start_time = time.time()
    run_root = Path(tempfile.mkdtemp(prefix='hermes-run-'))
    home_dir = run_root / 'home'
    config_path = home_dir / 'config.yaml'
    preserve_artifacts = False

    home_dir.mkdir(parents=True, exist_ok=True)
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    try:
        hermes_bin = await install_hermes_agent(
            version=version,
            install_root=install_root or DEFAULT_INSTALL_ROOT,
        )
    except Exception as exc:
        preserve_artifacts = True
        return {
            'success': False,
            'output': '',
            'stderr': _append_artifact_path(f'Failed to install Hermes Agent: {exc}', run_root),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    proxy_port = find_free_port()
    proxy = OpenAIProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        models=[model],
    )

    try:
        await proxy.start()
    except Exception as exc:
        preserve_artifacts = True
        return {
            'success': False,
            'output': '',
            'stderr': _append_artifact_path(f'Failed to start OpenAI proxy: {exc}', run_root),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    try:
        _write_config(
            config_path,
            build_hermes_config(
                model=model,
                working_dir=working_dir,
                proxy_port=proxy_port,
                toolsets=toolsets,
                max_turns=max_turns,
            ),
        )

        hermes_env = {
            'HOME': str(home_dir),
            'HERMES_HOME': str(home_dir),
            'OPENAI_API_KEY': DEFAULT_PROXY_API_KEY,
            'CUSTOM_BASE_URL': f'http://127.0.0.1:{proxy_port}/v1',
            'HERMES_INFERENCE_MODEL': model,
            'HERMES_INFERENCE_PROVIDER': DEFAULT_PROVIDER,
            'HERMES_YOLO_MODE': '1',
            'HERMES_ACCEPT_HOOKS': '1',
            'HERMES_QUIET': '1',
            'HERMES_REDACT_SECRETS': '0',
            'PYTHONUNBUFFERED': '1',
            'PATH': build_runtime_path(str(Path(hermes_bin).parent)),
            'CI': '1',
            'NO_COLOR': '1',
            'http_proxy': '',
            'https_proxy': '',
            'HTTP_PROXY': '',
            'HTTPS_PROXY': '',
            'no_proxy': 'localhost,127.0.0.1',
            'NO_PROXY': 'localhost,127.0.0.1',
        }
        if max_turns is not None:
            hermes_env['HERMES_MAX_TURNS'] = str(max_turns)

        cmd = build_hermes_command(
            hermes_path=hermes_bin,
            task=task,
            model=model,
            toolsets=toolsets,
        )
        cli_timeout = max(1, math.ceil(timeout))
        result = await run_command(
            RunCommandRequest(
                command=shlex.join(cmd),
                cwd=working_dir,
                timeout=cli_timeout,
                extra_env=hermes_env,
            )
        )

        stdout, stderr, exit_code = unpack_command_result(
            result,
            timeout_message='Hermes execution timed out',
            error_message='Hermes command execution error',
        )

        execution_time = time.time() - start_time
        success = exit_code == 0
        if not success:
            preserve_artifacts = True
            stderr = _append_artifact_path(stderr, run_root)

        return {
            'success': success,
            'output': stdout,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': execution_time,
            'transcript': _read_jsonl_files(home_dir / 'sessions'),
        }
    finally:
        try:
            await proxy.stop()
        finally:
            if not preserve_artifacts:
                shutil.rmtree(run_root, ignore_errors=True)


__all__ = [
    'DEFAULT_HOME_ROOT',
    'DEFAULT_PROVIDER',
    'DEFAULT_TOOLSETS',
    'HermesResult',
    'build_hermes_command',
    'build_hermes_config',
    'setup_and_run_hermes',
]
