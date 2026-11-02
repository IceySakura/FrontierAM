"""Runtime compatibility helpers for OSWorld evaluator getters."""

from __future__ import annotations

import asyncio
import base64
import os
import posixpath
import shlex
import tempfile

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.os_world.evaluation.errors import EvaluatorInfrastructureError
from nexus.extensions.tasks.os_world.execution.commands import command_to_shell, osworld_user_command
from nexus.runtime.executors.models.file import FileItem
from nexus.utils.async_http import http_client_with_trust_env as http_client


CACHE_DIR_CONFIG_KEY = '__nexus_cache_dir'


def cache_dir_from_config(config: dict[str, Any]) -> Path:
    cache_dir = config.get(CACHE_DIR_CONFIG_KEY)
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix='nexus-osworld-evaluator-')
    path = Path(str(cache_dir))
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_cache_path(config: dict[str, Any], dest: str) -> Path:
    return cache_dir_from_config(config) / dest


def response_stdout(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get('output') or response.get('stdout') or '')
    return str(getattr(response, 'stdout', '') or '')


def response_stderr(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get('error') or response.get('stderr') or '')
    return str(getattr(response, 'stderr', '') or '')


def response_returncode(response: Any) -> int | None:
    if isinstance(response, dict):
        value = response.get('returncode', response.get('return_code'))
    else:
        value = getattr(response, 'return_code', None)
    return int(value) if value is not None else None


def _command_as_user(runtime: Any, default: bool) -> bool:
    value = getattr(runtime, 'osworld_command_as_user', None)
    return default if value is None else bool(value)


def runtime_headers(runtime: Any) -> dict[str, str] | None:
    handle = getattr(runtime, 'handle', None)
    headers = getattr(handle, 'headers', None)
    return dict(headers) if headers else None


def runtime_port_url(runtime: Any, port: int) -> str | None:
    value = getattr(runtime, 'port_mappings', {}).get(port)
    return str(value).rstrip('/') if value else None


async def execute_command(
    runtime: Any,
    command: str | list[str],
    *,
    shell: bool = False,
    timeout: int = 120,
    as_user: bool = True,
) -> dict[str, Any]:
    """Execute a command through raw OSWorld or Nexus runtime surfaces."""

    if hasattr(runtime, 'execute'):
        result = await runtime.execute(command, shell=shell)
        return result if isinstance(result, dict) else {'output': str(result), 'error': '', 'returncode': 0}

    command_text = command_to_shell(command) if isinstance(command, list) else str(command)
    wrapped = osworld_user_command(command_text) if _command_as_user(runtime, as_user) else command_text
    response = await runtime.run_command(command=wrapped, timeout=timeout)
    return {
        'output': response_stdout(response),
        'error': response_stderr(response),
        'returncode': response_returncode(response),
    }


async def run_bash(runtime: Any, script: str, *, timeout: int = 120, working_dir: str | None = None) -> dict[str, Any]:
    if hasattr(runtime, 'run_bash_script'):
        result = await runtime.run_bash_script(script, timeout=timeout, working_dir=working_dir)
        return result if isinstance(result, dict) else {'output': str(result), 'error': '', 'returncode': 0}

    if working_dir:
        script = f'cd {shlex.quote(working_dir)}\n{script}'
    return await execute_command(runtime, script, shell=True, timeout=timeout)


async def run_python(runtime: Any, code: str, *, timeout: int = 120) -> dict[str, Any]:
    if hasattr(runtime, 'run_python'):
        result = await runtime.run_python(code)
        return result if isinstance(result, dict) else {'output': str(result), 'error': '', 'returncode': 0}
    return await execute_command(runtime, ['python3', '-c', code], timeout=timeout)


async def get_file_bytes(runtime: Any, remote_path: str) -> bytes | None:
    """Fetch bytes from the VM, returning None when the file is absent."""

    if hasattr(runtime, 'get_file'):
        try:
            content = await runtime.get_file(remote_path)
        except Exception as exc:
            response = getattr(exc, 'response', None)
            status = getattr(exc, 'status', None) or getattr(exc, 'status_code', None)
            status = status or getattr(response, 'status_code', None)
            if isinstance(exc, FileNotFoundError) or status == 404:
                raise FileNotFoundError(remote_path) from exc
            # Never collapse HTTP 5xx/timeouts/auth failures into a missing
            # output artifact (which would silently score zero).
            raise EvaluatorInfrastructureError(f'Sandbox file download failed ({type(exc).__name__})') from exc
        return content if content is not None else None

    if hasattr(runtime, 'download_file'):
        try:
            response = await runtime.download_file(files=[FileItem(path=remote_path, encoding='base64')])
            files = getattr(response, 'files', None) or []
            if files:
                content = getattr(files[0], 'content', None)
                if content is not None:
                    return base64.b64decode(content)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise EvaluatorInfrastructureError(f'Sandbox file download failed ({type(exc).__name__})') from exc

    result = await execute_command(runtime, f'base64 -w0 {shlex.quote(remote_path)}', shell=True, timeout=120)
    if response_returncode(result) not in (0, None):
        return None
    output = response_stdout(result).strip()
    return base64.b64decode(output) if output else b''


async def write_bytes(path: Path, content: bytes) -> None:
    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f'{path.name}.tmp')
        tmp_path.write_bytes(content)
        os.replace(tmp_path, path)

    await asyncio.to_thread(_write)


async def write_text(path: Path, content: str) -> None:
    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f'{path.name}.tmp')
        tmp_path.write_text(content, encoding='utf-8')
        os.replace(tmp_path, path)

    await asyncio.to_thread(_write)


def _rewrite_hf_url(url: str) -> str:
    endpoint = os.environ.get('HF_ENDPOINT', '').rstrip('/')
    if endpoint and 'huggingface.co' in url:
        return url.replace('https://huggingface.co', endpoint).replace('http://huggingface.co', endpoint)
    return url


async def download_url(url: str, path: Path) -> None:
    response = await http_client.get(_rewrite_hf_url(url))
    response.raise_for_status()
    await write_bytes(path, response.content)


async def open_file_from_vm(runtime: Any, remote_path: str) -> bytes | None:
    return await get_file_bytes(runtime, posixpath.normpath(remote_path))
