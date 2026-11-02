"""CUA-Gym public dataset setup action support.

CUA-Gym 公开子集的核心动作 ``download`` / ``execute`` 用本模块的**原生实现**
（download = 上传 bundle 内文件；execute = 在 DISPLAY=:0 的会话用户上下文跑命令），
保持已验证的语义，不引入 OSWorld 的屏幕占位符替换 / screen_size 调用。

新版数据集额外出现的 ``sleep`` / ``launch`` / ``open`` / ``activate_window`` 等动作，
复用 OSWorld 的 dispatch（``os_world.execution.setup.run_setup_action``），只注入 cua_gym 的
``user_command_fn``（DISPLAY=:0、/home/user），避免重复实现这些原子动作。
"""

from __future__ import annotations

import base64
import posixpath
import shlex

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, cast

from nexus.extensions.tasks.os_world.execution import setup as osworld_setup
from nexus.extensions.tasks.os_world.execution.commands import command_to_shell
from nexus.runtime.executors.models.file import FileItem
from nexus.runtime.utils.gui import session_user_command


# download/execute 为 cua_gym 原生；其余动作复用 OSWorld dispatch。
SUPPORTED_SETUP_ACTIONS = {'download', 'execute', 'command', 'launch', 'open', 'activate_window', 'sleep'}


class UnsupportedSetupActionError(NotImplementedError):
    """Raised when a CUA-Gym setup action is outside the supported public subset."""


def cua_gym_user_command(
    command: str,
    *,
    user: str = 'user',
    home_dir: str = '/home/user',
    cwd: str | None = None,
    display: str = ':0',
) -> str:
    """Wrap a setup/eval command so it runs in the desktop session user context."""
    return session_user_command(
        command,
        user=user,
        home_dir=home_dir,
        cwd=cwd,
        display=display,
        missing_user_policy='fallback',
    )


def _bundle_file(bundle_dir: str | Path, source: str) -> Path:
    if '://' in source:
        raise UnsupportedSetupActionError(f'CUA-Gym download only supports bundle-relative files: {source}')
    raw = Path(source)
    if raw.is_absolute():
        raise UnsupportedSetupActionError(f'CUA-Gym download source must be bundle-relative: {source}')
    root = Path(bundle_dir).resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UnsupportedSetupActionError(f'CUA-Gym download escapes bundle directory: {source}') from exc
    if not resolved.is_file():
        raise FileNotFoundError(f'CUA-Gym setup file not found: {resolved}')
    return resolved


async def _upload_bundle_file(runtime: Any, *, bundle_dir: str | Path, source: str, target: str) -> dict[str, object]:
    local_path = _bundle_file(bundle_dir, source)
    encoded = base64.b64encode(local_path.read_bytes()).decode('ascii')
    response = await runtime.upload_file([FileItem(path=target, content=encoded, encoding='base64')], overwrite=True)
    results = list(getattr(response, 'results', []) or [])
    success = bool(results) and all(bool(getattr(item, 'success', False)) for item in results)
    chmod_command = f'chown user:user {shlex.quote(target)} 2>/dev/null || true; chmod a+r {shlex.quote(target)}'
    chmod_response = await runtime.run_command(command=chmod_command, timeout=60)
    return {
        'source': str(local_path),
        'path': target,
        'success': success and getattr(chmod_response, 'return_code', None) == 0,
        'upload_results': [_dump_upload_result(item) for item in results],
    }


def _dump_upload_result(item: Any) -> dict[str, object]:
    if hasattr(item, 'model_dump'):
        return item.model_dump()
    return {
        'path': getattr(item, 'path', ''),
        'success': bool(getattr(item, 'success', False)),
        'message': getattr(item, 'message', None),
    }


async def _run_download(runtime: Any, parameters: dict[str, Any], *, bundle_dir: str | Path) -> dict[str, object]:
    """CUA-Gym download：上传 bundle 内的文件（不是 URL 下载）。"""
    files = parameters.get('files') or []
    if not isinstance(files, list):
        raise ValueError('CUA-Gym download parameters.files must be a list')
    results: list[dict[str, object]] = []
    for file_spec in files:
        if not isinstance(file_spec, dict):
            raise ValueError('CUA-Gym download file spec must be an object')
        source = str(file_spec.get('url') or file_spec.get('source') or '')
        target = str(file_spec.get('path') or file_spec.get('dest') or '')
        if not source or not target:
            raise ValueError('CUA-Gym download file spec requires url and path')
        parent = posixpath.dirname(target)
        if parent:
            mkdir_response = await runtime.run_command(command=f'mkdir -p {shlex.quote(parent)}', timeout=60)
            if getattr(mkdir_response, 'return_code', None) != 0:
                results.append({'source': source, 'path': target, 'success': False, 'error': 'mkdir failed'})
                continue
        results.append(await _upload_bundle_file(runtime, bundle_dir=bundle_dir, source=source, target=target))
    return {'files': results}


def _download_file_success(item: object) -> bool:
    if not isinstance(item, Mapping):
        return False
    return bool(cast(Mapping[str, object], item).get('success'))


async def _run_execute(
    runtime: Any,
    parameters: dict[str, Any],
    *,
    timeout: int,
    display: str,
) -> dict[str, object]:
    """CUA-Gym execute：原生语义——在会话用户上下文（DISPLAY=:0、/home/user）跑命令，
    不做 OSWorld 的屏幕占位符替换、不额外查 screen_size。"""
    raw_command = parameters.get('command')
    if raw_command is None:
        raise ValueError('CUA-Gym execute action requires parameters.command')
    command = command_to_shell(raw_command)
    response = await runtime.run_command(
        command=cua_gym_user_command(command, cwd='/home/user', display=display),
        timeout=timeout,
    )
    return {
        'command': command,
        'success': getattr(response, 'return_code', None) == 0,
        'return_code': getattr(response, 'return_code', None),
        'stdout': getattr(response, 'stdout', ''),
        'stderr': getattr(response, 'stderr', ''),
    }


def _unsupported_result(action_type: str) -> dict[str, object]:
    return {
        'type': action_type,
        'success': False,
        'supported': False,
        'error': f'Unsupported CUA-Gym setup action: {action_type}',
    }


async def run_setup_action(
    runtime: Any,
    action: dict[str, Any],
    *,
    bundle_dir: str | Path,
    unsupported_policy: str = 'fail',
    execute_timeout: int = 600,
    display: str = ':0',
) -> dict[str, object]:
    """Run one CUA-Gym setup action.

    download/execute 用 cua_gym 原生实现；sleep/launch/open/activate_window 复用 OSWorld dispatch。
    """
    action_type = str(action.get('type', ''))
    parameters = action.get('parameters') or {}
    if not isinstance(parameters, dict):
        raise ValueError(f'CUA-Gym setup parameters must be an object for {action_type!r}')
    if action_type not in SUPPORTED_SETUP_ACTIONS:
        if unsupported_policy == 'skip':
            return _unsupported_result(action_type)
        raise UnsupportedSetupActionError(f'Unsupported CUA-Gym setup action: {action_type}')

    # cua_gym 原生核心动作
    if action_type == 'download':
        result = await _run_download(runtime, parameters, bundle_dir=bundle_dir)
        files = result['files']
        success = isinstance(files, list) and all(_download_file_success(item) for item in files)
        return {'type': action_type, 'success': success, **result}
    if action_type in {'execute', 'command'}:
        return {
            'type': action_type,
            **await _run_execute(runtime, parameters, timeout=execute_timeout, display=display),
        }

    # 新增原子动作：复用 OSWorld dispatch，仅注入 cua_gym 的会话用户上下文（DISPLAY=:0）
    return await osworld_setup.run_setup_action(
        runtime,
        action,
        user_command_fn=partial(cua_gym_user_command, cwd='/home/user', display=display),
        execute_timeout=execute_timeout,
    )


async def run_setup_actions(
    runtime: Any,
    actions: list[dict[str, Any]] | None,
    *,
    bundle_dir: str | Path,
    unsupported_policy: str = 'fail',
    execute_timeout: int = 600,
    display: str = ':0',
    stop_on_failure: bool = True,
) -> list[dict[str, object]]:
    """Run CUA-Gym setup actions sequentially."""
    results = []
    for action in actions or []:
        result = await run_setup_action(
            runtime,
            action,
            bundle_dir=bundle_dir,
            unsupported_policy=unsupported_policy,
            execute_timeout=execute_timeout,
            display=display,
        )
        results.append(result)
        if stop_on_failure and result.get('success') is False:
            break
    return results


__all__ = [
    'SUPPORTED_SETUP_ACTIONS',
    'UnsupportedSetupActionError',
    'cua_gym_user_command',
    'run_setup_action',
    'run_setup_actions',
]
