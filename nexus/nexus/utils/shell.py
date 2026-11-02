import asyncio
import logging
import os
import shlex
import shutil

from pathlib import Path

import psutil


logger = logging.getLogger(__name__)


def quote_posix(value: str) -> str:
    """Quote one string for POSIX shell usage."""
    return shlex.quote(value)


def get_default_pwd() -> str:
    """
    Get the default working directory.

    Returns the current working directory, or '/' if it fails.
    """
    try:
        return os.getcwd()
    except Exception:
        return '/'


_NIX_BIN = '/nix/runtime/nix-env/bin'
_NIX_LIB = '/nix/runtime/nix-env/lib'
# 面向 agent 暴露的工具白名单目录（见 deployment/runtime/flake.nix 的 postBuild）。
# 整个 nix bin 会从 PATH 剥掉，但这个目录会追加到 PATH 末尾，使 docker 等平台
# 提供的工具无需 PATH 前缀即可用，同时用户镜像里的同名工具仍优先。
_NIX_AGENT_BIN = '/nix/runtime/nix-env/agent-bin'


def _strip_path_entry(value: str, entry: str) -> str:
    """Remove *entry* from a platform PATH-like string."""
    entry = entry.rstrip('/\\')
    parts = [p for p in value.split(os.pathsep) if p.rstrip('/\\') != entry]
    return os.pathsep.join(parts)


def get_clean_sandbox_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` sanitised for sandbox tool execution.

    Strips nix runtime paths from ``PATH`` and ``LD_LIBRARY_PATH`` so that
    agent tools see the user container's native environment (e.g. its own
    ``python``, ``node``, etc.) rather than nix-provided binaries.

    The agent-tools whitelist dir (``_NIX_AGENT_BIN``) is appended to the END of
    ``PATH`` when present, so platform-provided tools like ``docker`` work without
    a PATH prefix while user-image binaries of the same name still take priority.
    """
    env = os.environ.copy()

    if 'PATH' in env:
        env['PATH'] = _strip_path_entry(env['PATH'], _NIX_BIN)
        if os.path.isdir(_NIX_AGENT_BIN):
            env['PATH'] = f'{env["PATH"]}{os.pathsep}{_NIX_AGENT_BIN}' if env['PATH'] else _NIX_AGENT_BIN

    if 'LD_LIBRARY_PATH' in env:
        env['LD_LIBRARY_PATH'] = _strip_path_entry(env['LD_LIBRARY_PATH'], _NIX_LIB)
        if not env['LD_LIBRARY_PATH']:
            del env['LD_LIBRARY_PATH']

    return env


# Backward-compatible alias
get_os_env_without_ld_library_path = get_clean_sandbox_env


def resolve_first_existing_path(
    paths: list[str],
    *,
    fallback_to_last: bool = False,
) -> str:
    """
    Return the canonical absolute path of the first existing path.

    For each path:
    - If it's an absolute path, check if the file/directory exists
    - If it's a relative path or command name, resolve via PATH environment variable

    Args:
        paths: List of paths to check
        fallback_to_last: If True, return the last path when none exist;
                          If False (default), raise FileNotFoundError

    Returns:
        Canonical absolute path of the first existing path

    Raises:
        FileNotFoundError: When no paths exist and fallback_to_last=False
        ValueError: When paths is empty
    """
    if not paths:
        raise ValueError('paths cannot be empty')

    for path in paths:
        # Check if it's an absolute path or relative path (contains directory separator)
        if os.path.isabs(path) or os.sep in path or (os.altsep and os.altsep in path):
            if os.path.exists(path):
                return str(Path(path).resolve())
        else:
            # Resolve command via PATH
            resolved = shutil.which(path)
            if resolved:
                return str(Path(resolved).resolve())

    if fallback_to_last:
        return paths[-1]

    raise FileNotFoundError(f'None of the paths exist: {paths}')


async def kill_all_child_processes(pid: int) -> None:
    """
    Kill all child processes of a given PID using psutil.

    Args:
        pid: The parent process ID
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # Kill all child processes (in reverse order to kill leaf processes first)
        for child in reversed(children):
            try:
                if child.is_running():
                    child.terminate()
                    logger.debug(f'Terminated child process {child.pid}')
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f'Failed to terminate child process {child.pid}: {e}')

        # Give processes time to terminate gracefully using asyncio
        _, alive = await asyncio.to_thread(psutil.wait_procs, children, timeout=3)

        # Force kill any remaining processes
        for child in alive:
            try:
                if child.is_running():
                    child.kill()
                    logger.debug(f'Force killed child process {child.pid}')
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f'Failed to force kill child process {child.pid}: {e}')

    except psutil.NoSuchProcess:
        logger.warning(f'Process {pid} does not exist')
    except Exception as e:
        logger.warning(f'Failed to kill child processes of PID {pid}: {e}')
