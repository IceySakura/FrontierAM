"""Optional task bundle restoration under the actual working directory."""

from __future__ import annotations

import asyncio
import ctypes
import errno
import os
import shutil
import sys
import tempfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from nexus.dataset.item import TaskFilesSpec


class TaskFilesMaterialization(BaseModel):
    mode: Literal['none', 'cwd'] = 'none'
    target: str = 'host'
    timeout: float = Field(default=600, gt=0)

    @field_validator('target')
    @classmethod
    def validate_target(cls, value: str) -> str:
        if value != 'host' and (not value.startswith('runtime:') or not value.removeprefix('runtime:')):
            raise ValueError('task_files materialization target must be host or runtime:<name>')
        return value


@dataclass
class TaskFilesLease:
    path: Path
    device: int
    inode: int

    def release(self) -> None:
        try:
            current = self.path.lstat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) != (self.device, self.inode) or self.path.is_symlink():
            raise RuntimeError('Task-files directory ownership changed; refusing cleanup')
        shutil.rmtree(self.path)


def _rename_no_replace(source: Path, target: Path) -> None:
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
            code = ctypes.get_errno()
            if code == errno.EEXIST:
                raise FileExistsError(f'Task-files directory is already occupied: {target}')
            raise OSError(code, os.strerror(code), str(target))
    elif os.name == 'nt':
        os.rename(source, target)
    elif sys.platform == 'darwin':
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(os.fsencode(source), os.fsencode(target), 4):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(target))
    else:
        raise RuntimeError('Atomic task-files publication is unsupported on this platform')


class TaskFilesMaterializer:
    async def prepare(self, files: Any, spec: TaskFilesSpec, workspace: Path) -> TaskFilesLease:
        if spec.root_name is None:
            raise ValueError('files_to_cwd requires a published root_name')
        workspace = workspace.resolve(strict=True)
        target = workspace / spec.root_name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Task-files directory is already occupied: {target}')
        staging = Path(tempfile.mkdtemp(prefix='.nexus-task-files-', dir=workspace))
        try:
            # Empty bundles still produce their declared directory.
            if await files.list():
                await files.download('**/*', staging)
            _rename_no_replace(staging, target)
            identity = target.stat()
            return TaskFilesLease(target, identity.st_dev, identity.st_ino)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


_RUNTIME_LEASES: dict[str, TaskFilesLease] = {}


async def runtime_prepare_task_files(session: dict[str, Any], timeout: float, token: str) -> str:
    from nexus.dataset.task_files import TaskFilesSession
    from nexus.runtime.context import require_runtime_control

    parsed = TaskFilesSession.model_validate(session)
    runtime = require_runtime_control()
    runtime.task_files.set_session(parsed)
    lease = await asyncio.wait_for(
        TaskFilesMaterializer().prepare(runtime.task_files, parsed.task_files, Path.cwd()),
        timeout=timeout,
    )
    _RUNTIME_LEASES[token] = lease
    return token


async def runtime_release_task_files(token: str) -> None:
    if lease := _RUNTIME_LEASES.pop(token, None):
        lease.release()
