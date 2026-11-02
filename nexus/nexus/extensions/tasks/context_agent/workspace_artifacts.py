"""Persist a Context Agent runtime workspace as a Host-side artifact.

The general Context Agent runs its file tools inside a Nexus Runtime.  The
Runtime workspace is therefore ephemeral for remote providers such as
Gongfeng.  This module snapshots only the contractually-owned ``workspace/``
subdirectory before that Runtime is released and stores it under a stable
Host-side path keyed by ``run_id``.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import shutil
import tarfile
import uuid

from pathlib import Path, PurePosixPath
from typing import Any

from nexus.core import ExecutionContext
from nexus.runtime.executors import FileItem


logger = logging.getLogger(__name__)

DEFAULT_MAX_FILES = 10_000
DEFAULT_MAX_BYTES = 1024 * 1024 * 1024


def _safe_run_id(run_id: str) -> str:
    """Return a filesystem-safe directory component for a Nexus run id."""
    safe = ''.join(character if character.isalnum() or character in '._-' else '_' for character in run_id)
    return safe.strip('._') or 'run'


def _archive_workspace(
    workspace_dir: str,
    archive_path: str,
    max_files: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Runtime-side helper that archives regular files below ``workspace_dir``."""
    import os
    import tarfile

    workspace_dir = os.path.abspath(workspace_dir)
    archive_path = os.path.abspath(archive_path)
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)

    files: list[dict[str, Any]] = []
    total_bytes = 0
    with tarfile.open(archive_path, mode='w:gz') as archive:
        if os.path.isdir(workspace_dir):
            for root, dirs, names in os.walk(workspace_dir, followlinks=False):
                dirs[:] = sorted(
                    name
                    for name in dirs
                    if not os.path.islink(os.path.join(root, name))
                    and name not in {'__pycache__', '.git', '.ipynb_checkpoints'}
                )
                for name in sorted(names):
                    source = os.path.join(root, name)
                    if os.path.islink(source) or not os.path.isfile(source):
                        continue
                    rel = os.path.relpath(source, workspace_dir).replace(os.sep, '/')
                    if rel == '.keep':
                        continue
                    size = os.path.getsize(source)
                    if len(files) >= max_files:
                        raise ValueError(f'workspace contains more than {max_files} regular files')
                    if total_bytes + size > max_bytes:
                        raise ValueError(f'workspace exceeds the {max_bytes}-byte artifact limit')
                    archive.add(source, arcname=rel, recursive=False)
                    files.append({'path': rel, 'size_bytes': size})
                    total_bytes += size

    return {
        'archive_path': archive_path,
        'files': files,
        'file_count': len(files),
        'total_bytes': total_bytes,
        'archive_bytes': os.path.getsize(archive_path),
    }


def _remove_runtime_archive(path: str) -> None:
    """Runtime-side best-effort cleanup for the temporary archive."""
    import os

    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


def _validate_member(member: tarfile.TarInfo) -> PurePosixPath:
    """Validate one archive member before extracting it on the Host."""
    path = PurePosixPath(member.name)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError(f'unsafe workspace archive member: {member.name!r}')
    if member.issym() or member.islnk() or member.isdev():
        raise ValueError(f'unsupported workspace archive member: {member.name!r}')
    if not member.isfile() and not member.isdir():
        raise ValueError(f'unsupported workspace archive member type: {member.name!r}')
    return path


def _extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract a trusted-format workspace archive without path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode='r:gz') as archive:
        for member in archive.getmembers():
            relative = _validate_member(member)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f'workspace archive member has no content: {member.name!r}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open('wb') as output:
                shutil.copyfileobj(source, output)


def _context_names(context_files: list[Any], staged_files: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in context_files:
        if isinstance(item, dict):
            raw = item.get('name') or item.get('uri') or item.get('url') or item.get('path') or ''
        else:
            raw = str(item)
        basename = str(raw).split('?', 1)[0].rstrip('/').rsplit('/', 1)[-1].strip()
        if basename:
            names.add(basename.lower())
    for item in staged_files:
        if isinstance(item, dict) and item.get('name'):
            names.add(str(item['name']).lower())
    return names


def _manifest_files(workspace: Path, context_names: set[str]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(workspace.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(workspace).as_posix()
        files.append({
            'path': relative,
            'size_bytes': path.stat().st_size,
            'kind': 'context' if path.name.lower() in context_names else 'agent',
        })
    return files


async def persist_runtime_workspace(
    ctx: ExecutionContext,
    *,
    working_dir: str,
    artifact_root: str | Path,
    context_files: list[Any],
    staged_files: list[dict[str, Any]],
    task_id: str | None = None,
    runtime_name: str = 'main',
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Download ``<working_dir>/workspace`` and return its persistent mapping."""
    run_id = str(ctx.run_id or uuid.uuid4().hex)
    safe_run_id = _safe_run_id(run_id)
    configured_root = Path(artifact_root).expanduser()
    resolved_root = configured_root.resolve()
    final_dir = resolved_root / safe_run_id
    final_workspace = final_dir / 'workspace'
    temporary_dir = resolved_root / f'.{safe_run_id}.tmp-{uuid.uuid4().hex}'
    local_archive = temporary_dir / 'workspace.tar.gz'
    runtime_workspace = f'{working_dir.rstrip("/")}/workspace'
    runtime_archive = f'{working_dir.rstrip("/")}/.nexus_workspace_artifact.tar.gz'

    result: dict[str, Any] = {
        'run_id': run_id,
        'task_id': task_id,
        'artifact_root': str(resolved_root),
        'relative_path': f'{safe_run_id}/workspace',
        'path': str(final_workspace),
        'available': False,
        'files': [],
        'error': None,
    }

    runtime = await ctx.get_runtime(runtime_name)
    try:
        archive_info = await runtime.invoke(
            _archive_workspace,
            runtime_workspace,
            runtime_archive,
            int(max_files),
            int(max_bytes),
        )
        if not isinstance(archive_info, dict):
            raise RuntimeError('workspace archive operation returned invalid metadata')
        response = await runtime.download_file([FileItem(path=runtime_archive, encoding='base64')])
        if response.errors:
            messages = '; '.join(error.message or error.path for error in response.errors)
            raise RuntimeError(f'workspace archive download failed: {messages}')
        if len(response.files) != 1 or response.files[0].content is None:
            raise RuntimeError('workspace archive download returned no content')

        temporary_dir.mkdir(parents=True, exist_ok=False)
        local_archive.write_bytes(base64.b64decode(response.files[0].content, validate=True))
        extracted_workspace = temporary_dir / 'workspace'
        _extract_archive(local_archive, extracted_workspace)
        local_archive.unlink()

        context_names = _context_names(context_files, staged_files)
        files = _manifest_files(extracted_workspace, context_names)
        manifest = {
            'run_id': run_id,
            'task_id': task_id,
            'runtime_workspace': runtime_workspace,
            'host_workspace': str(final_workspace),
            'relative_path': f'{safe_run_id}/workspace',
            'file_count': len(files),
            'total_bytes': sum(int(file['size_bytes']) for file in files),
            'files': files,
        }
        (temporary_dir / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        resolved_root.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(temporary_dir, final_dir)

        result.update({
            'available': True,
            'files': files,
            'file_count': len(files),
            'total_bytes': manifest['total_bytes'],
            'archive_bytes': int(archive_info.get('archive_bytes') or 0),
        })
        logger.info(
            'context_agent_general: persisted %d workspace file(s) for run %s to %s',
            len(files),
            run_id,
            final_workspace,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception('context_agent_general: failed to persist workspace for run %s', run_id)
        result['error'] = f'{type(exc).__name__}: {exc}'
        return result
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        try:
            await runtime.invoke(_remove_runtime_archive, runtime_archive)
        except Exception as exc:  # noqa: BLE001
            logger.debug('context_agent_general: failed to remove runtime artifact archive: %s', exc)


__all__ = [
    'DEFAULT_MAX_BYTES',
    'DEFAULT_MAX_FILES',
    'persist_runtime_workspace',
]
