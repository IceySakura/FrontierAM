"""Persist minimal, auditable EnterpriseBench run evidence."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import shutil
import tarfile
import uuid

from pathlib import Path, PurePosixPath
from typing import Any

from nexus.runtime.executors import FileItem


DEFAULT_ARTIFACT_ROOT = 'workspace/enterprisebench_native_artifacts'
DEFAULT_MAX_FILES = 10_000
DEFAULT_MAX_BYTES = 1024 * 1024 * 1024
_IGNORED_TOP_LEVEL = {'tool_calls', '.git', '__pycache__', '.ipynb_checkpoints'}


def _safe_component(value: str) -> str:
    safe = ''.join(character if character.isalnum() or character in '._-' else '_' for character in value)
    return safe.strip('._') or 'run'


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_files(root: str, *, max_files: int) -> dict[str, dict[str, Any]]:
    if not os.path.isdir(root):
        return {}
    files: dict[str, dict[str, Any]] = {}
    for current, dirs, names in os.walk(root, followlinks=False):
        relative_dir = os.path.relpath(current, root)
        if relative_dir == '.':
            dirs[:] = sorted(
                name
                for name in dirs
                if name not in _IGNORED_TOP_LEVEL and not os.path.islink(os.path.join(current, name))
            )
        else:
            dirs[:] = sorted(
                name
                for name in dirs
                if name not in {'__pycache__', '.git', '.ipynb_checkpoints'}
                and not os.path.islink(os.path.join(current, name))
            )
        for name in sorted(names):
            source = os.path.join(current, name)
            if os.path.islink(source) or not os.path.isfile(source):
                continue
            relative = os.path.relpath(source, root).replace(os.sep, '/')
            if relative == '.nexus_enterprisebench_evidence.tar.gz':
                continue
            if len(files) >= max_files:
                raise ValueError(f'workspace contains more than {max_files} regular files')
            files[relative] = {
                'size_bytes': os.path.getsize(source),
                'sha256': _file_digest(source),
            }
    return files


def _build_workspace_evidence(
    final_root: str,
    initial_root: str,
    archive_path: str,
    max_files: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Runtime helper: compare final workspace to its pristine initial copy."""
    final_files = _scan_files(final_root, max_files=max_files)
    initial_files = _scan_files(initial_root, max_files=max_files)
    manifest: list[dict[str, Any]] = []
    changed_paths: list[str] = []

    for relative in sorted(set(final_files) | set(initial_files)):
        final = final_files.get(relative)
        initial = initial_files.get(relative)
        if final is None:
            manifest.append({'path': relative, 'status': 'deleted', 'initial': initial, 'final': None})
        elif initial is None:
            manifest.append({'path': relative, 'status': 'created', 'initial': None, 'final': final})
            changed_paths.append(relative)
        elif final['sha256'] != initial['sha256']:
            manifest.append({'path': relative, 'status': 'modified', 'initial': initial, 'final': final})
            changed_paths.append(relative)
        else:
            manifest.append({'path': relative, 'status': 'unchanged', 'initial': initial, 'final': final})

    total_bytes = sum(int(final_files[path]['size_bytes']) for path in changed_paths)
    if total_bytes > max_bytes:
        raise ValueError(f'changed workspace files exceed the {max_bytes}-byte artifact limit')

    archive_bytes = 0
    if changed_paths:
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        with tarfile.open(archive_path, mode='w:gz') as archive:
            for relative in changed_paths:
                archive.add(os.path.join(final_root, relative), arcname=relative, recursive=False)
        archive_bytes = os.path.getsize(archive_path)

    counts = {
        status: sum(item['status'] == status for item in manifest)
        for status in ('created', 'modified', 'unchanged', 'deleted')
    }
    return {
        'manifest': manifest,
        'counts': counts,
        'changed_paths': changed_paths,
        'changed_bytes': total_bytes,
        'archive_bytes': archive_bytes,
    }


def _remove_runtime_file(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


def _validate_member(member: tarfile.TarInfo) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError(f'unsafe evidence archive member: {member.name!r}')
    if member.issym() or member.islnk() or member.isdev():
        raise ValueError(f'unsupported evidence archive member: {member.name!r}')
    if not member.isfile() and not member.isdir():
        raise ValueError(f'unsupported evidence archive member type: {member.name!r}')
    return path


def _extract_archive(archive_path: Path, destination: Path) -> None:
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
                raise ValueError(f'evidence archive member has no content: {member.name!r}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open('wb') as output:
                shutil.copyfileobj(source, output)


def artifact_directory(artifact_root: str | Path, run_id: str) -> Path:
    return Path(artifact_root).expanduser().resolve() / _safe_component(run_id)


async def persist_workspace_evidence(
    runtime: Any,
    *,
    artifact_root: str | Path,
    run_id: str,
    task_id: str,
    final_root: str = '/workdir',
    initial_root: str = '/initial_workspace',
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Save a manifest plus only created/modified Agent workspace files."""
    final_dir = artifact_directory(artifact_root, run_id)
    temporary_dir = final_dir.parent / f'.{final_dir.name}.tmp-{uuid.uuid4().hex}'
    local_archive = temporary_dir / 'changed-files.tar.gz'
    runtime_archive = f'{final_root.rstrip("/")}/.nexus_enterprisebench_evidence.tar.gz'
    result: dict[str, Any] = {
        'run_id': run_id,
        'task_id': task_id,
        'artifact_root': str(Path(artifact_root).expanduser().resolve()),
        'path': str(final_dir),
        'available': False,
        'error': None,
    }

    try:
        evidence = await runtime.invoke(
            _build_workspace_evidence,
            final_root,
            initial_root,
            runtime_archive,
            int(max_files),
            int(max_bytes),
        )
        if not isinstance(evidence, dict):
            raise RuntimeError('workspace evidence operation returned invalid metadata')
        temporary_dir.mkdir(parents=True, exist_ok=False)
        changed_files = temporary_dir / 'changed_files'
        changed_files.mkdir()
        if evidence.get('changed_paths'):
            response = await runtime.download_file([FileItem(path=runtime_archive, encoding='base64')])
            if response.errors:
                messages = '; '.join(error.message or error.path for error in response.errors)
                raise RuntimeError(f'workspace evidence download failed: {messages}')
            if len(response.files) != 1 or response.files[0].content is None:
                raise RuntimeError('workspace evidence download returned no content')
            local_archive.write_bytes(base64.b64decode(response.files[0].content, validate=True))
            _extract_archive(local_archive, changed_files)
            local_archive.unlink()
        manifest = {
            'run_id': run_id,
            'task_id': task_id,
            'runtime_workspace': final_root,
            'initial_workspace': initial_root,
            **evidence,
        }
        manifest.pop('changed_paths', None)
        (temporary_dir / 'workspace_manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(temporary_dir, final_dir)
        result.update({
            'available': True,
            'relative_path': final_dir.name,
            'manifest_path': str(final_dir / 'workspace_manifest.json'),
            'changed_files_path': str(final_dir / 'changed_files'),
            'files': evidence.get('manifest') or [],
            'file_count': len(evidence.get('manifest') or []),
            'counts': evidence.get('counts') or {},
            'total_bytes': sum(
                int((item.get('final') or {}).get('size_bytes') or 0) for item in evidence.get('manifest') or []
            ),
            'changed_bytes': int(evidence.get('changed_bytes') or 0),
            'archive_bytes': int(evidence.get('archive_bytes') or 0),
        })
    except Exception as exc:  # noqa: BLE001 - persistence must not invalidate a graded rollout
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        with contextlib.suppress(Exception):
            await runtime.invoke(_remove_runtime_file, runtime_archive)
    return result


def persist_run_records(
    *,
    artifact_root: str | Path,
    run_id: str,
    final_answer: str,
    agent_trajectory: list[dict[str, Any]],
    judge_details: dict[str, Any],
    judge_trajectories: list[dict[str, Any]],
    external_services_diff: str | None,
    result_summary: dict[str, Any],
) -> dict[str, Any]:
    """Write non-workspace evidence beside the persisted changed files."""
    destination = artifact_directory(artifact_root, run_id)
    result: dict[str, Any] = {'path': str(destination), 'available': False, 'paths': {}, 'error': None}
    try:
        destination.mkdir(parents=True, exist_ok=True)
        paths = {
            'agent_final_answer': destination / 'agent_final_answer.txt',
            'agent_trajectory': destination / 'agent_trajectory.json',
            'judge_details': destination / 'judge_details.json',
            'judge_trajectories': destination / 'judge_trajectories.json',
            'judge_requests': destination / 'judge_requests.json',
            'result_summary': destination / 'result_summary.json',
            'external_services_diff': destination / 'judge_workspace' / 'external_services_diff.md',
        }
        paths['external_services_diff'].parent.mkdir(parents=True, exist_ok=True)
        paths['agent_final_answer'].write_text(final_answer, encoding='utf-8')
        paths['agent_trajectory'].write_text(
            json.dumps(agent_trajectory, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        paths['judge_details'].write_text(
            json.dumps(judge_details, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        paths['judge_trajectories'].write_text(
            json.dumps(judge_trajectories, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        paths['judge_requests'].write_text(
            json.dumps(
                [trajectory.get('request') or {} for trajectory in judge_trajectories],
                ensure_ascii=False,
                indent=2,
            )
            + '\n',
            encoding='utf-8',
        )
        paths['result_summary'].write_text(
            json.dumps(result_summary, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        if external_services_diff is not None:
            paths['external_services_diff'].write_text(external_services_diff, encoding='utf-8')
        result['available'] = True
        result['paths'] = {name: str(path) for name, path in paths.items() if path.exists()}
    except Exception as exc:  # noqa: BLE001 - persistence must not invalidate a graded rollout
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


__all__ = [
    'DEFAULT_ARTIFACT_ROOT',
    'DEFAULT_MAX_BYTES',
    'DEFAULT_MAX_FILES',
    'artifact_directory',
    'persist_run_records',
    'persist_workspace_evidence',
]
