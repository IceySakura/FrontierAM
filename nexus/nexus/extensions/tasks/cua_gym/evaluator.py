"""CUA-Gym public dataset reward runner."""

from __future__ import annotations

import base64
import re
import shlex

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.cua_gym.setup import cua_gym_user_command
from nexus.runtime.executors.models.file import FileItem


REWARD_PATTERN = re.compile(r'REWARD\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
DEFAULT_REMOTE_EVAL_ROOT = '/tmp/cua-gym'
PACKAGED_REWARD_JUDGE_PATH = Path(__file__).with_name('reward_judge.py')


def parse_reward_score(stdout: str) -> float:
    """Parse the last official ``REWARD: <float>`` line from reward stdout."""
    matches = REWARD_PATTERN.findall(stdout)
    if not matches:
        raise ValueError('CUA-Gym reward output did not contain a REWARD line')
    return float(matches[-1])


def _bundle_file(bundle_dir: str | Path, source: str) -> Path:
    raw = Path(source)
    if raw.is_absolute():
        raise ValueError(f'CUA-Gym evaluator path must be bundle-relative: {source}')
    root = Path(bundle_dir).resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'CUA-Gym evaluator path escapes bundle directory: {source}') from exc
    if not resolved.is_file():
        raise FileNotFoundError(f'CUA-Gym evaluator file not found: {resolved}')
    return resolved


def _bundle_files(bundle_dir: str | Path) -> list[Path]:
    root = Path(bundle_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'CUA-Gym bundle directory not found: {root}')
    return sorted(path for path in root.rglob('*') if path.is_file())


def _remote_bundle_path(*, bundle_dir: str | Path, local_path: Path, remote_dir: str) -> str:
    root = Path(bundle_dir).resolve()
    relative_path = local_path.resolve().relative_to(root).as_posix()
    return f'{remote_dir.rstrip("/")}/{relative_path}'


async def _upload_files(runtime: Any, files: list[FileItem]) -> None:
    response = await runtime.upload_file(
        files,
        overwrite=True,
    )
    results = list(getattr(response, 'results', []) or [])
    if not results or not all(bool(getattr(item, 'success', False)) for item in results):
        failed_paths = ', '.join(
            str(getattr(item, 'path', '')) for item in results if not bool(getattr(item, 'success', False))
        )
        raise RuntimeError(f'Failed to upload CUA-Gym evaluator files: {failed_paths or "unknown"}')


async def _upload_file(runtime: Any, *, local_path: Path, remote_path: str) -> None:
    encoded = base64.b64encode(local_path.read_bytes()).decode('ascii')
    await _upload_files(runtime, [FileItem(path=remote_path, content=encoded, encoding='base64')])


async def _upload_bundle(runtime: Any, *, bundle_dir: str | Path, remote_dir: str) -> None:
    files = [
        FileItem(
            path=_remote_bundle_path(bundle_dir=bundle_dir, local_path=local_path, remote_dir=remote_dir),
            content=base64.b64encode(local_path.read_bytes()).decode('ascii'),
            encoding='base64',
        )
        for local_path in _bundle_files(bundle_dir)
    ]
    if not files:
        raise FileNotFoundError(f'CUA-Gym bundle directory contains no files: {Path(bundle_dir).resolve()}')
    await _upload_files(runtime, files)


async def _maybe_upload_reward_judge(runtime: Any, reward_text: str) -> None:
    if 'reward_judge' not in reward_text:
        return
    await _upload_file(runtime, local_path=PACKAGED_REWARD_JUDGE_PATH, remote_path='/tmp/reward_judge.py')


async def evaluate_cua_gym(
    runtime: Any,
    evaluator: dict[str, Any] | None,
    *,
    bundle_dir: str | Path,
    task_id: str,
    remote_eval_root: str = DEFAULT_REMOTE_EVAL_ROOT,
    timeout: int = 600,
    display: str = ':0',
) -> dict[str, Any]:
    """Run the official CUA-Gym ``reward.py`` evaluator and parse its score."""
    if not evaluator:
        return {'score': 0.0, 'supported': False, 'error': 'No CUA-Gym evaluator configured'}
    if str(evaluator.get('type', 'python')) != 'python':
        return {'score': 0.0, 'supported': False, 'error': f'Unsupported CUA-Gym evaluator: {evaluator.get("type")}'}

    reward_path = _bundle_file(bundle_dir, str(evaluator.get('path') or 'reward.py'))
    remote_dir = f'{remote_eval_root.rstrip("/")}/{task_id}'
    remote_reward_path = _remote_bundle_path(bundle_dir=bundle_dir, local_path=reward_path, remote_dir=remote_dir)
    await _upload_bundle(runtime, bundle_dir=bundle_dir, remote_dir=remote_dir)
    await _maybe_upload_reward_judge(runtime, reward_path.read_text(encoding='utf-8'))

    command = f'python3 {shlex.quote(remote_reward_path)}'
    response = await runtime.run_command(
        command=cua_gym_user_command(command, cwd=remote_dir, display=display),
        timeout=timeout,
    )
    stdout = str(getattr(response, 'stdout', '') or '')
    stderr = str(getattr(response, 'stderr', '') or '')
    return_code = getattr(response, 'return_code', None)
    if return_code != 0:
        return {
            'score': 0.0,
            'supported': True,
            'stdout': stdout,
            'stderr': stderr,
            'return_code': return_code,
            'error': f'CUA-Gym reward script exited with non-zero return code: {return_code}',
        }
    try:
        score = parse_reward_score(stdout)
        return {
            'score': score,
            'supported': True,
            'stdout': stdout,
            'stderr': stderr,
            'return_code': return_code,
        }
    except ValueError as exc:
        return {
            'score': 0.0,
            'supported': True,
            'stdout': stdout,
            'stderr': stderr,
            'return_code': return_code,
            'error': str(exc),
        }


__all__ = ['DEFAULT_REMOTE_EVAL_ROOT', 'PACKAGED_REWARD_JUDGE_PATH', 'evaluate_cua_gym', 'parse_reward_score']
