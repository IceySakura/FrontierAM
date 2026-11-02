from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.extensions.tasks.aidata.engine.sandbox import run_bootstrap, upload_dir
from nexus.runtime.executors.command import CommandRunStatus


class _FakeRuntime:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    async def run_command(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        return self.result


@pytest.mark.asyncio
async def test_upload_dir_rejects_missing_source(tmp_path: Path) -> None:
    runtime = _FakeRuntime(SimpleNamespace())

    with pytest.raises(FileNotFoundError, match='upload source directory does not exist'):
        await upload_dir(runtime, tmp_path / 'missing', '/workspace/.aidata/confine')


@pytest.mark.asyncio
async def test_upload_dir_rejects_empty_source(tmp_path: Path) -> None:
    runtime = _FakeRuntime(SimpleNamespace())
    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()

    with pytest.raises(ValueError, match='upload source directory contains no files'):
        await upload_dir(runtime, empty_dir, '/workspace/.aidata/confine')


@pytest.mark.asyncio
async def test_run_bootstrap_fails_on_return_code_without_exit_code() -> None:
    runtime = _FakeRuntime(
        SimpleNamespace(
            status=CommandRunStatus.Finished,
            return_code=17,
            stdout='partial setup',
            stderr='missing seed',
        )
    )

    with pytest.raises(RuntimeError, match='return_code=17'):
        await run_bootstrap(runtime, 'assets/start.sh', '/workspace/.aidata/case')


@pytest.mark.asyncio
async def test_run_bootstrap_fails_on_non_finished_status_without_return_code() -> None:
    runtime = _FakeRuntime(
        SimpleNamespace(
            status=CommandRunStatus.TimeLimitExceeded,
            return_code=None,
            stdout='',
            stderr='timeout',
        )
    )

    with pytest.raises(RuntimeError, match='TimeLimitExceeded'):
        await run_bootstrap(runtime, 'assets/start.sh', '/workspace/.aidata/case')


@pytest.mark.asyncio
async def test_run_bootstrap_still_accepts_legacy_exit_code_success() -> None:
    runtime = _FakeRuntime(SimpleNamespace(exit_code=0, stdout='ok', stderr=''))

    await run_bootstrap(runtime, 'assets/start.sh', '/workspace/.aidata/case', extra_env={'BASE_URL': 'http://x'})

    assert runtime.calls == [
        (
            'chmod +x /workspace/.aidata/case/assets/start.sh 2>/dev/null; '
            'cd /workspace/.aidata/case && bash /workspace/.aidata/case/assets/start.sh',
            {'timeout': 180, 'extra_env': {'BASE_URL': 'http://x'}},
        )
    ]
