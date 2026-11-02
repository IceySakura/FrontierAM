import asyncio

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.os_world_v2.adapter.preparation import prepare_task048, run_preparation


class LocalRuntime:
    async def run_command(self, command, **kwargs):
        process = await asyncio.create_subprocess_exec(
            'bash',
            '-c',
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await process.communicate()
        return SimpleNamespace(return_code=process.returncode, stdout=out.decode(), stderr=err.decode())

    async def setup_upload_bytes(self, path, data, **kwargs):
        Path(path).write_bytes(data)
        return SimpleNamespace(status_code=200)


@pytest.mark.asyncio
async def test_preparation_returns_real_result_and_does_not_repeat_command(tmp_path):
    marker = tmp_path / 'attempts'
    result = await run_preparation(
        LocalRuntime(),
        command=f'echo once >> {marker}; read value; echo "answer=$value"; exit 7',
        stdin='fixture\n',
        timeout=10,
    )
    assert result.return_code == 7
    assert result.stdout == 'answer=fixture\n'
    assert marker.read_text() == 'once\n'


@pytest.mark.asyncio
async def test_preparation_guest_deadline_reports_124():
    result = await run_preparation(LocalRuntime(), command='sleep 30', timeout=0.05)
    assert result.return_code == 124


@pytest.mark.asyncio
async def test_task048_waits_for_boot_and_expands_only_small_shm(monkeypatch):
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=3, stdout='inactive\n'),
                SimpleNamespace(return_code=0, stdout='active\n'),
                SimpleNamespace(return_code=0, stdout=str(512 * 1024**2)),
                SimpleNamespace(return_code=0, stdout=''),
                SimpleNamespace(return_code=0, stdout=''),
            ]
        )
    )
    monkeypatch.setattr('nexus.extensions.tasks.os_world_v2.adapter.preparation.asyncio.sleep', AsyncMock())
    await prepare_task048(runtime, 'fixture-password')
    assert runtime.run_command.call_args_list[3].kwargs['stdin'] == 'fixture-password\n'
    assert 'size=1G' in runtime.run_command.call_args_list[3].kwargs['command']


@pytest.mark.asyncio
async def test_task048_does_not_shrink_larger_shm():
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout='active\n'),
                SimpleNamespace(return_code=0, stdout=str(2 * 1024**3)),
            ]
        )
    )
    await prepare_task048(runtime, 'fixture')
    assert runtime.run_command.await_count == 2
