import asyncio
import uuid

from pathlib import Path

import pytest

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.dataset.item import LocalFileRef, TaskFilesSpec
from nexus.dataset.preparation import TaskFilesMaterializer
from nexus.dataset.task_files import task_files_from_spec
from nexus.runner.single import RunInput, RunnerConfig, run


class ReadInput(TaskRunInput):
    path: str


class ReadOutput(TaskRunOutput):
    content: str


async def test_concurrent_runs_read_relative_paths_and_cleanup(tmp_path, monkeypatch):
    workspace = tmp_path / 'pwd'
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    name = f'files_cwd_{uuid.uuid4().hex}'
    barrier = asyncio.Event()
    started = []

    @task_registry.register(name)
    async def read_task(value: ReadInput) -> ReadOutput:
        started.append(value.path)
        if len(started) == 2:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), timeout=5)
        assert Path.cwd() == workspace
        return ReadOutput(content=Path(value.path).read_text())

    configs = []
    for number in range(2):
        root = f'case-{number}'
        source = tmp_path / root
        source.mkdir()
        (source / 'abcd').write_text(str(number))
        configs.append(
            RunInput(
                task_name=name,
                task_config={'path': f'{root}/abcd'},
                task_files=TaskFilesSpec(type='directory', source=LocalFileRef(path=str(source)), root_name=root),
                runner=RunnerConfig(task_files_materialization={'mode': 'cwd'}),
            )
        )
    outputs = await asyncio.gather(*(run(config) for config in configs))
    assert all(output.success for output in outputs)
    assert [output.task_output.content for output in outputs] == ['0', '1']
    assert list(workspace.iterdir()) == []


async def test_cancellation_cleans_only_owned_directory(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'abcd').write_text('data')
    workspace = tmp_path / 'pwd'
    workspace.mkdir()
    keep = workspace / 'unrelated'
    keep.mkdir()
    monkeypatch.chdir(workspace)
    name = f'files_cancel_{uuid.uuid4().hex}'
    started = asyncio.Event()

    @task_registry.register(name)
    async def blocked(_value: TaskRunInput) -> ReadOutput:
        assert Path('case-01/abcd').read_text() == 'data'
        started.set()
        await asyncio.Event().wait()
        return ReadOutput(content='unreachable')

    config = RunInput(
        task_name=name,
        task_files=TaskFilesSpec(type='directory', path=str(source), root_name='case-01'),
        runner=RunnerConfig(task_files_materialization={'mode': 'cwd'}),
    )
    running = asyncio.create_task(run(config))
    await asyncio.wait_for(started.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert keep.is_dir()
    assert not (workspace / 'case-01').exists()


async def test_concurrent_same_root_has_one_owner(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'abcd').write_text('data')
    workspace = tmp_path / 'pwd'
    workspace.mkdir()
    files = task_files_from_spec(TaskFilesSpec(type='directory', path=str(source), root_name='case-01'))
    results = await asyncio.gather(
        *(TaskFilesMaterializer().prepare(files, files.spec, workspace) for _ in range(2)), return_exceptions=True
    )
    assert sum(isinstance(value, FileExistsError) for value in results) == 1
    assert (workspace / 'case-01' / 'abcd').read_text() == 'data'
    for value in results:
        if not isinstance(value, BaseException):
            value.release()
    assert list(workspace.iterdir()) == []


async def test_cached_runtime_retries_failed_preparation_and_serializes_callers(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    from nexus.core.context import ExecutionContext
    from nexus.runtime import Runtime

    attempts = []
    releases = []

    async def invoke(target, *args):
        if target.endswith('runtime_release_task_files'):
            releases.append(args[0])
            return None
        attempts.append(args[-1])
        await asyncio.sleep(0)
        if len(attempts) == 1:
            raise RuntimeError('controlled preparation failure')
        return args[-1]

    runtime = MagicMock(spec=Runtime)
    runtime.invoke = AsyncMock(side_effect=invoke)
    runtime.drain_trace = AsyncMock()
    ctx = ExecutionContext(
        task_files=task_files_from_spec(TaskFilesSpec(type='directory', path=str(tmp_path), root_name='case')),
        task_files_materialization={'mode': 'cwd', 'target': 'runtime:main'},
    )
    ctx._runtimes['main'] = runtime
    async with ctx:
        with pytest.raises(RuntimeError, match='controlled preparation failure'):
            await ctx.get_runtime()
        results = await asyncio.gather(ctx.get_runtime(), ctx.get_runtime())
        assert results == [runtime, runtime]
        assert len(attempts) == 2
        assert releases == attempts[:1]
    assert releases == attempts
