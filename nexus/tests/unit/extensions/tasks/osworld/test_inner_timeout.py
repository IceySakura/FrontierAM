"""Nested OSWorld timeouts must not surface as the runner's task deadline."""

import asyncio
import json

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.os_world.outputs import (
    AgentExitStatus,
    OSWorldInnerTimeoutError,
    classify_exception,
    wrap_inner_timeout,
)
from nexus.utils.timeout import async_timeout


def test_wrap_inner_timeout_converts_only_nested_timeouts():
    wrapped = wrap_inner_timeout(TimeoutError(), stage='agent')
    assert isinstance(wrapped, OSWorldInnerTimeoutError)
    assert not isinstance(wrapped, TimeoutError)
    assert 'agent operation timed out before the task deadline' in str(wrapped)

    other = RuntimeError('boom')
    assert wrap_inner_timeout(other, stage='agent') is other
    cancelled = asyncio.CancelledError()
    assert wrap_inner_timeout(cancelled, stage='agent') is cancelled


def test_wrap_inner_timeout_keeps_expired_runner_deadline():
    exc = TimeoutError()
    assert wrap_inner_timeout(exc, stage='agent', timeout_handle=SimpleNamespace(expired=lambda: True)) is exc
    assert wrap_inner_timeout(exc, stage='agent', timeout_handle=SimpleNamespace(expired=True)) is exc


@pytest.mark.parametrize(
    ('stage', 'status'),
    [('agent', AgentExitStatus.RUNTIME_ERROR_BY_ENV), ('inference', AgentExitStatus.INFERENCE_ERROR)],
)
def test_inner_timeout_classification(stage, status):
    assert classify_exception(wrap_inner_timeout(TimeoutError(), stage=stage), stage=stage).exit_status == status


async def _task(inner_timeout: float, work: float, timeout_handle=None):
    try:
        await asyncio.wait_for(asyncio.sleep(work), timeout=inner_timeout)
    except BaseException as exc:
        primary = wrap_inner_timeout(exc, stage='agent', timeout_handle=timeout_handle)
        if primary is exc:
            raise
        raise primary from exc


@pytest.mark.asyncio
async def test_runner_deadline_pattern_distinguishes_inner_and_outer():
    # Mirrors nexus/runner/single.py: async_timeout(runner.timeout) around the task.
    with pytest.raises(OSWorldInnerTimeoutError):
        async with async_timeout(5) as handle:
            await _task(inner_timeout=0.01, work=1, timeout_handle=handle)

    with pytest.raises(TimeoutError):
        async with async_timeout(0.01) as handle:
            await _task(inner_timeout=5, work=1, timeout_handle=handle)


@pytest.mark.asyncio
async def test_run_os_world_reports_inner_timeout(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.data.artifacts import RunArtifacts
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    task_file = tmp_path / 'case.json'
    task_file.write_text(json.dumps({'id': 'timeout', 'instruction': 'fixture', 'config': [], 'evaluator': {}}))
    monkeypatch.setattr(RunArtifacts, 'publish', AsyncMock(side_effect=ConnectionError('injected upload failure')))
    ctx = ExecutionContext(task_name='os_world', agent_name='fixture')
    ctx._runtimes['main'] = SimpleNamespace(drain_trace=AsyncMock())

    async def agent(**kwargs):
        raise TimeoutError

    async with ctx:
        inp = OSWorldTaskInput(
            task_file=str(task_file),
            run_agent=agent,
            run_setup=False,
            include_initial_screenshot=False,
            skip_gui_start=True,
            setup_wait_time=0,
        )
        with pytest.raises(OSWorldInnerTimeoutError) as raised:
            await run_os_world(inp)
        assert isinstance(raised.value.__cause__, TimeoutError)
        assert 'agent operation timed out before the task deadline' in str(raised.value)
        partial = ctx.tool_state['osworld_partial_task_output']
        assert partial['status'] == 'agent_error'
        assert partial['agent_output']['exit_status'] == AgentExitStatus.RUNTIME_ERROR_BY_ENV
