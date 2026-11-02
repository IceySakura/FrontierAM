import asyncio
import uuid

from typing import Any

import pytest

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.runner.single import ProcessorConfig, RunInput, run
from nexus.trace.processors import SpanProcessor, span_processor_registry


_CALLS: dict[str, list[str]] = {}


class ScopeTaskOutput(TaskRunOutput):
    result: str


@span_processor_registry.register('test_scope_sync_processor')
class SyncProcessor(SpanProcessor):
    """A processor with only the synchronous lifecycle, like every processor on main."""

    def __init__(self, *, key: str) -> None:
        self.key = key
        _CALLS[key] = []

    def finalize(self) -> None:
        _CALLS[self.key].append('finalize')

    def get_result(self) -> Any:
        _CALLS[self.key].append('get_result')
        return self.key


@span_processor_registry.register('test_scope_async_processor')
class AsyncTeardownProcessor(SyncProcessor):
    """A processor whose teardown needs the loop, like GatewayTrajectoryProcessor."""

    async def afinalize(self) -> None:
        _CALLS[self.key].append('afinalize')


def run_input(key: str, task_name: str, processor: str) -> RunInput:
    return RunInput(
        run_id=key,
        task_name=task_name,
        processors={'scoped': ProcessorConfig(name=processor, args={'key': key})},
    )


@pytest.fixture
def task_name() -> str:
    name = f'test_scope_task_{uuid.uuid4().hex}'

    @task_registry.register(name)
    async def scope_task(_task_input: TaskRunInput) -> ScopeTaskOutput:
        return ScopeTaskOutput(result='done')

    return name


class TestProcessorScope:
    async def test_processor_without_afinalize_is_untouched(self, task_name):
        """The scope must stay inert for the processors that do not opt in."""
        key = f'sync-{uuid.uuid4().hex}'
        result = await run(run_input(key, task_name, 'test_scope_sync_processor'))

        assert result.success
        assert result.processor_results['scoped'] == key
        assert _CALLS[key] == ['finalize', 'get_result']

    async def test_async_teardown_runs_between_finalize_and_get_result(self, task_name):
        """Ordering is the contract: finalize settles sync facts, afinalize completes the result."""
        key = f'async-{uuid.uuid4().hex}'
        result = await run(run_input(key, task_name, 'test_scope_async_processor'))

        assert result.success
        assert _CALLS[key] == ['finalize', 'afinalize', 'get_result']

    async def test_async_teardown_runs_when_the_run_is_cancelled(self):
        """A cancelled run still tears down, so nothing it published is stranded."""
        key = f'cancelled-{uuid.uuid4().hex}'
        name = f'test_scope_blocking_task_{uuid.uuid4().hex}'
        started = asyncio.Event()

        @task_registry.register(name)
        async def blocking_task(_task_input: TaskRunInput) -> ScopeTaskOutput:
            started.set()
            await asyncio.Event().wait()
            return ScopeTaskOutput(result='unreachable')

        task = asyncio.create_task(run(run_input(key, name, 'test_scope_async_processor')))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert _CALLS[key] == ['finalize', 'afinalize']
