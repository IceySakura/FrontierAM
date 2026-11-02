import asyncio
import uuid

import pytest

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.runner.single import RunInput, RunnerConfig, run


@pytest.mark.asyncio
@pytest.mark.parametrize('runner_expires', [False, True])
async def test_shared_runner_keeps_legacy_timeout_contract(runner_expires: bool) -> None:
    name = f'timeout_origin_{uuid.uuid4().hex}'

    @task_registry.register(name)
    async def task(task_input: TaskRunInput) -> TaskRunOutput:
        if runner_expires:
            await asyncio.sleep(10)
        raise TimeoutError('DevCloud health probe deadline exceeded')

    budget = 0.01 if runner_expires else 4800
    result = await run(RunInput(task_name=name, runner=RunnerConfig(timeout=budget)))
    assert not result.success
    assert result.error is not None
    assert result.error.traceback is None
    assert result.error.message == f'Task execution timed out after {float(budget)} seconds'
