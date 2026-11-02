import asyncio
import uuid

import pytest

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.gateway.observers import (
    TrajectoryObserver,
    get_current_trajectory_observation_hub,
    trajectory_observer_registry,
)
from nexus.gateway.trajectory import Trajectory
from nexus.runner.single import ModuleObjectConfig, RunHooks, RunInput, run


class LifecycleTaskOutput(TaskRunOutput):
    result: str


def _register_observer(events: list[str]) -> str:
    observer_name = f'test_lifecycle_observer_{uuid.uuid4().hex}'

    @trajectory_observer_registry.register(observer_name)
    class LifecycleObserver(TrajectoryObserver):
        def on_final(self, _trajectory: Trajectory) -> None:
            events.append('final')

        def close(self) -> None:
            events.append('close')

    return observer_name


@pytest.mark.asyncio
async def test_run_without_observers_does_not_bind_a_hub() -> None:
    task_name = f'test_no_observer_hub_{uuid.uuid4().hex}'

    @task_registry.register(task_name)
    async def task(_task_input: TaskRunInput) -> LifecycleTaskOutput:
        assert get_current_trajectory_observation_hub() is None
        return LifecycleTaskOutput(result='done')

    result = await run(RunInput(task_name=task_name))

    assert result.success is True


@pytest.mark.asyncio
async def test_run_output_hook_publishes_final_before_observer_closes() -> None:
    task_name = f'test_observer_output_{uuid.uuid4().hex}'
    trajectory_id = uuid.uuid4().hex
    events: list[str] = []
    observer_name = _register_observer(events)

    @task_registry.register(task_name)
    async def task(_task_input: TaskRunInput) -> LifecycleTaskOutput:
        return LifecycleTaskOutput(result='done')

    async def on_run_output(_ctx, _output) -> None:
        hub = get_current_trajectory_observation_hub()
        assert hub is not None
        hub.publish_final(Trajectory(session_id=trajectory_id))

    result = await run(
        RunInput(
            task_name=task_name,
            gateway_trajectory_observers={'test': ModuleObjectConfig(name=observer_name)},
        ),
        hooks=RunHooks(on_run_output=on_run_output),
    )

    assert result.success is True
    assert events == ['final', 'close']


@pytest.mark.asyncio
async def test_context_closing_hook_publishes_final_before_observer_closes_on_cancellation() -> None:
    task_name = f'test_observer_cancel_{uuid.uuid4().hex}'
    trajectory_id = uuid.uuid4().hex
    events: list[str] = []
    observer_name = _register_observer(events)
    started = asyncio.Event()

    @task_registry.register(task_name)
    async def task(_task_input: TaskRunInput) -> LifecycleTaskOutput:
        started.set()
        await asyncio.Event().wait()
        return LifecycleTaskOutput(result='unreachable')

    async def on_context_closing(_ctx) -> None:
        hub = get_current_trajectory_observation_hub()
        assert hub is not None
        hub.publish_final(Trajectory(session_id=trajectory_id))

    running = asyncio.create_task(
        run(
            RunInput(
                task_name=task_name,
                gateway_trajectory_observers={'test': ModuleObjectConfig(name=observer_name)},
            ),
            hooks=RunHooks(on_context_closing=on_context_closing),
        )
    )
    await started.wait()
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running
    assert events == ['final', 'close']
