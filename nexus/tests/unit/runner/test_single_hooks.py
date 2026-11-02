import asyncio
import uuid

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    agent_registry,
    task_registry,
)
from nexus.core.context import ExecutionContext, require_context
from nexus.runner.single import RunHooks, RunInput, RunOutput, run


class HookTaskInput(TaskRunInput):
    message: str
    get_user_input: Callable[[Any], Awaitable[str | None]]


class HookTaskOutput(TaskRunOutput):
    result: str


class AgentHookTaskInput(TaskRunInput, WithAgentRunner):
    pass


class AgentHookTaskOutput(TaskRunOutput):
    result: str


class HookAgentInput(AgentRunInput):
    marker: str
    get_user_input: Callable[[Any], Awaitable[str | None]]


class HookAgentOutput(AgentRunOutput):
    result: str


@pytest.mark.asyncio
async def test_single_run_hooks_expose_context_and_inject_task_input() -> None:
    task_name = f'test_hook_task_{uuid.uuid4().hex}'
    seen: dict[str, Any] = {}

    @task_registry.register(task_name)
    async def hook_task(task_input: HookTaskInput) -> HookTaskOutput:
        seen['task_ctx'] = require_context()
        reply = await task_input.get_user_input('next')
        return HookTaskOutput(result=f'{task_input.message}:{reply}')

    async def get_user_input(_request: Any) -> str:
        return 'world'

    async def on_context_ready(ctx: ExecutionContext) -> None:
        seen['hook_ctx'] = ctx

    async def on_run_output(ctx: ExecutionContext, output: RunOutput) -> None:
        seen['output_ctx'] = ctx
        seen['output'] = output

    result = await run(
        RunInput(run_id='execution-1', task_name=task_name, task_config={'message': 'hello'}),
        hooks=RunHooks(
            task_input_overrides={'get_user_input': get_user_input},
            on_context_ready=on_context_ready,
            on_run_output=on_run_output,
        ),
    )

    assert result.success is True
    assert result.task_output.result == 'hello:world'
    assert seen['hook_ctx'] is seen['task_ctx']
    assert seen['hook_ctx'].run_id == 'execution-1'
    assert seen['output_ctx'] is seen['task_ctx']
    assert seen['output'] is result


@pytest.mark.asyncio
async def test_single_run_hooks_inject_agent_input() -> None:
    task_name = f'test_agent_hook_task_{uuid.uuid4().hex}'
    agent_name = f'test_agent_hook_agent_{uuid.uuid4().hex}'

    @agent_registry.register(agent_name)
    async def hook_agent(agent_input: HookAgentInput) -> HookAgentOutput:
        reply = await agent_input.get_user_input('agent input')
        return HookAgentOutput(result=f'{agent_input.marker}:{reply}')

    @task_registry.register(task_name)
    async def hook_agent_task(task_input: AgentHookTaskInput) -> AgentHookTaskOutput:
        output = await task_input.run_agent()
        return AgentHookTaskOutput(result=output.result)

    async def get_user_input(_request: Any) -> str:
        return 'world'

    result = await run(
        RunInput(
            task_name=task_name,
            agent_name=agent_name,
            agent_config={'marker': 'hello'},
        ),
        hooks=RunHooks(agent_input_overrides={'get_user_input': get_user_input}),
    )

    assert result.success is True
    assert result.task_output.result == 'hello:world'
    assert result.agent_output.result == 'hello:world'


@pytest.mark.asyncio
async def test_context_closing_hook_runs_inside_context_after_output() -> None:
    task_name = f'test_context_closing_{uuid.uuid4().hex}'
    calls: list[str] = []

    @task_registry.register(task_name)
    async def hook_task(_task_input: TaskRunInput) -> HookTaskOutput:
        calls.append('task')
        return HookTaskOutput(result='done')

    async def on_run_output(_ctx: ExecutionContext, _output: RunOutput) -> None:
        calls.append('output')

    async def on_context_closing(ctx: ExecutionContext) -> None:
        assert require_context() is ctx
        calls.append('closing')

    result = await run(
        RunInput(task_name=task_name),
        hooks=RunHooks(on_run_output=on_run_output, on_context_closing=on_context_closing),
    )

    assert result.success is True
    assert calls == ['task', 'output', 'closing']


@pytest.mark.asyncio
async def test_context_closing_hook_runs_when_task_is_cancelled() -> None:
    task_name = f'test_context_closing_cancel_{uuid.uuid4().hex}'
    started = asyncio.Event()
    closing = asyncio.Event()

    @task_registry.register(task_name)
    async def hook_task(_task_input: TaskRunInput) -> HookTaskOutput:
        started.set()
        await asyncio.Event().wait()
        return HookTaskOutput(result='unreachable')

    async def on_context_closing(_ctx: ExecutionContext) -> None:
        closing.set()

    task = asyncio.create_task(
        run(RunInput(task_name=task_name), hooks=RunHooks(on_context_closing=on_context_closing))
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert closing.is_set()


@pytest.mark.asyncio
async def test_run_output_hook_failure_is_recorded_without_raising() -> None:
    task_name = f'test_output_hook_failure_{uuid.uuid4().hex}'

    @task_registry.register(task_name)
    async def hook_task(_task_input: TaskRunInput) -> HookTaskOutput:
        return HookTaskOutput(result='done')

    async def on_run_output(_ctx: ExecutionContext, _output: RunOutput) -> None:
        raise RuntimeError('trace output failed')

    hooks = RunHooks(on_run_output=on_run_output)
    result = await run(RunInput(task_name=task_name), hooks=hooks)

    assert result.success is True
    assert len(hooks.hook_failures) == 1
    assert hooks.hook_failures[0].stage == 'run_output'
    assert str(hooks.hook_failures[0].error) == 'trace output failed'


@pytest.mark.asyncio
async def test_context_closing_hook_failure_is_recorded_without_raising() -> None:
    task_name = f'test_closing_hook_failure_{uuid.uuid4().hex}'

    @task_registry.register(task_name)
    async def hook_task(_task_input: TaskRunInput) -> HookTaskOutput:
        return HookTaskOutput(result='done')

    async def on_context_closing(_ctx: ExecutionContext) -> None:
        raise RuntimeError('final trace drain failed')

    hooks = RunHooks(on_context_closing=on_context_closing)
    result = await run(RunInput(task_name=task_name), hooks=hooks)

    assert result.success is True
    assert len(hooks.hook_failures) == 1
    assert hooks.hook_failures[0].stage == 'context_closing'
    assert str(hooks.hook_failures[0].error) == 'final trace drain failed'


@pytest.mark.asyncio
async def test_context_closing_hook_failure_does_not_mask_task_error() -> None:
    task_name = f'test_closing_hook_task_failure_{uuid.uuid4().hex}'

    @task_registry.register(task_name)
    async def hook_task(_task_input: TaskRunInput) -> HookTaskOutput:
        raise ValueError('business task failed')

    async def on_context_closing(_ctx: ExecutionContext) -> None:
        raise RuntimeError('final trace drain failed')

    hooks = RunHooks(on_context_closing=on_context_closing)
    result = await run(RunInput(task_name=task_name), hooks=hooks)

    assert result.success is False
    assert result.error is not None
    assert result.error.type == 'ValueError'
    assert result.error.message == 'business task failed'
    assert len(hooks.hook_failures) == 1
    assert hooks.hook_failures[0].stage == 'context_closing'


@pytest.mark.asyncio
async def test_context_closing_hook_failure_does_not_mask_cancellation() -> None:
    task_name = f'test_closing_hook_cancel_failure_{uuid.uuid4().hex}'
    started = asyncio.Event()

    @task_registry.register(task_name)
    async def hook_task(_task_input: TaskRunInput) -> HookTaskOutput:
        started.set()
        await asyncio.Event().wait()
        return HookTaskOutput(result='unreachable')

    async def on_context_closing(_ctx: ExecutionContext) -> None:
        raise RuntimeError('final trace drain failed')

    hooks = RunHooks(on_context_closing=on_context_closing)
    task = asyncio.create_task(run(RunInput(task_name=task_name), hooks=hooks))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(hooks.hook_failures) == 1
    assert hooks.hook_failures[0].stage == 'context_closing'
