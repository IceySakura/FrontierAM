from __future__ import annotations

import asyncio

from typing import Any

import pytest

from pydantic import BaseModel

from nexus.core import (
    ExecutionContext,
    Node,
    NodeResult,
    TaskRunInput,
    WithWorkflowOptions,
    Workflow,
    WorkflowLimitExceededError,
    WorkflowNodeOptions,
    WorkflowNodeTimeoutError,
    WorkflowOptions,
    WorkflowStateMismatchError,
    execute_workflow,
    require_context,
)
from nexus.trace import processor_context
from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import SpanProcessor


class ValueOutput(BaseModel):
    value: int


class ConfigurableTaskInput(TaskRunInput, WithWorkflowOptions):
    instance_id: str


def test_with_workflow_options_parses_task_owned_node_config() -> None:
    task_input = ConfigurableTaskInput.model_validate(
        {
            'instance_id': 'case-1',
            'workflow': {
                'max_steps': 20,
                'nodes': {'infer': {'timeout': 30, 'max_visits': 2}},
            },
        }
    )

    assert task_input.workflow.max_steps == 20
    assert task_input.workflow.nodes['infer'] == WorkflowNodeOptions(timeout=30, max_visits=2)


@pytest.mark.asyncio
async def test_execute_workflow_runs_graph_with_cycle_and_typed_history() -> None:
    loop_count = 0

    async def start(initial_value: int) -> NodeResult[ValueOutput]:
        with pytest.raises(LookupError, match='no committed output'):
            require_context().workflow.last_output()
        return NodeResult(output=ValueOutput(value=initial_value), next='loop')

    async def loop(_initial_value: int) -> NodeResult[ValueOutput]:
        nonlocal loop_count
        ctx = require_context()
        assert ctx.workflow.output('start', as_type=ValueOutput).value == 1
        loop_count += 1
        return NodeResult(
            output=ValueOutput(value=loop_count),
            next='loop' if loop_count < 2 else 'finish',
        )

    async def finish(_initial_value: int) -> NodeResult[ValueOutput]:
        ctx = require_context()
        assert [output.value for output in ctx.workflow.outputs('loop', as_type=ValueOutput)] == [1, 2]
        assert ctx.workflow.last_output(as_type=ValueOutput) == ValueOutput(value=2)
        return NodeResult(output=ValueOutput(value=3))

    workflow = Workflow(
        name='cycle',
        entry='start',
        nodes=[Node('start', start), Node('loop', loop), Node('finish', finish)],
    )

    ctx = ExecutionContext()
    async with ctx:
        state = await execute_workflow(workflow, 1)

    assert state.cursor is None
    assert [visit.node for visit in state.history] == ['start', 'loop', 'loop', 'finish']
    assert [visit.visit for visit in state.history] == [1, 1, 2, 1]
    assert state.output('finish') == {'value': 3}
    assert state.outputs('loop') == [{'value': 1}, {'value': 2}]
    assert state.output('finish', as_type=ValueOutput) == ValueOutput(value=3)
    assert state.last_output() == {'value': 3}
    assert state.last_output(as_type=ValueOutput) == ValueOutput(value=3)
    assert ctx.workflow_states['task/cycle'] == state


@pytest.mark.asyncio
async def test_execute_workflow_returns_last_commit_and_resumes_failed_node() -> None:
    should_fail = True
    calls: list[str] = []

    async def prepare(_workflow_input: None) -> NodeResult[dict[str, str]]:
        calls.append('prepare')
        return NodeResult(output={'status': 'ready'}, next='work')

    async def work(_workflow_input: None) -> NodeResult[dict[str, str]]:
        calls.append('work')
        if should_fail:
            raise RuntimeError('temporary failure')
        return NodeResult(output={'status': 'done'})

    workflow = Workflow(
        name='resume',
        entry='prepare',
        nodes=[Node('prepare', prepare), Node('work', work)],
    )

    first_ctx = ExecutionContext()
    async with first_ctx:
        with pytest.raises(RuntimeError, match='temporary failure'):
            await execute_workflow(workflow, None)

    failed_state = first_ctx.workflow_states['task/resume']
    assert failed_state.cursor == 'work'
    assert [visit.node for visit in failed_state.history] == ['prepare']

    should_fail = False
    second_ctx = ExecutionContext(workflow_states=first_ctx.workflow_states)
    async with second_ctx:
        completed_state = await execute_workflow(workflow, None)

    assert calls == ['prepare', 'work', 'work']
    assert completed_state.cursor is None
    assert [visit.node for visit in completed_state.history] == ['prepare', 'work']

    completed_ctx = ExecutionContext(workflow_states=second_ctx.workflow_states)
    async with completed_ctx:
        same_state = await execute_workflow(workflow, None)

    assert same_state == completed_state
    assert calls == ['prepare', 'work', 'work']


@pytest.mark.asyncio
async def test_execute_workflow_builds_stable_nested_scope_from_parent_visit() -> None:
    async def inner_node(_workflow_input: None) -> NodeResult[str]:
        assert require_context().workflow.workflow == 'judge'
        return NodeResult(output='inner')

    inner = Workflow(
        name='judge',
        namespace='agent',
        entry='judge',
        nodes=[Node('judge', inner_node)],
    )

    async def outer_node(_workflow_input: None) -> NodeResult[str]:
        assert require_context().workflow.workflow == 'task'
        await execute_workflow(inner, None)
        assert require_context().workflow.workflow == 'task'
        return NodeResult(output='outer')

    outer = Workflow(name='task', entry='improve', nodes=[Node('improve', outer_node)])

    ctx = ExecutionContext()
    async with ctx:
        await execute_workflow(outer, None)
        with pytest.raises(RuntimeError, match='No workflow node is active'):
            _ = ctx.workflow

    assert set(ctx.workflow_states) == {
        'task/task',
        'task/task/improve#1/agent/judge',
    }


@pytest.mark.asyncio
async def test_execute_workflow_validates_state_identity_and_cursor() -> None:
    async def done(_workflow_input: None) -> NodeResult[None]:
        return NodeResult(output=None)

    workflow = Workflow(name='current', version='2', entry='done', nodes=[Node('done', done)])
    ctx = ExecutionContext(
        workflow_states={
            'task/current': {
                'workflow': 'old',
                'version': '1',
                'cursor': 'done',
            }
        }
    )

    async with ctx:
        with pytest.raises(WorkflowStateMismatchError, match="'old'@1"):
            await execute_workflow(workflow, None)


@pytest.mark.asyncio
async def test_execute_workflow_enforces_visit_limit_before_reentering_node() -> None:
    async def loop(_workflow_input: None) -> NodeResult[int]:
        return NodeResult(output=1, next='loop')

    workflow = Workflow(name='limited', entry='loop', nodes=[Node('loop', loop)], max_steps=10)
    ctx = ExecutionContext()

    async with ctx:
        with pytest.raises(WorkflowLimitExceededError, match='max_visits=2'):
            await execute_workflow(
                workflow,
                None,
                options=WorkflowOptions(nodes={'loop': WorkflowNodeOptions(max_visits=2)}),
            )

    state = ctx.workflow_states['task/limited']
    assert len(state.history) == 2
    assert state.cursor == 'loop'


@pytest.mark.asyncio
async def test_execute_workflow_applies_node_timeout_without_commit() -> None:
    async def slow(_workflow_input: None) -> NodeResult[None]:
        await asyncio.sleep(1)
        return NodeResult(output=None)

    workflow = Workflow(name='timeout', entry='slow', nodes=[Node('slow', slow)])
    ctx = ExecutionContext()

    async with ctx:
        with pytest.raises(WorkflowNodeTimeoutError, match='timed out'):
            await execute_workflow(
                workflow,
                None,
                options=WorkflowOptions(nodes={'slow': WorkflowNodeOptions(timeout=0.01)}),
            )

    state = ctx.workflow_states['task/timeout']
    assert state.cursor == 'slow'
    assert state.history == ()


@pytest.mark.asyncio
async def test_execute_workflow_preserves_phase_span_name_and_adds_attributes() -> None:
    ended: list[Any] = []

    class Collector(SpanProcessor):
        def on_end(self, span: Any, _event: SpanEndEvent) -> None:
            ended.append(span)

    async def infer(_workflow_input: None) -> NodeResult[str]:
        return NodeResult(output='answer')

    workflow = Workflow(name='trace', entry='infer', nodes=[Node('infer', infer)])
    ctx = ExecutionContext()

    with processor_context([Collector()]):
        async with ctx:
            await execute_workflow(workflow, None)

    spans = [span for span in ended if span.name == 'task.infer']
    assert len(spans) == 1
    expected_attributes = {
        'workflow.name': 'trace',
        'workflow.version': '1',
        'workflow.scope': 'task/trace',
        'node.name': 'infer',
        'node.visit': 1,
        'node.sequence': 1,
        'type': 'task_phase',
        'phase': 'infer',
        'node.next': '',
        'workflow.completed': True,
    }
    assert spans[0].attributes.items() >= expected_attributes.items()
