import uuid

import pytest

from nexus.core import (
    Node,
    NodeResult,
    TaskRunInput,
    TaskRunOutput,
    Workflow,
    WorkflowState,
    execute_workflow,
    task_registry,
)
from nexus.runner.single import RunInput, _resolve_run_input_env_suffixes, run


class WorkflowTaskOutput(TaskRunOutput):
    result: str


class WorkflowTaskInput(TaskRunInput):
    pass


def test_run_input_treats_workflow_states_as_opaque_data(monkeypatch) -> None:
    monkeypatch.delenv('WORKFLOW_STATE_MUST_NOT_BE_RESOLVED', raising=False)
    state = {
        'workflow': 'main',
        'version': '1',
        'cursor': 'work',
        'history': [
            {
                'sequence': 1,
                'node': 'prepare',
                'visit': 1,
                'output': {
                    'template': '${oc.env:WORKFLOW_STATE_MUST_NOT_BE_RESOLVED}',
                    'literal__env': 'WORKFLOW_STATE_MUST_NOT_BE_RESOLVED',
                },
                'next_node': 'work',
            }
        ],
    }

    parsed = RunInput.from_dict({'task_name': 'unused', 'workflow_states': {'task/main': state}})
    resolved = _resolve_run_input_env_suffixes(parsed)

    assert parsed.workflow_states['task/main'] == WorkflowState.model_validate(state)
    assert resolved.workflow_states == parsed.workflow_states


@pytest.mark.asyncio
async def test_single_run_returns_and_accepts_workflow_states() -> None:
    task_name = f'test_workflow_{uuid.uuid4().hex}'
    fail_work = True

    @task_registry.register(task_name)
    async def workflow_task(_task_input: WorkflowTaskInput) -> WorkflowTaskOutput:
        async def prepare(_workflow_input: WorkflowTaskInput) -> NodeResult[str]:
            return NodeResult(output='ready', next='work')

        async def work(_workflow_input: WorkflowTaskInput) -> NodeResult[str]:
            if fail_work:
                raise RuntimeError('work failed')
            return NodeResult(output='done')

        state = await execute_workflow(
            Workflow(
                name='main',
                entry='prepare',
                nodes=[Node('prepare', prepare), Node('work', work)],
            ),
            _task_input,
        )
        return WorkflowTaskOutput(result=state.output('work', as_type=str))

    first = await run(RunInput(task_name=task_name))

    assert first.success is False
    assert first.error is not None
    assert first.error.type == 'RuntimeError'
    assert first.workflow_states['task/main'].cursor == 'work'
    assert [visit.node for visit in first.workflow_states['task/main'].history] == ['prepare']

    fail_work = False
    second = await run(
        RunInput(
            task_name=task_name,
            workflow_states=first.workflow_states,
        )
    )

    assert second.success is True
    assert second.task_output == WorkflowTaskOutput(result='done')
    assert second.workflow_states['task/main'].cursor is None
    assert [visit.node for visit in second.workflow_states['task/main'].history] == ['prepare', 'work']
