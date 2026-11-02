from nexus.core.agent import AgentRunInput, AgentRunOutput, validate_agent
from nexus.core.task import TaskRunInput, TaskRunOutput, validate_task
from nexus.runner.single import _get_input_type


def test_validate_task_resolves_postponed_annotations() -> None:
    namespace = {
        'TaskRunInput': TaskRunInput,
        'TaskRunOutput': TaskRunOutput,
    }
    exec(
        """from __future__ import annotations

class FutureInput(TaskRunInput):
    value: int

class FutureOutput(TaskRunOutput):
    value: int

async def future_task(task_input: FutureInput) -> FutureOutput:
    return FutureOutput(value=task_input.value)
""",
        namespace,
    )

    result = validate_task(namespace['future_task'], 'future_task', {})

    assert result.meta == {}


def test_single_runner_resolves_postponed_task_input_annotation() -> None:
    namespace = {'TaskRunInput': TaskRunInput, 'TaskRunOutput': TaskRunOutput}
    exec(
        """from __future__ import annotations

class FutureInput(TaskRunInput):
    value: int

async def future_task(task_input: FutureInput) -> TaskRunOutput:
    return TaskRunOutput()
""",
        namespace,
    )

    assert _get_input_type(namespace['future_task']) is namespace['FutureInput']


def test_validate_agent_resolves_postponed_annotations() -> None:
    namespace = {'AgentRunInput': AgentRunInput, 'AgentRunOutput': AgentRunOutput}
    exec(
        """from __future__ import annotations

class FutureAgentInput(AgentRunInput):
    value: int

class FutureAgentOutput(AgentRunOutput):
    value: int

async def future_agent(agent_input: FutureAgentInput) -> FutureAgentOutput:
    return FutureAgentOutput(value=agent_input.value)
""",
        namespace,
    )

    result = validate_agent(namespace['future_agent'], 'future_agent', {})

    assert result.meta == {}
