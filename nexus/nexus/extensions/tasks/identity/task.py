"""Identity task - passthrough task that just runs the agent."""

import logging

from pydantic import Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    UserInputRequest,
    WithAgentRunner,
    task_registry,
)


logger = logging.getLogger(__name__)


class IdentityTaskInput(TaskRunInput, WithAgentRunner):
    """Identity task input - no parameters needed."""

    user_inputs: list[str] = Field(default_factory=list)
    """Scripted follow-up user messages for agents that support WithInputProvider."""


class IdentityTaskOutput(TaskRunOutput):
    """Identity task output - nothing."""

    ...


@task_registry.register('identity')
async def identity_task(task_input: IdentityTaskInput) -> IdentityTaskOutput:
    """Identity task - just runs the agent."""

    pending_inputs = list(task_input.user_inputs)

    async def get_user_input(_hint: UserInputRequest | str | None) -> str | None:
        if not pending_inputs:
            return None
        return pending_inputs.pop(0)

    await task_input.run_agent(get_user_input=get_user_input)

    return IdentityTaskOutput()
