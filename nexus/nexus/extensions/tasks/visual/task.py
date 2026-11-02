"""Supply a visual task's messages and workspace specification to its agent."""

import copy

from typing import Any

from pydantic import Field, field_validator, model_validator

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, task_registry
from nexus.core.agent import validate_history


class VisualTaskInput(TaskRunInput, WithAgentRunner):
    """Visual task data; inference and media-processing policies live in agent_config."""

    model_config = {'extra': 'forbid'}

    user_prompt: str | list[dict[str, Any]]
    history: list[dict[str, Any]] | None = None
    upload_workspace: str | None = Field(
        default=None, description='Local directory or ZIP to upload to the agent workspace.'
    )

    reward_system_router_id: str | None = Field(
        default=None, description='Reward-system route; None disables scoring in Nexus.'
    )
    answer: str = Field(default='', description='Reference answer supplied only to the reward system.')
    reward_system_extra: dict[str, Any] | None = None
    public_test: dict[str, Any] | None = None
    reward_system_strip_multimodal_content: bool | list[str] | None = None
    reward_system_max_tool_images: int = Field(
        default=0, ge=0, description='With reward_system_include_trajectory, keep only this many latest tool images.'
    )
    reward_system_include_trajectory: bool = Field(
        default=False, description='Send the agent-generated trajectory to the reward system.'
    )

    @field_validator('reward_system_router_id')
    @classmethod
    def validate_reward_router(cls, value: str | None) -> str | None:
        if value is not None and (len(value.split('-')) != 3 or not all(part.strip() for part in value.split('-'))):
            raise ValueError('reward_system_router_id must be class1-class2-router_version, or None')
        return value

    @model_validator(mode='after')
    def validate_reward_media(self) -> 'VisualTaskInput':
        if self.reward_system_router_id is None:
            return self
        strip = self.reward_system_strip_multimodal_content
        if strip is True or (isinstance(strip, list) and self.reward_system_router_id in strip):
            return self
        from nexus.extensions.agents.visual_agent.resources.images import validate_reward_media

        for content in [self.user_prompt, *(message.get('content') for message in self.history or [])]:
            validate_reward_media(content)
        return self

    @model_validator(mode='after')
    def validate_tool_image_limit(self) -> 'VisualTaskInput':
        if self.reward_system_include_trajectory and self.reward_system_router_id is None:
            raise ValueError('reward_system_include_trajectory requires reward_system_router_id')
        if self.reward_system_max_tool_images and not self.reward_system_include_trajectory:
            raise ValueError('reward_system_max_tool_images requires reward_system_include_trajectory')
        return self

    @field_validator('history')
    @classmethod
    def validate_conversation(cls, history: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        validate_history(history)
        return history


class VisualTaskOutput(TaskRunOutput):
    """Optional score; answers and prepared workspace remain in agent_output."""

    score: float | None = None
    evaluation_details: dict[str, Any] | None = None


@task_registry.register('visual')
async def visual_task(task_input: VisualTaskInput) -> VisualTaskOutput:
    """Pass task inputs via the runner's standard keyword-argument interface."""
    client = None
    if task_input.reward_system_router_id is not None:
        from nexus.extensions.tasks.visual.reward import create_reward_client

        client = create_reward_client()
    # Prevent downstream agents from mutating the runner's task_config and original scoring input.
    agent_output = await task_input.run_agent(
        user_prompt=copy.deepcopy(task_input.user_prompt),
        history=copy.deepcopy(task_input.history),
        upload_workspace=task_input.upload_workspace,
    )
    if client is None:
        return VisualTaskOutput()

    from nexus.extensions.tasks.visual.reward import score_visual_response

    score, details = await score_visual_response(task_input, agent_output, client)
    return VisualTaskOutput(score=score, evaluation_details=details)


__all__ = ['VisualTaskInput', 'VisualTaskOutput', 'visual_task']
