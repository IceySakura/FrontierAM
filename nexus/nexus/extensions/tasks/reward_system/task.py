"""Unified one-turn task for Reward System and verifier-backed scores."""

from __future__ import annotations

import logging

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore, task_registry
from nexus.extensions.tasks.reward_system.schema import validate_prompt
from nexus.extensions.tasks.reward_system.score import compute_score


logger = logging.getLogger(__name__)


class RewardConfig(BaseModel):
    """Score configuration shared by external and local score backends."""

    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    scorer: Literal['reward_system', 'gsm8k', 'math'] = 'reward_system'
    agent_input_mode: Literal['prompt', 'user_prompt'] = 'prompt'
    reference_answer: Any
    question: str | None = None
    router_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    public_test: dict[str, Any] | list[Any] | str | None = None
    strip_multimodal: bool = False
    reasoning_end_token: str | None = None
    scale: float = 1.0
    max_retries: int = Field(default=1, ge=1)
    retry_delay: float = Field(default=5.0, ge=0)
    timeout: float = Field(default=1800.0, gt=0)


class RewardSystemTaskInput(TaskRunInput, WithAgentRunner):
    """Raw agent messages plus the score backend configuration."""

    prompt: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    reward: RewardConfig

    @field_validator('prompt')
    @classmethod
    def _canonical_prompt(cls, prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
        validate_prompt(prompt)
        return prompt


class RewardSystemTaskOutput(TaskRunOutput, WithScore):
    """Task score and response; trajectory/token metrics stay in Gateway."""

    score: float | None
    response: str
    extra_fields: dict[str, Any] = Field(default_factory=dict)


async def _response_for_scoring(agent_output: Any, reward: RewardConfig) -> str:
    response = getattr(agent_output, 'response', None)
    if response is None:
        response = getattr(agent_output, 'final_answer', None)
    if not isinstance(response, str):
        raise TypeError('reward_system requires an agent output with a string response')
    # Local verifier tasks historically scored the decoded response field.
    # Only the external Reward System path needs raw/reasoning reconstruction
    # so its wire payload stays compatible with the legacy reward loop.
    if reward.scorer != 'reward_system':
        return response
    raw_response = getattr(agent_output, 'raw_response', None)
    reasoning = getattr(agent_output, 'reasoning_content', None)
    if isinstance(raw_response, str):
        return raw_response
    if reasoning:
        if not reward.reasoning_end_token:
            raise ValueError('Split reasoning requires reward.reasoning_end_token')
        return reasoning + reward.reasoning_end_token + response
    return response


@task_registry.register('reward_system')
async def reward_system_task(task_input: RewardSystemTaskInput) -> RewardSystemTaskOutput:
    """Run the agent once, then dispatch one authoritative score computation."""
    if task_input.reward.agent_input_mode == 'user_prompt':
        if not task_input.reward.question:
            raise ValueError('reward.question is required for agent_input_mode=user_prompt')
        agent_output = await task_input.run_agent(user_prompt=task_input.reward.question)
    else:
        agent_output = await task_input.run_agent(prompt=task_input.prompt, tools=task_input.tools)
    # TODO(petersheng): skip compute_score when the Gateway already records the last main call as
    # length-limited (TRUNCATION). The legacy reward loop never sent overlong responses to the Reward
    # System, while this task still scores them and HyTuner then zeroes the reward, wasting a call.
    response = await _response_for_scoring(agent_output, task_input.reward)
    reward = task_input.reward
    logger.info('[reward_system] scorer=%s marker=%s', reward.scorer, reward.extra.get('ci_marker'))
    score = await compute_score(
        scorer=reward.scorer,
        response=response,
        reference_answer=reward.reference_answer,
        prompt=task_input.prompt,
        question=reward.question,
        router_id=reward.router_id,
        extra=reward.extra,
        public_test=reward.public_test,
        strip_multimodal=reward.strip_multimodal,
        max_retries=reward.max_retries,
        retry_delay=reward.retry_delay,
        timeout=reward.timeout,
    )
    # Preserve V1 failure sentinels below -80 while scaling real rewards.
    if score is not None and score >= -80:
        score *= reward.scale
    return RewardSystemTaskOutput(
        score=score,
        response=response,
        extra_fields={
            'score_backend': reward.scorer,
            **({'ci_marker': reward.extra['ci_marker']} if 'ci_marker' in reward.extra else {}),
        },
    )


__all__ = ['RewardConfig', 'RewardSystemTaskInput', 'RewardSystemTaskOutput', 'reward_system_task']
