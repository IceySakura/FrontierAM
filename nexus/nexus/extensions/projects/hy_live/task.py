"""Run hy-live and expose its result to the reward layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from hylive.cerebellum_contract import CEREBELLUM_TOOLS
from pydantic import Field, model_validator

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    task_registry,
)
from nexus.extensions.projects.hy_live.agent import LiveAgentAgentOutput
from nexus.extensions.projects.hy_live.cache_task_id import CacheTaskIdConfig
from nexus.extensions.projects.hy_live.coordination.backend import WorkAgentBudget
from nexus.extensions.projects.hy_live.coordination.models import AgentTask, OwnerEvent
from nexus.extensions.projects.hy_live.episode_view import EPISODE_VIEW_KEY, build_episode_view
from nexus.extensions.projects.hy_live.health import aggregate_episode_health
from nexus.extensions.projects.hy_live.user_simulator import (
    CerebellumResponse,
    SimulatedConversation,
    UserSimulatorConfig,
)
from nexus.extensions.projects.hy_live.web_search import WebSearchBackend, WebSearchProvider


class LiveAgentTaskInput(TaskRunInput, WithAgentRunner):
    """Configure the hy-live agent executed by this task."""

    prompt: str | None = None
    user_prompt: str | None = Field(default=None, exclude=True)
    user_simulator: UserSimulatorConfig | None = None
    owner_id: str | None = None
    external_client_name: str = 'external'
    policy_client_name: str = 'main'
    """Client of the trainable cerebellum."""
    midbrain_client_name: str | None = None
    """Client of the midbrain, or None to share ``external_client_name`` with the backend."""
    user_place: dict[str, str] | str | None = None
    """User location told to the cerebellum, e.g. ``{'country': '中国', 'city': '上海'}`` or ``'北京'``; None means Shanghai."""
    user_now: datetime | None = None
    """User wall-clock time (ISO datetime) told to the cerebellum; None means the current Shanghai time."""
    backend_adapter: Literal['codebuddy', 'workbuddy'] = 'workbuddy'
    """Working agent of every Task, configured by ``work_agent_configs[backend_adapter]``."""
    work_agent_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    default_budget: WorkAgentBudget = Field(default_factory=WorkAgentBudget)
    web_search_backend: WebSearchBackend | None = Field(default=None, exclude=True)
    web_search_provider: WebSearchProvider = 'none'
    """Backend built when none is injected: ``yuanbao`` is Yuanbao PromptSearch."""
    max_user_turns: int = Field(default=100, ge=1)
    task_delivery_timeout: float = Field(default=1800.0, gt=0)
    hold_next_user_until_permission: bool = False
    """After a delegate, wait for a harness permission notice before the next real user turn."""
    summarize_outcomes: bool = False
    """Ask the midbrain to summarize a completed or failed Task before the cerebellum is told.

    Off by default: the cerebellum is told the backend's own result or failure reason. A user-simulator
    dataset row turns it on beside ``user_simulator`` on ``task_config``.
    """
    cache_task_id: CacheTaskIdConfig = Field(default_factory=CacheTaskIdConfig)
    """Which conversations send their own gateway ``cache_task_id`` so their prefix cache is hit."""
    record_events_path: str | None = None
    """Write every exchange with the outside, in hylive's contract, to this JSONL file."""

    @model_validator(mode='before')
    @classmethod
    def inject_dataset_user_prompt(cls, value: Any) -> Any:
        """Use the dataset-injected ``user_prompt`` as ``prompt`` when no prompt is configured."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        prompt = normalized.get('prompt')
        user_prompt = normalized.pop('user_prompt', None)
        if prompt is not None and user_prompt is not None and prompt != user_prompt:
            raise ValueError('prompt and user_prompt must match when both are provided')
        if prompt is None and user_prompt is not None:
            normalized['prompt'] = user_prompt
        return normalized


class LiveAgentTaskOutput(TaskRunOutput):
    """Return the owner-scoped policy conversation and the background tasks kept by the harness."""

    prompt: str | None
    summary: str
    score: float | None = None
    policy_messages: list[dict[str, Any]]
    owner_id: str
    conversation: SimulatedConversation
    tasks: list[AgentTask]
    delivery_events: list[OwnerEvent]
    delivery_responses: list[CerebellumResponse]
    cerebellum_trajectory: dict[str, Any] = Field(default_factory=dict)
    """The cerebellum's tools and full message log in the ``cerebellum.jsonl`` format."""
    midbrain_trajectory: dict[str, Any] = Field(default_factory=dict)
    """The midbrain's tools and full message log in the ``midbrain.jsonl`` format."""
    dialogue_trajectory: dict[str, Any] = Field(default_factory=dict)
    """Both layers merged by time into user turns, in the ``dialogue.jsonl`` format."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    """Merged by the HyTuner runner into the trajectory's extra fields, which reach the reward loop."""


@task_registry.register('hy-live')
async def live_agent_task(task_input: LiveAgentTaskInput) -> LiveAgentTaskOutput:
    """Run hy-live and expose its result to the task/reward layer."""
    agent_output = await task_input.run_agent(
        prompt=task_input.prompt,
        user_simulator=task_input.user_simulator,
        owner_id=task_input.owner_id,
        external_client_name=task_input.external_client_name,
        policy_client_name=task_input.policy_client_name,
        midbrain_client_name=task_input.midbrain_client_name,
        user_place=task_input.user_place,
        user_now=task_input.user_now,
        backend_adapter=task_input.backend_adapter,
        work_agent_configs=task_input.work_agent_configs,
        default_budget=task_input.default_budget,
        web_search_backend=task_input.web_search_backend,
        web_search_provider=task_input.web_search_provider,
        max_user_turns=task_input.max_user_turns,
        task_delivery_timeout=task_input.task_delivery_timeout,
        hold_next_user_until_permission=task_input.hold_next_user_until_permission,
        summarize_outcomes=task_input.summarize_outcomes,
        cache_task_id=task_input.cache_task_id,
        record_events_path=task_input.record_events_path,
    )
    if not isinstance(agent_output, LiveAgentAgentOutput):
        raise TypeError(f'hy-live runner returned {type(agent_output).__name__}, expected LiveAgentAgentOutput')
    return build_live_agent_task_output(
        prompt=task_input.prompt,
        external_client_name=task_input.external_client_name,
        policy_client_name=task_input.policy_client_name,
        agent_output=agent_output,
    )


def build_live_agent_task_output(
    *,
    prompt: str | None,
    external_client_name: str,
    policy_client_name: str,
    agent_output: LiveAgentAgentOutput,
) -> LiveAgentTaskOutput:
    """Map agent output to task output with aggregated health metadata."""
    health = aggregate_episode_health(
        tasks=agent_output.tasks,
        episode_termination=agent_output.episode_termination,
    )
    return LiveAgentTaskOutput(
        prompt=prompt,
        summary=agent_output.summary,
        policy_messages=agent_output.main_trajectory,
        owner_id=agent_output.owner_id,
        conversation=agent_output.conversation,
        tasks=agent_output.tasks,
        delivery_events=agent_output.delivery_events,
        delivery_responses=agent_output.delivery_responses,
        cerebellum_trajectory=agent_output.cerebellum_trajectory,
        midbrain_trajectory=agent_output.midbrain_trajectory,
        dialogue_trajectory=agent_output.dialogue_trajectory,
        metadata={
            'external_client_name': external_client_name,
            'policy_client_name': policy_client_name,
            'task_statuses': [task.status for task in agent_output.tasks],
            'episode_clock': agent_output.episode_clock,
            'user_simulation': agent_output.user_simulation,
            **health,
        },
        extra_fields={
            EPISODE_VIEW_KEY: build_episode_view(
                messages=agent_output.main_trajectory,
                delivery_responses=agent_output.delivery_responses,
                termination_reason=health['termination_reason'],
                health_status=health['health_status'],
                tools=CEREBELLUM_TOOLS,
            )
        },
    )


__all__ = [
    'LiveAgentTaskInput',
    'LiveAgentTaskOutput',
    'build_live_agent_task_output',
    'live_agent_task',
]
