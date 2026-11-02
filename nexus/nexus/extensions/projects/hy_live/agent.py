"""Multi-turn trainable cerebellum on hylive, the hy-live cerebellum and midbrain state machines."""

from __future__ import annotations

import asyncio
import logging

from datetime import datetime
from typing import Any, Literal

from hylive.cerebellum_contract import CEREBELLUM_TOOLS
from hylive.midbrain_contract import MIDBRAIN_TOOLS
from hylive.prompts import cerebellum_instructions
from pydantic import Field, model_validator

from nexus.core import AgentRunInput, AgentRunOutput, WithMainTrajectory, agent_registry, require_context
from nexus.trace import get_current_span_id, processor_context, update_span
from nexus.trace.phases import phase

from .cache_task_id import CacheTaskIdConfig, conversation_client
from .coordination.backend import WorkAgentBudget
from .coordination.models import AgentTask, OwnerEvent
from .episode_clock import EpisodeClock
from .health import aggregate_episode_health
from .hylive_shell import LiveShell
from .recording import EventLog
from .trajectory_export import dialogue_record, trajectory_record
from .trajectory_processor import MainTrajectoryProcessor
from .user_simulator import (
    CerebellumResponse,
    LLMUserSimulator,
    LLMUserSimulatorConfig,
    ScriptedUserSimulatorConfig,
    SimulatedConversation,
    UserSimulatorConfig,
    build_user_simulator,
)
from .web_search import WebSearchBackend, WebSearchProvider, build_web_search_backend


logger = logging.getLogger(__name__)


class LiveAgentAgentInput(AgentRunInput):
    """Configure one owner-scoped, simulated multi-turn cerebellum episode."""

    prompt: str | None = None
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
    """Working agent of every Task: CodeBuddy (ACP version 1) or WorkBuddy ``wb --acp`` (ACP version 2).

    Its config is ``work_agent_configs[backend_adapter]``.
    """
    work_agent_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    default_budget: WorkAgentBudget = Field(default_factory=WorkAgentBudget)
    web_search_backend: WebSearchBackend | None = Field(default=None, exclude=True)
    """Search backend injected by code; takes precedence over ``web_search_provider``."""
    web_search_provider: WebSearchProvider = 'none'
    """Backend built when none is injected: ``yuanbao`` is Yuanbao PromptSearch."""
    max_user_turns: int = Field(default=100, ge=1)
    task_delivery_timeout: float = Field(default=1800.0, gt=0)
    hold_next_user_until_permission: bool = False
    """After a delegate, wait for a harness permission notice before the next real user turn."""
    summarize_outcomes: bool = False
    """Ask the midbrain to summarize a completed or failed Task before the cerebellum is told.

    Off by default: the cerebellum is told the backend's own result or failure reason. Dataset rows turn it
    on in ``task_config``.
    """
    cache_task_id: CacheTaskIdConfig = Field(default_factory=CacheTaskIdConfig)
    """Which conversations send their own gateway ``cache_task_id`` so their prefix cache is hit."""
    record_events_path: str | None = None
    """Write every event fed to hylive and every command it gave to this JSONL file; None records nothing."""

    @model_validator(mode='after')
    def require_user_source(self) -> LiveAgentAgentInput:
        if self.prompt is None and self.user_simulator is None:
            raise ValueError('Either prompt or user_simulator must be configured')
        return self


class LiveAgentAgentOutput(AgentRunOutput, WithMainTrajectory):
    """Return the complete cerebellum trajectory and the background tasks kept by the harness."""

    owner_id: str
    summary: str
    conversation: SimulatedConversation
    tasks: list[AgentTask]
    delivery_events: list[OwnerEvent] = Field(default_factory=list)
    delivery_responses: list[CerebellumResponse] = Field(default_factory=list)
    episode_termination: str | None = None
    episode_clock: dict[str, float] = Field(default_factory=dict)
    """Wall time, time the cerebellum sat paused by the trainer, and time the whole episode was frozen."""
    cerebellum_trajectory: dict[str, Any] = Field(default_factory=dict)
    """The cerebellum's tools and full message log in the ``cerebellum.jsonl`` format."""
    midbrain_trajectory: dict[str, Any] = Field(default_factory=dict)
    """The midbrain's tools and full message log in the ``midbrain.jsonl`` format."""
    dialogue_trajectory: dict[str, Any] = Field(default_factory=dict)
    """Both layers merged by time into user turns, in the ``dialogue.jsonl`` format."""
    user_simulation: dict[str, Any] = Field(default_factory=dict)
    """How the scenario user ended and its node progress; empty for a scripted user."""


def _resolve_simulator_config(agent_input: LiveAgentAgentInput) -> UserSimulatorConfig:
    config = agent_input.user_simulator
    if config is None:
        assert agent_input.prompt is not None
        return ScriptedUserSimulatorConfig(turns=[agent_input.prompt])
    if isinstance(config, LLMUserSimulatorConfig) and config.initial_prompt is None and agent_input.prompt:
        return config.model_copy(update={'initial_prompt': agent_input.prompt})
    return config


def project_backend_config(agent_input: LiveAgentAgentInput) -> dict[str, Any]:
    """Return the working-agent config of ``backend_adapter``, naming the ACP CLI it selects.

    The CodeBuddy config is passed through unchanged. A WorkBuddy config gets ``acp_cli='workbuddy'``;
    the field is rejected when it names another CLI than the adapter.

    Raises:
        ValueError: ``acp_cli`` in the config disagrees with ``backend_adapter``.
    """
    adapter = agent_input.backend_adapter
    config = dict(agent_input.work_agent_configs.get(adapter) or {})
    acp_cli = config.get('acp_cli', adapter)
    if acp_cli != adapter:
        raise ValueError(
            f'work_agent_configs.{adapter}.acp_cli is {acp_cli!r} but backend_adapter is {adapter!r}; '
            'drop acp_cli from the config, the adapter selects it'
        )
    if adapter != 'codebuddy':
        config['acp_cli'] = adapter
    config.setdefault('llm_client_name', agent_input.external_client_name)
    return config


@agent_registry.register('hy-live', meta={'gateway': {'scaffold_policy': 'nexus'}})
async def live_agent(agent_input: LiveAgentAgentInput) -> LiveAgentAgentOutput:
    """Run the cerebellum with an episode-local capture of its actual client calls."""
    trajectory_processor = MainTrajectoryProcessor(client_key=agent_input.policy_client_name)
    with processor_context([trajectory_processor]):
        return await _run_live_agent(agent_input, trajectory_processor=trajectory_processor)


async def _run_live_agent(
    agent_input: LiveAgentAgentInput,
    *,
    trajectory_processor: MainTrajectoryProcessor,
) -> LiveAgentAgentOutput:
    span_id = get_current_span_id()
    ctx = require_context()
    if not callable(getattr(ctx, 'get_runtime', None)):
        raise RuntimeError(
            'hy-live requires an execution context with a sandbox runtime (ctx.get_runtime); '
            f'got {type(ctx).__name__} without one'
        )
    cache_task_id = agent_input.cache_task_id
    policy_client = conversation_client(
        ctx.get_openai_client(agent_input.policy_client_name),
        conversation='cerebellum',
        enabled=cache_task_id.cerebellum,
        required=True,
    )
    midbrain_client = conversation_client(
        ctx.get_openai_client(agent_input.midbrain_client_name or agent_input.external_client_name),
        conversation='midbrain',
        enabled=cache_task_id.midbrain,
    )
    owner_id = agent_input.owner_id or ctx.run_id or f'hy-live-owner-{id(ctx)}'
    clock = EpisodeClock()
    simulator = build_user_simulator(_resolve_simulator_config(agent_input), ctx, clock=clock)
    if isinstance(simulator, LLMUserSimulator):
        simulator.runtime.client = conversation_client(
            simulator.runtime.client,
            conversation='user_simulator',
            enabled=cache_task_id.user_simulator,
        )
    event_log = EventLog(agent_input.record_events_path) if agent_input.record_events_path else None
    shell = LiveShell(
        ctx=ctx,
        owner_id=owner_id,
        clients={'cerebellum': policy_client, 'midbrain': midbrain_client},
        cerebellum_system=cerebellum_instructions(agent_input.user_place, agent_input.user_now),
        trusted_config=project_backend_config(agent_input),
        budget=agent_input.default_budget,
        search_backend=agent_input.web_search_backend or build_web_search_backend(agent_input.web_search_provider),
        summarize_outcomes=agent_input.summarize_outcomes,
        clock=clock,
        session_cache_task_id=cache_task_id.backend,
        event_log=event_log,
    )
    # Only the HyTuner client has an abort gate; with any other policy client the episode never freezes.
    clock.bind(abort_event=getattr(policy_client, 'abort_event', None), background_working=shell.background_working)

    try:
        # NOTE(wallyshi): Starting the sandbox takes minutes. Inside the episode the user would ask about
        # progress before any Task could run, so the conversation starts only once the sandbox is up.
        # NOTE(jaxlliu): the phase span is what reports this wait as hytuner's sandbox_init_time; it is
        # the only episode time that falls outside the conversation trace.
        with phase('task.init.get_runtime'):
            await shell.warm_up()
        live = await shell.run(
            simulator=simulator,
            max_user_turns=agent_input.max_user_turns,
            task_delivery_timeout=agent_input.task_delivery_timeout,
            hold_next_user_until_permission=agent_input.hold_next_user_until_permission,
        )
        episode, state = live.episode, live.state
        termination_reason = aggregate_episode_health(
            tasks=episode.tasks,
            episode_termination=episode.episode_termination,
        )['termination_reason']
        user_simulation = simulator.outcome()
        if user_simulation:
            logger.info(
                'hy-live user simulation owner=%s ending=%s elapsed_s=%s state=%s decisions=%s rejected=%s discarded=%s',
                owner_id,
                user_simulation['ending'],
                user_simulation['elapsed_s'],
                user_simulation['state'],
                user_simulation['decisions'],
                user_simulation['rejected_decisions'],
                user_simulation['discarded_decisions'],
            )
        if span_id:
            update_span(span_id, attributes={'hytuner.termination_reason': termination_reason})
        return LiveAgentAgentOutput(
            main_trajectory=trajectory_processor.snapshot_result(),
            owner_id=owner_id,
            summary=episode.summary,
            conversation=episode.conversation,
            tasks=episode.tasks,
            delivery_events=episode.delivery_events,
            delivery_responses=episode.delivery_responses,
            episode_termination=episode.episode_termination,
            episode_clock=episode.clock or clock.summary(),
            cerebellum_trajectory=trajectory_record(CEREBELLUM_TOOLS, state.cerebellum.messages),
            midbrain_trajectory=trajectory_record(MIDBRAIN_TOOLS, state.midbrain_log),
            dialogue_trajectory=dialogue_record(
                CEREBELLUM_TOOLS, state.cerebellum.messages, MIDBRAIN_TOOLS, state.midbrain_log
            ),
            user_simulation=user_simulation,
        )
    except (Exception, asyncio.CancelledError):
        try:
            if span_id:
                update_span(span_id, attributes={'hytuner.termination_reason': 'UNHANDLED_ERROR'})
        except Exception:
            logger.exception('Failed to record interrupted cerebellum termination; preserving the original exception')
        raise
    finally:
        # NOTE(wallyshi): Cleanup failures are logged, not raised, so they never replace the episode's
        # own result or exception; the shell bounds its own wait for the midbrain and the sessions.
        try:
            await simulator.close()
        except Exception:
            logger.exception('Failed to close the user simulator owner=%s', owner_id)
        try:
            await shell.close()
        except Exception:
            logger.exception('Failed to close the hy-live project Sessions owner=%s', owner_id)
        if event_log is not None:
            event_log.close()


__all__ = [
    'LiveAgentAgentInput',
    'LiveAgentAgentOutput',
    'live_agent',
]
