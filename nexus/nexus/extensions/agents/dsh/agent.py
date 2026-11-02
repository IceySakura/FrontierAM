"""DeepSeek Harness agent driven exclusively through ACP."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import Field

from nexus.core import UserInputRequest, WithInputProvider, agent_registry
from nexus.extensions.agents.installed_common import InstalledAgentOutput, InstalledNodeCliInput
from nexus.extensions.agents.installed_common.multiturn import run_installed_multiturn

from .installer import DEFAULT_DSH_ACP_VERSION, DEFAULT_DSH_PREBUILT_BASE_URL, DEFAULT_DSH_VERSION


async def _no_more_user_input(_request: UserInputRequest | str | None) -> None:
    """End after one ACP turn when no continuation provider is supplied."""
    return


class DshInput(InstalledNodeCliInput, WithInputProvider):
    """Input for the ACP-only DeepSeek Harness integration."""

    get_user_input: Callable[[UserInputRequest | str | None], Awaitable[str | None]] = Field(
        default=_no_more_user_input
    )
    """Continuation provider. The default returns ``None`` after the first turn."""

    session_id: str | None = None
    max_turns: int = Field(default=20, ge=1)
    close_on_finish: bool = True

    version: str = DEFAULT_DSH_VERSION
    acp_version: str = DEFAULT_DSH_ACP_VERSION
    prebuilt_base_url: str | None = DEFAULT_DSH_PREBUILT_BASE_URL
    extra_env: dict[str, str] | None = None

    permission_mode: Literal['read-only', 'workspace-write', 'danger-full-access'] = 'workspace-write'
    permission_policy: Literal['allow', 'reject'] = 'allow'

    context_window: int = Field(default=131072, ge=1)
    max_output_tokens: int = Field(default=32768, ge=1)
    transcript_timeout: float = Field(default=5.0, gt=0)
    record_trace_events: bool = True
    retain_session_artifacts: bool = False


class DshOutput(InstalledAgentOutput):
    """Output from one or more turns on the same DSH ACP session."""

    session_id: str
    turns: int
    events: list[dict[str, Any]] = Field(default_factory=list)
    ended_by_user: bool = False
    reached_max_turns: bool = False
    status: str | None = None


@agent_registry.register('dsh')
async def run_dsh(input_data: DshInput) -> DshOutput:
    """Run DeepSeek Harness through one persistent ACP subprocess."""
    from .session_runtime import close_dsh_session, run_dsh_session_turn

    result = await run_installed_multiturn(
        input_data,
        agent_key='dsh',
        agent_label='DeepSeek Harness',
        session_namespace='dsh-acp-session',
        session_turn_target=run_dsh_session_turn,
        close_session_target=close_dsh_session,
        runtime_kwargs={
            'working_dir': input_data.working_dir,
            'timeout': input_data.timeout,
            'version': input_data.version,
            'acp_version': input_data.acp_version,
            'npm_registry': input_data.npm_registry,
            'prebuilt_base_url': input_data.prebuilt_base_url,
            'extra_env': dict(input_data.extra_env or {}),
            'permission_mode': input_data.permission_mode,
            'permission_policy': input_data.permission_policy,
            'context_window': input_data.context_window,
            'max_output_tokens': input_data.max_output_tokens,
            'transcript_timeout': input_data.transcript_timeout,
            'record_trace_events': input_data.record_trace_events,
            'retain_session_artifacts': input_data.retain_session_artifacts,
        },
    )
    return DshOutput(**result)


__all__ = ['DshInput', 'DshOutput', 'run_dsh']
