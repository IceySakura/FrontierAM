"""Expose the official Kimi Code CLI as an installed Nexus scaffold."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from nexus.core import agent_registry
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)

from .installer import DEFAULT_VERSION


class KimiCodeInput(InstalledNodeCliInput):
    version: str = DEFAULT_VERSION
    """Exact @moonshot-ai/kimi-code npm version; the legacy Python CLI is not used."""

    model_id: str = 'nexus-model'
    """CLI model id; the configured Nexus client selects the actual backend model."""

    context_window: int = Field(default=262144, gt=0)
    max_tokens: int = Field(default=16384, gt=0)
    thinking_effort: str | None = None
    """Advertise reasoning support and select this effort; host enforcement still applies."""

    max_steps: int | None = Field(default=None, gt=0)
    agent_profile: str | None = None
    """Optional upstream agent profile; omitted uses Kimi Code's default agent."""

    extra_env: dict[str, str] = Field(default_factory=dict)


class KimiCodeOutput(InstalledAgentOutput):
    exit_status: str | None = None
    assistant_text: str | None = None
    session_id: str | None = None
    transcript_path: str | None = None
    artifact_dir: str | None = None
    trace: dict[str, Any] = Field(default_factory=dict)


@agent_registry.register('kimi_code')
async def run_kimi_code(input_data: KimiCodeInput) -> KimiCodeOutput:
    from .runtime_setup import setup_and_run_kimi_code

    result = await call_runtime_agent(
        setup_and_run_kimi_code,
        task=coerce_user_prompt(input_data.user_prompt),
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        npm_registry=input_data.npm_registry,
        model_id=input_data.model_id,
        context_window=input_data.context_window,
        max_tokens=input_data.max_tokens,
        thinking_effort=input_data.thinking_effort,
        max_steps=input_data.max_steps,
        agent_profile=input_data.agent_profile,
        extra_env=input_data.extra_env,
    )
    return KimiCodeOutput.model_validate(result)
