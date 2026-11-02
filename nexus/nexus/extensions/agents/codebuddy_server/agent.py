"""CodeBuddy Server Agent.

Runs the CodeBuddy agent-server inside a runtime sandbox.  The runtime-side
setup installs the agent-server via npm, starts an OpenAI-compatible proxy,
and communicates with the agent-server via HTTP.  LLM calls flow through the
host-side OpenAI client via Socket.IO.
"""

import logging

from typing import Any, Literal

from pydantic import BaseModel, Field

from nexus.core import agent_registry
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)

from .installer import DEFAULT_AGENT_SERVER_VERSION


logger = logging.getLogger(__name__)


class PreStartScript(BaseModel):
    """A script to execute inside the runtime before the agent-server starts."""

    interpreter: Literal['bash', 'python'] = 'bash'
    code: str
    """Inline source code for the chosen interpreter."""

    timeout: float = 300.0
    cwd: str | None = None
    """Override cwd.  Defaults to ``working_dir``."""

    env: dict[str, str] | None = None
    """Extra env vars merged on top of the agent-server launch env."""

    fail_on_error: bool = True
    """If True, non-zero exit aborts the agent run."""


class ReasoningParams(BaseModel):
    """Optional reasoning parameters for SWE mode."""

    effort: str = ''
    summary: str = ''


class CodeBuddyServerInput(InstalledNodeCliInput):
    """Input for CodeBuddy agent-server."""

    version: str = DEFAULT_AGENT_SERVER_VERSION
    """agent-server package version (``latest`` or e.g. ``0.0.23-beta``)."""

    mode: Literal['swe', 'craft'] = 'swe'
    """Completion mode: ``swe`` for SWE-Bench tasks, ``craft`` for general code generation."""

    model_id: str = 'nexus-proxy-model'
    """Model ID passed to the agent-server via ``MODEL_ID`` env var."""

    extra_env: dict[str, str] | None = None
    """Extra environment variables for the agent-server process."""

    reasoning_params: ReasoningParams | None = None
    """Optional reasoning parameters (SWE mode only)."""

    server_startup_timeout: float = 120
    """Seconds to wait for the agent-server to become ready."""

    pre_start_scripts: list[PreStartScript] = Field(default_factory=list)
    """Scripts executed in order before the agent-server starts."""

    max_step_limit: int = 0
    """Maximum number of agent loop iterations.  0 means effectively
    unlimited (patched to 999999)."""


class CodeBuddyServerOutput(InstalledAgentOutput):
    """Output from CodeBuddy agent-server."""

    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    """Tool calls made by the agent during execution."""

    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    """Tool results returned during execution."""


@agent_registry.register('codebuddy_server')
async def run_codebuddy_server(input_data: CodeBuddyServerInput) -> CodeBuddyServerOutput:
    """Run CodeBuddy agent-server inside the runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)
    extra_env = dict(input_data.extra_env or {})
    reasoning = input_data.reasoning_params.model_dump() if input_data.reasoning_params else {}

    logger.debug(f'Starting CodeBuddy Server agent for task: {task[:100]}...')

    from .runtime_setup import setup_and_run_agent_server

    result = await call_runtime_agent(
        setup_and_run_agent_server,
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        mode=input_data.mode,
        version=input_data.version,
        npm_registry=input_data.npm_registry,
        model_id=input_data.model_id,
        extra_env=extra_env,
        reasoning_params=reasoning,
        server_startup_timeout=input_data.server_startup_timeout,
        pre_start_scripts=[s.model_dump() for s in input_data.pre_start_scripts],
        max_step_limit=input_data.max_step_limit,
    )

    logger.info(f'CodeBuddy Server execution completed: success={result.get("success")}')
    return CodeBuddyServerOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
        tool_calls=result.get('tool_calls', []),
        tool_results=result.get('tool_results', []),
    )


__all__ = [
    'CodeBuddyServerInput',
    'CodeBuddyServerOutput',
    'PreStartScript',
    'ReasoningParams',
    'run_codebuddy_server',
]
