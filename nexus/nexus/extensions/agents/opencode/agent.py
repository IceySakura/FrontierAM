"""OpenCode Agent.

Runs the OpenCode CLI inside a runtime sandbox in headless mode.  The
runtime-side setup installs the CLI via npm, starts an OpenAI-compatible
proxy, generates an opencode config pointing at that proxy, and then executes
``opencode run`` with the task prompt.  LLM calls flow through the host-side
OpenAI client via Socket.IO.
"""

import logging

from typing import Literal

from pydantic import BaseModel, Field

from nexus.core import agent_registry
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)

from .installer import DEFAULT_OPENCODE_VERSION


logger = logging.getLogger(__name__)


class PreStartScript(BaseModel):
    """A script to execute inside the runtime before OpenCode starts."""

    interpreter: Literal['bash', 'python'] = 'bash'
    code: str
    """Inline source code for the chosen interpreter."""

    timeout: float = 300.0
    cwd: str | None = None
    """Override cwd.  Defaults to ``working_dir``."""

    env: dict[str, str] | None = None
    """Extra env vars merged on top of the OpenCode launch env."""

    fail_on_error: bool = True
    """If True, non-zero exit aborts the agent run."""


class OpenCodeInput(InstalledNodeCliInput):
    """Input for OpenCode agent."""

    version: str = DEFAULT_OPENCODE_VERSION
    """OpenCode CLI package version (``latest`` or e.g. ``1.14.22``)."""

    model_id: str = 'nexus-proxy-model'
    """Model ID used in the opencode config.  The proxy forwards all requests
    to the host-side OpenAI client regardless of the model name."""

    extra_env: dict[str, str] | None = None
    """Extra environment variables for the OpenCode process."""

    extra_cli_args: list[str] = Field(default_factory=list)
    """Additional CLI arguments appended to the ``opencode run`` command."""

    pre_start_scripts: list[PreStartScript] = Field(default_factory=list)
    """Scripts executed in order before OpenCode starts."""

    output_format: Literal['default', 'json'] = 'default'
    """Output format: ``default`` for human-readable, ``json`` for raw events."""


class OpenCodeOutput(InstalledAgentOutput):
    """Output from OpenCode agent."""


@agent_registry.register('opencode')
async def run_opencode(input_data: OpenCodeInput) -> OpenCodeOutput:
    """Run OpenCode agent inside the runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)
    extra_env = dict(input_data.extra_env or {})

    logger.debug(f'Starting OpenCode agent for task: {task[:100]}...')

    from .runtime_setup import setup_and_run_opencode

    result = await call_runtime_agent(
        setup_and_run_opencode,
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        npm_registry=input_data.npm_registry,
        model_id=input_data.model_id,
        extra_env=extra_env,
        extra_cli_args=list(input_data.extra_cli_args),
        pre_start_scripts=[s.model_dump() for s in input_data.pre_start_scripts],
        output_format=input_data.output_format,
    )

    logger.info(f'OpenCode execution completed: success={result.get("success")}')
    return OpenCodeOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
    )


__all__ = [
    'OpenCodeInput',
    'OpenCodeOutput',
    'PreStartScript',
    'run_opencode',
]
