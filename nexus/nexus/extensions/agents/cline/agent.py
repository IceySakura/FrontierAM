"""Cline Agent.

Runs the Cline CLI inside a runtime sandbox in headless mode.  The
runtime-side setup installs Cline from a bundled tarball, starts an
OpenAI-compatible proxy, and then executes the Cline CLI with the task
prompt.  LLM calls flow through the host-side OpenAI client via Socket.IO.
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

from .installer import DEFAULT_CLINE_VERSION


logger = logging.getLogger(__name__)


class PreStartScript(BaseModel):
    """A script to execute inside the runtime before Cline starts."""

    interpreter: Literal['bash', 'python'] = 'bash'
    code: str
    """Inline source code for the chosen interpreter."""

    timeout: float = 300.0
    cwd: str | None = None
    """Override cwd. Defaults to ``working_dir``."""

    env: dict[str, str] | None = None
    """Extra env vars merged on top of the Cline launch env."""

    fail_on_error: bool = True
    """If True, non-zero exit aborts the agent run."""


class ClineInput(InstalledNodeCliInput):
    """Input for Cline agent."""

    version: str = DEFAULT_CLINE_VERSION
    """Cline CLI package version (``latest`` or e.g. ``0.1.0``)."""

    model_id: str = 'nexus-proxy-model'
    """Model ID used in the Cline config. The proxy forwards all requests
    to the host-side OpenAI client regardless of the model name."""

    thinking_budget_tokens: int = 0
    """Extended thinking token budget. 0 = disabled; >0 adds ``--thinking`` flag."""

    extra_env: dict[str, str] | None = None
    """Extra environment variables for the Cline process."""

    extra_cli_args: list[str] = Field(default_factory=list)
    """Additional CLI arguments appended to the ``cline`` command."""

    pre_start_scripts: list[PreStartScript] = Field(default_factory=list)
    """Scripts executed in order before Cline starts."""


class ClineOutput(InstalledAgentOutput):
    """Output from Cline agent."""


@agent_registry.register('cline')
async def run_cline(input_data: ClineInput) -> ClineOutput:
    """Run Cline agent inside the runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)
    extra_env = dict(input_data.extra_env or {})

    logger.debug(f'Starting Cline agent for task: {task[:100]}...')

    from .runtime_setup import NOOP_FAILURE_STAGE, setup_and_run_cline

    result = await call_runtime_agent(
        setup_and_run_cline,
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        npm_registry=input_data.npm_registry,
        model_id=input_data.model_id,
        thinking_budget_tokens=input_data.thinking_budget_tokens,
        extra_env=extra_env,
        extra_cli_args=list(input_data.extra_cli_args),
        pre_start_scripts=[s.model_dump() for s in input_data.pre_start_scripts],
    )

    logger.info(f'Cline execution completed: success={result.get("success")}')

    # cline 3.0.3 ships a precompiled bun binary whose internal HTTP client
    # honors only the default ~5min headers timeout — neither AI_SDK_TIMEOUT
    # nor OPENAI_TIMEOUT actually feed into it. When a single LLM call exceeds
    # that, the bun fetch raises "operation timed out" and the cline process
    # exits non-zero with no patch produced. Treat that as a transient failure
    # so the batch runner's --retry-failed can re-run the case.
    stderr = result.get('stderr', '') or ''
    if not result.get('success', False) and 'operation timed out' in stderr.lower():
        raise RuntimeError(f'cline aborted by AI SDK fetch timeout (exit_code={result.get("exit_code")}) — retryable')

    # The runtime side already relaunched cline several times; a no-op that
    # survives that is still transient across sandboxes, so hand it to the batch
    # runner's --retry-failed rather than recording an empty run as a result.
    if result.get('failure_stage') == NOOP_FAILURE_STAGE:
        raise RuntimeError('cline exited 0 without producing any output — retryable')

    return ClineOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
        failure_kind=result.get('failure_kind'),
        failure_stage=result.get('failure_stage'),
    )


__all__ = [
    'ClineInput',
    'ClineOutput',
    'PreStartScript',
    'run_cline',
]
