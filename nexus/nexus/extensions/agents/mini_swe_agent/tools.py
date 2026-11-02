"""RUNTIME bash tool for mini_swe_agent.

This is the white-box equivalent of upstream mini-swe-agent's
``LocalEnvironment.execute``: every call runs the command in a *fresh*
``bash -c`` subshell inside the sandbox (non-persistent — directory and env
changes do not carry over between calls), with a per-command timeout and the
mini env vars. stdout+stderr are combined into a single ``output`` string and
returned together with the return code, so the host loop can render the
observation template and detect the submit sentinel — exactly like upstream.

The tool body executes on the runtime (sandbox) side; the agent loop in
``agent.py`` runs host-side and dispatches here via ``ToolExecutor``.

Import discipline: this module is imported *inside the sandbox* to run the tool,
so it must not transitively import any host-only dependency (e.g. ``jinja2``).
Shared constants therefore come from the dependency-free ``constants`` module,
NOT from ``mini_core`` (which imports ``jinja2`` for template rendering).
"""

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.agents.mini_swe_agent.constants import (
    DEFAULT_ENV_TIMEOUT,
    DEFAULT_MINI_ENV,
)


class BashInput(BaseModel):
    """Input for the bash tool (matches upstream ``BASH_TOOL``)."""

    command: str = Field(description='The bash command to execute')


def _get_working_dir(context: ToolContext) -> str:
    """Working directory for commands (seeded by the agent / swe_task)."""
    return context.tool_state.get('working_dir') or '/testbed'


def _get_env_timeout(context: ToolContext) -> int:
    return int(context.tool_state.get('env_timeout') or DEFAULT_ENV_TIMEOUT)


def _get_extra_env(context: ToolContext) -> dict[str, str]:
    extra = context.tool_state.get('bash_extra_env')
    return dict(extra) if isinstance(extra, dict) else {}


@tool_registry.register(
    'mini_swe_agent/bash',
    meta={
        'name': 'bash',
        'description': 'Execute a bash command',
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    """Run a single bash command in a fresh subshell inside the runtime sandbox.

    Returns the combined stdout+stderr as text; the return code and any
    execution exception message are returned in ``meta`` so the agent loop can
    build the observation dict (``output``/``returncode``/``exception_info``)
    and render the observation template.
    """
    # Imported lazily: the executor module is meant to run inside the sandbox and
    # should not be imported at host-side module-discovery time.
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    working_dir = _get_working_dir(context)
    timeout = _get_env_timeout(context)
    extra_env = {**DEFAULT_MINI_ENV, **_get_extra_env(context)}

    result = await run_command(
        RunCommandRequest(
            command=params.command,
            cwd=working_dir,
            timeout=timeout,
            extra_env=extra_env,
        )
    )

    stdout = result.stdout or ''
    stderr = result.stderr or ''

    # Mirror LocalEnvironment.execute: stdout+stderr merged into a single stream.
    if result.status == CommandRunStatus.Finished:
        combined = stdout + stderr
        returncode = result.return_code if result.return_code is not None else 0
        exception_info = ''
    elif result.status == CommandRunStatus.TimeLimitExceeded:
        combined = stdout + stderr
        returncode = -1
        # Match subprocess.TimeoutExpired's str used by upstream's exception_info.
        exception_info = (
            f'An error occurred while executing the command: '
            f"Command '{params.command}' timed out after {timeout} seconds"
        )
    else:  # CommandRunStatus.Error
        combined = stdout
        returncode = -1
        exception_info = f'An error occurred while executing the command: {stderr or "unknown error"}'

    return ToolResponse(
        output=[{'type': 'text', 'text': combined}],
        meta={'returncode': returncode, 'exception_info': exception_info},
    )


__all__ = ['BashInput', 'bash_tool']
