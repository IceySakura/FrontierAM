"""Runtime bash tool for the R2 STEM mini-SWE-agent extension."""

from __future__ import annotations

import json
import logging

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command


logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600.0
MAX_OBSERVATION_CHARS = 10_000


class BashInput(BaseModel):
    """Input for independent bash command execution."""

    command: str = Field(description='The bash command to execute.')
    timeout: float | None = Field(default=None, description='Optional timeout in seconds for this command.')


def _make_output(text: str) -> list[ChatCompletionContentPartParam | dict]:
    return [{'type': 'text', 'text': text}]


def _combined_output(stdout: str | None, stderr: str | None) -> str:
    output = stdout or ''
    if stderr:
        output = f'{output}\n[stderr]\n{stderr}' if output else f'[stderr]\n{stderr}'
    return output


def _format_observation(*, output: str, returncode: int, exception_info: str = '') -> str:
    if len(output) < MAX_OBSERVATION_CHARS:
        payload = {'returncode': returncode, 'output': output}
    else:
        payload = {
            'returncode': returncode,
            'output_head': output[:5000],
            'output_tail': output[-5000:],
            'elided_chars': len(output) - MAX_OBSERVATION_CHARS,
            'warning': 'Output too long.',
        }
    if exception_info:
        payload['exception_info'] = exception_info
    return json.dumps(payload, ensure_ascii=False)


@tool_registry.register(
    'mini_swe_agent_stem/bash',
    meta={
        'name': 'bash',
        'description': (
            'Execute a bash command in the runtime container. Each call runs in an independent subshell, '
            'so cd/export state does not persist between calls.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    """Execute one independent bash command in the runtime."""
    params = BashInput.model_validate(params)
    timeout = float(params.timeout or context.tool_state.get('command_timeout') or DEFAULT_TIMEOUT)
    cwd = context.tool_state.get('working_dir') or None
    logger.debug('mini_swe_agent_stem/bash command=%r cwd=%s timeout=%s', params.command, cwd, timeout)
    response = await run_command(RunCommandRequest(command=params.command, cwd=cwd, timeout=timeout))

    if response.status == CommandRunStatus.TimeLimitExceeded:
        raw_output = _combined_output(response.stdout, response.stderr)
        exception_info = f'Command timed out after {timeout}s'
        return ToolResponse(
            output=_make_output(_format_observation(output=raw_output, returncode=-1, exception_info=exception_info)),
            meta={
                'command': params.command,
                'exit_code': -1,
                'timeout': True,
                'execution_time': response.execution_time,
                'raw_output': raw_output,
                'stdout': response.stdout,
                'stderr': response.stderr,
            },
        )

    if response.status == CommandRunStatus.Error:
        raw_output = _combined_output(response.stdout, response.stderr)
        exception_info = raw_output or 'Command execution failed'
        return ToolResponse(
            output=_make_output(_format_observation(output=raw_output, returncode=-1, exception_info=exception_info)),
            meta={
                'command': params.command,
                'exit_code': -1,
                'error': True,
                'raw_output': raw_output,
                'stdout': response.stdout,
                'stderr': response.stderr,
            },
        )

    returncode = response.return_code or 0
    raw_output = _combined_output(response.stdout, response.stderr)

    return ToolResponse(
        output=_make_output(_format_observation(output=raw_output, returncode=returncode)),
        meta={
            'command': params.command,
            'exit_code': returncode,
            'execution_time': response.execution_time,
            'raw_output': raw_output,
            'stdout': response.stdout,
            'stderr': response.stderr,
        },
    )


__all__ = ['BashInput', 'bash_tool']
