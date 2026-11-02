"""Demo tools for the demo agent.

Includes:
- think: A LOCAL tool for reasoning (runs in agent process)
- bash: A RUNTIME tool for executing commands (runs in sandbox)
"""

import subprocess

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


# ============== Think Tool (LOCAL) ==============


class ThinkInput(BaseModel):
    """Input for think tool."""

    thought: str = Field(description='Your thought or reasoning process')


@tool_registry.register(
    'demo/think',
    meta={
        'name': 'Think',
        'description': 'Use this tool to think through a problem step by step. '
        'Your thoughts will be recorded but not shown to the user.',
        'tool_type': ToolType.LOCAL,
    },
)
async def think_tool(params: ThinkInput, context: ToolContext) -> ToolResponse:
    """Think through a problem without taking action."""
    return ToolResponse(
        output=[{'type': 'text', 'text': 'Thought recorded.'}],
        meta={'thought': params.thought},
    )


# ============== Bash Tool (RUNTIME) ==============


class BashInput(BaseModel):
    """Input for bash tool."""

    command: str = Field(description='The bash command to execute')
    timeout: int = Field(default=60, description='Timeout in seconds')


@tool_registry.register(
    'demo/bash',
    meta={
        'name': 'Bash',
        'description': 'Execute a bash command in the sandbox environment. '
        'Use this to run shell commands, read files, etc.',
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    """Execute a bash command in the sandbox.

    This runs on the runtime server (local or remote).
    """
    try:
        result = subprocess.run(
            params.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=params.timeout,
        )

        output = result.stdout
        if result.stderr:
            output += f'\n[stderr]\n{result.stderr}'

        return ToolResponse(
            output=[{'type': 'text', 'text': output or '(no output)'}],
            meta={
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            },
        )

    except subprocess.TimeoutExpired:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Command timed out after {params.timeout}s'}],
            meta={'exit_code': -1, 'timeout': True},
        )
