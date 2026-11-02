"""Bash tool for the white-box CC search agent.

Exposes a Claude-Code-compatible ``Bash`` tool surface with just
``command`` (required) and ``description`` (optional metadata). The Claude
Code ``timeout``, ``run_in_background`` and ``dangerouslyDisableSandbox``
parameters are intentionally omitted: ``run_in_background`` requires
companion ``BashOutput``/``KillShell`` tools that the terminal executor
does not expose, ``dangerouslyDisableSandbox`` conflicts with Nexus
runtime isolation, and ``timeout`` is not surfaced by the underlying
executor. Long-lived commands should append ``&`` and redirect output to
a file.

Execution is delegated to
:func:`nexus.extensions.agents.swe_agent.tools.bash_tool`, which runs each
command synchronously in a persistent tmux session and blocks until
completion before returning its full output. cwd, env vars and shell
state persist across calls because the session is reused.
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.agents.swe_agent.tools import BashInput as SweBashInput
from nexus.extensions.agents.swe_agent.tools import bash_tool as swe_bash_tool


BASH_TOOL_REGISTRY_NAME = 'cc_search_agent_whitebox/bash'
BASH_TOOL_NAME = 'Bash'

BASH_DESCRIPTION = """Executes a given bash command and returns its output.

The working directory persists between commands, but shell state does not. The shell environment is initialized from the user's profile (bash or zsh).

IMPORTANT: Avoid using this tool to run `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task.

# Instructions
 - If your command will create new directories or files, first use this tool to run `ls` to verify the parent directory exists and is the correct location.
 - Always quote file paths that contain spaces with double quotes in your command (e.g., cd "path with spaces/file.txt")
 - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it. In particular, never prepend `cd <current-directory>` to a `git` command — `git` already operates on the current working tree, and the compound triggers a permission prompt.
 - When issuing multiple commands:
  - If the commands are independent and can run in parallel, make multiple Bash tool calls in a single message. Example: if you need to run "git status" and "git diff", send a single message with two Bash tool calls in parallel.
  - If the commands depend on each other and must run sequentially, use a single Bash call with '&&' to chain them together.
  - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail.
  - DO NOT use newlines to separate commands (newlines are ok in quoted strings).
 - For git commands:
  - Prefer to create a new commit rather than amending an existing commit.
  - Before running destructive operations (e.g., git reset --hard, git push --force, git checkout --), consider whether there is a safer alternative that achieves the same goal. Only use destructive operations when they are truly the best approach.
  - Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign, -c commit.gpgsign=false) unless the user has explicitly asked for it. If a hook fails, investigate and fix the underlying issue.
 - Avoid unnecessary `sleep` commands:
  - Do not sleep between commands that can run immediately — just run them.
  - Do not retry failing commands in a sleep loop — diagnose the root cause.
  - If you must poll an external process, use a check command (e.g. `gh run view`) rather than sleeping first.
  - If you must sleep, keep the duration short to avoid blocking the user.
"""


_DESCRIPTION_FIELD_DESCRIPTION = """Clear, concise description of what this command does in active voice. Never use words like "complex" or "risk" in the description - just describe what it does.

For simple commands (git, npm, standard CLI tools), keep it brief (5-10 words):
- ls → "List files in current directory"
- git status → "Show working tree status"
- npm install → "Install package dependencies"

For commands that are harder to parse at a glance (piped commands, obscure flags, etc.), add enough context to clarify what it does:
- find . -name "*.tmp" -exec rm {} \\; → "Find and delete all .tmp files recursively"
- git reset --hard origin/main → "Discard all local changes and match remote main"
- curl -s url | jq '.data[]' → "Fetch JSON from URL and extract data array elements\""""


class BashInput(BaseModel):
    command: str = Field(
        description='The command to execute',
    )
    description: str | None = Field(
        default=None,
        description=_DESCRIPTION_FIELD_DESCRIPTION,
    )

    @model_validator(mode='before')
    @classmethod
    def _normalize_missing_required_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            normalized = dict(data)
            if not isinstance(normalized.get('command'), str):
                normalized['command'] = ''
            return normalized
        return data


@tool_registry.register(
    BASH_TOOL_REGISTRY_NAME,
    meta={
        'name': BASH_TOOL_NAME,
        'description': BASH_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    """Run a bash command by delegating to swe_agent's bash tool.

    ``params.description`` is accepted for parity with the Claude Code
    ``Bash`` schema; it is echoed back into ``response.meta`` for
    trajectory/logging consumers but never forwarded to the shell.
    """
    if not params.command.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Missing required parameter "command"'}],
            meta={'success': False},
        )

    response = await swe_bash_tool(
        SweBashInput(
            command=params.command,
        ),
        context,
    )

    if params.description is not None:
        response.meta.setdefault('description', params.description)

    return response


__all__ = [
    'BASH_DESCRIPTION',
    'BASH_TOOL_NAME',
    'BASH_TOOL_REGISTRY_NAME',
    'BashInput',
    'bash_tool',
]
