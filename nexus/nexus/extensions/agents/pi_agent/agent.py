"""Run the pi coding agent CLI as a Nexus installed agent."""

import logging

from typing import Any, Literal

from pydantic import Field

from nexus.core import agent_registry
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)

from .installer import DEFAULT_PI_VERSION
from .runtime_setup import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_TOOLS,
)


logger = logging.getLogger(__name__)


class PiInput(InstalledNodeCliInput):
    """Input for a single complete pi CLI run."""

    version: str = DEFAULT_PI_VERSION
    """pi npm package version (``latest`` or e.g. ``0.84.1``)."""

    model_id: str = DEFAULT_MODEL_ID
    """Model id advertised to pi.  The Nexus proxy forwards every request to the
    host-side client regardless of this name."""

    context_window: int = DEFAULT_CONTEXT_WINDOW
    """Context window pi assumes when deciding to compact."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    """Maximum output tokens per request."""

    reasoning: bool = False
    """Advertise the model as reasoning-capable, enabling ``thinking``."""

    thinking: Literal['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'] | None = None
    """Thinking level.  Requires ``reasoning=True`` to have any effect."""

    output_format: Literal['text', 'json'] = 'json'
    """``json`` streams JSONL events and populates ``trace``; ``text`` prints only
    the final assistant message."""

    tools: list[str] | None = None
    """Tool allowlist.  Defaults to every built-in tool, including the read-only
    ``grep``/``find``/``ls`` that pi ships disabled.  Pass ``[]`` to disable all."""

    load_context_files: bool = False
    """Load ``AGENTS.md``/``CLAUDE.md`` from the working dir and its ancestors.

    Off by default: pi walks every parent directory up to the filesystem root, so
    a stray context file outside the repo would silently join the system prompt
    and make results depend on the sandbox layout rather than the task.  Turn on
    for real development work, where those files are legitimate instructions.
    """

    trust_project_files: bool = False
    """Trust project-local ``.pi/`` settings, extensions and skills.

    Independent of ``load_context_files``: in pi, ``--no-approve`` does not
    suppress ``AGENTS.md`` loading, and ``--no-context-files`` does not suppress
    project settings.  Off by default because these resources execute code.
    """

    load_extensions: bool = False
    """Allow pi to discover extensions outside the project trust mechanism."""

    extra_env: dict[str, str] | None = None
    """Extra environment variables.  Nexus routing and isolation vars are protected."""

    extra_cli_args: list[str] = Field(default_factory=list)
    """Additional CLI arguments inserted before the prompt."""


class PiOutput(InstalledAgentOutput):
    """Output from a pi CLI run."""

    trace: dict[str, Any] | None = None
    """Summary of the ``--mode json`` event stream: event counts, turns, tool
    calls, malformed line count, the final assistant text, and the final
    ``stop_reason``/``error_message``.  None in text mode.

    ``stop_reason`` is what makes ``success`` trustworthy in JSON mode: pi exits
    0 even when the turn ended in ``error``/``aborted``, so a failed run is
    detected from the trace rather than the exit code.
    """


@agent_registry.register('pi')
async def run_pi(input_data: PiInput) -> PiOutput:
    """Run pi inside the runtime sandbox for one task."""
    task = coerce_user_prompt(input_data.user_prompt)

    if input_data.thinking and not input_data.reasoning:
        logger.warning('pi thinking=%s ignored: the model is not advertised as reasoning-capable', input_data.thinking)

    logger.debug('Starting pi agent for task: %s...', task[:100])

    from .runtime_setup import setup_and_run_pi

    result = await call_runtime_agent(
        setup_and_run_pi,
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        npm_registry=input_data.npm_registry,
        model_id=input_data.model_id,
        context_window=input_data.context_window,
        max_tokens=input_data.max_tokens,
        reasoning=input_data.reasoning,
        thinking=input_data.thinking,
        output_format=input_data.output_format,
        tools=list(input_data.tools) if input_data.tools is not None else list(DEFAULT_TOOLS),
        load_context_files=input_data.load_context_files,
        trust_project_files=input_data.trust_project_files,
        load_extensions=input_data.load_extensions,
        extra_env=dict(input_data.extra_env or {}),
        extra_cli_args=list(input_data.extra_cli_args),
    )

    logger.info('pi execution completed: success=%s', result.get('success'))
    return PiOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
        trace=result.get('trace'),
    )


__all__ = [
    'PiInput',
    'PiOutput',
    'run_pi',
]
