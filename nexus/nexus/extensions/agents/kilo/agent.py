"""Kilo Code Agent.

Runs the Kilo Code CLI (@kilocode/cli) inside a runtime sandbox in headless
mode (``kilo run``).  The runtime-side setup installs the CLI via npm, starts
an OpenAI-compatible proxy, generates a kilo config pointing at that proxy,
and then executes ``kilo run --auto`` with the task prompt.  LLM calls flow
through the host-side OpenAI client via Socket.IO.
"""

import asyncio
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

from .installer import DEFAULT_KILO_VERSION


logger = logging.getLogger(__name__)


class PreStartScript(BaseModel):
    """A script to execute inside the runtime before Kilo starts."""

    interpreter: Literal['bash', 'python'] = 'bash'
    code: str
    """Inline source code for the chosen interpreter."""

    timeout: float = 300.0
    cwd: str | None = None
    """Override cwd.  Defaults to ``working_dir``."""

    env: dict[str, str] | None = None
    """Extra env vars merged on top of the Kilo launch env."""

    fail_on_error: bool = True
    """If True, non-zero exit aborts the agent run."""


class KiloInput(InstalledNodeCliInput):
    """Input for Kilo Code agent."""

    version: str = DEFAULT_KILO_VERSION
    """Kilo CLI package version (``latest`` or e.g. ``7.2.21``)."""

    model_id: str = 'nexus-proxy-model'
    """Model ID used in the kilo config.  The proxy forwards all requests
    to the host-side OpenAI client regardless of the model name."""

    extra_env: dict[str, str] | None = None
    """Extra environment variables for the Kilo process."""

    extra_cli_args: list[str] = Field(default_factory=list)
    """Additional CLI arguments appended to the ``kilo run`` command."""

    pre_start_scripts: list[PreStartScript] = Field(default_factory=list)
    """Scripts executed in order before Kilo starts."""

    output_format: Literal['default', 'json'] = 'default'
    """Output format: ``default`` for human-readable, ``json`` for raw events."""

    thinking_budget_tokens: int = 0
    """When > 0, enable extended thinking with this token budget (min 1024)."""


class KiloOutput(InstalledAgentOutput):
    """Output from Kilo Code agent."""


@agent_registry.register('kilo')
async def run_kilo(input_data: KiloInput) -> KiloOutput:
    """Run Kilo Code agent inside the runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)
    extra_env = dict(input_data.extra_env or {})

    logger.debug(f'Starting Kilo agent for task: {task[:100]}...')

    from .runtime_setup import setup_and_run_kilo

    # When kilo emits a 'suggest' tool_call it considers itself done and waits for
    # user input that never arrives in headless mode. Without intervention the
    # invoke would idle until host-side runtime_provider.request_timeout (1h),
    # which both wastes the slot and risks losing the patch we already wrote.
    # Detect 'suggest' on the host side via LLMRequestHandler.on_tool_call, then
    # cancel invoke_task with a short drain window so kilo can flush stdout.
    suggest_event = asyncio.Event()

    def _on_tool_call(tool_name: str) -> None:
        if 'suggest' in (tool_name or '').lower():
            logger.debug('[kilo] Suggest tool_call detected (%s) — will short-circuit invoke', tool_name)
            suggest_event.set()

    invoke_task = asyncio.ensure_future(
        call_runtime_agent(
            setup_and_run_kilo,
            on_tool_call=_on_tool_call,
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
            thinking_budget_tokens=input_data.thinking_budget_tokens,
        )
    )
    suggest_wait = asyncio.ensure_future(suggest_event.wait())

    done, _pending = await asyncio.wait(
        [invoke_task, suggest_wait],
        return_when=asyncio.FIRST_COMPLETED,
    )

    result: dict[str, Any]
    if suggest_wait in done and invoke_task not in done:
        # Suggest fired first. Give kilo 10s to drain stdout and exit cleanly,
        # otherwise cancel the invoke and synthesize a successful result —
        # task.py will pull the patch from the working_dir on disk.
        logger.debug('[kilo] Suggest detected; draining 10s before cancelling invoke...')
        try:
            result = await asyncio.wait_for(asyncio.shield(invoke_task), timeout=10.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            invoke_task.cancel()
            result = {
                'success': True,
                'output': '',
                'stderr': '[kilo:suggest-terminated]',
                'exit_code': 0,
                'execution_time': 0,
            }
            logger.info('[kilo] Cancelled invoke after suggest detection (treating as success)')
    else:
        suggest_wait.cancel()
        result = invoke_task.result()

    logger.info(f'Kilo execution completed: success={result.get("success")}')
    return KiloOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
    )


__all__ = [
    'KiloInput',
    'KiloOutput',
    'PreStartScript',
    'run_kilo',
]
