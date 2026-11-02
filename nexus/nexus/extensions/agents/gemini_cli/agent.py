"""Gemini CLI agent backed by a runtime-local proxy."""

import logging

from typing import Any, Literal

from nexus.core import agent_registry, require_context
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)

from .installer import DEFAULT_GEMINI_CLI_VERSION
from .runtime_setup import (
    DEFAULT_APPROVAL_MODE,
    DEFAULT_HOME_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FORMAT,
)


logger = logging.getLogger(__name__)


class GeminiCliInput(InstalledNodeCliInput):
    """Input for Gemini CLI agent."""

    version: str = DEFAULT_GEMINI_CLI_VERSION
    model: str | None = None
    approval_mode: Literal['default', 'auto_edit', 'yolo', 'plan'] = DEFAULT_APPROVAL_MODE
    output_format: Literal['text', 'json', 'stream-json'] = DEFAULT_OUTPUT_FORMAT
    install_root: str = '/tmp/gemini-cli'
    home_root: str = DEFAULT_HOME_ROOT


class GeminiCliOutput(InstalledAgentOutput):
    """Output from Gemini CLI agent."""

    response: str | None = None


def _resolve_openai_client_config(ctx: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    client = ctx.openai_client
    config = getattr(client, 'config', None)
    client_args = getattr(config, 'client_args', {}) or {}
    request_args = client.get_effective_request_args()
    return client_args, request_args


@agent_registry.register('gemini_cli')
async def run_gemini_cli(input_data: GeminiCliInput) -> GeminiCliOutput:
    """Run Gemini CLI in the runtime with a local proxy."""
    ctx = require_context()

    try:
        _client_args, request_args = _resolve_openai_client_config(ctx)
    except Exception as exc:
        return GeminiCliOutput(
            success=False,
            output='',
            stderr=f'Gemini CLI requires openai_clients.main configuration: {exc}',
            exit_code=-1,
            execution_time=0.0,
            response=None,
        )

    model = input_data.model or request_args.get('model') or DEFAULT_MODEL

    if not getattr(ctx, 'openai_client', None):
        return GeminiCliOutput(
            success=False,
            output='',
            stderr='Gemini CLI requires openai_clients.main to be configured',
            exit_code=-1,
            execution_time=0.0,
            response=None,
        )

    task = coerce_user_prompt(input_data.user_prompt)
    logger.debug('Starting Gemini CLI agent for task: %s...', task[:100])
    from .runtime_setup import setup_and_run_gemini_cli

    result = await call_runtime_agent(
        setup_and_run_gemini_cli,
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        model=str(model),
        npm_registry=input_data.npm_registry,
        approval_mode=input_data.approval_mode,
        output_format=input_data.output_format,
        install_root=input_data.install_root,
        home_root=input_data.home_root,
    )

    return GeminiCliOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0.0),
        response=result.get('response'),
    )


__all__ = [
    'GeminiCliInput',
    'GeminiCliOutput',
    'run_gemini_cli',
]
