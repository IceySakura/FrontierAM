"""Hermes agent integration for Nexus."""

import logging

from nexus.core import agent_registry, require_context
from nexus.extensions.agents.installed_common import (
    InstalledAgentInput,
    InstalledAgentOutput,
    call_runtime_agent,
    coerce_user_prompt,
)

from .installer import DEFAULT_HERMES_VERSION
from .runtime_setup import DEFAULT_TOOLSETS


logger = logging.getLogger(__name__)


class HermesInput(InstalledAgentInput):
    """Input for the Hermes agent."""

    model: str | None = None
    """Backend model id used by the host-side OpenAI-compatible client."""

    version: str = DEFAULT_HERMES_VERSION
    """Hermes Agent package version to install."""

    toolsets: str = DEFAULT_TOOLSETS
    """Comma-separated Hermes toolsets enabled for the one-shot run; defaults to Hermes' CLI preset."""

    install_root: str | None = None
    """Optional runtime install root for the Hermes virtual environment."""

    max_turns: int | None = None
    """Optional maximum number of tool-calling iterations."""


class HermesOutput(InstalledAgentOutput):
    """Execution result for the Hermes agent."""


def _resolve_model_name(input_data: HermesInput) -> str:
    if input_data.model:
        return input_data.model

    ctx = require_context()
    config = getattr(ctx.openai_client, 'config', None)
    if config is None:
        raise RuntimeError('Hermes requires agent_config.model or a configured main OpenAI client model')

    for source in (ctx.openai_client.get_effective_request_args(), getattr(config, 'client_args', {})):
        model = source.get('model')
        if isinstance(model, str) and model.strip():
            return model

    raise RuntimeError('Hermes requires agent_config.model or openai_clients.main.config.request_args.model')


@agent_registry.register('hermes')
async def run_hermes(input_data: HermesInput) -> HermesOutput:
    """Run Hermes inside the configured runtime and proxy its LLM calls to the host."""
    task = coerce_user_prompt(input_data.user_prompt)
    model = _resolve_model_name(input_data)

    logger.debug('Starting Hermes agent with model=%s, toolsets=%s', model, input_data.toolsets)
    from .runtime_setup import setup_and_run_hermes

    result = await call_runtime_agent(
        setup_and_run_hermes,
        task=task,
        model=model,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        toolsets=input_data.toolsets,
        install_root=input_data.install_root,
        max_turns=input_data.max_turns,
    )
    return HermesOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
    )


__all__ = [
    'HermesInput',
    'HermesOutput',
    'run_hermes',
]
