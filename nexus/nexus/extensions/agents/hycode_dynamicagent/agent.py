"""HyCode DynamicAgent.

Runs ``@tencent/hycode-dynamicagent`` inside a runtime sandbox using the same
npm-install flow as the built-in HyCode extension. No host-side OpenCC binary
or plugin directory is uploaded; the npm package supplies the CLI, bundled
agents, and Code Arena MCP harness.
"""

import logging

from typing import Any, Literal

from pydantic import BaseModel, Field

from nexus.core import WithInputProvider, agent_registry
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)
from nexus.extensions.agents.installed_common.multiturn import run_installed_multiturn

from .installer import DEFAULT_HYCODE_DYNAMICAGENT_VERSION


logger = logging.getLogger(__name__)


class PreStartScript(BaseModel):
    """A script to execute inside the runtime before the CLI starts."""

    interpreter: Literal['bash', 'python'] = 'bash'
    code: str
    timeout: float = 300.0
    cwd: str | None = None
    env: dict[str, str] | None = None
    fail_on_error: bool = True


class ConfigFilePatch(BaseModel):
    """Merge *content* into the JSON file at *path* before launching."""

    path: str
    content: dict[str, Any]


class HyCodeDynamicAgentInput(InstalledNodeCliInput):
    """Input for the npm-installed HyCode DynamicAgent."""

    version: str = DEFAULT_HYCODE_DYNAMICAGENT_VERSION
    """``@tencent/hycode-dynamicagent`` package version."""

    extra_env: dict[str, str] | None = None
    """Extra environment variables for the CLI process."""

    extra_cli_args: list[str] = Field(default_factory=list)
    """Additional CLI arguments appended before the task prompt."""

    config_file_patches: list[ConfigFilePatch] = Field(default_factory=list)
    """JSON config files to patch before launch."""

    pre_start_scripts: list[PreStartScript] = Field(default_factory=list)
    """Scripts executed in order before launch."""

    conversion_log_level: str = 'INFO'
    """Log level for anthropic<->openai conversion dumps."""

    thinking_preset: str = 'hunyuan'
    """Thinking format preset for the anthropic<->openai converter."""

    no_thinking_request_args: dict[str, Any] | None = None
    """Override ``request_args`` for LLM calls when the CLI does NOT request thinking."""

    haiku_model: str | None = None
    """Model name to use when the CLI requests a haiku-class model."""

    tee_log_dir: str | None = None
    """Directory inside the sandbox to tee CLI stdout/stderr into.

    Files are named ``hycode_dynamicagent.<YYYYmmdd>.{stdout,stderr}.log`` and opened in
    append mode. Point it at ``working_dir`` to have the logs land in the archive. Empty
    or null disables teeing.
    """


class HyCodeDynamicAgentOutput(InstalledAgentOutput):
    """Output from HyCode DynamicAgent."""


class HyCodeDynamicAgentMultiturnInput(HyCodeDynamicAgentInput, WithInputProvider):
    """Input for interactive HyCode DynamicAgent stream-json multi-turn runs."""

    session_id: str | None = None
    """Stable Nexus-side session id. Generated when omitted."""

    max_turns: int = Field(default=20, ge=1)
    """Maximum number of user turns to send before stopping."""

    close_on_finish: bool = True
    """Close the runtime-side DynamicAgent process when this agent returns."""

    include_partial_messages: bool = False
    """Pass ``--include-partial-messages`` to the CLI."""

    include_hook_events: bool = False
    """Pass ``--include-hook-events`` to the CLI."""

    enable_user_input_hooks: bool = False
    """Bridge AskUserQuestion and ExitPlanMode PreToolUse hooks to ``get_user_input``."""

    record_trace_events: bool = True
    """Record stream events on the active trace span."""


class HyCodeDynamicAgentMultiturnOutput(HyCodeDynamicAgentOutput):
    """Output from a HyCode DynamicAgent interactive multi-turn run."""

    session_id: str
    turns: int
    events: list[dict[str, Any]] = Field(default_factory=list)
    ended_by_user: bool = False
    reached_max_turns: bool = False


@agent_registry.register('hycode_dynamicagent')
async def run_hycode_dynamicagent(input_data: HyCodeDynamicAgentInput) -> HyCodeDynamicAgentOutput:
    """Run ``@tencent/hycode-dynamicagent`` inside the runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)
    extra_env = dict(input_data.extra_env or {})

    logger.debug('Starting HyCode DynamicAgent for task: %s...', task[:100])
    from .runtime_setup import setup_and_run_hycode_dynamicagent

    result = await call_runtime_agent(
        setup_and_run_hycode_dynamicagent,
        no_thinking_request_args=input_data.no_thinking_request_args,
        haiku_model=input_data.haiku_model,
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        npm_registry=input_data.npm_registry,
        extra_env=extra_env,
        extra_cli_args=list(input_data.extra_cli_args),
        config_file_patches=[p.model_dump() for p in input_data.config_file_patches],
        pre_start_scripts=[s.model_dump() for s in input_data.pre_start_scripts],
        conversion_log_level=input_data.conversion_log_level,
        thinking_preset=input_data.thinking_preset,
        tee_log_dir=input_data.tee_log_dir,
    )

    if not result.get('success'):
        logger.warning('HyCode DynamicAgent failed, exit_code=%s', result.get('exit_code'))
        if result.get('stderr'):
            logger.warning('HyCode DynamicAgent stderr:\n%s', result.get('stderr'))
        if result.get('output'):
            logger.warning('HyCode DynamicAgent stdout:\n%s', result.get('output'))

    logger.info('HyCode DynamicAgent execution completed: success=%s', result.get('success'))
    return HyCodeDynamicAgentOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
    )


@agent_registry.register('hycode_dynamicagent_multiturn')
async def run_hycode_dynamicagent_multiturn(
    input_data: HyCodeDynamicAgentMultiturnInput,
) -> HyCodeDynamicAgentMultiturnOutput:
    """Run HyCode DynamicAgent as a stream-json multi-turn agent."""
    from .runtime_setup import close_hycode_dynamicagent_session, run_hycode_dynamicagent_session_turn

    result = await run_installed_multiturn(
        input_data,
        agent_key='hycode_dynamicagent',
        agent_label='HyCode DynamicAgent',
        session_namespace='hycode-dynamicagent-session',
        session_turn_target=run_hycode_dynamicagent_session_turn,
        close_session_target=close_hycode_dynamicagent_session,
        runtime_kwargs={
            'working_dir': input_data.working_dir,
            'timeout': input_data.timeout,
            'version': input_data.version,
            'npm_registry': input_data.npm_registry,
            'extra_env': dict(input_data.extra_env or {}),
            'extra_cli_args': list(input_data.extra_cli_args),
            'config_file_patches': [p.model_dump() for p in input_data.config_file_patches],
            'pre_start_scripts': [s.model_dump() for s in input_data.pre_start_scripts],
            'conversion_log_level': input_data.conversion_log_level,
            'thinking_preset': input_data.thinking_preset,
            'include_partial_messages': input_data.include_partial_messages,
            'include_hook_events': input_data.include_hook_events,
            'enable_user_input_hooks': input_data.enable_user_input_hooks,
            'record_trace_events': input_data.record_trace_events,
        },
    )
    return HyCodeDynamicAgentMultiturnOutput(**result)


__all__ = [
    'ConfigFilePatch',
    'HyCodeDynamicAgentInput',
    'HyCodeDynamicAgentMultiturnInput',
    'HyCodeDynamicAgentMultiturnOutput',
    'HyCodeDynamicAgentOutput',
    'PreStartScript',
    'run_hycode_dynamicagent',
    'run_hycode_dynamicagent_multiturn',
]
