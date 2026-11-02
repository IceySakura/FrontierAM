"""CodeBuddy Agent.

Runs the CodeBuddy CLI (@tencent-ai/codebuddy-code) inside a runtime sandbox
in headless mode.  The runtime-side setup installs the CLI via npm, starts an
OpenAI-compatible proxy, and then executes the CodeBuddy CLI with the task
prompt.  LLM calls flow through the host-side OpenAI client via Socket.IO.
"""

import logging

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from nexus.core import WithInputProvider, agent_registry, require_context
from nexus.extensions.agents.installed_common import (
    DEFAULT_PROXY_API_KEY,
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)
from nexus.extensions.agents.installed_common.multiturn import run_installed_multiturn

from .installer import DEFAULT_CODEBUDDY_VERSION
from .session_runtime import AcpCli
from .workbuddy_acp import WorkBuddyAcpOptions, local_package_sha256, upload_workbuddy_package


logger = logging.getLogger(__name__)


class PreStartScript(BaseModel):
    """A script to execute inside the runtime before CodeBuddy starts."""

    interpreter: Literal['bash', 'python'] = 'bash'
    code: str
    """Inline source code for the chosen interpreter."""

    timeout: float = 300.0
    cwd: str | None = None
    """Override cwd.  Defaults to ``working_dir``."""

    env: dict[str, str] | None = None
    """Extra env vars merged on top of the CodeBuddy launch env."""

    fail_on_error: bool = True
    """If True, non-zero exit aborts the agent run."""


class ConfigFilePatch(BaseModel):
    """Merge *content* into a JSON file at *path* before launching CodeBuddy.

    ``path`` may start with ``~`` to reference the runtime user's home.
    Non-existent files are created; existing JSON is deep-merged.
    """

    path: str
    content: dict[str, Any]


class CodeBuddyInput(InstalledNodeCliInput):
    """Input for CodeBuddy agent."""

    version: str = DEFAULT_CODEBUDDY_VERSION
    """CodeBuddy package version (``latest`` or e.g. ``2.86.0``)."""

    model_id: str = 'nexus-proxy-model'
    """Model ID written into models.json.  The proxy forwards all requests
    to the host-side OpenAI client regardless of the model name."""

    extra_env: dict[str, str] | None = None
    """Extra environment variables for the CodeBuddy process."""

    extra_cli_args: list[str] = Field(default_factory=list)
    """Additional CLI arguments appended before the task prompt."""

    config_file_patches: list[ConfigFilePatch] = Field(default_factory=list)
    """JSON config files to patch before launch (e.g. ``~/.codebuddy/settings.json``)."""

    pre_start_scripts: list[PreStartScript] = Field(default_factory=list)
    """Scripts executed in order before CodeBuddy starts."""

    prompt_suffix: str = ''
    """Appended after the task instruction. Does not replace ``user_prompt``."""


class CodeBuddyOutput(InstalledAgentOutput):
    """Output from CodeBuddy agent."""


class CodeBuddyMultiturnInput(CodeBuddyInput, WithInputProvider):
    """Input for a session-driven CodeBuddy run.

    Each turn reuses the same ``codebuddy --acp`` process, so context carries
    over between turns. The caller supplies follow-up turns through
    ``get_user_input``; returning ``None`` ends the run.
    """

    session_id: str | None = None
    """Stable Nexus-side session id. Generated when omitted."""

    acp_cli: AcpCli = 'codebuddy'
    """ACP server behind the session: ``codebuddy --acp`` (version 1) or WorkBuddy ``wb --acp`` (version 2).

    WorkBuddy ignores ``version`` and ``npm_registry`` (its package comes from ``workbuddy``) and
    accepts no ``mcp_servers`` or ``host_tools``. Anything but the default needs a runtime image
    whose session runtime knows ``acp_cli``; the key is only sent when it is not ``codebuddy``.
    """

    workbuddy: WorkBuddyAcpOptions | None = None
    """WorkBuddy package and session defaults; only valid with ``acp_cli='workbuddy'``, where None means defaults."""

    fork_source_session_id: str | None = None
    """Branch this session from that running runtime session with ``session/fork`` instead of ``session/new``.

    Only WorkBuddy forks; the new session shares the source's process and cwd. The id is the
    source's runtime session id, as normalized by the multi-turn loop.
    """

    llm_client_name: str = 'main'
    """Host-side OpenAI client used by the sandbox proxy."""

    skip_permissions: bool = True
    """When True, launch CodeBuddy with ``--dangerously-skip-permissions``.

    Set False if the host should see ACP ``session/request_permission``.
    extra_cli_args cannot override this knob.
    """

    max_turns: int = Field(default=20, ge=1)
    """Maximum number of user turns to send before stopping."""

    close_on_finish: bool = True
    """Close the runtime-side CodeBuddy process when this agent returns.

    Leave this off when the caller keeps driving the same session across
    separate agent invocations; the process then stays alive until an explicit
    :func:`~.session_runtime.close_codebuddy_session`.
    """

    permission_policy: Literal['allow', 'reject', 'defer'] = 'defer'
    """How the ACP peer's permission requests are answered.

    ``allow`` / ``reject`` answer immediately. ``defer`` relays the request to
    the host through :mod:`~.host_relay`; the host answers on the session channel.
    Default is defer so a skipped-off launch does not auto-allow;
    skip_permissions=True still short-circuits ACP.
    """

    deferred_permission_timeout: float | None = 300.0
    """Seconds a deferred permission may wait for the host; None waits until the turn ends."""

    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    """ACP session/new MCP server specifications."""

    relay_updates: bool = False
    """Push every ACP ``session/update`` to the host while a turn runs (``acp_session_update``).

    Needs a runtime image whose host relay knows ``relay_updates``; any other image rejects the key.
    Requires ``relay_turn_id``.
    """

    relay_turn_id: str | None = None
    """Id stamped on every update pushed during this run, so the host can tell its frames apart
    from a previous run's that are still arriving; ``relay_frames`` in the output counts them."""

    host_tools: dict[str, Any] | None = None
    """Host-owned MCP tools relayed into the session.

    ``{server_name, tools, request_timeout?, tool_timeouts?}``; ``tools`` are MCP
    ``Tool`` dicts. Calls arrive on the host as ``host_tool_request`` events.
    """

    client_questions: bool = False
    """Let CodeBuddy ask the user: declare form elicitation and ``AskUserQuestion`` at ACP initialize.

    Elicitations arrive on the host as ``acp_elicitation_request`` events; ``AskUserQuestion``
    arrives as a deferred permission the host answers with ``ext_requests``. Needs
    ``permission_policy='defer'`` and a runtime image whose host relay knows ``client_questions``;
    any other image rejects the key.
    """

    @model_validator(mode='after')
    def check_acp_cli_options(self) -> 'CodeBuddyMultiturnInput':
        """Reject WorkBuddy-only options on CodeBuddy and the MCP options WorkBuddy cannot serve."""
        if self.acp_cli == 'codebuddy':
            if self.workbuddy is not None:
                raise ValueError('workbuddy options need acp_cli="workbuddy"')
            if self.fork_source_session_id:
                raise ValueError('fork_source_session_id needs acp_cli="workbuddy": only WorkBuddy ACP can fork')
        elif self.host_tools is not None or self.mcp_servers:
            raise ValueError(
                'WorkBuddy ACP accepts no mcpServers, so host_tools and mcp_servers need acp_cli="codebuddy"'
            )
        return self


class CodeBuddyMultiturnOutput(CodeBuddyOutput):
    """Output from a session-driven CodeBuddy run."""

    session_id: str
    turns: int
    status: str | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    ended_by_user: bool = False
    reached_max_turns: bool = False
    relay_frames: int | None = None
    """Updates the relay pushed for ``relay_turn_id``, lost emits included; None without ``relay_updates``."""


# A sandboxed CodeBuddy CLI has no login state, and ACP mode refuses to serve
# requests without one ("Authentication required" on the first call). The CLI's
# real traffic goes to the runtime-side proxy, so any non-empty key satisfies
# the check. Callers can override it through `extra_env`.
_SANDBOX_AUTH_ENV = {'CODEBUDDY_API_KEY': DEFAULT_PROXY_API_KEY}
_HOST_RELAY_FACTORY = 'nexus.extensions.agents.codebuddy.host_relay::host_relay_extension'


def _host_relay_kwargs(input_data: CodeBuddyMultiturnInput) -> dict[str, Any]:
    """Select the public host relay when approvals or tools must reach the host."""
    extension_config: dict[str, Any] = {}
    if input_data.permission_policy == 'defer':
        extension_config['relay_permissions'] = True
    if input_data.client_questions:
        if input_data.permission_policy != 'defer':
            raise ValueError(
                'client_questions needs permission_policy="defer": AskUserQuestion is a relayed permission'
            )
        extension_config['client_questions'] = True
    turn_kwargs: dict[str, Any] = {}
    if input_data.relay_updates:
        if not input_data.relay_turn_id:
            raise ValueError('relay_updates needs relay_turn_id so the host can attribute pushed updates')
        extension_config['relay_updates'] = True
        turn_kwargs['relay_turn_id'] = input_data.relay_turn_id
    if input_data.host_tools is not None:
        extension_config['host_tools'] = input_data.host_tools
    return {
        'extension_factory': _HOST_RELAY_FACTORY if extension_config else None,
        'extension_config': extension_config or None,
        'deferred_permission_timeout': input_data.deferred_permission_timeout,
        **turn_kwargs,
    }


@agent_registry.register('codebuddy_multiturn')
async def run_codebuddy_multiturn(input_data: CodeBuddyMultiturnInput) -> CodeBuddyMultiturnOutput:
    """Run CodeBuddy as a multi-turn ACP session."""
    from .session_runtime import close_codebuddy_session, run_codebuddy_session_turn

    # Inherited from CodeBuddyInput, but this entry point builds each turn from the input
    # provider rather than from a single user_prompt, so there is nothing to append it to.
    # Fail loudly instead of accepting the key and ignoring it.
    if (input_data.prompt_suffix or '').strip():
        raise ValueError(
            'prompt_suffix is not supported by codebuddy_multiturn: turns come from the input '
            'provider, not from user_prompt. Put the instruction in the turns themselves.'
        )

    extra_env = {**_SANDBOX_AUTH_ENV, **(input_data.extra_env or {})}
    runtime_kwargs: dict[str, Any] = {
        'working_dir': input_data.working_dir,
        'timeout': input_data.timeout,
        'version': input_data.version,
        'npm_registry': input_data.npm_registry,
        'model_id': input_data.model_id,
        'extra_env': extra_env,
        'extra_cli_args': list(input_data.extra_cli_args),
        'config_file_patches': [patch.model_dump() for patch in input_data.config_file_patches],
        'pre_start_scripts': [script.model_dump() for script in input_data.pre_start_scripts],
        'permission_policy': input_data.permission_policy,
        'mcp_servers': [dict(server) for server in input_data.mcp_servers],
        'skip_permissions': input_data.skip_permissions,
        **_host_relay_kwargs(input_data),
    }
    # NOTE(wallyshi): An older runtime image rejects unknown session kwargs, so CodeBuddy sends none of these.
    if input_data.acp_cli != 'codebuddy':
        runtime_kwargs['acp_cli'] = input_data.acp_cli
        runtime_kwargs['workbuddy_options'] = await _workbuddy_runtime_options(
            input_data.workbuddy or WorkBuddyAcpOptions()
        )
    if input_data.fork_source_session_id:
        runtime_kwargs['fork_source_session_id'] = input_data.fork_source_session_id
    result = await run_installed_multiturn(
        input_data,
        agent_key='codebuddy',
        agent_label='CodeBuddy',
        session_namespace='codebuddy-acp-session',
        session_turn_target=run_codebuddy_session_turn,
        close_session_target=close_codebuddy_session,
        llm_channel=True,
        openai_client_name=input_data.llm_client_name,
        runtime_kwargs=runtime_kwargs,
    )
    return CodeBuddyMultiturnOutput(**result)


async def _workbuddy_runtime_options(options: WorkBuddyAcpOptions) -> dict[str, Any]:
    """Upload the Host package into this sandbox once, then name its parts and digest for the runtime."""
    if not options.package_path:
        return options.runtime_options(None)
    runtime = await require_context().get_runtime()
    await runtime.ws_connect()
    parts = await upload_workbuddy_package(runtime, options)
    return options.runtime_options(parts, package_sha256=await local_package_sha256(options.package_path))


@agent_registry.register('codebuddy')
async def run_codebuddy(input_data: CodeBuddyInput) -> CodeBuddyOutput:
    """Run CodeBuddy agent inside the runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)
    suffix = (input_data.prompt_suffix or '').strip()
    if suffix:
        task = f'{task.rstrip()}\n\n{suffix}'
    extra_env = dict(input_data.extra_env or {})

    logger.debug(f'Starting CodeBuddy agent for task: {task[:100]}...')

    from .runtime_setup import setup_and_run_codebuddy

    result = await call_runtime_agent(
        setup_and_run_codebuddy,
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        npm_registry=input_data.npm_registry,
        model_id=input_data.model_id,
        extra_env=extra_env,
        extra_cli_args=list(input_data.extra_cli_args),
        config_file_patches=[p.model_dump() for p in input_data.config_file_patches],
        pre_start_scripts=[s.model_dump() for s in input_data.pre_start_scripts],
    )

    logger.info(f'CodeBuddy execution completed: success={result.get("success")}')
    return CodeBuddyOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
    )


__all__ = [
    'CodeBuddyInput',
    'CodeBuddyMultiturnInput',
    'CodeBuddyMultiturnOutput',
    'CodeBuddyOutput',
    'ConfigFilePatch',
    'PreStartScript',
    'run_codebuddy',
    'run_codebuddy_multiturn',
]
