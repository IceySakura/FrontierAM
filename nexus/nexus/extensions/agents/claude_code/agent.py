"""Claude Code Agent.

Runs Claude Code inside a runtime sandbox. The runtime-side setup installs
the CLI (npm by default, with an optional direct-binary fallback), starts an
Anthropic-to-OpenAI proxy, optionally patches Claude's config, runs any
pre-start scripts, and then executes ``claude --print`` with the task prompt.
LLM calls flow through the host-side OpenAI client via Socket.IO.
"""

import json
import logging
import math
import shlex

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus.core import (
    CONTEXT_LENGTH_EXCEEDED_CODE,
    UserInputRequest,
    WithInputProvider,
    agent_registry,
    require_context,
)
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_native_bridge_runtime_agent,
    call_runtime_agent,
    coerce_user_prompt,
)
from nexus.extensions.agents.installed_common.execution import (
    execution_target_shared_path,
    get_agent_execution_target,
)
from nexus.extensions.agents.installed_common.multiturn import run_installed_multiturn
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.trace import get_current_span_id, update_span

from .installer import DEFAULT_CLAUDE_CODE_VERSION
from .interactive.hooks import DEFAULT_HOOK_COMMAND_TIMEOUT, DEFAULT_HOOK_WAIT_TIMEOUT
from .mcp_catalog import merge_mcp_servers, resolve_mcp_apps


logger = logging.getLogger(__name__)
_UNATTENDED_HOOKS_REMOTE_PATH = '/tmp/nexus_claude_unattended_hooks.py'
_STRICT_MCP_REMOTE_PATH = '/tmp/nexus_claude_strict_mcp.json'
# Stages this agent reports before Claude Code ran; same names as harbor_bridge's
# _INITIALIZATION_FAILURE_STAGES, which maps them to INITIALIZATION_ERROR.
_INITIALIZATION_FAILURE_STAGES = frozenset({'install', 'proxy', 'pre_start', 'execution_transport', 'runtime_protocol'})


def _budget_truncation_exit_status(result: dict[str, Any], failure_kind: str | None) -> str | None:
    """Return ``'truncation'`` when the host refused Claude Code's last call on token budget.

    Claude Code only sees the refusal as an API error. Only the last call to settle counts: a
    refused subagent call reaches the main loop as a tool error and the run goes on.
    """
    refused = (
        not result.get('success', False)
        and failure_kind != 'infrastructure'
        and result.get('exit_code', -1) > 0
        and result.get('last_llm_error_code') == CONTEXT_LENGTH_EXCEEDED_CODE
    )
    if not refused:
        return None
    logger.info('Claude Code: labelling exit as TRUNCATION (host refused the last call on token budget)')
    return 'truncation'


async def _no_more_user_input(_request: UserInputRequest | str | None) -> None:
    """Default continuation provider for single-turn PTY sessions."""
    return


def _has_explicit_user_input_provider(input_data: Any) -> bool:
    """Whether a caller replaced the single-turn compatibility provider."""
    return input_data.get_user_input is not _no_more_user_input


class PreStartScript(BaseModel):
    """A script to execute inside the runtime before Claude Code starts."""

    interpreter: Literal['bash', 'python'] = 'bash'
    code: str
    """Inline source code for the chosen interpreter."""

    timeout: float = 300.0
    cwd: str | None = None
    """Override cwd. Defaults to ``working_dir``."""

    env: dict[str, str] | None = None
    """Extra env vars merged on top of the Claude launch env."""

    fail_on_error: bool = True
    """If True, non-zero exit aborts the agent run."""


class NativeBridgePreStartScript(PreStartScript):
    """A pre-start script supported by native bridge runtime adapters."""

    interpreter: Literal['bash', 'python', 'powershell'] = 'bash'


class ConfigFilePatch(BaseModel):
    """Merge *content* into the JSON file at *path* before launching Claude.

    ``path`` may be absolute or start with ``~`` (e.g.
    ``~/.claude/settings.json``) to reference the runtime user's home.
    Non-existent files are created; existing JSON is deep-merged, preferring
    *content* for scalars. For env-var expansion use omegaconf's
    ``${oc.env:HOME}`` templating directly in the yaml.
    """

    path: str
    """Path inside the runtime. ``~`` is expanded against the runtime's HOME."""

    content: dict[str, Any]
    """Keys to deep-merge into the existing JSON (or create fresh)."""


class ClaudeCodeInput(InstalledNodeCliInput):
    """Input for Claude Code agent."""

    version: str | None = None
    """Optional version constraint; reuse installed Claude Code, otherwise install the default release."""

    binary_url: str | None = None
    """Optional direct download URL. When provided, bypasses npm install."""

    binary_checksum: str | None = None
    """Optional SHA256 checksum when using binary_url."""

    extra_env: dict[str, str] | None = None
    """Extra environment variables for the Claude Code process."""

    extra_cli_args: list[str] = Field(default_factory=list)
    """Additional CLI arguments appended before the task prompt."""

    config_file_patches: list[ConfigFilePatch] = Field(default_factory=list)
    """JSON config files to patch before launch.

    Use this for ``~/.claude/settings.json`` and any other JSON you want to
    seed (MCP servers, hooks, feature flags, etc.). Paths support ``~`` and
    ``$VAR`` expansion.
    """

    pre_start_scripts: list[PreStartScript] = Field(default_factory=list)
    """Scripts executed in order before Claude Code starts."""

    enable_unattended_interaction_hooks: bool = False
    """Upload and install Harbor's AskUserQuestion/ExitPlanMode hooks.

    The helper source is uploaded by the Host and appended after caller-owned
    pre-start scripts, so this does not require the Runtime image to contain
    Harbor Bridge modules.
    """

    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    """Shared Nexus MCP specs added to Claude's existing local configuration."""

    skills_dir: str | None = None
    """Optional task-provided Skills directory visible inside the Agent environment."""

    mcp_servers_mode: Literal['additive', 'strict_file'] = 'additive'
    """How shared ``mcp_servers`` reach Claude Code.

    ``additive`` preserves the historical Runtime-side registration path.
    ``strict_file`` converts on the Host, uploads one session-scoped config,
    and ignores all other MCP sources for a reproducible evaluation.
    """

    mcp_config_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Claude Code ``mcpServers`` map for servers that are *already running*.

    Unlike additive ``mcp_servers`` (registered through ``claude mcp add-json``),
    this follows ``mcp_config_mode`` and may strictly replace config already in
    the image. Values use Claude Code's native shape, e.g.
    ``{'canyon': {'type': 'http', 'url': 'http://localhost:9900/mcp'}}``.
    """

    mcp_config_mode: Literal['cli', 'file', 'both'] = 'cli'
    """How ``mcp_config_servers`` reaches Claude Code.

    ``cli`` passes ``--mcp-config <json> --strict-mcp-config`` (stateless, ignores
    any MCP config already in the image). ``file`` writes ``~/.claude/mcp.json``
    (survives cc-spawned subprocesses, but depends on HOME resolving as expected).
    ``both`` does each; ``cli`` still wins because of ``--strict-mcp-config``.
    """

    conversion_log_level: str = 'INFO'
    """Log level for anthropic<->openai conversion dumps."""

    thinking_preset: str = 'hunyuan'
    """Thinking format preset for the anthropic<->openai converter."""

    no_thinking_request_args: dict[str, Any] | None = None
    """Override ``request_args`` for LLM calls when Claude Code does NOT request thinking.
    ``no_thinking_request_args_mode`` controls replacement versus safe inheritance."""

    no_thinking_request_args_mode: Literal['replace', 'inherit'] = 'replace'
    """``replace`` preserves the historical isolated override; ``inherit`` keeps host routing fields."""

    haiku_model: str | None = None
    """Model name to use when Claude Code requests a haiku-class model.
    When set, overrides the host model for lightweight requests."""


class ClaudeCodeOutput(InstalledAgentOutput):
    """Output from Claude Code agent."""

    exit_status: str | None = None
    """Canonical Agent exit status for task adapters; None leaves the task-side default."""


class ClaudeCodeEpisodeOutput(ClaudeCodeOutput):
    """Structured episode result for explicitly selected native execution profiles."""

    main_trajectory: list[dict[str, Any]] = Field(default_factory=list)
    exit_status: str = 'unknown'
    iterations: int = 0
    final_answer: str | None = None
    cc_return_code: int | None = None
    termination_reason: str | None = None


class ClaudeCodeNativeBridgeInput(InstalledNodeCliInput):
    """Input for Claude Code running through a sandbox-local native bridge."""

    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='before')
    @classmethod
    def supply_profile_working_dir(cls, value):
        # Ordinary native calls keep requiring working_dir. A profile resolves
        # its own directory; this inherited field is inert only for that path.
        if isinstance(value, dict) and value.get('execution_profile') is not None:
            return {'working_dir': '/workspace', **value}
        return value

    execution_profile: str | None = None
    """Explicit benchmark execution policy; None retains ordinary native CC behavior."""
    execution_options: dict[str, Any] = Field(default_factory=dict)
    task_context: dict[str, Any] = Field(default_factory=dict)
    """Task-supplied metadata, kept separate from reusable execution policy."""
    max_turns: int | None = Field(default=None, gt=0)
    """Optional per-invocation limit; also supports existing --set agent_config.max_turns."""

    def build_execution_profile(self) -> Any:
        """Use the same explicit input overrides during validation and execution."""
        from .native_bridge.adapters.base import build_execution_profile

        if self.execution_profile is None:
            raise ValueError('execution_profile is required')
        overrides = {'timeout': self.timeout} if 'timeout' in self.model_fields_set else {}
        return build_execution_profile(
            self.execution_profile,
            {
                **self.execution_options,
                **({'max_turns': self.max_turns} if self.max_turns is not None else {}),
            },
            self.task_context,
            self.user_prompt,
            **({'input_overrides': overrides} if overrides else {}),
        )

    @model_validator(mode='after')
    def validate_execution_profile(self):
        if self.execution_profile is None and self.execution_options:
            raise ValueError('execution_options requires execution_profile')
        if self.execution_profile is not None:
            self.build_execution_profile()
            # Profile owns install/env/CLI/episode settings. Reject conflicting
            # explicit knobs instead of silently accepting options it won't use.
            defaults = type(self).model_fields
            for key in self.model_fields_set - {
                'user_prompt',
                'execution_profile',
                'execution_options',
                'task_context',
                'max_turns',
                'timeout',
            }:
                if key == 'working_dir' and self.working_dir == '/workspace':
                    continue
                field = defaults[key]
                default = field.get_default(call_default_factory=True)
                if getattr(self, key) != default:
                    raise ValueError(f'{key} must be configured through execution_options for an execution profile')
        return self

    version: str = DEFAULT_CLAUDE_CODE_VERSION
    """Claude Code package version (``latest`` or e.g. ``2.1.85``)."""

    binary_url: str | None = None
    """Optional download URL for a Claude Code executable or an npm native package tarball. Bypasses npm install."""

    binary_checksum: str | None = None
    """Optional SHA256 checksum of the file at binary_url."""

    extra_env: dict[str, str] | None = None
    """Extra environment variables for the Claude Code process."""

    extra_cli_args: list[str] = Field(default_factory=list)
    """Additional CLI arguments appended before the task prompt."""

    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    """MCP servers registered before the run using the standard native bridge.

    Registered via ``claude mcp add --transport stdio`` before launch. Uses the same
    ``name`` / ``command`` / ``args`` schema as ``ClaudeCodeInput``. Server processes
    inherit environment variables from ``extra_env``.
    """

    mcp_apps: list[str] = Field(default_factory=list)
    """Software names resolved to MCP server specs via :mod:`.mcp_catalog` and merged into
    ``mcp_servers`` (an explicit ``mcp_servers`` entry wins over a catalog entry of the same name).
    Lets a config register a server by naming the software, e.g. ``mcp_apps: [blender, kicad]``."""

    pre_start_scripts: list[NativeBridgePreStartScript] = Field(default_factory=list)
    """Scripts executed in order inside the sandbox before Claude Code starts."""

    no_thinking_request_args: dict[str, Any] | None = None
    """Override request args for LLM calls when Claude Code does not request thinking."""

    haiku_model: str | None = None
    """Model name to use when Claude Code requests a haiku-class model."""

    ensure_claude: bool = True
    """Install Claude Code if no executable can be auto-detected."""

    conversion_log_level: str = 'INFO'
    """Log level for anthropic<->openai conversion dumps."""

    thinking_preset: str = 'hunyuan'
    """Thinking format preset for the native bridge anthropic<->openai converter."""


class ClaudeCodeInteractiveConfigFilePatch(BaseModel):
    """Merge *content* into a JSON file before launching interactive Claude Code."""

    path: str
    content: dict[str, Any]


class ClaudeCodeInteractivePreStartScript(BaseModel):
    """A script to execute inside the runtime before interactive Claude Code starts."""

    interpreter: str = 'bash'
    code: str
    timeout: float = 300.0
    cwd: str | None = None
    env: dict[str, str] | None = None
    fail_on_error: bool = True


class ClaudeCodeInteractiveInput(InstalledNodeCliInput):
    """Input for a native Claude Code TUI driven through a PTY.

    Multi-turn capable: an interactive caller supplies ``get_user_input`` to
    drive several turns on the same PTY. Plain callers omit it (the default
    provider ends after the first turn), preserving single-turn behavior.
    """

    get_user_input: Callable[[UserInputRequest | str | None], Awaitable[str | None]] = Field(
        default=_no_more_user_input
    )
    """Interactive input provider. Defaults to ending after the first turn."""

    session_id: str | None = None
    """Stable Nexus-side session id. Generated when omitted."""

    max_turns: int = Field(default=20, ge=1)
    """Maximum number of user turns to send before stopping."""

    close_on_finish: bool = True
    """Close the runtime-side PTY session when this agent returns."""

    hook_input_timeout: float | None = Field(default=None, gt=0)
    """Optional provider timeout. ``None`` follows the effective Hook wait timeout."""

    hook_wait_timeout: float = Field(default=DEFAULT_HOOK_WAIT_TIMEOUT, gt=0)
    """Maximum seconds a Hook waits for an input response."""

    hook_command_timeout: float = Field(default=DEFAULT_HOOK_COMMAND_TIMEOUT, gt=0)
    """Claude Code watchdog for the complete Hook subprocess."""

    version: str = 'latest'
    """Claude Code package version, independent from the other Claude Code agents."""

    binary_url: str | None = None
    binary_checksum: str | None = None

    extra_env: dict[str, str] | None = None
    extra_cli_args: list[str] = Field(default_factory=list)
    config_file_patches: list[ClaudeCodeInteractiveConfigFilePatch] = Field(default_factory=list)
    pre_start_scripts: list[ClaudeCodeInteractivePreStartScript] = Field(default_factory=list)
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)

    conversion_log_level: str = 'INFO'
    thinking_preset: str = 'hunyuan'
    no_thinking_request_args: dict[str, Any] | None = None
    no_thinking_request_args_mode: Literal['replace', 'inherit'] = 'replace'
    haiku_model: str | None = None

    rows: int = 50
    cols: int = 200
    prompt_settle_seconds: float = 4.0
    log_frames: bool = False
    """Log every PTY output frame to the worker log."""

    snapshot_seconds: float | None = Field(default=None, gt=0)
    """Debug screen snapshot interval in seconds. Disabled by default."""

    @model_validator(mode='after')
    def _validate_hook_timeouts(self) -> 'ClaudeCodeInteractiveInput':
        values = {
            'hook_wait_timeout': self.hook_wait_timeout,
            'hook_command_timeout': self.hook_command_timeout,
        }
        if self.hook_input_timeout is not None:
            values['hook_input_timeout'] = self.hook_input_timeout
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite, got {value!r}')
        if self.hook_command_timeout <= self.hook_wait_timeout:
            raise ValueError('hook_command_timeout must be greater than hook_wait_timeout')
        if self.hook_input_timeout is not None and self.hook_input_timeout > self.hook_wait_timeout:
            raise ValueError('hook_input_timeout cannot be greater than hook_wait_timeout')
        return self


class ClaudeCodeInteractiveOutput(InstalledAgentOutput):
    """Output from the interactive Claude Code PTY agent."""

    session_id: str = ''
    turns: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)
    ended_by_user: bool = False
    reached_max_turns: bool = False
    status: str | None = None


class ClaudeCodeMultiturnInput(ClaudeCodeInput, WithInputProvider):
    """Input for interactive Claude Code stream-json multi-turn runs."""

    session_id: str | None = None
    """Stable Nexus-side session id. Generated when omitted."""

    execution_profile: str | None = None
    """Optional task-owned transport profile for runtimes without invoke."""
    execution_options: dict[str, Any] = Field(default_factory=dict)
    task_context: dict[str, Any] = Field(default_factory=dict)

    max_turns: int = Field(default=20, ge=1)
    """Maximum number of user turns to send before stopping."""

    close_on_finish: bool = True
    """Close the runtime-side Claude Code process when this agent returns."""

    include_partial_messages: bool = False
    """Pass ``--include-partial-messages`` to Claude Code."""

    include_hook_events: bool = False
    """Pass ``--include-hook-events`` to Claude Code."""

    enable_user_input_hooks: bool = False
    """Bridge AskUserQuestion and ExitPlanMode PreToolUse hooks to ``get_user_input``."""

    record_trace_events: bool = True
    """Record Claude Code stream events on the active trace span."""


class ClaudeCodeMultiturnOutput(ClaudeCodeOutput):
    """Output from a Claude Code interactive multi-turn run."""

    session_id: str
    turns: int
    events: list[dict[str, Any]] = Field(default_factory=list)
    ended_by_user: bool = False
    reached_max_turns: bool = False


def _apply_batch_features(
    input_data: ClaudeCodeInput,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Wire optional MCP config into the three list-valued launch settings."""
    from .batch_features import apply_batch_features

    return apply_batch_features(
        extra_cli_args=list(input_data.extra_cli_args),
        config_file_patches=[patch.model_dump() for patch in input_data.config_file_patches],
        pre_start_scripts=[script.model_dump() for script in input_data.pre_start_scripts],
        mcp_config_servers=({} if input_data.mcp_servers_mode == 'strict_file' else input_data.mcp_config_servers),
        mcp_config_mode=input_data.mcp_config_mode,
    )


async def _upload_runtime_text(path: str, content: str, operation: str, *, mode: int) -> None:
    """Upload text and set its mode on the Controller before service access."""
    from nexus.core import require_context
    from nexus.extensions.harbor_bridge.upload import raise_for_upload_failures
    from nexus.runtime.executors.file import FileItem

    runtime = await require_context().get_runtime()
    response = await runtime.upload_file(
        files=[FileItem(path=path, content=content, encoding='utf8')],
        overwrite=True,
    )
    raise_for_upload_failures(response, operation)
    chmod_result = await runtime.run_command(
        command=shlex.join(['chmod', f'{mode:04o}', '--', path]),
        timeout=30,
    )
    if chmod_result.return_code != 0:
        details = chmod_result.stderr or chmod_result.stdout or 'no output'
        raise RuntimeError(f'{operation} permission setup failed: {details}')


async def _prepare_unattended_interaction_hooks(
    pre_start_scripts: list[dict[str, Any]],
    *,
    remote_path: str = _UNATTENDED_HOOKS_REMOTE_PATH,
    file_mode: int = 0o600,
) -> None:
    """Upload Harbor's stdlib-only hook installer and append its execution."""
    from nexus.extensions.harbor_bridge import claude_unattended_hooks

    source_path = Path(claude_unattended_hooks.__file__)
    await _upload_runtime_text(
        remote_path,
        source_path.read_text(encoding='utf-8'),
        'Claude unattended hooks installer',
        mode=file_mode,
    )
    pre_start_scripts.append({
        'interpreter': 'bash',
        'code': (
            'py="$(command -v python3 || command -v python || true)"; '
            'if [ -z "$py" ]; then '
            'for candidate in /nix/runtime/nexus/.venv-*/bin/python; do '
            'if [ -x "$candidate" ]; then py="$candidate"; break; fi; '
            'done; '
            'fi; '
            'if [ -z "$py" ]; then echo "No Python interpreter available for Claude hooks" >&2; exit 127; fi; '
            f'exec "$py" {shlex.quote(remote_path)}'
        ),
        'timeout': 30.0,
        'cwd': None,
        'env': None,
        'fail_on_error': True,
    })


async def _prepare_strict_mcp_file(
    input_data: ClaudeCodeInput,
    extra_cli_args: list[str],
    *,
    remote_path: str = _STRICT_MCP_REMOTE_PATH,
    file_mode: int = 0o600,
) -> None:
    """Convert MCP specs on the Host and upload one exclusive Claude config."""
    from .mcp import mcp_config_servers_from_specs

    if any(
        arg == '--strict-mcp-config' or arg == '--mcp-config' or arg.startswith('--mcp-config=')
        for arg in extra_cli_args
    ):
        raise ValueError('mcp_servers_mode=strict_file cannot be combined with explicit MCP CLI arguments')

    task_servers = mcp_config_servers_from_specs(list(input_data.mcp_servers))
    duplicate_names = sorted(set(task_servers) & set(input_data.mcp_config_servers))
    if duplicate_names:
        raise ValueError(f'Duplicate strict MCP server names across inputs: {duplicate_names}')
    native_servers = {**task_servers, **input_data.mcp_config_servers}
    config_json = json.dumps({'mcpServers': native_servers}, ensure_ascii=False, separators=(',', ':'))
    await _upload_runtime_text(remote_path, config_json, 'Claude strict MCP config', mode=file_mode)
    extra_cli_args.extend(['--mcp-config', remote_path, '--strict-mcp-config'])


@agent_registry.register(
    'claude_code',
    meta={
        'gateway': {'scaffold_policy': 'claude_code'},
        'supports_execution_target': True,
        'supports_execution_user': True,
    },
)
async def run_claude_code(input_data: ClaudeCodeInput) -> ClaudeCodeOutput:
    """Run Claude Code agent inside the runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)

    extra_env = dict(input_data.extra_env or {})

    logger.debug(f'Starting Claude Code agent for task: {task[:100]}...')
    logger.debug('Executing Claude Code in runtime...')
    from .runtime_setup import setup_and_run_claude_code

    execution_target = get_agent_execution_target()
    shared_file_mode = 0o600
    uploaded_files: list[str] = []
    extra_cli_args, config_file_patches, pre_start_scripts = _apply_batch_features(input_data)
    runtime_mcp_servers = list(input_data.mcp_servers)
    if input_data.mcp_servers_mode == 'strict_file':
        await _prepare_strict_mcp_file(
            input_data,
            extra_cli_args,
            remote_path=str(
                execution_target_shared_path(
                    execution_target,
                    'strict-mcp.json',
                    default=_STRICT_MCP_REMOTE_PATH,
                )
            ),
            file_mode=shared_file_mode,
        )
        runtime_mcp_servers = []
        uploaded_files.append(extra_cli_args[extra_cli_args.index('--mcp-config') + 1])
    if input_data.enable_unattended_interaction_hooks:
        await _prepare_unattended_interaction_hooks(
            pre_start_scripts,
            remote_path=str(
                execution_target_shared_path(
                    execution_target,
                    'claude-unattended-hooks.py',
                    default=_UNATTENDED_HOOKS_REMOTE_PATH,
                )
            ),
            file_mode=shared_file_mode,
        )

        uploaded_files.append(
            str(
                execution_target_shared_path(
                    execution_target,
                    'claude-unattended-hooks.py',
                    default=_UNATTENDED_HOOKS_REMOTE_PATH,
                )
            )
        )

    runtime_kwargs: dict[str, Any] = {
        'task': task,
        'working_dir': input_data.working_dir,
        'timeout': input_data.timeout,
        'version': input_data.version,
        'npm_registry': input_data.npm_registry,
        'binary_url': input_data.binary_url,
        'binary_checksum': input_data.binary_checksum,
        'extra_env': extra_env,
        'extra_cli_args': extra_cli_args,
        'config_file_patches': config_file_patches,
        'pre_start_scripts': pre_start_scripts,
        'mcp_servers': runtime_mcp_servers,
        'conversion_log_level': input_data.conversion_log_level,
        'thinking_preset': input_data.thinking_preset,
    }
    if execution_target is not None and uploaded_files:
        runtime_kwargs['uploaded_files'] = uploaded_files
    if input_data.skills_dir:
        runtime_kwargs['skills_dir'] = input_data.skills_dir
    agent_span_id = get_current_span_id()
    try:
        result = await call_runtime_agent(
            setup_and_run_claude_code,
            no_thinking_request_args=input_data.no_thinking_request_args,
            no_thinking_request_args_mode=input_data.no_thinking_request_args_mode,
            haiku_model=input_data.haiku_model,
            **runtime_kwargs,
        )
    except RuntimeUnhealthyError as e:
        # The provider watchdog marked the sandbox unusable (stopped / OOMKilled / SandboxExit).
        logger.exception('Claude Code runtime became unhealthy; classifying as ENV_ERROR')
        if agent_span_id:
            update_span(agent_span_id, attributes={'hytuner.termination_reason': 'ENV_ERROR'})
        return ClaudeCodeOutput(
            success=False,
            output='',
            stderr=f'Runtime unhealthy: {e}',
            exit_code=-1,
            execution_time=0.0,
            failure_kind='infrastructure',
            failure_stage='runtime_unhealthy',
        )

    logger.info(f'Claude Code execution completed: success={result.get("success")}')
    failure_kind = result.get('failure_kind')
    failure_stage = result.get('failure_stage')
    stderr = result.get('stderr', '')
    if not result.get('success', False) and failure_kind is None:
        # Old Runtime images returned only success/exit_code for setup failures.
        # Fail closed instead of letting Harbor score a pristine workspace as a
        # model attempt. Current Runtime code labels post-start process failures
        # explicitly as ``agent``.
        failure_kind = 'infrastructure'
        failure_stage = 'runtime_protocol'
        protocol_error = 'Runtime returned an unclassified Claude failure; update the Nexus Runtime image'
        stderr = f'{stderr}\n{protocol_error}'.strip()
    exit_status = _budget_truncation_exit_status(result, failure_kind)
    termination_reason = None
    if failure_kind == 'infrastructure':
        # Without an explicit reason the run tracker would train this sandbox as a clean finish.
        termination_reason = (
            'INITIALIZATION_ERROR' if failure_stage in _INITIALIZATION_FAILURE_STAGES else 'UNHANDLED_ERROR'
        )
    elif exit_status is not None:
        termination_reason = 'TRUNCATION'
    if termination_reason is not None and agent_span_id:
        update_span(agent_span_id, attributes={'hytuner.termination_reason': termination_reason})
    return ClaudeCodeOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=stderr,
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
        failure_kind=failure_kind,
        failure_stage=failure_stage,
        exit_status=exit_status,
    )


@agent_registry.register('claude_code_native_bridge', meta={'gateway': {'scaffold_policy': 'claude_code'}})
async def run_claude_code_native_bridge(input_data: ClaudeCodeNativeBridgeInput) -> ClaudeCodeOutput:
    """Run Claude Code in a native sandbox through the native bridge."""
    if input_data.execution_profile is not None:
        profile = input_data.build_execution_profile()
        ctx = require_context()
        result = await profile.run(ctx.get_openai_client())
        return ClaudeCodeEpisodeOutput(**{
            key: value for key, value in result.items() if key in ClaudeCodeEpisodeOutput.model_fields
        })
    task = coerce_user_prompt(input_data.user_prompt)
    extra_env = dict(input_data.extra_env or {})

    logger.debug(f'Starting Claude Code native bridge agent for task: {task[:100]}...')
    from .native_bridge.agent_runtime import setup_and_run_native_bridge_claude_code

    # Names in ``mcp_apps`` expand to catalog specs and merge with any explicit ``mcp_servers``
    # (explicit wins by name). resolve_mcp_apps raises on an unknown name, failing fast here rather
    # than mid-rollout.
    mcp_servers = merge_mcp_servers(list(input_data.mcp_servers), resolve_mcp_apps(input_data.mcp_apps))

    cli_args = list(input_data.extra_cli_args)
    if input_data.max_turns is not None:
        if any(arg == '--max-turns' or arg.startswith('--max-turns=') for arg in cli_args):
            raise ValueError('Specify max_turns or --max-turns, not both')
        cli_args.extend(['--max-turns', str(input_data.max_turns)])
    result = await call_native_bridge_runtime_agent(
        setup_and_run_native_bridge_claude_code,
        no_thinking_request_args=input_data.no_thinking_request_args,
        haiku_model=input_data.haiku_model,
        preserve_thinking=True,
        conversion_log_level=input_data.conversion_log_level,
        thinking_config=input_data.thinking_preset,
        api_protocol='anthropic',
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        npm_registry=input_data.npm_registry,
        ensure_claude=input_data.ensure_claude,
        binary_url=input_data.binary_url,
        binary_checksum=input_data.binary_checksum,
        extra_env=extra_env,
        extra_cli_args=cli_args,
        mcp_servers=mcp_servers,
        pre_start_scripts=[s.model_dump() for s in input_data.pre_start_scripts],
    )

    logger.info(f'Claude Code native bridge execution completed: success={result.get("success")}')
    exit_status = _budget_truncation_exit_status(result, result.get('failure_kind'))
    span_id = get_current_span_id()
    if exit_status is not None and span_id:
        update_span(span_id, attributes={'hytuner.termination_reason': 'TRUNCATION'})
    return ClaudeCodeOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
        failure_kind=result.get('failure_kind'),
        failure_stage=result.get('failure_stage'),
        exit_status=exit_status,
    )


@agent_registry.register('claude_code_multiturn', meta={'gateway': {'scaffold_policy': 'claude_code'}})
async def run_claude_code_multiturn(input_data: ClaudeCodeMultiturnInput) -> ClaudeCodeMultiturnOutput:
    """Run Claude Code as a stream-json multi-turn agent."""
    if input_data.execution_profile is not None:
        from .native_bridge.adapters.base import multiturn_execution_profile_registry

        profile = multiturn_execution_profile_registry.get(input_data.execution_profile)
        result = await profile(input_data).run(require_context().openai_client)
        return ClaudeCodeMultiturnOutput(**result)
    from .session_runtime import close_claude_code_session, run_claude_code_session_turn

    extra_env = dict(input_data.extra_env or {})
    extra_cli_args, config_file_patches, pre_start_scripts = _apply_batch_features(input_data)
    result = await run_installed_multiturn(
        input_data,
        agent_key='claude_code',
        agent_label='Claude Code',
        session_namespace='claude-code-session',
        session_turn_target=run_claude_code_session_turn,
        close_session_target=close_claude_code_session,
        runtime_kwargs={
            'working_dir': input_data.working_dir,
            'timeout': input_data.timeout,
            'version': input_data.version,
            'npm_registry': input_data.npm_registry,
            'binary_url': input_data.binary_url,
            'binary_checksum': input_data.binary_checksum,
            'extra_env': extra_env,
            'extra_cli_args': extra_cli_args,
            'config_file_patches': config_file_patches,
            'pre_start_scripts': pre_start_scripts,
            'mcp_servers': list(input_data.mcp_servers),
            'conversion_log_level': input_data.conversion_log_level,
            'thinking_preset': input_data.thinking_preset,
            'include_partial_messages': input_data.include_partial_messages,
            'include_hook_events': input_data.include_hook_events,
            'enable_user_input_hooks': input_data.enable_user_input_hooks,
            'record_trace_events': input_data.record_trace_events,
        },
    )
    return ClaudeCodeMultiturnOutput(**result)


@agent_registry.register('claude_code_interactive', meta={'gateway': {'scaffold_policy': 'claude_code'}})
async def run_claude_code_interactive(input_data: ClaudeCodeInteractiveInput) -> ClaudeCodeInteractiveOutput:
    """Run the native Claude Code TUI behind a PTY, one or many turns on one session."""
    _runtime_module = 'nexus.extensions.agents.claude_code.interactive.runtime'

    extra_env = dict(input_data.extra_env or {})
    enable_input_bridge = _has_explicit_user_input_provider(input_data)
    logger.debug('Starting interactive Claude Code agent (multi-turn capable)')

    result = await run_installed_multiturn(
        input_data,
        agent_key='claude_code_interactive',
        agent_label='Claude Code (PTY)',
        session_namespace='claude-code-interactive-session',
        session_turn_target=f'{_runtime_module}::run_claude_code_interactive_session_turn',
        close_session_target=f'{_runtime_module}::close_claude_code_interactive_session',
        respond_session_target=f'{_runtime_module}::respond_and_wait_claude_code_interactive_session',
        runtime_kwargs={
            'working_dir': input_data.working_dir,
            'timeout': input_data.timeout,
            'version': input_data.version,
            'npm_registry': input_data.npm_registry,
            'binary_url': input_data.binary_url,
            'binary_checksum': input_data.binary_checksum,
            'mcp_servers': list(input_data.mcp_servers),
            'extra_env': extra_env,
            'extra_cli_args': list(input_data.extra_cli_args),
            'config_file_patches': [patch.model_dump() for patch in input_data.config_file_patches],
            'pre_start_scripts': [script.model_dump() for script in input_data.pre_start_scripts],
            'conversion_log_level': input_data.conversion_log_level,
            'thinking_preset': input_data.thinking_preset,
            'enable_input_bridge': enable_input_bridge,
            'rows': input_data.rows,
            'cols': input_data.cols,
            'prompt_settle_seconds': input_data.prompt_settle_seconds,
            'log_frames': input_data.log_frames,
            'snapshot_seconds': input_data.snapshot_seconds,
            'hook_wait_timeout': input_data.hook_wait_timeout,
            'hook_command_timeout': input_data.hook_command_timeout,
        },
    )

    logger.info('Interactive Claude Code execution completed: success=%s', result.get('success'))
    return ClaudeCodeInteractiveOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
        session_id=result.get('session_id', ''),
        turns=result.get('turns', 0),
        events=result.get('events', []),
        ended_by_user=result.get('ended_by_user', False),
        reached_max_turns=result.get('reached_max_turns', False),
        status=result.get('status'),
    )


__all__ = [
    'ClaudeCodeInput',
    'ClaudeCodeNativeBridgeInput',
    'ClaudeCodeOutput',
    'ClaudeCodeInteractiveConfigFilePatch',
    'ClaudeCodeInteractiveInput',
    'ClaudeCodeInteractiveOutput',
    'ClaudeCodeInteractivePreStartScript',
    'ClaudeCodeMultiturnInput',
    'ClaudeCodeMultiturnOutput',
    'ConfigFilePatch',
    'PreStartScript',
    'NativeBridgePreStartScript',
    'run_claude_code',
    'run_claude_code_native_bridge',
    'run_claude_code_interactive',
    'run_claude_code_multiturn',
]
