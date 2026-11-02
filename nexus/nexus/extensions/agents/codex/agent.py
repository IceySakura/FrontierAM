"""Codex agent."""

import json
import logging
import time

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nexus.core import CONTEXT_LENGTH_EXCEEDED_CODE, UserInputRequest, agent_registry
from nexus.core.agent import WithFinalAnswer, WithInputProvider
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_native_bridge_runtime_agent,
    call_runtime_agent,
    coerce_user_prompt,
)
from nexus.extensions.agents.installed_common.multiturn import run_installed_multiturn
from nexus.trace import RemoteTraceContext, get_current_span_id, update_span

from .events import CodexEventStream
from .installer import DEFAULT_CODEX_VERSION
from .interactive import INTERACTIVE_DEFAULT_CODEX_VERSION


logger = logging.getLogger(__name__)


async def _no_more_user_input(_request: UserInputRequest | str | None) -> None:
    return


# Reasons that a host-side budget refusal is allowed to overwrite with TRUNCATION. Listed
# positively so a reason added to classify_termination later is NOT relabelled by default:
# a new reason is usually a more specific diagnosis and should outrank a truncation label.
# The two deliberate omissions:
#   ENV_DONE     -- codex recovered from the refusal and still finished cleanly; the run is
#                   a success and must not be recast as truncated.
#   TASK_TIMEOUT -- the wall clock ran out, a harder fact that keeps precedence.
# Relabelling UNHANDLED_ERROR also moves the sample out of the trainer's maskout set (which
# holds UNHANDLED_ERROR but not TRUNCATION). That is intended: it makes the coarse
# exit-code path agree with the --json path, which already leaves these runs trainable.
_TRUNCATION_OVERRIDABLE_REASONS = frozenset({
    'CODEX_TURN_FAILED',
    'CODEX_ERROR',
    'UNHANDLED_ERROR',
})

_NATIVE_RESPONSES_COMPACTION_MIN_RELEASE = (0, 136)

CodexExitStatus = Literal['completed', 'timeout', 'truncation', 'format_error', 'error']


def _uses_native_responses_compaction(*, version: str | None, use_azure_provider: bool) -> bool:
    """Return whether this Codex configuration can call ``/responses/compact``."""
    if not use_azure_provider:
        return False
    if version is None:
        version = DEFAULT_CODEX_VERSION
    if version == 'latest':
        return True
    try:
        return Version(version).release >= _NATIVE_RESPONSES_COMPACTION_MIN_RELEASE
    except InvalidVersion:
        # A custom/latest build using the Azure provider may have the new behavior.
        # Fail closed at the client capability boundary instead of starting a run
        # that can only fail after it reaches the compaction threshold.
        return True


def _validate_native_compaction_client(client: Any, *, version: str | None, use_azure_provider: bool) -> None:
    """Reject provider-native compaction before acquiring a runtime when unsupported."""
    if not _uses_native_responses_compaction(version=version, use_azure_provider=use_azure_provider):
        return
    supports_native = bool(getattr(client, 'supports_native_responses_compaction', False))
    compact_response = getattr(client, 'compact_response', None)
    if supports_native and callable(compact_response):
        return
    raise ValueError(
        f'Codex {version} with use_azure_provider=true requires an OpenAI client that supports provider-native '
        f'Responses compaction; {type(client).__name__} does not. HyTuner training must set '
        'use_azure_provider=false so Codex uses trainable local summary compaction.'
    )


class CodexInput(InstalledNodeCliInput):
    """Input for the Codex agent."""

    mode: Literal['exec', 'exec_json', 'app_server'] = 'exec_json'
    """Codex execution mode.

    ``exec`` runs plain ``codex exec``; ``exec_json`` enables its curated JSONL event stream;
    ``app_server`` drives one thread over stdio JSON-RPC, optionally continuing turns with ``goal_mode``.
    """

    goal_mode: bool = False
    """Run a native goal across turns; requires app_server mode and a CLI with goal API support."""

    goal_token_budget: int | None = Field(default=None, gt=0)
    """Optional native goal token budget, only valid with goal_mode."""

    version: str | None = DEFAULT_CODEX_VERSION
    """Codex CLI version; omitted or null uses 0.114.0. Reuse only matching installed versions."""

    binary_url: str | None = None
    """Optional direct binary URL (including signed COS URLs), bypassing npm installation."""

    binary_checksum: str | None = None
    """Optional SHA256 checksum of the binary downloaded from binary_url."""

    extra_env: dict[str, str] | None = None
    """Additional Codex environment variables; configure MCP env in mcp_servers."""

    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    """Stdio MCP servers, using the same fields as codex_native_bridge."""

    model: str | None = None
    """Optional Codex model override written into config.toml."""

    model_catalog_json: dict[str, Any] | None = None
    """Optional inline Codex model catalog materialized inside the runtime."""

    model_catalog_overrides: dict[str, Any] | None = None
    """Optional catalog-entry overrides applied to the selected model at runtime."""

    codex_home_name: str | None = None
    """Optional stable CODEX_HOME directory name for isolated deterministic parity runs.

    Leave unset to preserve the default per-run UUID. Reusing a stable name in one long-lived
    runtime can overwrite files from an earlier invocation.
    """

    normalize_exec_command_metadata: bool = False
    """Normalize Codex exec-command chunk ID and wall time for deterministic parity tests."""

    model_context_window: int | None = None
    """Optional context-window override for Codex 0.147 and newer."""

    reasoning_effort: str | None = None
    """Optional reasoning effort written as ``model_reasoning_effort``."""

    extra_cli_args: list[str] = Field(default_factory=list)
    """Additional Codex CLI flags appended before the task prompt."""

    images: list[str] = Field(default_factory=list)
    """Runtime-local image paths attached to the initial prompt via ``codex exec -i``."""

    developer_instructions: str | None = None
    """Developer instructions prepended to the task prompt."""

    extra_config_toml: str | None = None
    """Additional root-level Codex configuration written before the Nexus provider block."""

    model_instructions_file: str | None = None
    """Runtime-local file whose content becomes the model's base instructions.

    A dedicated field rather than an ``extra_config_toml`` prefix: the
    task-side caller does not see ``agent_config``, so it could not merge into
    a configured ``extra_config_toml`` (osworld's ``model_reasoning_summary``,
    swe_cosmos's ``[features]``), and appending a root key after an existing
    table header would scope it into that table. Rendered as a root key that
    coexists with ``extra_config_toml``; a same-named key there is replaced.
    """

    experimental_use_freeform_apply_patch: bool | None = None
    """Override Codex's legacy freeform_apply_patch setting, or infer it from the CLI version."""

    use_azure_provider: bool = True
    """Whether to write provider name="azure" in config.toml.

    True (default) for models that use encrypted_content reasoning (e.g. GPT-5.5): keeps store=true
    and original reasoning item ids so reasoning survives across turns. Set False for plaintext
    reasoning models (e.g. HY3) to keep store=false and route auto-compaction to local
    summarization instead of native Responses compaction. HyTuner training also requires False
    because provider-native opaque compaction has no trainable token ids/logprobs. Does not affect
    tool-call parsing.

    Polarist/trpc GPT evals must set False: the gateway is stateless (``rs_ not found`` on
    id replay). Generators in docs/examples already pass ``use_azure_provider=false``.
    """

    wrap_plaintext_reasoning_content: bool = True
    """Whether the Responses converter wraps ``reasoning_content`` into a minted reasoning item.

    True (default, origin/main) keeps HY3/GLM plaintext CoT replayable. GPT tasks set False so a
    plaintext summary is not given a fake ``rs_*``. Native ``reasoning_items`` are always kept.
    """

    supports_tool_references: bool | None = None
    """Enable inline tool references and deferred schemas when the endpoint supports both.

    None uses the converter's default, which disables the extension. False uses standard Chat
    schemas and text search results; True enables structured search results and function.defer_loading.
    """

    tool_search_output_format: Literal['tool_reference', 'tool_search_output'] = 'tool_reference'
    """Select the deferred schema format understood by the model's chat template.

    Use tool_reference for HY4 text or tool_search_output for HY4 serving/VLM. Both require
    supports_tool_references=True; False keeps the standard text search result behavior.
    """

    auto_compact_token_limit: int | None = 272000
    """Token threshold written as ``model_auto_compact_token_limit`` in config.toml.

    Codex resolves the effective trigger as ``min(this, context_window * 90%)``, so the 272000
    default is at/above gpt-5.4's 272000 context window and just inherits the default 90% clamp
    (~244800) -- a safe no-op that documents the intent. To actually defer compaction, raise this
    AND ``model_context_window``. Set None to omit the line entirely.
    """

    auto_compact_token_limit_scope: Literal['total', 'body_after_prefix'] | None = None
    """Optional token-count scope used by Codex 0.147 auto-compaction."""

    compact_prompt: str | None = None
    """Optional custom local-compaction prompt written into config.toml."""

    timeline_collect_tool_args: bool = False
    """Whether to attach tool arguments (command / patch changes) to agent_timeline segments.

    Only consulted by event-producing modes. Mirrors swe_agent's ``timeline_collect_tool_args`` gating
    so tool args are opt-in (they can be large / sensitive).
    """

    logs_dir: str | None = None
    """Optional Runtime-side directory for merged CLI output and native sessions (Linux).

    Sessions are always persisted; this only controls whether they get exported. Files stay out of
    CodexOutput. Not supported by codex_native_bridge.
    """

    retain_json_events_stdout: bool = False
    """Whether to retain the raw ``codex exec --json`` JSONL in ``CodexOutput``.

    Off by default because the event stream can be large. Enable only for wire-level audits that
    intentionally persist the raw stream in batch output.
    """

    @field_validator('version')
    @classmethod
    def _default_version(cls, version: str | None) -> str:
        return DEFAULT_CODEX_VERSION if version is None else version


class CodexOutput(InstalledAgentOutput, WithFinalAnswer):
    """Output from the Codex agent."""

    event_stream: CodexEventStream | None = None
    """Ordered protocol events for app_server runs."""

    exit_status: CodexExitStatus
    """Standard completion status consumed by task-level result classifiers."""

    termination_reason: str | None = None
    """Structured completion status propagated to tasks and reward evaluators."""

    json_events_stdout: str | None = None
    """Raw ``codex exec --json`` JSONL when explicitly retained; otherwise ``None``."""


def _relabel_clean_exit_on_last_call(termination_reason: str, result: dict[str, Any], label: str) -> str:
    """Relabel a clean exit whose last model call shows Codex never reached a real final answer.

    Codex ends its turn on any reply without tool calls and exits 0, so ENV_DONE also covers
    two replies that are not answers:

    * ``finish_reason='length'``: the host clipped ``max_tokens`` to a small positive remaining
      budget instead of refusing the call, and the reply was cut mid-answer -> TRUNCATION.
    * an unparsed tool call: the model began a tool call the parser rejected, so the raw markup
      became the reply -> FORMAT_ERROR.

    A length cut wins because it also leaves any tool call it interrupted unparsed.
    """
    if termination_reason != 'ENV_DONE':
        return termination_reason
    if result.get('last_llm_finish_reason') == 'length':
        logger.info('%s: re-labelling ENV_DONE as TRUNCATION (last model call stopped at max_tokens)', label)
        return 'TRUNCATION'
    if result.get('last_llm_unparsed_tool_call'):
        logger.info('%s: re-labelling ENV_DONE as FORMAT_ERROR (last model call left a tool call unparsed)', label)
        return 'FORMAT_ERROR'
    return termination_reason


def _standardize_exit_status(termination_reason: str) -> CodexExitStatus:
    """Map detailed Codex telemetry reasons to the common Agent output contract."""
    if termination_reason == 'ENV_DONE':
        return 'completed'
    if termination_reason == 'TASK_TIMEOUT':
        return 'timeout'
    if termination_reason in {'TRUNCATION', 'CODEX_GOAL_BUDGET_LIMITED'}:
        return 'truncation'
    # Harbor maps a bare 'error' exit status to UNHANDLED_ERROR and never reads termination_reason.
    if termination_reason == 'FORMAT_ERROR':
        return 'format_error'
    return 'error'


class CodexEpisodeOutput(CodexOutput):
    """OSWorld completion vocabulary without changing ordinary CodexOutput semantics."""

    exit_status: str
    main_trajectory: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    codex_return_code: int = -1


class CodexNativeBridgePreStartScript(BaseModel):
    """A script to execute in the native sandbox before Codex starts."""

    interpreter: Literal['bash', 'python', 'powershell'] = 'bash'
    code: str
    timeout: float = 300.0
    cwd: str | None = None
    env: dict[str, str] | None = None
    fail_on_error: bool = True


class CodexNativeBridgeInput(CodexInput):
    """Input for Codex running through a sandbox-local native bridge."""

    emit_json_trace: bool = True
    """Whether to run codex exec --json through the native bridge."""

    model_config = ConfigDict(extra='forbid')
    execution_profile: str | None = None
    execution_options: dict[str, Any] = Field(default_factory=dict)
    task_context: dict[str, Any] = Field(default_factory=dict)
    get_user_input: Callable[[UserInputRequest | str | None], Awaitable[str | None]] | None = Field(
        default=None, exclude=True
    )
    """Optional user-input transport for an explicitly interactive execution profile."""

    @model_validator(mode='before')
    @classmethod
    def supply_profile_directory(cls, value):
        if isinstance(value, dict) and value.get('execution_profile'):
            return {'working_dir': '/workspace', **value}
        return value

    @model_validator(mode='after')
    def validate_execution_options(self):
        if self.execution_profile:
            from .native_bridge.adapters.base import build_execution_profile

            build_execution_profile(self.execution_profile, self.execution_options, self.task_context, self.user_prompt)
            for key in self.model_fields_set - {
                'working_dir',
                'execution_profile',
                'execution_options',
                'task_context',
                'user_prompt',
                'get_user_input',
            }:
                if getattr(self, key) != type(self).model_fields[key].get_default(call_default_factory=True):
                    raise ValueError(f'{key} must be set through execution_options for a profile')
            if self.working_dir != '/workspace':
                raise ValueError('working_dir must be set through execution_options for a profile')
        elif self.execution_options:
            raise ValueError('execution_options requires execution_profile')
        return self

    version: str = DEFAULT_CODEX_VERSION
    """Native bridge retains its existing pinned installation policy."""

    binary_url: str | None = None
    """Optional direct Codex binary URL. When provided, bypasses package installation."""

    binary_checksum: str | None = None
    """Optional SHA256 checksum when using binary_url."""

    extra_env: dict[str, str] | None = None
    """Extra environment variables for Codex, pre-start scripts, and MCP servers."""

    pre_start_scripts: list[CodexNativeBridgePreStartScript] = Field(default_factory=list)
    """Scripts executed in order inside the sandbox before Codex starts."""

    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    """Stdio MCP servers written into the isolated Codex config for this run."""


def _emit_codex_timeline_spans(segments: list[dict]) -> None:
    """Emit post-hoc ``agent_observation`` spans for the rebuilt codex timeline.

    Each segment becomes one ``agent_observation`` span with EXPLICIT start/end
    ``time_unix_nano`` (the synthetic, ordered-but-approximate window from
    build_timeline_segments), parented to the current codex agent span so it lands
    in the same trace the HytunerProcessor is consuming. We emit raw
    SpanStart/SpanEnd events (as replay_span_events does) rather than start_span(),
    because start_span() would stamp "now" and collapse every segment to a
    zero-width cluster, destroying the ordered spread used by the chrome-trace view.

    Only ``agent_observation`` spans are emitted. agent_inference / agent_compaction
    spans are deliberately never synthesized: they mutate the HytunerProcessor token
    buffer / segment state (hytuner.py on_end), so faking them post-hoc would corrupt
    the training trajectory. agent_observation spans only append to agent_timeline,
    which is telemetry-only and safe.
    """
    if not segments:
        return

    from nexus.trace import get_current_span
    from nexus.trace.controller import SpanController
    from nexus.trace.events import SpanEndEvent, SpanStartEvent

    parent = get_current_span()
    if parent is None:
        logger.debug('Codex timeline: no active parent span; skipping timeline span emission')
        return

    controller = SpanController.get_instance_sync()
    for seg in segments:
        start_ns = int(seg['start_time'] * 1e9)
        end_ns = int(seg['end_time'] * 1e9)
        attributes: dict = {
            'type': 'agent_observation',
            'tool_name': seg.get('name', 'observation'),
        }
        tool_args = seg.get('tool_args')
        if tool_args is not None:
            attributes['hytuner.tool_args'] = json.dumps(tool_args, ensure_ascii=False)

        start_event = SpanStartEvent(
            time_unix_nano=start_ns,
            parent_span_id=parent.span_id,
            trace_id=parent.trace_id,
            name='agent.observation',
            attributes=attributes,
        )
        controller.add_event(start_event)
        controller.add_event(SpanEndEvent(span_id=start_event.span_id, time_unix_nano=end_ns))


@agent_registry.register(
    'codex',
    meta={'gateway': {'scaffold_policy': 'codex'}, 'supports_execution_target': True, 'supports_execution_user': True},
)
async def run_codex(input_data: CodexInput) -> CodexOutput:
    """Run Codex inside the runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)

    logger.debug(f'Starting Codex agent for task: {task[:100]}...')

    from .runtime_setup import setup_and_run_codex

    # Execution mode belongs to agent_config; task_config.mode describes the task (e.g. Harbor's 'agent').
    mode = input_data.mode
    # Trace switches retain task_config overrides; a missing context keeps the input defaults.
    timeline_collect_tool_args = input_data.timeline_collect_tool_args
    retain_json_events_stdout = input_data.retain_json_events_stdout
    execution_context = None
    try:
        from nexus.core import require_context

        execution_context = require_context()
        task_config = execution_context.task_config
        timeline_collect_tool_args = bool(task_config.get('timeline_collect_tool_args', timeline_collect_tool_args))
        retain_json_events_stdout = bool(task_config.get('retain_json_events_stdout', retain_json_events_stdout))
    except Exception as e:  # pragma: no cover - defensive: fall back to input-field defaults
        logger.debug('Codex: could not read trace settings from task_config (%s); using input defaults', e)

    if mode == 'app_server' and input_data.logs_dir is not None:
        raise ValueError('Codex app_server mode does not support logs_dir')

    if input_data.goal_mode:
        if mode != 'app_server':
            raise ValueError('goal_mode requires mode=app_server')
    elif input_data.goal_token_budget is not None:
        raise ValueError('goal_token_budget requires goal_mode')

    if execution_context is not None and _uses_native_responses_compaction(
        version=input_data.version,
        use_azure_provider=input_data.use_azure_provider,
    ):
        _validate_native_compaction_client(
            execution_context.openai_client,
            version=input_data.version,
            use_azure_provider=input_data.use_azure_provider,
        )

    host_start_time = time.time()
    # Keep the host compatible with published runtime images. Codex 0.147 settings are CLI
    # config overrides instead of new Python arguments to the remotely invoked runtime function.
    extra_cli_args = list(input_data.extra_cli_args)
    for image_path in input_data.images:
        # `codex exec -i` splits values on commas, so a comma in the path silently becomes two.
        if ',' in image_path:
            raise ValueError(f'Codex image path cannot contain a comma: {image_path!r}')
        extra_cli_args.extend(['-i', image_path])
    if input_data.model_context_window is not None:
        extra_cli_args.extend(['-c', f'model_context_window={int(input_data.model_context_window)}'])
    if input_data.reasoning_effort is not None:
        extra_cli_args.extend(['-c', f'model_reasoning_effort={json.dumps(input_data.reasoning_effort)}'])
    if input_data.auto_compact_token_limit_scope is not None:
        extra_cli_args.extend([
            '-c',
            f'model_auto_compact_token_limit_scope={json.dumps(input_data.auto_compact_token_limit_scope)}',
        ])

    runtime_kwargs: dict[str, Any] = {
        'task': task,
        'working_dir': input_data.working_dir,
        'timeout': input_data.timeout,
        'version': input_data.version,
        'model': input_data.model,
        'npm_registry': input_data.npm_registry,
        'extra_cli_args': extra_cli_args,
        'developer_instructions': input_data.developer_instructions,
        'extra_config_toml': input_data.extra_config_toml,
        'use_azure_provider': input_data.use_azure_provider,
        'auto_compact_token_limit': input_data.auto_compact_token_limit,
    }
    if input_data.model_instructions_file is not None:
        runtime_kwargs['model_instructions_file'] = input_data.model_instructions_file
    if input_data.binary_url is not None:
        runtime_kwargs['binary_url'] = input_data.binary_url
        runtime_kwargs['binary_checksum'] = input_data.binary_checksum
    if input_data.extra_env is not None:
        runtime_kwargs['extra_env'] = input_data.extra_env
    if input_data.mcp_servers:
        runtime_kwargs['mcp_servers'] = input_data.mcp_servers
    if input_data.model_catalog_json is not None:
        runtime_kwargs['model_catalog_json'] = input_data.model_catalog_json
    if input_data.model_catalog_overrides is not None:
        runtime_kwargs['model_catalog_overrides'] = input_data.model_catalog_overrides
    if input_data.experimental_use_freeform_apply_patch is not None:
        runtime_kwargs['experimental_use_freeform_apply_patch'] = input_data.experimental_use_freeform_apply_patch

    if input_data.codex_home_name is not None:
        runtime_kwargs['codex_home_name'] = input_data.codex_home_name
    if input_data.normalize_exec_command_metadata:
        runtime_kwargs['normalize_exec_command_metadata'] = True
    if input_data.compact_prompt is not None:
        runtime_kwargs['compact_prompt'] = input_data.compact_prompt
    runtime_kwargs['mode'] = mode
    if input_data.goal_mode:
        runtime_kwargs['goal_mode'] = True
        if input_data.goal_token_budget is not None:
            runtime_kwargs['goal_token_budget'] = input_data.goal_token_budget
    if input_data.logs_dir is not None:
        runtime_kwargs['logs_dir'] = input_data.logs_dir
    if not input_data.wrap_plaintext_reasoning_content:
        runtime_kwargs['wrap_plaintext_reasoning_content'] = False
    if input_data.supports_tool_references is not None:
        runtime_kwargs['supports_tool_references'] = input_data.supports_tool_references
    if input_data.tool_search_output_format != 'tool_reference':
        runtime_kwargs['tool_search_output_format'] = input_data.tool_search_output_format

    # The remote setup inherits this span ID, so both sides address the same
    # command deadline without adding arguments to the runtime setup protocol.
    runtime_trace_context = RemoteTraceContext.from_current_span()
    if runtime_trace_context is None:
        raise RuntimeError('Codex active-time timeout requires an active trace span')
    if execution_context is not None:
        if execution_context.runtime_command_timeout_id is not None:
            raise RuntimeError('ExecutionContext already has an active runtime command timeout')
        execution_context.runtime_command_timeout_id = runtime_trace_context.parent_span_id
    try:
        result = await call_runtime_agent(
            setup_and_run_codex,
            runtime_trace_context=runtime_trace_context,
            # Only training can park on a partial-rollout boundary. Keep validation,
            # standalone, and unknown invocations bounded by the host deadline.
            disable_invoke_wait_timeout=(
                execution_context is not None and execution_context.task_config.get('is_validation') is False
            ),
            **runtime_kwargs,
        )
    finally:
        if execution_context is not None:
            execution_context.runtime_command_timeout_id = None
    host_end_time = time.time()

    logger.info(f'Codex execution completed: success={result.get("success")}')

    # Emit the trajectory termination reason and observation time as span attributes so they flow
    # through the standard HytunerProcessor pipeline (hytuner.termination_reason /
    # hytuner.observation_time -> traj/train/termination/<reason>/ratio, agent_loop/tool_calls).
    # This mirrors the reference scaffolds swe_agent (agent.py) and terminus2 (agent.py). Codex runs
    # its agent loop inside the @openai/codex CLI subprocess.
    #
    # Coarse path (mode='exec'): classify termination from the CLI exit status only
    # (ENV_DONE / TASK_TIMEOUT / UNHANDLED_ERROR). observation_time is approximated by
    # execution_time (which also includes LLM generation time, so it is an upper bound rather than
    # pure tool wall-clock).
    #
    # Fine-grained exec path (mode='exec_json', the default): additionally parse the --json event
    # stream to refine the termination reason (CODEX_TURN_FAILED / CODEX_ERROR) and rebuild an ordered
    # agent_timeline of tool observations. The codex stream has no timestamps/durations and no
    # compaction event (rust-v0.146.1), so per-tool timing stays approximate and NO
    # agent_inference/agent_compaction spans are synthesized (they would corrupt the
    # HytunerProcessor token buffer / segmentation).
    exit_code = result.get('exit_code', -1)
    timed_out = bool(result.get('timed_out', False))
    execution_time = float(result.get('execution_time', 0.0) or 0.0)

    termination_reason = result.get('termination_reason') or (
        'TASK_TIMEOUT' if timed_out else ('ENV_DONE' if exit_code == 0 else 'UNHANDLED_ERROR')
    )
    event_stream: CodexEventStream | None = None

    json_stdout = result.get('json_events_stdout') or ''
    if mode == 'exec_json' and json_stdout:
        # Import lazily so the coarse path never pays for it and an import error here can never
        # break a working codex run.
        try:
            from .json_events import (
                build_timeline_segments,
                classify_termination,
                parse_codex_jsonl,
            )

            events = parse_codex_jsonl(json_stdout)
            termination_reason = classify_termination(events, exit_code, timed_out)
            _emit_codex_timeline_spans(
                build_timeline_segments(
                    events,
                    start_time=host_start_time,
                    end_time=host_end_time,
                    include_tool_args=timeline_collect_tool_args,
                )
            )
        except Exception as e:  # pragma: no cover - defensive: never break a real run on trace rebuild
            logger.warning('Codex --json trace rebuild failed (%s); falling back to coarse termination', e)

    raw_event_stream = result.get('event_stream')
    if mode == 'app_server' and raw_event_stream:
        try:
            from .events import classify_app_server_termination, reduce_app_server_events

            event_stream = CodexEventStream.model_validate(raw_event_stream)
            termination_reason = classify_app_server_termination(event_stream.events, exit_code, timed_out)
            reduction = reduce_app_server_events(
                event_stream.events,
                include_tool_args=timeline_collect_tool_args,
            )
            _emit_codex_timeline_spans(reduction.timeline_segments)
        except Exception as e:  # pragma: no cover - telemetry must not break a successful run
            logger.warning('Codex app-server event projection failed (%s); keeping the run result', e)

    # Fallback for a run whose event stream did not name the cause -- an old in-sandbox proxy that
    # does not prefix the error code, or --json disabled entirely. The host's last-settled call is
    # the closest it can get to the one codex gave up on, so it only fills a gap the stream left:
    # a stream that already classified the run (including TRUNCATION on its own evidence) is never
    # overridden. See _TRUNCATION_OVERRIDABLE_REASONS for precedence.
    if (
        result.get('last_llm_error_code') == CONTEXT_LENGTH_EXCEEDED_CODE
        and termination_reason in _TRUNCATION_OVERRIDABLE_REASONS
    ):
        logger.debug(
            'Codex: re-labelling %s as TRUNCATION (host refused the last call on token budget)', termination_reason
        )
        termination_reason = 'TRUNCATION'
    termination_reason = _relabel_clean_exit_on_last_call(termination_reason, result, 'Codex')

    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={
                'hytuner.termination_reason': termination_reason,
                'hytuner.observation_time': execution_time,
            },
        )

    failure_kind = result.get('failure_kind')
    failure_stage = result.get('failure_stage')
    stderr = result.get('stderr', '')
    if not result.get('success', False) and failure_kind is None:
        # Old Runtime images did not classify failures. A timeout, truncation, JSON
        # failure event, or real process exit is enough evidence that Codex started;
        # otherwise fail closed instead of grading a potentially pristine workspace.
        process_failure = (
            timed_out
            or termination_reason in {'TRUNCATION', 'CODEX_TURN_FAILED', 'CODEX_ERROR'}
            or (isinstance(exit_code, int) and exit_code not in {-1, 0})
        )
        if process_failure:
            failure_kind = 'agent'
            failure_stage = 'process'
        else:
            failure_kind = 'infrastructure'
            failure_stage = 'runtime_protocol'
            protocol_error = 'Runtime returned an unclassified Codex failure; update the Nexus Runtime image'
            stderr = f'{stderr}\n{protocol_error}'.strip()

    exit_status = _standardize_exit_status(termination_reason)
    output = result.get('output', '')
    final_answer = output.strip() if exit_status == 'completed' and isinstance(output, str) and output.strip() else None
    return CodexOutput(
        success=result.get('success', False),
        output=output,
        final_answer=final_answer,
        stderr=stderr,
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
        event_stream=event_stream,
        failure_kind=failure_kind,
        failure_stage=failure_stage,
        exit_status=exit_status,
        termination_reason=termination_reason,
        json_events_stdout=json_stdout if retain_json_events_stdout and mode == 'exec_json' else None,
    )


@agent_registry.register('codex_native_bridge')
async def run_codex_native_bridge(input_data: CodexNativeBridgeInput) -> CodexOutput:
    return await run_codex_native_bridge_with_adapter(input_data)


async def run_codex_native_bridge_with_adapter(
    input_data: CodexNativeBridgeInput,
    *,
    api_protocol: str = 'responses',
    format_terminal_error: Callable[[list[dict[str, Any]], str], str] | None = None,
) -> CodexOutput:
    """Run Codex in a native sandbox through the sandbox-local bridge."""
    if input_data.execution_profile:
        from nexus.core import require_context

        from .native_bridge.adapters.base import build_execution_profile

        profile = build_execution_profile(
            input_data.execution_profile, input_data.execution_options, input_data.task_context, input_data.user_prompt
        )
        if input_data.get_user_input is not None:
            configure_input = getattr(profile, 'set_user_input', None)
            if not callable(configure_input):
                raise ValueError('Selected Codex profile does not support user input')
            configure_input(input_data.get_user_input)
        result = await profile.run(require_context().get_openai_client())
        return CodexEpisodeOutput(**result)
    if input_data.logs_dir is not None:
        raise ValueError('codex_native_bridge does not support logs_dir')
    task = coerce_user_prompt(input_data.user_prompt)

    logger.debug(f'Starting Codex native bridge agent for task: {task[:100]}...')

    from .native_bridge.agent_runtime import setup_and_run_native_bridge_codex

    emit_json_trace = input_data.emit_json_trace
    timeline_collect_tool_args = input_data.timeline_collect_tool_args
    try:
        from nexus.core import require_context

        task_config = require_context().task_config
        emit_json_trace = bool(task_config.get('emit_json_trace', emit_json_trace))
        timeline_collect_tool_args = bool(task_config.get('timeline_collect_tool_args', timeline_collect_tool_args))
    except Exception as e:  # pragma: no cover - defensive: fall back to input-field defaults
        logger.debug(
            'Codex native bridge: could not read trace switches from task_config (%s); using input defaults', e
        )

    host_start_time = time.time()
    result = await call_native_bridge_runtime_agent(
        setup_and_run_native_bridge_codex,
        api_protocol=api_protocol,
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        model=input_data.model,
        npm_registry=input_data.npm_registry,
        extra_cli_args=list(input_data.extra_cli_args),
        developer_instructions=input_data.developer_instructions,
        extra_config_toml=input_data.extra_config_toml,
        model_instructions_file=input_data.model_instructions_file,
        use_azure_provider=input_data.use_azure_provider,
        auto_compact_token_limit=input_data.auto_compact_token_limit,
        emit_json_trace=emit_json_trace,
        binary_url=input_data.binary_url,
        binary_checksum=input_data.binary_checksum,
        extra_env=dict(input_data.extra_env or {}),
        pre_start_scripts=[script.model_dump() for script in input_data.pre_start_scripts],
        mcp_servers=list(input_data.mcp_servers),
    )
    host_end_time = time.time()

    logger.info(f'Codex native bridge execution completed: success={result.get("success")}')

    exit_code = result.get('exit_code', -1)
    timed_out = bool(result.get('timed_out', False))
    execution_time = float(result.get('execution_time', 0.0) or 0.0)

    termination_reason = 'TASK_TIMEOUT' if timed_out else ('ENV_DONE' if exit_code == 0 else 'UNHANDLED_ERROR')

    json_stdout = result.get('json_events_stdout') or ''
    if emit_json_trace and json_stdout:
        try:
            from .json_events import (
                build_timeline_segments,
                classify_termination,
                parse_codex_jsonl,
            )

            events = parse_codex_jsonl(json_stdout)
            termination_reason = classify_termination(events, exit_code, timed_out)
            if format_terminal_error is not None and exit_code != 0:
                result['stderr'] = format_terminal_error(events, result.get('stderr') or '')
            _emit_codex_timeline_spans(
                build_timeline_segments(
                    events,
                    start_time=host_start_time,
                    end_time=host_end_time,
                    include_tool_args=timeline_collect_tool_args,
                )
            )
        except Exception as e:  # pragma: no cover - defensive: never break a real run on trace rebuild
            logger.warning(
                'Codex native bridge --json trace rebuild failed (%s); falling back to coarse termination', e
            )

    if (
        result.get('last_llm_error_code') == CONTEXT_LENGTH_EXCEEDED_CODE
        and termination_reason in _TRUNCATION_OVERRIDABLE_REASONS
    ):
        logger.debug(
            'Codex native bridge: re-labelling %s as TRUNCATION (host refused the last call on token budget)',
            termination_reason,
        )
        termination_reason = 'TRUNCATION'
    termination_reason = _relabel_clean_exit_on_last_call(termination_reason, result, 'Codex native bridge')

    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={
                'hytuner.termination_reason': termination_reason,
                'hytuner.observation_time': execution_time,
            },
        )

    exit_status = _standardize_exit_status(termination_reason)
    output = result.get('output', '')
    final_answer = output.strip() if exit_status == 'completed' and isinstance(output, str) and output.strip() else None
    return CodexOutput(
        success=result.get('success', False),
        output=output,
        final_answer=final_answer,
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
        failure_kind=result.get('failure_kind'),
        failure_stage=result.get('failure_stage'),
        exit_status=exit_status,
        termination_reason=termination_reason,
        json_events_stdout=json_stdout if input_data.retain_json_events_stdout and emit_json_trace else None,
    )


class CodexInteractiveInput(InstalledNodeCliInput, WithInputProvider):
    """Input for the native Codex TUI driven through a PTY.

    Separate from ``CodexInput`` on purpose: ``exec``-only knobs (``emit_json_trace``,
    ``retain_json_events_stdout``, ...) have no meaning for a long-lived TUI, and the
    interactive path tracks a much newer Codex than the batch agent.

    TUI 启动前，会话私有的 ``CODEX_HOME`` 仅信任 ``working_dir``。工具操作在
    Codex 的 workspace-write sandbox 中无人值守运行；提供 ``get_user_input`` 时，
    它负责回答用户问题并生成后续轮次输入。
    """

    get_user_input: Callable[[UserInputRequest | str | None], Awaitable[str | None]] = Field(
        default=_no_more_user_input
    )
    """可选的输入提供方；默认完成一个无人值守轮次。"""

    session_id: str | None = None
    """Stable Nexus-side session id. Generated when omitted."""

    max_turns: int = Field(default=20, ge=1)
    """Maximum number of user turns to send before stopping."""

    close_on_finish: bool = True
    """Close the runtime-side PTY session when this agent returns."""

    version: str = INTERACTIVE_DEFAULT_CODEX_VERSION
    """Codex CLI version, independent from the batch ``codex`` agent."""

    binary_url: str | None = None
    binary_checksum: str | None = None

    model: str | None = None
    """Optional Codex model override written into config.toml."""

    model_catalog_json: dict[str, Any] | None = None
    """Optional inline Codex model catalog materialized inside the session's CODEX_HOME."""

    extra_env: dict[str, str] | None = None
    extra_cli_args: list[str] = Field(default_factory=list)
    extra_config_toml: str | None = None
    model_instructions_file: str | None = None
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)

    use_azure_provider: bool = True
    """Same meaning as on ``CodexInput``; see that field's documentation."""

    rows: int = 50
    cols: int = 200

    hook_input_timeout: float | None = Field(default=None, gt=0)
    """Optional host-side timeout for answering an interaction request."""


class CodexInteractiveOutput(InstalledAgentOutput):
    """Output from the interactive Codex PTY agent."""

    session_id: str = ''
    turns: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)
    ended_by_user: bool = False
    reached_max_turns: bool = False
    status: str | None = None


@agent_registry.register(
    'codex_interactive',
    # Same gateway scaffold policy as the batch agent: callers such as HyTuner
    # reject an agent that declares none. Execution targets are not claimed --
    # this agent owns a local PTY and its own app-server.
    meta={'gateway': {'scaffold_policy': 'codex'}},
)
async def run_codex_interactive(input_data: CodexInteractiveInput) -> CodexInteractiveOutput:
    """Run the native Codex TUI behind a PTY, one or many turns on one session."""
    runtime_module = 'nexus.extensions.agents.codex.interactive.runtime'
    logger.info('Starting interactive Codex agent (multi-turn capable)')

    result = await run_installed_multiturn(
        input_data,
        agent_key='codex_interactive',
        agent_label='Codex (PTY)',
        session_namespace='codex-interactive-session',
        session_turn_target=f'{runtime_module}::run_codex_interactive_session_turn',
        close_session_target=f'{runtime_module}::close_codex_interactive_session',
        close_failure_affects_result=False,
        respond_session_target=f'{runtime_module}::respond_and_wait_codex_interactive_session',
        disable_invoke_wait_timeout=True,
        runtime_kwargs={
            'working_dir': input_data.working_dir,
            'timeout': input_data.timeout,
            'version': input_data.version,
            'npm_registry': input_data.npm_registry,
            'binary_url': input_data.binary_url,
            'binary_checksum': input_data.binary_checksum,
            'model': input_data.model,
            'model_catalog_json': input_data.model_catalog_json,
            'mcp_servers': list(input_data.mcp_servers),
            'extra_env': dict(input_data.extra_env or {}),
            'extra_cli_args': list(input_data.extra_cli_args),
            'extra_config_toml': input_data.extra_config_toml,
            'model_instructions_file': input_data.model_instructions_file,
            'use_azure_provider': input_data.use_azure_provider,
            'rows': input_data.rows,
            'cols': input_data.cols,
        },
    )

    logger.info('Interactive Codex execution completed: success=%s', result.get('success'))
    return CodexInteractiveOutput(
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
    'CodexInput',
    'CodexNativeBridgeInput',
    'CodexExitStatus',
    'CodexInteractiveInput',
    'CodexInteractiveOutput',
    'CodexOutput',
    'CodexEventStream',
    'run_codex',
    'run_codex_interactive',
    'run_codex_native_bridge',
]
