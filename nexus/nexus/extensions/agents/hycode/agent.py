"""HyCode Agent.

Runs HyCode (@tencent/hycode) inside a runtime sandbox. The runtime-side
setup installs the package via npm (from Tencent mirror), selects the
correct binary variant (glibc or musl), starts an Anthropic-to-OpenAI
proxy, optionally patches ~/.claude/ config, runs any pre-start
scripts, and then executes ``opencc-bin --print`` with the task prompt.
LLM calls flow through the host-side OpenAI client via Socket.IO.
"""

import json
import logging
import shlex
import uuid

from typing import Any, Literal

from pydantic import BaseModel, Field

from nexus.core import CONTEXT_LENGTH_EXCEEDED_CODE, agent_registry
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)
from nexus.extensions.agents.installed_common.execution import (
    ComposeExecutionTarget,
    execution_target_shared_path,
    get_agent_execution_target,
)
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.trace import RemoteTraceContext, get_current_span_id, update_span

from .exit_policy import scoreable_exit_name
from .installer import DEFAULT_HYCODE_VERSION


logger = logging.getLogger(__name__)

EXIT_CODE_MAX_TURNS = 64
EXIT_CODE_TOOL_ERROR = 65
EXIT_CODE_CONTEXT_LENGTH = 66

# Filled in when the host got no `exit_code` key back from the sandbox.
#
# Deliberately not -1. The sandbox itself writes -1 in four places, all of them meaning "the
# sandbox knows it broke" and all of them producing a worthless trajectory:
#   runtime_setup.py:259  installing HyCode failed
#   runtime_setup.py:284  starting the Anthropic proxy failed
#   runtime_setup.py:316  a pre-start script failed
#   runtime_setup.py:185  the task timed out and the process was killed (also sets timed_out,
#                         which is decided before the tables below are consulted)
# The first three also report failure_kind/failure_stage, which harbor_bridge maps to
# INITIALIZATION_ERROR before it ever reads the exit code; the tables below only decide the fate
# of a -1 that arrives without them.
# A missing key means something else entirely: the host never received the field, while the run
# itself completed -- measured at a median of 144 turns with 35% scoring full reward. Same number,
# opposite meaning, so the two need distinct values to be handled differently.
EXIT_CODE_MISSING = -5

# `subtype` values on opencc's final `{"type":"result"}` event (see its print.ts).
SUBTYPE_MAX_TURNS = 'error_max_turns'
SUBTYPE_TOOL_ERROR = 'error_tool_use'
SUBTYPE_CONTEXT_LENGTH = 'error_context_length'
SUBTYPE_DURING_EXECUTION = 'error_during_execution'
SUBTYPE_MAX_BUDGET_USD = 'error_max_budget_usd'
SUBTYPE_MAX_STRUCTURED_OUTPUT_RETRIES = 'error_max_structured_output_retries'

# (exit_code, subtype) -> reason, for the exits opencc uses to describe why it stopped.
# Both halves must agree: the pair is emitted from one place in print.ts, so a mismatch means
# something else produced the exit code (a task script exiting 64 on its own, say) and is not
# trusted. Such pairs fall through to the ENV_DONE default.
_CONTROL_EXIT_REASONS = {
    (EXIT_CODE_MAX_TURNS, SUBTYPE_MAX_TURNS): 'MAX_STEPS',
    (EXIT_CODE_TOOL_ERROR, SUBTYPE_TOOL_ERROR): 'FORMAT_ERROR',
    (EXIT_CODE_CONTEXT_LENGTH, SUBTYPE_CONTEXT_LENGTH): 'TRUNCATION',
}

# Keyed on the exit code alone, for runs that ended without a final result event.
#
# Both halves of the pairs above come from that event, so requiring them to agree is not "a
# mismatch to distrust" here -- it is unsatisfiable by construction. Demanding it anyway funnels
# every such run into UNHANDLED_ERROR, which is masked out of training. Measured on A5B: of 678
# runs reaching this branch two thirds exit 64, i.e. they merely used up their turn budget, while
# MAX_STEPS and subtype=error_max_turns never appeared once in the whole run -- proof that the
# paired mapping is dead code on this path.
_NO_RESULT_EXIT_REASONS = {
    # Codes opencc assigns on purpose: it said why it stopped, so trust it.
    EXIT_CODE_MAX_TURNS: 'MAX_STEPS',
    EXIT_CODE_TOOL_ERROR: 'FORMAT_ERROR',
    EXIT_CODE_CONTEXT_LENGTH: 'TRUNCATION',
    # The process ended abnormally but the trajectory is intact, so it trains as a clean finish.
    # run_hycode moves the ones that never settled a model call to ENV_ERROR.
    EXIT_CODE_MISSING: 'ENV_DONE',
    -9: 'ENV_DONE',  # SIGKILL
    143: 'ENV_DONE',  # SIGTERM, in the shell's 128+n spelling
    # print.ts exits 1 via `is_error ? 1 : 0`, but that path emits a result event and so cannot
    # reach this branch. Exit 1 with no result event means the CLI died before printing one.
    1: 'ENV_DONE',
    # Deliberately absent: -1, which the sandbox writes on the infra failures listed at
    # EXIT_CODE_MISSING, and 137/139, which stay UNHANDLED_ERROR for the same reason.
}

# Error subtypes opencc reports without a dedicated exit code: print.ts assigns 64/65 only to
# error_max_turns/error_tool_use, and the rest fall through to `is_error ? 1 : 0`.
#
# classify_termination does not map them: a subtype string alone is a weaker signal than the
# pairs above. run_hycode masks them only when the exit code agrees (exit 1 with is_error=true
# becomes ENV_ERROR); any that still land in ENV_DONE are recorded on the span.
_UNCLASSIFIED_ERROR_SUBTYPES = frozenset({
    SUBTYPE_DURING_EXECUTION,
    SUBTYPE_MAX_BUDGET_USD,
    SUBTYPE_MAX_STRUCTURED_OUTPUT_RETRIES,
})

# Reasons a host-side budget refusal may overwrite with TRUNCATION. Mirrors codex's
# _TRUNCATION_OVERRIDABLE_REASONS but holds ENV_DONE, which codex deliberately excludes: an
# unpatched opencc reports a context refusal as subtype=success, so it lands in ENV_DONE here,
# whereas codex lands in an error bucket. The widening is made safe by also requiring
# is_error is True -- a run that really finished reports is_error=false and cannot enter.
# Deliberate omissions:
#   MAX_STEPS / FORMAT_ERROR / TRUNCATION -- the CLI named these itself; trust it over a host
#                                            guess. TRUNCATION's absence is what retires the
#                                            fallback automatically once opencc emits exit 66.
#   TASK_TIMEOUT / UNHANDLED_ERROR        -- unreachable here: both are assigned in branches
#                                            that return before classify_termination runs.
_TRUNCATION_OVERRIDABLE_REASONS = frozenset({'ENV_DONE'})


def _has_context_length_refusal(result: dict[str, Any]) -> bool:
    if result.get('last_llm_error_code') == CONTEXT_LENGTH_EXCEEDED_CODE:
        return True
    outcomes = result.get('llm_request_outcomes')
    return isinstance(outcomes, dict) and CONTEXT_LENGTH_EXCEEDED_CODE in outcomes.values()


def extract_hycode_final_answer(output: str) -> str:
    """Return the user-facing final answer text of the LAST assistant message in
    HyCode's stream-json output, for task-layer evaluation.

    Only ``text`` (and ``summary``) blocks are included; ``thinking`` blocks are
    deliberately excluded so the judge grades the model's actual reply, not its
    internal reasoning. Returns '' if no answer text can be recovered."""
    last_assistant: dict[str, Any] | None = None
    for line in (output or '').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(event, dict) and event.get('type') == 'assistant':
            message = event.get('message')
            if isinstance(message, dict):
                last_assistant = message
    if last_assistant is None:
        return ''

    content = last_assistant.get('content')
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ''

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get('type')
        if block_type == 'text':
            parts.append(str(block.get('text') or ''))
        elif block_type == 'summary':
            parts.append(str(block.get('summary') or block.get('text') or ''))
    return '\n'.join(p for p in parts if p).strip()


def classify_termination(exit_code: int, result_subtype: str | None, is_error: bool | None = None) -> str:
    """Classify HyCode's explicit RL control exits without changing legacy defaults.

    ENV_DONE is returned for anything not in _CONTROL_EXIT_REASONS, including runs that
    reported is_error=true; test_only_explicit_control_exits_override_legacy_default pins this.
    is_error alone is not enough to drop a sample, so this function never guesses from it.
    run_hycode applies the stronger evidence afterwards: exit 1 together with is_error=true
    becomes ENV_ERROR, and any is_error=true run left in ENV_DONE is marked
    hycode.env_done_despite_error.
    """
    # A run that reported is_error=false finished cleanly, so no control exit applies. Tested
    # against False rather than falsiness on purpose: None means "the runtime image did not
    # report is_error", which must still reach the lookup below.
    if is_error is False:
        return 'ENV_DONE'
    return _CONTROL_EXIT_REASONS.get((exit_code, result_subtype), 'ENV_DONE')


class PreStartScript(BaseModel):
    """A script to execute inside the runtime before HyCode starts."""

    interpreter: Literal['bash', 'python'] = 'bash'
    code: str
    """Inline source code for the chosen interpreter."""

    timeout: float = 300.0
    cwd: str | None = None
    """Override cwd. Defaults to ``working_dir``."""

    env: dict[str, str] | None = None
    """Extra env vars merged on top of the HyCode launch env."""

    fail_on_error: bool = True
    """If True, non-zero exit aborts the agent run."""


class ConfigFilePatch(BaseModel):
    """Merge *content* into the JSON file at *path* before launching HyCode.

    ``path`` may be absolute or start with ``~`` (e.g.
    ``~/.claude/settings.json``) to reference the runtime user's home.
    """

    path: str
    """Path inside the runtime. ``~`` is expanded against the runtime's HOME."""

    content: dict[str, Any]
    """Keys to deep-merge into the existing JSON (or create fresh)."""


class HyCodeInput(InstalledNodeCliInput):
    """Input for HyCode agent."""

    version: str = DEFAULT_HYCODE_VERSION
    """HyCode package version."""

    disable_invoke_wait_timeout: bool | None = None
    """Disable cumulative host-side invoke waiting while retaining HTTP timeouts.

    None follows the execution context: disabled for training, enabled otherwise.
    An explicit bool overrides that default. The caller must enforce an outer
    task deadline when this is True.
    """

    extra_env: dict[str, str] | None = None
    """Extra environment variables for the HyCode process."""

    emit_spans: bool = True
    """Import local OTEL spans after exit. Explicit values override task_config.hycode_emit_spans."""

    extra_cli_args: list[str] = Field(default_factory=list)
    """Additional CLI arguments. The prompt is sent through stdin, so these must include
    ``--print`` and must not contain positional arguments, which the CLI would join to it."""

    config_file_patches: list[ConfigFilePatch] = Field(default_factory=list)
    """JSON config files to patch before launch (e.g. ~/.claude/settings.json)."""

    pre_start_scripts: list[PreStartScript] = Field(default_factory=list)
    """Scripts executed in order before HyCode starts."""

    conversion_log_level: str = 'INFO'
    """Log level for anthropic<->openai conversion dumps."""

    thinking_preset: str = 'hunyuan'
    """Thinking format preset for the anthropic<->openai converter."""

    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    """Nexus MCP server specs exposed to HyCode."""

    mcp_config_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Additional HyCode-native MCP server definitions."""

    no_thinking_request_args: dict[str, Any] | None = None
    """Override ``request_args`` for LLM calls when HyCode does NOT request thinking.
    ``no_thinking_request_args_mode`` controls replacement versus safe inheritance."""

    no_thinking_request_args_mode: Literal['replace', 'inherit'] = 'replace'
    """``replace`` preserves the historical isolated override; ``inherit`` keeps host routing fields."""

    haiku_model: str | None = None
    """Model name to use when HyCode requests a haiku-class model."""

    max_turns: int | None = None
    """Maximum agentic turns passed to HyCode's print-mode loop."""

    stop_on_tool_error: bool = False
    """Stop the trajectory when HyCode reports a failed tool batch."""


_STRICT_MCP_REMOTE_PATH = '/tmp/nexus_hycode_strict_mcp.json'


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


async def _prepare_strict_mcp_file(
    input_data: 'HyCodeInput',
    extra_cli_args: list[str],
) -> None:
    from nexus.extensions.agents.claude_code.mcp import mcp_config_servers_from_specs

    task_servers = mcp_config_servers_from_specs(list(input_data.mcp_servers))
    duplicate_names = sorted(set(task_servers) & set(input_data.mcp_config_servers))
    if duplicate_names:
        raise ValueError(f'Duplicate HyCode MCP server names across inputs: {duplicate_names}')
    native_servers = {**task_servers, **input_data.mcp_config_servers}
    if not native_servers:
        return
    if any(
        arg == '--strict-mcp-config' or arg == '--mcp-config' or arg.startswith('--mcp-config=')
        for arg in extra_cli_args
    ):
        raise ValueError('HyCode task MCP injection cannot be combined with explicit MCP CLI arguments')

    execution_target = get_agent_execution_target()
    remote_path = str(
        execution_target_shared_path(
            execution_target,
            'hycode-mcp.json',
            default=_STRICT_MCP_REMOTE_PATH,
        )
    )
    config_json = json.dumps({'mcpServers': native_servers}, ensure_ascii=False, separators=(',', ':'))
    file_mode = 0o644 if execution_target is not None else 0o600
    await _upload_runtime_text(remote_path, config_json, 'HyCode strict MCP config', mode=file_mode)
    extra_cli_args.extend(['--mcp-config', remote_path, '--strict-mcp-config'])


class HyCodeOutput(InstalledAgentOutput):
    """Output from HyCode agent."""

    exit_status: str
    """Canonical Agent exit status consumed by task adapters."""

    termination_reason: str
    result_subtype: str | None = None
    num_turns: int | None = None
    scoreable_process_exit: bool = False
    hycode_exit_name: str | None = None


@agent_registry.register(
    'hycode',
    meta={'gateway': {'scaffold_policy': 'hycode'}, 'supports_execution_target': True},
)
async def run_hycode(input_data: HyCodeInput) -> HyCodeOutput:
    """Run HyCode agent inside the runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)

    extra_env = dict(input_data.extra_env or {})
    extra_cli_args = list(input_data.extra_cli_args)
    pre_start_scripts = [script.model_dump() for script in input_data.pre_start_scripts]
    await _prepare_strict_mcp_file(input_data, extra_cli_args)
    if input_data.max_turns is not None:
        extra_cli_args.extend(['--max-turns', str(input_data.max_turns)])
    if input_data.stop_on_tool_error and '--stop-on-tool-error' not in extra_cli_args:
        extra_cli_args.append('--stop-on-tool-error')
    agent_span_id = get_current_span_id()

    logger.debug('Starting HyCode agent for task: %s...', task[:100])
    logger.debug('Executing HyCode in runtime...')
    from .runtime_setup import setup_and_run_hycode

    execution_context = None
    try:
        from nexus.core import require_context

        execution_context = require_context()
    except RuntimeError:
        # Direct library use outside a Nexus run has no rollout pause controller.
        pass

    task_config = execution_context.task_config if execution_context is not None else {}
    emit_spans = (
        input_data.emit_spans
        if 'emit_spans' in input_data.model_fields_set
        else bool(task_config.get('hycode_emit_spans', input_data.emit_spans))
    )
    collect_tool_args = bool(task_config.get('timeline_collect_tool_args', False))
    spans_path = None
    if emit_spans:
        from .telemetry import build_telemetry_env

        execution_target = get_agent_execution_target()
        spans_name = f'hycode-spans-{uuid.uuid4().hex}.jsonl'
        spans_path = str(execution_target_shared_path(execution_target, spans_name, default=f'/tmp/{spans_name}'))
        spans_env_path = (
            f'{execution_target.transport_container_dir}/{spans_name}'
            if isinstance(execution_target, ComposeExecutionTarget)
            else spans_path
        )
        extra_env.update(build_telemetry_env(spans_env_path, collect_tool_args=collect_tool_args))
    runtime_trace_context = RemoteTraceContext.from_current_span()
    if execution_context is not None and runtime_trace_context is None:
        raise RuntimeError('HyCode active-time timeout requires an active trace span')
    if execution_context is not None:
        assert runtime_trace_context is not None
        if execution_context.runtime_command_timeout_id is not None:
            raise RuntimeError('ExecutionContext already has an active runtime command timeout')
        execution_context.runtime_command_timeout_id = runtime_trace_context.parent_span_id

    try:
        try:
            result = await call_runtime_agent(
                setup_and_run_hycode,
                no_thinking_request_args=input_data.no_thinking_request_args,
                no_thinking_request_args_mode=input_data.no_thinking_request_args_mode,
                haiku_model=input_data.haiku_model,
                runtime_trace_context=runtime_trace_context,
                # Explicit configuration wins; otherwise only training can park
                # on a partial-rollout boundary without a host wait deadline.
                disable_invoke_wait_timeout=(
                    input_data.disable_invoke_wait_timeout
                    if input_data.disable_invoke_wait_timeout is not None
                    else execution_context is not None and execution_context.task_config.get('is_validation') is False
                ),
                task=task,
                working_dir=input_data.working_dir,
                timeout=input_data.timeout,
                version=input_data.version,
                npm_registry=input_data.npm_registry,
                extra_env=extra_env,
                extra_cli_args=extra_cli_args,
                config_file_patches=[p.model_dump() for p in input_data.config_file_patches],
                pre_start_scripts=pre_start_scripts,
                conversion_log_level=input_data.conversion_log_level,
                thinking_preset=input_data.thinking_preset,
            )
        finally:
            if execution_context is not None:
                execution_context.runtime_command_timeout_id = None
    except RuntimeUnhealthyError as e:
        # The sandbox was marked unusable by the provider watchdog (stopped / OOMKilled /
        # SandboxExit -- see RuntimeUnhealthyError). Without this handler the exception leaves
        # run_hycode entirely, classify_termination never runs, and hytuner's agent loop records
        # a bare UNHANDLED_ERROR. Both are masked out, so the sample's fate is unchanged; what
        # this buys is attribution -- roughly half of the observed UNHANDLED_ERROR occurrences
        # are sandbox deaths.
        #
        # Returning instead of re-raising hands control back to the task layer, which will go on
        # to score the (already dead) sandbox. That costs one doomed scoring attempt and may
        # overwrite the span below in tasks that both discard this return value and swallow the
        # failure. failure_kind='infrastructure' is set so a task layer that does read the
        # output can bail out before scoring.
        logger.exception('HyCode runtime became unhealthy; classifying as ENV_ERROR')
        if agent_span_id:
            try:
                update_span(
                    agent_span_id,
                    attributes={
                        'hytuner.termination_reason': 'ENV_ERROR',
                        'hycode.termination_source': 'runtime_unhealthy',
                    },
                )
            except Exception:
                logger.exception('Failed to record HyCode ENV_ERROR on span %s', agent_span_id)
        return HyCodeOutput(
            success=False,
            output='',
            stderr=f'Runtime unhealthy: {e}',
            exit_code=-1,
            execution_time=0.0,
            termination_reason='ENV_ERROR',
            failure_kind='infrastructure',
            failure_stage='runtime_unhealthy',
            # Spelled out rather than derived: the local termination_reason is not bound until
            # after this early return.
            exit_status='env_error',
        )

    if spans_path is not None:
        from .telemetry import import_hycode_spans

        await import_hycode_spans(
            spans_path,
            collect_tool_args=collect_tool_args,
            tool_args_min_duration=float(task_config.get('timeline_tool_args_min_duration', 0)),
        )

    return classify_hycode_result(result, agent_span_id=agent_span_id)


def classify_hycode_result(result: dict[str, Any], *, agent_span_id: str | None = None) -> HyCodeOutput:
    """Apply the established OpenCC exit policy to observed runtime facts."""
    raw_exit_code = result.get('exit_code')
    exit_code = raw_exit_code if isinstance(raw_exit_code, int) else EXIT_CODE_MISSING
    result_subtype = result.get('result_subtype')
    result_is_error = result.get('result_is_error')
    num_turns = result.get('num_turns')
    result_count = result.get('result_count', 0)
    timed_out = bool(result.get('timed_out', False))
    used_host_fallback = False
    if timed_out:
        termination_reason = 'TASK_TIMEOUT'
    elif result_count == 0 and _has_context_length_refusal(result):
        # Must precede the branch below. An unpatched opencc reports a context refusal as
        # subtype=success + is_error=true + exit 1, and the table below reads a bare exit 1 as
        # ENV_DONE -- so a refusal that killed the CLI before it emitted a result event would be
        # trained as a clean finish, which is the exact failure mode f481ce284 removed. The host
        # fallback in the else branch cannot cover this: it only runs once a result event exists.
        termination_reason = 'TRUNCATION'
        used_host_fallback = True
    elif not result.get('success', False) and result_count == 0:
        termination_reason = _NO_RESULT_EXIT_REASONS.get(exit_code, 'UNHANDLED_ERROR')
    else:
        termination_reason = classify_termination(exit_code, result_subtype, result_is_error)
        # An unpatched opencc (<=0.1.9) cannot name a context refusal: the error crosses the
        # Socket.IO boundary as an opaque 500, so the CLI reports subtype=success + is_error=true
        # + exit 1. The host refused the call and is the only side that knows why, so fill the
        # gap from its per-request outcome. Verified against @tencent/hycode@0.1.8.
        # Once opencc emits exit 66 this never fires: classify_termination already returned
        # TRUNCATION, which is not in _TRUNCATION_OVERRIDABLE_REASONS.
        if (
            termination_reason in _TRUNCATION_OVERRIDABLE_REASONS
            and result_is_error is True
            and _has_context_length_refusal(result)
        ):
            termination_reason = 'TRUNCATION'
            used_host_fallback = True
        # print.ts exits 1 only when the final result reports is_error=true and no control exit
        # applies: the last turn is a synthetic API error (e.g. the bridge gave up on a paused
        # router) or the loop aborted. The task stopped unfinished, so scoring it as ENV_DONE
        # would train an infrastructure failure as the model's own result.
        if termination_reason == 'ENV_DONE' and exit_code == 1 and result_is_error is True:
            termination_reason = 'ENV_ERROR'

    hycode_exit_name = (
        scoreable_exit_name(exit_code) if termination_reason == 'ENV_DONE' and result_count == 0 else None
    )
    scoreable_process_exit = hycode_exit_name is not None
    termination_source = 'host_llm_error_code' if used_host_fallback else 'opencc_exit_protocol'
    failure_kind = result.get('failure_kind')
    failure_stage = result.get('failure_stage')
    # A scoreable exit trains on the strength of an intact trajectory. When the host bridge settled
    # no model call at all there is no trajectory, and hytuner's V2 admission raises on an empty
    # ENV_DONE trajectory. Treat it like a dead sandbox instead: ENV_ERROR is masked before
    # training, and failure_kind lets the task layer skip scoring. A missing outcomes key means
    # the bridge reported nothing, so that case keeps the legacy classification.
    llm_request_outcomes = result.get('llm_request_outcomes')
    no_model_call_exit = scoreable_process_exit and isinstance(llm_request_outcomes, dict) and not llm_request_outcomes
    if no_model_call_exit:
        termination_reason = 'ENV_ERROR'
        termination_source = 'no_model_call'
        hycode_exit_name = None
        scoreable_process_exit = False
        failure_kind = 'infrastructure'
        failure_stage = 'no_model_call'
    logger.info(
        'HyCode execution completed: exit_code=%s subtype=%s result_count=%s termination=%s',
        exit_code,
        result_subtype,
        result_count,
        termination_reason,
    )
    if termination_reason != 'ENV_DONE' or scoreable_process_exit:
        logger.warning(
            'HyCode termination classified: termination=%s source=%s '
            'exit_code=%s subtype=%s is_error=%s result_count=%s',
            termination_reason,
            termination_source,
            exit_code,
            result_subtype,
            result_is_error,
            result_count,
        )
    if agent_span_id:
        attributes: dict[str, object] = {
            'hytuner.termination_reason': termination_reason,
            'hycode.termination_source': termination_source,
            'hycode.exit_code': exit_code,
        }
        if result_subtype is not None:
            attributes['hycode.result_subtype'] = result_subtype
        # ENV_DONE means "trained as a clean finish". An is_error=true run only gets here when its
        # exit code was not 1 (missing, a signal, or a control code its subtype disagrees with),
        # so there is no agreeing evidence to drop it; flag it so the gap stays countable.
        if termination_reason == 'ENV_DONE' and result_is_error is True:
            attributes['hycode.env_done_despite_error'] = True
            if result_subtype in _UNCLASSIFIED_ERROR_SUBTYPES:
                attributes['hycode.unclassified_error_subtype'] = result_subtype
        if scoreable_process_exit:
            attributes['hycode.scoreable_process_exit'] = True
        if num_turns is not None:
            attributes['hycode.num_turns'] = num_turns
        try:
            update_span(agent_span_id, attributes=attributes)
        except Exception:
            logger.exception('Failed to record HyCode termination on span %s', agent_span_id)

    return HyCodeOutput(
        success=not no_model_call_exit
        and (result.get('success', False) or scoreable_process_exit or exit_code == EXIT_CODE_MISSING),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=exit_code,
        execution_time=result.get('execution_time', 0),
        # Set only by the sandbox's install / proxy / pre_start early-returns, whose stages are
        # in harbor_bridge's _INITIALIZATION_FAILURE_STAGES -- without this passthrough those
        # three land in UNHANDLED_ERROR alongside real agent failures, and the task layer goes
        # on to score a sandbox that never had a working binary.
        failure_kind=failure_kind,
        failure_stage=failure_stage,
        # termination_reason stays ENV_DONE; exit_status carries the name into the T3 mapping.
        exit_status=(hycode_exit_name or termination_reason).lower(),
        termination_reason=termination_reason,
        result_subtype=result_subtype,
        num_turns=num_turns,
        scoreable_process_exit=scoreable_process_exit,
        hycode_exit_name=hycode_exit_name,
    )


__all__ = [
    'ConfigFilePatch',
    'classify_termination',
    'classify_hycode_result',
    'extract_hycode_final_answer',
    'HyCodeInput',
    'HyCodeOutput',
    'PreStartScript',
    'run_hycode',
]
