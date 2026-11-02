"""OSWorld stage results; semantic exit codes are not CLI process return codes."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from enum import Enum
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, model_validator

from nexus.runner.hooks.base import RunHook, run_hook_registry


def summarize_processors(results: dict[str, Any]) -> dict[str, Any]:
    """Retain metrics and trajectory payloads consumed by the bridge.

    The bridge persists and removes its internal chat/ATIF payloads after
    on_run_output. Keep those payloads intact until then. Token aggregates
    retain every field and numeric value without text truncation.
    """
    summary = {
        key: results[key]
        for key in ('token_counter', 'execution_timeout_tracker', '_default_chat_trace', '_default_atif_trace')
        if key in results
    }
    if 'bridge_toolcall_anomaly' in results:
        summary['bridge_toolcall_anomaly'] = {
            client: {key: value for key, value in metrics.items() if key != 'requests'}
            for client, metrics in results['bridge_toolcall_anomaly'].items()
        }
    return summary


def _result_fields(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    data = value.model_dump(mode='json') if hasattr(value, 'model_dump') else value
    if not isinstance(data, dict):
        return {}
    result = {}
    for key in fields:
        item = data.get(key)
        if key not in data or not isinstance(item, (str, int, float, bool, type(None))):
            continue
        # Full text is archived first. Bound even pathological CLI errors and
        # final answers; never truncate a status, code or numeric score.
        if isinstance(item, str) and len(json.dumps(item, ensure_ascii=False).encode('utf-8')) > 8192:
            item = item.encode('utf-8')[:8192].decode('utf-8', errors='ignore')
            while len(json.dumps(item, ensure_ascii=False).encode('utf-8')) > 8100:
                item = item[: len(item) // 2]
            item += '… [see artifacts]'
        result[key] = item
    return result


def summarize_task_output(value: Any) -> Any:
    """Result-only projection, called after archiving the complete task details."""
    data = value.model_dump(mode='json') if hasattr(value, 'model_dump') else dict(value)
    agent = data.get('agent_output')
    if agent is not None:
        agent = _result_fields(
            agent,
            (
                'exit_status',
                'exit_code',
                'error',
                'conclusion',
                'process_exit_code',
                'cc_return_code',
                'codex_return_code',
                'failure_stage',
                'runtime_reason',
                'iterations',
                'execution_time',
                'termination_reason',
                'final_answer',
            ),
        )
        if isinstance(getattr(value, 'agent_output', None), OSWorldAgentOutput):
            agent = OSWorldAgentOutput.model_validate(agent)
    judge = data.get('judge_output')
    if judge is not None:
        judge = _result_fields(judge, ('exit_status', 'exit_code', 'error'))
        if isinstance(getattr(value, 'judge_output', None), OSWorldJudgeOutput):
            judge = OSWorldJudgeOutput.model_validate(judge)
    updates = {
        **_result_fields(data, ('status', 'completed', 'task_id', 'domain', 'score', 'error')),
        'agent_output': agent,
        'judge_output': judge,
        'instruction': '',
        'setup_results': [],
        'evaluator_result': _result_fields(data.get('evaluator_result'), ('score', 'supported', 'error')),
    }
    if hasattr(value, 'model_copy'):
        for key, item in updates.items():
            setattr(value, key, item)
        return value
    return {**updates, 'artifacts': data.get('artifacts')}


class AgentExitStatus(str, Enum):
    FINISHED = 'finished'
    AGENT_TIMEOUT = 'agent_timeout'
    MAX_ITERATIONS = 'max_iterations'
    CONTEXT_LENGTH_EXCEEDED = 'context_length_exceeded'
    OUTPUT_LENGTH_EXCEEDED = 'output_length_exceeded'
    RUNTIME_ERROR_BY_AGENT = 'runtime_error_by_agent'
    RUNTIME_ERROR_BY_ENV = 'runtime_error_by_env'
    INFERENCE_ERROR = 'inference_error'
    ENVIRONMENT_ERROR = 'environment_error'
    UNKNOWN_ERROR = 'unknown_error'


EVALUABLE_AGENT_STATUSES = frozenset({
    AgentExitStatus.FINISHED,
    AgentExitStatus.AGENT_TIMEOUT,
    AgentExitStatus.MAX_ITERATIONS,
    AgentExitStatus.CONTEXT_LENGTH_EXCEEDED,
    AgentExitStatus.OUTPUT_LENGTH_EXCEEDED,
    AgentExitStatus.RUNTIME_ERROR_BY_AGENT,
})


class OSWorldAgentOutput(BaseModel):
    # Retain trajectories, usage, and harness-specific evidence without requiring
    # ordinary agent output models to adopt OSWorld's semantic exit-code contract.
    model_config = ConfigDict(extra='allow')

    exit_status: AgentExitStatus
    exit_code: Literal[0, -1]
    error: str | None = None
    conclusion: Literal['done', 'infeasible'] | None = None
    process_exit_code: int | None = None
    cc_return_code: int | None = None
    codex_return_code: int | None = None
    failure_stage: str | None = None
    runtime_reason: str | None = None

    @model_validator(mode='after')
    def validate_semantic_code(self) -> OSWorldAgentOutput:
        expected = 0 if self.exit_status in EVALUABLE_AGENT_STATUSES else -1
        if self.exit_code != expected:
            raise ValueError(f'{self.exit_status.value} requires semantic exit_code={expected}')
        if self.exit_code == -1 and not self.error:
            raise ValueError('An invalid agent result requires its actual error or an explicit unknown-error detail')
        return self


class OSWorldJudgeOutput(BaseModel):
    exit_status: Literal['finished', 'judge_error']
    exit_code: Literal[0, -1]
    error: str | None = None
    result: dict[str, Any] | None = None

    @model_validator(mode='after')
    def validate_semantic_code(self) -> OSWorldJudgeOutput:
        if self.exit_code != (0 if self.exit_status == 'finished' else -1):
            raise ValueError('Judge status and semantic exit_code disagree')
        if self.exit_code == -1 and not self.error:
            raise ValueError('A failed judge requires its actual error')
        return self


def task_status(
    agent: OSWorldAgentOutput, judge: OSWorldJudgeOutput | None
) -> Literal['success', 'agent_error', 'judge_error']:
    if agent.exit_code != 0:
        return 'agent_error'
    # An unexecuted judge is not a successful judge.
    return 'success' if judge is not None and judge.exit_code == 0 else 'judge_error'


def _parent_result(raw: str) -> dict[str, Any]:
    from nexus.extensions.agents.claude_code.stream_json import ParentSessionState

    parent = ParentSessionState()
    result: dict[str, Any] = {}
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and parent.feed(event) and event.get('type') == 'result':
            result = event
    return result


def _error_status(detail: str) -> AgentExitStatus:
    """Called only for actual exceptions/CLI errors, never ordinary model prose."""
    lowered = detail.lower()
    if any(
        token in lowered
        for token in (
            'context_length_exceeded',
            'context length exceeded',
            'context window exceeded',
            'maximum context length',
            'prompt is too long',
        )
    ):
        return AgentExitStatus.CONTEXT_LENGTH_EXCEEDED
    if 'output_length_exceeded' in lowered:
        return AgentExitStatus.OUTPUT_LENGTH_EXCEEDED
    return AgentExitStatus.INFERENCE_ERROR


class OSWorldInnerTimeoutError(RuntimeError):
    """A nested deadline (e.g. /execute, upload, inference) expired inside the task.

    Deliberately not a TimeoutError: the runner reports every TimeoutError as its
    own task deadline (runner.timeout), which misclassifies these failures.
    """


def _runner_deadline_expired(timeout_handle: Any) -> bool:
    expired = getattr(timeout_handle, 'expired', False)
    return bool(expired() if callable(expired) else expired)


def wrap_inner_timeout(exc: BaseException, *, stage: str, timeout_handle: Any = None) -> BaseException:
    # The runner deadline cancels the task (CancelledError) and is converted to
    # TimeoutError outside this task, so any TimeoutError seen here is nested.
    if not isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or _runner_deadline_expired(timeout_handle):
        return exc
    return OSWorldInnerTimeoutError(
        f'OSWorld {stage} operation timed out before the task deadline ({type(exc).__name__}: {exc or "no detail"})'
    )


def classify_exception(exc: Exception, *, stage: str) -> OSWorldAgentOutput:
    from nexus.core.openai_client import ContextLengthExceededError

    detail = f'{type(exc).__name__}: {exc}'
    raw_reason = getattr(exc, 'reason', None)
    reason = raw_reason if isinstance(raw_reason, str) else None
    if isinstance(exc, ContextLengthExceededError):
        status = AgentExitStatus.CONTEXT_LENGTH_EXCEEDED
    elif reason in {'Process.OOMKilled', 'Process.NonZeroExit', 'Process.CrashLoopBackOff'}:
        status = AgentExitStatus.RUNTIME_ERROR_BY_AGENT
    elif reason or stage == 'runtime':
        status = AgentExitStatus.RUNTIME_ERROR_BY_ENV
    elif isinstance(exc, (ConnectionError, TimeoutError, OSWorldInnerTimeoutError)):
        status = AgentExitStatus.INFERENCE_ERROR if stage == 'inference' else AgentExitStatus.RUNTIME_ERROR_BY_ENV
    elif stage == 'prepare':
        status = AgentExitStatus.ENVIRONMENT_ERROR
    elif stage == 'inference':
        status = _error_status(detail)
    else:
        status = AgentExitStatus.UNKNOWN_ERROR
    return OSWorldAgentOutput(
        exit_status=status,
        exit_code=0 if status in EVALUABLE_AGENT_STATUSES else -1,
        error=detail,
        failure_stage=stage,
        runtime_reason=reason,
    )


def normalize_agent_output(value: Any, *, harness: str | None = None) -> OSWorldAgentOutput:
    """Task-owned projection; retain raw evidence without mutating shared agents."""
    if isinstance(value, OSWorldAgentOutput):
        return value
    data = (
        value.model_dump(mode='python')
        if hasattr(value, 'model_dump')
        else dict(value)
        if isinstance(value, dict)
        else dict(vars(value))
    )
    # Explicit profiles can supply semantic output before entering the task.
    if isinstance(data.get('osworld_stage_output'), dict):
        return OSWorldAgentOutput.model_validate(data['osworld_stage_output'])
    rc = data.get('exit_code')
    status = AgentExitStatus.UNKNOWN_ERROR
    error = data.get('error') or data.get('stderr') or None
    old_status = data.get('exit_status')
    conclusion = 'infeasible' if old_status == 'infeasible' else None
    root = _parent_result(data.get('output') or '') if harness == 'cc' else {}
    root_error = bool(root.get('is_error') or root.get('api_error_status'))
    if data.get('failure_stage') == 'cleanup':
        status = AgentExitStatus.RUNTIME_ERROR_BY_ENV
    elif root_error and root.get('subtype') not in {'max_turns', 'error_max_turns'}:
        error = str(root.get('result') or root.get('errors') or error or 'Claude Code terminal API error')
        status = _error_status(error)
        # CC exhausts its continuation attempts on Messages max_tokens and emits
        # this CLI error instead of the provider's output_length_exceeded code.
        # Only the root error result is authoritative, not assistant/child text.
        if re.match(r"^API Error: Claude's response exceeded the [0-9][0-9,]* output token maximum\.", error):
            status = AgentExitStatus.OUTPUT_LENGTH_EXCEEDED
        if root.get('terminal_reason') == 'prompt_too_long':
            status = AgentExitStatus.CONTEXT_LENGTH_EXCEEDED
    elif data.get('success') is not False and (
        data.get('termination_reason') == 'cc_timeout' or old_status == 'budget_exhausted'
    ):
        status = AgentExitStatus.AGENT_TIMEOUT
    elif data.get('success') is not False and old_status == 'max_iterations':
        status = AgentExitStatus.MAX_ITERATIONS
    elif data.get('success') is not False and old_status in {'finished', 'infeasible'}:
        status = AgentExitStatus.FINISHED
        conclusion = conclusion or 'done'
        error = data.get('error') or None
    elif data.get('terminal_finish_reason') in {'length', 'max_tokens'}:
        status = AgentExitStatus.OUTPUT_LENGTH_EXCEEDED
        error = error or 'Final model response reached its output token limit'
    elif data.get('failure_stage') in {'install', 'proxy', 'pre_start'} or (
        isinstance(error, str)
        and error.startswith((
            'Failed to install Claude Code:',
            'Pre-start script failed:',
            'Failed to register MCP servers:',
        ))
    ):
        status = AgentExitStatus.ENVIRONMENT_ERROR
    elif data.get('terminal_api_error'):
        error = str(data['terminal_api_error'])
        status = _error_status(error)
    elif data.get('budget_stop_confirmed') is True:
        status = AgentExitStatus.AGENT_TIMEOUT
    elif data.get('failure_kind') == 'infrastructure' or (
        isinstance(error, str) and ('exited without writing an exit code' in error or 'AGS OSWorld /execute' in error)
    ):
        status = AgentExitStatus.RUNTIME_ERROR_BY_ENV
    if status not in EVALUABLE_AGENT_STATUSES and not error:
        error = f'Unclassified agent failure: status={old_status!r}, process_exit_code={rc!r}'
    data.update(
        exit_status=status,
        exit_code=0 if status in EVALUABLE_AGENT_STATUSES else -1,
        error=error,
        conclusion=conclusion,
        process_exit_code=rc,
    )
    data.pop('osworld_stage_output', None)
    return OSWorldAgentOutput.model_validate(data)


def record_agent_output(value: Any, *, harness: str) -> OSWorldAgentOutput | None:
    """Publish profile evidence only to the V1 task; V2/ordinary callers stay unchanged."""
    from nexus.core import require_context

    try:
        ctx = require_context()
    except RuntimeError:
        return None
    if ctx.task_name == 'os_world':
        result = normalize_agent_output(value, harness=harness)
        ctx.tool_state['osworld_agent_output'] = result
        return result
    return None


async def run_task_agent(run_agent: Any, **kwargs: Any) -> Any:
    """Observe bare CUA provider calls without changing their retries or requests."""
    from nexus.core import require_context

    ctx = require_context()
    if ctx.agent_name not in {'cua_gpt_agent', 'cua_anthropic_agent'}:
        try:
            return await run_agent(**kwargs)
        except Exception as exc:
            recorded = ctx.tool_state.get('osworld_agent_output')
            if recorded is None:
                recorded = classify_exception(exc, stage='agent')
                ctx.tool_state['osworld_agent_output'] = recorded
                if recorded.exit_code == 0:
                    return recorded
            raise
    originals = ctx.openai_clients
    last_error: Exception | None = None
    last_reason: str | None = None

    class ObservedClient:
        def __init__(self, client: Any):
            self.client = client

        def __getattr__(self, name: str) -> Any:
            return getattr(self.client, name)

        async def create_completion(self, *args: Any, **request: Any) -> Any:
            nonlocal last_error, last_reason
            try:
                response = await self.client.create_completion(*args, **request)
            except Exception as exc:
                last_error = exc
                raise
            last_error = None
            choices = getattr(response, 'choices', None) or []
            last_reason = getattr(choices[0], 'finish_reason', None) if choices else None
            return response

    # A delegating per-run view, not a replacement client implementation: tracing,
    # caching, retry and request policy remain on the original client.
    ctx.openai_clients = cast(Any, {name: ObservedClient(client) for name, client in originals.items()})
    try:
        result = await run_agent(**kwargs)
        if isinstance(result, OSWorldAgentOutput):
            return result
        data = result.model_dump() if hasattr(result, 'model_dump') else dict(result)
        if last_reason in {'length', 'max_tokens'}:
            data['terminal_finish_reason'] = last_reason
            data['exit_status'] = 'incomplete'
            # CUA may use FAIL/DONE for an incomplete response. Neither is a
            # business conclusion when the provider reports output truncation.
            ctx.tool_state.pop('os_world_last_action', None)
        return normalize_agent_output(data)
    except Exception as exc:
        if last_error is not None and (
            last_error is exc
            or (
                ctx.agent_name == 'cua_anthropic_agent'
                and isinstance(exc, RuntimeError)
                and str(exc).startswith('Anthropic API failed too many times:')
            )
        ):
            result = classify_exception(last_error, stage='inference')
            ctx.tool_state['osworld_agent_output'] = result
            if result.exit_code == 0:
                return result
        raise
    finally:
        ctx.openai_clients = originals


@run_hook_registry.register('osworld_output_contract')
class OSWorldOutputContractHook(RunHook):
    """Project task-owned results without changing ordinary runner outputs."""

    task_name = 'os_world'
    summarize = staticmethod(summarize_task_output)

    async def on_run_output(self, ctx: Any, output: Any) -> None:
        if ctx.task_name != self.task_name:
            return
        from nexus.extensions.tasks.os_world.data.artifacts import RunArtifacts

        partial = getattr(ctx, 'tool_state', {}).pop('osworld_partial_task_output', None)
        if partial is not None and not output.success and output.task_output is None:
            output.task_output = partial
        publisher = getattr(ctx, 'tool_state', {}).pop('osworld_output_publisher', None)
        details = None
        if hasattr(output, 'model_dump'):
            # Processor results only exist after task publication. Use a separate
            # artifact archive, never mutate/reupload the immutable host archive.
            try:
                publisher = publisher or RunArtifacts()
                details = await publisher.publish_run_output_details(output)
            except Exception as exc:
                # Archival is ancillary, just like task artifact publication.
                # Do not turn a finished evaluation into an infrastructure error.
                details = {'status': 'archive_failed', 'upload_failed': True}
                logging.getLogger(__name__).warning('Runner detail archival failed (%s)', type(exc).__name__)
        task = output.task_output
        if task is not None:
            task = self.summarize(task)
            if hasattr(task, 'model_dump'):
                task = task.model_dump(mode='json')
            task.pop('instruction', None)
            task.pop('setup_results', None)
            reference = task.get('artifacts') or {}
            release_records = getattr(ctx, 'tool_state', {}).get('osworld_release_records', reference.get('release'))
            uris = reference.get('uris') or []
            # Large per-file manifests belong to publication.json, not RunOutput.
            reference = _result_fields(reference, ('status', 'uri', 'local_directory', 'upload_failed', 'attempts'))
            if release_records is not None:
                # Filled by provider teardown after task/runner archives have
                # been published. Keep the per-run list shared until run returns.
                reference['release'] = release_records
            reference['uris'] = [
                _result_fields({'uri': uri}, ('uri',))['uri'] for uri in uris[:16] if isinstance(uri, str)
            ]
            if details:
                reference['runner_details'] = _result_fields(
                    details, ('status', 'uri', 'local_directory', 'upload_failed', 'attempts')
                )
                if details.get('upload_failed'):
                    reference['upload_failed'] = True
                    if reference.get('status') == 'published':
                        reference['status'] = 'partial'
            task['artifacts'] = reference
            output.task_output = task
            output.agent_output = task.get('agent_output')
        else:
            output.agent_output = None
        if hasattr(output, 'processor_results'):
            output.processor_results = summarize_processors(output.processor_results)
            output.workflow_states = {}
            output.extra = {}
            output.cache_stats = None
            output.hook_results = {}
            output.hook_errors = {}
            for key in ('input_cache_task_ids', 'cache_task_ids'):
                value = getattr(output, key, {})
                setattr(output, key, _result_fields(value, tuple(list(value)[:16])))
            for key in ('input_cache_task_id', 'cache_task_id'):
                if hasattr(output, key):
                    setattr(output, key, _result_fields({key: getattr(output, key)}, (key,)).get(key))
            if output.error is not None:
                output.error = output.error.model_copy(
                    update={
                        **_result_fields(output.error, ('type', 'message')),
                        'traceback': None,
                    }
                )
            if task is None and details:
                output.extra = {
                    'artifacts': _result_fields(details, ('status', 'uri', 'local_directory', 'upload_failed'))
                }

    async def on_context_closing(self, ctx: Any) -> None:
        if ctx.task_name == self.task_name:
            ctx.tool_state.pop('osworld_release_records', None)
            ctx.tool_state.pop('osworld_partial_task_output', None)
            ctx.tool_state.pop('osworld_output_publisher', None)
