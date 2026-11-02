"""Shared append-only tool loop for Context Agent serving profiles.

The three public Context Agent variants differ in how they prepare documents,
build prompts, execute tools, and finalize an exhausted rollout.  Their normal
LLM/tool loop, however, has one important common contract: assistant messages
must be appended exactly as generated and every generated tool call must receive
a tool response in the same order.

Keeping that contract here prevents the serving profiles from slowly drifting
apart while still allowing each profile to retain its own prompt and tool
semantics.
"""

from __future__ import annotations

import asyncio
import json
import logging

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from nexus.core import ContextLengthExceededError
from nexus.extensions.agents.base.schema import TerminationReason


logger = logging.getLogger(__name__)

ExecuteToolCall = Callable[[Any, int, int], Awaitable[str]]
ToolResultHook = Callable[[int, int, Any, str], None]
MessageSerializer = Callable[[Any], dict[str, Any]]
SkippedResultFactory = Callable[[int, int], str]


@dataclass(slots=True)
class ToolLoopResult:
    """Result of the shared normal-turn tool loop."""

    final_answer: str | None
    final_message: Any | None
    tool_rounds_used: int
    tool_calls_requested: int
    tool_calls_executed: int
    tool_calls_skipped: int
    exit_status: TerminationReason
    response_budget_exhausted: bool = False


def safe_parse_json(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def tool_call_id(tool_call: Any, fallback: str) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get('id') or fallback)
    return str(getattr(tool_call, 'id', None) or fallback)


def tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        function = tool_call.get('function') or {}
        return str(function.get('name') or tool_call.get('name') or '')
    return str(getattr(getattr(tool_call, 'function', None), 'name', '') or '')


def tool_call_arguments(tool_call: Any) -> Any:
    if isinstance(tool_call, dict):
        return (tool_call.get('function') or {}).get('arguments', '')
    return getattr(getattr(tool_call, 'function', None), 'arguments', '')


def dump_tool_call(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        return tool_call
    return tool_call.model_dump(exclude_none=True)


def dump_assistant_message(message: Any) -> dict[str, Any]:
    """Return an assistant message without rebuilding its tool calls."""
    dumped = dict(message) if isinstance(message, dict) else message.model_dump(exclude_none=True)
    # HyTuner's vision extraction indexes message['content'], so tool-call-only turns keep the key.
    dumped.setdefault('content', None)
    return dumped


def default_skipped_tool_result(executed_count: int, _index: int) -> str:
    """Explain a skipped call while preserving tool-call/response alignment."""
    return (
        '[Error] Tool call was not executed because this response exceeded '
        f'the configured limit of {executed_count} parallel calls. '
        'Use the available results before requesting more tools.'
    )


def _finish_reason(choice: Any) -> str | None:
    """Return a normalized provider finish reason when one is available."""
    raw = getattr(choice, 'finish_reason', None)
    if raw is None:
        return None
    value = getattr(raw, 'value', raw)
    normalized = str(value).strip().lower()
    return normalized or None


def append_tool_results(
    conversation: list[dict[str, Any]],
    *,
    tool_calls: list[Any],
    outputs: list[str],
    round_num: int,
    respond_to_skipped: bool = True,
    fallback_id: Callable[[int, int], str] | None = None,
    skipped_result_factory: SkippedResultFactory = default_skipped_tool_result,
) -> None:
    """Append ordered tool responses for an assistant tool-call turn.

    ``outputs`` contains results for the executed prefix of ``tool_calls``.
    When a parallel-call cap skips the remaining calls, synthetic error
    responses keep the append-only chat history structurally valid.
    """
    executed_count = len(outputs)
    response_count = len(tool_calls) if respond_to_skipped else executed_count
    for index, tool_call in enumerate(tool_calls[:response_count]):
        content = outputs[index] if index < executed_count else skipped_result_factory(executed_count, index)
        default_id = fallback_id(round_num, index) if fallback_id is not None else f'call_{round_num}_{index}'
        conversation.append({
            'role': 'tool',
            'tool_call_id': tool_call_id(tool_call, default_id),
            'content': content,
        })


async def run_tool_call_loop(
    *,
    client: Any,
    conversation: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]] | None,
    max_tool_rounds: int,
    execute_tool: ExecuteToolCall,
    max_parallel_calls: int | None = None,
    respond_to_skipped: bool = True,
    serialize_message: MessageSerializer = dump_assistant_message,
    on_tool_result: ToolResultHook | None = None,
    fallback_id: Callable[[int, int], str] | None = None,
    skipped_result_factory: SkippedResultFactory = default_skipped_tool_result,
    catch_context_length: bool = True,
    raise_on_no_choices: bool = False,
    raise_on_empty_response: bool = False,
    log_label: str = 'context_agent',
) -> ToolLoopResult:
    """Run normal assistant/tool turns until an answer, error, or round limit.

    Exhaustion finalization remains profile-owned because the legacy/general
    profiles disable tools for that turn while the frozen SFT profile must keep
    its original schema.
    """
    round_num = 0
    requested = 0
    executed = 0
    skipped = 0
    parallel_limit = max_parallel_calls if max_parallel_calls and max_parallel_calls > 0 else None

    while round_num < max_tool_rounds:
        try:
            response = await client.create_completion(messages=conversation, tools=tool_schemas or None)
        except ContextLengthExceededError as exc:
            if not catch_context_length:
                raise
            logger.warning('%s: response budget exhausted at round %d: %s', log_label, round_num, exc)
            return ToolLoopResult(
                final_answer=None,
                final_message=None,
                tool_rounds_used=round_num,
                tool_calls_requested=requested,
                tool_calls_executed=executed,
                tool_calls_skipped=skipped,
                exit_status=TerminationReason.TRUNCATION,
                response_budget_exhausted=True,
            )
        except Exception as exc:
            logger.error('%s: round %d LLM call failed: %s', log_label, round_num, exc)
            raise

        if not response.choices:
            if raise_on_no_choices:
                raise RuntimeError(f'{log_label}: model returned no choices')
            logger.error('%s: no choices returned', log_label)
            return ToolLoopResult(
                final_answer=None,
                final_message=None,
                tool_rounds_used=round_num,
                tool_calls_requested=requested,
                tool_calls_executed=executed,
                tool_calls_skipped=skipped,
                exit_status=TerminationReason.UNHANDLED_ERROR,
            )

        choice = response.choices[0]
        message = choice.message
        finish_reason = _finish_reason(choice)
        tool_calls = list(message.tool_calls or [])

        if finish_reason == 'length':
            logger.warning('%s: generation stopped at the response budget at round %d', log_label, round_num)
            return ToolLoopResult(
                final_answer=None,
                final_message=message,
                tool_rounds_used=round_num,
                tool_calls_requested=requested,
                tool_calls_executed=executed,
                tool_calls_skipped=skipped,
                exit_status=TerminationReason.OUTPUT_LENGTH_EXCEEDED,
                response_budget_exhausted=True,
            )

        if finish_reason == 'content_filter':
            logger.error('%s: generation was stopped by the provider content filter at round %d', log_label, round_num)
            return ToolLoopResult(
                final_answer=None,
                final_message=message,
                tool_rounds_used=round_num,
                tool_calls_requested=requested,
                tool_calls_executed=executed,
                tool_calls_skipped=skipped,
                exit_status=TerminationReason.PROVIDER_ERROR,
            )

        if finish_reason == 'stop':
            if tool_calls:
                logger.warning(
                    '%s: finish_reason=%r takes precedence over %d unexpected tool call(s) at round %d',
                    log_label,
                    finish_reason,
                    len(tool_calls),
                    round_num,
                )

            final_answer = (message.content or '').strip()
            if not final_answer:
                if raise_on_empty_response:
                    raise RuntimeError(f'{log_label}: model returned an empty final response')
                logger.warning('%s: empty stop response', log_label)
                return ToolLoopResult(
                    final_answer=None,
                    final_message=message,
                    tool_rounds_used=round_num,
                    tool_calls_requested=requested,
                    tool_calls_executed=executed,
                    tool_calls_skipped=skipped,
                    exit_status=TerminationReason.FORMAT_ERROR,
                )

            conversation.append(serialize_message(message))
            logger.info('%s: final answer (%d chars) at round %d', log_label, len(final_answer), round_num)
            return ToolLoopResult(
                final_answer=final_answer,
                final_message=message,
                tool_rounds_used=round_num,
                tool_calls_requested=requested,
                tool_calls_executed=executed,
                tool_calls_skipped=skipped,
                exit_status=TerminationReason.SUCCESS,
            )

        if finish_reason in {'tool_calls', 'function_call'} and not tool_calls:
            logger.error(
                '%s: finish_reason=%r but the response contained no tool calls at round %d',
                log_label,
                finish_reason,
                round_num,
            )
            return ToolLoopResult(
                final_answer=None,
                final_message=message,
                tool_rounds_used=round_num,
                tool_calls_requested=requested,
                tool_calls_executed=executed,
                tool_calls_skipped=skipped,
                exit_status=TerminationReason.FORMAT_ERROR,
            )

        if not tool_calls:
            if finish_reason is not None:
                logger.error(
                    '%s: unrecognized finish_reason=%r with no tool calls at round %d',
                    log_label,
                    finish_reason,
                    round_num,
                )
                return ToolLoopResult(
                    final_answer=None,
                    final_message=message,
                    tool_rounds_used=round_num,
                    tool_calls_requested=requested,
                    tool_calls_executed=executed,
                    tool_calls_skipped=skipped,
                    exit_status=TerminationReason.FORMAT_ERROR,
                )

            final_answer = (message.content or '').strip()
            if not final_answer:
                if raise_on_empty_response:
                    raise RuntimeError(f'{log_label}: model returned an empty final response')
                logger.warning('%s: empty response with no tool calls', log_label)
                return ToolLoopResult(
                    final_answer=None,
                    final_message=message,
                    tool_rounds_used=round_num,
                    tool_calls_requested=requested,
                    tool_calls_executed=executed,
                    tool_calls_skipped=skipped,
                    exit_status=TerminationReason.FORMAT_ERROR,
                )

            # Compatibility path for older/fake clients that do not expose a
            # finish reason. Real providers should end text responses with
            # finish_reason='stop'.
            conversation.append(serialize_message(message))
            logger.info('%s: final answer (%d chars) at round %d', log_label, len(final_answer), round_num)
            return ToolLoopResult(
                final_answer=final_answer,
                final_message=message,
                tool_rounds_used=round_num,
                tool_calls_requested=requested,
                tool_calls_executed=executed,
                tool_calls_skipped=skipped,
                exit_status=TerminationReason.SUCCESS,
            )

        if finish_reason is not None and finish_reason not in {'tool_calls', 'function_call'}:
            logger.warning(
                '%s: refusing %d tool call(s) with unrecognized finish_reason=%r',
                log_label,
                len(tool_calls),
                finish_reason,
            )
            return ToolLoopResult(
                final_answer=None,
                final_message=message,
                tool_rounds_used=round_num,
                tool_calls_requested=requested,
                tool_calls_executed=executed,
                tool_calls_skipped=skipped,
                exit_status=TerminationReason.FORMAT_ERROR,
            )

        conversation.append(serialize_message(message))
        selected = tool_calls if parallel_limit is None else tool_calls[:parallel_limit]
        requested += len(tool_calls)
        executed += len(selected)
        skipped += len(tool_calls) - len(selected)

        outputs = await asyncio.gather(*[
            execute_tool(tool_call, round_num, index) for index, tool_call in enumerate(selected)
        ])
        if on_tool_result is not None:
            for index, (tool_call, output) in enumerate(zip(selected, outputs, strict=True)):
                on_tool_result(round_num, index, tool_call, output)

        append_tool_results(
            conversation,
            tool_calls=tool_calls,
            outputs=list(outputs),
            round_num=round_num,
            respond_to_skipped=respond_to_skipped,
            fallback_id=fallback_id,
            skipped_result_factory=skipped_result_factory,
        )
        round_num += 1

    return ToolLoopResult(
        final_answer=None,
        final_message=None,
        tool_rounds_used=round_num,
        tool_calls_requested=requested,
        tool_calls_executed=executed,
        tool_calls_skipped=skipped,
        exit_status=TerminationReason.MAX_ITERATIONS,
    )


__all__ = [
    'ToolLoopResult',
    'append_tool_results',
    'default_skipped_tool_result',
    'dump_assistant_message',
    'dump_tool_call',
    'run_tool_call_loop',
    'safe_parse_json',
    'tool_call_arguments',
    'tool_call_id',
    'tool_call_name',
]
