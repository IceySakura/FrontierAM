"""S2T Search agent for the yuanbao-reward glue project.

Same multi-turn OpenAI function-calling loop as ``search_s2t_rl`` (the trainable
policy is the ``main`` client; the agent exposes a single search tool), extended
to:

* accept multi-modal (audio) starting messages and embed ``<audio>`` patches, and
* surface the fields the yuanbao ``agts_closed_qa`` reward needs
  (``tool_call_count`` / ``search_count`` / ``valid_search_count`` / ``final_answer``).

Only ``search_s2t_yb_glue/search`` is registered — the yuanbao
``web_search`` / ``picture_search`` / ``video_search`` / ``finance_search``
backends are intentionally not used.
"""

import asyncio
import json
import logging
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ContextLengthExceededError,
    Registry,
    ToolExecutor,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)
from nexus.extensions.projects.search_s2t_rl_yb_glue.conversation import build_agent_messages
from nexus.trace import get_current_span_id, span_context, update_span


logger = logging.getLogger(__name__)

SEARCH_TOOL_NAME = 'search_s2t_yb_glue/search'
# Names the model may emit for the search tool: the canonical registry key, the
# model-visible schema name (``web_search``, per sp.md) and the legacy ``search``
# alias. ``search`` is not in the schema and ToolExecutor cannot resolve it, so it
# only ever produces an error observation -- but it must stay listed: dropping it
# would stop it from spending the trajectory's single search slot, which lets a
# sibling web_search in the same message run instead of being rejected over budget,
# flipping that trajectory's reward from the -1.0 format gate to a judge score.
SEARCH_TOOL_NAMES = {SEARCH_TOOL_NAME, 'web_search', 'search'}

# Project-level safety caps. Per-row configs may request a smaller value, but
# legacy datasets carrying the old 10-iteration / 5-search settings are clamped
# here so the hard protocol also applies before those parquet files are rebuilt.
MAX_ITERATIONS = 2
MAX_SEARCH_CALLS = 1
MAX_SUBQUERIES_PER_CALL = 3
MAX_TOTAL_OBSERVATION_TOKENS = 10000
MIN_OBSERVATION_BUDGET_TO_ENABLE_TOOL = 64
OBSERVATION_TRUNCATION_SUFFIX = '\n...[tool observation truncated to the trajectory token budget]'

DEFAULT_SYSTEM_PROMPT = (
    'You are a helpful assistant that can call a web `search` tool when external information is needed.'
)

# Register the vendored PromptSearch tool on import.
Registry.import_module('nexus.extensions.projects.search_s2t_rl_yb_glue.tools')


class SearchS2TExitStatus(str, Enum):
    """How the agent loop ended. Reported for metrics only.

    The HyTuner ``termination_reason`` is deliberately left to the runner, which
    already derives it the way this project relies on: a normal ending defaults to
    ``ENV_DONE``, a propagating ``TimeoutError`` becomes ``TASK_TIMEOUT`` and any
    other exception becomes ``UNHANDLED_ERROR``. Writing ``agent.termination_reason``
    from here would override that and change which trajectories get masked out.
    """

    FINISHED = 'finished'
    MAX_ITERATIONS = 'max_iterations'
    TOOL_BUDGET_VIOLATION = 'tool_budget_violation'


@dataclass
class _TrajectoryCounters:
    """Per-trajectory counters surfaced to the reward through the task output."""

    tool_call_count: int = 0
    search_count: int = 0
    valid_search_count: int = 0
    web_len: int = 0
    subquery_count: int = 0
    executed_subquery_count: int = 0
    overlimit_subquery_count: int = 0
    malformed_tool_calls: int = 0
    rejected_tool_calls: int = 0
    overlimit_search_calls: int = 0
    tool_observation_tokens: int = 0
    truncated_tool_observations: int = 0


class SearchS2TYBAgentInput(AgentRunInput, WithUserPrompt):
    max_iterations: int = MAX_ITERATIONS
    max_search_calls: int = MAX_SEARCH_CALLS
    max_subqueries_per_call: int = MAX_SUBQUERIES_PER_CALL
    max_total_observation_tokens: int = MAX_TOTAL_OBSERVATION_TOKENS
    system_prompt: str | None = None
    messages: list[dict] | None = None
    audios: list[dict] | None = None
    # Per-row cutoff date for the search backend (from the parquet `time_range`
    # column). When set, it is forwarded into ExecutionContext.tool_configs so the
    # vendored PromptSearch tool applies the `time_range` filter. Empty = no filter.
    time_range: str = ''


class SearchS2TYBAgentOutput(AgentRunOutput, WithMainTrajectory):
    final_answer: str | None = Field(default=None, description='Last assistant message content.')
    tool_call_count: int = Field(default=0, description='Total tool calls executed by the agent.')
    search_count: int = Field(default=0, description='Number of search calls accepted by the execution gate.')
    valid_search_count: int = Field(
        default=0,
        description='Search calls that reached the tool with at least one non-empty subquery.',
    )
    web_len: int = Field(default=0, description='Total documents returned across successful searches.')
    subquery_count: int = Field(default=0, description='Subqueries attempted before the per-call hard cap.')
    executed_subquery_count: int = Field(default=0, description='Subqueries forwarded after the per-call hard cap.')
    overlimit_subquery_count: int = Field(default=0, description='Attempted subqueries removed by the hard cap.')
    malformed_tool_calls: int = Field(default=0, description='Tool calls whose JSON arguments could not be parsed.')
    rejected_tool_calls: int = Field(default=0, description='Tool calls rejected by execution hard limits.')
    overlimit_search_calls: int = Field(default=0, description='Rejected search calls beyond the hard limit.')
    tool_observation_tokens: int = Field(
        default=0,
        description='Exact tokenizer count of tool messages added after the initial prompt.',
    )
    truncated_tool_observations: int = Field(default=0, description='Tool observations truncated to the token budget.')
    iterations: int = Field(default=0, description='Number of agent iterations executed.')
    exit_status: str = Field(default='', description='How the agent loop ended (SearchS2TExitStatus value).')


def _tool_output_to_text(output) -> str:
    if isinstance(output, list):
        return ''.join(item.get('text', '') for item in output if isinstance(item, dict) and item.get('type') == 'text')
    return output if isinstance(output, str) else str(output)


def _is_valid_search_response(tool_name: str, tool_response: Any) -> bool:
    """Whether a returned search response represents at least one usable query.

    The search tool deliberately returns a ``ToolResponse`` for both backend
    failures and empty results so the policy can still finish its answer. Those
    attempts remain valid when at least one non-empty query reached the tool.
    A whitespace-only query is cleaned to zero subqueries and must not unlock the
    required-search reward.
    """
    if tool_name not in SEARCH_TOOL_NAMES:
        return False
    meta = getattr(tool_response, 'meta', None)
    if not isinstance(meta, dict):
        return False
    try:
        return int(meta.get('subquery_count', 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _decode_token_prefix(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


async def _fit_tool_observation_to_budget(
    client: Any,
    history: list[dict[str, Any]],
    *,
    tool_call_id: str,
    content: str,
    tools: list[dict[str, Any]] | None,
    remaining_tokens: int,
) -> tuple[str, int, bool]:
    """Fit one complete role=tool message into the remaining trajectory budget.

    HyTunerClient.count_tokens applies the same chat template and tokenizer as
    rollout. Measuring the before/after prompt delta therefore includes the tool
    role wrapper and generation prompt, not merely a character approximation.
    """

    def tool_message(text: str) -> dict[str, Any]:
        return {'role': 'tool', 'tool_call_id': tool_call_id, 'content': text}

    async def count_with_base(text: str, base_tokens: int) -> int:
        result = await client.count_tokens(messages=[*history, tool_message(text)], tools=tools)
        return max(0, int(result.input_tokens) - base_tokens)

    if remaining_tokens <= 0:
        return '', 0, bool(content)

    try:
        base = await client.count_tokens(messages=history, tools=tools)
        base_tokens = int(base.input_tokens)
        full_tokens = await count_with_base(content, base_tokens)
        if full_tokens <= remaining_tokens:
            return content, full_tokens, False

        empty_tokens = await count_with_base('', base_tokens)
        if empty_tokens > remaining_tokens:
            # This should be rare in normal training because tool execution is
            # disabled on the next iteration once the remaining budget is below
            # MIN_OBSERVATION_BUDGET_TO_ENABLE_TOOL.
            logger.warning(
                'Tool message wrapper requires %s tokens but only %s remain; emitting empty observation',
                empty_tokens,
                remaining_tokens,
            )
            return '', empty_tokens, True

        tokenizer = getattr(client, 'tokenizer', None)
        if tokenizer is not None:
            source_ids = list(tokenizer.encode(content, add_special_tokens=False))

            def prefix_text(size: int, *, with_suffix: bool = True) -> str:
                prefix = _decode_token_prefix(tokenizer, source_ids[:size])
                return f'{prefix}{OBSERVATION_TRUNCATION_SUFFIX}' if with_suffix else prefix

            # Prefer an explicit truncation marker when it fits; otherwise use a
            # raw token prefix. Binary search is over source-token count, while
            # count_tokens verifies the fully rendered message at each probe.
            marker_tokens = await count_with_base(OBSERVATION_TRUNCATION_SUFFIX, base_tokens)
            use_suffix = marker_tokens <= remaining_tokens
            low, high = 0, len(source_ids)
            best_text = ''
            best_tokens = empty_tokens
            while low <= high:
                mid = (low + high) // 2
                candidate = prefix_text(mid, with_suffix=use_suffix)
                candidate_tokens = await count_with_base(candidate, base_tokens)
                if candidate_tokens <= remaining_tokens:
                    best_text, best_tokens = candidate, candidate_tokens
                    low = mid + 1
                else:
                    high = mid - 1
            return best_text, best_tokens, True

        # Non-HyTuner clients may implement count_tokens without exposing their
        # tokenizer. Fall back to a character-prefix binary search, but still use
        # the provider's exact count_tokens result as the acceptance criterion.
        low, high = 0, len(content)
        best_text = ''
        best_tokens = empty_tokens
        while low <= high:
            mid = (low + high) // 2
            candidate = f'{content[:mid]}{OBSERVATION_TRUNCATION_SUFFIX}'
            candidate_tokens = await count_with_base(candidate, base_tokens)
            if candidate_tokens <= remaining_tokens:
                best_text, best_tokens = candidate, candidate_tokens
                low = mid + 1
            else:
                high = mid - 1
        return best_text, best_tokens, True
    except NotImplementedError as exc:
        raise RuntimeError(
            'max_total_observation_tokens requires an OpenAI client with exact count_tokens support'
        ) from exc


async def _append_tool_observation(
    client: Any,
    history: list[dict[str, Any]],
    counters: _TrajectoryCounters,
    *,
    tool_call_id: str,
    content: str,
    tools: list[dict[str, Any]] | None,
    max_total_observation_tokens: int,
) -> float:
    """Fit one observation into the remaining budget, append it, update counters.

    Returns the seconds spent fitting so the caller can bill it to observation
    time — the fit runs a binary search over ``count_tokens`` and is otherwise
    indistinguishable from the search RPC in the timeline.
    """
    started = time.monotonic()
    with span_context('agent.observation.token_fit', attributes={'type': 'agent_observation_token_fit'}):
        output, observation_tokens, was_truncated = await _fit_tool_observation_to_budget(
            client,
            history,
            tool_call_id=tool_call_id,
            content=content,
            tools=tools,
            remaining_tokens=max_total_observation_tokens - counters.tool_observation_tokens,
        )
    counters.tool_observation_tokens += observation_tokens
    counters.truncated_tool_observations += int(was_truncated)
    history.append({'role': 'tool', 'tool_call_id': tool_call_id, 'content': output})
    return time.monotonic() - started


async def _execute_tool_call(
    *,
    client: Any,
    executor: ToolExecutor,
    history: list[dict[str, Any]],
    counters: _TrajectoryCounters,
    tool_call: Any,
    tools: list[dict[str, Any]] | None,
    max_search_calls: int,
    max_subqueries_per_call: int,
    max_total_observation_tokens: int,
) -> float:
    """Run one model-emitted tool call and append its paired observation.

    Every failure is fed back to the model as an error observation and the loop
    continues, which is what the reward gates are calibrated against. Returns the
    seconds spent fitting observations into the token budget.
    """
    tool_name = tool_call.function.name
    fit_seconds = 0.0

    async def append_observation(content: str) -> None:
        nonlocal fit_seconds
        fit_seconds += await _append_tool_observation(
            client,
            history,
            counters,
            tool_call_id=tool_call.id,
            content=content,
            tools=tools,
            max_total_observation_tokens=max_total_observation_tokens,
        )

    try:
        tool_args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        counters.malformed_tool_calls += 1
        error_msg = f'Failed to parse tool arguments: {tool_call.function.arguments}'
        logger.warning(error_msg)
        await append_observation(f'Error: {error_msg}')
        return fit_seconds

    logger.debug(f'Tool call: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})')
    is_search_call = tool_name in SEARCH_TOOL_NAMES

    # Apply the per-call subquery cap before executing. The tool input model caps
    # the list too, so validating first would reject an over-long query list
    # instead of truncating and running it.
    if is_search_call:
        if not isinstance(tool_args, dict):
            # Valid JSON that is not an object. Raising keeps the historical outcome
            # (the runner marks the run UNHANDLED_ERROR and masks the trajectory)
            # while replacing the AttributeError that the lookup below used to raise
            # with a message that says what happened. The check stays inside this
            # branch because any other tool name still has to fall through to
            # ToolExecutor.call, which turns it into an error observation.
            raise TypeError(
                f'Tool call {tool_name} produced non-object arguments '
                f'({type(tool_args).__name__}): {tool_call.function.arguments}'
            )
        raw_queries = tool_args.get('query')
        if isinstance(raw_queries, list):
            attempted_count = len(raw_queries)
            counters.subquery_count += attempted_count
            if attempted_count > max_subqueries_per_call:
                counters.overlimit_subquery_count += attempted_count - max_subqueries_per_call
                tool_args = dict(tool_args)
                tool_args['query'] = raw_queries[:max_subqueries_per_call]

    if is_search_call and counters.search_count >= max_search_calls:
        counters.rejected_tool_calls += 1
        counters.overlimit_search_calls += 1
        error_msg = f'Search call limit exceeded: max_search_calls={max_search_calls}'
        logger.warning(error_msg)
        await append_observation(f'Error: {error_msg}')
        return fit_seconds

    # The search quota is spent before the call is validated. ToolExecutor.call
    # resolves the name and validates the arguments before it reaches the backend,
    # so an invalid call has always consumed the quota and left any sibling call in
    # the same message to be rejected over budget -- which the reward gates score as
    # a format failure. Validating up front to spare the quota would turn that
    # trajectory's -1.0 into a judge score.
    if is_search_call:
        counters.search_count += 1
        queries_to_execute = tool_args.get('query')
        if isinstance(queries_to_execute, list):
            counters.executed_subquery_count += len(queries_to_execute)

    try:
        tool_response = await executor.call(tool_name, tool_args)
        counters.tool_call_count += 1
        counters.valid_search_count += int(_is_valid_search_response(tool_name, tool_response))
        output = _tool_output_to_text(tool_response.output)
        meta = getattr(tool_response, 'meta', None) or {}
        counters.web_len += int(meta.get('result_count', 0) or 0)
        if is_search_call and meta.get('give_up'):
            # The search tool never raises: transport and HTTP failures that
            # survive every retry come back as a ToolResponse carrying give_up,
            # with the fixed empty-result text the model can still answer from.
            logger.warning('Search backend gave up after retries: %s', meta.get('error'))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error_msg = f'Error executing tool {tool_name}: {exc}'
        logger.warning(error_msg)
        output = f'Error: {error_msg}'

    logger.debug(f'Tool output: {output}')
    await append_observation(output)
    return fit_seconds


@agent_registry.register('search_s2t_agent_yb_glue')
async def run_search_s2t_agent_yb_glue(agent_input: SearchS2TYBAgentInput) -> SearchS2TYBAgentOutput:
    ctx = require_context()
    client = ctx.openai_client

    executor = ToolExecutor([tool_registry.get(SEARCH_TOOL_NAME)])

    # Per-row cutoff date: push it into the ExecutionContext tool_configs so the
    # vendored search tool reads it via context.tool_configs['search_s2t_yb_glue']
    # (the same channel run-config uses). Empty time_range leaves tool_configs
    # untouched (no filter). ExecutionContext.tool_configs is a plain dict (only
    # the derived ToolContext is frozen), so this per-run write is safe.
    # We ALWAYS log a one-liner (empty or not) so the operator can grep the
    # worker log to confirm the parquet field really made it here per rollout.
    if agent_input.time_range:
        _ctx = require_context()
        _tc = _ctx.tool_configs.setdefault('search_s2t_yb_glue', {})
        _tc['time_range'] = agent_input.time_range
        logger.debug('[time_range] applied=%r (injected into tool_configs)', agent_input.time_range)
    else:
        logger.debug('[time_range] applied=%r (empty -> no time filter)', agent_input.time_range)

    system_prompt = agent_input.system_prompt or DEFAULT_SYSTEM_PROMPT
    if agent_input.messages:
        history = build_agent_messages(
            agent_input.messages,
            agent_input.audios,
            default_system_prompt=system_prompt,
        )
    else:
        history = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': agent_input.user_prompt},
        ]

    counters = _TrajectoryCounters()
    observation_seconds = 0.0
    observation_fit_seconds = 0.0
    iterations = 0
    exit_status = SearchS2TExitStatus.MAX_ITERATIONS

    max_iterations = max(1, min(int(agent_input.max_iterations), MAX_ITERATIONS))
    max_search_calls = max(0, min(int(agent_input.max_search_calls), MAX_SEARCH_CALLS))
    max_subqueries_per_call = max(
        1,
        min(int(agent_input.max_subqueries_per_call), MAX_SUBQUERIES_PER_CALL),
    )
    max_total_observation_tokens = max(
        MIN_OBSERVATION_BUDGET_TO_ENABLE_TOOL,
        min(int(agent_input.max_total_observation_tokens), MAX_TOTAL_OBSERVATION_TOKENS),
    )
    tool_schemas = executor.get_tool_schemas() or None
    # Nexus's generic schema simplifier currently drops maxItems/minItems from
    # Pydantic JSON schema. Restore them on the model-visible schema while the
    # Pydantic model and agent-side slice independently enforce the same cap.
    if tool_schemas:
        for schema in tool_schemas:
            function = schema.get('function', {})
            if function.get('name') != 'web_search':
                continue
            query_schema = function.get('parameters', {}).get('properties', {}).get('query')
            if isinstance(query_schema, dict):
                query_schema['minItems'] = 1
                query_schema['maxItems'] = max_subqueries_per_call

    # HyTuner incrementally tokenizes a trajectory and caches the rendered
    # prefix after every completion. Tool schemas participate in that render,
    # so changing ``tools`` from a schema list to ``None`` mid-trajectory would
    # rewrite the cached prefix and invalidate its token alignment. Pick the
    # model-visible schema once and keep it stable for the whole trajectory;
    # the per-iteration hard limits below independently gate tool execution.
    trajectory_tool_schemas = tool_schemas if max_iterations > 1 and max_search_calls > 0 else None

    for iteration in range(max_iterations):
        iterations = iteration + 1
        with span_context('agent.iteration', attributes={'type': 'agent_iteration', 'iteration': iterations}):
            remaining_observation_tokens = max_total_observation_tokens - counters.tool_observation_tokens
            tool_execution_enabled = (
                iteration < max_iterations - 1
                and counters.search_count < max_search_calls
                and remaining_observation_tokens >= MIN_OBSERVATION_BUDGET_TO_ENABLE_TOOL
            )

            try:
                with span_context(
                    'agent.inference',
                    attributes={'type': 'agent_inference', 'iteration': iterations},
                ):
                    response = await client.create_completion(messages=history, tools=trajectory_tool_schemas)
            # Inference failures are logged with their classification and then
            # re-raised. The runner already maps a propagating TimeoutError to
            # TASK_TIMEOUT and everything else to UNHANDLED_ERROR, both of which
            # HyTuner masks out; handling them here instead would keep the run
            # "successful" and spend a reward judge call on a broken trajectory.
            except ContextLengthExceededError:
                logger.warning('Inference failed at iteration %s: context length exceeded', iterations)
                raise
            except asyncio.CancelledError:
                # Rollout abort and the runner-level task timeout both arrive as
                # cancellation; swallowing it would break both.
                raise
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning('Inference failed at iteration %s: timeout', iterations)
                raise
            except Exception:
                logger.exception('Inference failed at iteration %s', iterations)
                raise

            if not response.choices:
                logger.error('No choices returned in the response')
                break

            message = response.choices[0].message
            history.append(message.model_dump(exclude_none=True))

            if message.content:
                logger.debug(f'Assistant: {message.content}')

            if not message.tool_calls:
                exit_status = SearchS2TExitStatus.FINISHED
                break

            # Tool schemas stay model-visible to preserve HyTuner's cached prefix,
            # but execution remains strictly disabled after either hard limit is
            # reached. Ignore a leaked/repeated tool call instead of running it.
            if not tool_execution_enabled:
                counters.rejected_tool_calls += len(message.tool_calls)
                counters.overlimit_search_calls += sum(
                    tool_call.function.name in SEARCH_TOOL_NAMES for tool_call in message.tool_calls
                )
                logger.warning('Ignoring tool calls emitted after search/observation budget was exhausted')
                exit_status = SearchS2TExitStatus.TOOL_BUDGET_VIOLATION
                break

            for tool_call in message.tool_calls:
                observation_started = time.monotonic()
                with span_context(
                    'agent.observation',
                    attributes={
                        'type': 'agent_observation',
                        'iteration': iterations,
                        'tool_name': tool_call.function.name,
                    },
                ):
                    fit_seconds = await _execute_tool_call(
                        client=client,
                        executor=executor,
                        history=history,
                        counters=counters,
                        tool_call=tool_call,
                        tools=trajectory_tool_schemas,
                        max_search_calls=max_search_calls,
                        max_subqueries_per_call=max_subqueries_per_call,
                        max_total_observation_tokens=max_total_observation_tokens,
                    )
                observation_seconds += time.monotonic() - observation_started
                observation_fit_seconds += fit_seconds

    final_answer = next(
        (msg['content'] for msg in reversed(history) if msg.get('role') == 'assistant' and msg.get('content')),
        None,
    )
    if isinstance(final_answer, list):
        final_answer = ''.join(
            part.get('text', '') for part in final_answer if isinstance(part, dict) and part.get('type') == 'text'
        )

    logger.info(
        'Agent finished: exit_status=%s iterations=%s search_count=%s',
        exit_status.value,
        iterations,
        counters.search_count,
    )

    span_id = get_current_span_id()
    if span_id:
        try:
            # HytunerProcessor sums observation_time from span attributes; it is
            # written once for the whole run. termination_reason is intentionally
            # not written so the runner keeps deciding it.
            update_span(
                span_id,
                attributes={
                    'agent.exit_status': exit_status.value,
                    'agent.observation_time': observation_seconds,
                    'agent.observation_token_fit_time': observation_fit_seconds,
                    'agent.iterations': iterations,
                    'agent.tool_call_count': counters.tool_call_count,
                    'agent.search_count': counters.search_count,
                    'agent.valid_search_count': counters.valid_search_count,
                    'agent.malformed_tool_calls': counters.malformed_tool_calls,
                    'agent.rejected_tool_calls': counters.rejected_tool_calls,
                },
            )
        except Exception:
            logger.warning('Failed to update span attributes for agent', exc_info=True)

    return SearchS2TYBAgentOutput(
        main_trajectory=history,
        final_answer=final_answer,
        tool_call_count=counters.tool_call_count,
        search_count=counters.search_count,
        valid_search_count=counters.valid_search_count,
        web_len=counters.web_len,
        subquery_count=counters.subquery_count,
        executed_subquery_count=counters.executed_subquery_count,
        overlimit_subquery_count=counters.overlimit_subquery_count,
        malformed_tool_calls=counters.malformed_tool_calls,
        rejected_tool_calls=counters.rejected_tool_calls,
        overlimit_search_calls=counters.overlimit_search_calls,
        tool_observation_tokens=counters.tool_observation_tokens,
        truncated_tool_observations=counters.truncated_tool_observations,
        iterations=iterations,
        exit_status=exit_status.value,
    )
