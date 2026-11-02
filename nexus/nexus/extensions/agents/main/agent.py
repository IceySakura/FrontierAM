"""Main Agent — 通用的 tool-calling agent 循环。

通过 ``tools`` 参数配置任意工具组合，支持顺序/并行工具执行。
当模型不再调用工具时，循环结束。

指标通过 span 体系采集：
* ``agent.main_agent``    — 总 span，携带 exit_status / iterations / counts
* ``agent.iteration``     — 每轮迭代的 span
* ``agent.inference``     — 每次 LLM 调用的 span
* ``agent.observation``   — 每次工具执行阶段的 span
* ``agent.compaction``    — 每次 context management compaction 检查/执行的 span
配合 ``MainAgentMetricsProcessor`` 可在 ``RunOutput.processor_results`` 中获取
全部指标（含 per-step 耗时）。
"""

import asyncio
import json
import logging
import os
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeGuard

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, ConfigDict, Field

import nexus.extensions.agents.main.tools  # noqa: F401

from nexus.core import (
    UNPARSED_TOOL_CALL_FIELD,
    AgentRunInput,
    AgentRunOutput,
    ContentPartParam,
    ContextLengthExceededError,
    EntryNotFoundError,
    OpenAICompatibleClient,
    ToolArgumentValidationError,
    ToolExecutor,
    ToolNameResolutionError,
    ToolResponse,
    WithHistory,
    WithMainTrajectory,
    WithToolSearch,
    WithUserPrompt,
    agent_registry,
    as_tokenizer_panic_error,
    import_from_path,
    require_context,
    tool_registry,
    validate_history,
)
from nexus.core.tool import ToolParamOverride
from nexus.extensions.agents.main.multi_agent import (
    MULTI_AGENT_CONTROLLER_STATE_KEY,
    MULTI_AGENT_SUMMARY_STATE_KEY,
    SPAWN_AGENT_TOOL_NAME,
    WAIT_AGENT_TOOL_NAME,
    AgentExecutionIdentity,
    MultiAgentConfig,
    SubagentController,
)
from nexus.extensions.agents.main.utils import (
    extract_tool_output_content,
    extract_tool_output_text,
    fmt_msg,
    log_messages,
    maybe_truncate,
    minimum_tool_response_token_budget,
    truncate_tool_response_by_tokens,
)
from nexus.extensions.compaction.compaction_manager import CompactionManager, count_tokens_with_context_client
from nexus.runtime.client.base import RuntimeTransportError, RuntimeUnhealthyError
from nexus.trace import get_current_span_id, span_context, update_span
from nexus.utils.timeout import (
    InferenceTimeoutError,
    TimeBudget,
    await_completion_with_time_budget,
    with_timeout,
)
from nexus.utils.tool_schema_overrides import load_tool_schema_overrides


logger = logging.getLogger(__name__)

_UNOBSERVED_SUBAGENT_CONTINUATION: str = (
    'One or more spawned subagents have not been observed through wait_agent. '
    'Continue the task and call wait_agent before returning the final answer.'
)
_FULL_SUBAGENT_CAPACITY_CONTINUATION: str = (
    'The current spawned-agent capacity is full. Perform any requested independent main-agent work, then call '
    'wait_agent. Do not call spawn_agent again until wait_agent has observed a child.'
)


class _AgentForkingClient(Protocol):
    def fork_for_agent(self, *, agent_id: str, parent_agent_id: str | None) -> OpenAICompatibleClient: ...


def _supports_agent_fork(client: object) -> TypeGuard[_AgentForkingClient]:
    return callable(getattr(client, 'fork_for_agent', None))


# ---------------------------------------------------------------------------
# Schema: exit status, input / output
# ---------------------------------------------------------------------------


class MainExitStatus(str, Enum):
    FINISHED = 'finished'
    REFUSED = 'refused'
    MAX_ITERATIONS = 'max_iterations'
    TOOL_ERROR = 'tool_error'
    TOOL_EXECUTION_ERROR = 'tool_execution_error'
    INFERENCE_ERROR = 'inference_error'
    CONTEXT_LENGTH_EXCEEDED = 'context_length_exceeded'
    OUTPUT_LENGTH_EXCEEDED = 'output_length_exceeded'
    INFERENCE_TIMEOUT = 'inference_timeout'
    OBSERVATION_TIMEOUT = 'observation_timeout'
    RUNTIME_ERROR = 'runtime_error'

    @property
    def termination_reason(self) -> str:
        match self:
            case MainExitStatus.FINISHED:
                return 'ENV_DONE'
            case MainExitStatus.REFUSED:
                return 'ENV_DONE'
            case MainExitStatus.MAX_ITERATIONS:
                return 'MAX_STEPS'
            case MainExitStatus.TOOL_ERROR:
                return 'FORMAT_ERROR'
            case MainExitStatus.TOOL_EXECUTION_ERROR:
                return 'ENV_ERROR'
            case MainExitStatus.INFERENCE_ERROR:
                return 'UNHANDLED_ERROR'
            case MainExitStatus.CONTEXT_LENGTH_EXCEEDED:
                return 'TRUNCATION'
            case MainExitStatus.OUTPUT_LENGTH_EXCEEDED:
                return 'OUTPUT_LENGTH_EXCEEDED'
            case MainExitStatus.INFERENCE_TIMEOUT:
                return 'TASK_TIMEOUT'
            case MainExitStatus.OBSERVATION_TIMEOUT:
                return 'TASK_TIMEOUT'
            case MainExitStatus.RUNTIME_ERROR:
                return 'UNHANDLED_ERROR'

    @classmethod
    def termination_reason_for(cls, value: str) -> str:
        try:
            return cls(value).termination_reason
        except ValueError:
            return 'UNKNOWN'


@dataclass
class PostIterationHookResult:
    """Result from a post-iteration hook.

    Attributes:
        tool_messages: If set, replaces the original tool messages for this iteration.
        extra_messages: Additional messages appended after tool messages
            (e.g., auto-screenshot as a user message with an image).
        next_completion_kwargs: kwargs merged into the next ``create_completion`` call only, then cleared.
            Valid keys depend on the client (e.g. ``sampling_params`` for HyTunerClient).
    """

    tool_messages: list[dict[str, Any]] | None = None
    extra_messages: list[dict[str, Any]] | None = None
    next_completion_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class PostIterationHookContext:
    """Read-only context passed to a post-iteration hook.

    The tuple containers are immutable, but the dicts inside are the loop's live
    messages (not copied). Don't mutate them in place — return a
    PostIterationHookResult and copy any dict you rewrite (e.g.
    [dict(m) for m in ctx.tool_messages]).
    """

    iteration: int
    max_iterations: int
    # Conversation after this round's assistant response, before its tool messages.
    history: tuple[dict[str, Any], ...]
    # This round's tool observations, pre-rewrite.
    tool_messages: tuple[dict[str, Any], ...]


class MainAgentToolOverride(BaseModel):
    tool_name: str
    name: str | None = None
    description: str | None = None
    params: dict[str, str | ToolParamOverride] | None = None


class MainAgentCompactionConfig(BaseModel):
    """Context-management compaction owned by main_agent; mirrors ``CompactionManager`` arguments."""

    model_config = ConfigDict(extra='forbid')

    compaction_strategy: str = 'keep_recent'
    compaction_condition: str = 'always_not'
    target_tokens: int = 4096
    max_segments: int | None = None
    inject_trajectory_budget_state: bool = False
    condition: dict[str, Any] = Field(default_factory=dict)
    strategy: dict[str, Any] = Field(default_factory=dict)


class MainAgentInput(AgentRunInput, WithUserPrompt, WithHistory, WithToolSearch):
    system_prompt: str | list[ChatCompletionContentPartParam] = 'You are a helpful assistant.'
    max_iterations: int = 30

    # 工具执行
    tools: list[str]
    mcp_servers: list[dict[str, Any]] | None = None
    parallel_tool_calls: bool = False
    invalid_tool_args_retries: int = 0
    tool_call_error_retries: int = 0
    tool_output_truncation: int | None = None
    # Model-token cap applied to each text tool result before it enters history.
    max_tool_response_tokens: int | None = Field(default=None, gt=0)
    # On an unrecoverable tool-call error (unparseable args, failed name/arg
    # validation, or tool execution exception) the default is to abort the run
    # with a tool error status. When this is enabled the error is instead
    # returned to the model as a tool observation and the loop continues, giving
    # the agent a chance to self-correct. Overridable at runtime via the
    # ``NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE`` env var. It does not cover a tool
    # call the client parser left unparsed (UNPARSED_TOOL_CALL_FIELD): that turn
    # has no tool_call id to answer, so the run always ends with a tool error status.
    tool_error_continue: bool = False
    # HyTuner segment shaping can opt in to wrapping tool-reported failures as
    # Tool error observations. Default stays false to preserve baseline output.
    wrap_failed_tool_response_as_tool_error: bool = False
    tool_overrides: list[MainAgentToolOverride] | None = None
    tool_schema_overrides: list[dict[str, Any]] | str | None = None
    oom_runtime_recovery: bool = False
    multimodal_tool_output: bool = True
    image_history_limit: int | None = None
    warmup_runtime: bool = False

    compaction: MainAgentCompactionConfig | None = None

    multi_agent: MultiAgentConfig | None = None
    execution_identity: AgentExecutionIdentity | None = Field(default=None, exclude=True)
    openai_client_key: str = Field(default='main', exclude=True)

    # 回调
    post_iteration_hook: str | None = None

    # 超时
    observation_time_timeout: float | None = None
    inference_time_timeout: float | None = None
    single_tool_timeout: float | None = None


class MainAgentOutput(AgentRunOutput, WithMainTrajectory):
    exit_status: str
    iterations: int = 0
    final_answer: str | None = None
    multi_agent_summary: dict[str, Any] | None = None


def _prepare_messages_for_inference(
    messages: list[dict[str, Any]], image_history_limit: int | None
) -> list[dict[str, Any]]:
    """Return LLM-facing messages with older image payloads elided.

    The stored trajectory keeps full screenshots. This helper only trims the
    copy sent to the model, which prevents long GUI runs from exceeding
    provider image-byte limits while preserving text/tool history.
    """
    if image_history_limit is None:
        return messages

    keep_remaining = max(int(image_history_limit), 0)
    prepared: list[dict[str, Any]] = []
    for message in reversed(messages):
        content = message.get('content')
        if not isinstance(content, list):
            prepared.append(message)
            continue

        copied = dict(message)
        copied_content: list[Any] = []
        for part in reversed(content):
            if isinstance(part, dict) and part.get('type') in {'image_url', 'image'}:
                if keep_remaining > 0:
                    copied_content.append(part)
                    keep_remaining -= 1
                else:
                    copied_content.append({'type': 'text', 'text': '[previous image omitted from inference history]'})
            else:
                copied_content.append(part)
        copied['content'] = list(reversed(copied_content))
        prepared.append(copied)

    return list(reversed(prepared))


def _append_reminder_to_user_content(
    user_content: str | list[ContentPartParam],
    reminder: str | None,
) -> str | list[ContentPartParam]:
    """Append a tool-search reminder to the first user message content.

    Returns the content unchanged when there is no reminder. Strings are joined
    with a blank line; multimodal lists get an extra text part appended to a copy
    (the input list is not mutated).
    """
    if reminder is None:
        return user_content
    if isinstance(user_content, str):
        return f'{user_content}\n\n{reminder}'
    return [*user_content, {'type': 'text', 'text': reminder}]


def _message_refusal(message: Any) -> str | None:
    refusal = getattr(message, 'refusal', None)
    if not isinstance(refusal, str):
        return None
    return refusal.strip() or None


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


async def _run_main_agent_loop(agent_input: MainAgentInput) -> MainAgentOutput:
    ctx = require_context()
    client = ctx.get_openai_client(agent_input.openai_client_key)
    state = _LoopState()
    if agent_input.warmup_runtime:
        await ctx.get_runtime()

    if agent_input.tool_search is not None and agent_input.tool_search.enabled and agent_input.tool_schema_overrides:
        raise ValueError('tool_schema_overrides cannot be used when tool_search.enabled is true')

    # Load MCP tools when the agent itself carries mcp_servers config.
    if agent_input.mcp_servers:
        await ctx.load_mcp_tools(extra_servers=agent_input.mcp_servers)

    # Resolve static tools from registry while allowing context-loaded MCP tools.
    dynamic_tool_names = set(getattr(ctx, 'mcp_tools', {}).keys())
    tool_funcs = []
    for tool_name in agent_input.tools:
        if tool_name in dynamic_tool_names:
            continue
        try:
            tool_funcs.append(tool_registry.get(tool_name))
        except EntryNotFoundError as exc:
            raise EntryNotFoundError(
                f"Tool '{tool_name}' not found in registry or context-loaded MCP tools. "
                f'Context MCP tools: {sorted(dynamic_tool_names)}'
            ) from exc

    executor = ToolExecutor(
        tool_funcs,
        allowed_tool_names=agent_input.tools,
        tool_search_config=agent_input.tool_search,
    )

    if agent_input.tool_overrides:
        for override in agent_input.tool_overrides:
            executor.set_tool_override(
                override.tool_name,
                name=override.name,
                description=override.description,
                params=override.params,
            )

    tool_schemas = executor.get_tool_schemas()

    resolved_overrides = load_tool_schema_overrides(agent_input.tool_schema_overrides)
    if resolved_overrides:
        tool_schemas = resolved_overrides
        logger.debug(
            'Using overridden tool schemas: %s', [s.get('function', {}).get('name', '?') for s in tool_schemas]
        )

    # Resolve post-iteration hook
    hook_fn = None
    if agent_input.post_iteration_hook:
        hook_fn = import_from_path(agent_input.post_iteration_hook)
        logger.debug('Loaded post_iteration_hook: %s', agent_input.post_iteration_hook)

    # Conversation history
    tool_search_reminder = executor.build_tool_search_reminder()
    user_content = _append_reminder_to_user_content(agent_input.user_prompt, tool_search_reminder)
    validate_history(agent_input.history)
    prior_history = agent_input.history or []
    history: list[dict[str, Any]] = [
        {'role': 'system', 'content': agent_input.system_prompt},
        *prior_history,
        {'role': 'user', 'content': user_content},
    ]
    if ctx.compaction_manager:
        inject_initial_runtime_context = getattr(ctx.compaction_manager, 'inject_initial_runtime_context', None)
        if callable(inject_initial_runtime_context):
            history = inject_initial_runtime_context(history)
    log_messages(history, prefix='init ')

    exit_status = MainExitStatus.MAX_ITERATIONS.value
    final_answer: str | None = None
    invalid_args_max_retries = agent_input.invalid_tool_args_retries
    invalid_arg_observation_retries = 0
    tool_error_continue = _resolve_tool_error_continue(agent_input)
    if tool_error_continue:
        logger.debug(
            'Main agent: continue-on-tool-error enabled — tool errors are fed back to the model instead of aborting'
        )
    next_request_is_compaction_resume = False
    inference_budget = TimeBudget(agent_input.inference_time_timeout)
    observation_budget = TimeBudget(agent_input.observation_time_timeout)
    pending_completion_kwargs: dict[str, Any] | None = None
    tool_response_token_encode = None
    tool_response_token_decode = None
    if agent_input.max_tool_response_tokens is not None:
        tool_response_tokenizer = getattr(client, 'tokenizer', None)
        tool_response_token_encode = getattr(tool_response_tokenizer, 'encode', None)
        tool_response_token_decode = getattr(tool_response_tokenizer, 'decode', None)
        if not callable(tool_response_token_encode) or not callable(tool_response_token_decode):
            raise TypeError('max_tool_response_tokens requires an OpenAI client with a tokenizer')
        minimum_budget = minimum_tool_response_token_budget(
            lambda text: tool_response_token_encode(text, add_special_tokens=False)
        )
        if agent_input.max_tool_response_tokens < minimum_budget:
            raise ValueError(
                f'max_tool_response_tokens must be at least {minimum_budget} for this tokenizer to preserve '
                f'head/tail around the truncation marker, got {agent_input.max_tool_response_tokens}'
            )

    for iteration in range(1, agent_input.max_iterations + 1):
        state.iterations = iteration
        invocation_kind = 'compaction_resume' if next_request_is_compaction_resume else None

        with span_context(
            'agent.iteration',
            attributes={
                'type': 'agent_iteration',
                'iteration': iteration,
            },
        ):
            # --- LLM call with optional retry ---
            valid_response = False
            output_length_exceeded = False
            unparsed_tool_call = False
            message = None
            prepared_tool_calls: list[_PreparedToolCall] | None = None

            for retry in range(invalid_args_max_retries + 1):
                try:
                    with span_context(
                        'agent.inference',
                        attributes={
                            'type': 'agent_inference',
                            'iteration': iteration,
                            'retry': retry,
                        },
                    ):
                        inference_history = _prepare_messages_for_inference(
                            history,
                            agent_input.image_history_limit,
                        )
                        completion_kwargs = pending_completion_kwargs or {}
                        pending_completion_kwargs = None
                        hook_request_meta = completion_kwargs.pop('request_meta', None)
                        request_meta = dict(hook_request_meta or {})
                        identity = agent_input.execution_identity
                        if identity is not None:
                            declared_chain: dict[str, Any] = {
                                'chain_id': identity.chain_id,
                                'kind': identity.kind,
                                'parent_chain_id': identity.parent_chain_id,
                            }
                            if identity.spawn_order is not None:
                                declared_chain['spawn_order'] = identity.spawn_order
                            request_meta.update(
                                agent_id=identity.agent_id,
                                parent_agent_id=identity.parent_agent_id,
                                declared_chain=declared_chain,
                            )
                        request_meta.update(iteration=iteration, retry=retry)
                        request_meta.pop('invocation_kind', None)
                        if invocation_kind is not None:
                            request_meta['invocation_kind'] = invocation_kind
                        response = await await_completion_with_time_budget(
                            inference_budget,
                            lambda inference_history=inference_history, request_meta=request_meta, completion_kwargs=completion_kwargs: (
                                client.create_completion(
                                    messages=inference_history,
                                    request_meta=request_meta,
                                    tools=tool_schemas or None,
                                    **completion_kwargs,
                                )
                            ),
                        )
                except ContextLengthExceededError:
                    exit_status = MainExitStatus.CONTEXT_LENGTH_EXCEEDED.value
                    logger.warning('Context length exceeded — stopping agent loop')
                    break
                except InferenceTimeoutError:
                    exit_status = MainExitStatus.INFERENCE_TIMEOUT.value
                    break
                except Exception as exc:
                    logger.exception('Inference call failed: %s', exc)
                    exit_status = MainExitStatus.INFERENCE_ERROR.value
                    break

                try:
                    choice = response.choices[0]
                    message = choice.message
                except Exception as exc:
                    logger.exception('Invalid inference response shape: %s', exc)
                    exit_status = MainExitStatus.INFERENCE_ERROR.value
                    break

                if choice.finish_reason == 'length':
                    output_length_exceeded = True
                    break

                # A refusal is terminal even if a malformed provider response
                # also contains tool calls. Never execute tools after refusal.
                if _message_refusal(message):
                    break

                # No tool calls means the agent has finished, unless the client saw the model begin
                # a tool call its parser could not extract.
                if not message.tool_calls:
                    provider_fields = getattr(choice, 'provider_specific_fields', None)
                    unparsed_tool_call = isinstance(provider_fields, dict) and bool(
                        provider_fields.get(UNPARSED_TOOL_CALL_FIELD)
                    )
                    break

                # Invalid JSON in tool call arguments
                prepared_tool_calls = _prepare_tool_calls(message.tool_calls)
                if prepared_tool_calls is None:
                    if retry < invalid_args_max_retries:
                        logger.warning(
                            'Invalid JSON in tool arguments, retry %d/%d',
                            retry + 1,
                            invalid_args_max_retries,
                        )
                        continue
                    break

                valid_response = True
                break

            # The classification belongs to this logical request, including
            # all retries, and is consumed before the next iteration.
            next_request_is_compaction_resume = False

            # Exit was hit inside the retry loop
            if exit_status in {
                MainExitStatus.INFERENCE_TIMEOUT.value,
                MainExitStatus.CONTEXT_LENGTH_EXCEEDED.value,
                MainExitStatus.INFERENCE_ERROR.value,
            }:
                break

            assert message is not None
            refusal = _message_refusal(message)

            if (
                not output_length_exceeded
                and not unparsed_tool_call
                and not message.content
                and not refusal
                and not message.tool_calls
            ):
                logger.error('Inference returned an empty assistant message — stopping agent loop')
                exit_status = MainExitStatus.INFERENCE_ERROR.value
                break

            msg_dict = message.model_dump(exclude_none=True)
            history.append(msg_dict)
            logger.debug('iter %d resp %s', iteration, fmt_msg(msg_dict))

            if refusal:
                final_answer = refusal
            elif message.content:
                final_answer = message.content

            if output_length_exceeded:
                exit_status = MainExitStatus.OUTPUT_LENGTH_EXCEEDED.value
                logger.warning('Model output reached max_tokens — stopping agent loop with partial response')
                break

            if refusal:
                exit_status = MainExitStatus.REFUSED.value
                break

            if unparsed_tool_call:
                logger.warning('Model began a tool call the parser could not extract — stopping agent loop')
                state.tool_call_count += 1
                state.tool_error_count += 1
                exit_status = MainExitStatus.TOOL_ERROR.value
                break

            # No tool calls → finished
            if not message.tool_calls:
                controller = ctx.tool_state.get(MULTI_AGENT_CONTROLLER_STATE_KEY)
                identity = agent_input.execution_identity
                if (
                    identity is not None
                    and identity.kind == 'main'
                    and isinstance(controller, SubagentController)
                    and controller.has_unobserved_agents()
                ):
                    final_answer = None
                    history.append({'role': 'user', 'content': _UNOBSERVED_SUBAGENT_CONTINUATION})
                    continue
                exit_status = MainExitStatus.FINISHED.value
                break

            # JSON parse failed after all retries (at least one call's args are
            # not valid JSON). Count every call as attempted, but only the
            # actually-malformed ones as errors, matching the validation path so
            # tool_error_count stays accurate and never exceeds tool_call_count.
            if not valid_response:
                args_ok = _tool_call_args_parseable(message.tool_calls)
                state.tool_call_count += len(args_ok)
                state.tool_error_count += sum(1 for ok in args_ok if not ok)
                if tool_error_continue:
                    # Keep the assistant message JSON-loadable for downstream
                    # clients, then feed an observation back per tool_call (so
                    # history stays well-formed): malformed calls get the parse
                    # error, well-formed siblings get the skipped notice. Wrap in
                    # an observation span so this iteration stays visible in the
                    # trace / metrics (it skips the execution span below).
                    _sanitize_unparseable_tool_args(history[-1])
                    with _tool_error_observation_span(iteration):
                        err_messages = [
                            _build_tool_message(
                                tc.id,
                                _SKIPPED_TOOL_CALL_OBSERVATION if ok else _INVALID_JSON_ARGS_OBSERVATION,
                            )
                            for tc, ok in zip(message.tool_calls, args_ok, strict=False)
                        ]
                        log_messages(err_messages, prefix=f'iter {iteration} obs ')
                        history.extend(err_messages)
                    continue
                exit_status = MainExitStatus.TOOL_ERROR.value
                break

            assert prepared_tool_calls is not None
            validation_messages, validation_error_count = _build_tool_call_validation_messages(
                executor,
                prepared_tool_calls,
            )
            if validation_messages is not None:
                state.tool_call_count += len(prepared_tool_calls)
                state.tool_error_count += validation_error_count
                log_messages(validation_messages, prefix=f'iter {iteration} obs ')
                history.extend(validation_messages)

                if invalid_arg_observation_retries < invalid_args_max_retries:
                    invalid_arg_observation_retries += 1
                    logger.warning(
                        'Invalid tool call arguments, feedback retry %d/%d',
                        invalid_arg_observation_retries,
                        invalid_args_max_retries,
                    )
                    continue

                if tool_error_continue:
                    # Validation error observations were already appended above.
                    # Emit an observation span so this recoverable-error iteration
                    # still shows up in the trace / metrics (it otherwise skips the
                    # execution span below), then keep going so the model can
                    # correct the tool name/args.
                    with _tool_error_observation_span(iteration):
                        logger.warning(
                            'Invalid tool call; continue-on-tool-error enabled — feeding error back to the model'
                        )
                    continue

                exit_status = MainExitStatus.TOOL_ERROR.value
                break

            invalid_arg_observation_retries = 0
            # --- Execute tool calls ---
            t0 = time.monotonic()
            try:
                run_observation = (
                    _execute_tools_parallel if agent_input.parallel_tool_calls else _execute_tools_sequential
                )
                with span_context(
                    'agent.observation',
                    attributes={
                        'type': 'agent_observation',
                        'iteration': iteration,
                    },
                ):
                    remaining = observation_budget.remaining()
                    if remaining is None:
                        remaining = agent_input.single_tool_timeout
                    async with with_timeout(remaining):
                        tool_messages, tc_exit = await run_observation(
                            executor,
                            prepared_tool_calls,
                            agent_input,
                            state,
                        )
                observation_budget.consume(time.monotonic() - t0)
            except (asyncio.TimeoutError, TimeoutError):
                observation_budget.consume(time.monotonic() - t0)
                exit_status = MainExitStatus.OBSERVATION_TIMEOUT.value
                break

            # --- Post-iteration hook ---
            if hook_fn is not None and tc_exit is None:
                hook_context = PostIterationHookContext(
                    iteration=iteration,
                    max_iterations=agent_input.max_iterations,
                    history=tuple(history),
                    tool_messages=tuple(tool_messages),
                )
                try:
                    hook_result = await hook_fn(ctx, hook_context)
                except Exception:
                    logger.warning('Post-iteration hook failed', exc_info=True)
                    hook_result = None
                if hook_result is not None:
                    if hook_result.tool_messages is not None:
                        tool_messages = hook_result.tool_messages
                    if hook_result.extra_messages:
                        tool_messages = tool_messages + hook_result.extra_messages
                    if hook_result.next_completion_kwargs:
                        pending_completion_kwargs = dict(hook_result.next_completion_kwargs)

            # continue-on-tool-error: turn a fatal tool-execution error into a
            # recoverable observation and keep the loop running. A dead runtime
            # transport stays fatal: every remaining call would fail the same
            # way, and a rollout produced against a dead connection must not
            # reach scoring.
            tc_error_recoverable = (
                tool_error_continue
                and not state.tool_transport_fatal
                and tc_exit
                in {
                    MainExitStatus.TOOL_ERROR.value,
                    MainExitStatus.TOOL_EXECUTION_ERROR.value,
                }
            )
            if tc_error_recoverable:
                tool_messages = _pad_missing_tool_messages(tool_messages, prepared_tool_calls)
                logger.warning('Tool execution error; continue-on-tool-error enabled — continuing loop')

            if agent_input.max_tool_response_tokens is not None:
                assert tool_response_token_encode is not None
                assert tool_response_token_decode is not None
                tool_names_by_call_id = {tool_call.id: tool_call.name for tool_call in prepared_tool_calls}
                bounded_tool_messages = []
                for tool_message in tool_messages:
                    content = tool_message.get('content')
                    if tool_message.get('role') != 'tool' or not isinstance(content, str):
                        bounded_tool_messages.append(tool_message)
                        continue
                    tool_call_id = tool_message.get('tool_call_id')
                    try:
                        bounded_content, original_tokens, retained_tokens = truncate_tool_response_by_tokens(
                            content,
                            agent_input.max_tool_response_tokens,
                            lambda text: tool_response_token_encode(text, add_special_tokens=False),
                            lambda token_ids: tool_response_token_decode(token_ids, skip_special_tokens=False),
                        )
                    except BaseException as error:
                        tokenizer_error = as_tokenizer_panic_error(error, operation='tool result')
                        if tokenizer_error is None:
                            raise
                        logger.exception(
                            'Main agent: tool result tokenization failed '
                            '(trajectory_id=%s, instance_id=%s, iteration=%d, tool=%s, '
                            'tool_call_id=%s, content_chars=%d, content=%r)',
                            ctx.task_config.get('traj_id', 'unknown'),
                            ctx.task_config.get('instance_id', 'unknown'),
                            iteration,
                            tool_names_by_call_id.get(tool_call_id, 'unknown'),
                            tool_call_id,
                            len(content),
                            content,
                        )
                        raise tokenizer_error from error
                    if bounded_content != content:
                        logger.warning(
                            'Main agent: truncated tool result from %d to %d tokens (limit=%d, tool_call_id=%s)',
                            original_tokens,
                            retained_tokens,
                            agent_input.max_tool_response_tokens,
                            tool_message.get('tool_call_id'),
                        )
                        tool_message = {**tool_message, 'content': bounded_content}
                    bounded_tool_messages.append(tool_message)
                tool_messages = bounded_tool_messages

            log_messages(tool_messages, prefix=f'iter {iteration} obs ')

            history.extend(tool_messages)

            controller = ctx.tool_state.get(MULTI_AGENT_CONTROLLER_STATE_KEY)
            identity = agent_input.execution_identity
            if (
                identity is not None
                and identity.kind == 'main'
                and isinstance(controller, SubagentController)
                and controller.is_unobserved_capacity_full()
            ):
                history.append({'role': 'user', 'content': _FULL_SUBAGENT_CAPACITY_CONTINUATION})

            # Context management: compact history if needed before next iteration.
            tc_exit_terminates = tc_exit is not None and not tc_error_recoverable
            is_last_iter = tc_exit_terminates or iteration == agent_input.max_iterations
            if ctx.compaction_manager and not is_last_iter:
                with span_context(
                    'agent.compaction',
                    attributes={
                        'type': 'agent_compaction',
                        'iteration': iteration,
                    },
                ) as span:
                    tools_for_compact = tool_schemas or None
                    try:
                        compact_result = await ctx.compaction_manager.maybe_compact(
                            messages=history,
                            tools=tools_for_compact,
                            inference_budget=inference_budget,
                        )
                    except InferenceTimeoutError:
                        exit_status = MainExitStatus.INFERENCE_TIMEOUT.value
                        break
                    history = compact_result.messages
                    span.update(
                        attributes={
                            'compacted': compact_result.compacted,
                            'requested_strategy': compact_result.requested_strategy,
                            'effective_strategy': compact_result.effective_strategy,
                        }
                    )
                    if compact_result.compacted:
                        next_request_is_compaction_resume = True

            if tc_exit_terminates:
                assert tc_exit is not None  # implied by tc_exit_terminates; narrows for the type checker
                exit_status = tc_exit
                break

    span_id = get_current_span_id()
    if span_id:
        try:
            update_span(
                span_id,
                attributes={
                    'agent.exit_status': exit_status,
                    'agent.termination_reason': MainExitStatus.termination_reason_for(exit_status),
                    'agent.observation_time': observation_budget.elapsed,
                    'agent.iterations': state.iterations,
                    'agent.tool_call_count': state.tool_call_count,
                    'agent.tool_error_count': state.tool_error_count,
                },
            )
        except Exception:
            logger.warning('Failed to update span attributes for agent', exc_info=True)

    return MainAgentOutput(
        main_trajectory=history,
        exit_status=exit_status,
        iterations=state.iterations,
        final_answer=final_answer,
    )


def _install_compaction_manager(agent_input: MainAgentInput) -> None:
    config = agent_input.compaction
    if config is None:
        return
    ctx = require_context()
    if ctx.compaction_manager is not None:
        raise ValueError('agent_config.compaction cannot be combined with a runner-installed compaction manager')
    condition_kwargs = dict(config.condition)
    if condition_kwargs.get('ratio') is not None:
        max_model_len = getattr(ctx.get_openai_client(agent_input.openai_client_key), 'max_model_len', None)
        if max_model_len is None:
            raise ValueError(
                'compaction.condition.ratio requires an OpenAI client with max_model_len; '
                'use compaction.condition.max_generation_length instead'
            )
        condition_kwargs['max_model_len'] = max_model_len
    ctx.compaction_manager = CompactionManager(
        strategy_name=config.compaction_strategy,
        condition_name=config.compaction_condition,
        target_tokens=config.target_tokens,
        max_segments=config.max_segments,
        token_count_fn=count_tokens_with_context_client,
        condition_kwargs=condition_kwargs,
        inject_trajectory_budget_state=config.inject_trajectory_budget_state,
        **config.strategy,
    )


@agent_registry.register('main_agent', meta={'gateway': {'scaffold_policy': 'nexus_main_agent'}})
async def run_main_agent(agent_input: MainAgentInput) -> MainAgentOutput:
    """Run a root agent and await all declared first-level subagents on success."""
    _install_compaction_manager(agent_input)
    multi_agent_config = agent_input.multi_agent
    if multi_agent_config is None or not multi_agent_config.enabled:
        return await _run_main_agent_loop(agent_input)
    if agent_input.execution_identity is not None and agent_input.execution_identity.kind != 'main':
        raise ValueError('subagents cannot enable multi_agent')

    ctx = require_context()
    if MULTI_AGENT_CONTROLLER_STATE_KEY in ctx.tool_state:
        raise RuntimeError('a subagent controller is already active in this execution context')
    parent_client = ctx.get_openai_client(agent_input.openai_client_key)
    if not _supports_agent_fork(parent_client):
        raise TypeError('multi_agent requires an OpenAI client that implements fork_for_agent')
    forking_client = parent_client
    missing_control_tools = [
        tool_name for tool_name in (SPAWN_AGENT_TOOL_NAME, WAIT_AGENT_TOOL_NAME) if tool_name not in agent_input.tools
    ]
    if missing_control_tools:
        raise ValueError(f'multi_agent requires root tools: {missing_control_tools}')

    child_config = multi_agent_config.subagent
    control_tool_names = {SPAWN_AGENT_TOOL_NAME, WAIT_AGENT_TOOL_NAME}
    forbidden_child_tools = [tool_name for tool_name in child_config.tools if tool_name in control_tool_names]
    if forbidden_child_tools:
        raise ValueError(f'subagents cannot use control tools: {forbidden_child_tools}')
    for child_tool_name in child_config.tools:
        tool_registry.get(child_tool_name)

    root_input = agent_input.model_copy(update={'execution_identity': AgentExecutionIdentity.main()})

    async def run_child(
        identity: AgentExecutionIdentity,
        prompt: str,
    ) -> MainAgentOutput:
        child_client_key = f'{agent_input.openai_client_key}:{identity.agent_id}'
        if child_client_key in ctx.openai_clients:
            raise RuntimeError(f"OpenAI client '{child_client_key}' already exists")
        ctx.openai_clients[child_client_key] = forking_client.fork_for_agent(
            agent_id=identity.agent_id,
            parent_agent_id=identity.parent_agent_id,
        )
        child_input = MainAgentInput(
            system_prompt=child_config.system_prompt,
            user_prompt=prompt,
            tools=list(child_config.tools),
            max_iterations=child_config.max_iterations,
            parallel_tool_calls=child_config.parallel_tool_calls,
            execution_identity=identity,
            openai_client_key=child_client_key,
        )
        try:
            return await _run_main_agent_loop(child_input)
        finally:
            ctx.openai_clients.pop(child_client_key, None)

    controller = SubagentController(multi_agent_config, run_child=run_child)
    ctx.tool_state[MULTI_AGENT_CONTROLLER_STATE_KEY] = controller
    root_output: MainAgentOutput | None = None
    try:
        root_output = await _run_main_agent_loop(root_input)
        await controller.finish()
        root_output.multi_agent_summary = controller.summary()
        return root_output
    except BaseException:
        await controller.abort()
        raise
    finally:
        ctx.tool_state[MULTI_AGENT_SUMMARY_STATE_KEY] = controller.summary()
        ctx.tool_state.pop(MULTI_AGENT_CONTROLLER_STATE_KEY, None)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


@dataclass
class _LoopState:
    """Mutable counters shared between the main loop and tool-execution helpers.

    Timing metrics are tracked via span durations; processor remains the
    authoritative source for exported per-step timing.
    """

    iterations: int = 0
    tool_call_count: int = 0
    tool_error_count: int = 0
    # Set when the last tool error, after tool_call_error_retries were
    # exhausted, was a transport-level connection error (connection reset /
    # broken pipe): every remaining call would fail the same way, and a
    # rollout produced against a dead connection must not reach scoring.
    # Retries still run first, so consumers that configured
    # tool_call_error_retries keep their recovery window.
    tool_transport_fatal: bool = False


@dataclass
class _PreparedToolCall:
    id: str
    name: str
    args: Any


def _prefix_tool_error_text(text: str) -> str:
    return text if text.lower().startswith('tool error:') else f'Tool error: {text}'


def _wrap_failed_tool_response_content(
    output_content: str | list[dict[str, Any]],
) -> str | list[dict[str, Any]]:
    if isinstance(output_content, str):
        return _prefix_tool_error_text(output_content)

    wrapped_content = [dict(part) for part in output_content]
    for index, part in enumerate(wrapped_content):
        if part.get('type') == 'text':
            text = str(part.get('text') or 'tool reported failure')
            wrapped_content[index] = {**part, 'text': _prefix_tool_error_text(text)}
            return wrapped_content

    return [{'type': 'text', 'text': 'Tool error: tool reported failure'}, *wrapped_content]


def _build_tool_response_message(
    tool_call_id: str,
    resp: ToolResponse,
    output_content: str | list[dict[str, Any]],
    agent_input: MainAgentInput,
    state: _LoopState,
) -> dict[str, Any]:
    """Build the model-facing tool observation for a successful executor call."""
    if agent_input.wrap_failed_tool_response_as_tool_error and resp.meta.get('success') is False:
        state.tool_error_count += 1
        return _build_tool_message(tool_call_id, _wrap_failed_tool_response_content(output_content))
    return _build_tool_message(tool_call_id, output_content)


def _prepare_tool_calls(tool_calls: list[Any]) -> list[_PreparedToolCall] | None:
    """Parse tool-call arguments once. Return None when any args JSON is invalid."""
    prepared: list[_PreparedToolCall] = []
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, AttributeError, TypeError):
            return None
        prepared.append(_PreparedToolCall(id=tc.id, name=tc.function.name, args=args))
    return prepared


def _tool_call_args_parseable(tool_calls: list[Any]) -> list[bool]:
    """Per-call flag: True when a tool call's args are valid JSON, else False.

    Used on the parse-failure path (where ``_prepare_tool_calls`` returned None
    for the whole batch) to attribute the error to the calls that are actually
    malformed, instead of blaming the entire batch.
    """
    flags: list[bool] = []
    for tc in tool_calls:
        try:
            json.loads(tc.function.arguments)
            flags.append(True)
        except (json.JSONDecodeError, AttributeError, TypeError):
            flags.append(False)
    return flags


_TOOL_ERROR_CONTINUE_ENV = 'NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE'
_TRUE_VALUES = {'1', 'true', 'yes', 'y', 'on'}
_FALSE_VALUES = {'0', 'false', 'no', 'n', 'off'}


def _resolve_tool_error_continue(agent_input: MainAgentInput) -> bool:
    """Whether tool-call errors should be fed back to the model instead of aborting.

    The ``NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE`` env var, when set to a
    recognised truthy/falsy value, overrides the ``tool_error_continue`` config
    field; otherwise the config field (default ``False``) is used.
    """
    raw = os.environ.get(_TOOL_ERROR_CONTINUE_ENV)
    if raw is not None:
        value = raw.strip().lower()
        if value in _TRUE_VALUES:
            return True
        if value in _FALSE_VALUES:
            return False
    return bool(agent_input.tool_error_continue)


_INVALID_JSON_ARGS_OBSERVATION = (
    'Tool error: tool call arguments are not valid JSON and could not be parsed. '
    'Please resend the tool call with corrected JSON arguments.'
)

_SKIPPED_AFTER_ERROR_OBSERVATION = (
    'Tool error: this tool call was not executed because an earlier tool call in '
    'the same assistant message failed. Please resend it if still needed.'
)


def _sanitize_unparseable_tool_args(msg: dict[str, Any]) -> None:
    """Make every tool_call's ``arguments`` JSON-loadable in-place.

    When continuing past an unparseable-args error the offending assistant
    message stays in history. Downstream clients (e.g. HyTuner) normalise
    tool-call arguments via ``json.loads``; an invalid string would crash the
    next inference, so wrap it in a valid JSON object that preserves the raw
    text for the model to see.
    """
    for tc in msg.get('tool_calls') or []:
        fn = tc.get('function') or {}
        args = fn.get('arguments')
        if isinstance(args, str):
            try:
                json.loads(args)
            except (json.JSONDecodeError, ValueError):
                fn['arguments'] = json.dumps({'_unparseable_arguments': args}, ensure_ascii=False)


_SKIPPED_TOOL_CALL_OBSERVATION = (
    'Tool call was not executed because another tool call in the same assistant message '
    'had invalid arguments. Please resend the intended tool calls with corrected arguments.'
)


def _build_tool_call_validation_messages(
    executor: ToolExecutor,
    tool_calls: list[_PreparedToolCall],
) -> tuple[list[dict[str, Any]] | None, int]:
    """Return tool error observations when tool names or arguments are invalid."""
    if not executor.available_schema_names():
        return None, 0

    errors_by_call_id: dict[str, str] = {}
    for tc in tool_calls:
        try:
            executor.validate_tool_call(tc.name, tc.args)
        except (ToolNameResolutionError, ToolArgumentValidationError) as exc:
            errors_by_call_id[tc.id] = f'Tool error: {exc.message_for_model}'

    if not errors_by_call_id:
        return None, 0

    messages: list[dict[str, Any]] = []
    for tc in tool_calls:
        messages.append(
            _build_tool_message(
                tc.id,
                errors_by_call_id.get(tc.id, _SKIPPED_TOOL_CALL_OBSERVATION),
            )
        )
    return messages, len(errors_by_call_id)


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------


def _build_tool_message(tool_call_id: str, content: str | list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'role': 'tool',
        'tool_call_id': tool_call_id,
        'content': content,
    }


def _tool_error_observation_span(iteration: int):
    """Observation span for a recoverable tool-call error iteration.

    The continue-on-tool-error paths feed the error back and skip the normal
    tool-execution observation span; wrapping their bookkeeping in this span
    keeps the iteration visible in the trace / metrics.
    """
    return span_context(
        'agent.observation',
        attributes={'type': 'agent_observation', 'iteration': iteration, 'tool_error': True},
    )


def _pad_missing_tool_messages(
    tool_messages: list[dict[str, Any]],
    prepared_tool_calls: list[_PreparedToolCall],
) -> list[dict[str, Any]]:
    """Ensure every tool call has a matching tool response message.

    A tool-execution error can short-circuit the executor, leaving later tool
    calls in the same assistant message without a response. When continuing past
    the error we must keep the message history well-formed (one tool response per
    tool_call), so we append placeholder error observations positionally.
    """
    if len(tool_messages) >= len(prepared_tool_calls):
        return tool_messages
    padded = list(tool_messages)
    for tc in prepared_tool_calls[len(tool_messages) :]:
        padded.append(_build_tool_message(tc.id, _SKIPPED_AFTER_ERROR_OBSERVATION))
    return padded


def _termination_exit_status_via_tool(resp: ToolResponse) -> str | None:
    """Map explicit termination tool metadata to the agent loop exit status."""
    if resp.meta.get('termination_requested') is True:
        return MainExitStatus.FINISHED.value
    return None


_OOM_NOTICE = (
    '\nThe sandbox has been destroyed due to OOM. '
    'A fresh sandbox will be created for the next tool call. '
    'All previous state (variables, files, processes) has been lost.'
)


async def _release_runtime(reason: str = 'unknown') -> None:
    """Release all runtimes so the next tool call provisions a fresh sandbox."""
    ctx = require_context()
    if ctx.runtime_provider is not None:
        logger.warning('Releasing runtime: %s', reason)
        await ctx.runtime_provider.release_all()
        ctx._runtimes.clear()


async def _execute_tools_sequential(
    executor: ToolExecutor,
    tool_calls: list[_PreparedToolCall],
    agent_input: MainAgentInput,
    state: _LoopState,
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute tool calls one-by-one. Returns (tool_messages, exit_status_or_None)."""
    messages: list[dict[str, Any]] = []
    termination_exit: str | None = None

    for tc in tool_calls:
        tool_name = tc.name
        args = tc.args
        state.tool_call_count += 1
        resp: ToolResponse | None = None
        last_exc: Exception | None = None

        for attempt in range(agent_input.tool_call_error_retries + 1):
            try:
                resp = await executor.call(tool_name, args)
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, TimeoutError):
                raise
            except RuntimeUnhealthyError as exc:
                state.tool_error_count += 1
                logger.exception('Runtime unhealthy error occurred, stopping agent: %s', exc)
                messages.append(_build_tool_message(tc.id, f'Tool error: {exc}'))
                return messages, MainExitStatus.RUNTIME_ERROR.value
            except (ToolNameResolutionError, ToolArgumentValidationError) as exc:
                state.tool_error_count += 1
                message_for_model = getattr(exc, 'message_for_model', str(exc))
                messages.append(_build_tool_message(tc.id, f'Tool error: {message_for_model}'))
                return messages, MainExitStatus.TOOL_ERROR.value
            except Exception as exc:
                state.tool_error_count += 1
                last_exc = exc

                if agent_input.oom_runtime_recovery and 'OOMKilled' in str(exc):
                    await _release_runtime(reason='OOMKilled during tool execution')
                    messages.append(_build_tool_message(tc.id, f'Tool error: {exc}{_OOM_NOTICE}'))
                    break

                if attempt < agent_input.tool_call_error_retries:
                    logger.warning(
                        'Tool call failed (%s), retry %d/%d: %s',
                        tool_name,
                        attempt + 1,
                        agent_input.tool_call_error_retries,
                        exc,
                    )
                    continue

                # Transport death is retried like any tool error, so consumers
                # that configured tool_call_error_retries keep their recovery
                # window (a one-off connection reset used to be recoverable).
                # It only becomes fatal here, once retries are exhausted: the
                # flag vetoes the continue-on-tool-error path below, so a
                # rollout produced against a dead connection still cannot
                # reach scoring. Only the dedicated transport error counts:
                # ToolExecutor wraps the sandbox-bound OS errors at its
                # RUNTIME dispatch branch, so a LOCAL/MCP tool raising a
                # connection error of its own stays a recoverable tool error.
                if isinstance(exc, RuntimeTransportError):
                    state.tool_transport_fatal = True
                messages.append(_build_tool_message(tc.id, f'Tool error: {exc}'))
                return messages, MainExitStatus.TOOL_EXECUTION_ERROR.value

            break

        if resp is None:
            if agent_input.oom_runtime_recovery and last_exc is not None and 'OOMKilled' in str(last_exc):
                continue  # OOM already handled: message appended, skip to next tool_call
            messages.append(_build_tool_message(tc.id, f'Tool error: {last_exc}'))
            return messages, MainExitStatus.TOOL_EXECUTION_ERROR.value

        try:
            extract_fn = extract_tool_output_content if agent_input.multimodal_tool_output else extract_tool_output_text
            output_content = extract_fn(resp.output)
            if isinstance(output_content, str):
                output_content = maybe_truncate(output_content, agent_input.tool_output_truncation)
        except Exception as exc:
            state.tool_error_count += 1
            messages.append(_build_tool_message(tc.id, f'Tool error: {exc}'))
            return messages, MainExitStatus.TOOL_EXECUTION_ERROR.value
        messages.append(_build_tool_response_message(tc.id, resp, output_content, agent_input, state))
        termination_exit = termination_exit or _termination_exit_status_via_tool(resp)

    return messages, termination_exit


async def _execute_tools_parallel(
    executor: ToolExecutor,
    tool_calls: list[_PreparedToolCall],
    agent_input: MainAgentInput,
    state: _LoopState,
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute tool calls in parallel via executor.parallel_call(). Returns (tool_messages, exit_status_or_None)."""
    calls: list[tuple[str, Any]] = []
    call_ids: list[str] = []

    for tc in tool_calls:
        tool_name = tc.name
        call_ids.append(tc.id)
        args = tc.args
        calls.append((tool_name, args))

    state.tool_call_count += len(calls)

    responses: list[ToolResponse] | None = None
    last_exc: Exception | None = None
    for attempt in range(agent_input.tool_call_error_retries + 1):
        try:
            responses = await executor.parallel_call(calls)
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, TimeoutError):
            raise
        except RuntimeUnhealthyError as exc:
            state.tool_error_count += len(calls)
            logger.exception('Runtime unhealthy error occurred during parallel tool execution: %s', exc)
            messages = [_build_tool_message(cid, f'Tool error: {exc}') for cid in call_ids]
            return messages, MainExitStatus.RUNTIME_ERROR.value
        except (ToolNameResolutionError, ToolArgumentValidationError) as exc:
            state.tool_error_count += len(calls)
            message_for_model = getattr(exc, 'message_for_model', str(exc))
            messages = [_build_tool_message(cid, f'Tool error: {message_for_model}') for cid in call_ids]
            return messages, MainExitStatus.TOOL_ERROR.value
        except Exception as exc:
            state.tool_error_count += len(calls)
            last_exc = exc

            if agent_input.oom_runtime_recovery and 'OOMKilled' in str(exc):
                await _release_runtime(reason='OOMKilled during parallel tool execution')
                messages = [_build_tool_message(cid, f'Tool error: {exc}{_OOM_NOTICE}') for cid in call_ids]
                return messages, None

            if attempt < agent_input.tool_call_error_retries:
                logger.warning(
                    'Parallel tool call failed, retry %d/%d: %s',
                    attempt + 1,
                    agent_input.tool_call_error_retries,
                    exc,
                )
                continue

            # Same as the sequential path: transport errors get the configured
            # retries, and turn fatal only once those are exhausted. Only the
            # dedicated transport error counts -- see the sequential branch.
            if isinstance(exc, RuntimeTransportError):
                state.tool_transport_fatal = True
            messages = [_build_tool_message(cid, f'Tool error: {exc}') for cid in call_ids]
            return messages, MainExitStatus.TOOL_EXECUTION_ERROR.value

        break

    if responses is None:
        messages = [_build_tool_message(cid, f'Tool error: {last_exc}') for cid in call_ids]
        return messages, MainExitStatus.TOOL_EXECUTION_ERROR.value

    if len(responses) != len(call_ids):
        state.tool_error_count += 1
        messages = [_build_tool_message(cid, 'Tool error: parallel response count mismatch') for cid in call_ids]
        return messages, MainExitStatus.TOOL_EXECUTION_ERROR.value

    messages: list[dict[str, Any]] = []
    termination_exit: str | None = None
    for cid, resp in zip(call_ids, responses, strict=False):
        try:
            extract_fn = extract_tool_output_content if agent_input.multimodal_tool_output else extract_tool_output_text
            output_content = extract_fn(resp.output)
            if isinstance(output_content, str):
                output_content = maybe_truncate(output_content, agent_input.tool_output_truncation)
        except Exception as exc:
            state.tool_error_count += 1
            messages.append(_build_tool_message(cid, f'Tool error: {exc}'))
            return messages, MainExitStatus.TOOL_EXECUTION_ERROR.value
        messages.append(_build_tool_response_message(cid, resp, output_content, agent_input, state))
        termination_exit = termination_exit or _termination_exit_status_via_tool(resp)

    return messages, termination_exit
