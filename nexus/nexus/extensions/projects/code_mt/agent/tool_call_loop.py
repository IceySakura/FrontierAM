"""Tool-call mode loop for code_mt_agent."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time

from typing import Any

from nexus.core import ToolExecutor, require_context
from nexus.core.openai_client import ContextLengthExceededError
from nexus.extensions.projects.code_mt.agent.models import CodeMTAgentInput, CodeMTAgentOutput
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.trace import span_context
from nexus.utils.timeout import InferenceTimeoutError, TimeBudget, await_completion_with_time_budget


logger = logging.getLogger(__name__)


def _is_loaded_hytuner_client(client: object) -> bool:
    """Check the client type without importing the optional HyTuner module."""
    module = sys.modules.get('nexus.extensions.openai_clients.hytuner')
    if module is None:
        return False
    client_type = getattr(module, 'HyTunerClient', None)
    return isinstance(client_type, type) and isinstance(client, client_type)


def _log_cache_summary(total_in: int, total_out: int, cache_read: int, cache_create: int, cached: int) -> None:
    if total_in <= 0:
        return
    overall_rate = (cached / total_in * 100) if cached > 0 else 0
    lines = [
        f'  Total Input:   {total_in:>10,} tokens',
        f'  Total Output:  {total_out:>10,} tokens',
        f'  Cache Read:    {cache_read:>10,} ({overall_rate:.1f}%)',
        f'  Cache Create:  {cache_create:>10,}',
        f'  Overall Rate:  {overall_rate:>9.1f}%',
    ]
    width = max(len(ln) for ln in lines) + 2
    border = '═' * (width + 2)
    logger.debug('╔%s╗', border)
    logger.debug('║  %-*s║', width, 'Cache Summary')
    logger.debug('╠%s╣', border)
    for line in lines:
        logger.debug('║ %-*s║', width, line)
    logger.debug('╚%s╝', border)


# Truncation notice — copied verbatim from swe_agent (agent.py DEFAULT_TRUNCATE_NOTICE)
# so the observation bytes match swe_agent exactly. code_mt must stay self-contained
# (no cross-agent import), so the constant and the algorithm below are inlined here.
_TRUNCATE_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of the full '
    'response has been shown to you.</NOTE>'
)  # 113 chars


def _truncate_observation(text: str, max_len: int) -> str:
    """Truncate the middle of content, keeping head and tail.

    Byte-for-byte port of swe_agent.agent.maybe_truncate (head/tail split with the
    truncate notice in the middle; head gets the extra char on an odd split).
    """
    # 1) Early exits: no truncation requested, or content already within limit
    if not max_len or len(text) <= max_len or max_len < 0:
        return text
    # 2) If even the base notice doesn't fit, return a slice of it
    if len(_TRUNCATE_NOTICE) >= max_len:
        return _TRUNCATE_NOTICE[:max_len]
    # 3) Calculate head and tail sizes (prefer giving the extra char to head)
    available_chars = max_len - len(_TRUNCATE_NOTICE)
    head_chars = available_chars // 2 + (available_chars % 2)
    tail_chars = available_chars - head_chars
    return text[:head_chars] + _TRUNCATE_NOTICE + (text[-tail_chars:] if tail_chars > 0 else '')


def _validate_tool_args(tool_calls: list[Any]) -> bool:
    """Check that all tool call arguments are valid JSON."""
    for tc in tool_calls:
        try:
            if isinstance(tc.function.arguments, str):
                json.loads(tc.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            return False
    return True


# Key under which the running conversation is stashed on the ExecutionContext so
# it survives across MT turns (the context is per task-run, shared task→agent).
MESSAGES_STATE_KEY = '_code_mt_messages'


def _ensure_ready_for_user_turn(messages: list[dict[str, Any]]) -> None:
    """Make a carried-over trajectory safe to extend with a new user message.

    If the previous turn ended with an assistant message whose tool_calls were
    not ALL answered, append placeholder tool results for the missing ids so the
    assistant/tool pairing stays valid for the next completion request. This
    covers two shapes:

      * the turn ended ON the assistant message (mid-tool timeout before any
        result was recorded), and
      * the turn broke out of a PARALLEL tool_calls loop after answering only
        some ids (e.g. submit / per-tool timeout on tool_call #1 of #3) — here
        the last message is a `tool`, but the assistant above it still has
        unanswered ids. The old ``messages[-1]`` check missed this and the next
        turn's completion request carried an illegal assistant/tool pairing
        (provider 400).

    We locate the last assistant-with-tool_calls, collect the tool_call_ids
    already answered in the trailing run of tool messages, and backfill only the
    difference — preserving the order the assistant declared them.
    """
    if not messages:
        return
    # Find the last assistant message that issued tool_calls.
    a_idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get('role') == 'assistant' and m.get('tool_calls'):
            a_idx = i
            break
        # A user/system message after the last tool run means the previous turn
        # already closed cleanly — nothing to repair.
        if m.get('role') in ('user', 'system'):
            return
    if a_idx is None:
        return

    def _tc_id(tc: Any) -> Any:
        return tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)

    declared = [_tc_id(tc) for tc in messages[a_idx]['tool_calls']]
    answered = {m.get('tool_call_id') for m in messages[a_idx + 1 :] if m.get('role') == 'tool'}
    for tc_id in declared:
        if tc_id not in answered:
            messages.append({
                'role': 'tool',
                'tool_call_id': tc_id,
                'content': 'Tool call not completed (previous turn ended).',
            })


async def tool_call_loop(agent_input: CodeMTAgentInput) -> CodeMTAgentOutput:
    """Main loop for tool_call mode."""
    ctx = require_context()
    client = ctx.openai_client
    # Mirror swe_agent: on the HyTuner training client, a tool-call retry must NOT
    # commit rollout state (last_messages / last_tokens / routed_experts). Without
    # this, a retried malformed-tool-call turn is committed by the client while our
    # loop re-sends the SHORTER (unappended) messages, tripping prefix_shortened and
    # resetting the R3 routed_experts accumulator mid-turn. See openai_clients/hytuner.py.
    is_hytuner_client = _is_loaded_hytuner_client(client)

    # Seed the per-command bash hard-kill timeout into tool_state so the RUNTIME
    # bash tool (executed in-sandbox) can read it via _get_bash_timeout. Only
    # tool_call mode does this; text_parse never calls code_mt/bash.
    ctx.tool_state.setdefault('bash_timeout', agent_input.bash_timeout)

    # Load MCP tools
    if agent_input.mcp_servers:
        await ctx.load_mcp_tools(extra_servers=agent_input.mcp_servers)

    # Turn-0 system + user prompt: VERBATIM passthrough (aligns main_agent).
    # code_mt_agent does NO prompt assembly — the TASK owns all prompt composition
    # (instruction source selection, system_prompt, anti-hack injection) and passes
    # the final strings via run_agent(user_prompt=, system_prompt=). On a retry (MT)
    # turn we resume the stored conversation and append the task's user_prompt
    # (mt_feedback) verbatim.
    system_prompt = agent_input.system_prompt or ''
    initial_user_prompt = agent_input.user_prompt or ''

    # Build tools
    from nexus.core import tool_registry
    from nexus.core.registry import EntryNotFoundError

    dynamic_tool_names = set(getattr(ctx, 'mcp_tools', {}).keys())
    tool_funcs = []
    for tool_name in agent_input.tools:
        if tool_name in dynamic_tool_names:
            continue
        try:
            tool_funcs.append(tool_registry.get(tool_name))
        except EntryNotFoundError:
            logger.warning('Tool %s not found in registry, skipping', tool_name)

    executor = ToolExecutor(tool_funcs, allowed_tool_names=agent_input.tools)
    if agent_input.tool_description_language == 'zh-CN':
        from nexus.extensions.projects.code_mt.agent.tool_localization import apply_chinese_tool_descriptions

        apply_chinese_tool_descriptions(executor, agent_input.tools)
    tool_schemas = executor.get_tool_schemas()

    if agent_input.tool_schema_overrides:
        from nexus.utils.tool_schema_overrides import load_tool_schema_overrides

        resolved = load_tool_schema_overrides(agent_input.tool_schema_overrides)
        if resolved:
            tool_schemas = resolved

    # Resolve post-iteration hook
    hook_fn = None
    if agent_input.post_iteration_hook:
        from nexus.core.registry import import_from_path
        from nexus.extensions.projects.code_mt.task._module_aliases import canonicalize_import_path

        resolved = canonicalize_import_path(agent_input.post_iteration_hook)
        assert resolved is not None  # guarded by outer if (non-empty in -> non-None out)
        hook_fn = import_from_path(resolved)

    # Build / resume messages. The ExecutionContext persists for the whole task
    # run, so a prior MT turn's conversation is stashed under MESSAGES_STATE_KEY.
    # On a retry turn we CONTINUE that same conversation (replaying the full
    # trajectory — which hits the provider prompt cache) and append this turn's
    # prompt (e.g. mt_feedback) as the next user message, rather than starting
    # fresh. On the first turn we build [system?] + [user_prompt] and stash it by
    # reference so subsequent in-place appends keep the stored copy current.
    prior_messages = ctx.tool_state.get(MESSAGES_STATE_KEY)
    if prior_messages:
        messages = prior_messages
        _ensure_ready_for_user_turn(messages)
        messages.append({'role': 'user', 'content': agent_input.user_prompt})
    else:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': initial_user_prompt})
        ctx.tool_state[MESSAGES_STATE_KEY] = messages

    total_observation_time = 0.0
    inference_budget = TimeBudget(agent_input.inference_time_timeout)

    # Resolve timeouts, mirroring swe_agent (agent.py:294-300): a missing
    # per_observation_timeout falls back to the total observation_time_timeout so
    # the executor call is still bounded. Kept local so we never mutate the input.
    observation_time_timeout = agent_input.observation_time_timeout
    per_observation_timeout = agent_input.per_observation_timeout
    if per_observation_timeout is None:
        per_observation_timeout = observation_time_timeout

    # Resolve parallel_tool_calls — pass through natively to the API. Treat the
    # historic `single_tool_call` flag as a legacy alias when the new field is
    # unset; only forward to the LLM call when the operator explicitly opted in
    # so models that don't recognize the param aren't poked unnecessarily.
    parallel_tool_calls = agent_input.parallel_tool_calls
    if parallel_tool_calls is None and agent_input.single_tool_call:
        parallel_tool_calls = False
    completion_extra: dict[str, Any] = {}
    if parallel_tool_calls is not None:
        completion_extra['parallel_tool_calls'] = parallel_tool_calls

    # Cache tracking
    _total_input_tokens = 0
    _total_output_tokens = 0
    _total_cache_read = 0
    _total_cache_creation = 0
    _total_cached_tokens = 0

    # Length overflow is bounded exactly like swe_agent: the HyTuner client caps
    # each call's max_tokens to (max_model_len - initial_prompt_len) and raises
    # ContextLengthExceededError once the replayed prompt leaves no budget
    # (openai_clients/hytuner.py). tool_call mode replays full assistant messages
    # (no reasoning strip — see the model_dump append below), so the prompt grows
    # monotonically and that native check fires before the training seq ceiling
    # (max_prompt + max_response, set equal to max_model_len in the run script).
    # We therefore keep NO extra agent-side length clamp — that matches swe_agent
    # and avoids truncating valid trajectories early. The except-handler below
    # converts the client's ContextLengthExceededError into a clean 'truncation'.

    try:
        for iteration in range(1, agent_input.max_iterations + 1):
            # === INFERENCE (with retry) ===
            response = None
            message_obj = None
            tool_error = False
            no_tool_finish = False

            max_retries = agent_input.completion_retry_on_tool_error
            for retry in range(max_retries + 1):
                # On a HyTuner retry that is NOT the last attempt, tell the client to
                # early-return without committing rollout state, so re-sending the
                # unappended (shorter) messages does not trip R3 prefix_shortened.
                # Matches swe_agent/agent.py:318-320. Last attempt commits normally.
                retry_extra = completion_extra
                if is_hytuner_client and max_retries > 0 and retry < max_retries:
                    retry_extra = {**completion_extra, 'allow_retry_for_tool_call': True}
                try:
                    with span_context(
                        'agent.inference', attributes={'type': 'agent_inference', 'iteration': iteration}
                    ):
                        response = await await_completion_with_time_budget(
                            inference_budget,
                            lambda retry_extra=retry_extra: client.create_completion(
                                messages=messages,
                                tools=tool_schemas or None,
                                **retry_extra,
                            ),
                        )

                    # Track cache usage
                    if response and hasattr(response, 'usage') and response.usage:
                        u = response.usage
                        rd = u.model_dump() if hasattr(u, 'model_dump') else {}
                        inp = getattr(u, 'prompt_tokens', 0) or 0
                        out = getattr(u, 'completion_tokens', 0) or 0
                        cr = getattr(u, 'cache_read_input_tokens', 0) or rd.get('cache_read_input_tokens', 0) or 0
                        cc = (
                            getattr(u, 'cache_creation_input_tokens', 0)
                            or rd.get('cache_creation_input_tokens', 0)
                            or 0
                        )
                        ptd = rd.get('prompt_tokens_details') or {}
                        cached = (ptd.get('cached_tokens', 0) or 0) if isinstance(ptd, dict) else 0
                        if not cached:
                            cached = cr
                        _total_input_tokens += inp
                        _total_output_tokens += out
                        _total_cache_read += cr
                        _total_cache_creation += cc
                        _total_cached_tokens += cached
                        cache_pct = (cached / inp * 100) if inp > 0 and cached > 0 else 0
                        logger.debug(
                            '[iter %d] tokens: in=%d out=%d cache_read=%d cache_create=%d cached_rate=%.1f%%',
                            iteration,
                            inp,
                            out,
                            cr,
                            cc,
                            cache_pct,
                        )
                except ContextLengthExceededError:
                    # Context overflow is terminal — let the outer handler convert it
                    # to a clean 'truncation' exit. Mirrors swe_agent/agent.py:530.
                    raise
                except InferenceTimeoutError:
                    return CodeMTAgentOutput(
                        exit_status='timeout',
                        iterations=iteration,
                        main_trajectory=messages,
                    )
                except Exception as e:
                    if retry < agent_input.completion_retry_on_tool_error:
                        logger.warning('Inference error (retry %d): %s', retry, e)
                        continue
                    raise

                try:
                    message_obj = response.choices[0].message
                except (AttributeError, IndexError):
                    if retry < agent_input.completion_retry_on_tool_error:
                        continue
                    return CodeMTAgentOutput(
                        exit_status='inference_error', iterations=iteration, main_trajectory=messages
                    )

                # Whether generation was cut at the response budget this turn. When
                # set, a missing/half-formed tool call is a LENGTH problem, not a
                # format one — reporting tool_error (FORMAT_ERROR) would bury a
                # truncation under a format error and skew stats. Mirrors
                # swe_agent/agent.py (nexus MR !1018). Ported from qy4 code_mt
                # (this guard was lost in the agentbus merge).
                finish_reason = getattr(response.choices[0], 'finish_reason', None)

                # No tool calls. Two contracts, selected by no_tool_call_finish:
                #  - default (False): mirror swe_agent — a turn with no tool call is
                #    a format error (the task's contract is "finish by CALLING a
                #    submit-family tool"). Retry, then fall through to TOOL_ERROR
                #    (-> FORMAT_ERROR). Preserves historical FORMAT_ERROR stats.
                #  - no_tool_call_finish (True): align with main_agent — a bare
                #    assistant message is a natural completion (-> 'finished' ->
                #    ENV_DONE), capturing the message as the final answer. Used by
                #    open-ended tasks with no submit-family terminator.
                # A length-truncated turn (finish_reason=='length') is ALWAYS a
                # truncation, never a format error, regardless of no_tool_call_finish.
                if not message_obj.tool_calls:
                    if finish_reason == 'length':
                        # Generation cut at the response budget, not a real format
                        # error: retrying is pointless (budget spent) and tool_error
                        # would bury a length problem. Mirrors swe_agent (MR !1018).
                        logger.warning('code_mt: generation cut at the response budget -> truncation')
                        return CodeMTAgentOutput(
                            exit_status='truncation', iterations=iteration, main_trajectory=messages
                        )
                    if retry < agent_input.completion_retry_on_tool_error:
                        logger.warning('No tool calls - model output error, retry %d', retry)
                        continue  # Retry WITHOUT adding to history (mirrors swe_agent:349)
                    if agent_input.no_tool_call_finish:
                        # Natural finish: keep the assistant message in history so
                        # main_trajectory carries the final answer, then end cleanly.
                        logger.info('No tool calls - finishing (no_tool_call_finish)')
                        messages.append(message_obj.model_dump(exclude_none=True))
                        no_tool_finish = True
                        break
                    logger.error('No tool calls - model did not call any tool, all retries exhausted')
                    tool_error = True
                    break

                # Validate tool args JSON
                if not _validate_tool_args(message_obj.tool_calls):
                    if finish_reason == 'length':
                        # Arguments unparseable because generation was cut mid-JSON;
                        # same as the no-tool-calls branch above. Mirrors swe_agent.
                        logger.warning('code_mt: tool arguments cut at the response budget -> truncation')
                        return CodeMTAgentOutput(
                            exit_status='truncation', iterations=iteration, main_trajectory=messages
                        )
                    if retry < agent_input.completion_retry_on_tool_error:
                        logger.warning('Invalid tool args JSON, retry %d', retry)
                        continue
                    # Retries exhausted on malformed tool arguments → format error.
                    # Mirrors swe_agent's ExitStatus.TOOL_ERROR (→ FORMAT_ERROR).
                    logger.error('Invalid tool args JSON, all retries exhausted')
                    tool_error = True
                    break

                break  # Valid response

            if tool_error:
                _log_cache_summary(
                    _total_input_tokens,
                    _total_output_tokens,
                    _total_cache_read,
                    _total_cache_creation,
                    _total_cached_tokens,
                )
                return CodeMTAgentOutput(exit_status='tool_error', iterations=iteration, main_trajectory=messages)

            if no_tool_finish:
                # no_tool_call_finish: the bare assistant message was already
                # appended to `messages`; end the loop as a natural completion.
                _log_cache_summary(
                    _total_input_tokens,
                    _total_output_tokens,
                    _total_cache_read,
                    _total_cache_creation,
                    _total_cached_tokens,
                )
                return CodeMTAgentOutput(exit_status='finished', iterations=iteration, main_trajectory=messages)

            if message_obj is None:
                return CodeMTAgentOutput(exit_status='inference_error', iterations=iteration, main_trajectory=messages)

            # Append assistant message. A no-tool-call turn never reaches here — it is
            # handled as a format error in the retry loop above (mirrors swe_agent), so
            # by this point message_obj.tool_calls is always non-empty.
            msg_dict = message_obj.model_dump(exclude_none=True)
            messages.append(msg_dict)

            # === OBSERVATION ===
            tool_messages: list[dict[str, Any]] = []
            submitted = False

            with span_context('agent.observation', attributes={'type': 'agent_observation', 'iteration': iteration}):
                for tc in message_obj.tool_calls:
                    t0 = time.monotonic()
                    tool_name = tc.function.name
                    try:
                        tool_args = (
                            json.loads(tc.function.arguments)
                            if isinstance(tc.function.arguments, str)
                            else tc.function.arguments
                        )
                    except json.JSONDecodeError:
                        tool_args = {}

                    # Submit / eval-style tools drive exit via context_updates
                    # (`exit_status='submitted'`). Two-stage semantics live inside
                    # the tool itself (e.g. code_mt/submit_patch toggles
                    # tool_state['should_exit']), not in the loop, so builder
                    # (single-call submit) and SWE (two-stage submit_patch) both
                    # work without a name-based special case here.
                    #
                    # Execute with the per-observation timeout enforced by the
                    # executor (mirrors swe_agent/agent.py:401 `timeout=`). The error
                    # taxonomy below is a byte-for-byte port of swe_agent
                    # (agent.py:402-474): RuntimeUnhealthyError -> terminal ERROR,
                    # (Timeout)Error -> terminal TIMEOUT, generic Exception -> log then
                    # early_stop? terminal TOOL_ERROR : append error + continue. Every
                    # error observation is appended UNWRAPPED as `Error executing tool:
                    # {e}` (literal "tool", NO OBSERVATION template, NO truncation) so
                    # the trajectory keeps growing (HyTuner requirement) with content
                    # bytes identical to swe_agent.
                    try:
                        result = await executor.call(tool_name, tool_args, timeout=per_observation_timeout)
                    except RuntimeUnhealthyError as e:
                        total_observation_time += time.monotonic() - t0
                        logger.exception('Runtime unhealthy error occurred, stopping agent, error: %s', e)
                        tool_messages.append({
                            'role': 'tool',
                            'tool_call_id': tc.id,
                            'content': f'Error executing tool: {e}',
                        })
                        messages.extend(tool_messages)
                        _log_cache_summary(
                            _total_input_tokens,
                            _total_output_tokens,
                            _total_cache_read,
                            _total_cache_creation,
                            _total_cached_tokens,
                        )
                        return CodeMTAgentOutput(exit_status='error', iterations=iteration, main_trajectory=messages)
                    except (TimeoutError, asyncio.TimeoutError) as e:
                        total_observation_time += time.monotonic() - t0
                        logger.warning('Observation timeout reached, stopping agent: %s', e)
                        tool_messages.append({
                            'role': 'tool',
                            'tool_call_id': tc.id,
                            'content': f'Error executing tool: {e}',
                        })
                        messages.extend(tool_messages)
                        _log_cache_summary(
                            _total_input_tokens,
                            _total_output_tokens,
                            _total_cache_read,
                            _total_cache_creation,
                            _total_cached_tokens,
                        )
                        return CodeMTAgentOutput(exit_status='timeout', iterations=iteration, main_trajectory=messages)
                    except Exception as e:
                        total_observation_time += time.monotonic() - t0
                        logger.error('Error executing tool %s: %s', tool_name, e, exc_info=True)
                        tool_messages.append({
                            'role': 'tool',
                            'tool_call_id': tc.id,
                            'content': f'Error executing tool: {e}',
                        })
                        # early_stop: terminate immediately (swe_agent/agent.py:448-457).
                        if agent_input.early_stop:
                            messages.extend(tool_messages)
                            _log_cache_summary(
                                _total_input_tokens,
                                _total_output_tokens,
                                _total_cache_read,
                                _total_cache_creation,
                                _total_cached_tokens,
                            )
                            return CodeMTAgentOutput(
                                exit_status='tool_error', iterations=iteration, main_trajectory=messages
                            )
                        # Otherwise keep going, but still honor the TOTAL observation
                        # timeout (swe_agent/agent.py:468-474, uses `>=`).
                        if observation_time_timeout and total_observation_time >= observation_time_timeout:
                            messages.extend(tool_messages)
                            _log_cache_summary(
                                _total_input_tokens,
                                _total_output_tokens,
                                _total_cache_read,
                                _total_cache_creation,
                                _total_cached_tokens,
                            )
                            return CodeMTAgentOutput(
                                exit_status='timeout', iterations=iteration, main_trajectory=messages
                            )
                        continue

                    # Extract the observation TEXT from the tool output, mirroring
                    # swe_agent (agent.py:489): ToolResponse.output is a list of
                    # content parts (`[{'type':'text','text': ...}]`), NOT a plain
                    # string. Pull `output[0]['text']` so the tool message content
                    # is clean text, matching swe_agent byte-for-byte — never the
                    # Python repr of the list.
                    raw_output = ''
                    if hasattr(result, 'output') and result.output:
                        first = result.output[0]
                        raw_output = first.get('text', '') if isinstance(first, dict) else str(first)
                    # Defer the exit until AFTER this tool's result is recorded
                    # below, so the trajectory keeps a valid tool_call/tool pair
                    # and a subsequent MT turn can resume the conversation.
                    if ctx.tool_state.get('exit_status') == 'submitted':
                        submitted = True

                    total_observation_time += time.monotonic() - t0

                    # Truncate + format, mirroring swe_agent (agent.py:490-497):
                    # non-empty output -> head/tail-truncate then wrap in the
                    # OBSERVATION template; empty output -> the no-output message
                    # WITHOUT the OBSERVATION wrapper.
                    if raw_output:
                        truncated = _truncate_observation(raw_output, agent_input.max_observation_length)
                        formatted = agent_input.observation_template.format(output=truncated)
                    else:
                        formatted = agent_input.no_output_message

                    tool_messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.id,
                        'content': formatted,
                    })

                    # Submit fired — stop processing further tool_calls this turn.
                    if submitted:
                        break

                    # Total observation timeout, checked after each tool result like
                    # swe_agent (agent.py:514-520, `>=`). The PER-observation bound is
                    # enforced by the executor `timeout=` above (raises TimeoutError,
                    # handled by the terminal branch), so no manual per-obs check here.
                    if observation_time_timeout and total_observation_time >= observation_time_timeout:
                        messages.extend(tool_messages)
                        _log_cache_summary(
                            _total_input_tokens,
                            _total_output_tokens,
                            _total_cache_read,
                            _total_cache_creation,
                            _total_cached_tokens,
                        )
                        return CodeMTAgentOutput(exit_status='timeout', iterations=iteration, main_trajectory=messages)

            # Submit was requested this iteration. KEEP the submit tool result in
            # the trajectory (mirrors swe_agent/agent.py:503-512, which appends the
            # tool response to history BEFORE breaking on 'submitted'). This makes
            # the stashed conversation end on a paired assistant/tool tail, so a
            # subsequent MT turn resumes as
            #   ...assistant(submit_call) tool(result) user(mt_feedback) ...
            # and _ensure_ready_for_user_turn finds the tool_call already answered
            # -> it injects NO "not completed" placeholder (that placeholder would
            # otherwise be a lie: the submit tool DID complete). The post-iteration
            # hook is intentionally skipped on submit — it only shapes/augments a
            # CONTINUING turn's observations, and a terminating turn takes none.
            if submitted:
                messages.extend(tool_messages)
                _log_cache_summary(
                    _total_input_tokens,
                    _total_output_tokens,
                    _total_cache_read,
                    _total_cache_creation,
                    _total_cached_tokens,
                )
                return CodeMTAgentOutput(exit_status='submitted', iterations=iteration, main_trajectory=messages)

            # Post-iteration hook (only when the turn continues — a submit exits
            # above after recording its tool result).
            if hook_fn is not None:
                try:
                    hook_result = await hook_fn(ctx, tool_messages)
                    if hook_result is not None:
                        if hasattr(hook_result, 'tool_messages') and hook_result.tool_messages is not None:
                            tool_messages = hook_result.tool_messages
                        if hasattr(hook_result, 'extra_messages') and hook_result.extra_messages:
                            tool_messages = tool_messages + hook_result.extra_messages
                except Exception:
                    logger.warning('Post-iteration hook failed', exc_info=True)

            messages.extend(tool_messages)

            # NOTE: no post-tool-loop observation-timeout check here — swe_agent
            # only checks the total observation timeout after each tool result
            # (agent.py:514-520), which the in-loop `>=` check above already
            # mirrors. A trailing `>` re-check would diverge from swe_agent.

    except ContextLengthExceededError as e:
        logger.error('model context length exceeded: %s, exiting.', e)
        _log_cache_summary(
            _total_input_tokens, _total_output_tokens, _total_cache_read, _total_cache_creation, _total_cached_tokens
        )
        return CodeMTAgentOutput(
            exit_status='truncation', iterations=agent_input.max_iterations, main_trajectory=messages
        )
    except Exception as e:
        logger.exception('Agent encountered error: %s', e)
        _log_cache_summary(
            _total_input_tokens, _total_output_tokens, _total_cache_read, _total_cache_creation, _total_cached_tokens
        )
        return CodeMTAgentOutput(exit_status='error', iterations=agent_input.max_iterations, main_trajectory=messages)
    _log_cache_summary(
        _total_input_tokens, _total_output_tokens, _total_cache_read, _total_cache_creation, _total_cached_tokens
    )
    return CodeMTAgentOutput(
        exit_status='max_iterations', iterations=agent_input.max_iterations, main_trajectory=messages
    )
