"""Text-parse mode loop for code_mt_agent.

Self-contained: components (Chat, Parser, Summarization, Terminal Output)
live under text_parse_helpers/, copied from terminus2 at refactor time.
code_mt has no import dependencies on sibling agent packages.

Context-management (inference retry, ContextLength/OutputLength handling,
parse-error / warning feedback bytes, two-stage completion) is copied
BYTE-FOR-BYTE from terminus2 agent.py so a text_parse trajectory is
identical to the terminus2 agent it replaces. The ONLY code_mt additions are
the cross-turn MT resume path (terminus2 has no MT) and the MESSAGES_STATE_KEY
stash (so host-side monitors / the RepoQA answer extractor see this trajectory).
"""

from __future__ import annotations

import asyncio
import logging
import time

from dataclasses import dataclass
from typing import Literal

from tenacity import retry, retry_if_exception_type, retry_if_not_exception_type, stop_after_attempt

from nexus.core import require_context
from nexus.core.openai_client import ContextLengthExceededError
from nexus.extensions.projects.code_mt.agent.models import CodeMTAgentInput, CodeMTAgentOutput
from nexus.trace import get_current_span_id, span_context, update_span
from nexus.utils.timeout import InferenceTimeoutError, TimeBudget, with_timeout


logger = logging.getLogger(__name__)


@dataclass
class Command:
    keystrokes: list[str]
    duration_sec: float


def _format_initial_terminal_state(terminal_state: str) -> str:
    """Strip initialization lines from terminal output."""
    import re

    pattern = r'(Current Terminal Screen:\n)(?:.*\n)*?.*export PS1=.*\n'
    return re.sub(pattern, r'\1\n', terminal_state)


def _get_parser(parser_name: Literal['json', 'xml'] = 'json'):
    """Return the appropriate parser instance for the given format."""
    from nexus.extensions.projects.code_mt.agent.text_parse_helpers.parsing import TerminusJSONParser
    from nexus.extensions.projects.code_mt.agent.text_parse_helpers.xml_parsing import TerminusXMLParser

    if parser_name == 'xml':
        return TerminusXMLParser()
    return TerminusJSONParser()


def _unwind_messages_to_free_tokens(
    chat,
    target_free_tokens: int = 4000,
    model_name: str = '',
    context_limit: int | None = None,
) -> None:
    """Remove recent messages until we have enough free tokens.

    Copied from terminus2 agent.py:182 (operates on the live Chat, not a raw
    history list — this is the terminus2 variant, distinct from the helper
    module's history-list `unwind_messages_to_free_tokens`).
    """
    from nexus.extensions.projects.code_mt.agent.text_parse_helpers.chat import count_total_tokens
    from nexus.extensions.projects.code_mt.agent.text_parse_helpers.summarization import get_model_context_limit

    context_limit = context_limit or get_model_context_limit(model_name)

    while len(chat.messages) > 1:  # Keep at least the first message
        current_tokens = count_total_tokens(chat, model_name)
        free_tokens = context_limit - current_tokens

        if free_tokens >= target_free_tokens:
            break

        # Remove the most recent pair of messages (user + assistant)
        if len(chat.messages) >= 2:
            chat._messages = chat.messages[:-2]
        else:
            break

    free_tokens = context_limit - count_total_tokens(chat, model_name)
    logger.debug(
        'Unwound messages. Remaining messages: %d, Free tokens: approximately %d', len(chat.messages), free_tokens
    )


async def _query_llm(
    chat,
    prompt: str,
    original_instruction: str = '',
    model_name: str = '',
    enable_summarize: bool = True,
    session_id: str = '',
    parser_name: Literal['json', 'xml'] = 'json',
    _output_length_retry: int = 0,
    context_limit: int | None = None,
):
    """Copied from terminus2 agent.py:218 (_query_llm), byte-for-byte semantics.

    Handles: ContextLengthExceeded (3-tier summarization fallback + unwind),
    OutputLengthExceeded (XML salvage, then append truncated turn + retry up to
    3× with a length-error re-prompt). The tenacity `@retry` on generic
    exceptions is applied via the local `_do` wrapper below (equivalent to the
    decorator on terminus2's _query_llm — retries transient inference errors
    3×, does NOT retry ContextLength/OutputLength).
    """
    from nexus.extensions.projects.code_mt.agent.text_parse_helpers.chat import (
        LLMResponse,
        OutputLengthExceededError,
        lite_llm_call,
    )
    from nexus.extensions.projects.code_mt.agent.text_parse_helpers.summarization import (
        get_model_output_limit,
        summarize,
    )
    from nexus.extensions.projects.code_mt.agent.text_parse_helpers.xml_parsing import TerminusXMLParser

    @retry(
        stop=stop_after_attempt(3),
        retry=(
            retry_if_not_exception_type((ContextLengthExceededError, OutputLengthExceededError, InferenceTimeoutError))
            & retry_if_exception_type(Exception)
        ),
        reraise=True,
    )
    async def _do_chat(p: str):
        return await chat.chat(p)

    try:
        return await _do_chat(prompt)

    except ContextLengthExceededError:
        if not enable_summarize:
            logger.debug('Context length exceeded and summarization is OFF.')
            raise
        logger.debug('Context length exceeded. Using fallback summarization.')

        _unwind_messages_to_free_tokens(
            chat, target_free_tokens=4000, model_name=model_name, context_limit=context_limit
        )

        summary_prompt = None

        # Fallback 1: Try full summary (three-step summarization process)
        try:
            logger.debug('SUMMARIZATION: Attempting full summary')
            summary_prompt, _ = await summarize(chat, original_instruction, session_id)
            logger.debug('SUMMARIZATION: Full summary succeeded')
        except InferenceTimeoutError:
            raise
        except Exception as e:
            logger.debug('SUMMARIZATION: Full summary failed: %s', e)

        # Fallback 2: Try short summary
        if summary_prompt is None:
            try:
                logger.debug('SUMMARIZATION: Attempting short summary')
                runtime = await require_context().get_runtime()
                current_screen = (
                    await runtime.capture_terminal_pane(session_id=session_id, capture_entire=False)
                ).content
                limited_screen = current_screen[-1000:] if current_screen else ''

                short_prompt = (
                    f'Briefly continue this task: {original_instruction}\n\n'
                    f'Current state: {limited_screen}\n\nNext steps (2-3 sentences):'
                )

                short_llm_response = await lite_llm_call(
                    prompt=short_prompt,
                    inference_budget=chat.inference_budget,
                )

                summary_prompt = f'{original_instruction}\n\nSummary: {short_llm_response.content}'
                logger.debug('SUMMARIZATION: Short summary succeeded')
            except InferenceTimeoutError:
                raise
            except Exception as e:
                logger.debug('SUMMARIZATION: Short summary failed: %s', e)

        # Fallback 3: Ultimate fallback (no LLM)
        if summary_prompt is None:
            logger.debug('SUMMARIZATION: Using ultimate fallback')
            runtime = await require_context().get_runtime()
            current_screen = (await runtime.capture_terminal_pane(session_id=session_id, capture_entire=False)).content
            limited_screen = current_screen[-1000:] if current_screen else ''
            summary_prompt = f'{original_instruction}\n\nCurrent state: {limited_screen}'

        try:
            llm_response = await chat.chat(summary_prompt)
        except InferenceTimeoutError:
            raise
        except Exception as e:
            logger.exception('Even fallback LLM call failed: %s', e)
            llm_response = LLMResponse(content='Technical difficulties. Please continue with the task.')

        return llm_response

    except OutputLengthExceededError as e:
        logger.debug('Output length exceeded: %s', e)

        if _output_length_retry >= 3:
            logger.warning('Output length exceeded retry limit, raising.')
            raise

        truncated_response = getattr(e, 'truncated_response', '[TRUNCATED RESPONSE NOT AVAILABLE]')
        truncated_message = getattr(e, 'raw_message', None)
        if not isinstance(truncated_message, dict):
            truncated_message = {'role': 'assistant', 'content': truncated_response}

        # Try to salvage a valid response from truncated output (XML parser only)
        parser = _get_parser(parser_name)
        if isinstance(parser, TerminusXMLParser):
            salvaged_response, _has_multiple_blocks = parser.salvage_truncated_response(truncated_response)
            if salvaged_response:
                logger.debug('Output exceeded length but found valid response via salvage.')
                chat.messages.append({'role': 'user', 'content': prompt})
                chat.messages.append(truncated_message)
                return LLMResponse(
                    content=salvaged_response,
                    raw_message=truncated_message,
                )

        warnings_text = ''
        try:
            parse_result = parser.parse_response(truncated_response)
            if parse_result.warning:
                warnings_text = f'\n\nParser warnings from your truncated response:\n{parse_result.warning}'
        except Exception as parse_error:
            logger.debug('Failed to parse truncated response: %s', parse_error)

        output_limit = get_model_output_limit(model_name)
        limit_str = f'{output_limit} tokens' if output_limit is not None else 'the maximum output length'

        error_msg = (
            'ERROR!! NONE of the actions you just requested were performed '
            f'because you exceeded {limit_str}. '
            f'Your outputs must be less than {limit_str}. Re-issue this request, '
            f'breaking it into chunks each of which is less than {limit_str}.'
        )

        if warnings_text:
            error_msg += warnings_text

        chat.messages.append({'role': 'user', 'content': prompt})
        chat.messages.append(truncated_message)

        return await _query_llm(
            chat,
            error_msg,
            original_instruction,
            model_name,
            enable_summarize,
            session_id,
            parser_name=parser_name,
            _output_length_retry=_output_length_retry + 1,
            context_limit=context_limit,
        )

    except Exception as e:
        logger.exception('Unknown Error in LLM interaction: %s', e)
        raise


async def _handle_llm_interaction(
    chat,
    prompt: str,
    original_instruction: str = '',
    model_name: str = '',
    enable_summarize: bool = True,
    session_id: str = '',
    parser_name: Literal['json', 'xml'] = 'json',
    context_limit: int | None = None,
):
    """Copied from terminus2 agent.py:350. Returns (commands, is_task_complete,
    feedback, analysis, plan, llm_response)."""
    llm_response = await _query_llm(
        chat,
        prompt,
        original_instruction,
        model_name,
        enable_summarize,
        session_id,
        parser_name=parser_name,
        context_limit=context_limit,
    )

    parser = _get_parser(parser_name)
    result = parser.parse_response(llm_response.content)

    # Build feedback from errors and warnings
    feedback = ''
    if result.error:
        feedback += f'ERROR: {result.error}'
        if result.warning:
            feedback += f'\nWARNINGS: {result.warning}'
    elif result.warning:
        feedback += f'WARNINGS: {result.warning}'

    if result.warning:
        logger.warning('Parser warnings: %s', result.warning)

    # Convert parsed commands to Command objects
    commands: list[Command] = []
    for parsed_cmd in result.commands:
        commands.append(
            Command(
                keystrokes=parsed_cmd.keystrokes,
                duration_sec=min(parsed_cmd.duration, 60),
            )
        )

    return (
        commands,
        result.is_task_complete,
        feedback,
        result.analysis,
        result.plan,
        llm_response,
    )


async def text_parse_loop(agent_input: CodeMTAgentInput) -> CodeMTAgentOutput:
    """Main loop for text_parse mode (self-contained, see text_parse_helpers/).

    Mirrors terminus2 _run_agent_loop; the MT-resume path + MESSAGES_STATE_KEY
    stash are the only code_mt-specific additions.
    """
    from nexus.extensions.projects.code_mt.agent.text_parse_helpers.chat import Chat
    from nexus.extensions.projects.code_mt.agent.text_parse_helpers.prompts import (
        get_response_type_label,
        get_system_template,
    )
    from nexus.extensions.projects.code_mt.agent.text_parse_helpers.summarization import (
        check_proactive_summarization,
        get_model_context_limit,
    )
    from nexus.extensions.projects.code_mt.agent.text_parse_helpers.terminal_output import (
        Terminus2State,
        get_completion_confirmation_message,
        get_incremental_output,
        limit_output_length,
    )

    ctx = require_context()
    runtime = await ctx.get_runtime()
    span_id = get_current_span_id()

    # 1. Create terminal session
    session = await runtime.create_terminal_session(
        height=agent_input.tmux_pane_height,
        width=agent_input.tmux_pane_width,
    )
    session_id = session.session_id

    # 2. Initialize PS1 and read initial state
    await runtime.send_keys_to_terminal_session(
        session_id=session_id,
        keys=[r"export PS1='\u@\h:\w\$ '", 'Enter'],
    )
    await asyncio.sleep(1)

    state = Terminus2State()
    # Wrap the initial terminal dump in limit_output_length, mirroring terminus2
    # (agent.py:631). Without this a huge boot-time terminal buffer would enter
    # the opening prompt uncapped — a real context-length divergence.
    terminal_state = limit_output_length(await get_incremental_output(session_id, state))
    terminal_state = _format_initial_terminal_state(terminal_state)

    # 3. Build this turn's opening prompt. On the FIRST MT turn we wrap the
    # dataset instruction + initial terminal screen in the system template. On a
    # RESUMED turn (mt_max_turns>1) the system instructions already live in the
    # carried-over history, so we must NOT re-wrap: user_prompt is the dataset
    # mt_feedback and is appended as a plain continuing user turn, together with
    # the current terminal screen so the model sees fresh state.
    from nexus.extensions.projects.code_mt.agent.tool_call_loop import MESSAGES_STATE_KEY

    prior_messages = ctx.tool_state.get(MESSAGES_STATE_KEY)
    is_resume = bool(prior_messages)
    if is_resume:
        opening_prompt = f'{agent_input.user_prompt}\n\n{terminal_state}' if agent_input.user_prompt else terminal_state
    else:
        system_template = get_system_template(agent_input.parser_name)
        opening_prompt = system_template.format(
            instruction=agent_input.user_prompt,
            terminal_state=terminal_state,
        )

    parser_name = agent_input.parser_name
    response_type_label = get_response_type_label(parser_name)
    original_instruction = (
        agent_input.user_prompt if isinstance(agent_input.user_prompt, str) else str(agent_input.user_prompt)
    )

    # 5. Create Chat instance
    inference_budget = TimeBudget(agent_input.inference_time_timeout)
    chat = Chat(inference_budget=inference_budget)
    # On a RESUMED MT turn continue the SAME conversation the previous turn(s)
    # built, so the model sees the full cross-turn trajectory (real multi-turn).
    # The HyTuner client persists across MT turns, so replaying the identical
    # prefix keeps the token prefix stable (no prefix_shortened -> no R3
    # routed_experts reset). We copy the prior messages into this turn's Chat;
    # the previous turn's list object is left untouched.
    if is_resume:
        chat._messages.extend(prior_messages or [])
    # Stash the live conversation under the same key the tool_call loop uses
    # (MESSAGES_STATE_KEY = '_code_mt_messages') so the host-side reward-hacking /
    # baseline monitors AND the RepoQA answer/transcript extractor see the
    # text_parse trajectory too. chat.messages returns the live self._messages
    # list, and chat.chat() appends to it in place, so this reference stays
    # current for the whole turn. This also becomes the carried-over history for
    # the NEXT MT turn's resume above.
    ctx.tool_state[MESSAGES_STATE_KEY] = chat.messages

    # Resolve the effective context limit for logging/visibility, mirroring
    # terminus2 (agent.py:434-439): fall back to the model's default window when
    # the dataset leaves context_limit unset.
    resolved_context_limit = agent_input.context_limit or get_model_context_limit(agent_input.model_name)
    logger.debug(
        'code_mt text_parse config: model_name=%r, context_limit=%s, resolved_context_limit=%s, '
        'enable_summarize=%s, proactive_threshold=%s',
        agent_input.model_name,
        agent_input.context_limit,
        resolved_context_limit,
        agent_input.enable_summarize,
        agent_input.proactive_summarization_threshold,
    )

    # 6. Main loop (mirrors terminus2 _run_agent_loop body)
    prompt = opening_prompt
    _pending_completion = False
    total_observation_time = 0.0
    total_inference_time = 0.0
    n_episode = -1

    # Resolve timeouts, mirroring terminus2 (agent.py:659-664): a missing
    # per_observation_timeout falls back to the total observation_time_timeout so
    # each observation is still bounded. Kept local so we never mutate the input.
    observation_time_timeout = agent_input.observation_time_timeout
    per_observation_timeout = agent_input.per_observation_timeout
    inference_time_timeout = agent_input.inference_time_timeout
    if per_observation_timeout is None:
        per_observation_timeout = observation_time_timeout

    # Set before the loop so the TimeoutError handler can attribute the partial
    # elapsed of an interrupted observation (mirrors terminus2:591-593 `if start_time`).
    start_time = None

    try:
        for n_episode in range(1, agent_input.max_iterations + 1):
            logger.debug('=== code_mt text_parse STEP %d ===', n_episode)

            # 6a. Proactive summarization
            if original_instruction and agent_input.enable_summarize:
                proactive_summary_result = await check_proactive_summarization(
                    chat,
                    original_instruction,
                    agent_input.model_name,
                    session_id,
                    threshold=agent_input.proactive_summarization_threshold,
                    context_limit=agent_input.context_limit,
                )
                if proactive_summary_result:
                    prompt, _ = proactive_summary_result

            # 6b. Inference + parse (ContextLength/OutputLength handled inside)
            with span_context('agent.inference', attributes={'type': 'agent_inference', 'iteration': n_episode}):
                t0 = time.monotonic()
                (
                    commands,
                    is_task_complete,
                    feedback,
                    _analysis,
                    _plan,
                    _llm_response,
                ) = await _handle_llm_interaction(
                    chat,
                    prompt,
                    original_instruction,
                    agent_input.model_name,
                    agent_input.enable_summarize,
                    session_id,
                    parser_name=parser_name,
                    context_limit=agent_input.context_limit,
                )
                total_inference_time += time.monotonic() - t0

            # 6c. Parse-error feedback → re-prompt, mirroring terminus2:497-505
            if feedback and 'ERROR:' in feedback:
                prompt = (
                    f'Previous response had parsing errors:\n{feedback}\n\n'
                    f'Please fix these issues and provide a proper {response_type_label}.'
                )
                continue

            # Inference timeout check (after parse-error skip so re-prompts don't
            # consume the budget — the re-prompt cycle reuses the same inference).
            if inference_time_timeout and total_inference_time > inference_time_timeout:
                if span_id:
                    update_span(span_id, attributes={'hytuner.inference_time': total_inference_time})
                return CodeMTAgentOutput(exit_status='timeout', iterations=n_episode, main_trajectory=chat.messages)

            # 6d. Execute commands
            start_time = time.monotonic()
            with span_context('agent.observation', attributes={'type': 'agent_observation', 'iteration': n_episode}):
                async with with_timeout(per_observation_timeout):
                    for command in commands:
                        await runtime.send_keys_to_terminal_session(session_id=session_id, keys=command.keystrokes)
                        await asyncio.sleep(command.duration_sec)
                    terminal_output = await get_incremental_output(session_id, state)
                    terminal_output = limit_output_length(terminal_output, agent_input.max_terminal_output)

            total_observation_time += time.monotonic() - start_time

            # 6e. Timeout check
            if observation_time_timeout and total_observation_time > observation_time_timeout:
                if span_id:
                    update_span(span_id, attributes={'hytuner.observation_time': total_observation_time})
                return CodeMTAgentOutput(exit_status='timeout', iterations=n_episode, main_trajectory=chat.messages)

            # 6f. is_task_complete + WARNINGS observation, mirroring terminus2:555-585
            was_pending_completion = _pending_completion

            if is_task_complete:
                if was_pending_completion:
                    observation = terminal_output
                else:
                    _pending_completion = True
                    observation = get_completion_confirmation_message(terminal_output, parser_name)
            else:
                _pending_completion = False
                if feedback and 'WARNINGS:' in feedback:
                    observation = (
                        f'Previous response had warnings:\n{feedback}\n\n'
                        f'{limit_output_length(terminal_output, agent_input.max_terminal_output)}'
                    )
                else:
                    observation = limit_output_length(terminal_output, agent_input.max_terminal_output)

            if is_task_complete:
                if was_pending_completion:
                    if span_id:
                        update_span(span_id, attributes={'hytuner.observation_time': total_observation_time})
                    return CodeMTAgentOutput(
                        exit_status='submitted', iterations=n_episode, main_trajectory=chat.messages
                    )
                prompt = observation
                continue

            prompt = observation

        if span_id:
            update_span(span_id, attributes={'hytuner.observation_time': total_observation_time})
        return CodeMTAgentOutput(
            exit_status='max_iterations', iterations=agent_input.max_iterations, main_trajectory=chat.messages
        )
    except ContextLengthExceededError as e:
        logger.error('Context length exceeded after all summarization fallbacks: %s', e)
        if span_id:
            update_span(span_id, attributes={'hytuner.observation_time': total_observation_time})
        return CodeMTAgentOutput(
            exit_status='truncation',
            iterations=n_episode if n_episode > 0 else agent_input.max_iterations,
            main_trajectory=chat.messages,
        )
    except InferenceTimeoutError as e:
        logger.warning('Inference timeout reached: %s', e)
        if span_id:
            update_span(span_id, attributes={'hytuner.observation_time': total_observation_time})
        return CodeMTAgentOutput(
            exit_status='timeout',
            iterations=n_episode if n_episode > 0 else agent_input.max_iterations,
            main_trajectory=chat.messages,
        )
    except (TimeoutError, asyncio.TimeoutError) as e:
        # Attribute the partial elapsed of the interrupted observation, mirroring
        # terminus2 (agent.py:591-593): the timeout fires inside the with_timeout
        # block before start_time is cleared, so add the time spent so far.
        if start_time:
            total_observation_time += time.monotonic() - start_time
        logger.warning('Observation timeout reached: %s', e)
        if span_id:
            update_span(span_id, attributes={'hytuner.observation_time': total_observation_time})
        return CodeMTAgentOutput(
            exit_status='timeout',
            iterations=n_episode if n_episode > 0 else agent_input.max_iterations,
            main_trajectory=chat.messages,
        )
    except Exception as e:
        logger.exception('Unknown error in text_parse loop: %s', e)
        if span_id:
            update_span(span_id, attributes={'hytuner.observation_time': total_observation_time})
        return CodeMTAgentOutput(
            exit_status='error',
            iterations=n_episode if n_episode > 0 else agent_input.max_iterations,
            main_trajectory=chat.messages,
        )
