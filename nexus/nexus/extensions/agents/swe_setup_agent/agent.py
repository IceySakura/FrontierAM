"""SWE Setup Agent implementation (In-Place Mode).

This agent configures development environments by installing dependencies directly
in the runtime container (in-place), rather than generating Dockerfiles.

The agent sets up the environment, verifies fail2pass, writes /tmp/eval.sh, and
calls submit_eval. After submit_eval succeeds the agent exits and the task is done.
"""

import asyncio
import json
import logging
import re
import time

from dataclasses import dataclass

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ContextLengthExceededError,
    Registry,
    ToolExecutor,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)
from nexus.extensions.agents.swe_setup_agent.prompts import (
    SYSTEM_PROMPT,
    get_user_prompt,
)
from nexus.trace import get_current_span_id, span_context, start_span, update_span
from nexus.trace.types import Status


DEFAULT_AGENT_TIMELINE_PREFIX = 'hytuner_agent/'

# Import tools to ensure registration
Registry.import_module('nexus.extensions.agents.swe_setup_agent.tools')


logger = logging.getLogger(__name__)


# ============== Agent Loop Configuration ==============


@dataclass
class AgentLoopConfig:
    """Centralized configuration for all agent loop parameters.

    All circuit breaker thresholds, stuck-detection limits, and context window
    settings are defined here. Change values in ONE place instead of hunting
    through the loop body.
    """

    # --- Circuit breakers (submit_eval failure handling) ---
    max_submit_failures: int = 30  # Total submit_eval attempts before giving up
    max_pass2pass: int = 5  # Cumulative pass2pass results before giving up
    max_golden_patch_failures: int = 15  # Consecutive identical golden_patch apply failures
    max_full_state_failures: int = 15  # Cumulative full-state failures before giving up
    full_state_hint_threshold: int = 5  # Consecutive identical full-state failures → inject diagnostic hint

    # --- Stuck detection ---
    max_no_tool_call_retries: int = 3  # Consecutive no-tool-call responses before aborting
    max_same_bash: int = 8  # Consecutive identical bash commands → inject loop-break hint
    max_disconnect_errors: int = 3  # Consecutive runtime disconnections before aborting

    # --- Context window management ---
    context_trim_threshold: int = 1_000_000  # Prompt tokens threshold (kept for config compat, not used)


# Singleton config — override by subclassing or replacing before agent_run
AGENT_LOOP_CONFIG = AgentLoopConfig()

EXIT_STATUS_TO_TERMINATION_REASON: dict[str, str] = {
    'submit_eval_success': 'ENV_DONE',
    'max_iterations': 'MAX_STEPS',
    'no_tool_calls_incomplete': 'FORMAT_ERROR',
    'response_truncated': 'TRUNCATION',
    'submit_eval_failed': 'ENV_DONE',
    'pass2pass_exhausted': 'ENV_DONE',
    'llm_error': 'UNHANDLED_ERROR',
    'runtime_disconnected': 'ENV_ERROR',
    'context_length_exceeded': 'TRUNCATION',
    'observation_timeout': 'TASK_TIMEOUT',
}


# ============== Input/Output Models ==============


class SWESetupAgentInput(AgentRunInput, WithUserPrompt):
    """Input for SWE Setup Agent."""

    # Instance data
    repo: str = Field(description='Repository name (owner/repo)')
    base_commit: str = Field(description='Base commit hash')
    language: str = Field(description='Programming language')
    problem_statement: str = Field(description='Problem statement / issue description')
    test_patch: str = Field(default='', description='Test patch content')
    golden_patch: str = Field(default='', description='Golden patch content (the fix)')

    # Environment config
    testbed_dir: str = Field(default='/testbed', description='Testbed directory path')
    base_image: str = Field(default='ubuntu-git:22.04', description='Current base image')

    # Phase control (kept for API compatibility but ignored)
    phase: int = Field(default=1, description='Reserved for compatibility (unused)')

    # Agent config
    max_iterations: int = Field(default=50, description='Maximum iterations')
    timeout: int = Field(default=3600, description='Timeout in seconds')
    single_tool_call: bool = Field(default=False, description='If True, instruct agent to call only one tool per turn')


class SWESetupAgentOutput(AgentRunOutput):
    """Output for SWE Setup Agent."""

    exit_status: str = Field(default='success', description='Exit status')
    iterations: int = Field(default=0, description='Number of iterations')
    phase_completed: int = Field(default=0, description='Final phase completed')


# ============== Agent Implementation ==============


@agent_registry.register('swe_setup_agent')
async def swe_setup_agent(agent_input: SWESetupAgentInput) -> SWESetupAgentOutput:
    """SWE Setup Agent - Configure environment and reproduce issues (in-place mode).

    - Set up the environment in-place
    - Write /tmp/eval.sh (targeted fail2pass script)
    - Call submit_eval — agent exits on success
    """
    ctx = require_context()
    client = ctx.openai_client

    from nexus.extensions.tasks.swe_builder.constants import TaskPhase, format_available_images_list

    # Get available images
    available_images = format_available_images_list(agent_input.language)

    # ---- Phase 0: Auto Setup (deterministic, no agent iterations) ----
    from nexus.extensions.agents.swe_setup_agent.tools import auto_setup

    auto_setup_result = None
    try:
        runtime = await ctx.get_runtime('main')
        auto_setup_result = await auto_setup(ctx, runtime, ctx.task_config)
        logger.info(f'[SWE_SETUP_AGENT] Auto setup completed in {auto_setup_result.setup_duration_s}s')
        # Update base_image if auto_setup switched it
        if auto_setup_result.image_switched:
            agent_input.base_image = auto_setup_result.selected_image
    except Exception as e:
        logger.error(f'[SWE_SETUP_AGENT] Auto setup failed (agent will explore manually): {e}', exc_info=True)

    # Agent tools: no explore_repo (auto_setup already did it),
    # keep switch_runtime_image for manual correction
    executor = ToolExecutor([
        tool_registry.get('swe_setup_agent/bash'),
        tool_registry.get('swe_setup_agent/str_replace_editor'),
        tool_registry.get('swe_setup_agent/switch_runtime_image'),
        tool_registry.get('swe_setup_agent/submit_eval'),
    ])

    user_prompt = get_user_prompt(
        repo=agent_input.repo,
        base_commit=agent_input.base_commit,
        language=agent_input.language,
        testbed_dir=agent_input.testbed_dir,
        base_image=agent_input.base_image,
        problem_statement=agent_input.problem_statement,
        test_patch=agent_input.test_patch,
        golden_patch=agent_input.golden_patch,
        available_images=available_images,
        auto_setup_result=auto_setup_result,
    )

    if agent_input.single_tool_call:
        user_prompt += (
            '\n\n---\n\n'
            '**⚠️ IMPORTANT: Call only ONE tool per turn. Do NOT call multiple tools in a single response. '
            'Wait for the result of each tool call before proceeding to the next step.**'
        )

    logger.debug('[SWE_SETUP_AGENT] Starting Environment Setup')

    logger.debug(f'[SWE_SETUP_AGENT] Repo: {agent_input.repo}')
    logger.debug(f'[SWE_SETUP_AGENT] Base image: {agent_input.base_image}')
    logger.debug(f'[SWE_SETUP_AGENT] Max iterations: {agent_input.max_iterations}')

    # Build conversation history
    system_prompt = {'role': 'system', 'content': SYSTEM_PROMPT}
    history = [system_prompt, {'role': 'user', 'content': user_prompt}]

    # full_history is never trimmed — it captures the complete conversation for export.
    full_history: list[dict] = list(history)

    # Get tool schemas
    tools = executor.get_tool_schemas()

    # Agent loop config — all thresholds from centralized config
    cfg = AGENT_LOOP_CONFIG
    _max_submit_override = ctx.task_config.get('max_submit_attempts')
    if _max_submit_override is not None and _max_submit_override != cfg.max_submit_failures:
        from dataclasses import replace

        cfg = replace(cfg, max_submit_failures=_max_submit_override)

    exit_status = None
    iterations = 0
    submit_eval_failures = 0

    # Circuit breaker counters
    consecutive_fail2pass_failures = 0
    last_fail2pass_error = None
    consecutive_golden_patch_failures = 0
    last_golden_patch_error = None
    cumulative_pass2pass_count = 0
    cumulative_full_state_fail_count = 0

    # Stuck detection counters
    consecutive_no_tool_calls = 0
    last_bash_command: str | None = None
    consecutive_same_bash = 0
    consecutive_disconnect_errors = 0

    # Observation time tracking (aligned with swe_agent)
    total_observation_time = 0.0
    observation_time_timeout = ctx.task_config.get('observation_time_timeout', None)
    per_observation_timeout = ctx.task_config.get('per_observation_timeout', None)
    if per_observation_timeout is None:
        per_observation_timeout = observation_time_timeout
    timeline_collect_tool_args = bool(ctx.task_config.get('timeline_collect_tool_args', False))

    for iteration in range(agent_input.max_iterations):
        iterations = iteration + 1

        # Call LLM
        logger.debug(f'[SWE_SETUP_AGENT] Iteration {iteration + 1}/{agent_input.max_iterations}')

        _gen_name = f'{DEFAULT_AGENT_TIMELINE_PREFIX}gen'
        try:
            with span_context(
                _gen_name,
                attributes={'type': 'agent_inference', 'timeline_type': 'gen'},
            ):
                response = await client.create_completion(
                    messages=history,
                    tools=tools or None,
                )
            message = response.choices[0].message
        except ContextLengthExceededError as e:
            logger.error(f'[SWE_SETUP_AGENT] Context length exceeded, exiting: {e}')
            exit_status = 'context_length_exceeded'
            break
        except Exception as e:
            msg_types = [(m.get('role'), type(m.get('content')).__name__) for m in history]
            logger.error(
                f'[SWE_SETUP_AGENT] LLM error: {e} | iter={iteration + 1} msgs={len(history)} msg_types={msg_types}'
            )
            exit_status = 'llm_error'
            break

        # Add assistant message to history
        assistant_msg = message.model_dump(exclude_none=True)
        history.append(assistant_msg)
        full_history.append(assistant_msg)

        # Log assistant response
        if message.content:
            logger.debug(f'[SWE_SETUP_AGENT] Assistant: {message.content[:200]}...')

        # Read before the branch below: a turn can be cut at the response budget either before
        # reaching a tool call or midway through one, and both paths need this.
        finish_reason = getattr(response.choices[0], 'finish_reason', None) if response.choices else None

        # Check if no tool calls
        if not message.tool_calls:
            phase = ctx.tool_state.get('task_phase', TaskPhase.SETUP)
            if phase >= TaskPhase.EVAL_SUBMITTED:
                # submit_eval succeeded — task is done
                logger.info('[SWE_SETUP_AGENT] No tool calls, submit_eval completed — exiting')
                exit_status = 'submit_eval_success'
                break

            consecutive_no_tool_calls += 1
            logger.warning(
                f'[SWE_SETUP_AGENT] No tool calls ({consecutive_no_tool_calls}/{cfg.max_no_tool_call_retries})'
            )

            # Detect if output was likely truncated (very long content, possible max_tokens hit)
            content_len = len(message.content or '')
            assistant_text = message.content or ''

            # Detect self-review pattern: agent outputs "approved: false" JSON instead of calling a tool
            _is_self_review = '"approved"' in assistant_text and 'false' in assistant_text.lower()

            if _is_self_review:
                logger.warning(
                    '[SWE_SETUP_AGENT] Detected self-review pattern — agent reviewing its own eval script instead of calling a tool'
                )
                reminder = (
                    '⚠️ STOP: You are reviewing/judging your own eval script instead of calling a tool. '
                    'Do NOT output approval/rejection JSON — that is not your job. '
                    'You have two choices:\n'
                    '1. If you have written /tmp/eval.sh, call submit_eval immediately and let the system validate it.\n'
                    '2. If you need to fix /tmp/eval.sh first, call bash or str_replace_editor to make the fix.\n'
                    'Do NOT analyze or judge the script yourself. Call a tool NOW.'
                )
            elif finish_reason == 'length' or content_len > 12000:
                logger.warning(
                    f'[SWE_SETUP_AGENT] Response may be truncated (finish_reason={finish_reason}, content_len={content_len})'
                )
                reminder = (
                    '⚠️ Your last response was too long and may have been cut off before reaching a tool call. '
                    'Call a tool now — use bash, str_replace_editor, or submit_eval to continue solving the problem. '
                    'Do not output long explanations; act directly.'
                )
            else:
                reminder = (
                    '⚠️ You did not call any tool in your last response. '
                    'Continue working on solving the problem — call bash, str_replace_editor, or submit_eval. '
                    'Do not output analysis or explanations without a tool call.'
                )

            if consecutive_no_tool_calls < cfg.max_no_tool_call_retries:
                history.append({'role': 'user', 'content': reminder})
                full_history.append({'role': 'user', 'content': reminder})
                continue
            logger.error('[SWE_SETUP_AGENT] No tool calls after retries — giving up')
            # A final turn cut at the response budget never got the chance to reach a tool
            # call, which is a length problem rather than the formatting one FORMAT_ERROR
            # reports. The content_len heuristic above is only good enough to pick a
            # reminder, so the label follows finish_reason alone.
            exit_status = 'response_truncated' if finish_reason == 'length' else 'no_tool_calls_incomplete'
            break

        # Execute tools
        all_tools_succeeded = True
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            consecutive_no_tool_calls = 0  # reset: agent is calling a tool this turn

            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f'[SWE_SETUP_AGENT] Failed to parse tool arguments: {e}')
                tool_output = f'Error: Invalid JSON in tool arguments: {e}'
                all_tools_succeeded = False
                if finish_reason == 'length':
                    # The arguments are unparseable because generation was cut mid-JSON, the same
                    # length problem the no-tool-calls branch above reports. Without this the turn
                    # looks like a tool failure, the loop keeps going, and the run ends as
                    # MAX_STEPS. The loop exits at the `if exit_status` check after this for-body,
                    # so the tool result below is still appended and the history stays complete.
                    logger.error('[SWE_SETUP_AGENT] Tool arguments cut at the response budget — giving up')
                    exit_status = 'response_truncated'
            else:
                logger.debug(f'[SWE_SETUP_AGENT] Calling tool: {tool_name}')
                logger.debug(f'[SWE_SETUP_AGENT] Tool args: {json.dumps(tool_args, ensure_ascii=False)}')

                # Detect stuck-loop: same bash command repeated too many times
                if tool_name in ('bash', 'swe_setup_agent/bash'):
                    _cmd = tool_args.get('command', '').strip()
                    if _cmd and _cmd == last_bash_command:
                        consecutive_same_bash += 1
                        if consecutive_same_bash >= cfg.max_same_bash:
                            logger.warning(
                                f'[SWE_SETUP_AGENT] Same bash command repeated {consecutive_same_bash}x: {_cmd[:80]!r} — injecting loop-break hint'
                            )
                            history.append({
                                'role': 'user',
                                'content': (
                                    f'⚠️ You have run the same command {consecutive_same_bash} times in a row with no progress:\n'
                                    f'  `{_cmd[:200]}`\n\n'
                                    'This approach is not working. Try a different strategy:\n'
                                    '- If trying to write/delete a file: use `str_replace_editor create` with `overwrite=True`\n'
                                    '- If waiting for something: stop and proceed to the next step\n'
                                    '- If the environment is broken: call `switch_runtime_image` to get a fresh container\n'
                                    '- If you have a working eval.sh: call `submit_eval` now instead of exploring further'
                                ),
                            })
                            full_history.append(history[-1])
                            consecutive_same_bash = 0
                    else:
                        consecutive_same_bash = 0
                    last_bash_command = _cmd
                else:
                    last_bash_command = None
                    consecutive_same_bash = 0

                # Execute tool – with graduated retry on transient server disconnections
                # "already running" gets 3 retries with increasing delays (a previous
                # long install may still be finishing); other transient errors get 1 retry.
                tool_response = None
                _max_transient_retries = 1
                _tool_start_time = time.time()
                tool_span_attrs: dict = {
                    'type': 'agent_observation',
                    'tool_name': tool_name,
                }
                if timeline_collect_tool_args:
                    tool_span_attrs['hytuner.tool_args'] = json.dumps(tool_args, ensure_ascii=False)
                tool_span = start_span('agent.observation', tool_span_attrs)
                try:
                    tool_response = await executor.call(tool_name, tool_args, timeout=per_observation_timeout)
                    consecutive_disconnect_errors = 0
                except Exception as e:
                    _e_lower = str(e).lower()
                    _is_busy = 'a command is already running' in _e_lower
                    _is_cmd_start_fail = 'command failed to start' in _e_lower
                    _is_transient = (
                        _is_busy
                        or _is_cmd_start_fail
                        or isinstance(e, (TimeoutError, asyncio.TimeoutError))
                        or 'server disconnected' in _e_lower
                        or 'connect call failed' in _e_lower
                        or 'clientconnectorerror' in _e_lower
                        or 'connection refused' in _e_lower
                        or 'timeouterror' in type(e).__name__.lower()
                    )
                    if _is_transient:
                        _max_transient_retries = 3 if (_is_busy or _is_cmd_start_fail) else 1
                        _retry_ok = False
                        for _retry_i in range(_max_transient_retries):
                            _retry_delay = 5.0 * (_retry_i + 1) if (_is_busy or _is_cmd_start_fail) else 5.0
                            logger.warning(
                                f'[SWE_SETUP_AGENT] Runtime transient error (tool={tool_name}), '
                                f'retry {_retry_i + 1}/{_max_transient_retries} in {_retry_delay}s... ({e})'
                            )
                            await asyncio.sleep(_retry_delay)
                            try:
                                tool_response = await executor.call(
                                    tool_name, tool_args, timeout=per_observation_timeout
                                )
                                consecutive_disconnect_errors = 0
                                logger.info('[SWE_SETUP_AGENT] Runtime transient error: retry succeeded')
                                _retry_ok = True
                                break
                            except Exception as retry_e:
                                e = retry_e
                                _re_lower = str(retry_e).lower()
                                _same_transient = (
                                    'a command is already running' in _re_lower
                                    or 'command failed to start' in _re_lower
                                )
                                if not _same_transient:
                                    break
                        if not _retry_ok:
                            if _is_busy or _is_cmd_start_fail:
                                logger.warning(
                                    f'[SWE_SETUP_AGENT] Transient tool error retry exhausted '
                                    f'(non-disconnect, tool={tool_name}): {e}'
                                )
                            else:
                                consecutive_disconnect_errors += 1
                                logger.error(
                                    f'[SWE_SETUP_AGENT] Runtime transient error retry exhausted '
                                    f'(consecutive={consecutive_disconnect_errors}/{cfg.max_disconnect_errors}): {e}'
                                )
                                if consecutive_disconnect_errors >= cfg.max_disconnect_errors:
                                    logger.error(
                                        '[SWE_SETUP_AGENT] Runtime server repeatedly unreachable, aborting task'
                                    )
                                    exit_status = 'runtime_disconnected'
                            tool_span.update(status=Status(code='ERROR', message=str(e)))
                            tool_output = f'Error executing tool: {e}'
                            all_tools_succeeded = False
                    else:
                        logger.error(f'[SWE_SETUP_AGENT] Tool execution error: {e}', exc_info=True)
                        tool_span.update(status=Status(code='ERROR', message=str(e)))
                        tool_output = f'Error executing tool: {str(e)}'
                        all_tools_succeeded = False
                finally:
                    total_observation_time += time.time() - _tool_start_time
                    tool_span.finish()

                if tool_response is not None:
                    tool_output = tool_response.output
                    if isinstance(tool_output, list):
                        tool_output = '\n'.join(part.get('text', '') for part in tool_output if isinstance(part, dict))
                    logger.debug(
                        '[SWE_SETUP_AGENT] Tool output (type=%s): %s...',
                        type(tool_output).__name__,
                        str(tool_output)[:200],
                    )

                    # Check if tool indicated completion
                    if tool_name in ('submit_eval', 'swe_setup_agent/submit_eval'):
                        phase_after = ctx.tool_state.get('task_phase', TaskPhase.SETUP)
                        tool_success = tool_response.meta.get('success', False)

                        if phase_after >= TaskPhase.EVAL_SUBMITTED and tool_success:
                            # Submit succeeded — task complete, exit agent.
                            # Append tool result before breaking so full_history is complete.
                            tool_msg = {
                                'role': 'tool',
                                'tool_call_id': tool_call.id,
                                'content': tool_output,
                            }
                            history.append(tool_msg)
                            full_history.append(tool_msg)
                            logger.info('[SWE_SETUP_AGENT] submit_eval succeeded, exiting agent')
                            exit_status = 'submit_eval_success'
                            break
                        if not tool_success:
                            # Submit failed
                            submit_eval_failures += 1
                            logger.warning(
                                f'[SWE_SETUP_AGENT] submit_eval failed ({submit_eval_failures}/{cfg.max_submit_failures})'
                            )
                            tool_output_str = str(tool_output)

                            # --- Extract current failure_type from structured output ---
                            # Format: "## submit_eval FAILED: {failure_type} (attempt ...)"
                            # IMPORTANT: Do NOT match against the full output string — the History
                            # section contains ALL past failure_types, causing false matches.
                            _ft_match = re.search(r'submit_eval FAILED: (\w+)', tool_output_str)
                            _current_failure_type = _ft_match.group(1) if _ft_match else ''

                            # --- Circuit breaker 1: pass2pass ---
                            # Test passes on base_commit too → test_patch has no fail2pass signal.
                            if _current_failure_type == 'base_state_passes':
                                cumulative_pass2pass_count += 1
                                consecutive_fail2pass_failures = 0
                                consecutive_golden_patch_failures = 0
                                logger.warning(
                                    f'[SWE_SETUP_AGENT] pass2pass detected ({cumulative_pass2pass_count}/{cfg.max_pass2pass}, cumulative)'
                                )
                                if cumulative_pass2pass_count >= cfg.max_pass2pass:
                                    logger.error(
                                        f'[SWE_SETUP_AGENT] {cfg.max_pass2pass} cumulative pass2pass — no fail2pass signal, giving up'
                                    )
                                    exit_status = 'pass2pass_exhausted'
                                    break

                            # --- Circuit breaker 2: golden_patch cannot apply ---
                            # This is a data-quality issue (patch incompatible with base_commit).
                            # The agent CANNOT fix this by modifying eval.sh. Stop after 3 identical errors.
                            # Matches both verbose and structured output formats.
                            elif _current_failure_type == 'golden_patch_apply_failed':
                                current_error = tool_output_str[:300]
                                consecutive_fail2pass_failures = 0
                                if current_error == last_golden_patch_error:
                                    consecutive_golden_patch_failures += 1
                                    logger.warning(
                                        f'[SWE_SETUP_AGENT] Consecutive golden_patch failures: {consecutive_golden_patch_failures}'
                                    )
                                    if consecutive_golden_patch_failures >= cfg.max_golden_patch_failures:
                                        logger.error(
                                            f'[SWE_SETUP_AGENT] golden_patch failed to apply {cfg.max_golden_patch_failures} times — data quality issue, giving up'
                                        )
                                        exit_status = 'submit_eval_failed'
                                        break
                                else:
                                    consecutive_golden_patch_failures = 1
                                    last_golden_patch_error = current_error

                            # --- Circuit breaker 3: full state consistently fails (same env error) ---
                            # Full state (base+golden_patch) can't pass → environment/setup issue.
                            # After 5 identical consecutive errors, inject a diagnostic hint.
                            # After 15 cumulative full-state failures, give up (env unrecoverable).
                            # Matches both verbose and structured output formats.
                            elif _current_failure_type == 'full_state_fails':
                                current_error = tool_output_str[:200]
                                consecutive_golden_patch_failures = 0
                                cumulative_full_state_fail_count += 1
                                logger.warning(
                                    f'[SWE_SETUP_AGENT] Full-state failure cumulative={cumulative_full_state_fail_count}/{cfg.max_full_state_failures}'
                                )
                                if current_error == last_fail2pass_error:
                                    consecutive_fail2pass_failures += 1
                                    logger.warning(
                                        f'[SWE_SETUP_AGENT] Consecutive full-state failures: {consecutive_fail2pass_failures}'
                                    )
                                    if consecutive_fail2pass_failures == cfg.full_state_hint_threshold:
                                        hint_message = (
                                            '\n\n⚠️ **CHECKPOINT: 5 consecutive identical full-state failures**\n\n'
                                            'The full state (base + golden_patch) keeps failing with the same error.\n'
                                            'This is an ENVIRONMENT problem, not a test problem.\n\n'
                                            '**Diagnose before resubmitting — run these diagnostic commands NOW:**\n'
                                            '1. `cat /tmp/eval_output.log 2>/dev/null | head -50` — read the FIRST error, not the last\n'
                                            '2. Identify root cause: build failure? missing command? DB not started? wrong test path?\n'
                                            '3. Make ONE targeted fix based on the diagnosis\n\n'
                                            '**Common fixes by error type:**\n'
                                            '- `command not found` / `No such file or directory` → source runtime env in eval.sh (NVM/venv/cargo/bundle)\n'
                                            '- DB connection error (PostgreSQL/MySQL/Redis) → add `service postgresql start; sleep 2` in eval.sh\n'
                                            '- Rails: `PendingMigrationError` / schema mismatch → add `bundle exec rails db:migrate RAILS_ENV=test` in eval.sh\n'
                                            '- Build error (`CMakeError`, `error[E`, `BUILD FAILURE`) → add build step before the test run\n'
                                            '- `No tests ran` / `0 examples` → wrong test path or filter\n\n'
                                            '**Do NOT resubmit the same eval.sh — make a concrete change first.**\n'
                                            'If the environment is fundamentally broken (Bazel network, Windows-only project), '
                                            'this instance may not be solvable. Stop and move on.'
                                        )
                                        history.append({'role': 'user', 'content': hint_message})
                                        full_history.append(history[-1])
                                        logger.debug('[SWE_SETUP_AGENT] Injected full-state failure hint')
                                else:
                                    consecutive_fail2pass_failures = 1
                                    last_fail2pass_error = current_error
                                if cumulative_full_state_fail_count >= cfg.max_full_state_failures:
                                    logger.error(
                                        f'[SWE_SETUP_AGENT] {cfg.max_full_state_failures} cumulative full-state failures — env unrecoverable, giving up'
                                    )
                                    exit_status = 'submit_eval_failed'
                                    break

                            else:
                                # Other failure (static check, workdir dirty, LLM validation, etc.)
                                consecutive_fail2pass_failures = 0
                                consecutive_golden_patch_failures = 0
                                last_fail2pass_error = None
                                last_golden_patch_error = None

                            if submit_eval_failures >= cfg.max_submit_failures:
                                logger.error(
                                    f'[SWE_SETUP_AGENT] submit_eval failed {cfg.max_submit_failures} times, giving up'
                                )
                                exit_status = 'submit_eval_failed'
                                break

            # Add tool result to history
            tool_msg = {
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': tool_output,
            }
            history.append(tool_msg)
            full_history.append(tool_msg)

            # Cumulative observation timeout check (after appending tool_msg to keep history complete)
            if observation_time_timeout and total_observation_time >= observation_time_timeout:
                logger.warning(
                    f'[SWE_SETUP_AGENT] Observation timeout: {observation_time_timeout}s, used {total_observation_time:.0f}s'
                )
                exit_status = 'observation_timeout'
                break

        # Check if we should exit after tool execution
        if exit_status:
            break

        # Check if all tools failed (agent might be stuck)
        if not all_tools_succeeded:
            logger.warning(f'[SWE_SETUP_AGENT] Some tools failed in iteration {iteration + 1}')

    # Determine final exit status
    if exit_status is None:
        exit_status = 'max_iterations'
        logger.warning(f'[SWE_SETUP_AGENT] Reached max iterations: {agent_input.max_iterations}')

    # Get final phase
    final_phase = ctx.tool_state.get('task_phase', TaskPhase.SETUP)

    # Store iteration count and full conversation in shared state for task output
    ctx.tool_state['agent_iters'] = iterations
    ctx.tool_state['main_trajectory'] = full_history

    logger.info(f'[SWE_SETUP_AGENT] Completed with status: {exit_status}')
    logger.info(f'[SWE_SETUP_AGENT] Total iterations: {iterations}')
    logger.info(f'[SWE_SETUP_AGENT] Final phase: {final_phase}')

    # Write HyTuner span attributes (aligned with swe_agent)
    termination_reason = EXIT_STATUS_TO_TERMINATION_REASON.get(exit_status, 'UNKNOWN')
    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={
                'hytuner.termination_reason': termination_reason,
                'hytuner.observation_time': total_observation_time,
            },
        )

    return SWESetupAgentOutput(
        exit_status=exit_status,
        iterations=iterations,
        phase_completed=final_phase,
    )
