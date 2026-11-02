"""SWE-agent Rubrics: an agent that writes evaluation rubrics for SWE-bench instances.

The agent explores a code repository, examines a gold patch, and generates
a rubrics.yaml file that can be used to evaluate other patches.
"""

import json
import logging

from enum import Enum

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    Registry,
    ToolExecutor,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)
from nexus.extensions.tasks.swe.rubrics.agentic_patch.prompts import (
    GEN_INSTANCE_TEMPLATE,
    GEN_NEXT_STEP_NO_OUTPUT,
    GEN_NEXT_STEP_TEMPLATE,
    GEN_SYSTEM_PROMPT,
)


logger = logging.getLogger(__name__)

# Register tools on import
Registry.import_module('nexus.extensions.tasks.swe.swe_agent_rubrics.tools')


# ============== Observation Truncation ==============

DEFAULT_TEXT_CONTENT_LIMIT = 50_000

DEFAULT_TRUNCATE_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of the full response has been shown to you.</NOTE>'
)


def maybe_truncate(
    content: str,
    truncate_after: int | None = None,
    truncate_notice: str = DEFAULT_TRUNCATE_NOTICE,
) -> str:
    """Truncate the middle of content if it exceeds the specified length."""
    if not truncate_after or len(content) <= truncate_after or truncate_after < 0:
        return content

    if len(truncate_notice) >= truncate_after:
        return truncate_notice[:truncate_after]

    available_chars = truncate_after - len(truncate_notice)
    head_chars = available_chars // 2 + (available_chars % 2)
    tail_chars = available_chars - head_chars

    return content[:head_chars] + truncate_notice + (content[-tail_chars:] if tail_chars > 0 else '')


# ============== Exit Status ==============


class ExitStatus(str, Enum):
    SUBMITTED = 'submitted'
    FINISHED = 'finished'
    MAX_ITERATIONS = 'max_iterations'
    TOOL_ERROR = 'tool_error'
    ERROR = 'error'


# ============== Input/Output Models ==============


class SWEAgentRubricsInput(AgentRunInput, WithUserPrompt):
    max_iterations: int = Field(default=50, description='Maximum number of agent steps')
    completion_retry_on_tool_error: int = Field(
        default=0,
        description='Number of retries when model output is invalid (no tool calls or invalid JSON). '
        '0 means no retry, fail immediately on first error.',
    )


class SWEAgentRubricsOutput(AgentRunOutput):
    exit_status: ExitStatus = Field(description='How the agent exited')
    iterations: int = Field(default=0, description='Number of iterations completed')


# ============== Agent Implementation ==============


@agent_registry.register('swe_agent_rubrics')
async def run_swe_agent_rubrics(agent_input: SWEAgentRubricsInput) -> SWEAgentRubricsOutput:
    """Run SWE-agent rubrics on a problem statement.

    Agent loop:
    1. Initialize with system prompt and problem statement
    2. Query model for next action
    3. Execute tool and get observation
    4. Check for exit conditions (submit, max iterations)
    5. Repeat until done
    """
    ctx = require_context()

    # Determine which openai_client to use
    gen_client_name = ctx.task_config.get('gen_client_name', 'main')
    client = ctx.get_openai_client(gen_client_name)

    # Build user prompt from task_config if available (set by gen_task or integrated task)
    working_dir = ctx.task_config.get('working_dir')
    if working_dir:
        problem_statement = ctx.task_config.get('problem_statement', '')
        gold_patch = ctx.task_config.get('gold_patch') or '(No gold patch available)'
        agent_input.user_prompt = GEN_INSTANCE_TEMPLATE.format(
            working_dir=working_dir,
            gold_patch=gold_patch,
            problem_statement=problem_statement,
        )

    # Create tool executor
    executor = ToolExecutor([
        tool_registry.get('swe_agent_rubrics/bash'),
        tool_registry.get('swe_agent_rubrics/str_replace_editor'),
        tool_registry.get('swe_agent_rubrics/review_submit'),
    ])

    # Build message history
    history: list[dict] = [
        {'role': 'system', 'content': GEN_SYSTEM_PROMPT},
        {'role': 'user', 'content': agent_input.user_prompt},
    ]

    tools = executor.get_tool_schemas()
    exit_status: ExitStatus | None = None
    iteration = 0

    try:
        while iteration < agent_input.max_iterations:
            logger.debug(f'=== SWE-agent-rubrics Step {iteration + 1} ===')

            # Call LLM
            max_retries = agent_input.completion_retry_on_tool_error
            message = None
            parsed_tool_args: list[dict] = []

            for retry_attempt in range(max_retries + 1):
                response = await client.create_completion(messages=history, tools=tools or None)
                message = response.choices[0].message

                if message.content:
                    logger.debug(f'Model output: {message.content[:200]}...')

                # No tool calls = pure text response (thinking). This is allowed.
                if not message.tool_calls:
                    logger.info('No tool calls this turn (text-only response)')
                    parsed_tool_args = []
                    break

                # Validate JSON in tool call arguments
                json_parse_error = None
                parsed_tool_args = []
                for tool_call in message.tool_calls:
                    try:
                        parsed_tool_args.append(json.loads(tool_call.function.arguments))
                    except json.JSONDecodeError as e:
                        json_parse_error = e
                        break

                if json_parse_error is not None:
                    if retry_attempt < max_retries:
                        logger.warning(
                            f'Invalid tool arguments: {json_parse_error}, retry {retry_attempt + 1}/{max_retries}'
                        )
                        continue
                    logger.error(f'Invalid tool arguments: {json_parse_error}, all retries exhausted')
                    exit_status = ExitStatus.TOOL_ERROR
                    break

                break  # Valid response with tool calls

            if exit_status is not None:
                break

            assert message is not None
            history.append(message.model_dump(exclude_none=True))

            # If no tool calls, just continue to next iteration (text was added to history)
            if not message.tool_calls:
                iteration += 1
                continue

            # Execute tool calls
            for idx, tool_call in enumerate(message.tool_calls):
                tool_name = tool_call.function.name
                tool_args = parsed_tool_args[idx]

                logger.debug(f'Tool call: {tool_name}')
                logger.debug(f'Tool args: {json.dumps(tool_args, ensure_ascii=False)[:500]}')

                try:
                    tool_response = await executor.call(tool_name, tool_args)
                except Exception as e:
                    logger.error(f'Tool execution error: {e}')
                    history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': f'Error executing tool: {e}',
                    })
                    continue

                raw_output = tool_response.output[0].get('text', '') if tool_response.output else ''
                if raw_output:
                    truncated_output = maybe_truncate(raw_output, truncate_after=DEFAULT_TEXT_CONTENT_LIMIT)
                    if len(truncated_output) < len(raw_output):
                        logger.debug(f'Observation truncated: {len(raw_output)} -> {len(truncated_output)} chars')
                    output_text = GEN_NEXT_STEP_TEMPLATE.format(observation=truncated_output)
                else:
                    output_text = GEN_NEXT_STEP_NO_OUTPUT

                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': output_text,
                })

                if ctx.tool_state.get('exit_status') == 'submitted':
                    exit_status = ExitStatus.SUBMITTED
                    break

            if exit_status is not None:
                break

            iteration += 1

    except Exception as e:
        logger.exception(f'Agent encountered error: {e}')
        exit_status = ExitStatus.ERROR

    if exit_status is None:
        exit_status = ExitStatus.MAX_ITERATIONS
        logger.warning(f'Agent reached max iterations ({agent_input.max_iterations})')

    logger.info(f'SWE-agent-rubrics finished: status={exit_status.value}, iterations={iteration + 1}')

    return SWEAgentRubricsOutput(exit_status=exit_status, iterations=iteration + 1)
