"""Single-turn agent for one-shot LLM completion.

This agent performs a single LLM call without tool use, suitable for:
- Math reasoning (GSM8K, MATH, etc.)
- Text generation tasks
- Classification tasks
- Any task requiring one-shot completion
"""

import logging

from typing import Any

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    WithMainTrajectory,
    agent_registry,
    require_context,
)


logger = logging.getLogger(__name__)


class SingleTurnAgentInput(AgentRunInput):
    """Single-turn agent input.

    Accepts a list of messages in OpenAI chat format, allowing full control
    over the conversation structure (system prompt, user messages, etc.).
    """

    prompt: list[dict[str, Any]]  # Messages in OpenAI chat format.
    system_prompt: str | None = None  # Prepended without replacing system messages already in prompt.
    tools: list[dict] | None = None  # Optional tool schemas, forwarded to the chat template.
    # Needed when ``prompt`` replays a tool-using history: the Hunyuan chat template
    # only keeps interleaved <think> blocks for historical turns when ``tools`` is
    # provided (it force-disables interleaved_thinking when tools is None), so the
    # rendered prompt matches the original multi-turn rollout byte-for-byte.


class SingleTurnAgentOutput(AgentRunOutput, WithMainTrajectory):
    """Single-turn agent output."""

    response: str  # The assistant's response content
    raw_response: str | None = None
    reasoning_content: str | None = None


@agent_registry.register('single_turn', meta={'gateway': {'scaffold_policy': 'nexus_main_agent'}})
async def single_turn_agent(agent_input: SingleTurnAgentInput) -> SingleTurnAgentOutput:
    """Single-turn agent that performs one LLM completion.

    This agent:
    1. Prepends the optional system prompt to the input messages
    2. Calls the LLM once
    3. Returns the response

    No tool calling, no multi-turn conversation.
    """
    ctx = require_context()
    client = ctx.openai_client

    # Keep the original prompt untouched; the complete assistant message is the
    # canonical one-turn trajectory entry returned by this agent.
    messages = [dict(message) for message in agent_input.prompt]
    if agent_input.system_prompt is not None:
        messages.insert(0, {'role': 'system', 'content': agent_input.system_prompt})

    logger.debug(f'Single-turn agent: calling LLM with {len(messages)} messages')

    response = await client.create_completion(messages=messages, tools=agent_input.tools)
    choice = response.choices[0]
    message = choice.message
    assistant_content = message.content or ''

    logger.debug(f'Single-turn agent: received response of {len(assistant_content)} chars')

    # Keep all fields Gateway exposed on the assistant message (tool calls,
    # raw_response and reasoning_content included) so standalone and Gateway
    # trajectories have the same message dump.
    if hasattr(message, 'model_dump'):
        assistant_message = message.model_dump(exclude_none=True)
    else:
        assistant_message = {'role': 'assistant', 'content': assistant_content}
    assistant_message.setdefault('role', 'assistant')
    assistant_message.setdefault('content', assistant_content)
    messages.append(assistant_message)
    raw_response = getattr(message, 'raw_response', None)
    reasoning_content = getattr(message, 'reasoning_content', None)
    return SingleTurnAgentOutput(
        main_trajectory=messages,
        response=assistant_content,
        raw_response=raw_response,
        reasoning_content=reasoning_content,
    )
