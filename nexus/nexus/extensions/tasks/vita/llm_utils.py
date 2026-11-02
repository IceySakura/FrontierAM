"""LLM adapter: bridges Nexus async OpenAI client to VitaBench message format."""

import json
import logging

from typing import Any

from nexus.extensions.tasks.vita.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from nexus.extensions.tasks.vita.environment.tool import Tool


logger = logging.getLogger(__name__)


def format_messages(messages: list[Message]) -> list[dict]:
    """Convert VitaBench Message objects to OpenAI-compatible dicts."""
    formatted = []
    for message in messages:
        if isinstance(message, UserMessage):
            formatted.append({'role': 'user', 'content': message.content})
        elif isinstance(message, AssistantMessage):
            entry: dict[str, Any] = {'role': 'assistant', 'content': message.content}
            if message.is_tool_call():
                entry['tool_calls'] = [
                    {
                        'id': tc.id,
                        'type': 'function',
                        'function': {
                            'name': tc.name,
                            'arguments': json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in message.tool_calls or []
                ]
            if message.raw_data and message.raw_data.get('message'):
                reasoning = message.raw_data['message'].get('reasoning_content')
                if reasoning:
                    entry['reasoning_content'] = reasoning
            formatted.append(entry)
        elif isinstance(message, ToolMessage):
            formatted.append({
                'role': 'tool',
                'content': message.content,
                'tool_call_id': message.id,
                'name': message.name,
            })
        elif isinstance(message, SystemMessage):
            formatted.append({'role': 'system', 'content': message.content})
    return formatted


def _parse_response_to_message(response: Any) -> AssistantMessage:
    """Parse an OpenAI ChatCompletion response into a VitaBench AssistantMessage."""
    choice = response.choices[0]
    msg = choice.message

    content = msg.content
    tool_calls = None

    if msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args or {},
                )
            )
        tool_calls = tool_calls or None

    usage_dict = None
    cost = 0.0
    if response.usage:
        usage_dict = {
            'prompt_tokens': response.usage.prompt_tokens or 0,
            'completion_tokens': response.usage.completion_tokens or 0,
        }

    raw_data = {
        'message': {
            'content': content,
            'reasoning_content': getattr(msg, 'reasoning_content', None),
        }
    }

    return AssistantMessage(
        role='assistant',
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage_dict,
        raw_data=raw_data,
    )


async def generate(
    openai_client: Any,
    messages: list[Message],
    tools: list[Tool] | None = None,
    enable_think: bool = False,
    **kwargs: Any,
) -> AssistantMessage:
    """Generate a response using a Nexus OpenAI client.

    This replaces VitaBench's synchronous HTTP-based generate() with an async
    call through Nexus's OpenAI-compatible client.
    """
    formatted_messages = format_messages(messages)
    tool_schemas = [tool.openai_schema for tool in tools] if tools else None

    create_kwargs: dict[str, Any] = {
        'messages': formatted_messages,
    }

    if tool_schemas:
        create_kwargs['tools'] = tool_schemas
        create_kwargs['tool_choice'] = kwargs.pop('tool_choice', 'auto')

    if enable_think:
        reasoning_effort = kwargs.pop('reasoning_effort', None)
        if reasoning_effort:
            create_kwargs['reasoning_effort'] = reasoning_effort
        thinking = kwargs.pop('thinking', None)
        if thinking:
            create_kwargs['thinking'] = thinking
    else:
        kwargs.pop('reasoning_effort', None)
        kwargs.pop('thinking', None)

    temperature = kwargs.pop('temperature', None)
    if temperature is not None:
        create_kwargs['temperature'] = temperature

    max_tokens = kwargs.pop('max_tokens', None)
    if max_tokens is not None:
        create_kwargs['max_tokens'] = max_tokens

    max_completion_tokens = kwargs.pop('max_completion_tokens', None)
    if max_completion_tokens is not None:
        create_kwargs['max_completion_tokens'] = max_completion_tokens

    seed = kwargs.pop('seed', None)
    if seed is not None:
        create_kwargs['seed'] = seed

    msg_count = len(formatted_messages)
    total_chars = sum(len(m.get('content', '') or '') for m in formatted_messages)
    tool_count = len(tool_schemas) if tool_schemas else 0
    logger.debug(
        'LLM call start: messages=%d, total_content_chars=%d, tools=%d',
        msg_count,
        total_chars,
        tool_count,
    )

    try:
        response = await openai_client.create_completion(**create_kwargs)
    except Exception as e:
        logger.error('LLM call failed: %s', e)
        raise

    result = _parse_response_to_message(response)
    content_len = len(result.content or '')
    tc_count = len(result.tool_calls or [])
    usage = result.usage or {}
    logger.debug(
        'LLM call done: response_chars=%d, tool_calls=%d, usage=%s',
        content_len,
        tc_count,
        usage,
    )
    return result


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """Get the cost of agent vs user messages."""
    agent_cost = 0.0
    user_cost = 0.0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if hasattr(message, 'cost') and message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            return None
    return agent_cost, user_cost
