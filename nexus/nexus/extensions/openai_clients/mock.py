"""Mock OpenAI client for testing and demonstration.

This client allows you to define custom responses or exceptions for testing
cache functionality, error handling, and other scenarios without making
real API calls.

Example usage:

    # In configuration - responses can be strings, dicts with tool_calls, or exception configs
    openai_clients:
      main:
        name: "mock"
        config:
          client_args:
            responses:
              - "First response"  # Simple text response
              - content: "Let me use a tool"  # Response with tool_calls
                tool_calls:
                  - id: "call_1"
                    function:
                      name: "demo/think"
                      arguments: '{"thought": "thinking..."}'
              - exception: "TimeoutError"
                message: "Simulated timeout"
              - "Fourth response (after recovery)"

    # Or programmatically
    from nexus.extensions.openai_clients.mock import MockException

    run_input = RunInput(
        openai_clients={
            'main': {
                'name': 'mock',
                'config': {
                    'client_args': {
                        'responses': [
                            'Hello!',
                            MockException('TimeoutError', 'API timeout'),
                            'Recovered!',
                        ],
                    },
                },
            },
        },
    )
"""

# /// script
# dependencies = []
#
# [project]
# name = "mock"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# openai_clients = { mock = "nexus.extensions.openai_clients.mock" }
# ///

from __future__ import annotations

import time

from typing import Any, NoReturn

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from pydantic import BaseModel

from nexus.core import OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.types import ExtendedChatCompletion
from nexus.core.types.completion import ExtendedChoice
from nexus.core.types.message import ExtendedMessage
from nexus.core.types.usage import ExtendedUsage


class MockResponse(BaseModel):
    """A structured mock response with optional tool_calls.

    Example:
        MockResponse(content='Hello!')
        MockResponse(content='Using tool', tool_calls=[...])
    """

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class MockAPIError(Exception):
    """Mock API error for testing error handling."""

    pass


class MockException(BaseModel):
    """Configuration for a mock exception in the response sequence.

    Example:
        responses = [
            'First response',
            MockException(type='TimeoutError', message='API timeout'),
            'Third response',
        ]
    """

    type: str = 'MockAPIError'
    message: str = 'Simulated error'

    def raise_exception(self) -> NoReturn:
        """Raise the configured exception."""
        exception_map: dict[str, type[Exception]] = {
            'MockAPIError': MockAPIError,
            'RuntimeError': RuntimeError,
            'TimeoutError': TimeoutError,
            'ConnectionError': ConnectionError,
            'ValueError': ValueError,
            'Exception': Exception,
        }
        exc_class = exception_map.get(self.type, Exception)
        raise exc_class(self.message)


def _parse_response_item(item: str | dict | MockException | MockResponse) -> str | MockException | MockResponse:
    """Parse a response item from config.

    Args:
        item: Can be a string, a dict with 'exception'/'content'/'tool_calls' key,
              or a MockException/MockResponse

    Returns:
        str (simple response), MockException (raise error), or MockResponse (with tool_calls)
    """
    if isinstance(item, str):
        return item
    if isinstance(item, MockException):
        return item
    if isinstance(item, MockResponse):
        return item
    if isinstance(item, dict):
        if 'exception' in item:
            return MockException(
                type=item['exception'],
                message=item.get('message', 'Simulated error'),
            )
        if 'content' in item or 'tool_calls' in item:
            return MockResponse(
                content=item.get('content'),
                tool_calls=item.get('tool_calls'),
            )
        # Treat as regular dict, convert to string
        return str(item)
    return str(item)


@openai_client_registry.register('mock')
class MockOpenAIClient(OpenAICompatibleClient):
    """Mock OpenAI client for testing.

    Supports:
    - Predefined response sequences (strings or exceptions)
    - Simulated failures at any position in the sequence
    - Custom error messages and exception types
    - Call counting for verification

    Config options (in client_args):
        responses: list[str | dict] - List of responses or exception configs
            - String: Normal response content
            - Dict with 'exception' key: Raise exception
              Example: {'exception': 'TimeoutError', 'message': 'API timeout'}

    Example:
        responses:
          - "First response"
          - exception: "TimeoutError"
            message: "Simulated timeout"
          - "Third response (after recovery)"
    """

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        raw_responses = config.client_args.get('responses', ['Mock response'])
        self.responses: list[str | MockException | MockResponse] = [_parse_response_item(r) for r in raw_responses]
        self.call_count: int = 0

    def _get_response(self) -> str | MockException | MockResponse:
        """Get the response for current call (cycles through responses)."""
        idx = (self.call_count - 1) % len(self.responses)
        return self.responses[idx]

    async def _create_completion(
        self,
        messages: list,
        model: str | Any = None,
        **_kwargs: Any,
    ) -> ExtendedChatCompletion:
        """Create a mock completion response.

        Args:
            messages: The messages to complete
            model: The model to use (ignored in mock)
            **_kwargs: Additional arguments (ignored in mock)

        Returns:
            Mock ExtendedChatCompletion

        Raises:
            Configured exception if the current response is a MockException
        """
        self.call_count += 1

        # Get response (may be string, exception, or structured response)
        response = self._get_response()

        # If it's an exception config, raise it
        if isinstance(response, MockException):
            response.raise_exception()

        # Handle different response types
        content: str | None = None
        tool_calls: list[ChatCompletionMessageToolCall] | None = None

        if isinstance(response, str):
            content = response
        elif isinstance(response, MockResponse):
            content = response.content
            if response.tool_calls:
                tool_calls = [
                    ChatCompletionMessageToolCall(
                        id=tc.get('id', f'call_{i}'),
                        type='function',
                        function=Function(
                            name=tc['function']['name'],
                            arguments=tc['function']['arguments'],
                        ),
                    )
                    for i, tc in enumerate(response.tool_calls)
                ]

        # Determine finish_reason based on response type
        finish_reason = 'tool_calls' if tool_calls else 'stop'

        return ExtendedChatCompletion(
            id=f'mock-{self.call_count}-{int(time.time())}',
            choices=[
                ExtendedChoice(
                    finish_reason=finish_reason,
                    index=0,
                    message=ExtendedMessage(
                        content=content,
                        role='assistant',
                        tool_calls=tool_calls,
                    ),
                )
            ],
            created=int(time.time()),
            model=model or 'mock-model',
            usage=ExtendedUsage(
                completion_tokens=len((content or '').split()),
                prompt_tokens=sum(len(str(m.get('content', '')).split()) for m in messages),
                total_tokens=0,
            ),
        )

    def reset(self) -> None:
        """Reset call counter for reuse."""
        self.call_count = 0
