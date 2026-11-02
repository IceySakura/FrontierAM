"""Reading a model response, running its tool calls, then their handlers.

``ToolExecutor`` validates and executes; this adds a per-tool handler that may
shape the result, end the run, or append messages, and answers every call even
when something raises.
"""

import json
import logging

from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from nexus.core import ToolExecutor, ToolResponse
from nexus.core.types.message import ChatCompletionMessageToolCall
from nexus.extensions.agents.main.utils import extract_tool_output_content, extract_tool_output_text


logger = logging.getLogger(__name__)


# What a tool message carries: text, or the content parts a multimodal result needs.
ToolContent = str | list[dict[str, Any]]


def read_response(response: Any) -> tuple[dict[str, Any], list[ChatCompletionMessageToolCall]]:
    """The assistant message as a plain history entry, and its tool calls."""
    message = response.choices[0].message
    return message.model_dump(exclude_none=True), list(message.tool_calls or [])


def call_id(call: ChatCompletionMessageToolCall) -> str:
    return call.id


def call_name(call: ChatCompletionMessageToolCall) -> str:
    return call.function.name


def call_args(call: ChatCompletionMessageToolCall) -> dict[str, Any]:
    """The call's parsed arguments. Raises ``json.JSONDecodeError`` if malformed."""
    raw = call.function.arguments
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def tool_message(call: ChatCompletionMessageToolCall, content: ToolContent) -> dict[str, Any]:
    """The ``role: tool`` entry answering ``call``. Every call needs one."""
    return {'role': 'tool', 'tool_call_id': call_id(call), 'content': content}


@dataclass(frozen=True, kw_only=True)
class ExtendedToolResponse:
    """A ``ToolResponse`` plus what the loop acts on.

    ``response`` is the executor's own result, unchanged. ``kw_only`` leaves
    ``__match_args__`` empty, so positional matching raises instead of binding
    ``tool_call_id``, and the fields stay reorderable.
    """

    tool_call_id: str
    name: str
    content: ToolContent
    response: ToolResponse | None = None
    messages: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        # The loop does history.extend(result.messages), so an empty entry would
        # reach the provider as a message with no content. Builders that may have
        # nothing to say return {} (``reminder_message``), and dropping those
        # here keeps every call site from having to check.
        if any(not entry for entry in self.messages):
            object.__setattr__(self, 'messages', tuple(entry for entry in self.messages if entry))

    @property
    def failed(self) -> bool:
        return self.response is None


Handler = Callable[[ChatCompletionMessageToolCall, ToolResponse, dict[str, Any]], Awaitable['ExtendedToolResponse']]
"""``(call, response, args) -> ExtendedToolResponse``. Runs after the tool body."""


@dataclass(frozen=True)
class Dispatcher:
    """Executes calls against ``executor``, then hands them to their handler."""

    executor: ToolExecutor
    handlers: Mapping[str, Handler] = field(default_factory=dict)

    async def dispatch(self, call: ChatCompletionMessageToolCall) -> ExtendedToolResponse:
        """Run one call. Bad arguments and raised errors come back as content."""
        name = call_name(call)
        try:
            args = call_args(call)
        except Exception as exc:  # noqa: BLE001 - reported to the model as content
            return self._failure(call, name, f'{type(exc).__name__}: {exc}')

        try:
            self.executor.validate_tool_call(name, args)
            response = await self.executor.call(name, args)
        except Exception as exc:  # noqa: BLE001 - reported to the model as content
            logger.warning('tool %s failed: %s', name, exc)
            return self._failure(call, name, f'{type(exc).__name__}: {exc}')

        content = content_of(response)
        handler = self.handlers.get(name)
        if handler is None:
            return ExtendedToolResponse(tool_call_id=call_id(call), name=name, content=content, response=response)

        try:
            return await handler(call, response, args)
        except Exception as exc:  # noqa: BLE001 - appended to the tool's own result
            logger.exception('handler for %s failed', name)
            return ExtendedToolResponse(
                tool_call_id=call_id(call),
                name=name,
                content=with_error(content, f'{type(exc).__name__}: {exc}'),
                response=response,
            )

    def _failure(self, call: ChatCompletionMessageToolCall, name: str, detail: str) -> ExtendedToolResponse:
        return ExtendedToolResponse(tool_call_id=call_id(call), name=name, content=f'<error>\n{detail}\n</error>')


def text_of(response: ToolResponse) -> str:
    """A ``ToolResponse`` as text, with a placeholder for any non-text part.

    For a handler that reads what a tool said. The history gets
    :func:`content_of` instead, which keeps the parts themselves.
    """
    return extract_tool_output_text(response.output)


def content_of(response: ToolResponse) -> ToolContent:
    """A ``ToolResponse`` as the content a tool message carries.

    Text-only results stay a string; anything else keeps its content parts, so an
    image reaches the model as an image. ``str()`` on the part would put its whole
    base64 payload in the text -- measured at 240 KB for one 1280x720 screenshot,
    and the model still could not see it.
    """
    return extract_tool_output_content(response.output)


def with_error(content: ToolContent, detail: str) -> ToolContent:
    """``content`` with an error appended, whichever form it is in."""
    note = f'<error>\n{detail}\n</error>'
    if isinstance(content, str):
        return f'{content}\n\n{note}' if content else note
    return [*content, {'type': 'text', 'text': note}]


def check_handlers(handlers: Mapping[str, Handler], names: Collection[str]) -> None:
    """Raise when a handler is keyed by a tool the set does not hold."""
    unknown = sorted(set(handlers) - set(names))
    if unknown:
        available = ', '.join(sorted(names)) or '(none)'
        raise ValueError(f'handlers for tools that are not offered: {unknown}. Offered: {available}')


__all__ = [
    'Dispatcher',
    'ExtendedToolResponse',
    'Handler',
    'ToolContent',
    'call_args',
    'call_id',
    'call_name',
    'check_handlers',
    'content_of',
    'read_response',
    'text_of',
    'tool_message',
    'with_error',
]
