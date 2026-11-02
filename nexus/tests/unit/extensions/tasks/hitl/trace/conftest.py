"""Shared fakes for concurrent trace-handler tests.

Three details here decide whether the isolation tests in this directory mean
anything. Each one, if dropped, turns a real failure into a passing test:

1. **Spans come from the real decorator.** ``MockLLMBackend`` subclasses
   ``OpenAICompatibleClient`` and implements ``_create_completion``, so the
   ``@observe('openai_completion')`` on the public ``create_completion``
   produces genuine spans through the genuine dispatcher. Hand-building spans
   with ``start_span``/``update_span`` bypasses that decorator's context
   management, which is the very thing under test.

2. **Dispatch runs in a fresh context.** ``FakeRuntime.deliver`` starts each
   listener with ``context=contextvars.Context()``. Real Socket.IO callbacks
   arrive on their own task and do *not* inherit the contextvars of the
   ``send()`` that is waiting — confirmed against a real Socket.IO server.
   Plain ``create_task`` copies the caller's context, which would let the
   handler see processors it could never see in production, and every isolation
   assertion here would pass for the wrong reason.

3. **Every request carries a unique ``request_id``.** ``LLMRequestHandler``
   defaults a missing id to ``'<no-request-id>'`` and then de-duplicates by it,
   so concurrent requests without ids collapse into a single call and the rest
   replay its result. A test written that way passes while exercising one
   request instead of three.
"""

import asyncio
import contextvars
import uuid

from typing import Any

import pytest

from openai.types.chat import ChatCompletion

from nexus.core.openai_client import OpenAICompatibleClient, OpenAIConfig
from nexus.trace.tracer import end_span, start_span, update_span


class MockLLMBackend(OpenAICompatibleClient):
    """Deterministic LLM whose reply echoes the last prompt.

    Latency is configurable so tests can interleave turns precisely: a slow
    member models the long tail that must not block a fast one.
    """

    def __init__(self, *, latency: float = 0.0, tag: str = 'mock') -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self.latency = latency
        self.tag = tag
        self.prompts: list[str] = []

    async def _create_completion(
        self,
        messages: list[Any],
        model: Any = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        prompt = _last_text(messages)
        self.prompts.append(prompt)
        if self.latency:
            await asyncio.sleep(self.latency)
        return ChatCompletion.model_validate(
            {
                'id': f'chatcmpl-{uuid.uuid4().hex[:8]}',
                'object': 'chat.completion',
                'created': 0,
                'model': model or 'mock-model',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {'role': 'assistant', 'content': f'{self.tag}:reply<{prompt}>'},
                    }
                ],
                'usage': {'prompt_tokens': 3, 'completion_tokens': 5, 'total_tokens': 8},
            }
        )


def _last_text(messages: list[Any]) -> str:
    if not messages:
        return ''
    content = messages[-1].get('content') if isinstance(messages[-1], dict) else None
    return content if isinstance(content, str) else ''


class FakeRuntime:
    """Runtime stand-in that dispatches like Socket.IO does.

    ``deliver`` mirrors ``_dispatch_event_listeners``: it walks *every*
    registered listener, which is what makes a second handler on one runtime
    double-serve each request. Listeners run in a fresh context (see the module
    docstring) so nothing leaks in from the caller.
    """

    default_venv_id = None

    def __init__(self) -> None:
        self._listeners: dict[str, list[Any]] = {}
        self.emitted: list[tuple[str, Any]] = []
        self.connects = 0

    async def ws_connect(self) -> None:
        self.connects += 1

    def on(self, event: str, listener: Any) -> None:
        self._listeners.setdefault(event, []).append(listener)

    def off(self, event: str, listener: Any = None) -> None:
        if listener is None:
            self._listeners.pop(event, None)
            return
        listeners = self._listeners.get(event)
        if listeners and listener in listeners:
            listeners.remove(listener)

    def listener_count(self, event: str) -> int:
        return len(self._listeners.get(event, ()))

    async def emit(self, event: str, payload: Any = None) -> None:
        self.emitted.append((event, payload))

    def deliver(self, prompt: str, *, event: str = 'llm_request') -> asyncio.Task[None]:
        """Deliver one request, returning the task so tests can await it."""
        data = {
            'messages': [{'role': 'user', 'content': prompt}],
            # Unique per request: a shared id would be de-duplicated away.
            'request_id': str(uuid.uuid4()),
        }
        # ``Context().run`` rather than ``create_task(context=...)``: the keyword
        # only exists from 3.11 and this project supports 3.10. Both give the
        # listener a fresh context; only this form works on both.
        loop = asyncio.get_running_loop()
        return contextvars.Context().run(loop.create_task, self._run(event, data))

    async def _run(self, event: str, data: dict[str, Any]) -> None:
        for listener in list(self._listeners.get(event, ())):
            await listener(data)


def emit_host_span(prompt: str, reply: str) -> None:
    """Emit one host-side ``openai_completion`` span.

    Stands in for a user-agent LLM call made directly on the host (the path
    ``SimpleSessionManager`` and the recipes' own turns take), which never
    crosses the runtime boundary.
    """
    stub = start_span('openai_completion')
    span_id = stub.span_id if hasattr(stub, 'span_id') else stub
    update_span(
        span_id,
        attributes={
            'inputs': {'messages': [{'role': 'user', 'content': prompt}], 'kwargs': {}},
            'outputs': {'choices': [{'message': {'role': 'assistant', 'content': reply}}]},
        },
    )
    end_span(span_id)
