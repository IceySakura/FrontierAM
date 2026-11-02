"""Host-side LLM channel routing: one runtime listener fans requests out to per-channel handlers."""

from __future__ import annotations

import asyncio
import contextvars
import gc
import weakref

import pytest

from nexus.runtime.server.services.llm_handler import LLMRequestHandler, LLMRequestRouter


class FakeRuntime:
    """Host-side runtime: records emits and the single registered listener."""

    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []
        self.registered: dict[str, object] = {}

    async def emit(self, event, data):
        self.emitted.append((event, data))

    def on(self, event, handler):
        self.registered[event] = handler

    def off(self, event, handler):  # noqa: ARG002
        self.registered.pop(event, None)


class FakeClient:
    """Records model calls so a test can count them per channel."""

    class Config:
        request_args: dict = {}
        client_args: dict = {}

    config = Config()

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict] = []

    async def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {'choices': [{'message': {'content': self.name}}]}


@pytest.mark.asyncio
async def test_router_gives_each_channel_its_own_handler_once():
    runtime = FakeRuntime()
    client_a, client_b = FakeClient('a'), FakeClient('b')
    handler_a = LLMRequestHandler(runtime=runtime, openai_client=client_a)
    handler_b = LLMRequestHandler(runtime=runtime, openai_client=client_b)
    router = LLMRequestRouter.for_runtime(runtime)
    router.bind('A', handler_a)
    router.bind('B', handler_b)

    await asyncio.gather(
        router.dispatch({
            'request_id': 'R1',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'rsio': {'attempt': 1, 'channel_id': 'A'},
        }),
        router.dispatch({
            'request_id': 'R2',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'rsio': {'attempt': 1, 'channel_id': 'B'},
        }),
    )

    assert len(client_a.calls) == 1
    assert len(client_b.calls) == 1
    assert len(runtime.registered) == 1


@pytest.mark.asyncio
async def test_channel_handler_preserves_its_creation_context():
    marker: contextvars.ContextVar[str] = contextvars.ContextVar('llm_channel_test_marker')
    runtime = FakeRuntime()
    seen: list[tuple[str, str]] = []

    class ContextClient(FakeClient):
        async def create_completion(self, **kwargs):
            seen.append((self.name, marker.get()))
            return await super().create_completion(**kwargs)

    router = LLMRequestRouter.for_runtime(runtime)
    token_a = marker.set('context-a')
    try:
        router.bind('A', LLMRequestHandler(runtime=runtime, openai_client=ContextClient('a')))
    finally:
        marker.reset(token_a)
    token_b = marker.set('context-b')
    try:
        router.bind('B', LLMRequestHandler(runtime=runtime, openai_client=ContextClient('b')))
    finally:
        marker.reset(token_b)

    await asyncio.gather(
        router.dispatch({
            'request_id': 'R1',
            'messages': [{'role': 'user', 'content': 'a'}],
            'rsio': {'channel_id': 'A'},
        }),
        router.dispatch({
            'request_id': 'R2',
            'messages': [{'role': 'user', 'content': 'b'}],
            'rsio': {'channel_id': 'B'},
        }),
    )

    assert set(seen) == {('a', 'context-a'), ('b', 'context-b')}


@pytest.mark.asyncio
async def test_persistent_handler_refreshes_context_between_turns():
    marker: contextvars.ContextVar[str] = contextvars.ContextVar('persistent_llm_channel_marker')
    runtime = FakeRuntime()
    seen: list[str] = []

    class ContextClient(FakeClient):
        async def create_completion(self, **kwargs):
            seen.append(marker.get())
            return await super().create_completion(**kwargs)

    token = marker.set('turn-1')
    try:
        handler = LLMRequestHandler(runtime=runtime, openai_client=ContextClient('client'))
    finally:
        marker.reset(token)
    router = LLMRequestRouter.for_runtime(runtime)
    router.bind('A', handler)
    await router.dispatch({
        'request_id': 'R1',
        'messages': [{'role': 'user', 'content': 'a'}],
        'rsio': {'channel_id': 'A'},
    })

    token = marker.set('turn-2')
    try:
        handler.refresh_request_context()
    finally:
        marker.reset(token)
    await router.dispatch({
        'request_id': 'R2',
        'messages': [{'role': 'user', 'content': 'b'}],
        'rsio': {'channel_id': 'A'},
    })

    assert seen == ['turn-1', 'turn-2']


def test_router_rejects_duplicate_channel_binding():
    runtime = FakeRuntime()
    router = LLMRequestRouter.for_runtime(runtime)
    handler_a = LLMRequestHandler(runtime=runtime, openai_client=FakeClient('a'))
    handler_b = LLMRequestHandler(runtime=runtime, openai_client=FakeClient('b'))
    router.bind('A', handler_a)

    with pytest.raises(ValueError, match='already bound'):
        router.bind('A', handler_b)


def test_router_removes_listener_after_last_route_is_unbound():
    runtime = FakeRuntime()
    router = LLMRequestRouter.for_runtime(runtime)
    handler = LLMRequestHandler(runtime=runtime, openai_client=FakeClient('a'))
    router.bind('A', handler)

    router.unbind('A', handler)

    assert runtime.registered == {}


@pytest.mark.parametrize('route', ['channel', 'default'])
def test_a_route_never_unbound_does_not_keep_its_runtime_alive(route):
    runtime = FakeRuntime()
    handler = LLMRequestHandler(runtime=runtime, openai_client=FakeClient('a'))
    if route == 'channel':
        LLMRequestRouter.for_runtime(runtime).bind('A', handler)
    else:
        handler.register()
    runtime_ref = weakref.ref(runtime)
    del runtime, handler
    gc.collect()

    assert runtime_ref() is None


def test_default_handler_unregisters_even_when_the_runtime_keeps_no_listener_reference():
    class ForgetfulRuntime:
        """A runtime may track listeners without holding the router alive, e.g. a proxy client."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        def on(self, event, handler):  # noqa: ARG002
            self.calls.append(f'on:{event}')

        def off(self, event, handler):  # noqa: ARG002
            self.calls.append(f'off:{event}')

    runtime = ForgetfulRuntime()
    handler = LLMRequestHandler(runtime=runtime, openai_client=FakeClient('default'))
    handler.register()
    gc.collect()
    handler.unregister()
    assert runtime.calls == ['on:llm_request', 'off:llm_request'], 'Unregister must remove the listener it added'


@pytest.mark.asyncio
async def test_router_sends_unchanneled_requests_to_the_default_handler():
    runtime = FakeRuntime()
    client = FakeClient('default')
    handler = LLMRequestHandler(runtime=runtime, openai_client=client)
    handler.register()

    await LLMRequestRouter.for_runtime(runtime).dispatch({
        'request_id': 'R1',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'rsio': {'attempt': 1},
    })

    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_router_rejects_a_request_whose_channel_is_unbound(caplog):
    runtime = FakeRuntime()
    client = FakeClient('default')
    LLMRequestHandler(runtime=runtime, openai_client=client).register()

    await LLMRequestRouter.for_runtime(runtime).dispatch({
        'request_id': 'R1',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'rsio': {'attempt': 1, 'channel_id': 'gone'},
    })

    # No silent fallback: answering with another session's client would mix
    # trajectories. Fail this channel immediately instead.
    assert client.calls == []
    assert 'no handler bound' in caplog.text
    assert runtime.emitted == [
        (
            'llm_response',
            {
                'request_id': 'R1',
                'success': False,
                'error': "No LLM handler is bound for channel 'gone'",
                'error_code': 'UNBOUND_LLM_CHANNEL',
                'rsio': {'kind': 'final'},
            },
        )
    ]
