"""One LLM handler per runtime.

The failure being guarded is silent and expensive: two handlers on one runtime
make every ``llm_request`` run twice, because dispatch walks all listeners. The
first test below checks against the real handler class that the runtime's LLM
router rejects a second default handler, so the guard is measured, not assumed.
"""

import asyncio
import uuid

from typing import Any

import pytest

from nexus.extensions.tasks.hitl.actor.labelling.handler import LabelledLLMRequestHandler
from nexus.extensions.tasks.hitl.actor.resources import (
    acquire_llm_handler,
    current_llm_handler,
    foreign_listener_count,
    holder_count,
    release_llm_handler,
)


class StubClient:
    """A labellable client seat; ``object()`` cannot carry a label."""


class Runtime:
    """Minimal runtime whose dispatch mirrors ``_dispatch_event_listeners``."""

    def __init__(self) -> None:
        self._event_listeners: dict[str, list[Any]] = {}

    def on(self, event: str, listener: Any) -> None:
        self._event_listeners.setdefault(event, []).append(listener)

    def off(self, event: str, listener: Any = None) -> None:
        if listener is None:
            self._event_listeners.pop(event, None)
            return
        listeners = self._event_listeners.get(event)
        if listeners and listener in listeners:
            listeners.remove(listener)

    async def emit(self, event: str, payload: Any = None) -> None:
        return None

    async def dispatch(self, event: str, data: dict[str, Any]) -> None:
        for listener in list(self._event_listeners.get(event, ())):
            await listener(data)


@pytest.fixture
def runtime():
    rt = Runtime()
    yield rt
    release_llm_handler(rt)


@pytest.mark.asyncio
async def test_second_default_handler_on_one_runtime_is_rejected(monkeypatch):
    """The runtime's LLM router refuses a second default handler instead of double-dispatching."""
    served: list[str] = []

    async def base_handle(data):
        served.append(data['request_id'])

    monkeypatch.setattr(LabelledLLMRequestHandler.__mro__[1], 'handle_llm_request', staticmethod(base_handle))

    rt = Runtime()
    first = LabelledLLMRequestHandler(runtime=rt, openai_client=StubClient())
    second = LabelledLLMRequestHandler(runtime=rt, openai_client=StubClient())
    first.register()
    with pytest.raises(ValueError, match='already bound'):
        second.register()

    await rt.dispatch('llm_request', {'request_id': 'only-one-request'})

    assert served == ['only-one-request']


def test_acquire_returns_one_handler_per_runtime(runtime):
    first = acquire_llm_handler(runtime, openai_client=StubClient())
    second = acquire_llm_handler(runtime, openai_client=StubClient())

    assert first is second
    assert isinstance(first, LabelledLLMRequestHandler)
    # One object means one listener means one model call.
    assert foreign_listener_count(runtime) == 1


def test_separate_runtimes_get_separate_handlers():
    left, right = Runtime(), Runtime()
    try:
        assert acquire_llm_handler(left, openai_client=StubClient()) is not acquire_llm_handler(
            right, openai_client=StubClient()
        )
    finally:
        release_llm_handler(left)
        release_llm_handler(right)


def test_a_foreign_handler_makes_acquire_raise(runtime):
    # Something registered its own handler first, bypassing acquire. Both would
    # serve everything, so the second one has to be refused rather than added.
    foreign = LabelledLLMRequestHandler(runtime=runtime, openai_client=StubClient())
    foreign.register()

    with pytest.raises(RuntimeError, match='already registered'):
        acquire_llm_handler(runtime, openai_client=StubClient())


def test_release_unregisters_and_is_idempotent(runtime):
    acquire_llm_handler(runtime, openai_client=StubClient())
    assert current_llm_handler(runtime) is not None

    release_llm_handler(runtime)
    assert current_llm_handler(runtime) is None
    assert foreign_listener_count(runtime) == 0

    # Two managers may both call shutdown(); the second must not raise.
    release_llm_handler(runtime)


def test_release_allows_a_fresh_acquire(runtime):
    first = acquire_llm_handler(runtime, openai_client=StubClient())
    release_llm_handler(runtime)
    second = acquire_llm_handler(runtime, openai_client=StubClient())

    assert second is not first
    assert foreign_listener_count(runtime) == 1


@pytest.mark.asyncio
async def test_the_shared_handler_serves_each_request_exactly_once(runtime, monkeypatch):
    served: list[str] = []

    async def base_handle(data):
        served.append(data['request_id'])

    monkeypatch.setattr(LabelledLLMRequestHandler.__mro__[1], 'handle_llm_request', staticmethod(base_handle))

    acquire_llm_handler(runtime, openai_client=StubClient())
    acquire_llm_handler(runtime, openai_client=StubClient())  # a second manager

    ids = [str(uuid.uuid4()) for _ in range(3)]
    await asyncio.gather(*(runtime.dispatch('llm_request', {'request_id': rid}) for rid in ids))

    assert sorted(served) == sorted(ids)


def test_listener_count_is_best_effort_on_an_opaque_runtime():
    class Opaque:
        pass

    # No way to introspect: report 0 rather than guessing or raising.
    assert foreign_listener_count(Opaque()) == 0


# -- releasing a handler two managers share --------------------------------


def test_the_handler_survives_until_the_last_holder_releases(runtime):
    """Release is reference-counted, because acquire is shared.

    Both callers get the same object and each sets its own ``_owns_handler``, so
    either may release first. Unregistering then would take the listener away from
    the other, whose next request would reach nobody -- a stalled turn rather than
    an error.
    """
    first = acquire_llm_handler(runtime, openai_client=StubClient())
    second = acquire_llm_handler(runtime, openai_client=StubClient())
    assert first is second
    assert holder_count(runtime) == 2

    release_llm_handler(runtime)

    assert current_llm_handler(runtime) is first, 'the second holder is still serving through it'
    assert foreign_listener_count(runtime) == 1
    assert holder_count(runtime) == 1

    release_llm_handler(runtime)

    assert current_llm_handler(runtime) is None
    assert foreign_listener_count(runtime) == 0


def test_releasing_more_often_than_acquired_is_a_no_op(runtime):
    """Shutdown paths must not raise, and must not go negative into a later acquire."""
    acquire_llm_handler(runtime, openai_client=StubClient())
    release_llm_handler(runtime)
    release_llm_handler(runtime)
    release_llm_handler(runtime)

    assert holder_count(runtime) == 0
    assert current_llm_handler(runtime) is None

    # A fresh acquire after over-releasing starts from one holder, not from a debt.
    acquire_llm_handler(runtime, openai_client=StubClient())
    assert holder_count(runtime) == 1
    assert foreign_listener_count(runtime) == 1


@pytest.mark.asyncio
async def test_a_request_after_one_of_two_releases_still_reaches_the_handler(runtime, monkeypatch):
    """The consequence, as behaviour rather than as listener bookkeeping."""
    served: list[str] = []

    async def base_handle(data):
        served.append(data['request_id'])

    monkeypatch.setattr(LabelledLLMRequestHandler.__mro__[1], 'handle_llm_request', staticmethod(base_handle))

    acquire_llm_handler(runtime, openai_client=StubClient())
    acquire_llm_handler(runtime, openai_client=StubClient())

    await runtime.dispatch('llm_request', {'request_id': 'while-both-hold-it'})
    release_llm_handler(runtime)
    await runtime.dispatch('llm_request', {'request_id': 'after-one-released'})
    release_llm_handler(runtime)
    await runtime.dispatch('llm_request', {'request_id': 'after-both-released'})

    assert served == ['while-both-hold-it', 'after-one-released']


@pytest.mark.asyncio
async def test_two_holders_still_serve_each_request_once(runtime, monkeypatch):
    """Sharing must not become double-serving: one listener, whoever asked for it."""
    served: list[str] = []

    async def base_handle(data):
        served.append(data['request_id'])

    monkeypatch.setattr(LabelledLLMRequestHandler.__mro__[1], 'handle_llm_request', staticmethod(base_handle))

    acquire_llm_handler(runtime, openai_client=StubClient())
    acquire_llm_handler(runtime, openai_client=StubClient())

    await runtime.dispatch('llm_request', {'request_id': 'once'})

    assert served == ['once']
