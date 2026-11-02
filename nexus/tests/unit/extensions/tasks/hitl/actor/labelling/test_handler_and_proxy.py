"""Tests for the labelled handler and proxy, from injection to exported label.

These two pieces are the ends of one path: the proxy stamps a session label onto
the request in the sandbox, the handler strips it back off host-side and holds it
on the client for the duration of the completion call so the span carries it.
Testing them together is what shows the path is unbroken -- each alone can pass
while the join is wrong.

The handler tests drive the real ``_produce_response`` seam rather than a
reimplementation of it, because the point of choosing that seam is what it
inherits from the base class.
"""

import asyncio

import pytest

from openai.types.chat import ChatCompletion

from nexus.core.openai_client import OpenAICompatibleClient, OpenAIConfig
from nexus.extensions.tasks.hitl.actor.labelling.handler import LabelledLLMRequestHandler
from nexus.extensions.tasks.hitl.actor.labelling.proxy import _LabellingBridge, attach_label
from nexus.extensions.tasks.hitl.trace import labels
from nexus.extensions.tasks.hitl.trace.labels import (
    LABEL_FIELD,
    UNATTRIBUTED,
    worker_label,
)
from nexus.trace import processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor


SYS = {'role': 'system', 'content': 'You are Claude Code.'}

# Long enough that the slow call is still open when the fast one finishes: a
# label that only survives accidental serialisation proves nothing.
SLOW = 0.10
FAST = 0.01
DISPATCH_SETTLE = 0.6


class RecordingClient(labels.LabelReadingClient, OpenAICompatibleClient):
    """A real client whose backend is replaced, not a stand-in for one.

    Subclassing the real class matters twice over: the ``@observe`` decorator on
    the public ``create_completion`` is what puts ``_chat_phase`` on the span, and
    the handler reads ``config.client_args`` / ``config.request_args`` on the way
    through. A hand-rolled double satisfies neither and passes for the wrong
    reasons.

    ``LabelReadingClient`` comes first so its context-scoped ``_chat_phase``
    property wins -- the production arrangement, since a client-stored label
    cannot survive concurrent blocks.
    """

    def __init__(self, *, delay: float = 0.0) -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self.delay = delay
        self.calls: list[dict] = []

    async def _create_completion(self, messages, model=None, **kwargs):
        self.calls.append({'messages': messages, 'model': model, **kwargs})
        if self.delay:
            await asyncio.sleep(self.delay)
        reply = kwargs.get('_reply', 'ok')
        return ChatCompletion.model_validate(
            {
                'id': 'chatcmpl-test',
                'object': 'chat.completion',
                'created': 0,
                'model': model or 'mock-model',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': reply}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            }
        )


class StubRuntime:
    """The handler only touches the runtime to register/emit; neither is used here."""

    default_venv_id = None

    def on(self, event, listener):  # pragma: no cover - registration unused
        pass

    def off(self, event, listener=None):  # pragma: no cover
        pass

    async def emit(self, event, payload=None):  # pragma: no cover
        pass


def make_handler(client: RecordingClient) -> LabelledLLMRequestHandler:
    return LabelledLLMRequestHandler(runtime=StubRuntime(), openai_client=client)


def request(label: str | None, reply: str, prompt: str) -> dict:
    """A request as it arrives from the sandbox, label included."""
    params: dict = {'messages': [SYS, {'role': 'user', 'content': prompt}], '_reply': reply}
    if label is not None:
        params[LABEL_FIELD] = label
    return params


def labels_by_reply(tracker: ChatTrackerProcessor) -> dict[str, str | None]:
    return {str(t['messages'][-1].get('content')): t.get('role') for t in tracker.get_result()['trajectories']}


# --- the handler strips and applies the label ------------------------------


@pytest.mark.asyncio
async def test_label_reaches_the_span_and_is_stripped_from_the_call():
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    client = RecordingClient()
    handler = make_handler(client)

    with processor_context([tracker], finalize_on_exit=False):
        await handler._produce_response('r1', request(worker_label('a'), 'A1', 'ta'), rsio_mode=False)
        await asyncio.sleep(DISPATCH_SETTLE)

    assert labels_by_reply(tracker) == {'A1': worker_label('a')}
    assert LABEL_FIELD not in client.calls[0], 'the label must not reach the provider'


@pytest.mark.asyncio
async def test_label_is_not_left_on_the_client_afterwards():
    """A leftover label would be inherited by the next unlabelled request."""
    client = RecordingClient()
    handler = make_handler(client)

    await handler._produce_response('r1', request(worker_label('a'), 'A1', 'ta'), rsio_mode=False)

    assert client._chat_phase is None


@pytest.mark.asyncio
async def test_missing_label_is_counted_and_recorded_as_unattributed():
    """Warn and count rather than raise: one unattributable call should not abort a turn."""
    client = RecordingClient()
    handler = make_handler(client)

    await handler._produce_response('r1', request(None, 'A1', 'ta'), rsio_mode=False)

    assert handler.unattributed_requests == 1
    assert handler.label_for('r1') == UNATTRIBUTED


@pytest.mark.asyncio
async def test_a_stray_private_field_raises_before_the_call():
    """sanitize_openai_params has a fixed denylist, so an unknown key is forwarded.

    This assertion is the only thing between a forgotten strip and sending a
    private field to the provider as a completion parameter.
    """
    client = RecordingClient()
    handler = make_handler(client)
    payload = request(worker_label('a'), 'A1', 'ta')
    payload['_hitl_smuggled'] = 'x'

    with pytest.raises(AssertionError, match='must not reach the provider'):
        await handler._produce_response('r1', payload, rsio_mode=False)

    assert client.calls == [], 'must fail before calling the provider'


@pytest.mark.asyncio
async def test_label_is_recorded_per_request_for_usage_attribution():
    """Usage arrives on the response emit, outside the span -- request_id bridges it."""
    client = RecordingClient()
    handler = make_handler(client)

    await handler._produce_response('r1', request(worker_label('a'), 'A1', 'ta'), rsio_mode=False)
    await handler._produce_response('r2', request(worker_label('b'), 'B1', 'tb'), rsio_mode=False)

    assert handler.label_for('r1') == worker_label('a')
    assert handler.label_for('r2') == worker_label('b')
    assert handler.label_for('missing') is None


@pytest.mark.asyncio
async def test_concurrent_sessions_on_one_shared_client_stay_separate():
    """The property that removes per-session clients.

    One client, two sessions, overlapping calls: the labels still match. Safe
    because the label write is adjacent to the call and ``labelled`` cannot
    yield in between.
    """
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    client = RecordingClient(delay=FAST)
    handler = make_handler(client)

    with processor_context([tracker], finalize_on_exit=False):
        await asyncio.gather(
            handler._produce_response('r1', request(worker_label('a'), 'A1', 'ta'), rsio_mode=False),
            handler._produce_response('r2', request(worker_label('b'), 'B1', 'tb'), rsio_mode=False),
            handler._produce_response('r3', request('user/orchestrator', 'U1', 'plan'), rsio_mode=False),
        )
        await asyncio.sleep(DISPATCH_SETTLE)

    assert labels_by_reply(tracker) == {
        'A1': worker_label('a'),
        'B1': worker_label('b'),
        'U1': 'user/orchestrator',
    }


@pytest.mark.asyncio
async def test_failure_inside_the_call_still_clears_the_label():
    class Failing(RecordingClient):
        async def _create_completion(self, messages, model=None, **kwargs):
            raise RuntimeError('provider down')

    client = Failing()
    handler = make_handler(client)

    with pytest.raises(RuntimeError):
        await handler._produce_response('r1', request(worker_label('a'), 'A1', 'ta'), rsio_mode=False)

    assert client._chat_phase is None


@pytest.mark.asyncio
async def test_handler_constructs_the_way_acquire_llm_handler_calls_it():
    """The construction path a real setup() takes, which injection skips.

    Every other test injects a pre-built handler, so a mismatch between what
    ``acquire_llm_handler`` passes and what the handler accepts stays invisible
    until a real ``setup()`` raises a TypeError.
    """
    from nexus.extensions.tasks.hitl.actor.resources import acquire_llm_handler, release_llm_handler

    runtime = StubRuntime()
    client = RecordingClient()

    handler = acquire_llm_handler(
        runtime,
        openai_client=client,
        factory=LabelledLLMRequestHandler,
    )
    try:
        assert isinstance(handler, LabelledLLMRequestHandler)
        assert handler.unattributed_requests == 0
    finally:
        release_llm_handler(runtime)


@pytest.mark.asyncio
async def test_acquire_refuses_to_hand_back_a_different_handler_class():
    """The class selects how requests are attributed, so sharing across styles lies.

    A label-stripping manager handed a handler of some other style would forward its
    private field to the provider and export every span unlabelled, while its own
    unattributed counter read zero. The stand-in below is any such other style.
    """
    from nexus.extensions.tasks.hitl.actor.resources import acquire_llm_handler, release_llm_handler

    class OtherStyleHandler(LabelledLLMRequestHandler):
        """A handler of a different class, which is the whole point of the check."""

    runtime = StubRuntime()
    client = RecordingClient()
    acquire_llm_handler(runtime, openai_client=client, factory=LabelledLLMRequestHandler)
    try:
        with pytest.raises(RuntimeError, match='already has a'):
            acquire_llm_handler(runtime, openai_client=client, factory=OtherStyleHandler)
    finally:
        release_llm_handler(runtime)

    # The check is isinstance, not equality: a subclass of what is registered is
    # accepted, since it attributes requests the same way.
    runtime = StubRuntime()
    acquire_llm_handler(runtime, openai_client=client, factory=OtherStyleHandler)
    try:
        assert acquire_llm_handler(runtime, openai_client=client, factory=LabelledLLMRequestHandler)
    finally:
        release_llm_handler(runtime)


@pytest.mark.asyncio
async def test_each_request_keeps_its_own_label_under_concurrency():
    """Attribution rides request_id, not handler state.

    A per-turn handler slot meant every concurrent member's request landed under
    whichever member entered the scope first.
    """
    client = RecordingClient(delay=FAST)
    handler = make_handler(client)

    await asyncio.gather(
        handler._produce_response('r1', request(worker_label('a'), 'A1', 'ta'), rsio_mode=False),
        handler._produce_response('r2', request(worker_label('b'), 'B1', 'tb'), rsio_mode=False),
    )

    assert handler.label_for('r1') == worker_label('a')
    assert handler.label_for('r2') == worker_label('b')
    assert handler.unattributed_requests == 0


# --- the proxy injects the label -------------------------------------------


class FakeBridge:
    def __init__(self) -> None:
        self.seen: list[dict] = []
        self.extra = 'delegated'

    async def request(self, openai_params: dict, timeout: float = 300) -> dict:
        self.seen.append(dict(openai_params))
        return {'ok': True}


@pytest.mark.asyncio
async def test_labelling_bridge_stamps_the_label():
    inner = FakeBridge()
    wrapped = _LabellingBridge(inner, worker_label('a'))

    await wrapped.request({'model': 'm', 'messages': []})

    assert inner.seen[0][LABEL_FIELD] == worker_label('a')


@pytest.mark.asyncio
async def test_labelling_bridge_does_not_mutate_the_caller_dict():
    """A resend re-spreads the same params; mutating them would double-stamp."""
    inner = FakeBridge()
    wrapped = _LabellingBridge(inner, worker_label('a'))
    params = {'model': 'm'}

    await wrapped.request(params)

    assert LABEL_FIELD not in params


def test_labelling_bridge_delegates_everything_else():
    inner = FakeBridge()

    assert _LabellingBridge(inner, 'x').extra == 'delegated'


@pytest.mark.asyncio
async def test_attach_label_wraps_the_bridge_once():
    """Re-attaching must not stack wrappers, which would stamp twice."""

    class FakeProxy:
        port = 1234

        def __init__(self) -> None:
            self.bridge = FakeBridge()

    proxy = FakeProxy()
    attach_label(proxy, worker_label('a'))
    first = proxy.bridge
    attach_label(proxy, worker_label('a'))

    assert proxy.bridge is first
    assert isinstance(proxy.bridge, _LabellingBridge)


def test_attach_label_requires_a_started_proxy():
    """Before start() there is no bridge; silently doing nothing would lose labels."""

    class NotStarted:
        bridge = None

    with pytest.raises(RuntimeError, match='call start\\(\\)'):
        attach_label(NotStarted(), worker_label('a'))


def test_attach_label_requires_a_label():
    """An unlabelled proxy would emit unattributable requests silently."""

    class FakeProxy:
        bridge = FakeBridge()

    with pytest.raises(ValueError, match='non-empty session_label'):
        attach_label(FakeProxy(), '')


# --- the two halves joined -------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_injection_through_handler_to_exported_label():
    """End to end: what the proxy stamps is what the trajectory reports.

    Two sessions whose requests are byte-identical apart from the label -- the
    case tree shape alone cannot attribute.
    """
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    client = RecordingClient(delay=FAST)
    handler = make_handler(client)

    async def through_proxy(session: str, reply: str, request_id: str):
        """Stamp like the proxy, then hand to the handler like the runtime does."""
        stamped: dict = {}

        class Capture:
            async def request(self, openai_params, timeout=300):
                stamped.update(openai_params)
                return {}

        await _LabellingBridge(Capture(), worker_label(session)).request(
            {
                'messages': [SYS, {'role': 'user', 'content': 'same prompt'}],
                '_reply': reply,
            }
        )
        return await handler._produce_response(request_id, stamped, rsio_mode=False)

    with processor_context([tracker], finalize_on_exit=False):
        await asyncio.gather(
            through_proxy('a', 'A1', 'r1'),
            through_proxy('b', 'B1', 'r2'),
        )
        await asyncio.sleep(DISPATCH_SETTLE)

    assert labels_by_reply(tracker) == {'A1': worker_label('a'), 'B1': worker_label('b')}
    assert handler.unattributed_requests == 0


class StockClient(OpenAICompatibleClient):
    """A client with no ``_chat_phase`` of its own, as production supplies it.

    ``RecordingClient`` mixes in ``LabelReadingClient``, so every test using it
    would pass even if the handler never enabled labels. This one would not.
    """

    def __init__(self) -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self.calls: list[dict] = []

    async def _create_completion(self, messages, model=None, **kwargs):
        self.calls.append({'messages': messages, 'model': model, **kwargs})
        return ChatCompletion.model_validate(
            {
                'id': 'chatcmpl-test',
                'object': 'chat.completion',
                'created': 0,
                'model': model or 'mock-model',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {'role': 'assistant', 'content': kwargs.get('_reply', 'ok')},
                    }
                ],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            }
        )


def test_a_stock_client_has_no_label_attribute_before_the_handler_sees_it():
    # The premise: nothing in nexus sets _chat_phase, so an unprepared client
    # exports every span unlabelled.
    assert not hasattr(StockClient(), '_chat_phase')


@pytest.mark.asyncio
async def test_the_handler_labels_spans_from_a_stock_client():
    """The handler must prepare the client it is given, not assume a prepared one."""
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    client = StockClient()
    handler = make_handler(client)

    with processor_context([tracker], finalize_on_exit=False):
        await handler._produce_response('r1', request(worker_label('a'), 'A1', 'ta'), rsio_mode=False)
        await asyncio.sleep(DISPATCH_SETTLE)

    assert labels_by_reply(tracker) == {'A1': worker_label('a')}


@pytest.mark.asyncio
async def test_preparing_one_client_does_not_label_another():
    handler_client = StockClient()
    make_handler(handler_client)
    untouched = StockClient()

    with labels.labelled(handler_client, worker_label('a')):
        assert handler_client._chat_phase == worker_label('a')
        assert not hasattr(untouched, '_chat_phase')
