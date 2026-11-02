"""Contract tests for opening a labelled PTY session against real upstream state.

The migration's premise is that a labelled session can be opened without changing
upstream: HITL calls ``_prepare_runtime_session`` (which starts the proxy but does
not spawn Claude), wraps the proxy's bridge, and writes ``_PTY_SESSIONS`` — all of
which the upstream turn path already reads. These tests run against the genuine
module globals, so if upstream renames the registry, moves the proxy off the
session wrapper, or starts creating sessions eagerly, they fail.

Offline: no Claude binary, no sandbox, no credentials.
"""

import inspect

from typing import Any

import pytest

from nexus.extensions.agents.claude_code.interactive import runtime as interactive_runtime
from nexus.extensions.tasks.hitl.actor.pty_claude_code import labelled_runtime
from nexus.extensions.tasks.hitl.trace.labels import LABEL_FIELD, worker_label


class StubBridge:
    """The proxy's LLM bridge; records what reaches it."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def request(self, openai_params: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
        del timeout
        self.requests.append(openai_params)
        return {'ok': True}


class StubProxy:
    port = 9001

    def __init__(self) -> None:
        self.bridge: Any = StubBridge()
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class StubPty:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def upstream(monkeypatch):
    """Isolate the real registry and stub only the PTY/proxy creation.

    The stub is bound to the real ``_prepare_runtime_session`` signature before it
    records anything, so a call that upstream would reject raises here too. A
    ``**kwargs`` stub accepts arguments the real function requires and would let a
    missing one through to a sandbox -- which is how eighteen required keyword-only
    arguments went unnoticed until a real run.
    """
    interactive_runtime._PTY_SESSIONS.clear()
    real_signature = inspect.signature(interactive_runtime._prepare_runtime_session)
    built: list[Any] = []

    async def fake_prepare(_session_id, **kwargs):
        real_signature.bind(_session_id, **kwargs)
        session = interactive_runtime._InteractiveRuntimeSession(
            pty_session=StubPty(),
            proxy=StubProxy(),
        )
        built.append((session, kwargs))
        return session

    monkeypatch.setattr(interactive_runtime, '_prepare_runtime_session', fake_prepare)
    yield built
    interactive_runtime._PTY_SESSIONS.clear()


@pytest.mark.asyncio
async def test_open_session_registers_where_the_turn_path_looks(upstream):
    """The whole seam: upstream's turn path must find what we registered."""
    result = await labelled_runtime.open_session(session_id='s1', label=worker_label('a'))

    assert result == {
        'session_id': 's1',
        'label': worker_label('a'),
        'proxy_port': 9001,
        'reused': False,
    }
    # Registered under the id the host will pass to run_turn, in upstream's own dict.
    assert 's1' in interactive_runtime._PTY_SESSIONS
    assert len(upstream) == 1


@pytest.mark.asyncio
async def test_the_label_reaches_requests_made_through_the_proxy(upstream):
    await labelled_runtime.open_session(session_id='s1', label=worker_label('a'))
    session = interactive_runtime._PTY_SESSIONS['s1']

    await session.proxy.bridge.request({'messages': [{'role': 'user', 'content': 'hi'}]})

    inner = upstream[0][0].proxy.bridge._inner
    assert inner.requests[0][LABEL_FIELD] == 'worker/a'


@pytest.mark.asyncio
async def test_opening_twice_keeps_the_first_label_and_does_not_rebuild(upstream):
    await labelled_runtime.open_session(session_id='s1', label=worker_label('a'))

    again = await labelled_runtime.open_session(session_id='s1', label=worker_label('b'))

    assert again['reused'] is True
    assert again['label'] == worker_label('a'), 'a live session must not be relabelled'
    assert len(upstream) == 1, 'the second call must not start another proxy'


@pytest.mark.asyncio
async def test_an_empty_label_is_refused_before_anything_is_built(upstream):
    with pytest.raises(ValueError, match='non-empty label'):
        await labelled_runtime.open_session(session_id='s1', label='')

    assert upstream == []
    assert interactive_runtime._PTY_SESSIONS == {}


@pytest.mark.asyncio
async def test_an_unlabelled_existing_session_is_refused_not_silently_reused(upstream):
    """Reusing it would export its trajectories as unattributed."""
    interactive_runtime._PTY_SESSIONS['s1'] = interactive_runtime._InteractiveRuntimeSession(
        pty_session=StubPty(),
        proxy=StubProxy(),
    )

    with pytest.raises(RuntimeError, match='unlabelled proxy'):
        await labelled_runtime.open_session(session_id='s1', label=worker_label('a'))


@pytest.mark.asyncio
async def test_a_failure_while_labelling_does_not_leak_the_session(upstream, monkeypatch):
    def explode(_proxy, _label):
        raise RuntimeError('bridge missing')

    monkeypatch.setattr('nexus.extensions.tasks.hitl.actor.labelling.proxy.attach_label', explode)

    with pytest.raises(RuntimeError, match='bridge missing'):
        await labelled_runtime.open_session(session_id='s1', label=worker_label('a'))

    # Nothing registered, and the half-built session was closed rather than left
    # holding a port: it is not in the registry, so nothing else would close it.
    assert interactive_runtime._PTY_SESSIONS == {}
    assert upstream[0][0].proxy.stopped is True


@pytest.mark.asyncio
async def test_session_label_reports_open_state(upstream):
    absent = await labelled_runtime.session_label(session_id='s1')
    assert absent == {'session_id': 's1', 'label': '', 'open': False}

    await labelled_runtime.open_session(session_id='s1', label=worker_label('a'))

    present = await labelled_runtime.session_label(session_id='s1')
    assert present == {'session_id': 's1', 'label': worker_label('a'), 'open': True}


@pytest.mark.asyncio
async def test_prepare_kwargs_reach_upstream(upstream):
    await labelled_runtime.open_session(
        session_id='s1',
        label=worker_label('a'),
        working_dir='/testbed',
        timeout=99.0,
    )

    _, kwargs = upstream[0]
    assert kwargs['working_dir'] == '/testbed'
    assert kwargs['timeout'] == 99.0


def test_upstream_still_exposes_what_this_seam_depends_on():
    """Names this module reaches into; a rename upstream must fail here, loudly."""
    assert isinstance(interactive_runtime._PTY_SESSIONS, dict)
    assert callable(interactive_runtime._prepare_runtime_session)
    assert hasattr(interactive_runtime, '_CREATE_LOCK')
    # The wrapper must keep exposing the proxy, which is what carries the label.
    assert 'proxy' in interactive_runtime._InteractiveRuntimeSession.__dataclass_fields__
    # And the turn path must still reuse a registered session rather than rebuild.
    source = inspect.getsource(interactive_runtime.run_claude_code_interactive_session_turn)
    assert '_PTY_SESSIONS.get(session_id)' in source
    assert 'if session is None:' in source


def test_every_argument_the_private_helper_requires_is_supplied() -> None:
    """It has eighteen keyword-only arguments and no default for any of them.

    The worker config carries only a few, so the rest come from the public turn
    entry point's defaults. A real run failed with ``missing 11 required
    keyword-only arguments`` before this held. The real function is inspected
    directly rather than through the module attribute, which a fixture replaces.
    """
    real = interactive_runtime._prepare_runtime_session
    assert not hasattr(real, '__wrapped__'), 'inspecting a stub would prove nothing'
    signature = inspect.signature(real)
    required = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty and name != 'session_id'
    }
    defaults = labelled_runtime._prepare_defaults()

    assert required - set(defaults) == set(), 'an argument upstream requires has no default here'
    signature.bind('s1', **{name: defaults[name] for name in required})


def test_the_defaults_are_read_from_upstream_rather_than_restated() -> None:
    """Copied values would drift silently; a wrong npm registry stalls the install."""
    public = inspect.signature(interactive_runtime.run_claude_code_interactive_session_turn).parameters
    defaults = labelled_runtime._prepare_defaults()

    assert defaults == {name: public[name].default for name in defaults}
    assert 'prompt' not in defaults, 'a turn argument is not part of opening a session'


@pytest.mark.asyncio
async def test_a_config_value_still_wins_over_the_upstream_default(upstream) -> None:
    await labelled_runtime.open_session(
        session_id='s1', label=worker_label('a'), working_dir='/testbed', timeout=99.0, cols=120
    )

    _, kwargs = upstream[0]
    assert (kwargs['working_dir'], kwargs['timeout'], kwargs['cols']) == ('/testbed', 99.0, 120)
    assert kwargs['rows'] == 50, "an argument nobody configured takes upstream's default"
