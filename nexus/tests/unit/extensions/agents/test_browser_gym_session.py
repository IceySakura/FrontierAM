from __future__ import annotations

import asyncio

from types import SimpleNamespace

import pytest

from nexus.extensions.agents.browser_gym.agent import _open_declared_browser_session


class _Runtime:
    def __init__(
        self,
        tool_state: dict,
        *,
        fail_navigation: bool = False,
        raise_navigation: bool = False,
        cancel_navigation: bool = False,
    ) -> None:
        self.tool_state = tool_state
        self.fail_navigation = fail_navigation
        self.raise_navigation = raise_navigation
        self.cancel_navigation = cancel_navigation
        self.events: list[tuple] = []

    async def create_browser_session(self, **kwargs):
        self.events.append(('create', kwargs))
        return SimpleNamespace(session_id='browser-1')

    async def browser_goto(self, session_id: str, url: str, *, timeout: int | None = None):
        # The session must not become verifier-visible before navigation succeeds.
        assert 'browser_session_id' not in self.tool_state
        self.events.append(('goto', session_id, url, timeout))
        if self.cancel_navigation:
            raise asyncio.CancelledError
        if self.raise_navigation:
            raise RuntimeError('navigation failed')
        return SimpleNamespace(
            success=not self.fail_navigation,
            url='about:blank' if self.fail_navigation else url,
        )

    async def destroy_browser_session(self, session_id: str):
        self.events.append(('destroy', session_id))


class _Context:
    def __init__(self, runtime: _Runtime, tool_state: dict) -> None:
        self.runtime = runtime
        self.tool_state = tool_state
        self.runtime_names: list[str] = []

    async def get_runtime(self, name: str = 'main'):
        self.runtime_names.append(name)
        return self.runtime


@pytest.mark.asyncio
async def test_declared_browser_session_is_published_after_navigation() -> None:
    tool_state: dict = {}
    runtime = _Runtime(tool_state)
    context = _Context(runtime, tool_state)

    session_id = await _open_declared_browser_session(
        context,
        {
            'start_url': 'https://example.com/task',
            'headless': False,
            'viewport_width': 1440,
            'viewport_height': 900,
            'nav_timeout_ms': 15000,
            'runtime_name': 'main',
        },
    )

    assert session_id == 'browser-1'
    assert context.runtime_names == ['main']
    assert runtime.events == [
        (
            'create',
            {
                'headless': False,
                'viewport_width': 1440,
                'viewport_height': 900,
                'timeout': 15000,
            },
        ),
        ('goto', 'browser-1', 'https://example.com/task', 15000),
    ]
    assert tool_state == {'browser_session_id': 'browser-1'}


@pytest.mark.asyncio
async def test_initial_navigation_failure_destroys_unpublished_session() -> None:
    tool_state: dict = {}
    runtime = _Runtime(tool_state, fail_navigation=True)
    context = _Context(runtime, tool_state)

    with pytest.raises(RuntimeError, match='Initial browser navigation.*failed'):
        await _open_declared_browser_session(context, {'start_url': 'https://example.com/task'})

    assert runtime.events[-1] == ('destroy', 'browser-1')
    assert tool_state == {}


@pytest.mark.asyncio
async def test_initial_navigation_exception_destroys_unpublished_session() -> None:
    tool_state: dict = {}
    runtime = _Runtime(tool_state, raise_navigation=True)
    context = _Context(runtime, tool_state)

    with pytest.raises(RuntimeError, match='navigation failed'):
        await _open_declared_browser_session(context, {'start_url': 'https://example.com/task'})

    assert runtime.events[-1] == ('destroy', 'browser-1')
    assert tool_state == {}


@pytest.mark.asyncio
async def test_initial_navigation_cancellation_destroys_unpublished_session() -> None:
    tool_state: dict = {}
    runtime = _Runtime(tool_state, cancel_navigation=True)
    context = _Context(runtime, tool_state)

    with pytest.raises(asyncio.CancelledError):
        await _open_declared_browser_session(context, {'start_url': 'https://example.com/task'})

    assert runtime.events[-1] == ('destroy', 'browser-1')
    assert tool_state == {}


@pytest.mark.asyncio
async def test_no_browser_declaration_does_not_acquire_runtime() -> None:
    tool_state: dict = {}
    runtime = _Runtime(tool_state)
    context = _Context(runtime, tool_state)

    assert await _open_declared_browser_session(context, None) is None
    assert context.runtime_names == []
