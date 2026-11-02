import asyncio
import json

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.claude_code.native_bridge import agent_runtime
from nexus.extensions.agents.claude_code.native_bridge.adapters.posix import (
    BasePosixClaudeNativeBridgeAdapter,
    _SessionPaths,
)
from nexus.extensions.tasks.os_world.execution.claude_code import (
    OSWorldClaudeExecutionProfile,
    OSWorldClaudeNativeBridgeAdapter,
    _ParentDrainState,
)
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime


def message(text='TASK_INFEASIBLE', parent=None):
    return {
        'type': 'assistant',
        'session_id': 'root',
        'parent_tool_use_id': parent,
        'message': {'content': [{'type': 'text', 'text': text}]},
    }


def test_notifications_and_repeated_init_do_not_extend_first_deadline():
    state = _ParentDrainState(30)
    state.feed(message(), 10)
    state.feed({'type': 'system', 'subtype': 'init', 'session_id': 'root'}, 20)
    state.feed(message(), 29)
    state.feed(message('child still working', parent='child'), 35)
    assert state.deadline == 40
    assert not state.expired(39)
    assert state.expired(40)


@pytest.mark.parametrize(
    'event',
    [
        message('I must do more work'),
        message('TASK_DONE'),
        {'type': 'result', 'session_id': 'root', 'subtype': 'error_during_execution', 'is_error': True},
        {'type': 'assistant', 'session_id': 'root', 'message': {'content': [{'type': 'tool_use', 'id': 'one'}]}},
    ],
)
def test_conflicting_root_work_cannot_rearm_deadline(event):
    state = _ParentDrainState(30)
    state.feed(message(), 0)
    state.feed(event, 1)
    state.feed(message(), 2)
    assert state.deadline == 30
    assert not state.expired(60)


@pytest.mark.asyncio
@pytest.mark.parametrize('stop_ok', [True, False])
async def test_deadline_stops_identified_process_and_preserves_real_code(monkeypatch, tmp_path, stop_ok):
    raw = json.dumps(message()) + '\n'
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.parent_terminal_grace_seconds = 0.01
    adapter.diagnostic_dir = tmp_path
    adapter.diagnostics = {}
    adapter._read_parent_events = AsyncMock(side_effect=[(raw.encode(), len(raw)), (b'', len(raw))])
    adapter._read_file = AsyncMock(return_value=raw)
    adapter._capture_diagnostics = AsyncMock()

    async def command(*args, **kwargs):
        cmd = kwargs['command']
        stopped = 'pidfd_open' in cmd
        return SimpleNamespace(
            return_code=0 if not stopped or stop_ok else 1,
            status='Finished',
            stdout=json.dumps({'confirmed': stop_ok}) if stopped else 'RUNNING\n',
            stderr='',
        )

    adapter.runtime = SimpleNamespace(run_command=AsyncMock(side_effect=command))

    async def blocked(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(BasePosixClaudeNativeBridgeAdapter, '_poll_until_done', blocked)
    if stop_ok:
        assert await adapter._poll_until_done(_SessionPaths('/tmp/test'), 10) == (None, True, False)
        assert adapter.diagnostics['budget_stop_confirmed']
    else:
        with pytest.raises(RuntimeError, match='termination'):
            await adapter._poll_until_done(_SessionPaths('/tmp/test'), 10)
        assert not adapter.diagnostics.get('budget_stop_confirmed')
    assert adapter.diagnostics['budget_reason'] == 'parent_terminal_grace'
    assert adapter.diagnostics['parent_terminal_marker'] == 'TASK_INFEASIBLE'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('confirmed', 'api_error', 'expected'),
    [
        (True, False, True),
        (False, False, False),
        (True, True, False),
    ],
)
async def test_parent_drain_classification_requires_evidence(monkeypatch, confirmed, api_error, expected):
    async def runner(**kwargs):
        adapter = kwargs['adapter_factory'](kwargs['runtime'])
        adapter.diagnostics = {
            'budget_reason': 'parent_terminal_grace',
            'budget_stop_confirmed': confirmed,
            'parent_terminal_marker': 'TASK_INFEASIBLE',
        }
        events = [message(), message('child waiting', parent='child')]
        if api_error:
            events.append({
                'type': 'result',
                'session_id': 'root',
                'subtype': 'error_during_execution',
                'is_error': True,
                'api_error_status': 401,
                'result': 'API rejected',
            })
        return {'success': False, 'exit_code': -1, 'output': '\n'.join(map(json.dumps, events)), 'stderr': 'drained'}

    monkeypatch.setattr(agent_runtime, 'setup_and_run_native_bridge_claude_code', runner)
    profile = OSWorldClaudeExecutionProfile({'working_dir': '/tmp', 'guide_text': ''}, {}, 'task')
    result = await profile.execute(
        runtime=object.__new__(AGSOSWorldRuntime),
        launch=BridgeLaunchResult(base_url='http://test', local_base_url='http://test'),
        timeout=60,
    )
    assert result['success'] is expected
    assert result['exit_status'] == ('infeasible' if expected else 'tool_error')
    assert result['cc_return_code'] == -1
    assert result['termination_reason'] == 'parent_terminal_grace'


@pytest.mark.asyncio
async def test_observation_failure_falls_back_without_early_success(monkeypatch, tmp_path):
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.parent_terminal_grace_seconds = 0.01
    adapter.diagnostics = {}
    adapter.diagnostic_dir = tmp_path
    adapter._read_parent_events = AsyncMock(side_effect=ConnectionError('file endpoint'))

    async def ordinary_poll(*args):
        await asyncio.sleep(0.01)
        return None, False, True

    monkeypatch.setattr(BasePosixClaudeNativeBridgeAdapter, '_poll_until_done', ordinary_poll)
    assert await adapter._poll_with_parent_drain(_SessionPaths('/tmp/test'), 10) == (None, False, True)
    assert adapter.diagnostics['parent_terminal_observation_error'] == 'ConnectionError'
    assert 'parent_terminal_marker' not in adapter.diagnostics


@pytest.mark.asyncio
async def test_disabled_parent_drain_does_not_read_stream(monkeypatch):
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.parent_terminal_grace_seconds = None
    adapter._read_parent_events = AsyncMock()
    normal = AsyncMock(return_value=(0, False, False))
    monkeypatch.setattr(BasePosixClaudeNativeBridgeAdapter, '_poll_until_done', normal)
    assert await adapter._poll_with_parent_drain(_SessionPaths('/tmp/test'), 10) == (0, False, False)
    adapter._read_parent_events.assert_not_awaited()


@pytest.mark.asyncio
async def test_parent_drain_cancellation_awaits_both_tasks(monkeypatch):
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.parent_terminal_grace_seconds = 30
    entered = [asyncio.Event(), asyncio.Event()]
    finished = [asyncio.Event(), asyncio.Event()]

    async def block(index):
        entered[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            finished[index].set()

    monkeypatch.setattr(BasePosixClaudeNativeBridgeAdapter, '_poll_until_done', lambda *args: block(0))
    adapter._read_parent_events = lambda *args: block(1)
    task = asyncio.create_task(adapter._poll_with_parent_drain(_SessionPaths('/tmp/test'), 60))
    await asyncio.gather(*(e.wait() for e in entered))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(e.is_set() for e in finished)
