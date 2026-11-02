"""Each hy-live conversation sends its own cache_task_id, and the sandbox is up before the user speaks."""

from __future__ import annotations

import logging

from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import parse_qsl

import pytest

from nexus.core.openai_client import OpenAIConfig
from nexus.extensions.openai_clients.openai import OpenAIClient
from nexus.extensions.projects.hy_live import agent as agent_module
from nexus.extensions.projects.hy_live.cache_task_id import conversation_client
from nexus.extensions.projects.hy_live.coordination.acp_updates import AcpUpdateRouter
from nexus.extensions.projects.hy_live.coordination.backend import AcpSessionExecutor
from nexus.extensions.projects.hy_live.coordination.models import SessionRef, TaskResult
from nexus.extensions.projects.hy_live.hylive_shell import Episode, LiveEpisode, LiveShell
from nexus.extensions.projects.hy_live.user_simulator import LLMUserSimulator, SimulatedConversation
from nexus.utils.cache_task_id import get_cache_task_id


def _gateway_client(client_key: str, api_key: str = 'app:key?provider=aihub') -> OpenAIClient:
    client = OpenAIClient(OpenAIConfig(client_args={'api_key': api_key, 'base_url': 'http://gateway'}, request_args={}))
    client._client_key = client_key
    return client


def _query(client: Any) -> dict[str, str]:
    return dict(parse_qsl(client.client.api_key.partition('?')[2]))


class OpaqueClient:
    """A client that cannot carry a cache_task_id, like the HyTuner policy client."""

    _client_key = 'main'


def test_a_conversation_client_is_a_copy_with_a_fresh_id_and_the_same_client_key() -> None:
    base = _gateway_client('external', 'app:key?provider=aihub&cache_task_id=configured')

    first = conversation_client(base, conversation='midbrain', enabled=True)
    second = conversation_client(base, conversation='midbrain', enabled=True)

    assert type(first) is OpenAIClient
    assert first is not base
    assert first._client_key == 'external', 'trace processors keep attributing calls to the same client'
    assert _query(first)['provider'] == 'aihub', 'other gateway parameters are kept'
    assert get_cache_task_id(first) not in {None, 'configured'}
    assert get_cache_task_id(first) != get_cache_task_id(second), 'every conversation gets its own id'
    assert get_cache_task_id(base) == 'configured', 'the shared client is left alone'


def test_a_disabled_conversation_uses_the_client_as_is() -> None:
    base = _gateway_client('main')

    assert conversation_client(base, conversation='cerebellum', enabled=False) is base


def test_a_client_without_cache_task_id_support_is_kept_unless_required(caplog: pytest.LogCaptureFixture) -> None:
    client = OpaqueClient()

    with caplog.at_level(logging.WARNING):
        assert conversation_client(cast(Any, client), conversation='midbrain', enabled=True) is client
    assert 'cannot send a cache_task_id' in caplog.text
    with pytest.raises(ValueError, match='cache_task_id.cerebellum'):
        conversation_client(cast(Any, client), conversation='cerebellum', enabled=True, required=True)


class RecordingExecutor(AcpSessionExecutor):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.configs: list[dict[str, Any]] = []

    async def _run_codebuddy(self, prompt: str, config: dict[str, Any]) -> TaskResult:
        self.configs.append(config)
        return TaskResult(final_output='done', trajectory=[{'role': 'assistant', 'content': 'done'}])


async def _drop(*_args: Any) -> None:
    return None


class _Runtime:
    def on(self, event: str, handler: Any) -> None:
        return None


def _session_executor(*, enabled: bool) -> tuple[RecordingExecutor, dict[str, Any]]:
    clients: dict[str, Any] = {'external': _gateway_client('external')}

    async def get_runtime() -> _Runtime:
        return _Runtime()

    ctx = SimpleNamespace(
        run_id='run-1', openai_clients=clients, get_openai_client=lambda name: clients[name], get_runtime=get_runtime
    )
    executor = RecordingExecutor(
        ctx=cast(Any, ctx),
        update_router=AcpUpdateRouter(lambda *_: None),
        turn_ended_sink=_drop,
        trusted_config={'acp_cli': 'workbuddy', 'llm_client_name': 'external'},
        session_cache_task_id=enabled,
    )
    return executor, clients


async def test_every_project_session_has_its_own_client() -> None:
    executor, clients = _session_executor(enabled=True)
    first = SessionRef(session_id='project-1', owner_id='owner-1', working_dir='/workspace/task-1')
    second = SessionRef(session_id='project-2', owner_id='owner-1', working_dir='/workspace/task-2')

    await executor._run_turn(first, turn_id='turn-1', prompt='build')
    await executor._run_turn(second, turn_id='turn-2', prompt='write')
    await executor._run_turn(first, turn_id='turn-3', prompt='continue')

    names = [config['llm_client_name'] for config in executor.configs]
    assert names[0] == names[2], 'later turns reuse the Session client'
    assert names[0] != names[1]
    assert all(name.startswith('external@') for name in names)
    first_id, second_id = get_cache_task_id(clients[names[0]]), get_cache_task_id(clients[names[1]])
    assert first_id is not None
    assert second_id is not None
    assert first_id != second_id
    assert get_cache_task_id(clients['external']) is None, 'the shared backend client is left alone'


async def test_project_sessions_share_the_configured_client_when_disabled() -> None:
    executor, clients = _session_executor(enabled=False)

    await executor._run_turn(SessionRef(session_id='project-1', owner_id='owner-1'), turn_id='turn-1', prompt='build')

    assert executor.configs[0]['llm_client_name'] == 'external'
    assert set(clients) == {'external'}


class WarmRuntime:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def run_command(self, command: str, **kwargs: Any) -> SimpleNamespace:
        self.order.append('preload')
        return SimpleNamespace(return_code=0, stdout='', stderr='', status='done')


def _install_context(monkeypatch: pytest.MonkeyPatch, clients: dict[str, Any], get_runtime: Any) -> dict[str, Any]:
    context = SimpleNamespace(
        run_id='run-1',
        openai_clients=dict(clients),
        get_openai_client=lambda name: clients[name],
        get_runtime=get_runtime,
    )
    monkeypatch.setattr(agent_module, 'require_context', lambda: context)
    monkeypatch.setattr(agent_module, 'get_current_span_id', lambda: None)
    captured: dict[str, Any] = {}

    async def run(shell: LiveShell, **kwargs: Any) -> LiveEpisode:
        captured.update(kwargs, shell=shell)
        return LiveEpisode(episode=Episode(conversation=SimulatedConversation(), summary='', tasks=[]), state=shell.state)

    monkeypatch.setattr(LiveShell, 'run', run)
    return captured


_SCENARIO = {
    'meta': {'script_id': 's-1', 'topic_id': 't-1', 'expected_turns': [1, 3]},
    'topic': '做个小游戏',
    'user': {'background': '用户想玩个小游戏。', 'style': 'A'},
    'goal': '拿到一个能玩的小游戏',
    'ending': {'normal': '游戏做好了。'},
}


def _llm_user(**fields: Any) -> agent_module.LiveAgentAgentInput:
    return agent_module.LiveAgentAgentInput.model_validate({
        'user_simulator': {'type': 'llm', 'scenario': _SCENARIO, 'initial_prompt': 'build'},
        **fields,
    })


async def test_midbrain_and_user_simulator_get_their_own_ids_and_the_cerebellum_none_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    async def get_runtime() -> WarmRuntime:
        return WarmRuntime(order)

    clients = {name: _gateway_client(name) for name in ('main', 'external', 'user_simulator')}
    captured = _install_context(monkeypatch, clients, get_runtime)

    await agent_module.live_agent(_llm_user())

    clients_of = captured['shell'].runner.ports.clients
    cerebellum_client = clients_of['cerebellum']
    midbrain_client = clients_of['midbrain']
    simulator = captured['simulator']
    assert isinstance(simulator, LLMUserSimulator)
    assert cerebellum_client is clients['main'], 'the cerebellum is off by default'
    assert midbrain_client._client_key == 'external', 'the midbrain shares the backend client unless configured'
    ids = {get_cache_task_id(midbrain_client), get_cache_task_id(simulator.runtime.client)}
    assert None not in ids
    assert len(ids) == 2
    assert captured['shell'].executor.session_cache_task_id is True


async def test_the_cerebellum_id_is_opt_in_and_a_client_that_cannot_carry_it_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def get_runtime() -> WarmRuntime:
        return WarmRuntime([])

    clients: dict[str, Any] = {name: _gateway_client(name) for name in ('main', 'external', 'user_simulator')}
    captured = _install_context(monkeypatch, clients, get_runtime)
    await agent_module.live_agent(_llm_user(cache_task_id={'cerebellum': True}))
    assert get_cache_task_id(captured['shell'].runner.ports.clients['cerebellum']) is not None

    clients['main'] = OpaqueClient()
    with pytest.raises(ValueError, match='cache_task_id.cerebellum'):
        await agent_module.live_agent(_llm_user(cache_task_id={'cerebellum': True}))


async def test_the_sandbox_is_up_before_the_conversation_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []

    async def get_runtime() -> WarmRuntime:
        order.append('runtime')
        return WarmRuntime(order)

    captured = _install_context(monkeypatch, {'main': OpaqueClient(), 'external': OpaqueClient()}, get_runtime)
    stand_in = LiveShell.run

    async def run(shell: LiveShell, **kwargs: Any) -> LiveEpisode:
        order.append('episode')
        return await stand_in(shell, **kwargs)

    monkeypatch.setattr(LiveShell, 'run', run)

    await agent_module.live_agent(agent_module.LiveAgentAgentInput(prompt='build'))

    assert order == ['runtime', 'preload', 'episode']
    assert captured, 'the episode ran'


async def test_a_sandbox_that_does_not_start_ends_the_run_before_the_user_speaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def get_runtime() -> Any:
        raise RuntimeError('sandbox create failed')

    captured = _install_context(monkeypatch, {'main': OpaqueClient(), 'external': OpaqueClient()}, get_runtime)

    with pytest.raises(RuntimeError, match='sandbox create failed'):
        await agent_module.live_agent(agent_module.LiveAgentAgentInput(prompt='build'))
    assert captured == {}, 'no user turn and no cerebellum call without a sandbox'
