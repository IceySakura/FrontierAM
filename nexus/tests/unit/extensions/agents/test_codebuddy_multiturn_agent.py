from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.codebuddy import agent as codebuddy_agent
from nexus.extensions.agents.codebuddy.agent import CodeBuddyMultiturnInput, run_codebuddy_multiturn
from nexus.extensions.agents.codebuddy.workbuddy_acp import WorkBuddyAcpOptions


pytestmark = pytest.mark.asyncio

_HOST_RELAY = 'nexus.extensions.agents.codebuddy.host_relay::host_relay_extension'
_HOST_TOOLS = {
    'server_name': 'host',
    'tools': [{'name': 'lookup', 'inputSchema': {'type': 'object', 'properties': {}}}],
    'tool_timeouts': {'lookup': 5.0},
}


async def _end_after_one_turn(_request: object) -> None:
    return None


async def _runtime_kwargs(monkeypatch: pytest.MonkeyPatch, **fields: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_multiturn(input_data: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs['runtime_kwargs'])
        return {
            'success': True,
            'output': '',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 0.0,
            'session_id': 's',
            'turns': 1,
        }

    monkeypatch.setattr(codebuddy_agent, 'run_installed_multiturn', fake_multiturn)
    await run_codebuddy_multiturn(
        CodeBuddyMultiturnInput(
            user_prompt='hi', working_dir='/workspace', get_user_input=_end_after_one_turn, **fields
        )
    )
    return captured


async def test_deferred_permissions_route_through_the_public_host_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = await _runtime_kwargs(monkeypatch, permission_policy='defer')
    assert kwargs['extension_factory'] == _HOST_RELAY
    assert kwargs['extension_config'] == {'relay_permissions': True}
    assert kwargs['deferred_permission_timeout'] == 300.0
    assert 'enable_session_mcp' not in kwargs


async def test_host_tools_are_served_without_taking_over_immediate_permissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = await _runtime_kwargs(monkeypatch, permission_policy='allow', host_tools=_HOST_TOOLS)
    assert kwargs['extension_factory'] == _HOST_RELAY
    assert kwargs['extension_config'] == {'host_tools': _HOST_TOOLS}


async def test_immediate_permissions_without_host_tools_need_no_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = await _runtime_kwargs(monkeypatch, permission_policy='allow')
    assert kwargs['extension_factory'] is None
    assert kwargs['extension_config'] is None


async def test_relay_updates_alone_enables_the_host_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = await _runtime_kwargs(monkeypatch, permission_policy='allow', relay_updates=True, relay_turn_id='turn-1')
    assert kwargs['extension_factory'] == _HOST_RELAY
    assert kwargs['extension_config'] == {'relay_updates': True}
    assert kwargs['relay_turn_id'] == 'turn-1', 'Each prompt must carry the id its pushed updates are stamped with'


async def test_relay_updates_without_a_turn_id_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match='relay_turn_id'):
        await _runtime_kwargs(monkeypatch, permission_policy='allow', relay_updates=True)


@pytest.mark.parametrize('key', ['relay_updates', 'client_questions'])
async def test_optional_relay_keys_are_not_sent_unless_requested(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    # An older runtime image rejects unknown relay keys, so the default must leave the config untouched.
    kwargs = await _runtime_kwargs(monkeypatch, permission_policy='defer', relay_turn_id='turn-1')
    assert key not in kwargs['extension_config']
    assert 'relay_turn_id' not in kwargs, 'An older runtime rejects the unknown turn kwarg too'


async def test_unbounded_deferred_permission_wait_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = await _runtime_kwargs(monkeypatch, deferred_permission_timeout=None, host_tools=_HOST_TOOLS)
    assert kwargs['deferred_permission_timeout'] is None
    assert kwargs['extension_config'] == {'relay_permissions': True, 'host_tools': _HOST_TOOLS}


async def test_client_questions_reaches_the_host_relay_with_deferred_permissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = await _runtime_kwargs(monkeypatch, permission_policy='defer', client_questions=True)
    assert kwargs['extension_factory'] == _HOST_RELAY
    assert kwargs['extension_config'] == {'relay_permissions': True, 'client_questions': True}


async def test_client_questions_without_deferred_permissions_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match='client_questions needs permission_policy="defer"'):
        await _runtime_kwargs(monkeypatch, permission_policy='allow', client_questions=True)


_CODEBUDDY_RUNTIME_KEYS = {
    'working_dir',
    'timeout',
    'version',
    'npm_registry',
    'model_id',
    'extra_env',
    'extra_cli_args',
    'config_file_patches',
    'pre_start_scripts',
    'permission_policy',
    'mcp_servers',
    'skip_permissions',
    'extension_factory',
    'extension_config',
    'deferred_permission_timeout',
}


async def test_codebuddy_sends_no_workbuddy_kwargs_to_an_older_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    # NOTE(wallyshi): A runtime image that predates WorkBuddy rejects unknown session kwargs with a TypeError.
    kwargs = await _runtime_kwargs(monkeypatch, permission_policy='defer')
    assert set(kwargs) == _CODEBUDDY_RUNTIME_KEYS


async def test_workbuddy_sends_its_cli_and_default_options(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = await _runtime_kwargs(monkeypatch, acp_cli='workbuddy', permission_policy='defer')
    assert set(kwargs) == _CODEBUDDY_RUNTIME_KEYS | {'acp_cli', 'workbuddy_options'}
    assert kwargs['acp_cli'] == 'workbuddy'
    assert kwargs['workbuddy_options'] == WorkBuddyAcpOptions().runtime_options(None)


async def test_workbuddy_uploads_the_host_package_and_sends_its_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = SimpleNamespace(ws_connect=AsyncMock())
    monkeypatch.setattr(
        codebuddy_agent, 'require_context', lambda: SimpleNamespace(get_runtime=AsyncMock(return_value=runtime))
    )
    upload = AsyncMock(return_value=['/tmp/part000'])
    monkeypatch.setattr(codebuddy_agent, 'upload_workbuddy_package', upload)
    monkeypatch.setattr(codebuddy_agent, 'local_package_sha256', AsyncMock(return_value='c' * 64))
    kwargs = await _runtime_kwargs(
        monkeypatch,
        acp_cli='workbuddy',
        workbuddy={'package_path': '/host/wb.tgz', 'mode': 'craft'},
        fork_source_session_id='source-session',
    )
    upload.assert_awaited_once()
    assert upload.await_args.args[0] is runtime
    options = kwargs['workbuddy_options']
    assert options['package_parts'] == ['/tmp/part000']
    assert options['package_sha256'] == 'c' * 64, 'The runtime verifies the bytes the Host uploaded'
    assert options['mode'] == 'craft'
    assert 'package_path' not in options, 'A Host path means nothing inside the sandbox'
    assert kwargs['fork_source_session_id'] == 'source-session'


@pytest.mark.parametrize(
    ('fields', 'message'),
    [
        ({'workbuddy': {'mode': 'ask'}}, 'workbuddy options need acp_cli="workbuddy"'),
        ({'fork_source_session_id': 's'}, 'only WorkBuddy ACP can fork'),
        ({'acp_cli': 'workbuddy', 'host_tools': _HOST_TOOLS}, 'WorkBuddy ACP accepts no mcpServers'),
        ({'acp_cli': 'workbuddy', 'mcp_servers': [{'name': 'x'}]}, 'WorkBuddy ACP accepts no mcpServers'),
    ],
)
async def test_options_each_cli_cannot_serve_are_rejected(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CodeBuddyMultiturnInput(
            user_prompt='hi', working_dir='/workspace', get_user_input=_end_after_one_turn, **fields
        )
