from __future__ import annotations

import json

from types import SimpleNamespace

import pytest

from nexus.extensions.tasks.aidata.engine.state import _collect_state_via_mcp, _dump_state_error_info


class _FakeMCPManager:
    def __init__(self) -> None:
        self._tools = {
            'wechat_wechat_dump_state': SimpleNamespace(server_name='wechat'),
            'xhs_xhs_dump_state': SimpleNamespace(server_name='xhs'),
        }

    async def call_tool(self, full_name: str, arguments: dict) -> SimpleNamespace:
        assert arguments == {}
        payloads = {
            'wechat_wechat_dump_state': {
                'actions': [{'source': 'wechat', 'type': 'send_message'}],
                'shared': {'wechat': True},
            },
            'xhs_xhs_dump_state': {
                'actions': [{'source': 'xhs', 'type': 'publish_note'}],
                'shared': {'xhs': True},
                # 保留的顶级服务器名不得覆盖命名空间桶。
                'wechat': [{'source': 'xhs', 'type': 'cross_reference'}],
            },
        }
        return SimpleNamespace(output=[{'type': 'text', 'text': json.dumps(payloads[full_name])}])


class _FlakyMCPManager:
    def __init__(self) -> None:
        self._tools = {
            'pdd_pdd_dump_state': SimpleNamespace(server_name='pdd'),
        }
        self.calls = 0

    async def call_tool(self, full_name: str, arguments: dict) -> SimpleNamespace:
        assert full_name == 'pdd_pdd_dump_state'
        assert arguments == {}
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError('transient payload read failed')
        return SimpleNamespace(output=[{'type': 'text', 'text': json.dumps({'orders': [{'order_id': 'O1'}]})}])


class _FailingMCPManager:
    def __init__(self) -> None:
        self._tools = {
            'pdd_pdd_dump_state': SimpleNamespace(server_name='pdd'),
        }
        self.calls = 0

    async def call_tool(self, full_name: str, arguments: dict) -> SimpleNamespace:
        self.calls += 1
        raise RuntimeError('payload too large')


@pytest.mark.asyncio
async def test_collect_state_keeps_per_mcp_namespaces_and_flat_compat() -> None:
    result: dict = {}
    ctx = SimpleNamespace(_mcp_manager=_FakeMCPManager())

    await _collect_state_via_mcp(ctx, result)

    assert result['wechat']['actions'] == [{'source': 'wechat', 'type': 'send_message'}]
    assert result['xhs']['actions'] == [{'source': 'xhs', 'type': 'publish_note'}]
    assert result['xhs']['wechat'] == [{'source': 'xhs', 'type': 'cross_reference'}]

    # 现有 rubric 仍需要旧式扁平切片。
    assert result['actions'] == [
        {'source': 'wechat', 'type': 'send_message'},
        {'source': 'xhs', 'type': 'publish_note'},
    ]
    assert result['shared'] == {'wechat': True, 'xhs': True}

    # 名为 "wechat" 的扁平载荷键被跳过，因为它与命名空间 MCP 桶冲突。
    assert result['wechat'] == {
        'actions': [{'source': 'wechat', 'type': 'send_message'}],
        'shared': {'wechat': True},
    }


@pytest.mark.asyncio
async def test_collect_state_retries_transient_dump_state_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    result: dict = {}
    mgr = _FlakyMCPManager()
    ctx = SimpleNamespace(_mcp_manager=mgr)
    monkeypatch.setenv('AIDATA_DUMP_STATE_MAX_ATTEMPTS', '2')
    monkeypatch.setenv('AIDATA_DUMP_STATE_RETRY_BACKOFF_S', '0')

    await _collect_state_via_mcp(ctx, result)

    assert mgr.calls == 2
    assert result['pdd']['orders'] == [{'order_id': 'O1'}]
    assert '_dump_state_errors' not in result


@pytest.mark.asyncio
async def test_collect_state_records_dump_state_failure_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    result: dict = {}
    mgr = _FailingMCPManager()
    ctx = SimpleNamespace(_mcp_manager=mgr)
    monkeypatch.setenv('AIDATA_DUMP_STATE_MAX_ATTEMPTS', '2')
    monkeypatch.setenv('AIDATA_DUMP_STATE_RETRY_BACKOFF_S', '0')

    await _collect_state_via_mcp(ctx, result)

    assert mgr.calls == 2
    assert result['_dump_state_errors']['pdd']['tool'] == 'pdd_pdd_dump_state'
    assert result['_dump_state_errors']['pdd']['attempts'] == 2
    assert result['_dump_state_errors']['pdd']['kind'] == 'payload_too_large'
    assert result['_dump_state_errors']['pdd']['error_type'] == 'RuntimeError'
    assert 'payload too large' in result['_dump_state_errors']['pdd']['error']


def test_dump_state_error_info_classifies_timeout_network_and_payload() -> None:
    timeout_info = _dump_state_error_info(TimeoutError('operation timed out'))
    assert timeout_info['kind'] == 'timeout'
    assert 'timed out' in timeout_info['error']

    payload_info = _dump_state_error_info(RuntimeError('413 request entity too large'))
    assert payload_info['kind'] == 'payload_too_large'

    network_info = _dump_state_error_info(ConnectionError('connection reset by peer'))
    assert network_info['kind'] == 'network'
