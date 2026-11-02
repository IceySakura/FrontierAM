from __future__ import annotations

import json

from typing import Any

import pytest

from nexus.extensions.agents.codex.macosworld.agent import MacOSWorldCodexInput, run_macosworld_codex


@pytest.mark.asyncio
@pytest.mark.parametrize('exit_code', [0, 1])
async def test_terminal_error_survives_without_retaining_raw_json(monkeypatch, exit_code):
    from unittest.mock import AsyncMock

    event = {'type': 'turn.failed', 'error': {'message': 'Unsupported parameter: reasoning_effort'}}
    monkeypatch.setattr(
        'nexus.extensions.agents.codex.agent.call_native_bridge_runtime_agent',
        AsyncMock(
            return_value={
                'success': exit_code == 0,
                'exit_code': exit_code,
                'output': '',
                'stderr': 'Reading additional input from stdin...\n',
                'json_events_stdout': json.dumps(event),
            }
        ),
    )
    monkeypatch.setattr('nexus.extensions.agents.codex.agent.get_current_span_id', lambda: None)
    monkeypatch.setattr('nexus.extensions.agents.codex.agent._emit_codex_timeline_spans', lambda *a, **kw: None)
    output = await run_macosworld_codex(
        MacOSWorldCodexInput(user_prompt='test', working_dir='/tmp/test', emit_json_trace=True)
    )
    assert output.json_events_stdout is None
    if exit_code:
        assert output.termination_reason == 'CODEX_TURN_FAILED'
        assert output.stderr.startswith('Codex terminal error: Unsupported parameter: reasoning_effort')
    else:
        assert output.termination_reason == 'ENV_DONE'
        assert output.stderr == 'Reading additional input from stdin...\n'


@pytest.mark.asyncio
@pytest.mark.parametrize('protocol', ['responses', 'responses-chat'])
async def test_codex_native_bridge_agent_uses_common_bridge_helper(monkeypatch: pytest.MonkeyPatch, protocol) -> None:
    captured: dict[str, Any] = {}

    async def fake_call(target: Any, **kwargs: Any) -> dict[str, Any]:
        captured['target'] = target
        captured.update(kwargs)
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
        }

    monkeypatch.setattr('nexus.extensions.agents.codex.agent.call_native_bridge_runtime_agent', fake_call)
    monkeypatch.setattr('nexus.extensions.agents.codex.agent.get_current_span_id', lambda: None)

    output = await run_macosworld_codex(
        MacOSWorldCodexInput(
            user_prompt='fix it',
            api_protocol=protocol,
            working_dir='/work',
            model='gpt-5.4',
            binary_url='http://cos-internal.example/codex',
            binary_checksum='abc123',
            extra_env={'CUSTOM': 'value'},
            pre_start_scripts=[{'interpreter': 'bash', 'code': 'echo ready'}],
            mcp_servers=[{'name': 'everything', 'command': 'npx'}],
        )
    )

    assert output.success is True
    assert captured['task'] == 'fix it'
    assert captured['api_protocol'] == protocol
    assert captured['working_dir'] == '/work'
    assert captured['version'] == '0.114.0'
    assert captured['model'] == 'gpt-5.4'
    assert captured['binary_url'] == 'http://cos-internal.example/codex'
    assert captured['binary_checksum'] == 'abc123'
    assert captured['extra_env'] == {'CUSTOM': 'value'}
    assert captured['pre_start_scripts'][0]['code'] == 'echo ready'
    assert captured['mcp_servers'] == [{'name': 'everything', 'command': 'npx'}]
