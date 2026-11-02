import json

from types import SimpleNamespace

import pytest

from nexus.extensions.tasks.one_agent_infer import task


def _handoff_output():
    return SimpleNamespace(
        main_trajectory=[
            {'role': 'system', 'content': 'A3B system'},
            {'role': 'user', 'content': 'complex request'},
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'function': {
                            'name': 'handoff',
                            'arguments': '{"subagent_type":"expert_agent"}',
                        }
                    }
                ],
            },
        ]
    )


def test_handoff_mode_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv('ONE_AGENT_HANDOFF_MODE', raising=False)

    assert task._get_handoff_mode() == 'disabled'


def test_disabled_mode_uses_standard_harness_tools():
    assert task._a3b_tools_for_mode('disabled') is None


@pytest.mark.parametrize('mode', ['offline', 'online'])
def test_handoff_modes_use_router_tools(monkeypatch, mode):
    monkeypatch.setenv('ONE_AGENT_HANDOFF_MODE', mode)

    assert task._get_handoff_mode() == mode
    tools = task._a3b_tools_for_mode(mode)

    assert tools is not None
    assert 'handoff' in tools
    assert 'delegate' not in tools


def test_handoff_mode_rejects_unknown_protocol(monkeypatch):
    monkeypatch.setenv('ONE_AGENT_HANDOFF_MODE', 'sticky-ish')

    with pytest.raises(ValueError, match='ONE_AGENT_HANDOFF_MODE'):
        task._get_handoff_mode()


def test_resume_history_drops_system_and_terminal_handoff():
    assert task._build_resume_history_from_output(_handoff_output()) == [
        {'role': 'user', 'content': 'complex request'}
    ]


@pytest.mark.asyncio
async def test_live_continuation_switches_model_and_tools_then_restores(monkeypatch):
    monkeypatch.setenv('ONE_AGENT_A20B_MODEL', 'openai/a20b')
    request_args = {'model': 'openai/a3b'}
    context = SimpleNamespace(
        openai_client=SimpleNamespace(
            config=SimpleNamespace(request_args=request_args),
        )
    )
    monkeypatch.setattr(task, 'require_context', lambda: context)
    observed = {}
    expected_output = SimpleNamespace(main_trajectory=[])

    async def run_agent(**kwargs):
        observed['model'] = request_args['model']
        observed['kwargs'] = kwargs
        return expected_output

    task_input = SimpleNamespace(run_agent=run_agent, terminal_type='web')
    actual_output = await task._continue_with_live_a20b(
        task_input,
        _handoff_output(),
        'task-1',
    )

    assert actual_output is expected_output
    assert observed['model'] == 'openai/a20b'
    assert observed['kwargs']['history'] == [{'role': 'user', 'content': 'complex request'}]
    assert observed['kwargs']['resume_from_history'] is True
    assert 'delegate' in observed['kwargs']['tools']
    assert 'handoff' not in observed['kwargs']['tools']
    assert request_args['model'] == 'openai/a3b'


@pytest.mark.asyncio
async def test_live_continuation_restores_model_after_failure(monkeypatch):
    monkeypatch.setenv('ONE_AGENT_A20B_MODEL', 'openai/a20b')
    request_args = {'model': 'openai/a3b'}
    context = SimpleNamespace(
        openai_client=SimpleNamespace(
            config=SimpleNamespace(request_args=request_args),
        )
    )
    monkeypatch.setattr(task, 'require_context', lambda: context)

    async def run_agent(**kwargs):
        del kwargs
        raise RuntimeError('A20B failed')

    task_input = SimpleNamespace(run_agent=run_agent, terminal_type='web')
    with pytest.raises(RuntimeError, match='A20B failed'):
        await task._continue_with_live_a20b(
            task_input,
            _handoff_output(),
            'task-1',
        )

    assert request_args['model'] == 'openai/a3b'


def test_persist_routing_sidecar(tmp_path):
    task._persist_routing_sidecar(
        str(tmp_path),
        'task-1',
        'session-1',
        2,
        'a20b',
    )

    routing = json.loads((tmp_path / 'routing' / 'task-1.json').read_text())
    assert routing == {
        'task_id': 'task-1',
        'session_id': 'session-1',
        'turn': 2,
        'routed_to': 'a20b',
    }
