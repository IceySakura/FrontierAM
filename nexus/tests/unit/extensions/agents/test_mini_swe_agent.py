from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from nexus.extensions.agents.mini_swe_agent import agent as mini_swe


class _Span:
    def update(self, **kwargs):
        del kwargs

    def finish(self):
        return None


@contextmanager
def _span_context(*args, **kwargs):
    del args, kwargs
    yield _Span()


class _Message:
    def __init__(self, tool_calls):
        self.content = None
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        del exclude_none
        return {
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': call.id,
                    'type': 'function',
                    'function': {
                        'name': call.function.name,
                        'arguments': call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ],
        }


class _Client:
    def __init__(self):
        self.calls = []

    async def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        tool_call = SimpleNamespace(
            id='call-1',
            function=SimpleNamespace(
                name='bash',
                arguments='{"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo patch"}',
            ),
        )
        message = _Message([tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='tool_calls')])


class _Executor:
    def __init__(self, tools):
        del tools

    async def call(self, name, arguments, timeout):
        assert name == 'bash'
        assert arguments['command'].startswith('echo COMPLETE_TASK')
        assert timeout > 0
        return SimpleNamespace(
            output=[{'text': 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch'}],
            meta={'returncode': 0},
        )


@pytest.mark.asyncio
async def test_mini_swe_agent_emits_gateway_request_meta(monkeypatch):
    client = _Client()
    context = SimpleNamespace(
        openai_client=client,
        task_config={'working_dir': '/testbed'},
        tool_state={},
    )
    monkeypatch.setattr(mini_swe, 'require_context', lambda: context)
    monkeypatch.setattr(mini_swe, 'ToolExecutor', _Executor)
    monkeypatch.setattr(mini_swe, 'start_span', lambda *args, **kwargs: _Span())
    monkeypatch.setattr(mini_swe, 'span_context', _span_context)
    monkeypatch.setattr(mini_swe, 'get_current_span_id', lambda: None)

    output = await mini_swe.run_mini_swe_agent(
        mini_swe.MiniSweAgentInput(
            user_prompt='fix the bug',
            working_dir='/testbed',
            max_iterations=1,
            extra_completion_kwargs={'request_meta': {'source': 'test'}},
        )
    )

    assert client.calls[0]['request_meta'] == {'source': 'test', 'iteration': 1, 'retry': 0}
    assert output.exit_status == mini_swe.EXIT_STATUS_SUBMITTED
    assert output.submission == 'patch'
    assert not hasattr(output, 'main_trajectory')
