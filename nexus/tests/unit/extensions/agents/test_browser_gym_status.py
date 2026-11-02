import json

from types import SimpleNamespace

import pytest

from nexus.extensions.agents.browser_gym import agent as browser_gym


class _Message:
    def __init__(self, *, content: str | None, tool_name: str | None, arguments: str | None = None) -> None:
        self.content = content
        self.tool_calls = (
            [
                SimpleNamespace(
                    id='call-1',
                    function=SimpleNamespace(
                        name=tool_name,
                        arguments=(
                            arguments
                            if arguments is not None
                            else json.dumps({'message': 'submitted answer'} if tool_name == 'submit_result' else {})
                        ),
                    ),
                )
            ]
            if tool_name
            else []
        )

    def model_dump(self, **kwargs):
        return {
            'role': 'assistant',
            'content': self.content,
            **({'tool_calls': self.tool_calls} if self.tool_calls else {}),
        }


class _Client:
    def __init__(self, message: _Message) -> None:
        self.message = message

    async def create_completion(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])


class _Executor:
    def __init__(self, tools) -> None:
        pass

    def get_tool_schemas(self):
        return []

    async def call(self, tool_name: str, tool_args: dict):
        return SimpleNamespace(output=[])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('tool_name', 'expected_status'),
    [
        ('submit_result', 'submitted'),
        (None, 'no_tool_calls_incomplete'),
        ('click', 'max_iterations'),
    ],
)
async def test_browser_gym_reports_why_the_loop_stopped(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str | None,
    expected_status: str,
) -> None:
    context = SimpleNamespace(
        openai_client=_Client(_Message(content=None, tool_name=tool_name)),
        tool_state={'browser_session_id': 'browser-1'},
    )

    async def fake_get_observation(*args, **kwargs):
        return 'page observation', None

    monkeypatch.setattr(browser_gym, 'require_context', lambda: context)
    monkeypatch.setattr(browser_gym, 'get_tools_for_subset', lambda subset: [])
    monkeypatch.setattr(browser_gym, 'ToolExecutor', _Executor)
    monkeypatch.setattr(browser_gym, 'get_observation', fake_get_observation)

    result = await browser_gym.run_browser_gym_agent(
        browser_gym.BrowserGymAgentInput(
            user_prompt='complete the browser task',
            max_iterations=1,
        )
    )

    assert result.exit_status == expected_status


@pytest.mark.asyncio
async def test_browser_gym_reports_invalid_tool_arguments_as_format_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        openai_client=_Client(_Message(content=None, tool_name='click', arguments='{"bid":')),
        tool_state={'browser_session_id': 'browser-1'},
    )

    async def fake_get_observation(*args, **kwargs):
        return 'page observation', None

    monkeypatch.setattr(browser_gym, 'require_context', lambda: context)
    monkeypatch.setattr(browser_gym, 'get_tools_for_subset', lambda subset: [])
    monkeypatch.setattr(browser_gym, 'ToolExecutor', _Executor)
    monkeypatch.setattr(browser_gym, 'get_observation', fake_get_observation)

    result = await browser_gym.run_browser_gym_agent(
        browser_gym.BrowserGymAgentInput(
            user_prompt='complete the browser task',
            max_iterations=1,
        )
    )

    assert result.exit_status == 'format_error'
