from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_calculator_agent_stops_and_returns_history_when_context_length_exceeded(monkeypatch):
    from nexus.core import ContextLengthExceededError
    from nexus.extensions.projects.calculator import agent as calculator_module

    client = SimpleNamespace(create_completion=AsyncMock(side_effect=ContextLengthExceededError('too long')))
    monkeypatch.setattr(calculator_module, 'require_context', lambda: SimpleNamespace(openai_client=client))

    output = await calculator_module.run_calculator_agent(calculator_module.CalculatorAgentInput(user_prompt='q'))

    assert [message['role'] for message in output.main_trajectory] == ['system', 'user']


@pytest.mark.asyncio
async def test_calculator_agent_stops_without_running_tools_when_response_hits_max_tokens(monkeypatch):
    from nexus.extensions.projects.calculator import agent as calculator_module

    tool_call = SimpleNamespace(id='call-0', function=SimpleNamespace(name='calculator/calculate', arguments='{}'))
    message = SimpleNamespace(
        content='cut',
        tool_calls=[tool_call],
        model_dump=lambda exclude_none: {'role': 'assistant', 'content': 'cut'},
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='length')])
    client = SimpleNamespace(create_completion=AsyncMock(return_value=response))
    monkeypatch.setattr(calculator_module, 'require_context', lambda: SimpleNamespace(openai_client=client))

    output = await calculator_module.run_calculator_agent(calculator_module.CalculatorAgentInput(user_prompt='q'))

    assert client.create_completion.await_count == 1
    assert [message['role'] for message in output.main_trajectory] == ['system', 'user', 'assistant']
