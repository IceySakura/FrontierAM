import copy

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.single_turn.agent import SingleTurnAgentInput, single_turn_agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('system_prompt', 'inline_system'),
    [(None, True), ('Agent instructions', False), ('Agent instructions', True), ('', True)],
)
async def test_optional_system_prompt_preserves_message_history_and_trajectory(
    monkeypatch, system_prompt, inline_system
):
    prompt = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'First question'}]},
        {'role': 'assistant', 'content': 'First answer'},
        {'role': 'user', 'content': 'Follow-up question'},
    ]
    if inline_system:
        prompt.insert(0, {'role': 'system', 'content': 'Dataset instructions'})
    original = copy.deepcopy(prompt)
    tools = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}]
    assistant_message = {'role': 'assistant', 'content': 'Answer', 'reasoning_content': 'Reasoning'}
    seen = {}

    async def complete(**kwargs):
        seen.update(copy.deepcopy(kwargs))
        message = SimpleNamespace(
            content='Answer',
            reasoning_content='Reasoning',
            model_dump=lambda **_: dict(assistant_message),
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(create_completion=AsyncMock(side_effect=complete))
    monkeypatch.setattr(
        'nexus.extensions.agents.single_turn.agent.require_context', lambda: SimpleNamespace(openai_client=client)
    )
    config = {'prompt': prompt, 'tools': tools}
    if system_prompt is not None:
        config['system_prompt'] = system_prompt
    agent_input = SingleTurnAgentInput(**config)
    result = await single_turn_agent(agent_input)

    expected = copy.deepcopy(original)
    if system_prompt is not None:
        expected.insert(0, {'role': 'system', 'content': system_prompt})
    assert seen == {'messages': expected, 'tools': tools}
    assert result.main_trajectory == [*expected, assistant_message]
    assert result.response == 'Answer'
    assert result.reasoning_content == 'Reasoning'
    assert agent_input.prompt == original
    assert prompt == original
    client.create_completion.assert_awaited_once()
