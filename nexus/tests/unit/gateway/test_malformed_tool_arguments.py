import json

from copy import deepcopy
from types import SimpleNamespace

import pytest

from _fakes import FakeTokenizer, RecordingEndpoint, memory_store
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from nexus.core import OpenAIConfig
from nexus.core.types.message import ExtendedMessage
from nexus.extensions.openai_clients.hytuner import HyTunerClient
from nexus.gateway.client import GatewayOpenAIClient
from nexus.gateway.session_manager import session_manager


def _tool_call(arguments):
    return {'id': 'call-1', 'type': 'function', 'function': {'name': 'bash', 'arguments': arguments}}


@pytest.mark.parametrize('malformed', ['{"command":', '{"command": "ls",}', '', 'not json'])
@pytest.mark.parametrize('mode', ['gateway', 'hytuner-copy', 'hytuner-inplace'])
def test_normalization_preserves_malformed_arguments_and_continues(mode, malformed):
    arguments = [malformed, '{"command": "ls"}', {'command': 'pwd'}, '[1, 2]', 'null']
    messages = [
        {'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'tool_calls': [_tool_call(value) for value in arguments]},
    ]
    original = deepcopy(messages)
    if mode == 'gateway':
        normalized = GatewayOpenAIClient._messages_with_normalized_tool_calls(messages)
    elif mode == 'hytuner-copy':
        normalized = HyTunerClient._messages_with_normalized_tool_calls(messages)
    else:
        HyTunerClient._normalize_tool_call_arguments(messages)
        normalized = messages

    assert [call['function']['arguments'] for call in normalized[1]['tool_calls']] == [
        malformed,
        {'command': 'ls'},
        {'command': 'pwd'},
        [1, 2],
        None,
    ]
    assert normalized[0] == original[0]
    if mode != 'hytuner-inplace':
        assert messages == original


@pytest.mark.asyncio
async def test_gateway_retries_with_malformed_history_without_losing_trajectory():
    calls = []
    malformed = '{"command":'
    corrected = '{"command": "ls"}'

    class BackendClient:
        async def create_completion(self, **kwargs):
            calls.append(deepcopy(kwargs))
            arguments = malformed if len(calls) == 1 else corrected
            message = ExtendedMessage(
                role='assistant',
                content='',
                tool_calls=[ChatCompletionMessageToolCall(**_tool_call(arguments))],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason='tool_calls', logprobs=None)],
                usage=None,
                model='test-model',
            )

    trajectory_id = 'malformed-arguments-retry'
    gateway = GatewayOpenAIClient(
        OpenAIConfig(
            client_args={
                'openai_client': BackendClient(),
                'trajectory_id': trajectory_id,
                'record_store': memory_store(trajectory_id),
                'scaffold_policy': 'generic',
            },
            request_args={'model': 'test-model'},
        )
    )
    try:
        messages = [{'role': 'user', 'content': 'run ls'}]
        first = await gateway.create_completion(messages)
        messages.append(first.choices[0].message.model_dump(exclude_none=True))
        messages.append({'role': 'tool', 'tool_call_id': 'call-1', 'content': 'Invalid JSON; retry.'})
        original = deepcopy(messages)
        second = await gateway.create_completion(messages)
        assert messages == original
        assert calls[1]['messages'][1]['tool_calls'][0]['function']['arguments'] == malformed
        assert calls[1]['messages'][2] == messages[2]
        assert second.choices[0].message.tool_calls[0].function.arguments == corrected
        trajectory = session_manager.end_session(trajectory_id)
        nodes = list(trajectory.iter_nodes())
        assert len([node for node in nodes if node.generated_by == 'llm']) == 2
        assert len(nodes) == 4
    finally:
        if session_manager.get_session(trajectory_id) is not None:
            session_manager.end_session(trajectory_id)


@pytest.mark.asyncio
async def test_gateway_malformed_history_keeps_sampled_prefix_and_token_accounting():
    malformed = '{"command":'

    class Tokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            rendered = []
            for message in messages:
                content = message.get('content') or ''
                for call in message.get('tool_calls') or []:
                    arguments = call['function']['arguments']
                    content += arguments if isinstance(arguments, str) else json.dumps(arguments)
                rendered.append({**message, 'content': content, 'tool_calls': []})
            return super().apply_chat_template(rendered, **kwargs)

    class ToolParser:
        async def extract_tool_calls(self, token_ids, request=None):
            text = tokenizer.decode(token_ids)
            return (text, '', [_tool_call(text)]) if text == malformed else (text, text, [])

    tokenizer = Tokenizer()
    endpoint = RecordingEndpoint(tokenizer, [malformed, 'done'])
    trajectory_id = 'malformed-token-retry'
    gateway = GatewayOpenAIClient(
        OpenAIConfig(
            client_args={
                'tokenizer': tokenizer,
                'server_manager': endpoint,
                'tool_parser': ToolParser(),
                'trajectory_id': trajectory_id,
                'record_store': memory_store(trajectory_id),
                'scaffold_policy': 'generic',
            },
            request_args={'sampling_params': {'max_tokens': 100}},
        )
    )
    try:
        messages = [{'role': 'user', 'content': 'run ls'}]
        first = await gateway.create_completion(messages)
        messages.append(first.choices[0].message.model_dump(exclude_none=True))
        messages.append({'role': 'tool', 'tool_call_id': 'call-1', 'content': 'Invalid JSON; retry.'})
        original = deepcopy(messages)
        counted = await gateway.count_tokens(messages=messages)
        await gateway.update_token_consumption(messages)
        state = gateway._session()._token_budget_state
        accounted = [list(ids) for ids in state._accounted_prompt_ids.values()]
        await gateway.create_completion(messages)
        assert messages == original
        sampled_prefix = endpoint.calls[0]['prompt_ids'] + tokenizer.encode(malformed)
        actual = endpoint.calls[1]['prompt_ids']
        assert actual[: len(sampled_prefix)] == sampled_prefix
        assert counted.input_tokens == len(actual)
        assert accounted == [actual]
    finally:
        await gateway._session().finish_pending_publications()
        session_manager.end_session(trajectory_id)
