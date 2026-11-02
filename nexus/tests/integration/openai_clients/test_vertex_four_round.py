from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from openai.types.chat import ChatCompletion

from nexus.core import OpenAIConfig
from nexus.core.context import ExecutionContext
from nexus.extensions.agents.main.agent import MainAgentInput, run_main_agent
from nexus.extensions.openai_clients.vertex import VertexClient
from nexus.trace import processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from tests.integration.openai_clients.test_openai_responses_four_round import (
    _PATCH_TOOL,
    _SHELL_TOOL,
    _STEP_TOOL,
)
from tests.provider_call_capture import ProviderCallCaptureProcessor


def _anthropic_response(index: int, *, final: bool = False) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {'type': 'thinking', 'thinking': f'anthropic-think-{index}', 'signature': f'anthropic-sig-{index}'}
    ]
    if final:
        content.append({'type': 'text', 'text': 'VERTEX_ANTHROPIC_FOUR_ROUND_OK'})
        stop_reason = 'end_turn'
    else:
        tools = [
            ('integration_step', {'step': 1}),
            ('shell', {'commands': ['echo step-2'], 'timeout_ms': 1000}),
            (
                'apply_patch',
                {'type': 'update_file', 'path': 'state.txt', 'diff': '@@ -1 +1 @@\n-old\n+done'},
            ),
        ]
        name, tool_input = tools[index - 1]
        content.append({'type': 'tool_use', 'id': f'toolu-{index}', 'name': name, 'input': tool_input})
        stop_reason = 'tool_use'
    return {
        'id': f'msg-vertex-{index}',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-opus-4-8',
        'content': content,
        'stop_reason': stop_reason,
        'stop_sequence': None,
        'usage': {'input_tokens': 10 * index, 'output_tokens': 5 * index},
    }


class _ScriptedAnthropicPost:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def __call__(
        self,
        endpoint: str,
        payload: dict[str, Any],
        provider_call: dict[str, Any],
    ) -> dict[str, Any]:
        del endpoint, provider_call
        self.payloads.append(deepcopy(payload))
        index = len(self.payloads)
        return _anthropic_response(index, final=index == 4)


def _openai_response(index: int, *, final: bool = False) -> ChatCompletion:
    message: dict[str, Any] = {
        'role': 'assistant',
        'content': 'VERTEX_OPENAI_FOUR_ROUND_OK' if final else None,
        'reasoning_content': f'openai-think-{index}',
    }
    if not final:
        tools = [
            ('integration_step', '{"step":1}'),
            ('shell', '{"commands":["echo step-2"],"timeout_ms":1000}'),
            ('apply_patch', '{"type":"update_file","path":"state.txt","diff":"patch"}'),
        ]
        name, arguments = tools[index - 1]
        message['tool_calls'] = [
            {
                'id': f'call-{index}',
                'type': 'function',
                'function': {'name': name, 'arguments': arguments},
            }
        ]
    return ChatCompletion.model_validate({
        'id': f'chatcmpl-vertex-{index}',
        'object': 'chat.completion',
        'created': index,
        'model': 'gpt-vertex',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop' if final else 'tool_calls',
                'message': message,
            }
        ],
        'usage': {'prompt_tokens': 10 * index, 'completion_tokens': 5 * index, 'total_tokens': 15 * index},
    })


class _ScriptedOpenAICreate:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def __call__(self, **payload: Any) -> ChatCompletion:
        self.payloads.append(deepcopy(payload))
        index = len(self.payloads)
        return _openai_response(index, final=index == 4)


def _agent_input() -> MainAgentInput:
    return MainAgentInput(
        system_prompt='Complete exactly three tools before the final answer.',
        user_prompt='Run the unified Vertex four-round integration.',
        tools=[_STEP_TOOL, _SHELL_TOOL, _PATCH_TOOL],
        max_iterations=4,
        single_tool_timeout=10,
        inference_time_timeout=30,
        observation_time_timeout=30,
    )


@pytest.mark.integration
async def test_vertex_anthropic_preserves_signed_thinking_for_four_rounds() -> None:
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'vertex-anthropic-secret',
                'base_url': 'https://vertex.invalid',
                'capture_raw_io': True,
                'max_retries': 1,
            },
            request_args={
                'model': 'claude-opus-4-8',
                'max_tokens': 256,
                'thinking': {'type': 'adaptive'},
                'reasoning_effort': 'high',
            },
        )
    )
    scripted_post = _ScriptedAnthropicPost()
    client._adapter._post_json = scripted_post  # type: ignore[method-assign]
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    context = ExecutionContext(openai_clients={'main': client}, agent_name='main_agent')

    with processor_context([tracker, provider_capture]):
        async with context:
            output = await run_main_agent(_agent_input())

    assert output.exit_status == 'finished'
    assert output.iterations == 4
    assert output.final_answer == 'VERTEX_ANTHROPIC_FOUR_ROUND_OK'
    assert len(scripted_post.payloads) == 4
    for request_index, payload in enumerate(scripted_post.payloads[1:], start=1):
        assistant_messages = [message for message in payload['messages'] if message['role'] == 'assistant']
        thinking = assistant_messages[-1]['content'][0]
        assert thinking == {
            'type': 'thinking',
            'thinking': f'anthropic-think-{request_index}',
            'signature': f'anthropic-sig-{request_index}',
        }
        assert payload['thinking'] == {'type': 'adaptive'}
        assert payload['output_config'] == {'effort': 'high'}

    assert 'calls' not in tracker.get_result()
    calls = provider_capture.get_result()
    assert len(calls) == 4
    assert [call['provider'] for call in calls] == ['vertex'] * 4
    assert [call['protocol'] for call in calls] == ['anthropic'] * 4
    assert all('input' not in call and 'output' not in call for call in calls)
    assert all(call['provider_io']['request'] and call['provider_io']['response'] for call in calls)
    assert 'vertex-anthropic-secret' not in str(calls)


@pytest.mark.integration
async def test_vertex_openai_preserves_reasoning_history_for_four_rounds() -> None:
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'vertex-openai-secret',
                'base_url': 'https://vertex.invalid/v1',
                'capture_raw_io': True,
                'max_retries': 1,
            },
            request_args={'model': 'gpt-vertex', 'max_completion_tokens': 256, 'reasoning_effort': 'high'},
        )
    )
    scripted_create = _ScriptedOpenAICreate()
    client._adapter.client = SimpleNamespace(  # type: ignore[union-attr]
        chat=SimpleNamespace(completions=SimpleNamespace(create=scripted_create))
    )
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    context = ExecutionContext(openai_clients={'main': client}, agent_name='main_agent')

    with processor_context([tracker, provider_capture]):
        async with context:
            output = await run_main_agent(_agent_input())

    assert output.exit_status == 'finished'
    assert output.iterations == 4
    assert output.final_answer == 'VERTEX_OPENAI_FOUR_ROUND_OK'
    assert len(scripted_create.payloads) == 4
    for request_index, payload in enumerate(scripted_create.payloads[1:], start=1):
        assistant_messages = [message for message in payload['messages'] if message['role'] == 'assistant']
        assert assistant_messages[-1]['reasoning_content'] == f'openai-think-{request_index}'

    assert 'calls' not in tracker.get_result()
    calls = provider_capture.get_result()
    assert len(calls) == 4
    assert [call['provider'] for call in calls] == ['vertex'] * 4
    assert [call['protocol'] for call in calls] == ['openai'] * 4
    assert all('input' not in call and 'output' not in call for call in calls)
    assert all(call['provider_io']['request'] and call['provider_io']['response'] for call in calls)
    assert 'vertex-openai-secret' not in str(calls)
