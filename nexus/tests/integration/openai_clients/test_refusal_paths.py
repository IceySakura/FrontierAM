from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openai.types.chat import ChatCompletion

from nexus.core import OpenAICompatibleClient, OpenAIConfig
from nexus.core.context import ExecutionContext
from nexus.extensions.agents.main.agent import MainAgentInput, MainExitStatus, run_main_agent
from nexus.extensions.openai_clients import litellm as litellm_module
from nexus.extensions.openai_clients.litellm import LiteLLMClient
from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient
from nexus.extensions.openai_clients.vertex import VertexClient
from nexus.trace import processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from tests.provider_call_capture import ProviderCallCaptureProcessor


_REFUSAL = 'INTEGRATION_REFUSAL'


def _openai_refusal() -> ChatCompletion:
    return ChatCompletion.model_validate({
        'id': 'chatcmpl-refusal-integration',
        'object': 'chat.completion',
        'created': 1,
        'model': 'refusal-model',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': None, 'refusal': _REFUSAL},
            }
        ],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    })


def _responses_refusal() -> dict[str, Any]:
    return {
        'id': 'resp-refusal-integration',
        'object': 'response',
        'created_at': 1,
        'model': 'refusal-model',
        'status': 'completed',
        'output': [
            {
                'type': 'message',
                'id': 'msg-refusal-integration',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'refusal', 'refusal': _REFUSAL}],
            }
        ],
        'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2},
    }


def _anthropic_refusal() -> dict[str, Any]:
    return {
        'id': 'msg-refusal-integration',
        'type': 'message',
        'role': 'assistant',
        'model': 'refusal-model',
        'content': [],
        'stop_reason': 'refusal',
        'stop_sequence': None,
        'stop_details': {'type': 'refusal', 'explanation': _REFUSAL},
        'usage': {'input_tokens': 1, 'output_tokens': 1},
    }


async def _client_for_case(case: str, monkeypatch: pytest.MonkeyPatch) -> tuple[OpenAICompatibleClient, str, str]:
    if case == 'openai_responses':
        client = OpenAIResponsesClient(
            OpenAIConfig(
                client_args={
                    'api_key': 'responses-refusal-secret',
                    'base_url': 'https://responses.invalid/v1',
                    'capture_raw_io': True,
                    'max_retries': 1,
                },
                request_args={'model': 'refusal-model'},
            )
        )

        async def create_response(**request: Any) -> dict[str, Any]:
            del request
            return _responses_refusal()

        client.client = SimpleNamespace(responses=SimpleNamespace(create=create_response))
        return client, 'openai', 'responses'

    if case == 'vertex_anthropic':
        client = VertexClient(
            OpenAIConfig(
                client_args={
                    'protocol': 'anthropic',
                    'api_key': 'vertex-anthropic-refusal-secret',
                    'base_url': 'https://vertex.invalid',
                    'capture_raw_io': True,
                    'max_retries': 1,
                },
                request_args={'model': 'refusal-model', 'max_tokens': 32},
            )
        )

        async def post_json(endpoint: str, payload: dict[str, Any], provider_call: dict[str, Any]) -> dict[str, Any]:
            del endpoint, payload
            provider_call['attempts'].append({'attempt': 1, 'status': 'success'})
            return _anthropic_refusal()

        client._adapter._post_json = post_json  # type: ignore[method-assign]
        return client, 'vertex', 'anthropic'

    if case == 'vertex_openai':
        client = VertexClient(
            OpenAIConfig(
                client_args={
                    'protocol': 'openai',
                    'api_key': 'vertex-openai-refusal-secret',
                    'base_url': 'https://vertex.invalid/v1',
                    'capture_raw_io': True,
                    'max_retries': 1,
                },
                request_args={'model': 'refusal-model'},
            )
        )

        async def create_chat_completion(**request: Any) -> ChatCompletion:
            del request
            return _openai_refusal()

        client._adapter.client = SimpleNamespace(  # type: ignore[union-attr]
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_chat_completion))
        )
        return client, 'vertex', 'openai'

    async def litellm_completion(**request: Any) -> ChatCompletion:
        del request
        return _openai_refusal()

    monkeypatch.setattr(litellm_module.litellm, 'acompletion', litellm_completion)
    protocol = case.removeprefix('litellm_')
    return (
        LiteLLMClient(
            OpenAIConfig(
                client_args={
                    'api_key': 'litellm-refusal-secret',
                    'base_url': 'https://litellm.invalid/v1',
                    'max_retries': 1,
                },
                request_args={'model': 'refusal-model'},
            )
        ),
        'litellm',
        protocol,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    'case',
    ['openai_responses', 'vertex_anthropic', 'vertex_openai', 'litellm_openai', 'litellm_anthropic'],
)
async def test_client_refusal_stops_before_an_empty_assistant_or_later_round(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, provider, protocol = await _client_for_case(case, monkeypatch)
    records_provider_calls = not case.startswith('litellm_')
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()

    with processor_context([tracker, provider_capture]):
        async with ExecutionContext(openai_clients={'main': client}, agent_name='main_agent'):
            output = await run_main_agent(
                MainAgentInput(
                    system_prompt='Stop on a provider refusal.',
                    user_prompt='Trigger the scripted refusal.',
                    tools=[],
                    max_iterations=4,
                    inference_time_timeout=30,
                )
            )

    assert output.exit_status == MainExitStatus.REFUSED.value
    assert output.iterations == 1
    assert output.final_answer == _REFUSAL
    assistant_messages = [message for message in output.main_trajectory if message['role'] == 'assistant']
    assert len(assistant_messages) == 1
    assert assistant_messages[0]['refusal'] == _REFUSAL
    assert 'content' not in assistant_messages[0]
    assert 'tool_calls' not in assistant_messages[0]

    result = tracker.get_result()
    assert result['trajectories']
    assert 'calls' not in result
    calls = provider_capture.get_result()
    if records_provider_calls:
        assert len(calls) == 1
        assert calls[0]['provider'] == provider
        assert calls[0]['protocol'] == protocol
        assert 'input' not in calls[0]
        assert 'output' not in calls[0]
        assert calls[0]['provider_io']['response']
        assert calls[0]['provider_io']['attempts'] == [{'attempt': 1, 'status': 'success'}]
    else:
        assert calls == []
