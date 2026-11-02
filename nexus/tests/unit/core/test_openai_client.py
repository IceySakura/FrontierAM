from __future__ import annotations

from typing import Any

import pytest

from openai.types.chat import ChatCompletionMessageParam
from tests.provider_call_capture import ProviderCallCaptureProcessor

from nexus.core.openai_client import (
    OpenAICompatibleClient,
    OpenAIConfig,
    apply_enforced_request_args,
    resolve_effective_request_args,
)
from nexus.core.types import ExtendedChatCompletion, ExtendedChoice, ExtendedMessage
from nexus.trace import Span, SpanEndEvent, SpanProcessor, processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor


class CapturingProcessor(SpanProcessor):
    def __init__(self) -> None:
        self.ended: list[Span] = []

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        del event
        self.ended.append(span)


class RecordingClient(OpenAICompatibleClient):
    async def _create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        self._record_provider_call({'provider': 'test'})
        dynamic_secret = str(kwargs['extra_headers']['X-Session-Credential'])
        return ExtendedChatCompletion(
            id='completion_1',
            created=1,
            model=model,
            choices=[
                ExtendedChoice(
                    finish_reason='stop',
                    index=0,
                    message=ExtendedMessage(
                        content=f'provider echoed {dynamic_secret}',
                        provider_specific_fields={'screenshot': kwargs['screenshot']},
                    ),
                )
            ],
        )


class FailingClient(OpenAICompatibleClient):
    async def _create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        del messages, model, kwargs
        raise RuntimeError('upstream echoed internal-secret')


class RefusingClient(OpenAICompatibleClient):
    async def _create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        del messages, kwargs
        return ExtendedChatCompletion(
            id='completion_refusal',
            created=1,
            model=model,
            choices=[
                ExtendedChoice(
                    finish_reason='stop',
                    index=0,
                    message=ExtendedMessage(refusal='Request refused.'),
                )
            ],
        )


class TimedClient(OpenAICompatibleClient):
    async def _create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        del messages, kwargs
        return ExtendedChatCompletion(
            id='completion_timed',
            created=1,
            model=model,
            choices=[
                ExtendedChoice(
                    finish_reason='stop',
                    index=0,
                    message=ExtendedMessage(content='done'),
                )
            ],
            provider_specific_fields={'inference_time': 1.25},
        )


class SensitiveProviderCallClient(OpenAICompatibleClient):
    async def _create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        del messages, kwargs
        self._record_provider_call(
            {
                'provider': 'test',
                'endpoint': (
                    'https://user:password@vertex.test/v1/chat/completions?api_key=configured-secret&region=ap-guangzhou'
                ),
                'attempts': [
                    {
                        'attempt': 1,
                        'status': 'error',
                        'message': 'configured-secret rejected; Authorization: Bearer header-secret',
                    }
                ],
                'request': {
                    'headers': {'Authorization': 'Bearer header-secret'},
                    'messages': [{'role': 'user', 'content': 'keep this diagnostic text'}],
                },
            }
        )
        return ExtendedChatCompletion(
            id='completion_sensitive_provider_call',
            created=1,
            model=model,
            choices=[
                ExtendedChoice(
                    finish_reason='stop',
                    index=0,
                    message=ExtendedMessage(content='done'),
                )
            ],
        )


def test_apply_enforced_request_args_is_final_without_mutating_inputs() -> None:
    request_args = {
        'model': 'call-model',
        'reasoning': {'effort': 'low'},
        'extra_body': {'model': 'extra-model', 'reasoning': {'effort': 'medium'}, 'vendor': 'base'},
    }
    enforced = {
        'model': 'forced-model',
        'reasoning': {'effort': 'high'},
        'temperature': None,
        'extra_body': {'vendor': 'forced'},
    }

    result = apply_enforced_request_args(request_args, enforced)

    assert result == {
        'model': 'forced-model',
        'reasoning': {'effort': 'high'},
        'temperature': None,
        'extra_body': {'vendor': 'forced'},
    }
    assert request_args['extra_body']['model'] == 'extra-model'
    assert enforced['extra_body'] == {'vendor': 'forced'}


def test_empty_enforced_request_args_preserves_existing_request_shape() -> None:
    request_args = {'model': 'base', 'extra_body': None}

    result = apply_enforced_request_args(request_args, {})

    assert result == request_args
    assert result is not request_args


def test_resolve_effective_request_args_matches_sdk_extra_body_order() -> None:
    effective = resolve_effective_request_args(
        {'model': 'base', 'extra_body': {'model': 'extra', 'vendor': True}},
        {'model': 'forced'},
    )

    assert effective == {'model': 'forced', 'vendor': True}


@pytest.mark.asyncio
async def test_record_provider_call_updates_openai_completion_span() -> None:
    processor = CapturingProcessor()
    default_tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    client = RecordingClient(OpenAIConfig(client_args={}, request_args={}))

    with processor_context([processor, default_tracker, provider_capture]):
        await client.create_completion(
            [{'role': 'user', 'content': 'hello'}],
            model='m',
            extra_headers={'X-Session-Credential': 'value'},
            screenshot='data:image/png;base64,c2NyZWVuc2hvdA==',
        )

    assert processor.ended[-1].attributes['provider_call']['provider'] == 'test'
    assert 'calls' not in default_tracker.get_result()
    provider_io = provider_capture.get_result()[0]['provider_io']
    assert provider_io == {'provider': 'test', 'attempts': []}
    assert 'request' not in provider_io
    assert 'response' not in provider_io


@pytest.mark.asyncio
async def test_create_completion_credits_runner_timeout(monkeypatch) -> None:
    calls: list[tuple[ExtendedChatCompletion, float, float]] = []

    def _record(response: ExtendedChatCompletion, wall_time: float, *, started_at: float) -> None:
        calls.append((response, wall_time, started_at))

    monkeypatch.setattr('nexus.core.openai_client.credit_runner_timeout_for_response', _record)
    client = TimedClient(OpenAIConfig(client_args={}, request_args={}))

    response = await client.create_completion([{'role': 'user', 'content': 'hello'}], model='m')

    assert calls
    assert calls[0][0] is response
    assert calls[0][1] >= 0.0
    assert calls[0][2] > 0.0, 'Response compensation must receive the completion start time'


@pytest.mark.asyncio
async def test_record_provider_call_redacts_config_headers_endpoint_and_error_text() -> None:
    processor = CapturingProcessor()
    client = SensitiveProviderCallClient(OpenAIConfig(client_args={'api_key': 'configured-secret'}, request_args={}))

    with processor_context([processor]):
        await client.create_completion([{'role': 'user', 'content': 'hello'}], model='m')

    provider_call = processor.ended[-1].attributes['provider_call']
    serialized = str(provider_call)
    assert 'configured-secret' not in serialized
    assert 'header-secret' not in serialized
    assert 'user:password' not in serialized
    assert provider_call['endpoint'] == (
        'https://vertex.test/v1/chat/completions?api_key=<redacted>&region=ap-guangzhou'
    )
    assert provider_call['request']['headers']['Authorization'] == '<redacted>'
    assert provider_call['request']['messages'][0]['content'] == 'keep this diagnostic text'
    assert provider_call['attempts'][0]['message'] == ('<redacted> rejected; Authorization: Bearer <redacted>')


@pytest.mark.asyncio
async def test_openai_completion_trace_preserves_inputs_outputs_and_credentials() -> None:
    processor = CapturingProcessor()
    client = RecordingClient(OpenAIConfig(client_args={'api_key': 'configured-secret'}, request_args={}))
    screenshot = 'data:image/png;base64,c2NyZWVuc2hvdA=='
    message = 'inspect https://user:password@files.test/item?token=url-secret'

    with processor_context([processor]):
        completion = await client.create_completion(
            [{'role': 'user', 'content': message}],
            model='model-name',
            tools=[{'type': 'function', 'function': {'name': 'inspect', 'parameters': {'type': 'object'}}}],
            extra_headers={'X-Session-Credential': 'dynamic-secret'},
            screenshot=screenshot,
        )

    assert completion.choices[0].message.content == 'provider echoed dynamic-secret'
    span = processor.ended[-1]
    assert span.attributes['inputs']['messages'][0]['content'] == message
    assert span.attributes['inputs']['kwargs']['extra_headers']['X-Session-Credential'] == 'dynamic-secret'
    assert span.attributes['inputs']['kwargs']['tools'][0]['function']['name'] == 'inspect'
    assert span.attributes['inputs']['kwargs']['screenshot'] == screenshot
    assert span.attributes['outputs']['choices'][0]['message']['content'] == 'provider echoed dynamic-secret'
    assert span.attributes['outputs']['choices'][0]['message']['provider_specific_fields']['screenshot'] == screenshot


@pytest.mark.asyncio
async def test_openai_completion_trace_preserves_raw_error_message() -> None:
    processor = CapturingProcessor()
    client = FailingClient(OpenAIConfig(client_args={}, request_args={}))

    with processor_context([processor]), pytest.raises(RuntimeError, match='upstream echoed internal-secret'):
        await client.create_completion([{'role': 'user', 'content': 'hello'}], model='m')

    assert processor.ended[-1].status.message == 'upstream echoed internal-secret'


@pytest.mark.asyncio
async def test_openai_completion_trace_preserves_structured_refusal() -> None:
    processor = CapturingProcessor()
    client = RefusingClient(OpenAIConfig(client_args={}, request_args={}))

    with processor_context([processor]):
        completion = await client.create_completion([{'role': 'user', 'content': 'hello'}], model='m')

    assert completion.choices[0].message.refusal == 'Request refused.'
    span = processor.ended[-1]
    assert span.attributes['outputs']['choices'][0]['message']['refusal'] == 'Request refused.'
    assert span.status.code == 'OK'
