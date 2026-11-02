import asyncio
import json
import os
import socket
import time

from typing import Any

import aiohttp
import pytest

from aiohttp import web

from nexus.core import TaskRunInput, TaskRunOutput, require_context, task_registry
from nexus.runner.live.server import create_app


class LiveStreamProbeInput(TaskRunInput):
    user_prompt: str = 'Build a sports-themed website and describe the layout.'


class LiveStreamProbeOutput(TaskRunOutput):
    content: str


@task_registry.register('live_stream_probe_task')
async def live_stream_probe_task(task_input: LiveStreamProbeInput) -> LiveStreamProbeOutput:
    ctx = require_context()
    client = ctx.openai_clients['main']
    response = await client.create_completion(
        messages=[
            {
                'role': 'user',
                'content': task_input.user_prompt,
            }
        ]
    )
    return LiveStreamProbeOutput(content=response.choices[0].message.content or '')


class FakeModelResponse:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self.payload


class SlowAsyncStream:
    def __init__(self, chunks: list[dict[str, Any]], *, delay: float = 0.02) -> None:
        self._chunks = list(chunks)
        self._delay = delay

    def __aiter__(self) -> 'SlowAsyncStream':
        return self

    async def __anext__(self) -> FakeModelResponse:
        if not self._chunks:
            raise StopAsyncIteration
        await asyncio.sleep(self._delay)
        return FakeModelResponse(self._chunks.pop(0))


async def _start_live_server() -> tuple[str, web.AppRunner]:
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    port = _find_free_port()
    site = web.TCPSite(runner, '127.0.0.1', port)
    await site.start()
    return f'http://127.0.0.1:{port}', runner


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


async def _read_sse_events(
    session: aiohttp.ClientSession, url: str, *, until_type: str, timeout: float
) -> list[dict[str, Any]]:
    return await _read_sse_events_until(
        session,
        url,
        stop=lambda events: bool(events) and events[-1].get('type') == until_type,
        timeout=timeout,
    )


async def _read_sse_events_until(
    session: aiohttp.ClientSession,
    url: str,
    *,
    stop: Any,
    timeout: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    events: list[dict[str, Any]] = []
    current: dict[str, str] = {}
    async with session.get(url) as response:
        response.raise_for_status()
        async for raw_line in response.content:
            if time.monotonic() > deadline:
                break
            line = raw_line.decode('utf-8').rstrip('\n')
            if not line or line == '\r':
                if current.get('data'):
                    event = json.loads(current['data'])
                    events.append(event)
                    if stop(events):
                        return events
                current = {}
                continue
            if line.startswith(':'):
                continue
            field, _, value = line.partition(':')
            current[field] = value.lstrip()
    return events


def _trace_event_name(event: dict[str, Any]) -> str:
    if event.get('type') != 'trace.event':
        return ''
    payload = event.get('data', {}).get('event', {})
    trace_event = payload.get('event', {})
    return trace_event.get('name') or ''


def _trace_payload(event: dict[str, Any]) -> dict[str, Any]:
    if event.get('type') != 'trace.event':
        return {}
    return event.get('data', {}).get('event', {})


def _run_output_content(events: list[dict[str, Any]]) -> str:
    output_event = next(event for event in events if event.get('type') == 'run.output')
    return output_event['data']['output']['task_output']['content']


def _openai_completion_end_content(events: list[dict[str, Any]]) -> str | None:
    message = _openai_completion_end_message(events)
    if message:
        return message.get('content')
    return None


def _openai_completion_end_outputs(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        payload = _trace_payload(event)
        trace_event = payload.get('event', {})
        if payload.get('span_name') != 'openai_completion' or trace_event.get('type') != 'END':
            continue
        outputs = trace_event.get('attributes', {}).get('outputs')
        if isinstance(outputs, dict):
            return outputs
    return None


def _openai_completion_end_message(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    outputs = _openai_completion_end_outputs(events)
    choices = (outputs or {}).get('choices') or []
    if choices:
        return choices[0].get('message', {})
    return None


def _chat_completion_completed_attrs(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        if _trace_event_name(event) != 'chat.completion.completed':
            continue
        attrs = _trace_payload(event).get('event', {}).get('attributes')
        if isinstance(attrs, dict):
            return attrs
    return None


def _path_keys(value: Any, prefix: tuple[Any, ...] = ()) -> set[tuple[Any, ...]]:
    if isinstance(value, dict):
        out = {prefix}
        for key, child in value.items():
            out |= _path_keys(child, (*prefix, key))
        return out
    if isinstance(value, list):
        out = {prefix}
        for index, child in enumerate(value):
            out |= _path_keys(child, (*prefix, index))
        return out
    return {prefix}


def _chat_completion_text_deltas(events: list[dict[str, Any]]) -> list[str]:
    deltas: list[str] = []
    for event in events:
        if _trace_event_name(event) != 'chat.completion.delta':
            continue
        attrs = _trace_payload(event).get('event', {}).get('attributes', {})
        delta = attrs.get('delta')
        if isinstance(delta, str):
            deltas.append(delta)
    return deltas


def _chat_completion_reasoning_deltas(events: list[dict[str, Any]]) -> list[str]:
    deltas: list[str] = []
    for event in events:
        if _trace_event_name(event) != 'chat.completion.reasoning.delta':
            continue
        attrs = _trace_payload(event).get('event', {}).get('attributes', {})
        delta = attrs.get('delta')
        if isinstance(delta, str):
            deltas.append(delta)
    return deltas


def _has_run_output(events: list[dict[str, Any]]) -> bool:
    return any(event.get('type') == 'run.output' for event in events)


def _has_stream_trace_tail(events: list[dict[str, Any]]) -> bool:
    names = [_trace_event_name(event) for event in events]
    return (
        _has_run_output(events)
        and names.count('chat.completion.delta') >= 2
        and 'chat.completion.completed' in names
        and _openai_completion_end_content(events) is not None
    )


def _has_non_stream_trace_tail(events: list[dict[str, Any]]) -> bool:
    return _has_run_output(events) and _openai_completion_end_content(events) is not None


def _rich_response_payload(response_id: str) -> dict[str, Any]:
    return {
        'id': response_id,
        'created': 1,
        'model': 'gpt-test',
        'object': 'chat.completion',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {
                    'role': 'assistant',
                    'content': '<section>Sports landing page</section>',
                    'reasoning_content': 'plan layout',
                    'thinking_blocks': [
                        {
                            'type': 'thinking',
                            'thinking': '',
                            'signature': 'claude-live-signature',
                        }
                    ],
                    'provider_specific_fields': {
                        'reasoning': {
                            'encrypted_content': 'gpt-live-signature',
                        },
                        'reasoningContent': {
                            'signature': 'claude-live-signature',
                        },
                    },
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'write_file', 'arguments': '{"path":"/tmp/a"}'},
                        }
                    ],
                },
            }
        ],
        'usage': {
            'prompt_tokens': 9,
            'completion_tokens': 6,
            'total_tokens': 15,
            'completion_tokens_details': {
                'reasoning_tokens': 2,
                'text_tokens': 4,
                'image_tokens': 0,
                'video_tokens': 0,
            },
            'prompt_tokens_details': {
                'audio_tokens': 0,
                'cached_tokens': 0,
                'image_tokens': 0,
                'video_tokens': 0,
            },
        },
    }


@pytest.mark.asyncio
async def test_live_server_stream_mode_emits_chat_completion_events_before_run_output(monkeypatch) -> None:
    async def fake_acompletion(**kwargs: Any) -> SlowAsyncStream:
        assert kwargs['stream'] is True
        return SlowAsyncStream(
            [
                {
                    'id': 'chatcmpl-live-stream',
                    'created': 1,
                    'model': 'gpt-test',
                    'choices': [
                        {
                            'index': 0,
                            'delta': {
                                'role': 'assistant',
                                'reasoning_content': 'plan ',
                                'thinking_blocks': [
                                    {
                                        'type': 'thinking',
                                        'thinking': '',
                                        'signature': 'claude-live-signature',
                                    }
                                ],
                                'provider_specific_fields': {
                                    'reasoning': {
                                        'encrypted_content': 'gpt-live-signature',
                                    },
                                    'reasoningContent': {
                                        'signature': 'claude-live-signature',
                                    },
                                },
                                'content': '<section>Sports',
                            },
                        }
                    ],
                },
                {
                    'id': 'chatcmpl-live-stream',
                    'created': 1,
                    'model': 'gpt-test',
                    'choices': [
                        {
                            'index': 0,
                            'delta': {'reasoning_content': 'layout', 'content': ' landing page</section>'},
                            'finish_reason': 'tool_calls',
                        }
                    ],
                    'usage': {'prompt_tokens': 9, 'completion_tokens': 6, 'total_tokens': 15},
                },
            ]
        )

    def fake_stream_chunk_builder(chunks: list[Any], messages: list[dict[str, Any]]) -> FakeModelResponse:
        assert len(chunks) == 2
        assert messages == [{'role': 'user', 'content': 'Build a sports-themed website.'}]
        return FakeModelResponse(
            {
                'id': 'chatcmpl-live-stream-built',
                'created': 1,
                'model': 'gpt-test',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {
                            'role': 'assistant',
                            'content': '<section>Sports landing page</section>',
                            'reasoning_content': 'plan layout',
                            'thinking_blocks': [
                                {
                                    'type': 'thinking',
                                    'thinking': '',
                                    'signature': 'claude-live-signature',
                                }
                            ],
                            'provider_specific_fields': {
                                'reasoning': {
                                    'encrypted_content': 'gpt-live-signature',
                                },
                                'reasoningContent': {
                                    'signature': 'claude-live-signature',
                                },
                            },
                        },
                    }
                ],
                'usage': {'prompt_tokens': 9, 'completion_tokens': 6, 'total_tokens': 15},
            }
        )

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm.litellm.stream_chunk_builder',
        fake_stream_chunk_builder,
    )
    base_url, runner = await _start_live_server()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{base_url}/v1/runs',
                json={
                    'run_id': 'run_live_stream_mock',
                    'task_name': 'live_stream_probe_task',
                    'task_config': {'user_prompt': 'Build a sports-themed website.'},
                    'openai_clients': {
                        'main': {
                            'name': 'litellm',
                            'config': {
                                'client_args': {
                                    'api_key': 'test',
                                    'stream': True,
                                    'max_retries': 1,
                                },
                                'request_args': {'model': 'gpt-test'},
                            },
                        }
                    },
                    'runner': {'timeout': 10},
                },
            ) as response:
                assert response.status == 201

            events = await _read_sse_events_until(
                session,
                f'{base_url}/v1/runs/run_live_stream_mock/events?cursor=0',
                stop=_has_stream_trace_tail,
                timeout=10,
            )

        names = [_trace_event_name(event) for event in events]
        assert 'chat.completion.created' in names
        assert names.count('chat.completion.reasoning.delta') == 2
        assert names.count('chat.completion.delta') == 3
        assert len(_chat_completion_text_deltas(events)) == 2
        assert 'chat.completion.completed' in names
        first_delta_index = names.index('chat.completion.delta')
        run_output_index = next(index for index, event in enumerate(events) if event.get('type') == 'run.output')
        assert first_delta_index < run_output_index
        assert _run_output_content(events) == '<section>Sports landing page</section>'
        assert ''.join(_chat_completion_reasoning_deltas(events)) == 'plan layout'
        assert ''.join(_chat_completion_text_deltas(events)) == '<section>Sports landing page</section>'
        assert _openai_completion_end_content(events) == '<section>Sports landing page</section>'
        outputs = _openai_completion_end_outputs(events)
        assert outputs is not None
        assert outputs['choices'][0]['finish_reason'] == 'tool_calls'
        completed_attrs = _chat_completion_completed_attrs(events)
        assert completed_attrs is not None
        assert completed_attrs['finish_reason'] == 'tool_calls'
        message = _openai_completion_end_message(events)
        assert message is not None
        assert message['thinking_blocks'][0]['signature'] == 'claude-live-signature'
        assert message['provider_specific_fields']['reasoning']['encrypted_content'] == 'gpt-live-signature'
        assert message['provider_specific_fields']['reasoningContent']['signature'] == 'claude-live-signature'
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_live_server_stream_and_non_stream_trace_outputs_have_same_path_keys(monkeypatch) -> None:
    final_payload = _rich_response_payload('chatcmpl-live-shape')

    async def fake_acompletion(**kwargs: Any) -> SlowAsyncStream | FakeModelResponse:
        if kwargs.get('stream'):
            return SlowAsyncStream(
                [
                    {
                        'id': 'chatcmpl-live-stream',
                        'created': 1,
                        'model': 'gpt-test',
                        'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': '<section>Sports'}}],
                    },
                    {
                        'id': 'chatcmpl-live-stream',
                        'created': 1,
                        'model': 'gpt-test',
                        'choices': [
                            {
                                'index': 0,
                                'delta': {'reasoning_content': 'plan layout', 'content': ' landing page</section>'},
                                'finish_reason': 'stop',
                            }
                        ],
                        'usage': final_payload['usage'],
                    },
                ]
            )
        assert 'stream' not in kwargs
        return FakeModelResponse(final_payload)

    def fake_stream_chunk_builder(chunks: list[Any], messages: list[dict[str, Any]]) -> FakeModelResponse:
        assert len(chunks) == 2
        return FakeModelResponse(final_payload)

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.litellm.litellm.stream_chunk_builder',
        fake_stream_chunk_builder,
    )
    base_url, runner = await _start_live_server()
    try:
        async with aiohttp.ClientSession() as session:
            for run_id, stream in [('run_live_shape_stream', True), ('run_live_shape_non_stream', False)]:
                client_args: dict[str, Any] = {'api_key': 'test', 'max_retries': 1}
                if stream:
                    client_args['stream'] = True
                async with session.post(
                    f'{base_url}/v1/runs',
                    json={
                        'run_id': run_id,
                        'task_name': 'live_stream_probe_task',
                        'task_config': {'user_prompt': 'Build a sports-themed website.'},
                        'openai_clients': {
                            'main': {
                                'name': 'litellm',
                                'config': {
                                    'client_args': client_args,
                                    'request_args': {'model': 'gpt-test'},
                                },
                            }
                        },
                        'runner': {'timeout': 10},
                    },
                ) as response:
                    assert response.status == 201

            stream_events = await _read_sse_events_until(
                session,
                f'{base_url}/v1/runs/run_live_shape_stream/events?cursor=0',
                stop=_has_stream_trace_tail,
                timeout=10,
            )
            non_stream_events = await _read_sse_events_until(
                session,
                f'{base_url}/v1/runs/run_live_shape_non_stream/events?cursor=0',
                stop=_has_non_stream_trace_tail,
                timeout=10,
            )

        stream_outputs = _openai_completion_end_outputs(stream_events)
        non_stream_outputs = _openai_completion_end_outputs(non_stream_events)
        assert stream_outputs is not None
        assert non_stream_outputs is not None
        assert _path_keys(stream_outputs) == _path_keys(non_stream_outputs)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_live_server_non_stream_mode_has_matching_final_output_without_chat_completion_events(monkeypatch) -> None:
    async def fake_acompletion(**kwargs: Any) -> FakeModelResponse:
        assert 'stream' not in kwargs
        return FakeModelResponse(
            {
                'id': 'chatcmpl-live-full',
                'created': 1,
                'model': 'gpt-test',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {'role': 'assistant', 'content': '<section>Sports landing page</section>'},
                    }
                ],
                'usage': {'prompt_tokens': 9, 'completion_tokens': 6, 'total_tokens': 15},
            }
        )

    monkeypatch.setattr('nexus.extensions.openai_clients.litellm.litellm.acompletion', fake_acompletion)
    base_url, runner = await _start_live_server()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{base_url}/v1/runs',
                json={
                    'run_id': 'run_live_full_mock',
                    'task_name': 'live_stream_probe_task',
                    'task_config': {'user_prompt': 'Build a sports-themed website.'},
                    'openai_clients': {
                        'main': {
                            'name': 'litellm',
                            'config': {
                                'client_args': {
                                    'api_key': 'test',
                                    'max_retries': 1,
                                },
                                'request_args': {'model': 'gpt-test'},
                            },
                        }
                    },
                    'runner': {'timeout': 10},
                },
            ) as response:
                assert response.status == 201

            events = await _read_sse_events_until(
                session,
                f'{base_url}/v1/runs/run_live_full_mock/events?cursor=0',
                stop=_has_non_stream_trace_tail,
                timeout=10,
            )

        names = [_trace_event_name(event) for event in events]
        assert all(not name.startswith('chat.completion.') for name in names if name)
        assert _run_output_content(events) == '<section>Sports landing page</section>'
        assert _openai_completion_end_content(events) == '<section>Sports landing page</section>'
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_live_server_real_model_provider_stream_and_non_stream_when_enabled() -> None:
    if os.environ.get('NEXUS_LIVE_REAL_STREAM_E2E') != '1':
        pytest.skip('set NEXUS_LIVE_REAL_STREAM_E2E=1 to run the real model provider stream test')
    api_key = os.environ.get('MODEL_SERVICE_API_KEY')
    if not api_key:
        api_key = (
            f'{os.environ["MODEL_SERVICE_API_APP_ID"]}:{os.environ["MODEL_SERVICE_API_APP_KEY"]}'
            '?provider=aws_third&timeout=3600'
        )
    base_url_env = os.environ.get('MODEL_SERVICE_BASE_URL', 'http://trpc-gpt-eval.production.polaris:8080')
    model = os.environ.get('MODEL_SERVICE_MODEL', 'anthropic.claude-sonnet-4-6')

    base_url, runner = await _start_live_server()
    try:
        async with aiohttp.ClientSession() as session:
            prompt = (
                'Build a sports-themed website. Compare basketball, running, and cycling sections. '
                'Return a concise HTML sketch with at least five short lines.'
            )
            stream_events = await _create_and_collect_real_probe_events(
                session,
                base_url,
                run_id=f'run_live_real_stream_{int(time.time())}',
                prompt=prompt,
                api_key=api_key,
                base_url_env=base_url_env,
                model=model,
                stream=True,
            )
            non_stream_events = await _create_and_collect_real_probe_events(
                session,
                base_url,
                run_id=f'run_live_real_full_{int(time.time())}',
                prompt=prompt,
                api_key=api_key,
                base_url_env=base_url_env,
                model=model,
                stream=False,
            )

        stream_names = [_trace_event_name(event) for event in stream_events]
        non_stream_names = [_trace_event_name(event) for event in non_stream_events]
        assert {'run.status', 'run.output'}.issubset(
            {event['type'] for event in stream_events if event['type'].startswith('run.')}
        )
        assert {'run.status', 'run.output'}.issubset(
            {event['type'] for event in non_stream_events if event['type'].startswith('run.')}
        )
        assert all(event['type'] not in {'run.failed', 'run.killed'} for event in stream_events)
        assert all(event['type'] not in {'run.failed', 'run.killed'} for event in non_stream_events)
        assert stream_names.count('chat.completion.delta') >= 2
        assert 'chat.completion.completed' in stream_names
        assert all(not name.startswith('chat.completion.') for name in non_stream_names if name)
        assert _run_output_content(stream_events)
        assert _run_output_content(non_stream_events)
        assert ''.join(_chat_completion_text_deltas(stream_events)) == _run_output_content(stream_events)
        assert _openai_completion_end_content(stream_events) == _run_output_content(stream_events)
        assert _openai_completion_end_content(non_stream_events) == _run_output_content(non_stream_events)
    finally:
        await runner.cleanup()


async def _create_and_collect_real_probe_events(
    session: aiohttp.ClientSession,
    base_url: str,
    *,
    run_id: str,
    prompt: str,
    api_key: str,
    base_url_env: str,
    model: str,
    stream: bool,
) -> list[dict[str, Any]]:
    client_args: dict[str, Any] = {
        'api_key': api_key,
        'base_url': base_url_env,
        'max_retries': 1,
    }
    client_args['stream'] = stream
    async with session.post(
        f'{base_url}/v1/runs',
        json={
            'run_id': run_id,
            'task_name': 'live_stream_probe_task',
            'task_config': {'user_prompt': prompt},
            'openai_clients': {
                'main': {
                    'name': 'litellm',
                    'config': {
                        'client_args': client_args,
                        'request_args': {'model': model},
                    },
                }
            },
            'runner': {'timeout': 120},
        },
    ) as response:
        assert response.status == 201
        created = await response.json()

    return await _read_sse_events_until(
        session,
        f'{base_url}/v1/runs/{created["run_id"]}/events?cursor=0',
        stop=_has_stream_trace_tail if stream else _has_non_stream_trace_tail,
        timeout=120,
    )
