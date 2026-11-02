"""Tests for the ``anthropic_messages`` openai_client.

This client exists so the Taiji platform can reach Claude by naming it in
``parameter_configuration.nexus_main_client_name`` and passing flat kwargs through
``main_client_args`` (see the module docstring). The tests below therefore pin the
things that break when driven from that direction and were the reason ``vertex``
could not be reused as-is:

* the base URL arrives as ``.../v1`` and must not produce ``/v1/v1/messages``,
* ``extra_headers['anthropic-beta']`` from cua_anthropic_agent must reach the wire,
* ``cache_task_id`` must land in the gateway credential,
* a Messages-API body carries ``model`` and no ``anthropic_version``, while a
  Bedrock InvokeModel body is the mirror image.
"""

from __future__ import annotations

import json
import logging

from typing import Any

import pytest

from nexus.core import ContextLengthExceededError, OpenAIConfig, openai_client_registry
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.openai_clients.anthropic_messages import (
    AnthropicMessagesClient,
    AnthropicMessagesRequestError,
    _merge_beta_values,
    _normalize_base_url,
)
from nexus.extensions.openai_clients.vertex import VertexAnthropicHTTPError


_RESPONSE = {
    'id': 'msg_1',
    'type': 'message',
    'role': 'assistant',
    'model': 'anthropic.claude-opus-4-8',
    'content': [{'type': 'text', 'text': 'OK'}],
    'stop_reason': 'end_turn',
    'usage': {'input_tokens': 7, 'output_tokens': 2},
}


def _client(**client_args: Any) -> AnthropicMessagesClient:
    args = {'api_key': 'cred', 'base_url': 'http://gateway/v1', **client_args}
    request_args = args.pop('_request_args', {'model': 'anthropic.claude-opus-4-8'})
    return AnthropicMessagesClient(OpenAIConfig(client_args=args, request_args=request_args))


class _Recorder:
    """Stands in for the network: records one POST and replays a canned result."""

    def __init__(self, result: Any = _RESPONSE) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, endpoint: str, headers: Any, payload: dict[str, Any], model: str = '') -> dict[str, Any]:
        self.calls.append({'endpoint': endpoint, 'headers': dict(headers), 'payload': payload, 'model': model})
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


async def _noop_sleep(_delay: float) -> None:
    """Collapse the retry backoff so the retry test does not actually wait."""


def test_is_registered() -> None:
    assert openai_client_registry.get('anthropic_messages') is AnthropicMessagesClient


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('http://host/v1', 'http://host'),
        ('http://host/v1/', 'http://host'),
        ('http://host', 'http://host'),
        ('http://host/gateway/v1', 'http://host/gateway'),
    ],
)
def test_normalize_base_url_strips_one_trailing_v1(raw: str, expected: str) -> None:
    assert _normalize_base_url(raw) == expected


def test_merge_beta_values_unions_and_preserves_order() -> None:
    assert _merge_beta_values(['a'], 'b,c', 'a,d') == ['a', 'b', 'c', 'd']


def test_platform_base_url_does_not_double_the_v1_segment() -> None:
    # The platform's external_url_custom_params.url ends in /v1; appending blindly
    # would yield /v1/v1/messages, which the gateway answers with HTTP 404.
    client = _client(base_url='http://llm-api.example/v1', ensure_v1_path=True, max_retries=5)
    assert client._resolve_endpoint('anthropic.claude-opus-4-8') == 'http://llm-api.example/v1/messages'


def test_missing_api_key_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match='api_key'):
        AnthropicMessagesClient(OpenAIConfig(client_args={'base_url': 'http://gw'}, request_args={}))


@pytest.mark.parametrize(('field', 'value'), [('api_style', 'grpc'), ('auth_style', 'oauth')])
def test_unknown_style_is_rejected(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        _client(**{field: value})


def test_missing_model_is_rejected() -> None:
    client = _client(_request_args={})
    with pytest.raises(ValueError, match='model'):
        client._prepare([{'role': 'user', 'content': 'hi'}], None, {})


def test_messages_style_puts_model_in_body_and_drops_anthropic_version() -> None:
    client = _client()
    model, payload, headers, endpoint = client._prepare(
        [{'role': 'user', 'content': 'hi'}],
        None,
        {'max_tokens': 64},
    )

    assert model == 'anthropic.claude-opus-4-8'
    assert endpoint == 'http://gateway/v1/messages'
    assert payload['model'] == 'anthropic.claude-opus-4-8'
    # anthropic_version is a Vertex/Bedrock field; the Messages API rejects it.
    assert 'anthropic_version' not in payload
    assert payload['max_tokens'] == 64
    assert headers['anthropic-version'] == '2023-06-01'
    assert headers['Authorization'].startswith('Bearer cred?')


def test_bedrock_style_puts_model_in_url_and_keeps_anthropic_version() -> None:
    client = _client(base_url='http://gateway', api_style='bedrock_invoke')
    model, payload, headers, endpoint = client._prepare([{'role': 'user', 'content': 'hi'}], None, {})

    assert endpoint == f'http://gateway/model/{model}/invoke'
    assert 'model' not in payload
    assert payload['anthropic_version'] == 'bedrock-2023-05-31'
    # InvokeModel is not the Messages API, so the version header must stay off.
    assert 'anthropic-version' not in headers


def test_agent_supplied_betas_reach_the_wire_merged_with_configured_ones() -> None:
    # vertex drops extra_headers entirely, which is how cua_anthropic_agent ships
    # its per-turn betas; losing them silently changes the model's capabilities.
    client = _client(betas='computer-use-2025-11-24')
    _, _, headers, _ = client._prepare(
        [{'role': 'user', 'content': 'hi'}],
        None,
        {'extra_headers': {'anthropic-beta': 'token-efficient-tools-2025-02-19', 'x-trace': 'abc'}},
    )

    assert headers['anthropic-beta'] == 'computer-use-2025-11-24,token-efficient-tools-2025-02-19'
    assert headers['x-trace'] == 'abc'


def test_extra_headers_is_not_forwarded_as_a_payload_field() -> None:
    client = _client()
    _, payload, _, _ = client._prepare(
        [{'role': 'user', 'content': 'hi'}],
        None,
        {'extra_headers': {'anthropic-beta': 'b1'}},
    )
    assert 'extra_headers' not in payload


def test_config_level_extra_headers_reach_the_wire() -> None:
    # hy_unify_eval injects SessionID/StrategyType into decoding_params, which it
    # then hands over as request_args (base.py:327-334 / :86-94). Those headers
    # drive consistent-hash LB routing, so losing them silently re-points traffic.
    client = _client(
        _request_args={
            'model': 'anthropic.claude-opus-4-8',
            'extra_headers': {'SessionID': 'sess-1', 'StrategyType': 'ConsistentHash'},
        }
    )
    _, payload, headers, _ = client._prepare([{'role': 'user', 'content': 'hi'}], None, {})

    assert headers['SessionID'] == 'sess-1'
    assert headers['StrategyType'] == 'ConsistentHash'
    # _build_payload ignores unknown keys, so a leak here would be invisible.
    assert 'extra_headers' not in payload


def test_per_call_extra_headers_win_per_key_over_config_level_ones() -> None:
    # The two sources carry different keys in practice; a collision must not make
    # either side disappear wholesale.
    client = _client(
        _request_args={
            'model': 'anthropic.claude-opus-4-8',
            'extra_headers': {'SessionID': 'sess-1', 'x-keep': 'config'},
        }
    )
    _, _, headers, _ = client._prepare(
        [{'role': 'user', 'content': 'hi'}],
        None,
        {'extra_headers': {'SessionID': 'sess-2', 'x-call': 'call'}},
    )

    assert headers['SessionID'] == 'sess-2'
    assert headers['x-keep'] == 'config'
    assert headers['x-call'] == 'call'


@pytest.mark.parametrize('header', ['Authorization', 'authorization', 'x-api-key'])
def test_extra_headers_cannot_clobber_the_routed_credential(header: str, caplog) -> None:
    # The resolved credential carries ?provider=&model=; letting a passed-through
    # header replace it strips the routing and the gateway answers with an opaque
    # PlatformNoAvailableAccount 500 that does not look like a config bug.
    client = _client()
    with caplog.at_level(logging.WARNING):
        _, _, headers, _ = client._prepare(
            [{'role': 'user', 'content': 'hi'}],
            None,
            {'extra_headers': {header: 'Bearer attacker-supplied', 'x-trace': 'keep'}},
        )

    assert headers['Authorization'].startswith('Bearer cred?')
    assert 'attacker-supplied' not in json.dumps(headers)
    assert headers['x-trace'] == 'keep', 'unrelated headers must still pass through'
    assert header in caplog.text, 'silently dropping an auth header would be its own trap'


@pytest.mark.parametrize('raw', [':', ' : '])
def test_empty_appid_appkey_pair_is_rejected(raw: str) -> None:
    # The example configs interpolate '${APP_ID}:${APP_KEY}'; with both env vars
    # unset that resolves to a bare ':' which is non-empty and would otherwise
    # start fine and then 401 on every request with no pointer to the real cause.
    with pytest.raises(ValueError, match='env vars are unset'):
        _client(api_key=raw)


@pytest.mark.asyncio
async def test_exhausted_retries_surface_as_a_sanitized_error(monkeypatch) -> None:
    # The retry-then-succeed test covers the happy path; this pins the terminal one:
    # the loop must stop at max_retries and the raw upstream error must be wrapped
    # so the body never escapes to the caller.
    client = _client(max_retries=3)
    recorder = _Recorder(VertexAnthropicHTTPError(503, 'upstream busy forever'))
    monkeypatch.setattr(client, '_post_once', recorder)
    monkeypatch.setattr('asyncio.sleep', _noop_sleep)

    with pytest.raises(AnthropicMessagesRequestError) as excinfo:
        await client._create_completion([{'role': 'user', 'content': 'hi'}])

    assert len(recorder.calls) == 3, 'must stop at max_retries, not loop forever'
    assert excinfo.value.status == 503
    assert 'upstream busy forever' not in str(excinfo.value)


def test_gateway_routing_is_derived_from_the_api_style() -> None:
    assert 'provider=anthropic' in _client()._resolve_api_key('m')
    assert 'provider=aws_third' in _client(api_style='bedrock_invoke')._resolve_api_key('m')
    assert 'model=m' in _client()._resolve_api_key('m')


def test_cache_task_id_is_hashed_into_the_credential() -> None:
    import hashlib

    digest = hashlib.md5(b'task-7').hexdigest()
    assert f'cache_task_id={digest}' in _client(cache_task_id='task-7')._resolve_api_key('m')


def test_preexisting_query_string_is_treated_as_fully_routed() -> None:
    # A hand-tuned credential already names its account; re-deriving provider from
    # api_style would silently re-point the request at a different one.
    client = _client(api_key='cred?provider=aws_third&model=custom')
    assert client._resolve_api_key('other') == 'cred?provider=aws_third&model=custom'


def test_explicit_client_args_still_override_a_routed_credential() -> None:
    client = _client(api_key='cred?provider=aws_third', provider='newapi')
    assert 'provider=newapi' in client._resolve_api_key('m')


def test_x_api_key_auth_style_replaces_the_bearer_header() -> None:
    _, _, headers, _ = _client(auth_style='x-api-key')._prepare([{'role': 'user', 'content': 'hi'}], None, {})
    assert 'Authorization' not in headers
    assert headers['x-api-key'].startswith('cred?')


def test_explicit_endpoint_wins_over_base_url() -> None:
    client = _client(endpoint='http://direct/custom/messages')
    assert client._resolve_endpoint('m') == 'http://direct/custom/messages'


@pytest.mark.asyncio
async def test_create_completion_returns_extended_completion_with_provider_call(monkeypatch) -> None:
    client = _client(capture_raw_io=True)
    recorder = _Recorder()
    monkeypatch.setattr(client, '_post_once', recorder)

    completion = await client._create_completion([{'role': 'user', 'content': 'hi'}], max_tokens=32)

    assert isinstance(completion, ExtendedChatCompletion)
    assert completion.choices[0].message.content == 'OK'
    assert completion.choices[0].finish_reason == 'stop'

    provider_call = completion.provider_specific_fields['provider_call']
    assert provider_call['provider'] == 'anthropic_messages'
    assert provider_call['api_style'] == 'messages'
    assert provider_call['endpoint'] == 'http://gateway/v1/messages'
    assert provider_call['attempts'] == [{'attempt': 1, 'status': 'success'}]
    assert recorder.calls[0]['payload']['max_tokens'] == 32


@pytest.mark.asyncio
async def test_bedrock_streaming_is_labelled_with_the_model_from_the_url(monkeypatch) -> None:
    # On the bedrock_invoke path the model is popped from the body into the URL, so
    # _read_stream_response can only learn it from the explicit argument. If
    # _post_json stops forwarding it, every streamed chunk is labelled '' and the
    # completion carries no model -- silent, and invisible in non-streaming tests.
    client = _client(base_url='http://gateway', api_style='bedrock_invoke')
    recorder = _Recorder()
    monkeypatch.setattr(client, '_post_once', recorder)

    await client._create_completion([{'role': 'user', 'content': 'hi'}], stream=True)

    assert 'model' not in recorder.calls[0]['payload']
    assert recorder.calls[0]['model'] == 'anthropic.claude-opus-4-8'


@pytest.mark.asyncio
async def test_credential_echoed_by_the_gateway_never_reaches_the_log(monkeypatch, caplog) -> None:
    """A 4xx body that quotes our own auth header must not be logged verbatim.

    The credential rides in the Authorization header, and gateways routinely echo
    the offending request back in an error body. ``provider_call`` is safe because
    ``_record_provider_call`` redacts against ``client_args``, but the WARNING that
    exists to make a 400 diagnosable is NOT on that path -- so it needs its own
    scrub, or the api_key lands in plaintext in every run log.
    """

    secret = 'sk-live-DEADBEEF'
    client = _client(api_key=secret, capture_raw_io=True)
    monkeypatch.setattr(
        client,
        '_post_once',
        _Recorder(VertexAnthropicHTTPError(400, f'invalid request; authorization was Bearer {secret}')),
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(AnthropicMessagesRequestError):
            await client._create_completion([{'role': 'user', 'content': 'hi'}])

    assert caplog.text, 'the upstream detail must still be logged, or 400s are undiagnosable'
    assert 'status=400' in caplog.text
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_recorded_provider_call_is_scrubbed_of_the_credential(monkeypatch) -> None:
    # Runs the real _record_provider_call (no stubbing) so this asserts the actual
    # redaction guarantee rather than a mock's behaviour.
    secret = 'sk-live-CAFEBABE'
    seen: list[dict[str, Any]] = []
    client = _client(api_key=secret, capture_raw_io=True)
    real_record = client._record_provider_call

    def spy(provider_call: dict[str, Any]) -> None:
        real_record(provider_call)
        seen.append(json.loads(json.dumps(provider_call, default=str)))

    monkeypatch.setattr(client, '_record_provider_call', spy)
    monkeypatch.setattr(
        client,
        '_post_once',
        _Recorder(VertexAnthropicHTTPError(400, f'authorization was Bearer {secret}?provider=anthropic')),
    )

    with pytest.raises(AnthropicMessagesRequestError):
        await client._create_completion([{'role': 'user', 'content': 'hi'}])

    assert seen, 'a failed call must still record its provider_call'
    assert secret not in json.dumps(seen, default=str)


@pytest.mark.asyncio
async def test_error_body_logging_can_be_suppressed(monkeypatch, caplog) -> None:
    # Operators need a kill switch for body logging in production, matching
    # vertex's NEXUS_VERTEX_LOG_ERROR_BODY gate.
    monkeypatch.setenv('NEXUS_ANTHROPIC_LOG_ERROR_BODY', '0')
    client = _client()
    monkeypatch.setattr(client, '_post_once', _Recorder(VertexAnthropicHTTPError(400, 'upstream detail')))

    with caplog.at_level(logging.WARNING):
        with pytest.raises(AnthropicMessagesRequestError):
            await client._create_completion([{'role': 'user', 'content': 'hi'}])

    assert 'upstream detail' not in caplog.text


@pytest.mark.asyncio
async def test_post_once_parses_a_real_sse_stream_and_labels_it_from_the_url_model(monkeypatch) -> None:
    """Exercises the streaming branch of _post_once itself, not a stub of it.

    On the bedrock path the model is popped out of the body into the URL, so when
    the upstream stream omits it (``build`` only ``setdefault``s the label) the
    explicit ``model`` argument is the sole remaining source. Every other test
    replaces ``_post_once`` wholesale, leaving this branch unexecuted.
    """

    events = [
        # No "model" key here: this is what forces the label to come from the URL.
        '{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant",'
        '"content":[],"usage":{"input_tokens":3,"output_tokens":0}}}',
        '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
        '{"type":"content_block_stop","index":0}',
        '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}',
        '{"type":"message_stop"}',
    ]

    class _Content:
        def __aiter__(self):
            async def gen():
                for event in events:
                    # SSE frames are blank-line delimited; the reader only flushes
                    # an event when it sees the empty line.
                    yield f'data: {event}\n'.encode()
                    yield b'\n'

            return gen()

    class _StreamResponse:
        status = 200
        headers: dict[str, str] = {}
        content = _Content()

        async def text(self) -> str:
            return ''

    class _PostCtx:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, *exc: Any) -> None: ...

    class _Session:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc: Any) -> None: ...

        def post(self, *args: Any, **kwargs: Any) -> _PostCtx:
            return _PostCtx()

    monkeypatch.setattr('aiohttp.ClientSession', _Session)

    client = _client(base_url='http://gateway', api_style='bedrock_invoke')
    model, payload, headers, endpoint = client._prepare([{'role': 'user', 'content': 'hi'}], None, {'stream': True})
    assert 'model' not in payload, 'precondition: bedrock keeps the model in the URL only'

    result = await client._post_once(endpoint, headers, payload, model)

    assert result['model'] == 'anthropic.claude-opus-4-8'
    assert result['content'] == [{'type': 'text', 'text': 'hi'}]
    assert result['stop_reason'] == 'end_turn'


@pytest.mark.asyncio
async def test_retryable_status_is_retried_then_succeeds(monkeypatch) -> None:
    client = _client(max_retries=3)
    attempts: list[int] = []

    async def flaky(endpoint: str, headers: Any, payload: dict[str, Any], model: str = '') -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) < 3:
            raise VertexAnthropicHTTPError(503, 'upstream busy')
        return _RESPONSE

    monkeypatch.setattr(client, '_post_once', flaky)
    monkeypatch.setattr('asyncio.sleep', _noop_sleep)

    completion = await client._create_completion([{'role': 'user', 'content': 'hi'}])

    assert len(attempts) == 3
    statuses = [a['status'] for a in completion.provider_specific_fields['provider_call']['attempts']]
    assert statuses == ['error', 'error', 'success']


@pytest.mark.asyncio
async def test_non_retryable_status_fails_without_leaking_the_body(monkeypatch) -> None:
    client = _client(max_retries=3)
    recorder = _Recorder(VertexAnthropicHTTPError(400, 'secret-bearing upstream detail'))
    monkeypatch.setattr(client, '_post_once', recorder)

    with pytest.raises(AnthropicMessagesRequestError) as excinfo:
        await client._create_completion([{'role': 'user', 'content': 'hi'}])

    assert excinfo.value.status == 400
    assert 'secret-bearing upstream detail' not in str(excinfo.value)
    assert len(recorder.calls) == 1


@pytest.mark.asyncio
async def test_context_length_error_is_translated(monkeypatch) -> None:
    client = _client()
    monkeypatch.setattr(
        client,
        '_post_once',
        _Recorder(VertexAnthropicHTTPError(413, 'input is too long for requested model')),
    )

    with pytest.raises(ContextLengthExceededError):
        await client._create_completion([{'role': 'user', 'content': 'hi'}])


@pytest.mark.asyncio
async def test_enforced_request_args_override_caller_values(monkeypatch) -> None:
    client = AnthropicMessagesClient(
        OpenAIConfig(
            client_args={'api_key': 'cred', 'base_url': 'http://gateway/v1'},
            request_args={'model': 'anthropic.claude-opus-4-8', 'temperature': 0.7},
            enforced_request_args={'temperature': 1.0},
        )
    )
    recorder = _Recorder()
    monkeypatch.setattr(client, '_post_once', recorder)

    await client._create_completion([{'role': 'user', 'content': 'hi'}], temperature=0.2)

    assert recorder.calls[0]['payload']['temperature'] == 1.0


@pytest.mark.asyncio
async def test_accepts_the_exact_call_shape_cua_anthropic_agent_sends(monkeypatch) -> None:
    """End-to-end contract with the agent, without a network call.

    This is the test that would have caught the reason ``vertex`` could not be
    reused: the agent hands its betas over in ``extra_headers`` and expects the
    thinking/text/tool_use blocks back in the Anthropic shape. Driving the real
    ``_to_chat_messages`` / ``_chat_response_to_anthropic`` boundary here means a
    future change to either side breaks this test rather than an OSWorld run.
    """

    from nexus.extensions.agents.cua.cua_anthropic_agent.agent import (
        CLAUDE_OPUS_48_PROFILE,
        _chat_response_to_anthropic,
        _client_tools,
        _to_chat_messages,
    )

    anthropic_messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Open Chrome'},
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'aGk='}},
            ],
        },
        {
            'role': 'assistant',
            'content': [
                {'type': 'thinking', 'thinking': 'plan', 'signature': 'sig'},
                {'type': 'tool_use', 'id': 'toolu_1', 'name': 'computer', 'input': {'action': 'screenshot'}},
            ],
        },
        {
            'role': 'user',
            'content': [
                {'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': [{'type': 'text', 'text': 'done'}]}
            ],
        },
    ]
    chat_messages = _to_chat_messages(anthropic_messages, 'SYSTEM PROMPT')

    client = _client(capture_raw_io=True)
    recorder = _Recorder(
        {
            'id': 'msg_2',
            'type': 'message',
            'role': 'assistant',
            'model': 'anthropic.claude-opus-4-8',
            'content': [
                {'type': 'thinking', 'thinking': 'deciding', 'signature': 'sg'},
                {'type': 'text', 'text': 'clicking'},
                {
                    'type': 'tool_use',
                    'id': 'toolu_2',
                    'name': 'computer',
                    'input': {'action': 'left_click', 'coordinate': [10, 20]},
                },
            ],
            'stop_reason': 'tool_use',
            'usage': {'input_tokens': 5, 'output_tokens': 3},
        }
    )
    monkeypatch.setattr(client, '_post_once', recorder)

    completion = await client._create_completion(
        chat_messages,
        max_tokens=4096,
        tools=_client_tools(CLAUDE_OPUS_48_PROFILE, 1280, 720),
        extra_headers={'anthropic-beta': CLAUDE_OPUS_48_PROFILE.computer_use_beta_flag},
        model='anthropic.claude-opus-4-8',
        temperature=1.0,
    )

    sent = recorder.calls[0]
    assert sent['endpoint'] == 'http://gateway/v1/messages'
    assert sent['headers']['anthropic-beta'] == CLAUDE_OPUS_48_PROFILE.computer_use_beta_flag
    # The system prompt must be hoisted out of messages into its own field.
    assert sent['payload']['system']
    assert [m['role'] for m in sent['payload']['messages']] == ['user', 'assistant', 'user']
    assert sent['payload']['tools']
    assert any(
        block.get('type') == 'tool_result'
        for message in sent['payload']['messages']
        for block in (message['content'] if isinstance(message['content'], list) else [])
    )

    # The agent loop consumes the response back in the Anthropic block shape.
    converted = _chat_response_to_anthropic(completion)
    assert converted['stop_reason'] == 'tool_use'
    assert [block['type'] for block in converted['content']] == ['thinking', 'text', 'tool_use']
    assert converted['content'][-1]['input'] == {'action': 'left_click', 'coordinate': [10, 20]}


# --- native request passthrough -------------------------------------------------
#
# Claude Code speaks Anthropic Messages and this client speaks Anthropic Messages,
# so when the runtime bridge hands over the agent's own body there is nothing to
# convert. These tests pin what that body is and is not allowed to pick up on the
# way to the wire.


def _native(**body_overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        'model': 'claude-sonnet-4-6',
        'system': [{'type': 'text', 'text': 'Be brief.', 'cache_control': {'type': 'ephemeral'}}],
        'messages': [
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'hi', 'cache_control': {'type': 'ephemeral'}}],
            }
        ],
        'max_tokens': 64,
        'thinking': {'type': 'enabled', 'budget_tokens': 2048},
        'metadata': {'user_id': 'u-1'},
        'top_k': 5,
    }
    body.update(body_overrides)
    return {'protocol': 'anthropic', 'body': body}


def test_client_advertises_the_anthropic_wire_protocol() -> None:
    assert AnthropicMessagesClient.native_request_protocol == 'anthropic'


def test_native_body_goes_to_the_wire_with_only_the_model_replaced() -> None:
    client = _client()
    envelope = _native()

    model, payload, _headers, endpoint = client._prepare([], None, {'native_request': envelope})

    assert model == 'anthropic.claude-opus-4-8'  # host config wins over the agent's choice
    assert endpoint == 'http://gateway/v1/messages'
    assert payload['model'] == 'anthropic.claude-opus-4-8'
    # Everything the anthropic->openai->anthropic round trip used to drop.
    assert payload['thinking'] == {'type': 'enabled', 'budget_tokens': 2048}
    assert payload['metadata'] == {'user_id': 'u-1'}
    assert payload['top_k'] == 5
    assert payload['system'] == envelope['body']['system']
    assert payload['messages'] == envelope['body']['messages']


def test_native_body_keeps_exactly_the_agents_own_cache_breakpoints() -> None:
    # The agent placed its own breakpoints; a second apply_cache_control pass would
    # add another and Anthropic caps them at 4.
    client = _client()
    envelope = _native()
    before = json.dumps(envelope, sort_keys=True)

    _model, payload, _headers, _endpoint = client._prepare([], None, {'native_request': envelope})

    marked = [
        block
        for message in payload['messages']
        for block in message['content']
        if isinstance(block, dict) and 'cache_control' in block
    ]
    assert len(marked) == 1
    assert sum('cache_control' in block for block in payload['system']) == 1
    assert json.dumps(envelope, sort_keys=True) == before  # caller's body not mutated


def test_native_body_never_inherits_stream_from_the_agent() -> None:
    # `stream` belongs to the transport: the bridge is non-streaming to the
    # provider and re-synthesises SSE on the sandbox side.
    client = _client()

    _model, payload, _headers, _endpoint = client._prepare([], None, {'native_request': _native(stream=True)})

    assert 'stream' not in payload


def test_enforced_request_args_still_win_over_a_native_body() -> None:
    client = AnthropicMessagesClient(
        OpenAIConfig(
            client_args={'api_key': 'cred', 'base_url': 'http://gateway/v1'},
            request_args={'model': 'anthropic.claude-opus-4-8'},
            enforced_request_args={'temperature': 1.0},
        )
    )

    _model, payload, _headers, _endpoint = client._prepare([], None, {'native_request': _native(temperature=0.2)})

    assert payload['temperature'] == 1.0


def test_bedrock_style_swaps_model_for_anthropic_version_on_a_native_body() -> None:
    client = _client(api_style='bedrock_invoke')

    _model, payload, _headers, _endpoint = client._prepare([], None, {'native_request': _native()})

    assert 'model' not in payload
    assert payload['anthropic_version'] == 'bedrock-2023-05-31'


@pytest.mark.parametrize(
    ('envelope', 'error', 'match'),
    [
        ('not-a-mapping', TypeError, 'must be a mapping'),
        ({'protocol': 'openai', 'body': {}}, ValueError, "must be 'anthropic'"),
        ({'protocol': 'anthropic', 'body': ['nope']}, TypeError, 'body must be a mapping'),
    ],
)
def test_a_malformed_native_envelope_is_rejected(envelope: Any, error: type[Exception], match: str) -> None:
    # There is no OpenAI-shaped payload to fall back to, so a broken envelope must
    # raise rather than quietly send an empty request.
    with pytest.raises(error, match=match):
        _client()._prepare([], None, {'native_request': envelope})


@pytest.mark.asyncio
async def test_native_request_travels_the_normal_create_completion_path(monkeypatch) -> None:
    client = _client()
    recorder = _Recorder()
    monkeypatch.setattr(client, '_post_once', recorder)

    completion = await client._create_completion([], native_request=_native())

    assert recorder.calls[0]['payload']['thinking'] == {'type': 'enabled', 'budget_tokens': 2048}
    # The response side still produces the native_response envelope it always did.
    native_response = completion.provider_specific_fields['native_response']
    assert native_response['protocol'] == 'anthropic'
    assert native_response['body'] == _RESPONSE
