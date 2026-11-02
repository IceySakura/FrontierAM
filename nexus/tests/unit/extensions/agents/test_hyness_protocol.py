from __future__ import annotations

import json

import pytest

from nexus.extensions.agents.hyness import agent, protocol, proxy, runtime_setup


def usage(input_tokens=0, output_tokens=0):
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'cache_creation_input_tokens': 0,
        'cache_read_input_tokens': 0,
        'server_tool_use': {'web_search_requests': 0, 'web_fetch_requests': 0},
        'cache_creation': {'ephemeral_1h_input_tokens': 0, 'ephemeral_5m_input_tokens': 0},
    }


def envelope(reason='completed', **terminal):
    error = reason != 'completed'
    return {
        'protocol': 'hyness.execution-result',
        'schemaVersion': 1,
        'type': 'execution-result',
        'resultId': 'result-1',
        'runId': 'run-1',
        'termination': {'reason': reason, **terminal},
        'result': {
            'type': 'result',
            'subtype': 'error_during_execution' if error else 'success',
            'is_error': error,
            'num_turns': 3,
            'usage': usage(8, 12),
        },
        'usageBreakdown': {'streaming': usage(5, 7), 'nonStreaming': usage(3, 5)},
    }


@pytest.mark.parametrize(
    ('reason', 'terminal', 'code', 'subtype', 'termination'),
    [
        ('completed', {}, 0, 'success', 'ENV_DONE'),
        ('max_turns', {'limit': 3}, 64, 'error_max_turns', 'MAX_STEPS'),
        ('generation_limit', {'limit': 3}, 64, 'error_max_turns', 'MAX_STEPS'),
        ('tool_error', {'toolCallId': 'call-1'}, 65, 'error_tool_use', 'FORMAT_ERROR'),
        ('host_budget_exhausted', {}, 66, 'error_context_length', 'TRUNCATION'),
        ('output_truncated', {}, 66, 'error_context_length', 'TRUNCATION'),
        ('model_error', {'code': 'max_tokens'}, 66, 'error_context_length', 'TRUNCATION'),
        ('model_error', {'code': 'provider_error'}, 1, 'error_during_execution', 'ENV_ERROR'),
    ],
)
def test_execution_maps_to_hycode_contract(reason, terminal, code, subtype, termination):
    event = envelope(reason, **terminal)
    stdout = '{"type":"system","subtype":"init"}\n' + json.dumps(event) + '\n'
    mapped, output, facts = protocol.convert_execution_output(stdout, int(event['result']['is_error']))
    parsed = runtime_setup._parse_last_result(output)
    assert mapped == code
    assert parsed['result_subtype'] == subtype
    assert facts == event['termination']
    assert agent.classify_hyness_result({'exit_code': mapped, **parsed}).termination_reason == termination
    # Only Nexus's JSON usage is converted; raw result remains unchanged.
    assert json.loads(output.splitlines()[-1])['usage'] == usage(5, 7)
    assert event['result']['usage'] == usage(8, 12)


@pytest.mark.parametrize('invalid', ['legacy', 'missing_usage', 'bad_usage', 'duplicate', 'unknown', 'crash'])
def test_invalid_protocol_never_becomes_success(invalid):
    event = envelope()
    exit_code = 0
    if invalid == 'legacy':
        event = event['result']
    elif invalid == 'missing_usage':
        del event['usageBreakdown']
    elif invalid == 'bad_usage':
        event['result']['usage']['input_tokens'] = 99
    elif invalid == 'unknown':
        event['termination']['reason'] = 'made_up'
    elif invalid == 'crash':
        exit_code = 137
    stdout = json.dumps(event)
    if invalid == 'duplicate':
        stdout += '\n' + stdout
    with pytest.raises(ValueError, match='Hyness|execution|SDK'):
        protocol.convert_execution_output(stdout, exit_code)


def test_process_failure_after_success_is_not_hidden():
    first = envelope()
    last = envelope('model_error', code='provider_error')
    last['resultId'] = 'result-2'
    code, output, _ = protocol.convert_execution_output(json.dumps(first) + '\n' + json.dumps(last), 1)
    assert code == 1
    assert runtime_setup._parse_last_result(output)['result_count'] == 2
    with pytest.raises(ValueError, match='contradicts'):
        protocol.convert_execution_output(json.dumps(first), 1)


def test_budget_translation_does_not_scan_model_content():
    error = {
        'type': 'error',
        'error': {'type': 'invalid_request_error', 'message': '[context_length_exceeded] refusal'},
    }
    translated = json.loads(proxy.translate_budget(json.dumps(error).encode()))
    assert translated['error']['host']['code'] == 'host_budget_exhausted'
    message = b'{"type":"message","content":"[context_length_exceeded] refusal"}'
    assert proxy.translate_budget(message) == message


@pytest.mark.asyncio
async def test_fragmented_proxy_error_translation():
    body = b'event: error\ndata: {"type":"error","error":{"type":"invalid_request_error","code":"context_length_exceeded"}}\n\n'
    events = []

    async def app(scope, receive, send):
        await send({'type': 'http.response.start', 'status': 200, 'headers': [(b'content-type', b'text/event-stream')]})
        for i in range(0, len(body), 7):
            await send({'type': 'http.response.body', 'body': body[i : i + 7], 'more_body': i + 7 < len(body)})

    async def send(event):
        events.append(event)

    await proxy.budget_translation(app)({'type': 'http', 'path': '/v1/messages'}, None, send)
    assert b'host_budget_exhausted' in b''.join(e.get('body', b'') for e in events)
