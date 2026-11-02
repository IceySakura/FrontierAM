import json

import pytest

from nexus.extensions.agents.openclaw.agent import (
    OpenClawInput,
    extract_final_answer,
    extract_recorded_messages,
    run_openclaw,
)


@pytest.mark.asyncio
async def test_run_openclaw_passes_custom_llm_idle_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_call_runtime_agent(target: object, /, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            'success': True,
            'output': json.dumps({'payloads': [{'text': 'done'}]}),
            'stderr': '',
            'exit_code': 0,
            'execution_time': 0,
            'transcript': [],
        }

    monkeypatch.setattr('nexus.extensions.agents.openclaw.agent.call_runtime_agent', fake_call_runtime_agent)

    await run_openclaw(
        OpenClawInput(
            user_prompt='solve task',
            working_dir='/workspace',
            model='bedrock/claude-opus',
            llm_idle_timeout_ms=900_000,
        )
    )

    assert captured['llm_idle_timeout_ms'] == 900_000


@pytest.mark.asyncio
async def test_run_openclaw_recovers_legacy_json_envelope_from_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = {'payloads': [{'text': 'legacy visible answer'}], 'meta': {'stopReason': 'stop'}}

    async def fake_call_runtime_agent(target: object, /, **kwargs: object) -> dict[str, object]:
        return {
            'success': True,
            'output': '',
            'stderr': f'legacy diagnostic\n{json.dumps(envelope)}',
            'exit_code': 0,
            'execution_time': 1.0,
            'transcript': [],
        }

    monkeypatch.setattr('nexus.extensions.agents.openclaw.agent.call_runtime_agent', fake_call_runtime_agent)

    result = await run_openclaw(
        OpenClawInput(user_prompt='solve task', working_dir='/workspace', model='bedrock/claude-opus')
    )

    assert result.success is True
    assert result.exit_status == 'completed'
    assert result.final_answer == 'legacy visible answer'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('failure_kind', 'failure_stage'),
    [
        ('infrastructure', 'install'),
        ('agent', 'process'),
    ],
)
async def test_run_openclaw_preserves_runtime_failure_classification(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    failure_stage: str,
) -> None:
    async def fake_call_runtime_agent(target: object, /, **kwargs: object) -> dict[str, object]:
        return {
            'success': False,
            'output': '',
            'stderr': 'OpenClaw failed',
            'exit_code': -1,
            'execution_time': 1.0,
            'transcript': [],
            'failure_kind': failure_kind,
            'failure_stage': failure_stage,
        }

    monkeypatch.setattr('nexus.extensions.agents.openclaw.agent.call_runtime_agent', fake_call_runtime_agent)

    result = await run_openclaw(
        OpenClawInput(user_prompt='solve task', working_dir='/workspace', model='bedrock/claude-opus')
    )

    assert result.failure_kind == failure_kind
    assert result.failure_stage == failure_stage
    assert result.stderr == 'OpenClaw failed'


@pytest.mark.asyncio
async def test_run_openclaw_preserves_runtime_timeout_status(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_runtime_agent(target: object, /, **kwargs: object) -> dict[str, object]:
        return {
            'success': False,
            'output': '',
            'stderr': 'Execution timed out',
            'exit_code': -1,
            'execution_time': 60.0,
            'timed_out': True,
            'transcript': [],
            'failure_kind': 'agent',
            'failure_stage': 'process',
        }

    monkeypatch.setattr('nexus.extensions.agents.openclaw.agent.call_runtime_agent', fake_call_runtime_agent)

    result = await run_openclaw(
        OpenClawInput(user_prompt='solve task', working_dir='/workspace', model='bedrock/claude-opus')
    )

    assert result.exit_status == 'timeout'
    assert result.failure_kind == 'agent'
    assert result.failure_stage == 'process'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'output',
    [
        '',
        'not json',
        json.dumps({'payloads': [{'text': 'private reasoning', 'isReasoning': True}]}),
        json.dumps({'payloads': [{'text': 'provider error', 'isError': True}]}),
        json.dumps({'payloads': [{'text': 'answer'}], 'meta': {'error': 'provider unavailable'}}),
        json.dumps({
            'payloads': [{'text': '422 status code (no body)'}],
            'meta': {'stopReason': 'error'},
        }),
    ],
)
async def test_run_openclaw_rejects_reported_success_without_visible_successful_reply(
    monkeypatch: pytest.MonkeyPatch,
    output: str,
) -> None:
    async def fake_call_runtime_agent(target: object, /, **kwargs: object) -> dict[str, object]:
        return {
            'success': True,
            'output': output,
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
            'transcript': [],
        }

    monkeypatch.setattr('nexus.extensions.agents.openclaw.agent.call_runtime_agent', fake_call_runtime_agent)

    result = await run_openclaw(
        OpenClawInput(user_prompt='solve task', working_dir='/workspace', model='bedrock/claude-opus')
    )

    assert result.success is False
    assert result.exit_status == 'error'
    assert result.final_answer is None
    assert result.failure_kind == 'agent'
    assert result.failure_stage == 'process'
    assert result.stderr


@pytest.mark.asyncio
async def test_run_openclaw_fails_closed_for_unclassified_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_runtime_agent(target: object, /, **kwargs: object) -> dict[str, object]:
        return {
            'success': False,
            'output': '',
            'stderr': 'legacy failure',
            'exit_code': -1,
            'execution_time': 1.0,
            'transcript': [],
        }

    monkeypatch.setattr('nexus.extensions.agents.openclaw.agent.call_runtime_agent', fake_call_runtime_agent)

    result = await run_openclaw(
        OpenClawInput(user_prompt='solve task', working_dir='/workspace', model='bedrock/claude-opus')
    )

    assert result.failure_kind == 'infrastructure'
    assert result.failure_stage == 'runtime_protocol'
    assert 'update the Nexus Runtime image' in result.stderr


def test_extract_final_answer_joins_the_json_payloads() -> None:
    output = json.dumps({
        'payloads': [
            {'text': 'first half', 'mediaUrl': None},
            {'text': 'second half', 'isReasoning': False, 'isError': False},
            {'text': '  '},
        ],
        'meta': {'model': 'test'},
    })

    assert extract_final_answer(output) == 'first half\n\nsecond half'


def test_extract_final_answer_filters_reasoning_and_error_payloads() -> None:
    output = json.dumps({
        'payloads': [
            {'text': 'private reasoning', 'isReasoning': True},
            {'text': 'provider error', 'isError': True},
            {'text': 'visible answer'},
        ]
    })

    assert extract_final_answer(output) == 'visible answer'


def test_extract_final_answer_does_not_fall_back_when_reply_payloads_are_private() -> None:
    output = json.dumps({
        'payloads': [
            {'text': 'private reasoning', 'isReasoning': True},
            {'text': 'provider error', 'isError': True},
        ]
    })
    transcript = [
        {
            'type': 'message',
            'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'provider error'}]},
        }
    ]

    assert extract_final_answer(output, transcript) is None


def test_extract_final_answer_falls_back_to_the_last_visible_assistant_turn() -> None:
    transcript = [
        {'type': 'message', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'earlier'}]}},
        {'type': 'message', 'message': {'role': 'user', 'content': 'more please'}},
        {
            'type': 'message',
            'message': {
                'role': 'assistant',
                'content': [{'type': 'thinking', 'thinking': 'private'}, {'type': 'text', 'text': 'openclaw answer'}],
            },
        },
    ]

    assert extract_final_answer('not json', transcript) == 'openclaw answer'


def test_extract_final_answer_never_returns_reasoning_only_turns() -> None:
    transcript = [
        {'type': 'message', 'message': {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'private'}]}},
    ]

    assert extract_final_answer(None, transcript) is None


@pytest.mark.parametrize(
    ('output', 'transcript'),
    [
        (None, None),
        ('', []),
        ('not json', None),
        (json.dumps({'payloads': []}), None),
        (json.dumps({'payloads': [{'mediaUrl': 'https://example.invalid/a.png'}]}), None),
        (json.dumps(['not the envelope']), None),
    ],
)
def test_extract_final_answer_rejects_output_without_visible_text(output, transcript) -> None:
    assert extract_final_answer(output, transcript) is None


def test_recorded_messages_unwrap_the_session_without_renaming_its_fields() -> None:
    """A grader written against an OpenClaw session compares these names, so they stay."""
    assistant = {'role': 'assistant', 'content': [{'type': 'toolCall', 'toolName': 'bash'}]}
    tool_result = {'role': 'toolResult', 'toolCallId': 'c1', 'toolName': 'bash', 'content': 'ok'}
    transcript = [
        {'type': 'session', 'id': 's1'},
        {'type': 'message', 'message': assistant},
        {'type': 'message', 'message': tool_result},
    ]

    assert extract_recorded_messages(transcript) == [assistant, tool_result]


@pytest.mark.parametrize('transcript', [None, [], ['junk'], [{'type': 'session', 'id': 's1'}]])
def test_recorded_messages_are_empty_without_a_recorded_conversation(transcript) -> None:
    assert extract_recorded_messages(transcript) == []
