from __future__ import annotations

import json

from types import SimpleNamespace

import pytest

from harbor.models.task.config import TaskOS

from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.harbor_bridge.agent_result import (
    T3_AGENT_RESULT_ENV,
    agent_termination_reason,
    build_t3_agent_result,
    classify_agent_output,
    extract_final_answer,
    extract_messages,
    upload_t3_agent_result,
    verifier_env_with_t3_agent_result,
    write_t3_agent_result,
)


class _FakeRuntime:
    def __init__(self, *, success: bool = True) -> None:
        self.success = success
        self.uploads = []

    async def upload_file(self, files, overwrite=False):
        self.uploads.append((files, overwrite))
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    path=item.path,
                    success=self.success,
                    message='' if self.success else 'disk full',
                )
                for item in files
            ]
        )


def test_extract_final_answer_normalizes_supported_agent_shapes() -> None:
    trajectory = [{'role': 'assistant', 'content': 'trajectory answer'}]

    assert extract_final_answer(SimpleNamespace(final_answer='direct answer')) == 'direct answer'
    assert extract_final_answer(SimpleNamespace(final_answer='')) == ''
    assert (
        extract_final_answer(SimpleNamespace(final_answer=[{'type': 'text', 'text': 'structured answer'}]))
        == 'structured answer'
    )
    assert extract_final_answer(SimpleNamespace(main_trajectory=trajectory)) == 'trajectory answer'
    assert (
        extract_final_answer(
            SimpleNamespace(
                main_trajectory=[
                    {'role': 'assistant', 'content': 'answer before tool'},
                    {'role': 'tool', 'content': 'tool output'},
                ]
            )
        )
        == 'answer before tool'
    )
    assert extract_final_answer(SimpleNamespace(final_answer=[{'type': 'thinking', 'text': 'private'}])) is None
    assert (
        extract_final_answer(SimpleNamespace(main_trajectory=[{'role': 'assistant', 'reasoning_content': 'private'}]))
        is None
    )
    assert extract_final_answer(SimpleNamespace(output='raw cli output')) is None
    assert extract_final_answer(None) is None


def test_extract_final_answer_normalizes_claude_code_json_result() -> None:
    output = json.dumps({
        'type': 'result',
        'subtype': 'success',
        'is_error': False,
        'result': 'claude answer',
        'session_id': 'session-1',
    })

    assert extract_final_answer(SimpleNamespace(output=output)) == 'claude answer'
    assert (
        build_t3_agent_result(
            mode='agent',
            status='success',
            agent_output=SimpleNamespace(output=output, success=True),
        ).final_answer
        == 'claude answer'
    )


@pytest.mark.parametrize(
    'output',
    [
        json.dumps([
            {'type': 'system', 'subtype': 'init'},
            {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'working'}]}},
            {'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'list answer'},
        ]),
        '\n'.join([
            json.dumps({'type': 'system', 'subtype': 'init'}),
            'non-json progress line',
            json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'list answer'}),
        ]),
    ],
)
def test_extract_final_answer_normalizes_current_claude_event_streams(output: str) -> None:
    assert extract_final_answer(SimpleNamespace(output=output)) == 'list answer'
    result = build_t3_agent_result(
        mode='agent',
        status='success',
        agent_output=SimpleNamespace(output=output, success=True),
    )
    assert result.final_answer == 'list answer'
    assert 'working' not in result.model_dump_json()


@pytest.mark.parametrize(
    'output',
    [
        '{"type": "result",',
        json.dumps({'type': 'assistant', 'result': 'not a final result'}),
        json.dumps({'type': 'result', 'result': 'missing error marker'}),
        json.dumps({'type': 'result', 'is_error': True, 'result': 'failed'}),
        json.dumps([{'type': 'result', 'is_error': True, 'result': 'failed list'}]),
        json.dumps({'type': 'result', 'result': {'text': 'not a string'}}),
        json.dumps({'type': 'result', 'reasoning': 'private'}),
    ],
)
def test_extract_final_answer_rejects_unsafe_cli_output(output: str) -> None:
    assert extract_final_answer(SimpleNamespace(output=output)) is None


def test_explicit_final_answer_wins_over_cli_output() -> None:
    output = json.dumps({'type': 'result', 'is_error': False, 'result': 'fallback'})

    assert extract_final_answer(SimpleNamespace(final_answer='', output=output)) == ''


def test_messages_come_from_whichever_way_an_agent_records_them() -> None:
    """Reported conversations are believed; only an Agent that reports none is parsed."""
    in_process = [{'role': 'assistant', 'content': 'from the loop'}]
    own_shape = [{'role': 'assistant', 'content': [{'type': 'text', 'text': 'in its own shape'}]}]
    tracker_fallback = [{'role': 'assistant', 'content': 'from ChatTracker'}]
    stream = json.dumps([
        {'type': 'system', 'subtype': 'init'},
        {
            'type': 'assistant',
            'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'from stdout'}]},
        },
        {'type': 'result', 'is_error': False, 'result': 'done'},
    ])

    assert (
        extract_messages(SimpleNamespace(main_trajectory=in_process), fallback_messages=tracker_fallback) == in_process
    )
    assert (
        extract_messages(SimpleNamespace(recorded_messages=own_shape), fallback_messages=tracker_fallback) == own_shape
    )
    assert extract_messages(SimpleNamespace(output=stream), fallback_messages=tracker_fallback) == tracker_fallback
    # Without a ChatTracker fallback, Claude Code stdout is the last resort.
    assert extract_messages(SimpleNamespace(output=stream)) == [
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'from stdout'}]}
    ]
    # Without --verbose the CLI prints the closing envelope alone: no conversation to find.
    assert extract_messages(SimpleNamespace(output=json.dumps({'type': 'result', 'result': 'done'}))) == []
    assert extract_messages(None) == []


def test_agent_result_is_small_and_reasoning_free() -> None:
    result = build_t3_agent_result(
        mode='agent',
        status='success',
        agent_output=SimpleNamespace(
            success=True,
            final_answer='42',
            main_trajectory=[{'reasoning_content': 'private reasoning'}],
        ),
    )

    assert result.model_dump() == {
        'mode': 'agent',
        'status': 'success',
        'agent_exit_status': None,
        'final_answer': '42',
        'failure_kind': None,
        'failure_stage': None,
        'failure_message': None,
    }
    assert 'reasoning' not in result.model_dump_json()


def test_agent_result_normalizes_and_limits_failure_message() -> None:
    empty = build_t3_agent_result(mode='agent', status='error', failure_message='   ')
    long_message = 'prefix-' + ('x' * 2500) + '-root-cause'
    truncated = build_t3_agent_result(
        mode='agent',
        status='error',
        failure_message=long_message,
    )

    assert empty.failure_message is None
    assert truncated.failure_message is not None
    assert len(truncated.failure_message) == 2000
    assert truncated.failure_message.startswith(f'[truncated, original length={len(long_message)}]\n')
    assert truncated.failure_message.endswith('-root-cause')


def test_timeout_and_nop_results_have_explicit_status() -> None:
    timeout = build_t3_agent_result(mode='agent', status='timeout')
    nop = build_t3_agent_result(mode='nop', status='skipped')
    oracle = build_t3_agent_result(mode='oracle', status='success')

    assert timeout.final_answer is None
    assert timeout.status == 'timeout'
    assert nop.mode == 'nop'
    assert nop.status == 'skipped'
    assert nop.agent_exit_status is None
    assert oracle.agent_exit_status is None


def test_agent_result_builder_classifies_agent_output_when_status_is_omitted() -> None:
    result = build_t3_agent_result(
        mode='agent',
        agent_output=SimpleNamespace(exit_status='max_iterations'),
    )

    assert result.status == 'incomplete'
    assert result.agent_exit_status == 'max_iterations'
    assert result.failure_message == 'agent exit status: max_iterations'


@pytest.mark.parametrize(
    ('termination_reason', 'exit_code', 'expected_status'),
    [
        ('MAX_STEPS', 64, 'incomplete'),
        ('FORMAT_ERROR', 65, 'error'),
    ],
)
def test_agent_result_accepts_legacy_termination_reason_as_exit_status(
    termination_reason: str,
    exit_code: int,
    expected_status: str,
) -> None:
    result = build_t3_agent_result(
        mode='agent',
        agent_output=SimpleNamespace(
            success=False,
            exit_code=exit_code,
            stderr='',
            termination_reason=termination_reason,
        ),
    )

    assert result.status == expected_status
    assert result.agent_exit_status == termination_reason
    assert agent_termination_reason(result) == termination_reason


def test_agent_result_builder_requires_or_validates_explicit_status() -> None:
    incomplete = build_t3_agent_result(mode='agent')

    assert incomplete.status == 'incomplete'
    assert incomplete.failure_message == 'agent returned no output'

    with pytest.raises(ValueError, match='status is required'):
        build_t3_agent_result(mode='oracle')

    with pytest.raises(ValueError, match='conflicts with Agent output status'):
        build_t3_agent_result(
            mode='agent',
            status='success',
            agent_output=SimpleNamespace(exit_status='tool_error'),
        )

    with pytest.raises(ValueError, match='agent_output is not valid'):
        build_t3_agent_result(
            mode='nop',
            status='skipped',
            agent_output=SimpleNamespace(exit_status='finished'),
        )


@pytest.mark.parametrize(
    ('status', 'agent_output', 'expected_reason'),
    [
        ('success', SimpleNamespace(exit_status='submitted'), 'ENV_DONE'),
        ('timeout', None, 'TASK_TIMEOUT'),
        ('incomplete', SimpleNamespace(exit_status='handoff'), 'ENV_DONE'),
        ('incomplete', SimpleNamespace(exit_status='max_iterations'), 'MAX_STEPS'),
        ('incomplete', SimpleNamespace(exit_status='truncation'), 'TRUNCATION'),
        ('error', SimpleNamespace(exit_status='tool_error'), 'FORMAT_ERROR'),
        (
            'error',
            SimpleNamespace(
                exit_status='runtime_disconnected',
                failure_kind='infrastructure',
                failure_stage='proxy',
            ),
            'INITIALIZATION_ERROR',
        ),
        (
            'error',
            SimpleNamespace(
                exit_status='runtime_error',
                failure_kind='infrastructure',
                failure_stage='execution_transport',
            ),
            'INITIALIZATION_ERROR',
        ),
        ('incomplete', SimpleNamespace(exit_status='future_status'), 'UNHANDLED_ERROR'),
    ],
)
def test_agent_termination_reason_is_resolved_from_normalized_result(
    status,
    agent_output,
    expected_reason,
) -> None:
    result = build_t3_agent_result(
        mode='agent',
        status=status,
        agent_output=agent_output,
    )

    assert agent_termination_reason(result) == expected_reason


@pytest.mark.parametrize(
    ('output', 'status'),
    [
        (SimpleNamespace(success=True, exit_code=0), 'success'),
        (SimpleNamespace(success=False, exit_code=1, stderr='failed'), 'error'),
        (SimpleNamespace(success=False, exit_code=-1, stderr='timed out', exit_status='timeout'), 'timeout'),
        (SimpleNamespace(success=False, exit_code=1, exit_status='truncation'), 'incomplete'),
        (SimpleNamespace(success=False, exit_code=1, exit_status='error'), 'error'),
        (SimpleNamespace(exit_status='timeout'), 'timeout'),
        (SimpleNamespace(exit_status='submitted'), 'success'),
        (SimpleNamespace(exit_status='handoff'), 'incomplete'),
        (SimpleNamespace(exit_status='max_iterations'), 'incomplete'),
        (SimpleNamespace(exit_status='no_tool_calls_incomplete'), 'incomplete'),
        (SimpleNamespace(exit_status='tool_error'), 'error'),
        (SimpleNamespace(exit_status='finished'), 'success'),
        (SimpleNamespace(exit_status='PROVIDER_ERROR'), 'error'),
        (SimpleNamespace(exit_status='FORMAT_ERROR'), 'error'),
        (SimpleNamespace(exit_status='ENV_ERROR'), 'error'),
        (SimpleNamespace(exit_status='output_length_exceeded'), 'incomplete'),
        (SimpleNamespace(exit_status='refused'), 'incomplete'),
        (SimpleNamespace(exit_status='future_status'), 'incomplete'),
        (SimpleNamespace(), 'incomplete'),
        (None, 'incomplete'),
    ],
)
def test_agent_output_status_is_not_assumed_success(output, status) -> None:
    actual, _error = classify_agent_output(output)

    assert actual == status


def test_handoff_is_a_known_routing_exit_without_failure_message() -> None:
    status, failure_message = classify_agent_output(SimpleNamespace(exit_status='handoff'))

    assert status == 'incomplete'
    assert failure_message is None


def test_hycode_general_error_exit_is_env_error() -> None:
    agent_output = SimpleNamespace(
        success=True,
        exit_code=1,
        stderr='',
        exit_status='general_error',
        termination_reason='ENV_DONE',
    )

    result = build_t3_agent_result(mode='agent', agent_output=agent_output)

    assert result.status == 'error'
    assert result.failure_kind is None
    assert result.failure_message == 'agent exit status: general_error'
    assert agent_termination_reason(result) == 'ENV_ERROR'


@pytest.mark.parametrize(
    ('exit_status', 'expected_status'),
    [
        ('timeout', 'timeout'),
        ('truncation', 'incomplete'),
        ('error', 'error'),
    ],
)
def test_failed_process_exit_status_reaches_t3_agent_result(exit_status: str, expected_status: str) -> None:
    agent_output = SimpleNamespace(
        success=False,
        exit_code=-1,
        exit_status=exit_status,
        failure_kind='agent',
    )

    status, failure_message = classify_agent_output(agent_output)
    result = build_t3_agent_result(
        mode='agent',
        status=status,
        agent_output=agent_output,
        failure_message=failure_message,
    )

    assert result.status == expected_status
    assert result.agent_exit_status == exit_status
    assert result.failure_kind == 'agent'


def test_agent_result_preserves_raw_exit_status() -> None:
    agent_output = SimpleNamespace(exit_status=TerminationReason.PROVIDER_ERROR)
    status, failure_message = classify_agent_output(agent_output)

    result = build_t3_agent_result(
        mode='agent',
        status=status,
        agent_output=agent_output,
        failure_message=failure_message,
    )

    assert result.status == 'error'
    assert result.agent_exit_status == 'PROVIDER_ERROR'
    assert result.failure_message == 'agent exit status: provider_error'


def test_agent_result_preserves_infrastructure_failure_kind() -> None:
    result = build_t3_agent_result(
        mode='agent',
        status='error',
        agent_output=SimpleNamespace(
            exit_status='provider_error',
            failure_kind='infrastructure',
            failure_stage='proxy',
        ),
        failure_message='provider unavailable',
    )

    assert result.failure_kind == 'infrastructure'
    assert result.failure_stage == 'proxy'


def test_agent_result_preserves_explicit_failure_stage_without_agent_output() -> None:
    result = build_t3_agent_result(
        mode='agent',
        status='error',
        failure_kind='infrastructure',
        failure_stage='proxy',
        failure_message='proxy unavailable',
    )

    assert result.failure_kind == 'infrastructure'
    assert result.failure_stage == 'proxy'
    assert result.failure_message == 'proxy unavailable'


@pytest.mark.parametrize(
    'exit_status',
    [
        'inference_error',
        'llm_error',
        'provider_error',
        'runtime_disconnected',
        'runtime_error',
    ],
)
def test_agent_result_infers_unambiguous_infrastructure_exit_statuses(exit_status: str) -> None:
    result = build_t3_agent_result(
        mode='agent',
        agent_output=SimpleNamespace(exit_status=exit_status),
        failure_message=f'agent exit status: {exit_status}',
    )

    assert result.failure_kind == 'infrastructure'


def test_inference_timeout_does_not_imply_infrastructure_without_explicit_evidence() -> None:
    timeout = build_t3_agent_result(
        mode='agent',
        status='timeout',
        agent_output=SimpleNamespace(exit_status='inference_timeout'),
    )
    infrastructure_timeout = build_t3_agent_result(
        mode='agent',
        status='timeout',
        agent_output=SimpleNamespace(
            exit_status='inference_timeout',
            failure_kind='infrastructure',
            failure_stage='proxy',
        ),
    )

    assert timeout.failure_kind is None
    assert agent_termination_reason(timeout) == 'TASK_TIMEOUT'
    assert infrastructure_timeout.failure_kind == 'infrastructure'
    assert agent_termination_reason(infrastructure_timeout) == 'INITIALIZATION_ERROR'


@pytest.mark.parametrize(
    'exit_status',
    ['context_length_exceeded', 'env_error', 'max_iterations', 'tool_error', 'unhandled_error'],
)
def test_agent_result_does_not_infer_ambiguous_or_model_exit_statuses(exit_status: str) -> None:
    result = build_t3_agent_result(
        mode='agent',
        agent_output=SimpleNamespace(exit_status=exit_status),
    )

    assert result.failure_kind is None


@pytest.mark.asyncio
async def test_agent_result_is_written_once_on_host_and_uploaded_to_standard_path(tmp_path) -> None:
    result = build_t3_agent_result(
        mode='agent',
        status='success',
        agent_output=SimpleNamespace(final_answer='answer', success=True),
    )
    trial_paths = SimpleNamespace(agent_dir=tmp_path / 'agent')
    host_path = write_t3_agent_result(trial_paths, result)
    runtime = _FakeRuntime()

    container_path = await upload_t3_agent_result(
        runtime,
        host_path,
        environment_os=TaskOS.LINUX,
    )

    assert container_path == '/logs/agent/result.json'
    assert json.loads(host_path.read_text())['final_answer'] == 'answer'
    (files, overwrite) = runtime.uploads[0]
    assert overwrite is True
    assert files[0].path == '/logs/agent/result.json'
    assert json.loads(files[0].content)['status'] == 'success'


@pytest.mark.asyncio
async def test_agent_result_upload_failure_is_not_ignored(tmp_path) -> None:
    result = build_t3_agent_result(mode='agent', status='success')
    host_path = write_t3_agent_result(SimpleNamespace(agent_dir=tmp_path / 'agent'), result)

    with pytest.raises(RuntimeError, match='Failed to upload T3 AgentResult'):
        await upload_t3_agent_result(
            _FakeRuntime(success=False),
            host_path,
            environment_os=TaskOS.LINUX,
        )


def test_verifier_env_injects_reserved_agent_result_path() -> None:
    env = verifier_env_with_t3_agent_result(
        {'JUDGE_MODEL': 'test'},
        environment_os=TaskOS.LINUX,
    )

    assert env == {
        'JUDGE_MODEL': 'test',
        T3_AGENT_RESULT_ENV: '/logs/agent/result.json',
    }


def test_verifier_env_rejects_task_override_of_agent_result_path() -> None:
    with pytest.raises(ValueError, match='reserved by Harbor Bridge'):
        verifier_env_with_t3_agent_result(
            {T3_AGENT_RESULT_ENV: '/app/fake-result.json'},
            environment_os=TaskOS.LINUX,
        )
