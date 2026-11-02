import json

from nexus.extensions.agents.workbuddy.acp_output import (
    assistant_text,
    classify_wb_result,
    conversation_id,
    failure_fields,
    final_turn_result,
    parse_wb_json_lines,
    task_outcome,
    terminal_error,
    transcript_path,
)


SESSION = '51422d18-0000-4000-8000-000000000000'
TRANSCRIPT = f'/root/.workbuddy/projects/ws/{SESSION}.jsonl'


def _stdout(*lines: object) -> str:
    return '\n'.join(line if isinstance(line, str) else json.dumps(line) for line in lines)


def _turn(outcome: str = 'SUCCESS', stop_reason: str = 'end_turn', **meta: object) -> dict:
    return {
        'stopReason': stop_reason,
        '_meta': {
            'codebuddy.ai/outcome': outcome,
            'codebuddy.ai/conversationId': SESSION,
            'codebuddy.ai/transcriptPath': TRANSCRIPT,
            **meta,
        },
    }


def test_parse_splits_updates_from_turn_results_and_skips_noise() -> None:
    stdout = _stdout(
        {'sessionId': SESSION, 'update': {'sessionUpdate': 'user_message_chunk'}},
        'npm WARN something a tool printed on stdout',
        {'sessionId': SESSION, 'method': '_codebuddy.ai/command', 'params': {}},
        {'sessionId': SESSION, 'channel': 'docs:selectionChanged', 'data': {}},
        '',
        '[1, 2, 3]',
        _turn(),
    )
    updates, results = parse_wb_json_lines(stdout)
    assert len(updates) == 3
    assert len(results) == 1
    assert final_turn_result(results) == results[0]


def test_parse_tolerates_empty_stdout() -> None:
    assert parse_wb_json_lines('') == ([], [])
    assert final_turn_result([]) is None


def test_assistant_text_keeps_only_visible_reply() -> None:
    updates, _ = parse_wb_json_lines(
        _stdout(
            {
                'sessionId': SESSION,
                'update': {'sessionUpdate': 'user_message_chunk', 'content': {'type': 'text', 'text': 'IGNORE ME'}},
            },
            {
                'sessionId': SESSION,
                'update': {'sessionUpdate': 'agent_thought_chunk', 'content': {'type': 'text', 'text': 'thinking...'}},
            },
            {
                'sessionId': SESSION,
                'update': {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': 'po'}},
            },
            {'sessionId': SESSION, 'update': {'sessionUpdate': 'tool_call', 'title': 'Write'}},
            {
                'sessionId': SESSION,
                'update': {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': 'ng'}},
            },
        )
    )
    assert assistant_text(updates) == 'pong'


def test_reads_the_bare_task_outcome_key_a_real_run_emits() -> None:
    # Verbatim _meta from a 5.7.0 gongfeng run: the outcome ships as a bare
    # "taskOutcome", not as "codebuddy.ai/outcome".
    result = {
        'stopReason': 'end_turn',
        '_meta': {
            'taskOutcome': 'SUCCESS',
            'codebuddy.ai/conversationId': 'b7349900-70b2-4271-85ab-6536ef42a756',
            'codebuddy.ai/transcriptPath': '/root/.workbuddy/projects/workspace/b7349900.jsonl',
        },
    }
    assert task_outcome(result) == 'SUCCESS'
    assert conversation_id(result) == 'b7349900-70b2-4271-85ab-6536ef42a756'
    assert classify_wb_result(result, 0, json_output=True) == ('completed', None)


def test_bare_task_outcome_still_catches_a_zero_exit_failure() -> None:
    # The whole point of parsing: wb exits 0 here, only _meta says it failed.
    result = {'stopReason': 'end_turn', '_meta': {'taskOutcome': 'FAILED_TOOL_EXECUTION'}}
    status, _detail = classify_wb_result(result, 0, json_output=True)
    assert status == 'error'
    assert failure_fields(status, task_outcome(result)) == ('agent', 'process')


def test_meta_readers_accept_the_nested_and_bare_spellings() -> None:
    assert task_outcome(_turn()) == 'SUCCESS'
    assert transcript_path(_turn()) == TRANSCRIPT
    assert conversation_id(_turn()) == SESSION

    nested = {'stopReason': 'end_turn', '_meta': {'codebuddy.ai': {'outcome': 'PARTIAL_SUCCESS'}}}
    assert task_outcome(nested) == 'PARTIAL_SUCCESS'

    bare = {'stopReason': 'end_turn', '_meta': {'outcome': 'CANCELLED'}}
    assert task_outcome(bare) == 'CANCELLED'

    assert task_outcome(None) is None
    assert task_outcome({'stopReason': 'end_turn'}) is None


def test_terminal_error_stringifies_structured_payloads() -> None:
    assert terminal_error(_turn()) is None
    assert terminal_error(_turn(**{'codebuddy.ai/terminalError': 'boom'})) == 'boom'
    structured = _turn(**{'codebuddy.ai/terminalError': {'code': 'E1'}})
    assert terminal_error(structured) == '{"code": "E1"}'


def test_success_is_the_only_completed_outcome() -> None:
    status, detail = classify_wb_result(_turn('SUCCESS'), 0, json_output=True)
    assert status == 'completed'
    assert detail is None
    assert failure_fields(status, 'SUCCESS') == (None, None)


def test_zero_exit_with_a_failing_outcome_is_not_success() -> None:
    # wb exits 0 whenever the turn completed, including when it completed badly.
    for outcome, expected in [
        ('PARTIAL_SUCCESS', 'partial'),
        ('FAILED_TOOL_EXECUTION', 'error'),
        ('FAILED_AGENT_LOOP', 'error'),
        ('REFUSED_OR_BLOCKED', 'refused'),
        ('PERMISSION_DENIED', 'refused'),
        ('CANCELLED', 'cancelled'),
        ('UNKNOWN', 'error'),
    ]:
        status, detail = classify_wb_result(_turn(outcome), 0, json_output=True)
        assert status == expected, outcome
        assert detail is not None
        kind, _stage = failure_fields(status, outcome)
        assert kind == 'agent'


def test_model_request_failure_is_attributed_to_infrastructure() -> None:
    status, _ = classify_wb_result(_turn('FAILED_MODEL_REQUEST'), 0, json_output=True)
    assert status == 'error'
    assert failure_fields(status, 'FAILED_MODEL_REQUEST') == ('infrastructure', 'model_request')


def test_stop_reason_drives_truncation_and_limits() -> None:
    status, _ = classify_wb_result(_turn('UNKNOWN', 'max_tokens'), 0, json_output=True)
    assert status == 'truncation'
    assert failure_fields(status, None) == ('agent', 'context_overflow')

    status, _ = classify_wb_result(_turn('UNKNOWN', 'max_turn_requests'), 0, json_output=True)
    assert status == 'error'


def test_terminal_error_surfaces_as_the_detail() -> None:
    result = _turn('FAILED_TOOL_EXECUTION', **{'codebuddy.ai/terminalError': 'disk full'})
    status, detail = classify_wb_result(result, 0, json_output=True)
    assert status == 'error'
    assert detail == 'disk full'


def test_exit_codes_map_to_documented_meanings() -> None:
    assert classify_wb_result(None, 130, json_output=True)[0] == 'cancelled'
    assert classify_wb_result(None, 143, json_output=True)[0] == 'timeout'
    assert classify_wb_result(None, 2, json_output=True)[0] == 'error'
    assert classify_wb_result(None, -1, json_output=True)[0] == 'timeout'


def test_missing_result_line_is_an_error_not_a_success() -> None:
    status, detail = classify_wb_result(None, 0, json_output=True)
    assert status == 'error'
    assert 'no turn result' in (detail or '')


def test_without_json_the_exit_code_is_all_there_is() -> None:
    assert classify_wb_result(None, 0, json_output=False) == ('completed', None)
    status, detail = classify_wb_result(None, 1, json_output=False)
    assert status == 'error'
    assert 'exited with code 1' in (detail or '')


def test_unrecognized_outcome_degrades_to_error() -> None:
    status, detail = classify_wb_result(_turn('SOMETHING_NEW'), 0, json_output=True)
    assert status == 'error'
    assert 'SOMETHING_NEW' in (detail or '')
