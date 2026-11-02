from __future__ import annotations

import json
import logging

import pytest

from infra.logging import bind_log_context, configure_logging, get_log_context


@pytest.fixture(autouse=True)
def _reset_logging():
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    yield
    root.handlers[:] = old_handlers
    root.setLevel(old_level)


def _read_json_lines(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_json_logging_includes_service_context_and_extra(capsys) -> None:
    configure_logging('test-service', level='INFO')

    with bind_log_context(request_id='req-1'):
        logging.getLogger('test').info('hello', extra={'execution_id': 'exe-1'})

    payload = _read_json_lines(capsys)[-1]
    assert payload['service'] == 'test-service'
    assert payload['level'] == 'INFO'
    assert payload['logger'] == 'test'
    assert payload['message'] == 'hello'
    assert payload['request_id'] == 'req-1'
    assert payload['extra']['execution_id'] == 'exe-1'
    assert 'timestamp' in payload


def test_text_logging_can_be_enabled(capsys) -> None:
    configure_logging('test-service', json_output=False)
    logging.getLogger('test').warning('hello')
    assert 'WARNING | test | hello' in capsys.readouterr().out


def test_level_filtering_drops_below_threshold(capsys) -> None:
    configure_logging('test-service', level='WARNING')

    logging.getLogger('test').info('dropped')
    logging.getLogger('test').warning('kept')

    lines = _read_json_lines(capsys)
    messages = [entry['message'] for entry in lines]
    assert 'dropped' not in messages
    assert 'kept' in messages


def test_invalid_level_raises() -> None:
    with pytest.raises(ValueError, match='invalid log level'):
        configure_logging('test-service', level='NOPE')


def test_exception_is_serialized(capsys) -> None:
    configure_logging('test-service')

    try:
        raise RuntimeError('boom')
    except RuntimeError:
        logging.getLogger('test').exception('failed')

    payload = _read_json_lines(capsys)[-1]
    assert payload['level'] == 'ERROR'
    assert 'RuntimeError: boom' in payload['exception']


def test_context_is_isolated_after_exit(capsys) -> None:
    configure_logging('test-service')

    with bind_log_context(request_id='req-1'):
        assert get_log_context()['request_id'] == 'req-1'
    assert 'request_id' not in get_log_context()

    logging.getLogger('test').info('after')
    payload = _read_json_lines(capsys)[-1]
    assert 'request_id' not in payload


def test_nested_context_merges_and_restores() -> None:
    configure_logging('test-service')

    with bind_log_context(request_id='req-1'):
        with bind_log_context(execution_id='exe-1'):
            ctx = get_log_context()
            assert ctx == {'request_id': 'req-1', 'execution_id': 'exe-1'}
        assert get_log_context() == {'request_id': 'req-1'}


def test_none_context_values_are_skipped() -> None:
    configure_logging('test-service')

    with bind_log_context(request_id='req-1', tenant=None):
        assert get_log_context() == {'request_id': 'req-1'}


def test_non_serializable_extra_is_stringified(capsys) -> None:
    configure_logging('test-service')

    logging.getLogger('test').info('obj', extra={'payload': object()})

    payload = _read_json_lines(capsys)[-1]
    assert isinstance(payload['extra']['payload'], str)


def test_uvicorn_loggers_route_through_json_formatter(capsys) -> None:
    configure_logging('test-service')

    for name in ('uvicorn', 'uvicorn.error', 'uvicorn.access'):
        assert logging.getLogger(name).handlers == []
        assert logging.getLogger(name).propagate is True

    logging.getLogger('uvicorn.error').info('startup complete')
    payload = _read_json_lines(capsys)[-1]
    assert payload['service'] == 'test-service'
    assert payload['logger'] == 'uvicorn.error'
    assert payload['message'] == 'startup complete'
