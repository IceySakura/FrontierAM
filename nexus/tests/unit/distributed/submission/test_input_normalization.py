from __future__ import annotations

import json

import pytest

from pydantic import ValidationError

from nexus.distributed.errors import DomainValidationError
from nexus.distributed.repositories.batch import _parse_task_list_record
from nexus.distributed.submission.specs import LiveSubmission, SingleSubmission


def _predefined_submission(**values):
    return {
        'submission_version': 1,
        'idempotency_key': 'input-normalization',
        'environment_id': 'environment-1',
        'code': {'mode': 'config_only'},
        'pool': 'execution',
        **values,
    }


def test_single_submission_normalizes_run_input_without_resolving_worker_environment(monkeypatch):
    monkeypatch.setenv('NEXUS_SUBMISSION_TEST_SECRET', 'must-not-be-read')
    placeholder = '${oc.env:NEXUS_SUBMISSION_TEST_SECRET}'

    submission = SingleSubmission.model_validate(
        _predefined_submission(
            run_input={
                'task_name': 'identity',
                'task_config': {'secret': placeholder},
                'unknown_top_level_field': 'discarded',
            }
        )
    )

    assert submission.run_input['task_config']['secret'] == placeholder
    assert submission.run_input['runner']['hold_signal_handlers'] is True
    assert 'unknown_top_level_field' not in submission.run_input
    assert 'compaction_manager' not in submission.run_input


def test_live_submission_normalizes_omitted_live_defaults():
    submission = LiveSubmission.model_validate(_predefined_submission(run_input={'task_name': 'identity'}))

    assert submission.live == {'working_dir': '/workspace', 'cache': None}


@pytest.mark.parametrize(
    'submission',
    [
        _predefined_submission(run_input={}),
        _predefined_submission(run_input={'task_name': 'identity'}, live={'cache': 'not-an-object'}),
    ],
)
def test_single_and_live_reject_invalid_normalized_input(submission):
    model = LiveSubmission if 'live' in submission else SingleSubmission

    with pytest.raises(ValidationError, match='invalid_(run_input|live_config)'):
        model.model_validate(submission)


def test_task_list_record_uses_the_same_run_input_normalization():
    placeholder = '${oc.env:TASK_LIST_WORKER_SECRET}'
    line = (
        json.dumps(
            {
                'version': 1,
                'item_id': 'item-1',
                'run_input': {
                    'task_name': 'identity',
                    'task_config': {'secret': placeholder},
                    'unknown_top_level_field': 'discarded',
                },
            }
        ).encode()
        + b'\n'
    )

    record = _parse_task_list_record(line, offset=0)

    assert record.item_id == 'item-1'
    assert record.metadata == {}
    assert record.run_input['task_config']['secret'] == placeholder
    assert 'unknown_top_level_field' not in record.run_input
    assert 'compaction_manager' not in record.run_input


def test_task_list_v2_record_validates_and_preserves_metadata():
    metadata = {'category': 'terminal', 'tags': ['dataset']}
    line = (
        json.dumps(
            {
                'version': 2,
                'item_id': 'item-1',
                'run_input': {'task_name': 'identity'},
                'metadata': metadata,
            }
        ).encode()
        + b'\n'
    )

    record = _parse_task_list_record(line, offset=0)

    assert record.version == 2
    assert record.item_id == 'item-1'
    assert record.metadata == metadata


def test_task_list_v2_record_rejects_oversized_metadata():
    line = (
        json.dumps(
            {
                'version': 2,
                'item_id': 'item-1',
                'run_input': {'task_name': 'identity'},
                'metadata': {'value': 'x' * (128 * 1024)},
            }
        ).encode()
        + b'\n'
    )

    with pytest.raises(DomainValidationError) as exc_info:
        _parse_task_list_record(line, offset=31)

    assert exc_info.value.code == 'invalid_task_list_metadata'
    assert exc_info.value.detail == 'offset=31'


def test_task_list_record_rejects_invalid_run_input_during_materialization():
    line = b'{"version":1,"item_id":"item-1","run_input":{}}\n'

    with pytest.raises(DomainValidationError) as exc_info:
        _parse_task_list_record(line, offset=17)

    assert exc_info.value.code == 'invalid_run_input'
    assert exc_info.value.detail == 'offset=17'


@pytest.mark.parametrize('item_id', ['Item', 'item ', 'café', 'cafe\u0301', 'a' * 65])
def test_task_list_record_rejects_noncanonical_item_id(item_id):
    line = (
        json.dumps(
            {
                'version': 1,
                'item_id': item_id,
                'run_input': {'task_name': 'identity'},
            }
        ).encode()
        + b'\n'
    )

    with pytest.raises(DomainValidationError) as exc_info:
        _parse_task_list_record(line, offset=23)

    assert exc_info.value.code == 'invalid_task_list_item'
    assert exc_info.value.detail == 'offset=23'
