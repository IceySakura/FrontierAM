from __future__ import annotations

import hashlib
import json

import pytest

from tests.e2e.distributed.compare_cloud_results import fingerprint, fingerprint_streaming


def test_checksum_ignores_only_top_level_lifecycle_times(tmp_path):
    reference = tmp_path / 'reference.jsonl'
    candidate = tmp_path / 'candidate.jsonl'
    row = {'run_id': 'one', 'start_time': 'old', 'end_time': 'old', 'output': {'time_unix_nano': 123}}
    reference.write_text(json.dumps(row) + '\n')
    candidate.write_text(json.dumps({**row, 'start_time': 'new', 'end_time': 'new'}, indent=None) + '\n')
    assert fingerprint(reference)['md5'] == fingerprint(candidate)['md5']
    row['output']['time_unix_nano'] = 124
    candidate.write_text(json.dumps(row) + '\n')
    assert fingerprint(reference)['md5'] != fingerprint(candidate)['md5']


@pytest.mark.parametrize('change', ['row_order', 'field_order', 'nested_field_order', 'array_order', 'extra_field'])
def test_checksum_detects_order_and_schema_changes(tmp_path, change):
    reference = tmp_path / 'reference.jsonl'
    candidate = tmp_path / 'candidate.jsonl'
    rows = [{'run_id': 'one', 'output': {'a': 1, 'b': [2, 3]}}, {'run_id': 'two', 'output': {}}]
    reference.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    if change == 'row_order':
        rows.reverse()
    elif change == 'field_order':
        rows[0] = dict(reversed(list(rows[0].items())))
    elif change == 'nested_field_order':
        rows[0]['output'] = dict(reversed(list(rows[0]['output'].items())))
    elif change == 'array_order':
        rows[0]['output']['b'].reverse()
    else:
        rows[0]['extra'] = {}
    candidate.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert fingerprint(reference)['md5'] != fingerprint(candidate)['md5']


@pytest.mark.parametrize('timestamp', [None, '2026-09-28T00:00:00+00:00'])
def test_streaming_checksum_removes_only_header_times_across_large_chunks(tmp_path, timestamp):
    path = tmp_path / 'results.jsonl'
    row = {
        'run_id': '中文 "quoted"',
        'status': 'success',
        'start_time': timestamp,
        'end_time': timestamp,
        'retry_count': 0,
        'output': {'start_time': 'must keep', 'time_unix_nano': 123, 'text': 'x' * (9 * 1024 * 1024)},
    }
    path.write_text((json.dumps(row, ensure_ascii=False) + '\n') * 2)
    row.pop('start_time')
    row.pop('end_time')
    expected = (json.dumps(row, ensure_ascii=False) + '\n').encode() * 2
    result = fingerprint_streaming(path)
    assert result['md5'] == hashlib.md5(expected).hexdigest()
    assert result['count'] == 2
    assert result['rows'][0]['run_id'] == row['run_id']


def test_streaming_checksum_rejects_nonstandard_header(tmp_path):
    path = tmp_path / 'results.jsonl'
    path.write_text('{"run_id":"one"}\n')
    with pytest.raises(ValueError, match='standard exporter'):
        fingerprint_streaming(path)
