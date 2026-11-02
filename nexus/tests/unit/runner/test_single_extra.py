from __future__ import annotations

import io
import json
import sqlite3

import pytest

from nexus.cli.commands.export import write_run_to_jsonl
from nexus.distributed.batch.materialize import materialize_items
from nexus.distributed.batch.task_store import BatchTaskRecord, decode_task_record
from nexus.distributed.repositories.batch import _parse_task_list_record
from nexus.distributed.submission.input import normalize_run_input
from nexus.distributed.utils.identity import canonical_sha256
from nexus.runner.batch import BatchItem
from nexus.runner.batch.database import BatchDatabase
from nexus.runner.single import RunInput, RunOutput, _resolve_run_input_env_suffixes, run


def test_empty_extra_preserves_input_and_output_hashes():
    # The input baseline includes default runner fields, including task_files_materialization and use_gateway.
    assert (
        canonical_sha256(normalize_run_input({'task_name': 'echo'}))
        == 'f54bc569ac5300962fd6001b48043bee11db73ee249d14cc89efeea2ee6dd6b9'
    )
    assert (
        canonical_sha256(RunOutput().model_dump(mode='json'))
        == 'a6738574bbaebe996129ad21b1e04b14a1b12b317128115d89b20048f398854f'
    )
    assert normalize_run_input({'task_name': 'echo', 'extra': {}}) == normalize_run_input({'task_name': 'echo'})


def test_extra_is_finite_json_limited_by_utf8_bytes():
    limit = 2 * 1024**2
    value = {'x': 'a' * (limit - 8)}
    assert RunInput(task_name='echo', extra=value).extra == value
    with pytest.raises(ValueError, match='2 MiB'):
        RunInput(task_name='echo', extra={'x': value['x'] + '文'})
    for value in [None, [], 'text', {'x': float('nan')}, {'x': object()}]:
        with pytest.raises(ValueError, match='extra'):
            RunInput(task_name='echo', extra=value)


def test_extra_is_opaque_to_both_configuration_resolvers(monkeypatch):
    monkeypatch.setenv('EXTRA_SHOULD_NOT_EXPAND', 'expanded')
    extra = {
        'text': '${oc.env:EXTRA_SHOULD_NOT_EXPAND}',
        'nested': {'x__env': 'MISSING_EXTRA_ENV', 'ref': '${unknown}'},
    }
    config = RunInput.from_dict({'task_name': 'echo', 'extra': extra})
    assert config.extra == extra
    assert _resolve_run_input_env_suffixes(config).extra == extra
    assert RunInput.model_validate_json(config.model_dump_json()).extra == extra


@pytest.mark.parametrize('version', [1, 2, 3])
def test_existing_task_records_keep_extra_inside_run_input(tmp_path, version):
    config = RunInput(task_name='echo', extra={'text': '文' * 100_000})
    metadata = {'tag': 1} if version > 1 else {}
    path = tmp_path / 'tasks.jsonl'
    materialize_items([BatchItem(config=config, metadata=metadata)], path, task_list_version=version)
    record = _parse_task_list_record(path.read_bytes(), offset=0)
    identity = record.identity()
    assert identity['run_input']['extra'] == config.extra
    assert 'extra' not in identity
    assert record.metadata == metadata
    if version == 3:
        stored = BatchTaskRecord.from_v3_identity(
            parent_execution_id='parent', identity=identity, frame=None, max_record_bytes=8 * 1024**2
        )
    else:
        stored = BatchTaskRecord.from_identity(parent_execution_id='parent', identity=identity)
    assert decode_task_record(stored, expected_sha256=stored.record_sha256).run_input['extra'] == config.extra


async def test_local_storage_retry_and_export_reuse_config_without_metadata_changes(tmp_path):
    path = tmp_path / 'runs.db'
    extra = {'reference': 'long text'}
    metadata = {'extra': 'user metadata key', 'tag': 1}
    config = RunInput(task_name='echo', extra=extra)
    async with BatchDatabase(path) as db:
        await db.insert_run('case', config.model_dump(mode='json'), metadata)
    async with BatchDatabase(path) as db:
        record = await db.get_run('case')
        restored = RunInput.from_dict(record['config'])
        assert restored.extra == extra
        assert record['metadata'] == metadata
        output = io.StringIO()
        write_run_to_jsonl(record, output)
        exported = json.loads(output.getvalue())
        assert exported['extra'] == extra
        assert exported['metadata'] == metadata
    with sqlite3.connect(path) as conn:
        stored = conn.execute('SELECT config, metadata FROM runs').fetchone()
        assert json.loads(stored[0])['extra'] == extra
        assert json.loads(stored[1]) == metadata
        assert 'extra' not in {row[1] for row in conn.execute('PRAGMA table_info(runs)')}


async def test_initialization_failure_retains_extra_in_run_output():
    result = await run(RunInput(task_name='unused', imports=['missing_extra_test_module'], extra={'case': 1}))
    assert result.success is False
    assert result.extra == {'case': 1}
    assert result.model_dump(mode='json')['extra'] == {'case': 1}
