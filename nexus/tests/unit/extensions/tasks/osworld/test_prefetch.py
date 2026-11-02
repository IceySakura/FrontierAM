"""OSWorld prefetch regressions."""

from __future__ import annotations

import json
import logging

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.dataset.exceptions import DatasetAuthenticationError
from nexus.extensions.tasks.os_world.data import prefetch
from nexus.extensions.tasks.os_world.execution.setup import _download_cache_path
from nexus.extensions.tasks.os_world.scripts.scan_declared_assets import scan


@pytest.fixture
def files(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=prefetch.__name__)
    files = SimpleNamespace(list=AsyncMock(return_value=[]), download=AsyncMock())
    monkeypatch.setattr(prefetch, 'require_context', lambda: SimpleNamespace(task_files=files))
    return files


async def test_task_json_only_is_not_a_download_failure(files, caplog, tmp_path):
    await prefetch.prefetch_task_files(
        tmp_path,
        tmp_path,
        task_data={
            'config': [{'type': 'launch', 'parameters': {'command': ['google-chrome']}}],
            'evaluator': {'result': {'type': 'default_search_engine'}, 'expected': {'type': 'rule'}},
        },
    )
    files.download.assert_not_called()
    assert caplog.text.count('status=empty') == 2
    assert 'declared=0' in caplog.text
    assert 'task_files may still contain task.json' in caplog.text
    assert 'falling back' not in caplog.text
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


async def test_uninspected_task_cannot_claim_no_required_assets(files, caplog, tmp_path):
    await prefetch.prefetch_task_files(tmp_path, tmp_path)
    assert 'declared=unknown' in caplog.text


async def test_partially_packaged_assets_are_reported_missing(files, caplog, tmp_path):
    data = {
        'evaluator': {
            'expected': {
                'type': 'cloud_file',
                'path': ['https://test/a', 'https://test/b'],
                'dest': ['a', 'b'],
            }
        }
    }
    files.list.side_effect = [[], ['evaluator/a']]
    await prefetch.prefetch_task_files(tmp_path, tmp_path, task_data=data)
    assert 'status=missing_declared pattern=evaluator/** missing=1 packaged=1' in caplog.text
    files.download.assert_awaited_once_with('evaluator/**', str(tmp_path))
    assert 'status=downloaded' in caplog.text


async def test_missing_setup_and_postconfig_and_local_reference_are_counted(files, caplog, tmp_path):
    action = {'type': 'download', 'parameters': {'files': [{'url': 'https://test/a', 'path': '/tmp/a'}]}}
    data = {
        'config': [action],
        'evaluator': {
            'postconfig': [action],
            'expected': {'type': 'local_file', 'path': 'repo/gold.txt', 'dest': 'gold.txt'},
        },
    }
    expected = prefetch._declared_assets(data)
    name = _download_cache_path(tmp_path, 'https://test/a', '/tmp/a').name
    assert expected == {'setup': {f'setup/{name}'}, 'evaluator': {f'evaluator/{name}', 'evaluator/gold.txt'}}
    await prefetch.prefetch_task_files(tmp_path, tmp_path, task_data=data)
    assert 'pattern=setup/** missing=1' in caplog.text
    assert 'pattern=evaluator/** missing=2' in caplog.text
    assert 'status=empty' not in caplog.text
    files.download.assert_not_called()


@pytest.mark.parametrize('stage', ['list', 'download'])
@pytest.mark.parametrize('error', [DatasetAuthenticationError('secret signed-url'), OSError('secret signed-url')])
async def test_failures_are_separate_from_empty_and_do_not_leak_errors(files, caplog, tmp_path, stage, error):
    files.list.return_value = ['setup/asset']
    getattr(files, stage).side_effect = error
    await prefetch.prefetch_task_files(tmp_path, tmp_path)
    assert f'stage={stage}' in caplog.text
    assert 'status=failed' in caplog.text
    assert (
        'category=authentication' in caplog.text
        if isinstance(error, DatasetAuthenticationError)
        else ('category=io' in caplog.text)
    )
    assert 'secret' not in caplog.text
    assert 'signed-url' not in caplog.text
    assert 'status=empty' not in caplog.text
    assert 'falling back' not in caplog.text


async def test_no_task_files_is_explicitly_not_attached(monkeypatch, caplog, tmp_path):
    caplog.set_level(logging.DEBUG, logger=prefetch.__name__)
    monkeypatch.setattr(prefetch, 'require_context', lambda: SimpleNamespace(task_files=None))
    await prefetch.prefetch_task_files(tmp_path, tmp_path)
    assert 'status=not_attached' in caplog.text


@pytest.mark.parametrize('rule_type', [{'method': 'eq', 'ref': 'list'}, [], None, 1, True])
def test_non_string_rule_type_does_not_hide_nested_or_sibling_assets(rule_type):
    data = {
        'evaluator': {
            'options': {
                'rules': [
                    {
                        'type': rule_type,
                        'nested': {
                            'type': 'local_file',
                            'path': 'fixtures/reference.txt',
                            'dest': 'reference.txt',
                        },
                    }
                ]
            },
            'expected': {'type': 'cloud_file', 'path': 'https://test/gold.xlsx', 'dest': 'gold.xlsx'},
        },
    }
    assert prefetch._declared_assets(data) == {
        'setup': set(),
        'evaluator': {'evaluator/reference.txt', 'evaluator/gold.xlsx'},
    }


async def test_calc_data_validation_rule_prefetch_keeps_expected_file(files, caplog, tmp_path):
    data = {
        'evaluator': {
            'func': 'compare_table',
            'options': {
                'rules': [
                    {
                        'type': 'data_validation',
                        'dv_props': [{'type': {'method': 'eq', 'ref': 'list'}}],
                    }
                ]
            },
            'expected': {'type': 'cloud_file', 'path': 'https://test/gold.xlsx', 'dest': 'gold.xlsx'},
        },
    }
    files.list.side_effect = [[], ['evaluator/gold.xlsx']]
    await prefetch.prefetch_task_files(tmp_path, tmp_path, task_data=data)
    files.download.assert_awaited_once_with('evaluator/**', str(tmp_path))
    assert 'status=downloaded pattern=evaluator/**' in caplog.text
    assert 'status=missing_declared' not in caplog.text
    assert 'status=failed' not in caplog.text


def test_nested_file_inside_mapping_type_is_still_discovered():
    data = {
        'evaluator': {
            'type': {
                'nested': {'type': 'cloud_file', 'path': 'https://test/a', 'dest': 'a'},
            }
        }
    }
    assert prefetch._declared_assets(data) == {'setup': set(), 'evaluator': {'evaluator/a'}}


def prepare(tmp_path, task_id='task-a'):
    root = tmp_path / 'cache'
    task = root / task_id / 'task.json'
    task.parent.mkdir(parents=True)
    task.write_text(
        json.dumps(
            {
                'id': task_id,
                'instruction': 'test instruction',
                'evaluator': {'expected': {'type': 'cloud_file', 'path': 'https://test/a', 'dest': 'a'}},
            }
        )
    )
    cases = tmp_path / 'cases.json'
    case = {
        'question_id': 1,
        'payload': json.dumps(
            {
                'meta_data_': {'instruction': 'test instruction'},
                'nexus_request': {'payload': {'run_input': {'task_config': {'task_id': task_id}}}},
            }
        ),
    }
    cases.write_text(json.dumps({'data': {'total': 1, 'truncated': False, 'cases': [case]}}))
    return root, cases, task


def test_scan_cached_definition_without_network(tmp_path, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError('Offline scan must not access the network')

    monkeypatch.setattr(socket, 'create_connection', forbidden)
    root, cases, _ = prepare(tmp_path)
    report = scan(root, cases)
    assert (report['total'], report['passed'], report['failed']) == (1, 1, 0)
    assert report['tasks'][0]['declared'] == {'setup': [], 'evaluator': ['evaluator/a']}


@pytest.mark.parametrize('change', ['missing', 'wrong_id', 'wrong_instruction', 'malformed_json'])
def test_scan_reports_bad_cached_definitions(tmp_path, change):
    root, cases, task = prepare(tmp_path)
    if change == 'missing':
        task.unlink()
    elif change == 'malformed_json':
        task.write_text('{')
    else:
        data = json.loads(task.read_text())
        data['id' if change == 'wrong_id' else 'instruction'] = 'different'
        task.write_text(json.dumps(data))
    report = scan(root, cases)
    assert report['failed'] == 1
    assert report['tasks'][0]['status'] == 'error'


@pytest.mark.parametrize('change', ['truncated', 'missing_case', 'duplicate', 'unsafe_id'])
def test_scan_rejects_incomplete_or_ambiguous_membership(tmp_path, change):
    root, cases, _ = prepare(tmp_path)
    envelope = json.loads(cases.read_text())
    data = envelope['data']
    if change == 'truncated':
        data['truncated'] = True
    elif change == 'missing_case':
        data['total'] = 2
    elif change == 'duplicate':
        data['total'] = 2
        data['cases'] *= 2
    else:
        payload = json.loads(data['cases'][0]['payload'])
        payload['nexus_request']['payload']['run_input']['task_config']['task_id'] = '../escape'
        data['cases'][0]['payload'] = payload
    cases.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='complete|Duplicate|single directory'):
        scan(root, cases)
