import gzip
import json

import pytest

from aiohttp.test_utils import TestClient, TestServer
from tests.unit.dashboard.trajectory_fixtures import branched_trajectory, write_trajectory

from nexus.dashboard.server import create_app
from nexus.dashboard.store import ArtifactStore


def trajectory(session='sample'):
    data = branched_trajectory().dump()
    data['session_id'] = session
    data['node_payloads'] = {'2': {'token_ids': [1, 2, 3]}}
    return data


def run_result(gateway):
    return {
        'task_output': {'score': 1},
        'agent_output': {'final_answer': 'done'},
        'processor_results': {'gateway': gateway, 'other': {'unrelated': True}},
        'success': True,
        'error': None,
    }


@pytest.mark.parametrize('compressed', [False, True])
@pytest.mark.parametrize('multiple', [False, True])
def test_run_json_supports_single_and_multiple_clients(tmp_path, compressed, multiple):
    first = trajectory()
    gateway = {'a': first, 'b': trajectory()} if multiple else first
    raw = json.dumps(run_result(gateway)).encode()
    original_bytes = gzip.compress(raw) if compressed else raw
    path = tmp_path / ('result.json.gz' if compressed else 'result.json')
    path.write_bytes(original_bytes)
    store = ArtifactStore(path)
    catalog = store.scan()
    assert catalog['errors'] == []
    assert len(catalog['items']) == (2 if multiple else 1)
    assert len({item['id'] for item in catalog['items']}) == len(catalog['items'])
    if multiple:
        assert {item['client'] for item in catalog['items']} == {'a', 'b'}
    for item in catalog['items']:
        loaded = store.load(item['id'])
        assert loaded['trajectory'] == first
        assert loaded['source']['filename'] == path.name
        assert loaded['run'] == {'success': True, 'error': None}
        assert loaded['export'] == {}
    assert path.read_bytes() == original_bytes


def test_bad_or_missing_client_is_isolated(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text(
        json.dumps(run_result({'good': trajectory(), 'bad': {'schema_version': 'unsupported'}, 'none': None}))
    )
    store = ArtifactStore(path)
    catalog = store.scan()
    assert len(catalog['items']) == 1
    assert catalog['items'][0]['client'] == 'good'
    assert {error['client'] for error in catalog['errors']} == {'bad', 'none'}
    assert store.load(catalog['items'][0]['id'])['trajectory']['session_id'] == 'sample'


@pytest.mark.parametrize('gateway', [None, {}, []])
def test_missing_gateway_is_reported_without_fabricating_trajectory(tmp_path, gateway):
    path = tmp_path / 'failed.json'
    data = run_result(gateway)
    data.update(success=False, error='execution failed')
    path.write_text(json.dumps(data))
    result = ArtifactStore(path).scan()
    assert result['items'] == []
    assert 'no Gateway trajectory' in result['errors'][0]['error']


def test_run_json_does_not_search_arbitrary_nested_fields(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps({'task_output': {'trajectory': trajectory()}}))
    result = ArtifactStore(path).scan()
    assert result['items'] == []
    assert result['errors']


def test_directory_supports_runs_and_native_artifacts_without_summary_duplicates(tmp_path):
    write_trajectory(str(tmp_path))
    (tmp_path / 'run.json').write_text(json.dumps(run_result(trajectory())))
    (tmp_path / 'native.json').write_text(json.dumps(trajectory('native')))
    (tmp_path / 'subdir').mkdir()
    (tmp_path / 'subdir' / 'nested.json').write_text(json.dumps(trajectory('nested')))
    catalog = ArtifactStore(tmp_path).scan()
    assert len(catalog['items']) == 3
    assert catalog['errors'] == []


def test_replaced_json_requires_refresh(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(run_result(trajectory('before'))))
    store = ArtifactStore(path)
    original = store.scan()['items'][0]
    path.write_text(json.dumps(run_result(trajectory('after'))))
    with pytest.raises(ValueError, match='refresh'):
        store.load(original['id'])
    current = store.scan()['items'][0]
    assert current['id'] == original['id']
    assert store.load(current['id'])['trajectory']['session_id'] == 'after'


@pytest.mark.asyncio
async def test_run_result_detail_still_projects_segments(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(run_result({'main-client': trajectory()})))
    async with TestClient(TestServer(create_app(path))) as client:
        catalog = await (await client.get('/api/trajectories')).json()
        (item,) = catalog['items']
        response = await client.get('/api/trajectories/' + item['id'])
        assert response.status == 200
        detail = await response.json()
        assert detail['graph']['segments']
        assert detail['source']['client'] == 'main-client'
        assert detail['trajectory']['node_payloads'] == {'2': {'token_ids': [1, 2, 3]}}


@pytest.mark.parametrize('multiple', [False, True])
def test_actual_single_writer_with_custom_processor_name(tmp_path, multiple):
    from nexus.runner.single import RunOutput, write_run_output

    data = trajectory('single-writer')
    path = tmp_path / 'result.json'
    write_run_output(path, RunOutput(processor_results={'custom-trace': {'llm': data} if multiple else data}))
    store = ArtifactStore(path)
    result = store.scan()
    assert result['errors'] == []
    (item,) = result['items']
    assert item['record_path'] == 'processor_results.custom-trace' + ('.llm' if multiple else '')
    assert store.load(item['id'])['trajectory'] == data


def test_actual_batch_writer_retains_failed_partial_results_and_run_id(tmp_path):
    from nexus.cli.commands.export import write_run_to_jsonl

    data = trajectory('partial')
    path = tmp_path / 'results.jsonl'
    with path.open('w') as output:
        write_run_to_jsonl(
            {
                'run_id': 'failed-run',
                'status': 'failed',
                'error': 'task failed',
                'processor_results': {'gateway': {'client': data}},
            },
            output,
        )
        write_run_to_jsonl({'run_id': 'no-trajectory', 'status': 'failed'}, output)
    store = ArtifactStore(path)
    catalog = store.scan()
    (item,) = catalog['items']
    assert item['run_id'] == 'failed-run'
    assert item['line'] == 1
    assert catalog['errors'][0]['run_id'] == 'no-trajectory'
    assert catalog['errors'][0]['line'] == 2
    detail = store.load(item['id'])
    assert detail['trajectory'] == data
    assert detail['run']['status'] == 'failed'
    assert detail['run']['error'] == 'task failed'
    assert detail['source']['run_id'] == 'failed-run'


def test_multiple_processors_keep_distinct_ids(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text(
        json.dumps(
            {
                'processor_results': {
                    'gateway': trajectory(),
                    'custom': trajectory(),
                    'counter': {'schema_version': 'counter-v1', 'calls': 2},
                }
            }
        )
    )
    result = ArtifactStore(path).scan()
    assert result['errors'] == []
    assert len({item['id'] for item in result['items']}) == 2
    assert {item['record_path'] for item in result['items']} == {
        'processor_results.gateway',
        'processor_results.custom',
    }
