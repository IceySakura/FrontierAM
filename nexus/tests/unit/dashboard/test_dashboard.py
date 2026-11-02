import argparse
import gzip
import json

import pytest

from aiohttp.test_utils import TestClient, TestServer
from tests.unit.dashboard.trajectory_fixtures import branched_trajectory, write_trajectory

from nexus.cli.commands.dashboard import register_dashboard_parser
from nexus.dashboard.projection import project_trajectory
from nexus.dashboard.server import create_app
from nexus.dashboard.store import ArtifactStore
from nexus.gateway.observers.file import visualization_dump
from nexus.gateway.trajectory import Chain, JoinLink, SpawnLink, Trajectory, TrajectoryNode


def test_projection_preserves_physical_forest_and_explicit_links():
    trajectory = branched_trajectory()
    data = visualization_dump(trajectory)
    graph = project_trajectory(data)
    assert graph['layout'] == 'assembly-order-not-time'
    physical = [edge for edge in graph['raw']['edges'] if edge['kind'] == 'physical']
    assert {(edge['source'], edge['target']) for edge in physical} == {
        (f'n:{node.parent}', f'n:{node.node_id}') for node in trajectory.nodes.values() if node.parent is not None
    }
    assert {edge['kind'] for edge in graph['raw']['edges']} == {'physical', 'spawn', 'join'}
    join = next(edge for edge in graph['raw']['edges'] if edge['kind'] == 'join')
    assert join['source'] == 'c:research'
    assert join['target'] == 'n:13'
    assert graph['lanes'][0]['id'] == 'main'
    assert [lane['id'] for lane in graph['lanes']] == ['main', 'research', 'tests']
    assert trajectory.node_chain(14) == 'main'
    assert next(node for node in graph['raw']['nodes'] if node['id'] == 'n:14')['boundary']['kind'] == 'compaction'
    assert next(node for node in graph['raw']['nodes'] if node['id'] == 'n:15')['selection'] == 'rejected_sibling'
    assert sorted(node_id for item in graph['overview']['nodes'] for node_id in item['node_ids']) == sorted(trajectory.nodes)
    assert any(len(item['node_ids']) > 1 for item in graph['overview']['nodes'])
    assert trajectory.dump() == data


def test_unresolved_spawn_and_join_are_not_invented():
    trajectory = branched_trajectory()
    trajectory.chains['research'].spawn = SpawnLink(node=None, candidates=['a', 'b'])
    trajectory.chains['research'].join = JoinLink(node=None)
    graph = project_trajectory(trajectory.dump())
    assert not any(edge.get('chain_id') == 'research' for edge in graph['raw']['edges'])
    assert len(graph['warnings']) == 2
    lane = next(lane for lane in graph['lanes'] if lane['id'] == 'research')
    assert lane['facts']['spawn']['candidates'] == ['a', 'b']


def test_no_declared_chains_retains_default_membership_without_agent_inference():
    trajectory = branched_trajectory()
    trajectory.chains = {}
    graph = project_trajectory(trajectory.dump())
    assert len(graph['lanes']) == 1
    assert graph['lanes'][0]['declared'] is False
    assert graph['lanes'][0]['kind'] == 'unclassified'
    assert {edge['kind'] for edge in graph['raw']['edges']} == {'physical'}


def test_non_monotonic_node_ids_are_not_sorted_into_false_edges():
    trajectory = Trajectory(session_id='nonmonotonic', roots=[9], nodes={
        9: TrajectoryNode(node_id=9, message_id=9, parent=None, children=[2], generated_by='client'),
        2: TrajectoryNode(node_id=2, message_id=2, parent=9, children=[], generated_by='llm'),
    }, chains={'main': Chain(chain_id='main', roots=[9])})
    graph = project_trajectory(trajectory.dump())
    assert graph['raw']['edges'] == [{'source': 'n:9', 'target': 'n:2', 'kind': 'physical'}]


def test_directory_catalog_reads_summary_without_full_payload(tmp_path, monkeypatch):
    write_trajectory(str(tmp_path))
    store = ArtifactStore(tmp_path)
    original = store._read_json
    reads = []

    def tracked(path, **kwargs):
        reads.append(path.name)
        return original(path, **kwargs)

    monkeypatch.setattr(store, '_read_json', tracked)
    result = store.scan()
    assert len(result['items']) == 1
    assert len(reads) == 1
    assert reads[0].endswith('.summary.json')
    assert store.scan() == result
    assert len(reads) == 1
    assert store.load(result['items'][0]['id'])['trajectory']['session_id'] == branched_trajectory().session_id


@pytest.mark.parametrize('compressed', [True, False])
@pytest.mark.parametrize('wrapped', [True, False])
def test_direct_input_preserves_trajectory_and_export_metadata(tmp_path, compressed, wrapped):
    path = tmp_path / ('input.json.gz' if compressed else 'input.json')
    trajectory = branched_trajectory().dump()
    metadata = {'artifact_id': 'example', 'omitted': ['external_node_payloads']} if wrapped else {}
    data = {**metadata, 'trajectory': trajectory} if wrapped else trajectory
    content = json.dumps(data).encode()
    path.write_bytes(gzip.compress(content) if compressed else content)
    store = ArtifactStore(path)
    result = store.scan()
    assert result['errors'] == []
    assert len(result['items']) == 1
    assert store.load(result['items'][0]['id']) == {'trajectory': trajectory, 'export': metadata}


def test_bad_files_and_external_symlinks_are_not_served(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    outside = tmp_path / 'secret.json'
    outside.write_text(json.dumps(branched_trajectory().dump()))
    (root / 'link.trajectory.json').symlink_to(outside)
    (root / 'bad.trajectory.json').write_text('{partial')
    (root / 'truncated.trajectory.json.gz').write_bytes(gzip.compress(b'{}')[:-5])
    (root / '.trajectory-hidden.tmp').write_text('partial')
    store = ArtifactStore(root)
    result = store.scan()
    assert result['items'] == []
    assert len(result['errors']) == 3
    assert any('Truncated' in error['error'] for error in result['errors'])
    with pytest.raises(KeyError):
        store.load('../secret.json')


def test_orphan_artifact_recovered_if_summary_publish_failed(tmp_path):
    write_trajectory(str(tmp_path))
    next(tmp_path.glob('*.summary.json')).unlink()
    result = ArtifactStore(tmp_path).scan()
    assert len(result['items']) == 1
    assert result['errors'] == []


@pytest.mark.asyncio
async def test_http_read_only_list_detail_static_and_unknown_ids(tmp_path):
    write_trajectory(str(tmp_path))
    async with TestClient(TestServer(create_app(tmp_path))) as client:
        response = await client.get('/')
        assert response.status == 200
        html = await response.text()
        assert 'Trajectory Explorer' in html
        assert html.index('assets/media.js') < html.index('assets/app.js')
        assert html.index('id="segments"') > html.index('id="viewport"')
        assert 'id="segment-count"' in html
        assert 'script-src' in response.headers['Content-Security-Policy']
        assert "img-src 'self' data:" in response.headers['Content-Security-Policy']
        assert (await client.get('/assets/app.js')).status == 200
        assert (await client.get('/assets/media.js')).status == 200
        assert (await client.get('/assets/secret.json')).status == 404
        response = await client.get('/api/trajectories')
        result = await response.json()
        assert result['total'] == 1
        artifact_id = result['items'][0]['id']
        response = await client.get(f'/api/trajectories/{artifact_id}')
        data = await response.json()
        assert response.status == 200
        assert len(data['trajectory']['nodes']) == 17
        assert len(data['graph']['lanes']) == 3
        assert [segment['segment_id'] for segment in data['graph']['segments']] == [
            segment.segment_id for segment in branched_trajectory().to_segment_node_lists()
        ]
        assert (await client.get('/api/trajectories/unknown')).status == 404
        assert (await client.get('/api/trajectories?limit=no')).status == 400
        assert (await client.get('/api/trajectories?offset=5')).status == 200
        assert (await (await client.get('/api/trajectories?q=missing')).json())['total'] == 0
        assert (await client.post('/api/trajectories', json={})).status == 405


@pytest.mark.asyncio
async def test_invalid_topology_returns_clear_error_instead_of_hanging(tmp_path):
    data = branched_trajectory().dump()
    data['nodes'][0]['parent'] = 16
    (tmp_path / 'bad.trajectory.json').write_text(json.dumps(data))
    async with TestClient(TestServer(create_app(tmp_path))) as client:
        catalog = await (await client.get('/api/trajectories')).json()
        response = await client.get('/api/trajectories/' + catalog['items'][0]['id'])
        assert response.status == 422
        assert 'cycle' in await response.text()


def test_dashboard_cli_requires_explicit_path(capsys):
    parser = argparse.ArgumentParser()
    register_dashboard_parser(parser.add_subparsers(dest='command'))
    args = parser.parse_args(['dashboard', '--path', '/mnt/ceph/run'])
    assert args.path == '/mnt/ceph/run'
    assert args.host == '127.0.0.1'
    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(['dashboard', '--help'])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert '--path' in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(['dashboard'])


@pytest.mark.parametrize('wrapped', [True, False])
@pytest.mark.parametrize('schema_version', [None, 'unsupported'])
def test_missing_or_unsupported_trajectory_schema_is_rejected(tmp_path, wrapped, schema_version):
    trajectory = branched_trajectory().dump()
    if schema_version is None:
        trajectory.pop('schema_version')
    else:
        trajectory['schema_version'] = schema_version
    data = {'trajectory': trajectory} if wrapped else trajectory
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(data))
    result = ArtifactStore(path).scan()
    assert result['items'] == []
    assert result['errors'][0]['error'] == 'Unsupported trajectory schema'


def test_summary_file_must_match_trajectory_filename(tmp_path):
    write_trajectory(str(tmp_path))
    path = next(tmp_path.glob('*.summary.json'))
    summary = json.loads(path.read_text())
    summary['file'] = 'other.trajectory.json.gz'
    path.write_text(json.dumps(summary))
    result = ArtifactStore(tmp_path).scan()
    assert result['items'] == []
    assert result['errors'][0]['error'] == 'Invalid summary catalog entry'
