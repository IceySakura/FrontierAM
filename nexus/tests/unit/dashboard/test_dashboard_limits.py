import argparse
import gzip
import json

from pathlib import Path

import pytest

from aiohttp.test_utils import TestClient, TestServer
from tests.unit.dashboard.trajectory_fixtures import branched_trajectory

from nexus.cli.commands.dashboard import register_dashboard_parser, serve_dashboard
from nexus.dashboard.server import create_app
from nexus.dashboard.store import DEFAULT_MAX_BYTES, ArtifactStore
from nexus.gateway.observers.file import FileTrajectoryObserver


def parse_args(*args):
    parser = argparse.ArgumentParser()
    register_dashboard_parser(parser.add_subparsers(dest='command'))
    return parser.parse_args(['dashboard', *args])


@pytest.mark.parametrize('mib', [64, 1024])
def test_cli_forwards_read_limit_for_path(tmp_path, monkeypatch, mib):
    from nexus.cli.commands import dashboard
    from nexus.dashboard import server

    calls = []
    app = object()

    def make_app(path, *, max_bytes):
        assert Path(path).is_dir()
        calls.append(max_bytes)
        return app

    def run_app(application, *, host, port):
        assert application is app
        assert host == '127.0.0.1'
        assert port == 8766

    monkeypatch.setattr(server, 'create_app', make_app)
    monkeypatch.setattr(dashboard.web, 'run_app', run_app)
    source = ['--path', str(tmp_path)]
    option = [] if mib == 64 else ['--max-artifact-mib', str(mib)]
    args = parse_args(*source, *option)
    assert args.max_artifact_mib == mib
    assert serve_dashboard(args) == 0
    assert calls == [mib * 1024 * 1024]
    assert DEFAULT_MAX_BYTES == 64 * 1024 * 1024


@pytest.mark.parametrize('limit', [0, -1])
def test_nonpositive_limits_fail_before_reading_or_serving(tmp_path, capsys, limit):
    args = parse_args('--path', str(tmp_path), '--max-artifact-mib', str(limit))
    assert serve_dashboard(args) == 2
    assert '--max-artifact-mib must be a positive integer' in capsys.readouterr().out
    with pytest.raises(ValueError, match='must be positive'):
        ArtifactStore(tmp_path, max_bytes=limit)
    with pytest.raises(ValueError, match='must be positive'):
        create_app(tmp_path, max_bytes=limit)


@pytest.mark.parametrize('value', ['1.5', 'unlimited', 'abc'])
def test_cli_requires_integer_mib(tmp_path, value):
    with pytest.raises(SystemExit):
        parse_args('--path', str(tmp_path), '--max-artifact-mib', value)


@pytest.mark.parametrize('compressed', [False, True])
@pytest.mark.asyncio
async def test_http_custom_limit_preserves_catalog_and_bounds_whole_detail(tmp_path, compressed):
    trajectory = branched_trajectory()
    trajectory.nodes[0].original_message['content'] = 'x' * 16384
    FileTrajectoryObserver(str(tmp_path), compress=compressed).on_final(trajectory)
    artifact = next(tmp_path.glob('*.trajectory.json*'))
    raw = artifact.read_bytes()
    content = gzip.decompress(raw) if compressed else raw
    size = len(content)

    for limit, status in [(size - 1, 422), (size, 200)]:
        async with TestClient(TestServer(create_app(tmp_path, max_bytes=limit))) as client:
            response = await client.get('/api/trajectories')
            assert response.status == 200
            catalog = await response.json()
            assert catalog['total'] == 1
            assert catalog['errors'] == []
            item = catalog['items'][0]
            assert 'trajectory' not in item
            response = await client.get('/api/trajectories/' + item['id'])
            assert response.status == status
            if status == 200:
                data = await response.json()
                assert len(data['trajectory']['nodes']) == len(trajectory.nodes)
                assert data['trajectory']['nodes'][0]['original_message']['content'] == 'x' * 16384
                assert len(data['graph']['lanes']) == 3
            else:
                message = await response.text()
                assert 'read limit' in message
                assert str(limit) in message
                if compressed:
                    assert len(raw) < limit
                    assert 'Decompressed' in message


def test_raising_artifact_limit_does_not_raise_summary_limit(tmp_path):
    FileTrajectoryObserver(str(tmp_path)).on_final(branched_trajectory())
    summary_path = next(tmp_path.glob('*.summary.json'))
    summary = json.loads(summary_path.read_text())
    summary['padding'] = 'x' * (1024 * 1024)
    summary_path.write_text(json.dumps(summary))
    result = ArtifactStore(tmp_path, max_bytes=2 * 1024 * 1024).scan()
    assert result['items'] == []
    assert len(result['errors']) == 1
    assert '1048576-byte read limit' in result['errors'][0]['error']
