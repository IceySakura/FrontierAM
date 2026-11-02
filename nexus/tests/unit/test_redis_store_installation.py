from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

from pathlib import Path

import pytest


@pytest.mark.parametrize('extra', [None, 'redis', 'redis-fs'])
def test_locked_sync_does_not_require_submodules(extra) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lock_text = (repo_root / 'uv.lock').read_text(encoding='utf-8')
    assert 'hy-kv-store' not in lock_text

    with tempfile.TemporaryDirectory() as raw_directory:
        archive_root = Path(raw_directory)
        archive = subprocess.run(
            ['git', 'archive', 'HEAD'],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        subprocess.run(['tar', '-x', '-C', str(archive_root)], input=archive.stdout, check=True)
        shutil.copyfile(repo_root / 'pyproject.toml', archive_root / 'pyproject.toml')
        shutil.copyfile(repo_root / 'uv.lock', archive_root / 'uv.lock')
        submodule = archive_root / 'third_party'
        if submodule.exists():
            shutil.rmtree(submodule)
        assert not submodule.exists()

        sync = subprocess.run(
            [
                'uv',
                'sync',
                '--project',
                str(archive_root),
                '--locked',
                '--offline',
                '--dry-run',
                *(['--extra', extra] if extra else []),
            ],
            cwd=archive_root,
            check=False,
            capture_output=True,
            text=True,
        )

    assert sync.returncode == 0, sync.stderr
    assert 'hy-kv-store' not in sync.stdout
    assert 'does not appear to be a Python project' not in sync.stderr


def test_wheel_contains_fs_services_and_cluster_config_without_hykvstore(tmp_path):
    pytest.importorskip('zmq')
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / 'dist'
    built = subprocess.run(
        ['uv', 'build', '--wheel', '--offline', '--out-dir', str(output)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next(output.glob('*.whl'))
    installed = tmp_path / 'installed'
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
        resource_prefix = 'nexus/gateway/trajectory_store_backends/redis_store/'
        for filename in ('redis_server_cluster.json', 'redis_server_launcher_prepare.sh', 'utils.py'):
            path = resource_prefix + filename
            assert archive.read(path) == (root / path).read_bytes()
            assert archive.getinfo(path).external_attr >> 16 & 0o777 == (root / path).stat().st_mode & 0o777
        assert 'nexus/_bundle/deployment/redis/redis_server_cluster.json' not in archive.namelist()
        assert not any('hy_kv_store' in path for path in archive.namelist())
    script = """
import argparse
import importlib.abc
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

class BlockHyKVStore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'hy_kv_store':
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockHyKVStore())
sys.modules['ray'] = SimpleNamespace(remote=lambda **kw: lambda function: function)
sys.modules['ray.util.scheduling_strategies'] = SimpleNamespace(NodeAffinitySchedulingStrategy=object)
import nexus
from nexus.gateway.trajectory_store_backends.redis import HybridNodeRef, NodeRecord
from nexus.gateway.trajectory_store_backends.redis_store.file_storage import FSClient, LocalCluster, RayCluster
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.launcher import run as run_fs
from nexus.gateway.trajectory_store_backends.redis_store.launcher import run as run_services
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import Codec, Message
from nexus.gateway.trajectory_store_backends.redis_store.cluster import CONFIG, config_path
from nexus.gateway.trajectory_store_backends.redis_store.utils import get_ray_head_ip
assert str(Path(nexus.__file__).resolve()).startswith(sys.argv[1])
redis_directory = Path(sys.argv[1]) / 'nexus/gateway/trajectory_store_backends/redis_store'
assert config_path() == redis_directory / 'redis_server_cluster.json'
assert CONFIG['redis_port'] == 16379
message = Message.create('put', payload=b'data')
assert Codec.decode(Codec.encode(message))[0] == message
assert NodeRecord.deserialize(NodeRecord('run', 1, [1]).serialize()).token_ids == [1]
assert HybridNodeRef('run', 'node').blob_urls == {}
assert callable(run_fs)
assert callable(run_services)
assert callable(get_ray_head_ip)

from nexus.cli.commands import redis as redis_cli
parser = argparse.ArgumentParser()
redis_cli.register_redis_parser(parser.add_subparsers())
combined = parser.parse_args(['redis', 'up', '--primary-count', '6', '--num-data-units', '6'])
assert combined.primary_count == combined.num_data_units == 6
args = parser.parse_args(['redis', 'cluster', 'prepare'])

def prepare(command, *, check):
    assert command[0] == 'bash'
    assert check is False
    script = Path(command[1])
    assert script.is_absolute()
    assert script == redis_directory / 'redis_server_launcher_prepare.sh'
    assert script.is_file()
    return SimpleNamespace(returncode=7)

with patch.object(redis_cli.subprocess, 'run', side_effect=prepare) as invoked:
    assert redis_cli.handle_redis_command(args) == 7
invoked.assert_called_once()
"""
    result = subprocess.run(
        [sys.executable, '-c', script, str(installed)],
        cwd=tmp_path,
        env={
            **os.environ,
            'PYTHONPATH': str(installed),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
