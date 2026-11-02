import json

import pytest

from nexus.runner.trace.tdsql.config import _manager_memory_bytes, resolve_writer_shards


@pytest.mark.parametrize(
    ('total_gib', 'managers', 'expected_gib', 'shards'),
    [(4, 1, 4, 1), (16, 2, 8, 2), (24, 2, 12, 3), (32, 2, 16, 4), (128, 4, 32, 4)],
)
def test_uses_submitted_memory_per_manager(monkeypatch, total_gib, managers, expected_gib, shards):
    monkeypatch.delenv('NEXUS_TRAJ_TDSQL_WRITER_SHARDS', raising=False)
    monkeypatch.setenv(
        'NEXUS_DIST_ALLOCATION_SPEC_JSON',
        json.dumps({'resources': {'memory_bytes': total_gib * 1024**3, 'manager_count': managers}}),
    )
    assert _manager_memory_bytes() == expected_gib * 1024**3
    assert resolve_writer_shards() == shards


@pytest.mark.parametrize(
    'spec',
    [
        None,
        'bad-json',
        'null',
        '[]',
        '{}',
        '{"resources":[]}',
        json.dumps({'resources': {'memory_bytes': 100, 'manager_count': 0}}),
        json.dumps({'resources': {'memory_bytes': -1, 'manager_count': 1}}),
        json.dumps({'resources': {'memory_bytes': 100, 'manager_count': True}}),
        json.dumps({'resources': {'memory_bytes': '100', 'manager_count': 1}}),
        json.dumps({'resources': {'memory_bytes': 101, 'manager_count': 2}}),
    ],
)
def test_missing_or_invalid_resource_configuration_falls_back_to_one(monkeypatch, spec):
    monkeypatch.delenv('NEXUS_TRAJ_TDSQL_WRITER_SHARDS', raising=False)
    monkeypatch.delenv('NEXUS_DIST_ALLOCATION_SPEC_JSON', raising=False)
    if spec is not None:
        monkeypatch.setenv('NEXUS_DIST_ALLOCATION_SPEC_JSON', spec)
    assert _manager_memory_bytes() == 0
    assert resolve_writer_shards() == 1


def test_explicit_shards_override_memory_default(monkeypatch):
    monkeypatch.setenv('NEXUS_DIST_ALLOCATION_SPEC_JSON', 'invalid')
    monkeypatch.setenv('NEXUS_TRAJ_TDSQL_WRITER_SHARDS', '6')
    assert resolve_writer_shards() == 6
    assert resolve_writer_shards(8) == 8
