import json

from types import SimpleNamespace

import pytest

from nexus.extensions.tasks.frontieram_offline.assets import file_sha256
from nexus.extensions.tasks.frontieram_offline.local_calibration import host_fingerprint, qualified_baseline


@pytest.fixture
def calibration(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    report = tmp_path / 'reference.json'
    report.write_text(json.dumps({'runs': [{'valid': True} for _ in range(69)], 'aggregates': {}}))
    assets = SimpleNamespace(source_root=tmp_path, image_id='sha256:abc', manifest=manifest, baseline=report)
    info = {'ID': 'host-a', 'Architecture': 'x86_64', 'KernelVersion': '6.8', 'NCPU': 8, 'MemTotal': 16 * 2**30}
    docker = SimpleNamespace(run=lambda *args: json.dumps(info).encode())
    lock = {
        'host_fingerprint': host_fingerprint(docker, assets.image_id),
        'report': str(report),
        'report_sha256': file_sha256(report),
        'manifest_sha256': file_sha256(manifest),
    }
    (tmp_path / 'PORTABLE.json').write_text('{}')
    return assets, docker, lock, info


def test_migrated_host_never_silently_uses_arm_timing(calibration):
    assets, docker, _, _ = calibration
    assert qualified_baseline(assets, docker) is None


def test_qualified_reference_works_only_after_explicit_calibration(calibration):
    assets, docker, lock, _ = calibration
    (assets.source_root / 'local_calibration.json').write_text(json.dumps(lock))
    assert len(qualified_baseline(assets, docker)['runs']) == 69


@pytest.mark.parametrize('change', ['host', 'image', 'data', 'report'])
def test_host_image_data_or_report_changes_invalidate_reference(calibration, change):
    assets, docker, lock, info = calibration
    (assets.source_root / 'local_calibration.json').write_text(json.dumps(lock))
    if change == 'host':
        info['ID'] = 'host-b'
    elif change == 'image':
        assets.image_id = 'sha256:other'
    elif change == 'data':
        assets.manifest.write_text('{"changed":true}')
    else:
        assets.baseline.write_text('{}')
    with pytest.raises(RuntimeError, match='does not match'):
        qualified_baseline(assets, docker)
