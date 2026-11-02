"""A migrated host must qualify its own fixed-policy timing reference."""

import hashlib
import json

from pathlib import Path

from .assets import file_sha256


def host_fingerprint(docker, image_id):
    info = json.loads(docker.run('info', '--format', '{{json .}}'))
    fields = {key: info.get(key) for key in ('ID', 'Architecture', 'OSType', 'KernelVersion', 'NCPU', 'MemTotal')}
    fields['image_id'] = image_id
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def qualified_baseline(assets, docker):
    if not (assets.source_root / 'PORTABLE.json').exists():
        return json.loads(assets.baseline.read_text())
    lock = assets.source_root / 'local_calibration.json'
    if not lock.exists():
        return None
    data = json.loads(lock.read_text())
    report = Path(data['report'])
    if (
        data['host_fingerprint'] != host_fingerprint(docker, assets.image_id)
        or data['manifest_sha256'] != file_sha256(assets.manifest)
        or data['report_sha256'] != file_sha256(report)
    ):
        raise RuntimeError('Local calibration does not match this host, image or dataset')
    baseline = json.loads(report.read_text())
    if len(baseline.get('runs', [])) != 69 or not all(row.get('valid') is True for row in baseline['runs']):
        raise RuntimeError('The local reference has not passed all 69 checks')
    return baseline
