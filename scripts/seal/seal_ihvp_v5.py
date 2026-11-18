#!/usr/bin/env python3
"""Seal ihvp-sequence-v5 (real-trajectory replay, balanced split) from the author's verified data package.

Usage: seal_ihvp_v5.py --extract var/raw/ihvp-v5-data/extract --data-root var/raw/ihvp-sequence-v5/data [--image TAG]

The package (release ihvp-sequence-statement-20261117, SHA256 in task.json) holds data/<source case>/ and
balanced-evaluator/split_manifest.json, whose pool labels override the source names. The pools are built as
data/{public,private}/<case>/ hard links into the extracted package (mode 0700, never mounted into a workspace), each
with a manifest; the model-visible public.tar is exactly the five MODEL_ONLY files.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = ROOT / 'tasks/ihvp-sequence/versions/v5'
WORKSPACE = ('README.md', 'statement.md', 'interface.md', 'solver.py', 'smoke.json')
COUNTS = {'public': 15, 'private': 30}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(f'seal refused: {message}')


def public_material(destination):
    with tarfile.open(destination, 'x') as tar:
        for name in WORKSPACE:
            body = (VERSION / 'workspace' / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(body), 0o444, 0
            tar.addfile(info, io.BytesIO(body))


def build_pool(extract, data, split, cases, revision, manifest_sha):
    pool = data / split
    pool.mkdir(parents=True)
    entries = []
    for case in sorted(cases, key=lambda c: c['case']):
        source = extract / 'data' / case['case']
        meta = json.loads((source / 'meta.json').read_text())
        require(meta['dimension'] == 16384 and meta['steps'] == 50, f"{case['case']}: not d=16384, T=50")
        files = ['COMPLETE.json', 'meta.json', 'reference.npz'] + [f'step_{t:03d}.npz' for t in range(50)]
        require(all((source / name).is_file() for name in files), f"{case['case']}: incomplete case directory")
        complete = sha256(source / 'COMPLETE.json')
        require(complete == case['data_id'], f"{case['case']}: COMPLETE.json differs from the split manifest")
        target = pool / case['case']
        target.mkdir()
        for name in files:
            os.link(source / name, target / name)
        entries.append({'id': case['case'], 'path': case['case'], 'sha256': complete,
                        'dataset': case['dataset'], 'profile': case['profile']})
    manifest = {'version': 'v5', 'split': split, 'revision': revision, 'split_manifest_sha256': manifest_sha,
                'expected_cases': len(entries), 'cases': entries}
    (pool / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    pool.chmod(0o700)
    return pool / 'manifest.json'


def seal(args):
    spec_path = VERSION / 'task.json'
    spec = json.loads(spec_path.read_text())
    extract = args.extract.resolve()
    data = args.data_root.resolve()
    require(data.is_relative_to(ROOT / 'var/raw'), 'sealed data root must be in ignored maintainer storage (var/raw)')
    require(not data.exists(), 'refusing to replace existing pools')
    split_path = extract / 'balanced-evaluator/split_manifest.json'
    manifest_sha = sha256(split_path)
    require(manifest_sha == spec['data']['split_manifest_sha256'], 'split manifest differs from task.json')
    split = json.loads(split_path.read_text())
    require(len(split['cases']) == 45 and len({c['case'] for c in split['cases']}) == 45, 'expected 45 distinct cases')
    pools = {name: [c for c in split['cases'] if c['split'] == name] for name in COUNTS}
    require({name: len(cases) for name, cases in pools.items()} == COUNTS, 'pool sizes differ from 15/30')
    tag = args.image or spec['images']['task']['tag']
    info = json.loads(subprocess.run(['docker', 'image', 'inspect', tag], check=True, capture_output=True, text=True).stdout)[0]
    require(info['Architecture'] == 'amd64' and info['Os'] == 'linux', 'task image must be Linux amd64')
    sealed = ROOT / '.runtime-ops' / spec['name']
    require(not sealed.exists(), 'refusing to replace an existing seal')
    manifests = {name: build_pool(extract, data, name, cases, split['revision'], manifest_sha) for name, cases in pools.items()}
    sealed.mkdir(parents=True)
    public_material(sealed / 'public.tar')
    relative = data.relative_to(ROOT)
    lock = {'name': spec['name'], 'status': spec['status'], 'sealed_utc': datetime.now(timezone.utc).isoformat(),
            'spec_sha256': sha256(spec_path), 'image': {'tag': tag, 'id': info['Id'], 'replaces': []},
            'public_tar_sha256': sha256(sealed / 'public.tar'), 'split_manifest_sha256': manifest_sha,
            'data': {name: str(relative / name) for name in COUNTS},
            'data_manifest_sha256': {name: sha256(path) for name, path in manifests.items()},
            'data_present': {name: True for name in COUNTS}, 'roots': {}, 'sample_service': {},
            'activation': 'Candidate replay: both references are measured on this machine per pool before any score.'}
    (sealed / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    return {'status': 'SEALED', 'task': spec['name'], **{name: len(cases) for name, cases in pools.items()},
            'learner_files': len(WORKSPACE), 'image_id': info['Id']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--extract', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--image')
    print(json.dumps(seal(parser.parse_args())))
