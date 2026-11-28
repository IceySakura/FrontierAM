#!/usr/bin/env python3
"""Seal blackbox-cubature-v3: the model-visible workspace (public.tar) and lock.json under .runtime-ops/.

The workspace is versions/v3/workspace/ plus samples/samples.json, reduced from data/samples/instances.json to what a
model may see (no function parameters: the samples' functions are served from the host during development) and
joined with the references' errors on the samples (data/samples/reference_rows.json, measured with the formal driver
by scripts/grade/cubature-v3-samples.py). lock.json records the task image, the workspace hash and the hashes of the
public and private instance files (which stay in the internal repository).

A version whose task.json names data_dir (v3.2: the v3.1 instance sets) takes its samples and instance files from
there and only its workspace from its own directory. A theory-only version (task.json theory_only) has neither
samples nor instance files.

Usage: seal_cubature_v3.py [--version v3|v3.1|v3.2|v3.3|v3.4|v3.5|v3.6|theory-v1] [--force] [--provisional]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = ROOT / 'tasks/blackbox-cubature/versions/v3'
DATA = VERSION / 'data'
SEALED = ROOT / '.runtime-ops/blackbox-cubature-v3'


def select(version: str) -> None:
    global VERSION, DATA, SEALED
    VERSION = ROOT / 'tasks/blackbox-cubature/versions' / version
    spec = json.loads((VERSION / 'task.json').read_text())
    DATA = ROOT / 'tasks/blackbox-cubature' / spec['data_dir'] if 'data_dir' in spec else VERSION / 'data'
    SEALED = ROOT / '.runtime-ops' / spec['name']
EXECUTABLE = {'tools/local_eval.py', 'tools/lean_check.py', 'examples/mc_submission/run.sh'}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def samples_json() -> bytes:
    data = json.loads((DATA / 'samples/instances.json').read_text())
    refs = json.loads((DATA / 'samples/reference_rows.json').read_text())
    strong, weak = ({r['id']: r for r in refs[k]} for k in ('strong', 'weak'))
    out = []
    for inst in data['instances']:
        cell = f'L{inst["L"]}-m{inst["m"]}' if inst['track'] == 'options' else f'd{inst["d"]}'
        out.append({'id': inst['id'], 'track': inst['track'], 'family': inst['family'], 'cell': cell,
                    'dimension': inst['d'], 'L': inst['L'], 'm': inst['m'], 'k': inst['k'], 'seed': inst['seed'],
                    'reference_value': inst['reference']['value'],
                    'reference_rel_uncertainty': inst['reference']['rel_uncertainty'],
                    'e_floor': inst['reference']['e_floor'],
                    'e_reference': strong[inst['id']]['log10_rel_error'], 'e_mc': weak[inst['id']]['log10_rel_error']})
    tracks = 'hard and curved tracks' if any(i['track'] == 'curved' for i in data['instances']) else 'hard track'
    note = ('Sample instances (same distribution as the hidden ones, other seeds). reference_value is the true '
            'integral to the stated relative uncertainty; e_floor is the clip floor of the score; e_reference and e_mc '
            'are the raw log10 relative errors of the GPCA-RQMC reference and of plain Monte Carlo on that instance '
            '(the anchors of the score), measured with the formal evaluator. k is the dimension of the subspace '
            f'carrying the non-smooth part ({tracks}); L and m are the numbers of assets and monitoring dates.')
    return (json.dumps({'note': note, 'instances': out}, indent=1) + '\n').encode()


def workspace_tar(destination: Path, samples: bool = True) -> int:
    members = sorted(p for p in (VERSION / 'workspace').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files = {str(p.relative_to(VERSION / 'workspace')): p.read_bytes() for p in members}
    if samples:
        files['samples/samples.json'] = samples_json()
    with tarfile.open(destination, 'x') as tar:
        for directory in sorted({str(Path(n).parent) for n in files} - {'.'}):
            parts = Path(directory).parts
            for i in range(1, len(parts) + 1):
                name = str(Path(*parts[:i]))
                if name not in tar.getnames():
                    info = tarfile.TarInfo(name)
                    info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o755, 0
                    tar.addfile(info)
        for name, body in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(body), 0o555 if name in EXECUTABLE else 0o444, 0
            tar.addfile(info, io.BytesIO(body))
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', default='v3', choices=('v3', 'v3.1', 'v3.2', 'v3.3', 'v3.4', 'v3.5', 'v3.6',
                                                            'theory-v1'))
    parser.add_argument('--force', action='store_true', help='replace an existing seal')
    parser.add_argument('--provisional', action='store_true',
                        help='no samples.json yet: lets the task image be loaded into the grading daemons early; '
                             'reseal with --force before any attempt')
    args = parser.parse_args()
    select(args.version)
    spec_path = VERSION / 'task.json'
    spec = json.loads(spec_path.read_text())
    tag = spec['images']['task']['tag']
    info = json.loads(subprocess.run(['docker', 'image', 'inspect', tag], check=True, capture_output=True,
                                     text=True).stdout)[0]
    if info['Architecture'] != 'amd64':
        raise SystemExit('task image must be linux/amd64')
    if SEALED.exists():
        if not args.force:
            raise SystemExit(f'{SEALED} exists; pass --force to replace it')
        for path in SEALED.iterdir():
            path.unlink()
    SEALED.mkdir(parents=True, exist_ok=True)
    theory_only = spec.get('theory_only', False)
    count = workspace_tar(SEALED / 'public.tar', samples=not (args.provisional or theory_only))
    data = {} if theory_only else {name: DATA / name / 'instances.json' for name in ('samples', 'public', 'private')}
    lock = {'name': spec['name'], 'status': spec['status'], 'sealed_utc': datetime.now(timezone.utc).isoformat(),
            'spec_sha256': sha256(spec_path), 'image': {'tag': tag, 'id': info['Id'], 'replaces': []},
            'public_tar_sha256': sha256(SEALED / 'public.tar'),
            'data': {name: str(path.relative_to(ROOT)) for name, path in data.items()},
            'data_sha256': {name: sha256(path) for name, path in data.items() if path.exists()},
            'roots': {}, 'provisional': args.provisional,
            'activation': ('Theory only (trial runs): scored by the Lean check and the theory review.' if theory_only
                           else 'Candidate (trial runs): both references are measured on this machine per data set.')}
    (SEALED / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps({'status': 'SEALED', 'task': spec['name'], 'workspace_files': count, 'image_id': info['Id'],
                      'data': sorted(lock['data_sha256'])}))


if __name__ == '__main__':
    main()
