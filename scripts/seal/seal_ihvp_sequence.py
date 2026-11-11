#!/usr/bin/env python3
"""Seal ihvp-sequence v1 for this machine: .runtime-ops/ihvp-sequence/{lock.json, public.tar}.

public.tar is the development workspace (problem.md and the materialized samples). The lock records the task image ID
and the data roots graded by arena/operator_drivers/ihvp_sequence.py (FRONTIERAM_IHVP_DATA): private and public as
prepared by prepare_ihvp_sequence.py --source-bundle, and a samples split holding only the two full-size samples for
infrastructure checks. Each data root is made mode 0700: inside the grading container only the evaluator (root) can
read the teacher files, not the submission's worker (UID 1000).

Usage: seal_ihvp_sequence.py [--samples DIR] [--image TAG] [--force]
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = ROOT / 'tasks/ihvp-sequence/versions/v1'
SEALED = ROOT / '.runtime-ops/ihvp-sequence'
DATA = 'var/raw/ihvp-sequence-v1/data'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def add(tar: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), mode, 0
    tar.addfile(info, io.BytesIO(data))


def samples_split(samples: Path) -> Path:
    """The two full-size samples as a grading split (the evaluator accepts only d = 16384, T = 50 cases)."""
    target = ROOT / DATA / 'samples'
    manifest = json.loads((samples / 'manifest.json').read_text())
    cases = [c for c in manifest['cases'] if c.get('kind') == 'signed_dct']
    for case in cases:
        destination = target / case['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(samples / case['path'], destination)
    (target / 'manifest.json').write_text(json.dumps({**manifest, 'cases': cases}, indent=2) + '\n')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=ROOT / 'var/outputs/ihvp-gen/samples')
    parser.add_argument('--image', default=json.loads((VERSION / 'task.json').read_text())['images']['task']['tag'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if (SEALED / 'lock.json').exists() and not args.force:
        parser.error(f'{SEALED} exists; use --force')
    samples = args.samples.resolve()
    SEALED.mkdir(parents=True, exist_ok=True)
    public = SEALED / 'public.tar'
    with tarfile.open(public, 'w') as tar:
        add(tar, 'problem.md', (VERSION / 'statement.md').read_bytes())
        for path in sorted(p for p in samples.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
            add(tar, f'samples/{path.relative_to(samples).as_posix()}', path.read_bytes(),
                0o755 if os.access(path, os.X_OK) else 0o644)
    samples_split(samples)
    data = {split: f'{DATA}/{split}' for split in ('private', 'public', 'samples')}
    for relative in data.values():
        if (ROOT / relative).is_dir():
            (ROOT / relative).chmod(0o700)
    image = subprocess.run(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'],
                           capture_output=True, text=True, check=True).stdout.strip()
    spec = VERSION / 'task.json'
    lock = {'name': 'ihvp-sequence', 'status': json.loads(spec.read_text())['status'],
            'sealed_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'spec_sha256': sha(spec),
            'image': {'tag': args.image, 'id': image, 'replaces': []}, 'public_tar_sha256': sha(public),
            'samples_manifest_sha256': sha(samples / 'manifest.json'), 'data': data,
            'data_present': {k: (ROOT / v / 'manifest.json').is_file() for k, v in data.items()}, 'roots': {}}
    (SEALED / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps({k: lock[k] for k in ('image', 'public_tar_sha256', 'data_present')}))


if __name__ == '__main__':
    main()
