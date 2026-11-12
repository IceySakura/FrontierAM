#!/usr/bin/env python3
"""Seal ihvp-sequence for this machine: .runtime-ops/<name>/{lock.json, public.tar}.

public.tar is the development workspace (problem.md and the materialized samples). The lock records the task image ID
and the data roots graded by arena/operator_drivers/ihvp_sequence.py (FRONTIERAM_IHVP_DATA): private and public as
prepared by prepare_ihvp_sequence.py --source-bundle, and a samples split holding only the two full-size samples for
infrastructure checks. Each data root is made mode 0700: inside the grading container only the evaluator (root) can
read the teacher files, not the submission's worker (UID 1000).

v2 (ihvp-sequence-v2) has the same data, references and grader; its workspace carries no generator code or
construction arrays (versions/v2/samples), and the lock lists the sample cases served from the host
(arena/operator_drivers/ihvp_oracle.py). v3 (the author's hard extension) uses the data prepared by
prepare_ihvp_hard_v3.py: private holds all 30 sequences, the feedback split only the 15 added hard ones (user decision
2026-11-12: one feedback run stays within the 15-minute cap), and the two samples come from the added feedback cases.

Usage: seal_ihvp_sequence.py [--version v1|v2|v3] [--samples DIR] [--image TAG] [--force]
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
DATA = 'var/raw/ihvp-sequence-v1/data'
DATA_V3 = 'var/raw/ihvp-sequence-v3/data'


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


def hard_feedback_split() -> Path:
    """v3 feedback: the added hard sequences of the prepared 30-sequence public pool (hard links, same bytes)."""
    source, target = ROOT / DATA_V3 / 'public', ROOT / DATA_V3 / 'public-hard'
    manifest = json.loads((source / 'manifest.json').read_text())
    cases = [c for c in manifest['cases'] if c.get('group') == 'hard']
    for case in cases:
        destination = target / case['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(source / case['path'], destination)
    (target / 'manifest.json').write_text(json.dumps({**manifest, 'sequence_count': len(cases), 'group_counts': {'hard': len(cases)},
                                                      'cases': cases}, indent=2) + '\n')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('v1', 'v2', 'v3'), default='v1')
    parser.add_argument('--samples', type=Path, default=ROOT / 'var/outputs/ihvp-gen/samples')
    parser.add_argument('--image', help='default: the task version\'s image tag')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    version = ROOT / 'tasks/ihvp-sequence/versions' / args.version
    spec = version / 'task.json'
    name = json.loads(spec.read_text())['name']
    args.image = args.image or json.loads(spec.read_text())['images']['task']['tag']
    SEALED = ROOT / '.runtime-ops' / name
    if (SEALED / 'lock.json').exists() and not args.force:
        parser.error(f'{SEALED} exists; use --force')
    samples = args.samples.resolve()
    SEALED.mkdir(parents=True, exist_ok=True)
    public = SEALED / 'public.tar'
    workspace_samples = samples if args.version == 'v1' else version / 'samples'
    with tarfile.open(public, 'w') as tar:
        add(tar, 'problem.md', (version / 'statement.md').read_bytes())
        for path in sorted(p for p in workspace_samples.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
            add(tar, f'samples/{path.relative_to(workspace_samples).as_posix()}', path.read_bytes(),
                0o755 if os.access(path, os.X_OK) else 0o644)
    if args.version == 'v3':
        hard_feedback_split()
        data = {'private': f'{DATA_V3}/private', 'public': f'{DATA_V3}/public-hard', 'samples': f'{DATA_V3}/samples'}
    else:
        samples_split(samples)
        data = {split: f'{DATA}/{split}' for split in ('private', 'public', 'samples')}
    for relative in data.values():
        if (ROOT / relative).is_dir():
            (ROOT / relative).chmod(0o700)
    image = subprocess.run(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'],
                           capture_output=True, text=True, check=True).stdout.strip()
    lock = {'name': name, 'status': json.loads(spec.read_text())['status'],
            'sealed_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'spec_sha256': sha(spec),
            'image': {'tag': args.image, 'id': image, 'replaces': []}, 'public_tar_sha256': sha(public),
            'samples_manifest_sha256': sha(ROOT / data['samples'] / 'manifest.json'), 'data': data,
            'data_present': {k: (ROOT / v / 'manifest.json').is_file() for k, v in data.items()}, 'roots': {}}
    if args.version in ('v2', 'v3'):
        lock['sample_service'] = {'smoke': f'tasks/ihvp-sequence/versions/{"v1/data" if args.version == "v2" else "v3"}/samples/smoke.json',
                                  **{c: f'{data["samples"]}/cases/{c}/teacher.npz' for c in ('sample_01', 'sample_02')}}
    (SEALED / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps({k: lock[k] for k in ('image', 'public_tar_sha256', 'data_present')}))


if __name__ == '__main__':
    main()
