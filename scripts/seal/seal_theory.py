#!/usr/bin/env python3
"""Seal a theory-only task version (task.json theory_only): the model-visible workspace (public.tar) and lock.json
under .runtime-ops/<name>/. The workspace is <version>/workspace/ as is; lock.json records the task image and the
workspace hash. There are no data sets: the deliverable is checked with scripts/grade/lean-comparator.py and reviewed.

Usage: seal_theory.py VERSION_DIR [--force]   (e.g. tasks/decentralized-sum-of-nonconvex/versions/theory-v1)
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
EXECUTABLE = {'tools/lean_check.py'}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workspace_tar(workspace: Path, destination: Path) -> int:
    members = sorted(p for p in workspace.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files = {str(p.relative_to(workspace)): p.read_bytes() for p in members}
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
    parser.add_argument('version', type=Path)
    parser.add_argument('--force', action='store_true', help='replace an existing seal')
    args = parser.parse_args()
    version = args.version.resolve()
    spec_path = version / 'task.json'
    spec = json.loads(spec_path.read_text())
    if not spec.get('theory_only'):
        raise SystemExit(f'{spec_path} is not a theory-only task')
    tag = spec['images']['task']['tag']
    info = json.loads(subprocess.run(['docker', 'image', 'inspect', tag], check=True, capture_output=True,
                                     text=True).stdout)[0]
    if info['Architecture'] != 'amd64':
        raise SystemExit('task image must be linux/amd64')
    sealed = ROOT / '.runtime-ops' / spec['name']
    if sealed.exists():
        if not args.force:
            raise SystemExit(f'{sealed} exists; pass --force to replace it')
        for path in sealed.iterdir():
            path.unlink()
    sealed.mkdir(parents=True, exist_ok=True)
    count = workspace_tar(version / 'workspace', sealed / 'public.tar')
    lock = {'name': spec['name'], 'status': spec['status'], 'sealed_utc': datetime.now(timezone.utc).isoformat(),
            'spec_sha256': sha256(spec_path), 'image': {'tag': tag, 'id': info['Id'], 'replaces': []},
            'public_tar_sha256': sha256(sealed / 'public.tar'), 'data': {}, 'data_sha256': {}, 'roots': {},
            'provisional': False,
            'activation': 'Theory only (trial runs): scored by the Lean check and the theory review.'}
    (sealed / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps({'status': 'SEALED', 'task': spec['name'], 'workspace_files': count, 'image_id': info['Id']}))


if __name__ == '__main__':
    main()
