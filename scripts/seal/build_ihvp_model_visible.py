#!/usr/bin/env python3
"""Build the exact six-file IHVP v3 learner view; never copy a source tree.

The adjacent manifest is for the maintainer and is not an archive member.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = ROOT / 'tasks/ihvp-sequence/versions/v3'
ALLOWLIST = {
    'problem.md': 'statement.md',
    'samples/README.md': 'samples/README.md',
    'samples/run_sample.py': 'samples/run_sample.py',
    'samples/smoke.json': 'samples/smoke.json',
    'samples/support/api.py': 'samples/support/api.py',
    'samples/support/worker.py': 'samples/support/worker.py',
}


def build(out):
    out = Path(out).expanduser().resolve()
    if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / 'var'):
        raise ValueError('output must be ignored var/ or outside the checkout')
    if out.suffix != '.zip':
        raise ValueError('output must be a .zip')
    manifest_path = out.with_suffix('.manifest.json')
    checksum_path = out.with_suffix('.zip.sha256')
    if any(path.exists() for path in (out, manifest_path, checksum_path)):
        raise FileExistsError('refusing to replace an existing delivery')
    contents = {}
    records = {}
    for destination, relative in ALLOWLIST.items():
        source = VERSION / relative
        if source.is_symlink() or not source.resolve().is_relative_to(VERSION.resolve()):
            raise ValueError('learner material must be a contained regular file')
        content = source.read_bytes()
        contents[destination] = content
        records[destination] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    out.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        descriptor = os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        created.append(out)
        with os.fdopen(descriptor, 'wb') as stream, zipfile.ZipFile(stream, 'w') as archive:
            for name, content in sorted(contents.items()):
                entry = zipfile.ZipInfo(name)
                entry.create_system = 3
                entry.external_attr = 0o100444 << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, content)
        with zipfile.ZipFile(out) as archive:
            if set(archive.namelist()) != set(ALLOWLIST):
                raise ValueError('learner archive is not the exact allowlist')
            for name, expected in records.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected['sha256']:
                    raise ValueError('learner archive bytes differ from the whitelist source')
        checksum = hashlib.sha256(out.read_bytes()).hexdigest()
        manifest = {'schema': 'frontieram/ihvp-v3-model-visible-1', 'file_count': 6,
                    'files': records, 'zip_sha256': checksum,
                    'only_teacher_exception': 'public d=4,T=3 smoke.json',
                    'full_scale_teachers': 'none; host oracle only',
                    'maintainer_only_manifest': True}
        for path, text in ((manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'),
                           (checksum_path, f'{checksum}  {out.name}\n')):
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            created.append(path)
            with os.fdopen(descriptor, 'w') as stream:
                stream.write(text)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return {'status': 'PASS', 'file_count': 6, 'bytes': out.stat().st_size,
            'zip_sha256': checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out), sort_keys=True))


if __name__ == '__main__':
    main()
