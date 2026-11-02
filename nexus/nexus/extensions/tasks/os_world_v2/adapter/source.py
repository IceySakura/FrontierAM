"""Validate the independent V2 task fork before executing trusted task code."""

import hashlib
import json
import re
import subprocess

from pathlib import Path


COMMIT = '6f50176ede5a3009ea92849699524ca4903e7a5e'
RELEASE = 'osworld-v2.1'
MANIFEST_SHA256 = 'd1d879baa39fba097defd04e591d9fd5d6914970d2229aabb36e687687fb8646'


def task_id(value: str) -> str:
    value = str(value).removeprefix('task_')
    if not re.fullmatch(r'\d{1,3}', str(value)):
        raise ValueError('V2.1 task id must be numeric')
    value = str(value).zfill(3)
    if not 1 <= int(value) <= 108:
        raise ValueError('V2.1 task id must be in 001..108')
    return value


def verify(root: str | Path) -> dict:
    root = Path(root).expanduser().resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMIT:
        raise ValueError(f'V2.1 source must be pinned at {COMMIT}; got {commit}')
    return verify_files(root)


def verify_files(root: str | Path) -> dict:
    """Check pinned source files after a verified dataset bundle was extracted."""
    root = Path(root).resolve()
    raw = (root / 'benchmark_releases/team-osworld-v2.1.task_hashes.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('V2.1 effective manifest digest mismatch')
    manifest = json.loads(raw)
    prefixes = {
        'files': 'evaluation_examples/task_class',
        'setup_support_files': 'evaluation_examples/setup_support',
        'evaluator_support_files': 'evaluation_examples/evaluator_support',
        'configuration_files': '',
        'official_runtime_files': '',
    }
    for group, prefix in prefixes.items():
        for relative, expected in manifest[group].items():
            path = root / prefix / relative
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f'Unsafe V2.1 source path: {relative}')
            content = path.read_bytes()
            if len(content) != expected['size'] or hashlib.sha256(content).hexdigest() != expected['sha256']:
                raise ValueError(f'V2.1 source identity mismatch: {relative}')
    return manifest


def belongs_to_task(name: str, selected: str) -> bool:
    """Include pinned task variants such as task_007_hitl, not just task_007."""
    return bool(re.fullmatch(rf'task_{selected}(?:_[A-Za-z0-9_-]+)?', name.split('/')[0]))


def validate_task_assets(root: Path, asset_root: Path, selected: str) -> list[str]:
    entries = json.loads((root / 'benchmark_releases/team-osworld-v2.1.assets.json').read_text())
    manifest_hash = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if manifest_hash != '2cba90b03937e2d87a285d37a7fc1becec9460b2a066d7c3e2f555001a0c1722':
        raise ValueError('V2.1 asset manifest digest mismatch')
    validated = []
    for entry in entries:
        name = entry['path']
        if not belongs_to_task(name, selected):
            continue
        path = asset_root / name
        if not path.resolve().is_relative_to(asset_root.resolve()):
            raise ValueError(f'Unsafe V2.1 asset path: {name}')
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError(f'Missing or mismatched V2.1 asset: {name}')
        etag = entry['etag']
        if len(etag) == 64:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != etag:
                raise ValueError(f'V2.1 asset digest mismatch: {name}')
        elif len(etag) == 40:
            data = path.read_bytes()
            if hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() != etag:
                raise ValueError(f'V2.1 asset Git blob mismatch: {name}')
        else:
            raise ValueError('Unknown asset digest format')
        validated.append(name)
    return validated
