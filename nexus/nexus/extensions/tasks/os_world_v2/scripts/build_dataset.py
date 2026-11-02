"""Build 108 pinned V2 task bundles for `nexus dataset publish`."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import re
import subprocess
import tarfile

from pathlib import Path

from nexus.extensions.tasks.os_world_v2.adapter.source import (
    COMMIT,
    RELEASE,
    belongs_to_task,
    validate_task_assets,
    verify,
)
from nexus.extensions.tasks.os_world_v2.data.bundle import MANIFEST, PART_BYTES, Manifest, digest_file, part_name


def split_archive(path: Path, *, limit: int = PART_BYTES):
    """Bound dataset ZIP entries, including an individually huge asset, without weakening ZIP checks."""
    if path.stat().st_size <= limit:
        return []
    parts = []
    with path.open('rb') as stream:
        index = 0
        while chunk := stream.read(limit):
            target = path.with_name(part_name(path.name, index))
            target.write_bytes(chunk)
            parts.append(digest_file(target))
            index += 1
    path.unlink()
    return parts


def archive_files(output: Path, entries):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for name, path in sorted(entries):
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f'Refusing nonregular source: {name}')
                info = tarfile.TarInfo(name)
                info.size, info.mode = path.stat().st_size, (0o755 if path.stat().st_mode & 0o111 else 0o644)
                info.mtime, info.uid, info.gid = 0, 0, 0
                with path.open('rb') as source:
                    tar.addfile(info, source)


def source_paths(root: Path):
    raw = subprocess.check_output([
        'git',
        '-C',
        str(root),
        'ls-tree',
        '-r',
        '--name-only',
        COMMIT,
        'desktop_env',
        'evaluation_examples',
        'benchmark_releases',
        'task_loader.py',
    ])
    result = []
    for name in raw.decode().splitlines():
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue  # gitlink desktop_env/server is supplied by the VM image, not a host dependency.
        if path.name.startswith('.env') or any(p in {'.git', '__pycache__'} for p in path.parts):
            raise ValueError('Secret/cache path in source snapshot')
        # Do not silently publish uncommitted source/config changes.
        expected = subprocess.check_output(['git', '-C', str(root), 'show', f'{COMMIT}:{name}'])
        if path.read_bytes() != expected:
            raise ValueError(f'Uncommitted source differs from pinned commit: {name}')
        result.append((name, path))
    return result


def instruction(root: Path, selected: str):
    tree = ast.parse((root / f'evaluation_examples/task_class/task_{selected}.py').read_text())
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments.setdefault(node.target.id, node.value)

    def render(node, seen=frozenset()):
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
            return render(assignments[node.id], seen | {node.id})
        if isinstance(node, ast.JoinedStr):
            return ''.join(render(value, seen) for value in node.values)
        if isinstance(node, ast.FormattedValue):
            return render(node.value, seen)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return render(node.left, seen) + render(node.right, seen)
        # Display metadata only: dynamic task instructions are computed by the
        # actual pinned Python class at execution, never replaced by this text.
        return '{runtime:' + ast.unparse(node) + '}'

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'instruction' for t in statement.targets
                ):
                    return render(statement.value)
    raise ValueError(f'No instruction assignment for {selected}; do not execute tasks during packaging')


def build(
    root: Path, assets: Path, output: Path, dependency_root: Path | None = None, dependency_profile: str | None = None
):
    from nexus.extensions.tasks.os_world_v2.adapter.task033 import PROFILE, dependency_entries

    if dependency_profile not in (None, PROFILE):
        raise ValueError('Unsupported dependency profile')
    verify(root)
    ids = json.loads((root / 'evaluation_examples/test_v2.json').read_text())['tasks']
    instructions = {selected: instruction(root, selected) for selected in ids}
    entries = json.loads((root / 'benchmark_releases/team-osworld-v2.1.assets.json').read_text())
    shared = [e for e in entries if e['path'].startswith('shared/')]
    by_task = {selected: [e for e in entries if belongs_to_task(e['path'], selected)] for selected in ids}
    for selected in ids:
        validate_task_assets(root, assets, selected)
    for entry in shared:
        path = assets / entry['path']
        data = path.read_bytes()
        digest = (
            hashlib.sha256(data).hexdigest()
            if len(entry['etag']) == 64
            else hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        )
        if digest != entry['etag'] or len(data) != entry['size']:
            raise ValueError(f'Shared asset digest mismatch: {entry["path"]}')
    output.mkdir(parents=True, exist_ok=False)
    common = output / '_build'
    archive_files(common / 'source.tar.gz', source_paths(root))
    archive_files(common / 'shared-assets.tar.gz', [(e['path'], assets / e['path']) for e in shared])
    common_digests = {name: digest_file(common / name) for name in ('source.tar.gz', 'shared-assets.tar.gz')}
    policy = json.loads((root / 'evaluation_examples/setup_support/policies.json').read_text())
    item_paths, report = [], []
    for selected in ids:
        directory = output / 'items' / selected
        files = directory / 'files'
        files.mkdir(parents=True)
        for name in common_digests:
            os.link(common / name, files / name)
        # Include literal cross-task asset references, without executing any task code.
        text = (root / f'evaluation_examples/task_class/task_{selected}.py').read_text()
        related = {selected, *re.findall(r'task_(\d{3})(?:_[A-Za-z0-9_-]+)?/', text)}
        task_entries = [e for key in related for e in by_task.get(key, [])]
        archive_files(files / 'task-assets.tar.gz', [(e['path'], assets / e['path']) for e in task_entries])
        missing, dependencies = [], []
        entries = (
            dependency_entries()
            if selected == '033' and dependency_profile
            else policy.get('external_dependencies', {}).get(selected, [])
        )
        for entry in entries:
            path = dependency_root / entry['path'] if dependency_root else None
            if path is None or not path.is_file():
                missing.append(entry['path'])
            elif digest_file(path).sha256 != entry['sha256']:
                raise ValueError(f'External dependency checksum mismatch: {entry["path"]}')
            else:
                dependencies.append((entry['path'], path))
        archives = {**common_digests, 'task-assets.tar.gz': digest_file(files / 'task-assets.tar.gz')}
        if dependencies and not missing:
            archive_files(files / 'dependencies.tar.gz', dependencies)
            archives['dependencies.tar.gz'] = digest_file(files / 'dependencies.tar.gz')
        archive_parts = {}
        for name in archives:
            parts = split_archive(files / name)
            if parts:
                archive_parts[name] = parts
        manifest = Manifest(
            source_commit=COMMIT,
            benchmark_release=RELEASE,
            task_id=selected,
            archives=archives,
            archive_parts=archive_parts,
            missing_external_dependencies=missing,
            dependency_profile=dependency_profile if selected == '033' else None,
        )
        raw = json.dumps(manifest.model_dump(), sort_keys=True, separators=(',', ':')).encode()
        (files / MANIFEST).write_bytes(raw)
        metadata = {
            'task_id': selected,
            'domain': 'tasks',
            'instruction': instructions[selected],
            'instruction_is_display_template': '{runtime:' in instructions[selected],
            'benchmark_release': RELEASE,
            'source_commit': COMMIT,
            'bundle_sha256': hashlib.sha256(raw).hexdigest(),
            'missing_external_dependencies': missing,
        }
        item = {
            'id': selected,
            'files': {'type': 'directory', 'path': 'files'},
            'body': {
                'task_name': 'os_world_v2',
                'task_config': {
                    'task_id': selected,
                    'domain': 'tasks',
                    'bundle_sha256': metadata['bundle_sha256'],
                },
            },
            'metadata': metadata,
        }
        (directory / 'item.yaml').write_text(json.dumps(item, ensure_ascii=False, indent=2))
        item_paths.append(f'items/{selected}/item.yaml')
        report.append(metadata)
    (output / 'dataset.yaml').write_text(
        json.dumps(
            {
                'schema_version': 1,
                'name': 'osworld_v2',
                'default_split': 'test',
                'splits': {'test': {'items': item_paths}},
            },
            indent=2,
        )
    )
    (output / 'build-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return {
        'task_count': len(ids),
        'missing_dependency_task_ids': [r['task_id'] for r in report if r['missing_external_dependencies']],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path)
    parser.add_argument('--dependency-profile')
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.assets, args.output, args.dependencies, args.dependency_profile)))


if __name__ == '__main__':
    main()
