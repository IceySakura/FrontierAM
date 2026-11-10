"""Seal operator-delivered tasks (tasks/operator/<name>/task.json) into .runtime-ops/<name>/.

For each task: verify the raw packages in var/raw by SHA-256, extract them, verify the package's own file list,
apply the declared portability edits (each must match exactly once), assemble the development workspace files and
their tar, stage the image build context, optionally build the amd64 image, and write lock.json with every digest.

Usage: prepare_operator_tasks.py [--task NAME ...] [--build-images] [--force]
"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import zipfile

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'var/raw'
SPECS = ROOT / 'tasks/operator'
OUT = ROOT / '.runtime-ops'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract_zip(z: zipfile.ZipFile, target: Path) -> None:
    """extractall drops Unix modes; restore each file's permission bits recorded by a Unix zip tool (build.sh 0755)."""
    z.extractall(target)
    for member in z.infolist():
        mode = (member.external_attr >> 16) & 0o777
        if member.create_system == 3 and mode and not member.is_dir():
            (target / member.filename).chmod(mode)


def extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True)
    if archive.name.endswith('.zip'):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if member.filename.startswith('/') or '..' in Path(member.filename).parts:
                    raise SystemExit(f'unsafe path in {archive.name}: {member.filename}')
            extract_zip(z, target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target, filter='data')


def verify_integrity(spec: dict, roots: dict) -> dict:
    """One rule or a list of rules (one per package); every listed file must match."""
    rules = spec['integrity'] if isinstance(spec['integrity'], list) else [spec['integrity']]
    results = [verify_rule(spec, rule, roots) for rule in rules]
    return results[0] if len(results) == 1 else {'rules': results}


def verify_rule(spec: dict, rule: dict, roots: dict) -> dict:
    base = roots[rule['package']]
    listing = base / rule['file']
    if rule['type'] == 'sha256sums':
        entries = {}
        for line in listing.read_text().splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                entries[name.lstrip('*').strip()] = digest
    elif rule['type'] == 'manifest_files':
        entries = json.loads(listing.read_text())['files']
    else:
        raise SystemExit(f'unknown integrity rule {rule["type"]}')
    bad = [name for name, digest in entries.items() if sha(base / name) != digest]
    if bad:
        raise SystemExit(f'{spec["name"]}: {len(bad)} files fail {rule["file"]}: {bad[:5]}')
    return {'listing': rule['file'], 'files_verified': len(entries)}


def source_path(ref: str, roots: dict) -> tuple[Path, bool]:
    unzip = ref.startswith('unzip:')
    if unzip:
        ref = ref[len('unzip:'):]
    package, relative = ref.split(':', 1)
    return roots[package] / relative, unzip


def assemble_public(spec: dict, roots: dict, target: Path) -> None:
    target.mkdir(parents=True)
    for item in spec['public_files']:
        source, unzip = source_path(item['from'], roots)
        dest = target / item['to']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if unzip:
            with zipfile.ZipFile(source) as z:
                extract_zip(z, dest)
        elif source.is_dir():
            shutil.copytree(source, dest)
        else:
            shutil.copy2(source, dest)


def public_tar(folder: Path, archive: Path) -> None:
    def clean(info):
        info.uid = info.gid = 1000
        info.uname = info.gname = ''
        info.mtime = 0
        return info

    with tarfile.open(archive, 'w', format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(folder.rglob('*')):
            tar.add(path, arcname=str(path.relative_to(folder)), recursive=False, filter=clean)


def apply_edits(spec: dict, operator: Path) -> list:
    applied = []
    for edit in spec['portability_edits']:
        path = operator / edit['file']
        text = path.read_text()
        if text.count(edit['old']) != 1:
            raise SystemExit(f'{spec["name"]}: edit for {edit["file"]} must match exactly once')
        before = sha(path)
        path.write_text(text.replace(edit['old'], edit['new']))
        applied.append({'file': edit['file'], 'sha256_before': before, 'sha256_after': sha(path), 'reason': edit['reason']})
    return applied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', action='append')
    parser.add_argument('--build-images', action='store_true')
    parser.add_argument('--force', action='store_true', help='replace an existing sealed copy')
    args = parser.parse_args()
    # Derived tasks (e.g. species-tree-v2, scripts/prepare_species_v2.py) are sealed by their own scripts.
    names = args.task or sorted(p.parent.name for p in SPECS.glob('*/task.json')
                                if 'derived_from' not in json.loads(p.read_text()))
    for name in names:
        spec_dir = SPECS / name
        spec = json.loads((spec_dir / 'task.json').read_text())
        target = OUT / name
        if target.exists():
            if not args.force:
                raise SystemExit(f'{target} exists; pass --force to rebuild it')
            shutil.rmtree(target)
        roots, packages = {}, {}
        for key, package in spec['packages'].items():
            archive = RAW / package['file']
            digest = sha(archive)
            if digest != package['sha256']:
                raise SystemExit(f'{name}: {archive.name} sha256 {digest} != {package["sha256"]}')
            if 'extracted' in package:
                # Packages too large to copy (mcmc maintainer package: 29 GB) are referenced where the verified archive
                # was unpacked; the integrity listing below still covers every file. They take no portability edits.
                if key == 'operator' and spec.get('portability_edits'):
                    raise SystemExit(f'{name}: portability edits are not supported on an extracted package')
                (target / 'packages').mkdir(parents=True, exist_ok=True)
                (target / 'packages' / key).symlink_to(Path(package['extracted']).resolve(strict=True))
            else:
                extract(archive, target / 'packages' / key)
            roots[key] = target / 'packages' / key / package['root']
            packages[key] = {'file': package['file'], 'sha256': digest}
        integrity = verify_integrity(spec, roots)
        edits = apply_edits(spec, roots['operator']) if 'operator' in roots else []
        workspace = target / 'workspace'
        assemble_public(spec, roots, workspace)
        public_tar(workspace, target / 'public.tar')
        context = target / 'image-context'
        shutil.copytree(workspace, context)
        for extra in spec_dir.iterdir():
            if extra.name not in ('task.json',):
                shutil.copy2(extra, context / extra.name)
        image = spec['images']['task']
        image_id = None
        if args.build_images:
            subprocess.run(['docker', 'build', '-q', '-f', str(context / image['dockerfile']), '-t', image['tag'],
                            str(context)], check=True)
            image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image['tag']],
                                               text=True).strip()
        else:
            # Re-sealing without a rebuild keeps the image already built (and loaded into the grading daemons).
            found = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image['tag']],
                                   capture_output=True, text=True)
            image_id = found.stdout.strip() or None
        lock = {
            'name': name, 'status': spec['status'], 'sealed_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'spec_sha256': sha(spec_dir / 'task.json'), 'packages': packages, 'integrity': integrity,
            'portability_edits': edits, 'public_tar_sha256': sha(target / 'public.tar'),
            'image': {'tag': image['tag'], 'id': image_id, 'replaces': image['replaces']},
            'roots': {k: str(v.relative_to(target)) for k, v in roots.items()},
        }
        (target / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')
        print(json.dumps({'task': name, 'integrity': integrity, 'edits': len(edits), 'image': image_id}))


if __name__ == '__main__':
    main()
