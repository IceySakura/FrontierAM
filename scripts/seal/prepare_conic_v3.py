"""Seal conic v3, the protocol v6.3 version of the conic task (results/EVAL_TIME_LIMITS.md, results/PROTOCOL_V63.md).

Derived from the sealed v2 runtimes (.runtime-conic-solver1105-skip for development and formal grading,
.runtime-conic-public for public feedback) as hard-linked copies; edited files are unlinked before writing:
  - cases the reference cannot solve kept the package's 300-second cap only because there is no reference time to
    calibrate; their limit becomes 18 s, so the worst-case evaluation fits the caps. Calibrated limits are unchanged;
  - the public-feedback runtime keeps every public case the reference solves plus 7 it does not (spread over groups);
  - the workspace: statement, pilot note, resources, interface examples, I/O tools and 5 samples under samples/;
  - the hash chain that names the manifest (score anchor, package lock, profile, runtime lock) is re-chained.

Usage: prepare_conic_v3.py [--force]
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEALS = {ROOT / '.runtime-conic-solver1105-skip': ROOT / '.runtime-conic-v3',
         ROOT / '.runtime-conic-public': ROOT / '.runtime-conic-v3-public'}
UNSOLVED_LIMIT = 18.0
PUBLIC_UNSOLVED = 7
SAMPLES = ['dev-8396c2285236', 'dev-03815674e04a', 'dev-4a42fbac0fcf', 'dev-afa1453420cc', 'dev-610c3f371053']
DOC_EDITS = {
    'INSTRUCTION.md': [
        ('；其余实例为 300 秒。', '；其余实例为 18 秒。'),
        ('163 个公开实例供开发，不计入最终指标。', '5 个样例实例供开发，不计入最终指标。'),
        ('列出 163 个输入的路径', '列出 5 个样例输入的路径'),
    ],
    'public-dev/README.md': [
        ('本目录包含 **163 个开发实例**及接口示例', '本目录包含 **5 个样例实例**及接口示例'),
        ('| `inputs/` | 163 个问题的 NPZ 输入 |', '| `inputs/` | 5 个问题的 NPZ 输入（每个锥类型 1 个） |'),
    ],
}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def once(text: str, old: str, new: str, where: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f'{where}: expected exactly one occurrence of {old[:70]!r}')
    return text.replace(old, new)


def rewrite(path: Path, data) -> None:
    path.unlink()
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data, indent=2) + '\n')
    else:
        path.write_bytes(data if isinstance(data, bytes) else data.encode())


def workspace(parent_tar: Path, limits: dict[str, float]) -> bytes:
    members = {}
    with tarfile.open(parent_tar) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if name.startswith('public-dev/inputs/') and Path(name).stem not in SAMPLES:
                continue
            members[name] = archive.extractfile(member).read()
    manifest = json.loads(members['public-dev/manifest.json'])
    manifest['instances'] = [dict(row, time_limit_seconds=limits.get(row['id'], row['time_limit_seconds']))
                             for row in manifest['instances'] if row['id'] in SAMPLES]
    if len(manifest['instances']) != len(SAMPLES):
        raise SystemExit('not every sample is in the public manifest')
    members['public-dev/manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    for name, edits in DOC_EDITS.items():
        text = members[name].decode()
        for old, new in edits:
            text = once(text, old, new, name)
        members[name] = text.encode()
    text = members['INSTRUCTION.md'].decode()
    if 'public-dev/' not in text:
        raise SystemExit('INSTRUCTION.md no longer names public-dev/')
    members['INSTRUCTION.md'] = text.replace('/workspace/public-dev/', '/workspace/samples/').replace('public-dev/', 'samples/').encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(members):
            info = tarfile.TarInfo(name.replace('public-dev/', 'samples/', 1))
            info.size, info.mode, info.mtime, info.uid, info.gid = len(members[name]), 0o644, 0, 1000, 1000
            archive.addfile(info, io.BytesIO(members[name]))
    return buffer.getvalue()


def rechain(task: Path, profile: dict) -> None:
    """The score anchor names the manifest, the package lock names both, the profile locks all three."""
    manifest = task / 'tests/private/data/manifest.json'
    anchor_path = task / 'tests/private/score-anchor.json'
    anchor = json.loads(anchor_path.read_text())
    anchor['evaluation_manifest_sha256'] = sha(manifest)
    rewrite(anchor_path, anchor)
    lock_path = task / 'tests/private/package-lock.json'
    lock = json.loads(lock_path.read_text())
    for relative in ('private/data/manifest.json', 'private/score-anchor.json'):
        path = task / 'tests' / relative
        lock['files'][relative] = {'sha256': sha(path), 'size_bytes': path.stat().st_size}
    rewrite(lock_path, lock)
    for relative in ('tests/private/data/manifest.json', 'tests/private/score-anchor.json', 'tests/private/package-lock.json'):
        if relative in profile['locked_files']:
            profile['locked_files'][relative] = sha(task / relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    held_out = json.loads((ROOT / '.runtime-conic-solver1105-skip/tasks/conic/original/tests/private/data/manifest.json').read_text())
    public_dev = json.loads((ROOT / '.runtime-conic-solver1105-skip/tasks/conic/original/environment/public-dev/manifest.json').read_text())
    limits = {row['id']: UNSOLVED_LIMIT for row in held_out['instances'] + public_dev['instances']
              if row['time_limit_seconds'] >= 300}
    ws = workspace(ROOT / '.runtime-conic-solver1105-skip/conic-public.tar', limits)
    for parent, seal in SEALS.items():
        if seal.exists():
            if not args.force:
                raise SystemExit(f'{seal} exists; pass --force to reseal')
            shutil.rmtree(seal)
        shutil.copytree(parent, seal, symlinks=True, copy_function=os.link)
        found = subprocess.run(['rg', '-l', '--max-depth', '6', '-g', '*.json', str(parent), str(seal)],
                               capture_output=True, text=True).stdout.split()
        for path in map(Path, found):
            if 'tests/private/data' not in str(path):
                rewrite(path, path.read_text().replace(str(parent), str(seal)))
        task = seal / 'tasks/conic/original'
        manifest_path = task / 'tests/private/data/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        rows = [dict(row, time_limit_seconds=limits.get(row['id'], row['time_limit_seconds'])) for row in manifest['instances']]
        kept_unsolved = None
        if seal.name.endswith('-public'):
            unsolved = sorted((r for r in rows if r['id'] in limits), key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
            by_group: dict[str, list] = {}
            for row in unsolved:
                by_group.setdefault(row['group'], []).append(row)
            picked = []
            while len(picked) < PUBLIC_UNSOLVED and any(by_group.values()):
                for group in sorted(by_group):
                    if by_group[group] and len(picked) < PUBLIC_UNSOLVED:
                        picked.append(by_group[group].pop(0)['id'])
            rows = [r for r in rows if r['id'] not in limits or r['id'] in picked]
            kept_unsolved = picked
        manifest['instances'] = rows
        rewrite(manifest_path, manifest)
        profile_path = seal / 'conic-profile.json'
        profile = json.loads(profile_path.read_text())
        rechain(task, profile)
        rewrite(seal / 'conic-public.tar', ws)
        profile['public_tar_sha256'] = sha(seal / 'conic-public.tar')
        if profile.get('public_feedback'):
            profile['public_feedback'].update(instances=len(rows), manifest_sha256=sha(manifest_path))
        profile['v63'] = {'unsolved_limit_s': UNSOLVED_LIMIT, 'changed_limits': sum(r['id'] in limits for r in rows),
                          'instances': len(rows), 'public_unsolved_kept': kept_unsolved, 'samples': SAMPLES,
                          'derived_from': parent.name}
        rewrite(profile_path, profile)
        lock_path = seal / 'runtime-lock.json'
        lock = json.loads(lock_path.read_text())
        lock['profile_sha256'] = sha(profile_path)
        rewrite(lock_path, lock)
        print(json.dumps({'sealed': str(seal), 'instances': len(rows), 'limit_changes': profile['v63']['changed_limits'],
                          'sum_limits_s': round(sum(r['time_limit_seconds'] for r in rows))}))


if __name__ == '__main__':
    main()
