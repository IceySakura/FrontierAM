"""Copy each v6.3 task version's data into the repository: tasks/<family>/versions/<v>/data/{samples,public,private}.

samples come from the version's workspace package (what the model receives); public is the hidden public-feedback set;
private is the held-out set. Each tier keeps the sealed manifest that lists its files. public/ and private/ go through
Git LFS (.gitattributes) and never reach the GitHub mirror (scripts/ops/sync_github.sh); samples are ordinary files.

Usage: export_v63_data.py [TASK ...]      (default: all six)
"""

import hashlib
import json
import os
import shutil
import sys
import tarfile

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / '.runtime-ops'
sys.path.insert(0, str(ROOT))


def manifest_files(base: Path, manifest: str, key: str, fields=('path',)) -> list[str]:
    data = json.loads((base / manifest).read_text())
    rows = data[key] if isinstance(data, dict) else data
    return [manifest] + [row[f] for row in rows for f in fields if f in row]


def operator_root(name: str, package: str = 'operator') -> Path:
    lock = json.loads((OPS / name / 'lock.json').read_text())
    return OPS / name / lock['roots'][package]


def trajectory(pool: str) -> tuple[Path, list[str], dict]:
    import arena.tasks  # noqa: F401
    from arena.tasks import TASKS

    task = TASKS['trajectory-covariance-v2']
    if pool == 'feedback':
        os.environ['FRONTIERAM_TRAJ_POOL'] = 'feedback'
    else:
        os.environ.pop('FRONTIERAM_TRAJ_POOL', None)
    ids = set(task.case_ids())
    base = task.root() / 'results/mcmc-long-run-covariance/formal-v1/private'
    cases = [c for c in json.loads((base / 'manifest.json').read_text())['cases'] if c['id'] in ids]
    return base, [p for c in cases for p in (c['input'], c['reference'])], {'cases': cases}


TASKS = {
    'conic': {'version': 'v3', 'workspace': (ROOT / '.runtime-conic-v3/conic-public.tar', 'samples/'),
              'public': lambda: (ROOT / '.runtime-conic-v3-public/tasks/conic/original/tests/private/data', 'instances'),
              'private': lambda: (ROOT / '.runtime-conic-v3/tasks/conic/original/tests/private/data', 'instances')},
    'exponential': {'version': 'v3', 'workspace': (ROOT / '.runtime-exp3/public.tar', 'task/samples/'),
                    'public': lambda: (ROOT / '.runtime-exp3-public/data', 'cases', ('file',)),
                    'private': lambda: (ROOT / '.runtime-exp3/data', 'cases', ('file',))},
    'galaxy-gravity': {'version': 'v3', 'workspace': (OPS / 'galaxy-gravity-v3/public.tar', 'samples/'),
                       'public': lambda: (operator_root('galaxy-gravity-v3', 'public') / 'public-dev', 'cases', ('input', 'truth')),
                       'private': lambda: (operator_root('galaxy-gravity-v3'), 'cases', ('input', 'truth'), 'private/scoring-config.json')},
    'polarization-demosaicking': {
        'version': 'v2', 'workspace': (OPS / 'polarization-demosaicking-v2/public.tar', 'samples/'),
        'public': lambda: (OPS / 'polarization-demosaicking-feedback', 'scenes', ('file',)),
        'private': 'polarization'},
    'species-tree': {'version': 'v3', 'workspace': (OPS / 'species-tree-v3/public.tar', 'samples/'),
                     'public': lambda: (operator_root('species-tree-v3', 'public') / 'public-dev', 'cases', ('input', 'truth')),
                     'private': lambda: (OPS / 'species-tree-v3/private', 'cases', ('input', 'truth'))},
    'trajectory-covariance': {'version': 'v2', 'workspace': (OPS / 'trajectory-covariance-v2/public.tar', 'samples/'),
                              'public': 'trajectory-feedback', 'private': 'trajectory-formal'},
}


def copy(base: Path, relatives: list[str], target: Path) -> int:
    total = 0
    for relative in dict.fromkeys(relatives):
        source, dest = base / relative, target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        total += dest.stat().st_size
    return total


def tier(spec, target: Path) -> dict:
    if spec == 'polarization':
        root = operator_root('polarization-demosaicking')
        data = root / 'results/polarization-demosaicking/table-iii-40/data'
        files = [str(p.relative_to(data)) for p in sorted(data.rglob('*')) if p.is_file()]
        size = copy(data, files, target)
        size += copy(root / 'tasks/polarization-demosaicking', ['test_scenes.txt'], target)
        return {'files': len(files) + 1, 'bytes': size}
    if spec in ('trajectory-feedback', 'trajectory-formal'):
        base, files, manifest = trajectory('feedback' if spec == 'trajectory-feedback' else 'formal')
        size = copy(base, files, target)
        (target / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
        return {'files': len(files) + 1, 'bytes': size}
    base, key, *rest = spec()
    fields = rest[0] if rest else ('path',)
    files = manifest_files(base, 'manifest.json' if len(rest) < 2 else rest[1], key, fields)
    if len(rest) >= 2:  # galaxy private: the scoring config lists the cases
        files = [rest[1]] + [row[f] for row in json.loads((base / rest[1]).read_text())[key] for f in fields]
    size = copy(base, files, target)
    return {'files': len(set(files)), 'bytes': size}


def main():
    names = sys.argv[1:] or list(TASKS)
    for name in names:
        cfg = TASKS[name]
        data = ROOT / 'tasks' / name / 'versions' / cfg['version'] / 'data'
        shutil.rmtree(data, ignore_errors=True)
        report = {}
        archive, prefix = cfg['workspace']
        with tarfile.open(archive) as package:
            for member in package.getmembers():
                if member.isfile() and member.name.startswith(prefix):
                    dest = data / 'samples' / member.name[len(prefix):]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(package.extractfile(member).read())
        report['samples'] = sum(1 for p in (data / 'samples').rglob('*') if p.is_file())
        report['public'] = tier(cfg['public'], data / 'public')
        report['private'] = tier(cfg['private'], data / 'private')
        print(json.dumps({'task': name, **report}))


if __name__ == '__main__':
    main()
