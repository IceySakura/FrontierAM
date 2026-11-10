"""Seal galaxy-gravity-v2: 131,072-particle snapshots so that exact direct summation no longer fits the budget.

Operator side: the sealed v1 package (grader, sandbox, Treecode2 references) with the formal snapshot size check
lifted to N particles; private data are new snapshots from the public generator with private seeds and reference
accelerations from the package's own direct-summation program (private/provenance/direct). The public package gets
the updated statement, the same generator and smoke files, and public snapshots with their references.
(results/galaxy-gravity/GALAXY_SCALE.md)

Usage: prepare_galaxy_v2.py --data /data/datasets/galaxy-v2 [--n 131072] [--cpu-limit 5.0] [--force]
"""

import argparse
import hashlib
import json
import shutil
import tarfile

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / '.runtime-ops/galaxy-gravity'
SEAL = ROOT / '.runtime-ops/galaxy-gravity-v2'
SPEC = ROOT / 'tasks/operator/galaxy-gravity-v2'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_once(path: Path, old: str, new: str) -> dict:
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f'{path}: expected exactly one occurrence of {old!r}')
    before = sha(path)
    path.write_text(text.replace(old, new))
    return {'file': str(path.relative_to(SEAL)), 'old': old, 'new': new, 'sha256_before': before, 'sha256_after': sha(path)}


def snapshots(source: Path, n: int) -> list[dict]:
    cases = []
    for path in sorted(source.glob('case-*.in')):
        ref = path.with_suffix('.ref')
        if path.stat().st_size != 32 * n or not ref.exists() or ref.stat().st_size != 24 * n:
            raise SystemExit(f'{path}: incomplete snapshot or reference')
        cases.append({'case': path.stem, 'family': path.with_suffix('.family').read_text().strip(), 'in': path, 'ref': ref})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--n', type=int, default=131072)
    parser.add_argument('--cpu-limit', type=float, default=5.0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if SEAL.exists():
        if not args.force:
            raise SystemExit(f'{SEAL} exists; pass --force')
        shutil.rmtree(SEAL)
    v1 = json.loads((V1 / 'lock.json').read_text())
    shutil.copytree(V1 / 'packages', SEAL / 'packages', ignore=shutil.ignore_patterns('__pycache__'))
    operator = SEAL / v1['roots']['operator']
    edits = [replace_once(operator / 'tasks/galaxy-gravity/harbor/tests/grade.py', 'size != 65536*32', f'size != {args.n}*32')]
    data = operator / 'private/data'
    shutil.rmtree(data)
    data.mkdir()
    formal = []
    for case in snapshots(args.data / 'formal', args.n):
        shutil.copyfile(case['in'], data / f'{case["case"]}.in')
        shutil.copyfile(case['ref'], data / f'{case["case"]}.ref')
        formal.append({'case': case['case'], 'family': case['family'], 'input': f'private/data/{case["case"]}.in',
                       'truth': f'private/data/{case["case"]}.ref', 'input_sha256': sha(data / f'{case["case"]}.in'),
                       'truth_sha256': sha(data / f'{case["case"]}.ref')})
    config_path = operator / 'private/scoring-config.json'
    config = json.loads(config_path.read_text())
    config.update(cases=formal, cpu_limit_s=args.cpu_limit, particles=args.n)
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    public_root = SEAL / v1['roots']['public']
    dev = public_root / 'public-dev'
    cases_dir = dev / 'cases'
    cases_dir.mkdir()
    public = []
    for case in snapshots(args.data / 'public', args.n):
        for suffix, path in (('in', case['in']), ('ref', case['ref'])):
            shutil.copyfile(path, cases_dir / f'{case["case"]}.{suffix}')
        public.append({'case': case['case'], 'family': case['family'], 'input': f'cases/{case["case"]}.in',
                       'truth': f'cases/{case["case"]}.ref', 'input_sha256': sha(cases_dir / f'{case["case"]}.in'),
                       'truth_sha256': sha(cases_dir / f'{case["case"]}.ref')})
    (dev / 'manifest.json').write_text(json.dumps({'particles': args.n, 'cases': public}, indent=1) + '\n')
    edits.append(replace_once(public_root / 'problem.md', '正式快照每张含 65,536 个等质量粒子', f'正式快照每张含 {args.n:,} 个等质量粒子'))
    edits.append(replace_once(dev / 'README.md', 'python3 generate.py jaffe 65536 123456 dev.in',
                              f'python3 generate.py jaffe {args.n} 123456 dev.in'))
    (dev / 'README.md').write_text((dev / 'README.md').read_text() + (SPEC / 'public-dev-addendum.md').read_text())
    edits.append(replace_once(dev / 'check_output.py',
                              'parser.add_argument("output", type=Path)\nargs = parser.parse_args()\n'
                              'reference = Path(__file__).with_name("smoke.ref").read_bytes()',
                              'parser.add_argument("output", type=Path)\n'
                              'parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("smoke.ref"))\n'
                              'args = parser.parse_args()\nreference = args.reference.read_bytes()'))
    with tarfile.open(SEAL / 'public.tar', 'w', format=tarfile.PAX_FORMAT) as archive:
        def clean(member):
            member.uid = member.gid = 1000
            member.uname = member.gname = ''
            member.mtime = 0
            return member
        archive.add(public_root / 'problem.md', arcname='problem.md', filter=clean)
        archive.add(dev, arcname='public-dev', filter=clean)
    files = {str(p.relative_to(SEAL)): sha(p) for p in sorted(SEAL.rglob('*')) if p.is_file()}
    lock = {'name': 'galaxy-gravity-v2', 'status': 'formal', 'sealed_utc': datetime.now(timezone.utc).isoformat(),
            'spec_sha256': sha(SPEC / 'task.json'), 'derived_from': {'galaxy-gravity': v1['spec_sha256']},
            'particles': args.n, 'cpu_limit_s': args.cpu_limit, 'formal_cases': len(formal), 'public_cases': len(public),
            'edits': edits, 'public_tar_sha256': files['public.tar'], 'image': v1['image'], 'roots': v1['roots'],
            'files': files}
    (SEAL / 'lock.json').write_text(json.dumps(lock, indent=1) + '\n')
    print(json.dumps({k: v for k, v in lock.items() if k not in ('files', 'edits')}, indent=1))


if __name__ == '__main__':
    main()
