"""Seal species-tree-v2: the rebuilt, larger species-tree task (results/SPECIES_TREE_SIM.md, SPECIES_TREE_SCALE.md).

The operator side is the sealed v1 package (its batch grader, sandboxed worker, TREE-QMC references and portability
edits), with the fixed taxon count lifted from 101 to TAXA in the scorer (every edit must match its original text
exactly once). Data are our own S100-style simulations: DATA/formal (held out) and DATA/public (shipped). Layout:

  packages/operator/        v1 operator directory without its private data, edited for TAXA
  private/                  cases/NNN/{genes.tre,truth.tre} and manifest.json (sha256 per file); never shipped
  packages/public/          problem.md, public-dev/{README.md,check_rf.py,cases/NNN/{genes.tre,truth.tre}}
  public.tar                packages/public as shipped to the development container
  lock.json                 hashes of every sealed file, image, edits

Usage: prepare_species_v2.py --data /data/datasets/species-sim/v2 [--taxa 250] [--force]
"""

import argparse
import hashlib
import io
import json
import shutil
import tarfile

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / '.runtime-ops/species-tree'
SEAL = ROOT / '.runtime-ops/species-tree-v2'
SPEC = ROOT / 'tasks/operator/species-tree-v2'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def edits(taxa: int) -> list[tuple[str, str, str]]:
    splits = taxa - 3
    return [
        ('runtime/score.py', 'if taxa != {str(i) for i in range(101)} or len(truth) != 98:',
         f'if taxa != {{str(i) for i in range({taxa})}} or len(truth) != {splits}:'),
        ('runtime/score.py', '"Expected a binary truth tree on labels 0 through 100"',
         f'"Expected a binary truth tree on labels 0 through {taxa - 1}"'),
        ('runtime/score.py', '"n_leaves": 101, "missed_splits": 98, "possible_splits": 98, "normalized_rf": 1.0}',
         f'"n_leaves": {taxa}, "missed_splits": {splits}, "possible_splits": {splits}, "normalized_rf": 1.0}}'),
        ('runtime/score.py', 'return {"status": "ok", "n_leaves": 101, "missed_splits": missed,',
         f'return {{"status": "ok", "n_leaves": {taxa}, "missed_splits": missed,'),
        ('runtime/score.py', '"possible_splits": 98, "normalized_rf": missed / 98}',
         f'"possible_splits": {splits}, "normalized_rf": missed / {splits}}}'),
        ('grade.py', 'denominator=n*98,mean_nrf=None if total is None else total/(n*98),',
         f'denominator=n*{splits},mean_nrf=None if total is None else total/(n*{splits}),'),
        ('grade.py', 'r.update(missed_splits=98,possible_splits=98,normalized_rf=1.0)',
         f'r.update(missed_splits={splits},possible_splits={splits},normalized_rf=1.0)'),
    ]


def copy_cases(source: Path, target: Path) -> dict:
    cases = []
    for directory in sorted(p for p in source.iterdir() if p.is_dir()):
        if not (directory / 'genes.tre').exists() or not (directory / 'truth.tre').exists():
            raise SystemExit(f'{directory}: incomplete case')
        out = target / 'cases' / directory.name
        out.mkdir(parents=True)
        for name in ('genes.tre', 'truth.tre'):
            shutil.copyfile(directory / name, out / name)
        cases.append({'case': directory.name, 'input': f'cases/{directory.name}/genes.tre',
                      'input_sha256': sha(out / 'genes.tre'), 'truth': f'cases/{directory.name}/truth.tre',
                      'truth_sha256': sha(out / 'truth.tre')})
    manifest = {'schema': 'frontieram/species-tree-v2-cases-1', 'cases': cases}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--taxa', type=int, default=250)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if SEAL.exists():
        if not args.force:
            raise SystemExit(f'{SEAL} exists; pass --force to replace it')
        shutil.rmtree(SEAL)
    v1 = json.loads((V1 / 'lock.json').read_text())
    operator = SEAL / 'packages/operator'
    shutil.copytree(V1 / v1['roots']['operator'], operator, ignore=shutil.ignore_patterns('private-data', '__pycache__'))
    applied = []
    for relative, old, new in edits(args.taxa):
        path = operator / 'operator' / relative
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(f'{relative}: expected exactly one occurrence of {old!r}')
        before = sha(path)
        path.write_text(text.replace(old, new))
        applied.append({'file': f'operator/{relative}', 'old': old, 'new': new, 'sha256_before': before,
                        'sha256_after': sha(path)})
    formal = copy_cases(args.data / 'formal', SEAL / 'private')
    public_root = SEAL / 'packages/public'
    dev = public_root / 'public-dev'
    public = copy_cases(args.data / 'public', dev)
    shutil.copyfile(V1 / v1['roots']['public'] / 'public-dev/check_rf.py', dev / 'check_rf.py')
    for name in ('problem.md', 'README.md'):
        target = public_root / 'problem.md' if name == 'problem.md' else dev / 'README.md'
        target.write_text((SPEC / ('problem.md' if name == 'problem.md' else 'public-dev-README.md')).read_text())
    with tarfile.open(SEAL / 'public.tar', 'w', format=tarfile.PAX_FORMAT) as archive:
        def clean(member):
            member.uid = member.gid = 1000
            member.uname = member.gname = ''
            member.mtime = 0
            return member
        archive.add(public_root / 'problem.md', arcname='problem.md', filter=clean)
        archive.add(dev, arcname='public-dev', filter=clean)
    spec = json.loads((SPEC / 'task.json').read_text())
    files = {str(p.relative_to(SEAL)): sha(p) for p in sorted(SEAL.rglob('*')) if p.is_file()}
    lock = {
        'name': spec['name'], 'status': spec['status'], 'sealed_utc': datetime.now(timezone.utc).isoformat(),
        'spec_sha256': sha(SPEC / 'task.json'), 'derived_from': {'species-tree': v1['spec_sha256']},
        'taxa': args.taxa, 'formal_cases': len(formal['cases']), 'public_cases': len(public['cases']),
        'scorer_edits': applied, 'public_tar_sha256': files['public.tar'], 'image': v1['image'],
        'roots': {'operator': 'packages/operator', 'public': 'packages/public'}, 'files': files,
    }
    (SEAL / 'lock.json').write_text(json.dumps(lock, indent=1) + '\n')
    print(json.dumps({k: v for k, v in lock.items() if k not in ('files', 'scorer_edits')}, indent=1))


if __name__ == '__main__':
    main()
