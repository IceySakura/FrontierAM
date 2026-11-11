"""Seal .runtime-exp3-cheb2: exponential-v3 with a one-member panel, the Chebyshev v2 reference (independent human
implementation of published techniques, tasks/exponential/references/candidates/chebyshev2), so that member alone can be measured
on the held-out cases; its measurement then joins the v2 panel's for the virtual best (results/BASELINE_REVIEW.md).

Usage: prepare_exp_v3_cheb2.py [--force]
"""

import argparse
import hashlib
import json
import os
import shutil

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT, SEAL = ROOT / '.runtime-exp3', ROOT / '.runtime-exp3-cheb2'
MEMBER = ROOT / 'tasks/exponential/references/candidates/chebyshev2'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if SEAL.exists():
        if not args.force:
            raise SystemExit(f'{SEAL} exists; pass --force')
        shutil.rmtree(SEAL)
    shutil.copytree(PARENT, SEAL, symlinks=True, copy_function=os.link,
                    ignore=lambda d, names: ['panel'] if Path(d) == PARENT else [])
    shutil.copytree(MEMBER.resolve(), SEAL / 'panel/chebyshev2', ignore=shutil.ignore_patterns('__pycache__'))
    lock = json.loads((SEAL / 'lock.json').read_text())
    (SEAL / 'lock.json').unlink()
    lock['files'] = {k: v for k, v in lock['files'].items() if not k.startswith('panel/')}
    lock['files'].update({str(p.relative_to(SEAL)): sha(p) for p in sorted((SEAL / 'panel').rglob('*')) if p.is_file()})
    lock.update(panel=['chebyshev2'], panel_sealing={'chebyshev2': {'links': 0}},
                derived_from='.runtime-exp3', note='one-member panel to measure the Chebyshev v2 reference on the held-out cases')
    (SEAL / 'lock.json').write_text(json.dumps(lock, indent=1) + '\n')
    print(json.dumps({'sealed': str(SEAL), 'panel': lock['panel'],
                      'member_files': sum(k.startswith('panel/') for k in lock['files'])}))


if __name__ == '__main__':
    main()
