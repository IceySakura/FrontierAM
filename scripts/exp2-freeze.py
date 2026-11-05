"""Freeze the exponential-v2 panel after public calibration (scripts/exp2-public.sh).

Applies each member's selection from tasks/exponential/sota_v1/calibration_v2/ as its default, rebuilds Expokit inside
the task image, checks the unparameterised members' public validation, and writes calibration_v2/FROZEN.json with
the selected parameters and a SHA-256 of every panel file. scripts/exp2-private.sh refuses to run without it.

Refuses to freeze if a calibration found no qualifying setting; the member then stays out of the panel and the
reason is recorded instead (a member that fails public data is not part of the frozen reference).

Usage: exp2-freeze.py [--dry-run]
"""

import argparse
import hashlib
import json
import re
import subprocess

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'tasks/exponential/sota_v1'
PANEL = V2 / 'panel'
CAL = V2 / 'calibration_v2'
IMAGE = 'frontieram-exponential:port-v1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path: Path, pattern: str, value: str, dry: bool) -> None:
    text = path.read_text()
    new, count = re.subn(pattern, value, text, count=1)
    if count != 1:
        raise SystemExit(f'{path}: pattern not found exactly once: {pattern}')
    if not dry:
        path.write_text(new)


def validation(member: str) -> dict:
    runs = json.loads((CAL / f'{member}-validation.json').read_text())['runs']
    worst = max((e['scaled_error'] for r in runs if r['valid'] for e in r['errors']), default=None)
    failed = sorted({r['id'] for r in runs if not r['valid']})
    return {'cases': len({r['id'] for r in runs}), 'failed_cases': failed, 'max_scaled_error': worst}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    dry = args.dry_run
    members, excluded = {}, {}

    expokit = json.loads((CAL / 'expokit.json').read_text())
    if expokit['selected_multiplier'] is None:
        excluded['expokit'] = 'no tolerance multiplier on the grid passes every public case with margin 0.1'
    else:
        replace_once(PANEL / 'expokit/src/stream_expokit.c', r'#define TOL_MULTIPLIER_DEFAULT [0-9.eE+-]+',
                     f'#define TOL_MULTIPLIER_DEFAULT {expokit["selected_multiplier"]!r}', dry)
        if not dry:
            subprocess.run(['docker', 'run', '--rm', '--network', 'none', '-v', f'{PANEL / "expokit"}:/src', '-w', '/src',
                            '--entrypoint', '/bin/sh', IMAGE, 'build.sh'], check=True)
        members['expokit'] = {'tol_multiplier': expokit['selected_multiplier']}

    kiops = json.loads((CAL / 'kiops.json').read_text())
    if kiops['selected_multiplier'] is None:
        excluded['kiops'] = 'no tolerance multiplier on the grid passes every public case with margin 0.1'
    else:
        replace_once(PANEL / 'kiops/stream_kiops.m', r'if nargin < 1, multiplier = [0-9.eE+-]+; end',
                     f'if nargin < 1, multiplier = {kiops["selected_multiplier"]!r}; end', dry)
        members['kiops'] = {'tol_multiplier': kiops['selected_multiplier']}

    expmv = json.loads((CAL / 'expmv.json').read_text())
    selected = expmv.get('selected') or {}
    if set(selected) != {'p4', 'p7', 'p10'} or None in selected.values():
        excluded['expmv'] = f'no qualifying precision for some level: {selected}'
    else:
        replace_once(PANEL / 'expmv/stream_expmv.m', r"policy = struct\('p4', '[a-z]+', 'p7', '[a-z]+', 'p10', '[a-z]+'\);",
                     f"policy = struct('p4', '{selected['p4']}', 'p7', '{selected['p7']}', 'p10', '{selected['p10']}');", dry)
        members['expmv'] = {'precision': selected}

    for member in ('scipy', 'chebyshev'):
        check = validation(member)
        if check['failed_cases'] or (check['max_scaled_error'] or 0) > 1:
            excluded[member] = f'public validation failed: {check}'
        else:
            members[member] = {'public_validation': check}

    files = {str(p.relative_to(PANEL)): sha(p) for p in sorted(PANEL.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts}
    frozen = {'frozen_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'calibration_data': 'tasks/exponential/sota_v1/dataset_v2/cases/public (public only)',
              'members': members, 'excluded': excluded, 'panel_files_sha256': files}
    print(json.dumps({k: v for k, v in frozen.items() if k != 'panel_files_sha256'}, indent=1))
    if not dry:
        (CAL / 'FROZEN.json').write_text(json.dumps(frozen, indent=2) + '\n')


if __name__ == '__main__':
    main()
