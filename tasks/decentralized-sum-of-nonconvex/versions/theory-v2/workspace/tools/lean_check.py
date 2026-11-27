#!/usr/bin/env python3
"""The evaluation's formal check, run locally on your Lean project.

Checks that the fixed model files (Theory/Model.lean, Theory/Sanity.lean) equal the template's, checks
manifest.json, runs `lake build`, then `lake comparator` on the results the manifest marks `proved`:
Theory.Statements is the challenge, Theory.Proofs the solution, and the only permitted axioms are propext, Quot.sound
and Classical.choice. The evaluation runs the same comparator with a fixed lakefile and the same package versions,
inside a sandbox, taking only theory/formal/Theory/**/*.lean and replacing the fixed model files with the originals;
here it runs without the sandbox. A joint run takes about three to five minutes (the model imports all of Mathlib);
with --each, a rejected joint run is followed by one run per proved result.

Usage: python3 /workspace/tools/lean_check.py [--each] [FORMAL_DIR]   (default /workspace/submission/theory/formal)
"""

import argparse
import json
import re
import subprocess
import sys
import time

from pathlib import Path

AXIOMS = ['propext', 'Quot.sound', 'Classical.choice']
TEMPLATE = Path('/workspace/examples/lean_project')
FIXED = ('Theory/Model.lean', 'Theory/Sanity.lean')
STATUSES = ('proved', 'stated', 'informal')
CLASSES = ('stronger', 'weaker', 'equal', 'incomparable')
COORDINATES = ('kind', 'class', 'inputs', 'bound', 'validity', 'guarantee', 'anchors', 'rows')
ROWS = ('assumptions', 'upper_data', 'upper_network', 'lower_data', 'lower_network', 'resources')
MARKS = ('+', '=', '-', 'none')
DECL = re.compile(r'^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*(?:theorem|lemma)\s+([^\s(:{\[]+)',
                  re.MULTILINE)


def manifest_results(formal: Path) -> tuple[list[dict], list[str]]:
    problems = []
    try:
        rows = json.loads((formal / 'manifest.json').read_text())['results']
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [], [f'manifest.json: {type(error).__name__}: {error}']
    results = []
    for index, row in enumerate(rows):
        rid = row.get('id', f'#{index}') if isinstance(row, dict) else f'#{index}'
        if not isinstance(row, dict):
            problems.append(f'{rid}: not an object')
            continue
        lean = [row['lean']] if isinstance(row.get('lean'), str) else list(row.get('lean') or [])
        if row.get('status') not in STATUSES:
            problems.append(f'{rid}: status must be one of {STATUSES}')
        if row.get('status') == 'proved' and not lean:
            problems.append(f'{rid}: proved but no Lean theorem named')
        comparison = row.get('comparison') if isinstance(row.get('comparison'), dict) else {}
        if comparison.get('class') not in CLASSES:
            problems.append(f'{rid}: comparison.class must be one of {CLASSES}')
        if not comparison.get('human'):
            problems.append(f'{rid}: comparison.human names no human result')
        coordinates = row.get('coordinates') if isinstance(row.get('coordinates'), dict) else None
        if coordinates is None:
            problems.append(f'{rid}: coordinates (result card) missing')
        else:
            missing = [key for key in COORDINATES if key not in coordinates]
            if missing:
                problems.append(f'{rid}: coordinates lacks {missing}')
            rows = coordinates.get('rows') if isinstance(coordinates.get('rows'), dict) else {}
            bad = {key: value for key, value in rows.items() if key not in ROWS or value not in MARKS}
            if bad or set(ROWS) - set(rows):
                problems.append(f'{rid}: coordinates.rows needs each of {ROWS} marked one of {MARKS}')
        results.append({'id': rid, 'lean': lean, 'status': row.get('status')})
    return results, problems


def comparator(formal: Path, names: list[str], tag: str) -> bool:
    config = Path('/tmp/lean-check') / f'comparator-{tag}.json'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({'challenge_module': 'Theory.Statements', 'solution_module': 'Theory.Proofs',
                                  'theorem_names': names, 'permitted_axioms': AXIOMS}))
    start = time.monotonic()
    done = subprocess.run(['lake', 'comparator', '--inadvisably-no-sandbox', '--config', str(config)], cwd=formal,
                          capture_output=True, text=True)
    lines = [line for line in (done.stdout + done.stderr).splitlines()
             if line.strip() and not line.startswith('Exporting #[') and 'designed for use with the module system' not in line]
    print('\n'.join(f'  {line[:300]}' for line in lines[-12:]))
    verdict = 'accepted' if done.returncode == 0 else 'REJECTED'
    print(f'[{tag}] {verdict} ({time.monotonic() - start:.0f} s): {", ".join(names)}', flush=True)
    return done.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('formal', nargs='?', type=Path, default=Path('/workspace/submission/theory/formal'))
    parser.add_argument('--each', action='store_true', help='after a rejected joint run, judge each proved result alone')
    args = parser.parse_args()
    formal = args.formal.resolve()
    results, problems = manifest_results(formal)
    statements = formal / 'Theory/Statements.lean'
    declared = {name.rsplit('.', 1)[-1] for name in DECL.findall(statements.read_text())} if statements.exists() else set()
    proved = [r for r in results if r['status'] == 'proved' and r['lean']]
    for row in proved:
        missing = [name for name in row['lean'] if name.rsplit('.', 1)[-1] not in declared]
        if missing:
            problems.append(f'{row["id"]}: not declared in Theory/Statements.lean: {missing}')
    for problem in problems:
        print(f'manifest: {problem}')
    for module in ('Statements', 'Proofs'):
        if not (formal / f'Theory/{module}.lean').exists():
            sys.exit(f'LEAN-CHECK: missing Theory/{module}.lean')
    changed = [name for name in FIXED
               if not (formal / name).is_file() or (formal / name).read_bytes() != (TEMPLATE / name).read_bytes()]
    if changed:
        sys.exit(f'LEAN-CHECK: {", ".join(changed)} differ from the template; the evaluation uses the originals. Restore '
                 f'them: cp {TEMPLATE}/Theory/Model.lean {TEMPLATE}/Theory/Sanity.lean {formal}/Theory/ '
                 '&& chmod u+w ' + ' '.join(str(formal / name) for name in FIXED))
    print('lake build ...', flush=True)
    build = subprocess.run(['lake', 'build'], cwd=formal, capture_output=True, text=True)
    errors = [line for line in (build.stdout + build.stderr).splitlines() if 'error' in line.lower()]
    sorries = [line for line in (build.stdout + build.stderr).splitlines() if 'declaration uses `sorry`' in line]
    print('\n'.join(f'  {line[:300]}' for line in errors[:20]))
    print(f'lake build: {"ok" if build.returncode == 0 else "FAILED"}; {len(sorries)} declarations use sorry '
          '(expected in Statements only)', flush=True)
    if build.returncode != 0:
        sys.exit('LEAN-CHECK: build failed')
    if not proved:
        sys.exit('LEAN-CHECK: no result marked proved')
    names = list(dict.fromkeys(name for row in proved for name in row['lean']))
    accepted = {row['id'] for row in proved} if comparator(formal, names, 'all') else set()
    if not accepted and args.each and len(proved) > 1:
        accepted = {row['id'] for row in proved if comparator(formal, row['lean'], row['id'])}
    rejected = [row['id'] for row in proved if row['id'] not in accepted]
    print(f'LEAN-CHECK: {len(accepted)}/{len(proved)} proved results accepted'
          + (f'; rejected or not judged alone: {", ".join(rejected)}' if rejected else '')
          + (f'; {len(problems)} manifest problems' if problems else ''))


if __name__ == '__main__':
    main()
