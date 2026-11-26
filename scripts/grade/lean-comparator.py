#!/usr/bin/env python3
"""Formal check of a theory deliverable with the official `lake comparator` (Lean v4.35, Mathlib and CSLib v4.35).

The challenge is `Theory.Statements` and the solution `Theory.Proofs`, both from the submission's `theory/formal/`.
Only `theory/formal/Theory/**/*.lean` is taken from the submission; the lakefile, the package manifest and the
toolchain are fixed here. The comparator evaluates the configuration and builds and exports the untrusted code inside
its bwrap sandbox and replays the solution in the kernel; this script also runs the comparator itself as the
unprivileged pilot user in a fresh network namespace. All results the manifest marks `proved` are judged together;
if that run rejects, each such result is judged alone.

The statement layer (Statements and the submission modules it imports) is what a reviewer reads to decide whether a
formal statement says what THEORY.md claims; commands that can make source text mean something else there
(notation, macros, elaboration-time code) are listed for the reviewer, not rejected.

Usage: lean-comparator.py SUBMISSION_SOURCE OUT_DIR [--paranoid] [--timeout SECONDS]
Writes OUT_DIR/lean-check.json and OUT_DIR/comparator-*.log.
"""

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT = Path('/data/workspace/frontieram-theory-pilot')
TOOLCHAIN = PILOT / '.elan/toolchains/leanprover--lean4---v4.35.0'
PACKAGES = PILOT / 'lean/.lake/packages'
MANIFEST = PILOT / 'lean/lake-manifest.json'
UID = 2600
AXIOMS = ['propext', 'Quot.sound', 'Classical.choice']
PATH = f'{ROOT}/scripts/grade/lean-bin:{TOOLCHAIN}/bin:{PILOT}/tools/bin:/usr/bin:/bin'
MAX_BYTES = 4 * 2**20
STATUSES = ('proved', 'stated', 'informal')
CLASSES = ('stronger', 'weaker', 'equal', 'incomparable')
LAKEFILE = '''name = "theory"
defaultTargets = ["Theory"]

[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "v4.35.0"

[[require]]
name = "cslib"
scope = "leanprover"
rev = "v4.35.0"

[[lean_lib]]
name = "Theory"
'''
FLAGGED = re.compile(
    r'^\s*(?:@\[[^\]]*\]\s*)?(?:local\s+|scoped\s+)?'
    r'(#eval|#exit|run_cmd|run_elab|run_meta|elab|elab_rules|macro|macro_rules|syntax|notation|infix|infixl|infixr|'
    r'prefix|postfix|initialize|builtin_initialize|unsafe|axiom|opaque|set_option\s+debug\.\S+|'
    r'attribute\s+\[[^\]]*(?:extern|implemented_by|csimp)[^\]]*\])\b',
    re.MULTILINE)
IMPORT = re.compile(r'^\s*(?:public\s+|meta\s+)*import\s+(?:all\s+)?(\S+)', re.MULTILINE)
DECL = re.compile(r'^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*(?:theorem|lemma)\s+([^\s(:{\[]+)',
                  re.MULTILINE)


def load_manifest(formal: Path) -> tuple[list[dict], list[str]]:
    """The manifest's results, normalized; problems are reported, not fatal."""
    problems = []
    try:
        data = json.loads((formal / 'manifest.json').read_text())
    except (OSError, ValueError) as error:
        return [], [f'manifest.json unreadable: {type(error).__name__}: {str(error)[:200]}']
    rows = data.get('results') if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return [], ['manifest.json has no "results" list']
    results = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f'result {index} is not an object')
            continue
        lean = row.get('lean') or []
        lean = [lean] if isinstance(lean, str) else [str(name) for name in lean if isinstance(name, str)]
        status = row.get('status')
        if status not in STATUSES:
            problems.append(f'result {row.get("id", index)}: status {status!r} not in {STATUSES}')
            status = 'informal'
        if status == 'proved' and not lean:
            problems.append(f'result {row.get("id", index)}: proved but names no Lean theorem')
            status = 'informal'
        comparison = row.get('comparison') if isinstance(row.get('comparison'), dict) else {}
        if comparison.get('class') not in CLASSES:
            problems.append(f'result {row.get("id", index)}: comparison class {comparison.get("class")!r} not in {CLASSES}')
        results.append({'id': str(row.get('id', f'#{index}')), 'claim': row.get('claim'), 'lean': lean,
                        'status': status, 'comparison': comparison})
    return results, problems


def module_file(work: Path, module: str) -> Path:
    return work / (module.replace('.', '/') + '.lean')


def statement_layer(work: Path) -> list[str]:
    """Theory.Statements and every submission module it imports, transitively."""
    seen, stack = [], ['Theory.Statements']
    while stack:
        module = stack.pop()
        path = module_file(work, module)
        if module in seen or not path.exists():
            continue
        seen.append(module)
        stack.extend(m for m in IMPORT.findall(path.read_text()) if m.startswith('Theory.'))
    return seen


def prepare(source: Path, work: Path) -> list[str]:
    """Copy the submission's Lean modules into a fresh project owned by the unprivileged user."""
    formal = source / 'theory/formal'
    files = sorted(p for p in (formal / 'Theory').rglob('*.lean') if p.is_file() and not p.is_symlink()) \
        if (formal / 'Theory').is_dir() else []
    total = sum(p.stat().st_size for p in files)
    if total > MAX_BYTES:
        raise ValueError(f'theory/formal/Theory holds {total} bytes of Lean, above {MAX_BYTES}')
    for path in files:
        target = work / 'Theory' / path.relative_to(formal / 'Theory')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (work / 'lakefile.toml').write_text(LAKEFILE)
    manifest = json.loads(MANIFEST.read_text())
    manifest['name'] = 'theory'
    (work / 'lake-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    shutil.copyfile(PILOT / 'lean/lean-toolchain', work / 'lean-toolchain')
    (work / '.lake').mkdir()
    (work / '.lake/packages').symlink_to(PACKAGES)
    (work / 'home').mkdir()
    for path in [work, *work.rglob('*')]:
        os.lchown(path, UID, UID)
    return [str(p.relative_to(formal)) for p in files]


def judge(work: Path, out: Path, tag: str, names: list[str], paranoid: bool, timeout: float) -> dict:
    config = {'challenge_module': 'Theory.Statements', 'solution_module': 'Theory.Proofs',
              'theorem_names': names, 'permitted_axioms': AXIOMS}
    (work / f'comparator-{tag}.json').write_text(json.dumps(config, indent=1) + '\n')
    os.lchown(work / f'comparator-{tag}.json', UID, UID)
    command = ['unshare', '--net', '--', 'setpriv', f'--reuid={UID}', f'--regid={UID}', '--clear-groups', '--',
               'env', '-i', f'PATH={PATH}', f'HOME={work / "home"}', 'LANG=C.UTF-8', 'TMPDIR=/tmp',
               'lake', 'comparator', '--config', f'comparator-{tag}.json', *(['--paranoid'] if paranoid else [])]
    start = time.monotonic()
    try:
        done = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=timeout)
        code, log = done.returncode, done.stdout + done.stderr
    except subprocess.TimeoutExpired as error:
        code = None
        log = (error.stdout or b'').decode(errors='replace') + (error.stderr or b'').decode(errors='replace')
        log += f'\n[lean-comparator] timeout after {timeout:.0f} s\n'
    (out / f'comparator-{tag}.log').write_text(log)
    lines = [line for line in log.splitlines() if line.strip()]
    return {'theorems': names, 'accepted': code == 0, 'exit_code': code, 'seconds': round(time.monotonic() - start, 1),
            'errors': [line[:300] for line in lines if line.lstrip().startswith('error')][:10],
            'last_line': lines[-1][:300] if lines else ''}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--paranoid', action='store_true')
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    formal = source / 'theory/formal'
    results, problems = load_manifest(formal)
    report = {'source': str(source), 'toolchain': 'leanprover/lean4:v4.35.0',
              'packages': {p['name']: p['rev'] for p in json.loads(MANIFEST.read_text())['packages']},
              'permitted_axioms': AXIOMS, 'paranoid': args.paranoid, 'manifest_problems': problems}
    work = ROOT / 'var/lean-check' / f'{out.name}-{secrets.token_hex(3)}'
    work.mkdir(parents=True)
    report['work'] = str(work)
    try:
        report['lean_files'] = prepare(source, work)
    except ValueError as error:
        report['lean_files'], report['prepare_error'] = [], str(error)
    layer = statement_layer(work)
    report['statement_layer'] = {module: sorted(set(FLAGGED.findall(module_file(work, module).read_text())))
                                 for module in layer}
    statements = DECL.findall(module_file(work, 'Theory.Statements').read_text()) \
        if module_file(work, 'Theory.Statements').exists() else []
    proved = [r for r in results if r['status'] == 'proved']
    names = list(dict.fromkeys(name for r in proved for name in r['lean']))
    declared = {name.rsplit('.', 1)[-1] for name in statements}
    missing = [name for name in names if name.rsplit('.', 1)[-1] not in declared]
    if missing:
        problems.append(f'not declared in Theory/Statements.lean: {missing}')
    runs = {}
    ready = names and all(module_file(work, m).exists() for m in ('Theory.Statements', 'Theory.Proofs'))
    if ready:
        runs['all'] = judge(work, out, 'all', names, args.paranoid, args.timeout)
        if not runs['all']['accepted'] and len(proved) > 1:
            for row in proved:
                tag = re.sub(r'[^A-Za-z0-9_.-]', '_', row['id'])
                runs[tag] = judge(work, out, tag, row['lean'], args.paranoid, args.timeout)
    for row in results:
        if row['status'] != 'proved' or not ready:
            row['verified'] = False
            continue
        run = runs['all'] if runs['all']['accepted'] or len(proved) == 1 else \
            runs[re.sub(r'[^A-Za-z0-9_.-]', '_', row['id'])]
        row['verified'] = run['accepted']
    report.update(results=results, runs=runs, verified=[r['id'] for r in results if r['verified']])
    (out / 'lean-check.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'results': len(results), 'proved_claimed': len(proved), 'verified': report['verified'],
                      'problems': len(problems), 'seconds': sum(r['seconds'] for r in runs.values())}))


if __name__ == '__main__':
    main()
