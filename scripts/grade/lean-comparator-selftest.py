#!/usr/bin/env python3
"""Judge the pilot's exploit cases and two more through lean-comparator.py; only Honest may pass.

SkipKernel adds an ill-typed proof with kernel checking switched off. IOEscape, while it is being built, writes where
the unprivileged user could write on the host (/tmp, /var/tmp, the project directory), tries the network, and then
fails with the network result in its message; no write may reach the host and the network must be unreachable.
Writes var/lean-check/selftest/summary.tsv.

Usage: lean-comparator-selftest.py [CASE ...]
"""

import json
import shutil
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = Path('/data/workspace/frontieram-theory-pilot/lean/Pilot/Exploits')
BASE = ROOT / 'var/lean-check/selftest'
MARKERS = [Path('/tmp/frontieram-lean-escape'), Path('/var/tmp/frontieram-lean-escape')]
FALSE_TARGET = 'theorem target : ∀ x : ℝ, x ^ 2 ≥ x + 1 := by\n  sorry\n'
EXTRA = {
    'SkipKernel': {
        'Statements.lean': 'import Mathlib\n' + FALSE_TARGET,
        'Proofs.lean': '''import Mathlib
open Lean Elab Command

theorem helper : ∀ x : ℝ, x ^ 2 ≥ x + 1 := by
  sorry

set_option debug.skipKernelTC true in
run_cmd liftCoreM do
  let info ← getConstInfo `helper
  addDecl (.thmDecl { name := `target, levelParams := [], type := info.type, value := mkConst ``True.intro })
''',
    },
    'IOEscape': {
        'Statements.lean': 'import Mathlib\ntheorem target : ∀ x : ℝ, x ^ 2 + 1 > 0 := by\n  sorry\n',
        'Proofs.lean': f'''import Mathlib

#eval show IO Unit from do
  try IO.FS.writeFile "{MARKERS[0]}" "escaped" catch _ => pure ()
  try IO.FS.writeFile "{MARKERS[1]}" "escaped" catch _ => pure ()
  try IO.FS.writeFile "escape-file" "escaped" catch _ => pure ()
  let net ← try
      let out ← IO.Process.output {{ cmd := "curl", args := #["-sS", "-m", "5", "-o", "/dev/null", "https://github.com"] }}
      pure s!"curl exit {{out.exitCode}}"
    catch e => pure s!"curl failed: {{e}}"
  throw <| IO.userError s!"ESCAPE-PROBE network: {{net}}"

theorem target : ∀ x : ℝ, x ^ 2 + 1 > 0 := by
  intro x
  positivity
''',
    },
}
EXPECTED = {'Honest': True}


def files(case: str) -> dict[str, str]:
    if case in EXTRA:
        return EXTRA[case]
    rename = lambda text: text.replace(f'Pilot.Exploits.{case}.', 'Theory.')
    out = {'Statements.lean': rename((CASES / case / 'Challenge.lean').read_text()),
           'Proofs.lean': rename((CASES / case / 'Solution.lean').read_text())}
    for extra in (CASES / case).glob('*.lean'):
        if extra.stem not in ('Challenge', 'Solution'):
            out[extra.name] = rename(extra.read_text())
    return out


def main() -> None:
    cases = sys.argv[1:] or sorted(p.name for p in CASES.iterdir() if p.is_dir()) + list(EXTRA)
    for marker in MARKERS:
        marker.unlink(missing_ok=True)
    rows = []
    for case in cases:
        submission = BASE / case / 'source'
        shutil.rmtree(BASE / case, ignore_errors=True)
        (submission / 'theory/formal/Theory').mkdir(parents=True)
        for name, text in files(case).items():
            (submission / 'theory/formal/Theory' / name).write_text(text)
        manifest = {'results': [{'id': 'R1', 'claim': 'target', 'lean': ['target'], 'status': 'proved',
                                 'comparison': {'class': 'incomparable'}}]}
        (submission / 'theory/formal/manifest.json').write_text(json.dumps(manifest))
        subprocess.run([sys.executable, str(ROOT / 'scripts/grade/lean-comparator.py'), str(submission),
                        str(BASE / case / 'check'), '--timeout', '900'], check=True, capture_output=True)
        report = json.loads((BASE / case / 'check/lean-check.json').read_text())
        accepted = report['runs']['all']['accepted'] if report['runs'] else False
        ok = accepted == EXPECTED.get(case, False)
        rows.append((case, 'ACCEPT' if accepted else 'REJECT', 'ok' if ok else 'UNEXPECTED',
                     str(report['runs'].get('all', {}).get('seconds')), report['runs'].get('all', {}).get('last_line', '')[:120]))
        print('\t'.join(rows[-1]), flush=True)
    if 'IOEscape' in cases:
        report = json.loads((BASE / 'IOEscape/check/lean-check.json').read_text())
        leaks = [str(m) for m in [*MARKERS, Path(report['work']) / 'escape-file'] if m.exists()]
        rows.append(('host writes', 'none' if not leaks else ' '.join(leaks), 'ok' if not leaks else 'UNEXPECTED', '', ''))
        probe = [line for line in (BASE / 'IOEscape/check/comparator-all.log').read_text().splitlines()
                 if 'ESCAPE-PROBE' in line]
        reached = any('curl exit 0' in line for line in probe)
        rows.append(('network', probe[0].strip()[:160] if probe else 'probe did not report',
                     'ok' if probe and not reached else 'UNEXPECTED', '', ''))
        print('\n'.join('\t'.join(r) for r in rows[-2:]))
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / 'summary.tsv').write_text('case\tverdict\tcheck\tseconds\tlast_line\n' + ''.join('\t'.join(r) + '\n' for r in rows))


if __name__ == '__main__':
    main()
