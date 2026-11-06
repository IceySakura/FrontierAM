"""Public calibration of the exponential-v2 panel under one amended rule for every parameterised member.

Rule (amended 2026-11-06, public data only, see RULE.md):
  * cases: the truth-qualified public cases (scored cases without an independently verified reference are excluded);
  * a setting qualifies if no request fails accuracy and the largest error over the requests it completes is at most
    MARGIN x (atol + rtol * |y_ref|_inf); a wall/CPU/memory limit is a case failure of the member (it then takes no
    part in that case's virtual best), not a disqualification of the setting;
  * tolerance members (expokit, kiops): the largest multiplier that qualifies together with every smaller one;
  * expmv: per level (p4, p7, p10) the cheapest of half < single < double that qualifies on that level's cases.
Timing plays no part, so settings are evaluated in parallel.

Run inside the task image:
  python3 calibrate_v2.py run MEMBER VALUE[,VALUE...] --manifest M --out DIR [--jobs N]
  python3 calibrate_v2.py select --manifest M --out DIR    (combines every evaluation in DIR, writes selection.json)
"""

import argparse
import json
import subprocess
import sys
import tempfile

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / 'panel'
TOOLS = HERE.parents[1] / 'public/tools'
MARGIN = 0.1
RESOURCE = {'wall_timeout', 'cpu_limit', 'cpu_timeout', 'memory_limit', 'startup_timeout'}
LEVELS = {1e-4: 'p4', 1e-7: 'p7', 1e-10: 'p10'}
PRECISIONS = ['half', 'single', 'double']


def command(member: str, value: str) -> str:
    if member == 'expokit':
        return f'{PANEL / "expokit/expokit_avx2"} --tol-multiplier {value}'
    if member == 'kiops':
        return f'{PANEL / "kiops/run.sh"} {value}'
    if member == 'expmv':
        return f'''{PANEL / "expmv/run.sh"} "struct('p4','{value}','p7','{value}','p10','{value}')"'''
    raise ValueError(member)


def evaluate(member: str, value: str, manifest: str) -> dict:
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / 'report.json'
        subprocess.run([sys.executable, str(TOOLS / 'run_local.py'), '--manifest', manifest, '--command',
                        command(member, value), '--repeats', '1', '--output', str(report)], check=True, capture_output=True)
        runs = json.loads(report.read_text())['runs']
    cases = [{'id': r['id'], 'valid': r['valid'], 'reason': r['reason'], 'rtol': r.get('rtol'),
              'worst': max((e['scaled_error'] for e in r['errors'] if e.get('scaled_error') is not None), default=None)}
             for r in runs]
    return {'member': member, 'value': value, 'cases': cases}


def qualifies(cases: list[dict]) -> tuple[bool, dict]:
    accuracy = [c['id'] for c in cases if not c['valid'] and c['reason'] not in RESOURCE]
    resource = [c['id'] for c in cases if not c['valid'] and c['reason'] in RESOURCE]
    worst = max((c['worst'] for c in cases if c['valid'] and c['worst'] is not None), default=0.0)
    return not accuracy and worst <= MARGIN, {'accuracy_failures': accuracy, 'resource_failures': resource,
                                              'max_scaled_error': worst}


def select(out: Path, manifest: Path) -> dict:
    ids = {c['id']: c for c in json.loads(manifest.read_text())['cases']}
    evaluations = [json.loads(p.read_text()) for p in sorted(out.glob('eval-*.json'))]
    result = {}
    for member in ('expokit', 'kiops'):
        rows = sorted((e for e in evaluations if e['member'] == member), key=lambda e: -float(e['value']))
        table = []
        for e in rows:
            # Converted first-pass results carry only their failures plus one record holding the largest error.
            ok, detail = qualifies([c for c in e['cases'] if c['id'] in ids or c['id'].startswith('__')])
            table.append({'multiplier': float(e['value']), 'qualifies': ok, **detail})
        chosen = next((r['multiplier'] for i, r in enumerate(table) if all(s['qualifies'] for s in table[i:])), None)
        result[member] = {'selected_multiplier': chosen, 'grid': table}
    levels = {}
    for level in LEVELS.values():
        table = []
        for precision in PRECISIONS:
            e = next((e for e in evaluations if e['member'] == 'expmv' and e['value'] == precision), None)
            if e is None:
                continue
            cases = [c for c in e['cases'] if c['id'] in ids and ids[c['id']]['precision'] == level]
            ok, detail = qualifies(cases)
            table.append({'precision': precision, 'qualifies': ok, **detail})
        levels[level] = {'selected': next((r['precision'] for r in table if r['qualifies']), None), 'grid': table}
    result['expmv'] = {'selected': {k: v['selected'] for k, v in levels.items()}, 'levels': levels}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('run', 'select'))
    parser.add_argument('member', nargs='?')
    parser.add_argument('values', nargs='?')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'run':
        values = args.values.split(',')
        with ThreadPoolExecutor(args.jobs) as pool:
            for e in pool.map(lambda v: evaluate(args.member, v, str(args.manifest)), values):
                (args.out / f'eval-{e["member"]}-{e["value"]}.json').write_text(json.dumps(e, indent=1) + '\n')
                print(e['member'], e['value'], qualifies(e['cases']), flush=True)
    else:
        result = select(args.out, args.manifest)
        (args.out / 'selection.json').write_text(json.dumps({'rule': __doc__.split('Run inside')[0].strip(), 'margin': MARGIN,
                                                             **result}, indent=2) + '\n')
        print(json.dumps({m: (r.get('selected_multiplier') if 'selected_multiplier' in r else r['selected'])
                          for m, r in result.items()}))


if __name__ == '__main__':
    main()
