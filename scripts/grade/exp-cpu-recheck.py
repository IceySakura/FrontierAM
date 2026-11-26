"""Re-measure the CPU-limited failures of an exponential-v3 candidate suite under a quiet slot, then rescore.

Per-case CPU caps are inflated by concurrent load on this machine (a case at 10.3 s CPU with load probes 1.03-1.10
hit the 15 s cap three times beside a probe of 1.20; results/OPEN_ITEMS.md, 11-25). A case that failed on its CPU or
wall limit is measured again alone (bench suite --cases, same shard and slot rules) until a measurement has both load
probes at most QUIET, up to TRIES times; that measurement replaces the case (exp2-suite-score.py --recheck). Accuracy
failures are never re-measured. Log: OUT/recheck.json.

Usage: exp-cpu-recheck.py --store STORE --candidate SUITE --out OUT [--tries 3]
"""

import argparse
import json
import os
import subprocess
import sys
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUIET = 1.10
PANEL = ('var/stores/store', 'suite-exponential-v2-20261109T032227Z-c09381')   # as scripts/rounds/v63-grade.sh


def bench(store: Path, *args: str) -> list[dict]:
    done = subprocess.run([str(ROOT / 'bench'), '--store', str(store), *args], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, 'FRONTIERAM_RUNTIME': str(ROOT / '.runtime-recal')})
    return [json.loads(line) for line in done.stdout.splitlines() if line.startswith('{')]


def finished(store: Path, suite: str) -> bool:
    record = json.loads((store / 'suites' / f'{suite}.json').read_text())
    return all((store / 'jobs/done' / f'{s["job"]}.json').exists() or (store / 'jobs/failed' / f'{s["job"]}.json').exists()
               for s in record['shards'])


def max_probe(store: Path, suite: str) -> float:
    record = json.loads((store / 'suites' / f'{suite}.json').read_text())
    ratios = []
    for shard in record['shards']:
        probe = json.loads((store / 'results' / shard['job'] / 'result.json').read_text()).get('load_probe') or {}
        ratios += [p.get('ratio') or 99.0 for p in probe.values()]
    return max(ratios, default=99.0)


def score(store: Path, candidate: str, out: Path, recheck: list[str]) -> dict:
    subprocess.run([str(ROOT / '.venv-ops/bin/python'), str(ROOT / 'scripts/grade/exp2-suite-score.py'), '--task',
                    'exponential-v3', '--store', str(store), '--panel-store', str(ROOT / PANEL[0]), '--panel', PANEL[1],
                    '--candidate', candidate, '--out', str(out), '--per-case', *(f'--recheck={s}' for s in recheck)],
                   cwd=ROOT, check=True, capture_output=True, env={**os.environ, 'FRONTIERAM_RUNTIME': str(ROOT / '.runtime-recal')})
    return json.loads((out / 'score.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tries', type=int, default=3)
    args = parser.parse_args()
    store = args.store.resolve()
    result = json.loads((args.out / 'score.json').read_text())
    limited = sorted(c for c, reason in (result.get('failed_cases') or {}).items() if 'limit' in str(reason))
    log = {'candidate': args.candidate, 'cases': limited, 'quiet_probe': QUIET, 'tries': []}
    if not limited:
        return
    submission = json.loads((store / 'suites' / f'{args.candidate}.json').read_text())['submission']
    cases = args.out / 'recheck-cases.json'
    cases.write_text(json.dumps(limited) + '\n')
    for attempt in range(args.tries):
        suite = bench(store, 'suite', '--task', 'exponential-v3', '--submission', submission, '--shards', '7',
                      '--cases', str(cases), '--isolation', 'shared', '--cap-s', '3600')[-1]['id']
        bench(store, 'grade-worker', '--task', 'exponential-v3', '--workers', '7', '--drain')
        while not finished(store, suite):
            time.sleep(30)
        probe = max_probe(store, suite)
        log['tries'].append({'suite': suite, 'max_probe': probe})
        (args.out / 'recheck.json').write_text(json.dumps(log, indent=1) + '\n')
        if probe <= QUIET:
            log['used'] = suite
            break
        time.sleep(600)
    if 'used' in log:
        result = score(store, args.candidate, args.out, [log['used']])
        log['score'] = result.get('score')
    (args.out / 'recheck.json').write_text(json.dumps(log, indent=1) + '\n')
    print(json.dumps(log))


if __name__ == '__main__':
    sys.exit(main())
