"""Grade further conic submissions end to end, unattended: the full suite under the original limits on slot-pinned
shards (store), then the undecided cases under the locally recalibrated limits (var/stores/store-recal), then the merge.

Workers and release-when-free must be running for both stores. Usage:
  conic-grade-remaining.py LABEL=SUBMISSION [LABEL=SUBMISSION ...]
"""

import json
import os
import shutil
import subprocess
import sys
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLOTS = 'grade-06,grade-14,grade-22,grade-30'
RECAL = 'tasks/conic/versions/original/local_recal/recal-manifest.json'


def bench(store: str, runtime: str, *args: str) -> dict:
    env = {**os.environ, 'FRONTIERAM_RUNTIME': str(ROOT / runtime)}
    out = subprocess.run(['./bench', '--store', store, *args], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    return json.loads(out.stdout[out.stdout.index('{'):])


def queue(store: str, runtime: str, submission: str, cases: str | None = None) -> str:
    args = ['suite', '--task', 'conic', '--submission', submission, '--shards', '4', '--slots', SLOTS, '--hold']
    return bench(store, runtime, *args, *(['--cases', cases] if cases else []))['id']


def wait(store: str, suite: str) -> None:
    jobs = [s['job'] for s in json.loads((ROOT / store / 'suites' / f'{suite}.json').read_text())['shards']]
    finished = lambda j: any((ROOT / store / 'jobs' / s / f'{j}.json').exists() for s in ('done', 'failed'))  # noqa: E731
    while not all(finished(j) for j in jobs):
        time.sleep(60)


def log(**fields) -> None:
    print(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%S'), **fields}), flush=True)


def main():
    pairs = [item.split('=', 1) for item in sys.argv[1:]]
    suites = {label: queue('store', '.runtime-suite', sub) for label, sub in pairs}
    log(event='queued_full_suites', suites=suites)
    for label, sub in pairs:
        wait('store', suites[label])
        report = bench('store', '.runtime-suite', 'suite-report', '--suite', suites[label])
        log(event='full_suite_done', label=label, suite=suites[label], complete=report['complete'],
            solved=report['successes'], cases=report['cases_graded'])
        if not report['complete']:
            continue
        undecided = ROOT / f'tasks/conic/versions/original/local_recal/{label}-undecided.json'
        subprocess.run([sys.executable, 'scripts/grade/conic-recal.py', 'subset', 'store', suites[label], RECAL, str(undecided)],
                       cwd=ROOT, check=True)
        target = ROOT / 'var/stores/store-recal/submissions' / sub
        if not target.exists():
            shutil.copytree(ROOT / 'var/stores/store/submissions' / sub, target)
        rerun = queue('var/stores/store-recal', '.runtime-recal', sub, str(undecided))
        if not any(p for p in os.popen('pgrep -f "release-when-free.py var/stores/store-recal"').read().split()):
            subprocess.Popen([sys.executable, 'scripts/ops/release-when-free.py', 'var/stores/store-recal', 'store'], cwd=ROOT,
                             stdout=open(ROOT / 'var/stores/store/reruns/release-recal.log', 'a'), stderr=subprocess.STDOUT,
                             start_new_session=True)
        wait('var/stores/store-recal', rerun)
        subprocess.run([sys.executable, 'scripts/grade/conic-recal.py', 'merge', 'store', suites[label], RECAL, 'var/stores/store-recal',
                        rerun, f'tasks/conic/analysis/conic-recal-{label}.json'], cwd=ROOT, check=True)
        log(event='recalibrated_done', label=label, full_suite=suites[label], regrade=rerun,
            result=f'tasks/conic/analysis/conic-recal-{label}.json')


if __name__ == '__main__':
    main()
