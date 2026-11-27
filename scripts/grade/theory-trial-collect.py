#!/usr/bin/env python3
"""Collect one theory-only trial (scripts/rounds/theory-trial.sh) and run its Lean check.

Waits for the attempt's develop_finished event, extracts the frozen submission into
tasks/<family>/solutions/<version>-trial/<model>-single-<n>/source/, copies the run's proxy log, writes result.json
(run facts only; the theory score is set by review, tasks/blackbox-cubature/analysis/THEORY_REVIEW.md section 2) and
runs scripts/grade/lean-comparator.py on a pinned grade lease, passing the task's fixed model files (task.json
lean.fixed) from the version's workspace template.

Usage: theory-trial-collect.py TASK MODEL N [--poll SECONDS] [--wait-hours H]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / 'var/stores/store-v63'
TAGS = {'gpt-6.2-sol': 'sol62', 'claude-opus-5-5': 'opus'}
VERSIONS = {'blackbox-cubature-theory-v1': ('blackbox-cubature', 'theory-v1'),
            'decentralized-sum-of-nonconvex-theory-v1': ('decentralized-sum-of-nonconvex', 'theory-v1'),
            'blackbox-cubature-theory-v2': ('blackbox-cubature', 'theory-v2'),
            'decentralized-sum-of-nonconvex-theory-v2': ('decentralized-sum-of-nonconvex', 'theory-v2')}


def finished(log: Path) -> dict | None:
    for line in log.read_text().splitlines() if log.exists() else []:
        if '"develop_finished"' in line:
            return json.loads(line)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('task', choices=sorted(VERSIONS))
    parser.add_argument('model', choices=sorted(TAGS))
    parser.add_argument('n', type=int)
    parser.add_argument('--poll', type=float, default=60)
    parser.add_argument('--wait-hours', type=float, default=8)
    args = parser.parse_args()
    family, version = VERSIONS[args.task]
    log = ROOT / f'var/stores/store/reruns/theory-{args.task}-{TAGS[args.model]}-{args.n}.log'
    deadline = time.time() + args.wait_hours * 3600
    while (event := finished(log)) is None:
        if time.time() > deadline:
            sys.exit(f'no develop_finished in {log}')
        time.sleep(args.poll)
    run = STORE / 'dev-runs' / event['run']
    out = ROOT / f'tasks/{family}/solutions/{version}-trial/{args.model}-single-{args.n}'
    out.mkdir(parents=True, exist_ok=True)
    develop = json.loads((run / 'develop_result.json').read_text()) if (run / 'develop_result.json').exists() else {}
    started = re.search(r'\d{8}T\d{6}Z', event['run']).group(0)
    proxy = (run / 'netproxy.jsonl').read_text() if (run / 'netproxy.jsonl').exists() else ''
    (out / 'netproxy.jsonl').write_text(proxy)
    rows = [json.loads(line) for line in proxy.splitlines() if line.strip()]
    allowed = sorted({r.get('host') for r in rows if str(r.get('decision', '')).startswith('allow')})
    result = {'task': args.task, 'model': args.model, 'setting': 'single (v6.3 harness, 6 h limit, validation run; theory only; '
              'public internet during development)', 'dev_run': event['run'], 'submission': event.get('submission'),
              'status': event.get('status'), 'agent_stop_reason': develop.get('agent_stop_reason'),
              'development_seconds': develop.get('development_seconds'), 'started_utc': started,
              'network': {'log': 'netproxy.jsonl', 'requests': len(rows),
                          'allowed': sum(str(r.get('decision', '')).startswith('allow') for r in rows),
                          'denied': sum(str(r.get('decision', '')).startswith('deny') for r in rows),
                          'allowed_hosts': allowed}}
    (out / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    if event.get('status') != 'frozen':
        print(json.dumps({'collected': str(out), 'status': event.get('status'), 'lean_check': None}))
        return
    source = out / 'source'
    if not source.exists():
        with tarfile.open(STORE / 'submissions' / event['submission'] / 'source.tar') as tar:
            try:
                tar.extractall(source, filter='data')
            except TypeError:
                tar.extractall(source)
    spec = json.loads((ROOT / f'tasks/{family}/versions/{version}/task.json').read_text())
    template = ROOT / f'tasks/{family}/versions/{version}/workspace/examples/lean_project'
    fixed = [arg for name in spec.get('lean', {}).get('fixed', []) for arg in ('--fixed', f'{name}={template / name}')]
    command = [sys.executable, 'scripts/ops/with_lease.py', '--kind', 'grade', '--pin', '--label',
               f'lean check {args.task} {args.model}', '--timeout', '14400', '--', sys.executable,
               'scripts/grade/lean-comparator.py', str(source), str(out / 'lean-check'), '--timeout', '5400', *fixed]
    done = subprocess.run(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)},
                          capture_output=True, text=True)
    (out / 'lean-check.out').write_text(done.stdout + done.stderr)
    print(json.dumps({'collected': str(out), 'status': 'frozen', 'lean_check_exit': done.returncode,
                      'summary': done.stdout.strip().splitlines()[-1:] }))


if __name__ == '__main__':
    main()
