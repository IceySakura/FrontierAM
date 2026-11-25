#!/usr/bin/env python3
"""Run one submission on a blackbox-cubature v3 instance file with the formal driver, on the slot this process
leases (by default a dev slot, whose daemon holds the task image), outside the job store: for the samples' reference rows and
for checks before the evaluation system is involved.

Usage: PYTHONPATH=. FRONTIERAM_RUNTIME=... cubature-v3-run.py INSTANCES SUBMISSION OUT [--only SUBSTRING ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from arena.cli import context, placement_for

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'arena/operator_drivers/blackbox_cubature_v3.py'
GRADER_PYTHON = ROOT / '.venv-ops/bin/python'
TAG = json.loads((ROOT / 'tasks/blackbox-cubature/versions/v3/task.json').read_text())['images']['task']['tag']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instances', type=Path)
    parser.add_argument('submission', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--kind', choices=('dev', 'grade'), default='dev',
                        help='grade: a grading slot and its daemon (the image must be loaded there)')
    args = parser.parse_args()
    data = json.loads(args.instances.read_text())
    instances = data['instances'] if isinstance(data, dict) else data
    if args.only:
        instances = [i for i in instances if any(s in i['id'] for s in args.only)]
    args.out.mkdir(parents=True, exist_ok=True)
    listing = args.out / 'cases.json'
    listing.write_text(json.dumps(instances) + '\n')
    image = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', TAG], check=True, capture_output=True,
                           text=True).stdout.strip()
    layout, broker, _ = context()
    with broker.acquire(args.kind, 4 * 2**30, f'cubature-v3-run {args.submission.name}', timeout=7200) as lease:
        placement = placement_for(lease, layout)
        done = subprocess.run([str(GRADER_PYTHON), '-B', str(DRIVER), str(listing), str(args.submission.resolve()),
                               str(args.out), '--image', image, '--cpuset', placement.cpuset,
                               '--cgroup-parent', placement.cgroup_parent],
                              env={**os.environ, 'DOCKER_HOST': placement.docker_host}, capture_output=True, text=True)
    (args.out / 'grader.log').write_text(done.stdout + done.stderr)
    summary = json.loads((args.out / 'summary.json').read_text())
    print(json.dumps({k: summary.get(k) for k in ('status', 'cases', 'failures', 'reason', 'max_wall_s', 'bounds')}))
    sys.exit(0 if summary.get('status') == 'ok' else 1)


if __name__ == '__main__':
    main()
