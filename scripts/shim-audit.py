"""Summarise an operator grade's docker shim logs: every container placement the package's grader requested.

For each <part>-docker.jsonl (one rewritten argv per line) it reports the distinct platform, cpuset, cgroup parent,
memory, CPU quota, network and image values of all create/run calls, so a grade can be checked to have stayed on its
slot's CPU, slice and sealed image.

Usage: shim-audit.py GRADE_DIR [GRADE_DIR ...]
"""

import collections
import json
import sys

from pathlib import Path

FLAGS = {'--platform': 'platform', '--cpuset-cpus': 'cpuset', '--cgroup-parent': 'cgroup_parent', '--memory': 'memory',
         '--cpus': 'cpus', '--network': 'network', '--memory-swap': 'memory_swap', '--pids-limit': 'pids'}
BOOLEAN = {'--read-only', '--rm', '--init', '--interactive', '-i', '--tty', '-t', '-it', '--detach', '-d', '--privileged',
           '--oom-kill-disable', '--no-healthcheck', '--quiet', '-q'}


def parse(argv: list[str]) -> dict:
    values, i = {}, 1
    while i < len(argv):
        arg = argv[i]
        key, value = (arg.split('=', 1) + [None])[:2] if arg.startswith('--') else (arg, None)
        if key in FLAGS:
            if value is None:
                i += 1
                value = argv[i]
            values[FLAGS[key]] = value
        elif not arg.startswith('-'):
            values['image'] = arg
            break
        elif value is None and key not in BOOLEAN:
            i += 1
        i += 1
    return values


def main():
    for grade in map(Path, sys.argv[1:]):
        for log in sorted(grade.glob('*-docker.jsonl')):
            calls = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            argvs = [c if isinstance(c, list) else c.get('args', []) for c in calls]
            starts = [a for a in argvs if a and a[0] in ('run', 'create')]
            seen = collections.defaultdict(collections.Counter)
            for argv in starts:
                for key, value in parse(argv).items():
                    seen[key][value] += 1
            print(f'{log.parent.name}/{log.name}: {len(argvs)} docker calls, {len(starts)} create/run')
            for key in ('platform', 'cpuset', 'cgroup_parent', 'cpus', 'memory', 'memory_swap', 'network', 'pids', 'image'):
                if key in seen:
                    print(f'  {key:13s} {dict(seen[key])}')


if __name__ == '__main__':
    main()
