"""Run one host command while holding a development lease from the slot broker (never on unleased cores).

The command sees the lease's cores in $LEASE_CPUSET and its cgroup parent in $LEASE_CGROUP, e.g. for
`docker build --cpuset-cpus "$LEASE_CPUSET"`. BuildKit steps cannot be pinned this way; the lease still keeps the
broker from handing the slot to anything else while the build runs.

Usage: PYTHONPATH=. FRONTIERAM_RUNTIME=... with_lease.py [--memory-gib N] [--timeout S] -- COMMAND ...
"""

import argparse
import os
import subprocess
import sys

from arena.cli import context, placement_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-gib', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=7200, help='seconds to wait for a lease')
    parser.add_argument('--label', default='with_lease')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('no command')
    layout, broker, _ = context()
    with broker.acquire('dev', args.memory_gib * 2**30, args.label, timeout=args.timeout) as lease:
        placement = placement_for(lease, layout)
        env = {**os.environ, 'LEASE_CPUSET': placement.cpuset, 'LEASE_CGROUP': placement.cgroup_parent}
        sys.exit(subprocess.run(command, env=env).returncode)


if __name__ == '__main__':
    main()
