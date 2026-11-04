"""Set up one concurrent grading round: pin chosen pending jobs to slots, hold back the rest, add a control sentinel.

The sentinel grades the frozen reference on the noisiest slot of the round; if it does not reproduce the baseline,
the round's candidates are regraded one at a time.

Usage: concurrent-round.py --task conic --sentinel-slot grade-06 JOB=SLOT [JOB=SLOT ...]
       concurrent-round.py --release        (return held jobs to pending)
"""

import argparse
import json
import shutil
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena.cli import enqueue_reference  # noqa: E402
from arena.runtime import load_runtime  # noqa: E402
from arena.store import Store  # noqa: E402
from arena.tasks import TASKS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', default='store')
    parser.add_argument('--task')
    parser.add_argument('--sentinel-slot')
    parser.add_argument('--release', action='store_true')
    parser.add_argument('assign', nargs='*')
    args = parser.parse_args()
    store = Store(args.store)
    pending, held = Path(args.store) / 'jobs/pending', Path(args.store) / 'jobs/held'
    held.mkdir(exist_ok=True)
    if args.release:
        for path in held.glob('*.json'):
            shutil.move(path, pending / path.name)
            print('released', path.stem)
        return
    assignments = dict(item.split('=', 1) for item in args.assign)
    for path in sorted(pending.glob('*.json')):
        job = json.loads(path.read_text())
        if job['task'] != args.task:
            continue
        if job['id'] in assignments:
            job.update(slot=assignments.pop(job['id']), isolation='shared')
            path.write_text(json.dumps(job, indent=2) + '\n')
            print('pinned', job['id'], job['slot'])
        else:
            shutil.move(path, held / path.name)
            print('held', job['id'])
    if assignments:
        raise SystemExit(f'not pending: {sorted(assignments)}')
    sentinel = enqueue_reference(store, load_runtime(), TASKS[args.task], kind='sentinel', isolation='shared',
                                 note='concurrent round control')
    path = pending / f'{sentinel}.json'
    job = json.loads(path.read_text())
    job['slot'] = args.sentinel_slot
    path.write_text(json.dumps(job, indent=2) + '\n')
    print('sentinel', sentinel, args.sentinel_slot)


if __name__ == '__main__':
    main()
