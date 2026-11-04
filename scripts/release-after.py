"""Release a held suite slot by slot: its shard for a slot enters the queue once the earlier suite's shard holds
that slot (or is done), so slot order follows suite priority instead of the broker's polling race.

Usage: release-after.py STORE AFTER_SUITE RELEASE_SUITE
"""

import json
import shutil
import sys
import time

from pathlib import Path

store, after, release = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
slots_dir = Path('/run/frontieram/slots')


def started(job: str, slot: str) -> bool:
    if (store / 'jobs/done' / f'{job}.json').exists() or (store / 'jobs/failed' / f'{job}.json').exists():
        return True
    record = slots_dir / f'{slot}.json'
    try:
        return job in json.loads(record.read_text()).get('owner', '')
    except (OSError, ValueError):
        return False


earlier = {s['job']: None for s in json.loads((store / 'suites' / f'{after}.json').read_text())['shards']}
for job in earlier:
    for state in ('pending', 'running', 'done', 'failed', 'held'):
        path = store / 'jobs' / state / f'{job}.json'
        if path.exists():
            earlier[job] = json.loads(path.read_text())['slot']
            break
waiting = {s['job'] for s in json.loads((store / 'suites' / f'{release}.json').read_text())['shards']}
while waiting:
    for job in sorted(waiting):
        held = store / 'jobs/held' / f'{job}.json'
        slot = json.loads(held.read_text())['slot']
        if any(s == slot and started(j, s) for j, s in earlier.items()):
            shutil.move(held, store / 'jobs/pending' / held.name)
            waiting.discard(job)
            print(time.strftime('%H:%M:%S'), 'released', job, slot, flush=True)
    time.sleep(30)
