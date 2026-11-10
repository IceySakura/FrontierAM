"""Release a held suite slot by slot: its shard for a slot enters the queue once the earlier suite's shard on that
slot is finished, so the queue only ever holds shards whose slot is free. Workers never sit on a busy slot while
another slot idles, and slot order follows suite priority.

Usage: release-after.py STORE AFTER_SUITE RELEASE_SUITE [AFTER_STORE]   (AFTER_STORE defaults to STORE)
"""

import json
import shutil
import sys
import time

from pathlib import Path

store, after, release = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
after_store = Path(sys.argv[4]) if len(sys.argv) > 4 else store
slots_dir = Path('/run/frontieram/slots')


def finished(job: str, slot: str) -> bool:
    if not ((after_store / 'jobs/done' / f'{job}.json').exists()
            or (after_store / 'jobs/failed' / f'{job}.json').exists()):
        return False
    try:
        return job not in json.loads((slots_dir / f'{slot}.json').read_text()).get('owner', '')
    except (OSError, ValueError):
        return True


earlier = {s['job']: None for s in json.loads((after_store / 'suites' / f'{after}.json').read_text())['shards']}
for job in earlier:
    for state in ('pending', 'running', 'done', 'failed', 'held'):
        path = after_store / 'jobs' / state / f'{job}.json'
        if path.exists():
            earlier[job] = json.loads(path.read_text())['slot']
            break
waiting = {s['job'] for s in json.loads((store / 'suites' / f'{release}.json').read_text())['shards']}
while waiting:
    for job in sorted(waiting):
        held = store / 'jobs/held' / f'{job}.json'
        slot = json.loads(held.read_text())['slot']
        if any(s == slot and finished(j, s) for j, s in earlier.items()):
            shutil.move(held, store / 'jobs/pending' / held.name)
            waiting.discard(job)
            print(time.strftime('%H:%M:%S'), 'released', job, slot, flush=True)
    time.sleep(30)
