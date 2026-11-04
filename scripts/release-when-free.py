"""Queue held slot-pinned jobs only when their slot is free, so workers never wait on a busy slot while another idles.

A held job pinned to slot X moves to pending when no lease holds X and no queued or claimed job in any of the
given stores is pinned to X. Exits when nothing is held.

With --defer STORE, a slot is also left alone while STORE still holds a job pinned to it (STORE goes first).

Usage: release-when-free.py STORE [OTHER_STORE ...] [--defer PRIORITY_STORE ...]
"""

import json
import shutil
import sys
import time

from pathlib import Path

args = sys.argv[1:]
deferred = [Path(p) for p in args[args.index('--defer') + 1:]] if '--defer' in args else []
stores = [Path(p) for p in (args[:args.index('--defer')] if '--defer' in args else args)] + deferred
target = stores[0]
slots_dir = Path('/run/frontieram/slots')


def priority_holds(slot: str) -> bool:
    for store in deferred:
        for path in (store / 'jobs/held').glob('*.json'):
            try:
                if json.loads(path.read_text()).get('slot') == slot:
                    return True
            except (OSError, ValueError):
                continue
    return False


def leased(slot: str) -> bool:
    try:
        return bool(json.loads((slots_dir / f'{slot}.json').read_text()).get('owner'))
    except (OSError, ValueError):
        return False


def waiting_on(slot: str) -> bool:
    for store in stores:
        for state in ('pending', 'running'):
            for path in (store / 'jobs' / state).glob('*.json'):
                try:
                    if json.loads(path.read_text()).get('slot') == slot:
                        return True
                except (OSError, ValueError):
                    continue
    return False


while held := sorted((target / 'jobs/held').glob('*.json')):
    for path in held:
        slot = json.loads(path.read_text()).get('slot')
        if slot and not leased(slot) and not waiting_on(slot) and not priority_holds(slot):
            shutil.move(path, target / 'jobs/pending' / path.name)
            print(time.strftime('%H:%M:%S'), 'released', path.stem, slot, flush=True)
    time.sleep(10)
