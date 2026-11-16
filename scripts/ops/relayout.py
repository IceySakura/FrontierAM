"""Change the machine's slot layout safely (results/EVAL_TIME_LIMITS.md: 7 grade + 7 dev slots, paired placement).

1. Hold every slot that changes: all grade slots (their daemons restart) and the development slots that leave or
   change kind. Development slots that exist unchanged in the new layout keep running: install never touches the
   development daemon or their CPUs.
2. Install the new layout while holding those leases (machine.install allows only them and the unchanged slots).
3. Copy every image of an existing grading daemon into each new grading daemon, by ID.
4. Record the new grading daemons in every sealed runtime (runtime-lock.json grade_docker_hosts) and check that
   `arena.cli.context()` accepts the layout.

Usage (root): relayout.py [--housekeeping-cores 2] [--grade-cores 7] [--placement paired] [--dry-run]
"""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import arena.tasks  # noqa: E402,F401  (task registry before arena.cli)
from arena import machine  # noqa: E402
from arena.slots import Broker  # noqa: E402

LOG = ROOT / 'var/stores/store/reruns/relayout.log'


def say(**event) -> None:
    line = json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S'), **event})
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as stream:
        stream.write(line + '\n')


def copy_images(source: str, target: str) -> int:
    ids = subprocess.run(['docker', '--host', source, 'images', '-q', '--no-trunc'], capture_output=True, text=True,
                         check=True).stdout.split()
    copied = 0
    for image in dict.fromkeys(ids):
        if subprocess.run(['docker', '--host', target, 'image', 'inspect', image], capture_output=True).returncode == 0:
            continue
        tags = json.loads(subprocess.check_output(['docker', '--host', source, 'image', 'inspect', image]))[0]['RepoTags']
        save = subprocess.Popen(['docker', '--host', source, 'image', 'save', *(tags or [image])], stdout=subprocess.PIPE)
        subprocess.run(['docker', '--host', target, 'image', 'load', '--quiet'], stdin=save.stdout, check=True)
        if save.wait():
            raise RuntimeError(f'could not save {image} from {source}')
        if subprocess.run(['docker', '--host', target, 'image', 'inspect', image], capture_output=True).returncode:
            raise RuntimeError(f'{image} did not arrive in {target} with the same ID')
        copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--housekeeping-cores', type=int, default=2)
    parser.add_argument('--grade-cores', type=int, default=7)
    parser.add_argument('--placement', default='paired')
    parser.add_argument('--dry-run', action='store_true', help='print the new layout only')
    args = parser.parse_args()
    old = machine.load()
    new = machine.plan(housekeeping_cores=args.housekeeping_cores, grade_cores=args.grade_cores,
                       grade_placement=args.placement)
    added = [s for s in new['slots'] if s['kind'] == 'grade' and s['id'] not in {o['id'] for o in old['slots']}]
    say(event='plan', slots=[s['id'] for s in new['slots']], added_grade=[s['id'] for s in added])
    if args.dry_run:
        return
    broker = Broker(old)
    unchanged = frozenset(s['id'] for s in old['slots'] if s['kind'] == 'dev' and
                          any(n['id'] == s['id'] and n['kind'] == 'dev' for n in new['slots']))
    changing = [s for s in old['slots'] if s['id'] not in unchanged]
    removed = {s['id'] for s in old['slots']} - {s['id'] for s in new['slots']}
    say(event='holding', slots=[s['id'] for s in changing], unchanged=sorted(unchanged))
    # Processes started before the change keep the old layout in their broker and would lease a removed slot (whose
    # CPUs are now housekeeping, or whose grading daemon is gone); this process keeps those slots leased until stopped.
    tombstones = contextlib.ExitStack()
    with contextlib.ExitStack() as held:
        for slot in changing:
            lease = broker.acquire(slot['kind'], 2**28, 'relayout (slot changes in the new layout)',
                                   slot_id=slot['id'], priority=slot['kind'] == 'grade')
            (tombstones if slot['id'] in removed else held).enter_context(lease)
        say(event='held')
        machine.install(new, own_pid=os.getpid(), keep=unchanged)
        say(event='installed', layout_sha256=new['layout_sha256'])
    source = next(s['docker_host'] for s in old['slots'] if s['kind'] == 'grade' and s['id'] in {n['id'] for n in new['slots']})
    for slot in added:
        say(event='images_copied', slot=slot['id'], count=copy_images(source, slot['docker_host']))
    for lock in sorted(ROOT.glob('.runtime*/runtime-lock.json')):
        data = json.loads(lock.read_text())
        if 'grade_docker_hosts' in data:
            data['grade_docker_hosts'] = new['docker_hosts']['grade']
            lock.write_text(json.dumps(data, indent=2) + '\n')
            say(event='runtime_updated', runtime=lock.parent.name)
    check = subprocess.run([sys.executable, '-c', 'import arena.tasks\nfrom arena.cli import context\ncontext()'], cwd=ROOT,
                           capture_output=True, text=True)
    say(event='done', context_ok=check.returncode == 0, error=check.stderr.strip()[-300:] or None)
    if removed:
        say(event='holding_removed', slots=sorted(removed), pid=os.getpid(),
            note='stop this process once no process started before the change is still waiting for a slot')
        with tombstones:
            signal.pause()


if __name__ == '__main__':
    main()
