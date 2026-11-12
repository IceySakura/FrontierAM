"""Live core occupancy for the dashboard's monitor page: writes status.json every few seconds.

Reads the machine layout (/etc/frontieram/machine.json: cores, L3 domains, housekeeping, dev and grade slots) and the
slot broker's lease records (/run/frontieram/slots/*.json, live only while their holder keeps the flock), and names
what holds each core: development runs (task, model, setting), grading jobs (task, kind), replays, held-idle slots.
Pending development holds (grading windows, quiet drains) are listed too. Light: no FrontierAM imports.

Usage: slot_status.py [--out DIR] [--every SECONDS] [--once]
"""
import argparse
import fcntl
import json
import os
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = Path('/etc/frontieram/machine.json')
SLOTS = Path(os.environ.get('FRONTIERAM_STATE', '/run/frontieram')) / 'slots'


def held(path: Path) -> bool:
    """True if some process holds the flock on path (a live lease or request)."""
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def describe(owner: str) -> dict:
    """What the owner string names: a development run, a grading job, a replay, or a held-idle slot."""
    words = owner.split()
    if owner.startswith('develop ') and len(words) > 1:
        for spec in ROOT.glob(f'var/stores/*/dev-runs/{words[1]}/spec.json'):
            s = json.loads(spec.read_text())
            return {'kind': 'develop', 'task': s.get('task'), 'model': s.get('model'),
                    'setting': s.get('setting') or 'single', 'label': f'开发 · {s.get("task")} · {s.get("model")} · '
                    f'{"迭代" if s.get("setting") == "iterative" else "单轮"}'}
        return {'kind': 'develop', 'label': owner}
    if owner.startswith('grade job-') and len(words) > 1:
        for job in ROOT.glob(f'var/stores/*/jobs/running/{words[1]}.json'):
            j = json.loads(job.read_text())
            kind = {'baseline': '参照', 'candidate': '评分', 'probe': '干扰测量', 'shard': '分片评分'}.get(j.get('kind'), j.get('kind'))
            return {'kind': 'grade', 'task': j.get('task'), 'label': f'{kind} · {j.get("task")}'}
        return {'kind': 'grade', 'label': owner}
    if owner.startswith('held idle by'):
        return {'kind': 'held', 'label': '为隔离空着（' + owner[len('held idle by '):] + '）'}
    return {'kind': 'other', 'label': owner}


def snapshot() -> dict:
    layout = json.loads(LAYOUT.read_text())
    slots = {s['cpu']: s for s in layout['slots']}
    housekeeping = set()
    for part in layout['housekeeping_cpus'].split(','):
        lo, _, hi = part.partition('-')
        housekeeping.update(range(int(lo), int(hi or lo) + 1))
    now = time.time()
    cores = []
    for index, core in enumerate(layout['topology']):
        slot = next((slots[c] for c in core['cpus'] if c in slots), None)
        entry = {'core': index, 'cpus': core['cpus'], 'l3': core['l3'],
                 'role': slot['kind'] if slot else ('housekeeping' if set(core['cpus']) & housekeeping else 'other'),
                 'slot': slot['id'] if slot else None}
        if slot:
            record = SLOTS / f'{slot["id"]}.json'
            if record.exists() and held(SLOTS / f'{slot["id"]}.lock'):
                try:
                    lease = json.loads(record.read_text())
                    owner = lease.get('owner') or ''
                    entry['lease'] = {**describe(owner), 'owner': owner, 'isolation': lease.get('isolation'),
                                      'minutes': round((now - lease['acquired_unix']) / 60, 1) if lease.get('acquired_unix') else None}
                except (OSError, ValueError):
                    entry['lease'] = {'kind': 'other', 'label': '（记录读取中）'}
        cores.append(entry)
    holds = []
    for path in SLOTS.glob('drain-*.request'):
        if held(path):
            try:
                request = json.loads(path.read_text())
                holds.append({'owner': request.get('owner'), 'dev_only': bool(request.get('dev_only')),
                              'blocks_dev_now': now >= request.get('dev_after_unix', 0)})
            except (OSError, ValueError):
                pass
    attempts = sum(1 for p in SLOTS.glob('attempt-*.request') if held(p))
    return {'generated_unix': now, 'generated': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
            'cores': cores, 'holds': holds, 'iterative_attempts': attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/srv/frontieram-dashboard'))
    parser.add_argument('--every', type=float, default=5.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        if args.out.is_dir():
            data = json.dumps(snapshot(), ensure_ascii=False)
            tmp = args.out / '.status.json.tmp'
            tmp.write_text(data)
            tmp.replace(args.out / 'status.json')
        if args.once:
            return
        time.sleep(args.every)


if __name__ == '__main__':
    main()
