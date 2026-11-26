#!/usr/bin/env python3
"""Free disk by removing the bulky candidate artifacts kept from finished grading jobs (user decision 2026-11-26).

Under the grade/candidate/ directory of every finished job (result.json present, job listed under jobs/done or
jobs/failed, not under jobs/running), files of at least 1 MiB that are not text records are removed: candidate outputs
(raw.npy, worker/pred.npy, handoff/output.bin) and build products. Every JSON/CSV/text/log file stays, so summaries,
per-case records and verdicts remain readable; a regrade from grade/source.tar recreates the rest. Each pruned
directory gets PRUNED.json, and the full list goes to var/stores/pruned-candidate-<UTC date>.jsonl.

Usage: prune-candidate-artifacts.py [--dry-run] STORE...
"""

import argparse
import collections
import json
import os
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIN_BYTES = 1 << 20
TEXT = {'.json', '.jsonl', '.csv', '.tsv', '.txt', '.log', '.md', '.yaml', '.yml', '.toml', '.py', '.sh', '.cfg', '.ini',
        '.h', '.hpp', '.c', '.cc', '.cpp', '.rs', '.jl', '.f90'}


def task_of(job: Path) -> str:
    try:
        return json.loads((job / 'spec.json').read_text()).get('task', '?')
    except (OSError, ValueError):
        return '?'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stores', nargs='+', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    stamp = time.strftime('%Y%m%d', time.gmtime())
    listing = ROOT / f'var/stores/pruned-candidate-{stamp}.jsonl'
    totals = collections.Counter()
    files = collections.Counter()
    skipped = []
    with (open(os.devnull, 'w') if args.dry_run else listing.open('a')) as log:
        for store in args.stores:
            finished = {p.stem for state in ('done', 'failed') for p in (store / 'jobs' / state).glob('*.json')}
            running = {p.stem for p in (store / 'jobs/running').glob('*.json')}
            for job in sorted((store / 'results').iterdir()):
                candidate = job / 'grade/candidate'
                if not candidate.is_dir():
                    continue
                if job.name in running or job.name not in finished or not (job / 'result.json').exists():
                    skipped.append(f'{store.name}/{job.name}')
                    continue
                removed, size = [], 0
                for path in candidate.rglob('*'):
                    if path.is_symlink() or not path.is_file() or path.suffix.lower() in TEXT:
                        continue
                    bytes_ = path.stat().st_size
                    if bytes_ < MIN_BYTES:
                        continue
                    removed.append(str(path.relative_to(candidate)))
                    size += bytes_
                    if not args.dry_run:
                        path.unlink()
                        log.write(json.dumps({'store': store.name, 'job': job.name,
                                              'path': f'grade/candidate/{removed[-1]}', 'bytes': bytes_}) + '\n')
                if not removed:
                    continue
                key = (store.name, task_of(job))
                totals[key] += size
                files[key] += len(removed)
                if not args.dry_run:
                    (candidate / 'PRUNED.json').write_text(json.dumps({
                        'date_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        'reason': '/data was full; bulky candidate artifacts of finished jobs removed (user decision 2026-11-26)',
                        'rule': f'files of at least {MIN_BYTES} bytes without a text suffix; text records kept',
                        'files': len(removed), 'bytes': size,
                        'listing': str(listing.relative_to(ROOT))}, indent=1) + '\n')
    for (store, task), size in totals.most_common():
        print(f'{size / 2**30:7.1f} GiB {files[(store, task)]:7d} files  {store:17s} {task}')
    print(f'total {sum(totals.values()) / 2**30:.1f} GiB in {sum(files.values())} files; skipped {len(skipped)} jobs: {skipped[:6]}')


if __name__ == '__main__':
    main()
