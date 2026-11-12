"""Per-case CPU limits for galaxy-gravity-v4 on this machine, with the author's B.4 rule.

The limit of a snapshot is 1.25 x the largest process CPU time of the strong reference (Treecode2 modified tree,
theta 0.7) over its three calls, rounded up to 0.1 s: 100000 * ceil(5 * max_cpu_us / 400000) microseconds
(private/limits.json of the B.4 bundle). The timing suite is a reference suite graded with a uniform generous limit
(FRONTIERAM_GALAXY_V4_TIMING) in the same shared conditions as candidate grading; the weak reference's times are
reported alongside, as a check that it fits the frozen limits.

Usage: galaxy-v4-calibrate.py --store STORE --suite SUITE [--public]
  writes tasks/galaxy-gravity/versions/v4/data/{private/private,public}/local-limits.json
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / 'tasks/galaxy-gravity/versions/v4/data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--public', action='store_true')
    args = parser.parse_args()
    record = json.loads((args.store / 'suites' / f'{args.suite}.json').read_text())
    strong, weak = {}, {}
    for shard in record['shards']:
        parts = json.loads((args.store / 'results' / shard['job'] / 'grade/metrics.json').read_text())['parts']
        for name, target in (('strong', strong), ('weak', weak)):
            if parts[name]['status'] != 'ok':
                raise SystemExit(f'{shard["job"]} {name}: {parts[name]["status"]} {parts[name].get("reason")}')
            for c in parts[name]['cases']:
                target[c['case']] = c
    limits, rows = {}, []
    for case, c in sorted(strong.items()):
        max_us = round(c['cpu_s']['max'] * 1e6)
        limit = 100000 * math.ceil(5 * max_us / 400000) / 1e6
        limits[case] = limit
        w = weak.get(case)
        rows.append({'case': case, 'n': c['n'], 'family': c['family'], 'strong_max_cpu_s': c['cpu_s']['max'],
                     'limit_s': limit, 'weak_max_cpu_s': w['cpu_s']['max'] if w else None,
                     'weak_fraction': w['cpu_s']['max'] / limit if w else None})
    out = V4 / ('public' if args.public else 'private/private') / 'local-limits.json'
    out.write_text(json.dumps({'rule': '100000 * ceil(5 * max_cpu_us / 400000) us of the strong reference over 3 calls',
                               'timing_suite': args.suite, 'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                               'limits': limits, 'cases': rows}, indent=1) + '\n')
    by_scale = {}
    for r in rows:
        s = by_scale.setdefault(r['n'], {'min': math.inf, 'max': 0, 'weak_max_fraction': 0})
        s['min'], s['max'] = min(s['min'], r['limit_s']), max(s['max'], r['limit_s'])
        s['weak_max_fraction'] = max(s['weak_max_fraction'], r['weak_fraction'] or 0)
    print(json.dumps({'out': str(out), 'cases': len(rows), 'by_scale': by_scale,
                      'weak_over_limit': [r['case'] for r in rows if (r['weak_fraction'] or 0) > 1]}, indent=1))


if __name__ == '__main__':
    main()
