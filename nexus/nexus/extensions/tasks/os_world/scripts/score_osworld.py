#!/usr/bin/env python3
"""Summarize OSWorld batch scores: per-domain and total pass rate.

Reads a ``results.jsonl`` produced by ``nexus run-batch`` (each line a run record with
``task_output.score`` and ``metadata.domain``) and prints per-domain + overall numbers.

Two rates are shown:
  - avg  : mean of the raw score (metrics may return continuous [0,1] partial credit)
  - strict: fraction with score == 1.0 (partial credit and failed runs count as 0)

Failed runs (status != success / missing score) count toward the denominator as 0.

Usage:
    python score_osworld.py <results.jsonl> [more.jsonl ...]
    python score_osworld.py <run_dir>            # will look for results.jsonl inside
"""

from __future__ import annotations

import json
import sys

from collections import defaultdict
from pathlib import Path


def _resolve(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            path = path / 'results.jsonl'
        if not path.is_file():
            sys.exit(f'not found: {path}')
        out.append(path)
    return out


def summarize(files: list[Path]) -> int:
    # domain -> [n, score_sum, full_count, partial_count, failed_count]
    dom: dict[str, list] = defaultdict(lambda: [0, 0.0, 0, 0, 0])
    n = 0
    for f in files:
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            meta = rec.get('metadata') or {}
            rid = rec.get('run_id') or ''
            d = meta.get('domain') or (rid.split('__')[0] if '__' in rid else 'unknown')
            row = dom[d]
            row[0] += 1
            to = rec.get('task_output') or {}
            s = to.get('score')
            if rec.get('status') != 'success' or not isinstance(s, (int, float)):
                row[4] += 1  # failed / no score -> 0
                continue
            s = float(s)
            row[1] += s
            if s >= 1.0:
                row[2] += 1
            elif s > 0:
                row[3] += 1

    if not n:
        print('no records found')
        return 1

    tot_n = sum(r[0] for r in dom.values())
    tot_sum = sum(r[1] for r in dom.values())
    tot_full = sum(r[2] for r in dom.values())
    tot_partial = sum(r[3] for r in dom.values())
    tot_failed = sum(r[4] for r in dom.values())

    w = max(len(d) for d in dom)
    print(f'{"domain":<{w}}   n   full  avg%    strict%   (partial/failed)')
    print('-' * (w + 46))
    for d in sorted(dom, key=lambda k: -dom[k][1] / max(dom[k][0], 1)):
        c, ssum, full, part, fail = dom[d]
        print(f'{d:<{w}} {c:>4} {full:>5}  {100 * ssum / c:5.1f}  {100 * full / c:6.1f}   ({part}/{fail})')
    print('-' * (w + 46))
    print(
        f'{"TOTAL":<{w}} {tot_n:>4} {tot_full:>5}  {100 * tot_sum / tot_n:5.1f}  '
        f'{100 * tot_full / tot_n:6.1f}   ({tot_partial}/{tot_failed})'
    )
    print(
        f'\navg pass@1   : {tot_sum:.1f}/{tot_n} = {100 * tot_sum / tot_n:.1f}%   (raw score mean, allows partial credit)'
    )
    print(f'strict pass@1: {tot_full}/{tot_n} = {100 * tot_full / tot_n:.1f}%   (score == 1.0 only)')
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    raise SystemExit(summarize(_resolve(sys.argv[1:])))
