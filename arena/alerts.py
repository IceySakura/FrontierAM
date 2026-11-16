"""Anomaly alerts under full concurrency: appended to var/stores/alerts.jsonl and shown on the dashboard monitor page.

Nothing is regraded automatically; each alert is handled case by case (docs/platform/CONCURRENCY.md section 7).
"""
from __future__ import annotations

import json
import time

from pathlib import Path

ALERTS = Path(__file__).resolve().parents[1] / 'var/stores/alerts.jsonl'
HANDLED = ALERTS.with_name('alerts-handled.jsonl')


def alert(kind: str, **fields) -> None:
    ALERTS.parent.mkdir(parents=True, exist_ok=True)
    with ALERTS.open('a') as stream:
        stream.write(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'kind': kind, **fields}, ensure_ascii=False) + '\n')


def handled(job: str, note: str) -> None:
    """Record the case-by-case conclusion for the alerts of one job; the dashboard shows it next to them."""
    HANDLED.parent.mkdir(parents=True, exist_ok=True)
    with HANDLED.open('a') as stream:
        stream.write(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'job': job, 'note': note}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    import sys
    handled(sys.argv[1], ' '.join(sys.argv[2:]))
