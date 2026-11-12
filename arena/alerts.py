"""Anomaly alerts under full concurrency: appended to var/stores/alerts.jsonl and shown on the dashboard monitor page.

Nothing is regraded automatically; each alert is handled case by case (docs/platform/CONCURRENCY.md section 7).
"""
from __future__ import annotations

import json
import time

from pathlib import Path

ALERTS = Path(__file__).resolve().parents[1] / 'var/stores/alerts.jsonl'


def alert(kind: str, **fields) -> None:
    ALERTS.parent.mkdir(parents=True, exist_ok=True)
    with ALERTS.open('a') as stream:
        stream.write(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'kind': kind, **fields}, ensure_ascii=False) + '\n')
