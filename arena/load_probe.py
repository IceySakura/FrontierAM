"""A ~0.2-second load probe on a grade slot's host thread (the sibling hardware thread of the grade core).

It times a memory copy that runs through the core's L2, its L3 domain and memory, the resources concurrent work can
contend for, and compares it with the best time seen on that slot. A grading job probes before and after; a ratio
above SLOW raises an alert (arena/alerts.py). The result itself is kept; nothing is regraded automatically.
"""
from __future__ import annotations

import json
import subprocess
import sys

from pathlib import Path

BEST = Path(__file__).resolve().parents[1] / 'var/stores/load-probe-best.json'
SLOW = 1.25
CODE = ("import time\nb = bytearray(64 << 20)\nbest = 1e9\nfor _ in range(3):\n    t = time.perf_counter()\n"
        "    for _ in range(4):\n        bytes(b)\n    best = min(best, time.perf_counter() - t)\nprint(best)")


def measure(cpu: int) -> float | None:
    """Run the copy pinned to cpu, in a scope of the host slice as grading jobs run (arena/cli.py run_in_slot)."""
    try:
        done = subprocess.run(['systemd-run', '--scope', '--quiet', '--collect', '--slice=frontieram-host.slice',
                               '-p', f'AllowedCPUs={cpu}', sys.executable, '-S', '-c', CODE],
                              capture_output=True, text=True, timeout=30)
        return float(done.stdout.strip().splitlines()[-1])
    except (subprocess.SubprocessError, ValueError, OSError, IndexError):
        return None


def probe(slot: str, cpu: int) -> dict:
    """Time the copy on cpu; ratio against the slot's best (updated when beaten)."""
    seconds = measure(cpu)
    if seconds is None:
        return {'seconds': None}
    try:
        best = json.loads(BEST.read_text())
    except (OSError, ValueError):
        best = {}
    reference = best.get(slot)
    if reference is None or seconds < reference:
        best[slot] = seconds
        BEST.parent.mkdir(parents=True, exist_ok=True)
        tmp = BEST.with_suffix('.tmp')
        tmp.write_text(json.dumps(best, indent=1) + '\n')
        tmp.replace(BEST)
        reference = seconds
    return {'seconds': round(seconds, 4), 'best': round(reference, 4), 'ratio': round(seconds / reference, 3)}
