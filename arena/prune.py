"""Remove the bulky candidate artifacts of a finished grading job (user decision 2026-11-26).

Files of at least 1 MiB under grade/candidate/ without a text suffix are candidate outputs (polarization raw.npy and
pred.npy, galaxy handoff/output.bin) and build products. The grader has read them once metrics.json exists, and a
regrade from the submission archive recreates them; text records (summaries, per-case JSON, CSV, logs, sources) stay.
PRUNED.json beside them lists what was removed. Used after every grading job (arena/cli.py grade_job) and by
scripts/ops/prune-candidate-artifacts.py for older results.
"""

import json
import time

from pathlib import Path

MIN_BYTES = 1 << 20
TEXT = {'.json', '.jsonl', '.csv', '.tsv', '.txt', '.log', '.md', '.yaml', '.yml', '.toml', '.py', '.sh', '.cfg', '.ini',
        '.h', '.hpp', '.c', '.cc', '.cpp', '.rs', '.jl', '.f90'}


def bulky(candidate: Path) -> list[Path]:
    """The files the rule removes."""
    if not candidate.is_dir():
        return []
    return [path for path in candidate.rglob('*')
            if not path.is_symlink() and path.is_file() and path.suffix.lower() not in TEXT
            and path.stat().st_size >= MIN_BYTES]


def prune_candidate(candidate: Path, reason: str) -> dict:
    """Remove the bulky files under candidate/ and record them in candidate/PRUNED.json; returns files and bytes."""
    removed = []
    for path in bulky(candidate):
        size = path.stat().st_size
        path.unlink()
        removed.append({'path': str(path.relative_to(candidate)), 'bytes': size})
    if removed:
        (candidate / 'PRUNED.json').write_text(json.dumps({
            'date_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'reason': reason,
            'rule': f'files of at least {MIN_BYTES} bytes without a text suffix; text records kept',
            'files': len(removed), 'bytes': sum(r['bytes'] for r in removed), 'removed': removed}, indent=1) + '\n')
    return {'files': len(removed), 'bytes': sum(r['bytes'] for r in removed)}
