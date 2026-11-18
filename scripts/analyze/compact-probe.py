#!/usr/bin/env python3
"""Probe that Claude Code compacts its context for a model, with a deliberately low auto-compact window.

Runs one short real development session (dev slot from the broker, nothing graded) and reports the compaction
events and the largest request it sent.
Usage: PYTHONPATH=. nexus/.venv/bin/python scripts/analyze/compact-probe.py MODEL TASK SECONDS WINDOW OUT [REQUEST_BYTES_LIMIT]
"""
import glob
import json
import re
import sys
from pathlib import Path

import arena.tasks  # noqa: F401
from arena.cli import context, placement_for
from arena.native import develop
from arena.tasks import TASKS


def main():
    model, task_name, seconds, window, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]).resolve()
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    layout, broker, runtime = context()
    task = TASKS[task_name]
    out.mkdir(parents=True)
    with broker.acquire('dev', task.dev_memory, f'compact probe {out.name}', timeout=3600) as lease:
        spec = {'task': task_name, 'model': model, 'placement': placement_for(lease, layout).__dict__, 'output': str(out),
                'protocol': 'v6.3', 'setting': 'single', 'development_seconds': seconds, 'autocompact_window': window,
                **({'request_bytes_limit': limit} if limit else {})}
        result = develop(spec, runtime)
    log = (out / 'cli.log').read_text(errors='replace')
    events = [json.loads(line) for path in glob.glob(str(out / 'atr/*/events.jsonl')) for line in open(path)]
    sizes = [e.get('request_bytes') or 0 for e in events if e['kind'] == 'request']
    print(json.dumps({'stop': result.get('agent_stop_reason'), 'requests': len(sizes), 'max_request_kb': max(sizes, default=0) // 1000,
                      'compact_boundary': len(re.findall(r'"subtype":"compact_boundary"', log)),
                      'compact_mentions': len(re.findall(r'compact', log, re.I)),
                      'context_errors': log.count("exceed the model's context limit"),
                      'transport_clears': log.count('cleared_old_context')}))


if __name__ == '__main__':
    main()
