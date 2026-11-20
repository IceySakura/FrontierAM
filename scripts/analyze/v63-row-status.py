#!/usr/bin/env python3
"""Status of one model's v6.3 board row: per queued cell the latest attempt's state, its ATR request health and the
score once graded.

Usage: v63-row-status.py MODEL TAG QUEUE   (e.g. gpt-6.2-sol sol62 var/stores/store/reruns/v63-queue-sol62.txt)
State: queued, running (minutes since start), or the develop_finished status with the agent's stop reason. Health
comes from scripts/analyze/v63-infra-audit.py: unanswered (cut) requests, error statuses, truncated answers and the
minutes they cost; speed is the median generation rate of answers with at least 200 output tokens.
"""

import importlib.util
import json
import re
import statistics
import sys

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RERUNS = ROOT / 'var/stores/store/reruns'
RUNS = ROOT / 'var/stores/store-v63/dev-runs'


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load(ROOT / 'scripts/analyze/v63-infra-audit.py').audit
publish = load(ROOT / 'scripts/rounds/v63-publish.py')


def latest_log(tag: str, setting: str, task: str) -> tuple[int, Path | None]:
    logs = [(int(m.group(1)), path) for path in RERUNS.glob(f'v63-{tag}-{setting}-{task}-*.log')
            if (m := re.fullmatch(rf'v63-{tag}-{setting}-{re.escape(task)}-(\d+)\.log', path.name))]
    return max(logs) if logs else (1, None)


def events(log: Path) -> list[dict]:
    out = []
    for line in log.read_text(errors='replace').splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return [e for e in out if isinstance(e, dict)]


def speed(run: Path) -> float | None:
    rates = []
    for path in run.glob('atr/*/events.jsonl'):
        for line in open(path):
            e = json.loads(line)
            usage, generating = e.get('usage') or {}, (e.get('duration_ms') or 0) - (e.get('ttfb_ms') or 0)
            if e['kind'] == 'response' and (usage.get('output') or 0) >= 200 and generating > 1000:
                rates.append(usage['output'] / (generating / 1000))
    return statistics.median(rates) if rates else None


def main() -> None:
    model, tag, queue = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    cells = [line.split()[1:3] for line in queue.read_text().splitlines() if line.strip() and not line.startswith('#')]
    now, states = datetime.now(timezone.utc), {}
    for setting, task in cells:
        n, log = latest_log(tag, setting, task)
        record = events(log) if log else []
        started = [e for e in record if e.get('event') == 'develop_started']
        finished = [e for e in record if e.get('event') == 'develop_finished']
        line = f'{setting:9s} {task:30s} #{n} '
        if not started:
            state = 'queued'
            line += state
        else:
            run = RUNS / started[-1]['run']
            stamp = datetime.strptime(re.search(r'(\d{8}T\d{6}Z)', run.name).group(1), '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
            if finished:
                result = run / 'develop_result.json'
                minutes = json.loads(result.read_text()).get('development_seconds', 0) / 60 if result.exists() else 0
                state = finished[-1].get('status', '?')
                line += f'{state} ({finished[-1].get("agent_stop")}, {minutes:.0f} min)'
            else:
                state = 'running'
                line += f'running {(now - stamp).total_seconds() / 60:.0f} min'
            health = audit(run) if run.exists() else {}
            rate = speed(run) if run.exists() else None
            line += (f' | req {health.get("requests", 0)} cut {health.get("cut", 0)} err {health.get("error_status", 0)}'
                     f' trunc {health.get("truncated", 0)} lost {health.get("lost_min", 0)} min')
            line += f' | {rate:.0f} tok/s' if rate else ' | - tok/s'
        folder = ROOT / 'tasks' / publish.TASKS[task] / 'solutions/v6.3' / '/'.join(
            filter(None, (publish.SUBFOLDER.get(task), f'{model}-{setting}-{n}')))
        if (folder / 'score.json').exists():
            score = json.loads((folder / 'score.json').read_text())
            try:
                raw, _ = publish.cell(task, score)
            except Exception:  # an invalid or unusual record: the score alone is still shown
                raw = ''
            value = score.get('score')
            line += f' | score {value:.1f} ({raw})' if isinstance(value, (int, float)) else f' | score {value} ({raw})'
            state = 'graded'
        states[state] = states.get(state, 0) + 1
        print(line)
    print(json.dumps({'time': f'{datetime.now():%H:%M}', **states}))


if __name__ == '__main__':
    main()
