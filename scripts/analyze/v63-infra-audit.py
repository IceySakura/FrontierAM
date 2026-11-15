#!/usr/bin/env python3
"""Infrastructure audit of v6.3 attempts: requests that got no answer, cut streams, error statuses and client
retries, from each attempt's ATR record and client log.

Usage: v63-infra-audit.py TAG [TAG ...]   (tags as in the runner logs: qwen, hy4, sol, opus, intern)
An unanswered request is cut when the client sent another request after it; the one still open when development
ended is not counted. Lost time is the span from a cut request to the next request. A request is abandoned when the
client sent the same body again before its answer arrived (Claude Code re-sends silently after its first-byte
timeout); its lost time is the span until the re-send.
"""

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RERUNS = ROOT / 'var/stores/store/reruns'
RUNS = ROOT / 'var/stores/store-v63/dev-runs'


def audit(run: Path) -> dict:
    events = []
    for path in glob.glob(str(run / 'atr/*/events.jsonl')):
        events += [json.loads(line) for line in open(path)]
    requests = sorted((e for e in events if e['kind'] == 'request'), key=lambda e: e['ts'])
    responses = {e['trace_id']: e for e in events if e['kind'] == 'response'}
    cut, lost = [], 0.0
    for current, following in zip(requests, requests[1:]):
        if current['trace_id'] not in responses:
            cut.append(round(current['ts'] - requests[0]['ts']))
            lost += following['ts'] - current['ts']
    abandoned, abandoned_lost = [], 0.0
    for index, current in enumerate(requests):
        answered = responses.get(current['trace_id'], {}).get('ts')
        resend = next((later for later in requests[index + 1:]
                       if later['request_bytes'] == current['request_bytes']), None)
        if resend is not None and answered is not None and answered > resend['ts']:
            abandoned.append(round(current['ts'] - requests[0]['ts']))
            abandoned_lost += resend['ts'] - current['ts']
    truncated = sum(1 for r in responses.values() if r.get('outcome') == 'truncated')
    errors = sum(1 for r in responses.values() if (r.get('status') or 200) >= 400)
    client = (run / 'cli.log').read_text(errors='replace') if (run / 'cli.log').exists() else ''
    longest = max((r.get('duration_ms') or 0 for r in responses.values()), default=0) / 1000
    return {'requests': len(requests), 'cut': len(cut), 'cut_at_s': cut, 'lost_min': round(lost / 60, 1),
            'abandoned': len(abandoned), 'abandoned_at_s': abandoned, 'abandoned_lost_min': round(abandoned_lost / 60, 1),
            'truncated': truncated, 'error_status': errors, 'client_retries': len(re.findall(r'"subtype":"api_retry"', client)),
            'non_stream_upstream': sum(1 for e in requests if e.get('stream') is False), 'longest_turn_s': round(longest)}


def main() -> None:
    for tag in sys.argv[1:]:
        for log in sorted(RERUNS.glob(f'v63-{tag}-*-*.log')):
            match = re.fullmatch(rf'v63-{tag}-(single|iterative)-(.+)-(\d+)\.log', log.name)
            if not match:
                continue
            started = [json.loads(line) for line in log.read_text().splitlines() if 'develop_started' in line]
            if not started:
                continue
            run = RUNS / started[-1]['run']
            if not run.exists():
                continue
            setting, task, attempt = match.groups()
            print(json.dumps({'tag': tag, 'task': task, 'setting': setting, 'attempt': int(attempt), **audit(run)}))


if __name__ == '__main__':
    main()
