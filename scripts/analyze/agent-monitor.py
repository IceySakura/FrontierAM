"""Background monitor of running v6.3 attempts of one model: a snapshot per attempt every INTERVAL seconds.

Each snapshot (var/raw/agent-monitor/<model>/<run>.jsonl) records the model's progress from the ATR records and the
Claude Code stream: requests and completed responses, output tokens, generation speed overall and over the last
10 minutes, how long the request in flight has been running, the longest and the latest turn's thinking, repetition
in thinking, tool calls, output-limit hits, upstream errors, and the files in /workspace/submission. When an attempt
ends, a summary line records where the time went (model generation vs the rest), when the submission was first
written, and the final state.

Usage: agent-monitor.py --model intern-s2 [--store var/stores/store-v63] [--interval 300]
"""

import argparse
import glob
import json
import re
import subprocess
import time

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_start(run: Path) -> float:
    stamp = re.search(r'-(\d{8}T\d{6})Z-', run.name).group(1)
    return datetime.strptime(stamp, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc).timestamp()


def atr_records(run: Path) -> tuple[list[dict], list[dict]]:
    requests, responses = [], []
    for path in glob.glob(str(run / 'atr/*-serve-*/events.jsonl')):
        for line in open(path, errors='replace'):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('kind') == 'request':
                requests.append(event)
            elif event.get('kind') == 'response':
                responses.append(event)
    return requests, responses


def stream(run: Path) -> dict:
    turns = tools = limit_hits = retries = 0
    thinking = []
    repetitive = 0
    first_write = None
    last_tool = None
    path = run / 'cli.log'
    if path.exists():
        for line in open(path, errors='replace'):
            if not line.startswith('{'):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get('type')
            if kind == 'assistant':
                turns += 1
                for block in event['message'].get('content', []):
                    if block.get('type') == 'thinking':
                        text = block.get('thinking', '')
                        thinking.append(len(text))
                        paragraphs = [p for p in text.split('\n\n') if p.strip()]
                        if len(paragraphs) > 50 and len(set(paragraphs)) / len(paragraphs) < 0.2:
                            repetitive += 1
                    elif block.get('type') == 'tool_use':
                        tools += 1
                        last_tool = block.get('name')
                        text = json.dumps(block.get('input', {}))
                        if first_write is None and 'submission' in text and (
                                block.get('name') in ('Write', 'Edit') or re.search(r'cat >|> */workspace/submission|cp |tee ', text)):
                            first_write = event.get('timestamp')
            elif kind == 'user' and event.get('isSynthetic') and 'Output token limit' in json.dumps(event):
                limit_hits += 1
            elif kind == 'system' and event.get('subtype') == 'api_retry':
                retries += 1
    return {'turns': turns, 'tool_calls': tools, 'last_tool': last_tool, 'output_limit_hits': limit_hits,
            'client_retries': retries, 'thinking_chars_max': max(thinking, default=0),
            'thinking_chars_last': thinking[-1] if thinking else 0, 'repetitive_turns': repetitive,
            'first_submission_write': first_write}


def submission_files(run: Path) -> list[str] | None:
    """Files in /workspace/submission of the attempt's dev container (matched by the run's harness image)."""
    try:
        image = json.loads((run / 'v6_conditions.json').read_text())['image']
    except (OSError, ValueError, KeyError):
        return None
    names = subprocess.run(['docker', 'ps', '--filter', f'ancestor={image}', '--format', '{{.Names}} {{.CreatedAt}}'],
                           capture_output=True, text=True, timeout=30).stdout.split('\n')
    start = run_start(run)
    for line in filter(None, names):
        name, created = line.split(' ', 1)
        created_ts = datetime.strptime(created[:25], '%Y-%m-%d %H:%M:%S %z').timestamp()
        if 0 <= created_ts - start < 600:
            listing = subprocess.run(['docker', 'exec', name, 'sh', '-c', 'ls -la /workspace/submission 2>/dev/null'],
                                     capture_output=True, text=True, timeout=30).stdout
            files = [row.split()[-1] for row in listing.splitlines()[1:] if row and not row.split()[-1] in ('.', '..')]
            return files
    return None


def snapshot(run: Path) -> dict:
    now = time.time()
    requests, responses = atr_records(run)
    output = sum((r.get('usage') or {}).get('output') or 0 for r in responses)
    generating = sum((r.get('duration_ms') or 0) / 1000 for r in responses)
    recent = [r for r in responses if float(r.get('ts', 0)) > now - 600]
    recent_out = sum((r.get('usage') or {}).get('output') or 0 for r in recent)
    recent_s = sum((r.get('duration_ms') or 0) / 1000 for r in recent)
    done = {r.get('trace_id') for r in responses}
    open_requests = [float(r['ts']) for r in requests if r.get('trace_id') not in done]
    serve_log = (run / 'atr-serve.log').read_text(errors='replace') if (run / 'atr-serve.log').exists() else ''
    return {'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'elapsed_min': round((now - run_start(run)) / 60, 1),
            'requests': len(requests), 'responses': len(responses), 'output_tokens': output,
            'tok_per_s': round(output / generating, 1) if generating else None,
            'tok_per_s_last_10min': round(recent_out / recent_s, 1) if recent_s else None,
            'model_minutes': round(generating / 60, 1),
            'in_flight_s': round(now - max(open_requests)) if open_requests else 0,
            'longest_request_min': round(max(((r.get('duration_ms') or 0) / 60000 for r in responses), default=0), 1),
            'upstream_errors': len(re.findall(r'<< (?:4|5)\d\d', serve_log)),
            **stream(run), 'submission_files': submission_files(run)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--store', default='var/stores/store-v63')
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    out = ROOT / 'var/raw/agent-monitor' / args.model
    out.mkdir(parents=True, exist_ok=True)
    finished: set[str] = {p.stem for p in out.glob('*.jsonl') if '"final"' in p.read_text()}
    while True:
        for run in sorted((ROOT / args.store / 'dev-runs').glob('dev-*')):
            if run.name in finished or not glob.glob(str(run / f'atr/{args.model}-serve-*')):
                continue
            if time.time() - run_start(run) > 6 * 3600 and not (out / f'{run.name}.jsonl').exists():
                continue
            record = snapshot(run)
            result = run / 'develop_result.json'
            if result.exists():
                outcome = json.loads(result.read_text())
                record.update(final=True, **{k: outcome.get(k) for k in (
                    'setting', 'agent_stop_reason', 'development_seconds', 'task_status', 'freeze_error', 'http_attempts')})
                finished.add(run.name)
            with (out / f'{run.name}.jsonl').open('a') as stream_out:
                stream_out.write(json.dumps(record) + '\n')
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
