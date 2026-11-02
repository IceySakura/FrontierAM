#!/usr/bin/env python3
"""Initialize benchmark cases from a JSON file of new cases.

Each case must have: url, question. Optional: inst_id, task_id, ws_id, sender.
If inst_id/task_id/ws_id are not provided, they will be extracted from the URL.

Usage:
    python3 nexus/extensions/tasks/taiji_analysis/add_cases/init_cases.py --input new_cases.json --output ./new_cases/

Input format (new_cases.json):
    [
        {"url": "https://hunyuanaide.taiji.woa.com/...", "question": "帮我看下..."},
        {"url": "https://...", "question": "这个实例为什么..."}
    ]

Output structure:
    new_cases/
    ├── dataset.json
    └── dataset.parquet

Then run GT generation:
    TAIJI_SKIP_JUDGE=1 TAIJI_DATASET_PATH=./new_cases/dataset.parquet \
      uv run nexus run-batch -g nexus/extensions/tasks/taiji_analysis/generator.py \
      -d ./new_cases/taiji_bench.db --max-concurrent 10
"""

import argparse
import json
import re
import sys

from pathlib import Path

import pandas as pd


INST_ID_RE = re.compile(r'instId=([a-f0-9]+)')
TASK_ID_RE = re.compile(r'taskID=(basic_train_[^\s&"\']+)')
WSID_RE = re.compile(r'wsId=(\d+)')


def extract_from_url(url: str) -> dict:
    """Extract instId, taskID, wsId from a Taiji URL."""
    inst = INST_ID_RE.search(url)
    task = TASK_ID_RE.search(url)
    wsid = WSID_RE.search(url)
    return {
        'inst_id': inst.group(1) if inst else '',
        'task_id': task.group(1) if task else '',
        'ws_id': wsid.group(1) if wsid else '',
    }


def main():
    parser = argparse.ArgumentParser(description='Initialize benchmark cases for GT generation')
    parser.add_argument('--input', '-i', required=True, help='JSON file with cases (url + question)')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    args = parser.parse_args()

    with open(args.input, encoding='utf-8') as f:
        raw_cases = json.load(f)

    if not isinstance(raw_cases, list):
        print('Error: input must be a JSON array.', file=sys.stderr)
        sys.exit(1)

    cases = []
    seen_inst_ids = set()

    for i, raw in enumerate(raw_cases):
        url = raw.get('url', '')
        question = raw.get('question', '')
        if not url or not question:
            print(f'  SKIP case {i}: missing url or question')
            continue

        extracted = extract_from_url(url)
        inst_id = raw.get('inst_id') or extracted['inst_id']
        task_id = raw.get('task_id') or extracted['task_id']
        ws_id = raw.get('ws_id') or extracted['ws_id']

        if not inst_id:
            print(f'  SKIP case {i}: cannot determine inst_id')
            continue
        if inst_id in seen_inst_ids:
            print(f'  SKIP case {i}: duplicate inst_id {inst_id}')
            continue
        seen_inst_ids.add(inst_id)

        cases.append({
            'instance_id': inst_id,
            'run_id': inst_id,
            'session_id': '',
            'question': question,
            'answer': '',
            'inst_id': inst_id,
            'task_id': task_id,
            'ws_id': ws_id,
            'url': url,
            'sender': raw.get('sender', 'benchmark'),
            'started_at': 0,
        })

    if not cases:
        print('Error: no valid cases.', file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_json = output_dir / 'dataset.json'
    dataset_parquet = output_dir / 'dataset.parquet'

    with open(dataset_json, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(cases)
    df.to_parquet(dataset_parquet, index=False)

    print(f'\nDone: {len(cases)} cases initialized')
    print(f'  {dataset_json}')
    print(f'  {dataset_parquet}')
    print('\nNext step — run GT generation:')
    print(f'  TAIJI_SKIP_JUDGE=1 TAIJI_DATASET_PATH={dataset_parquet} \\')
    print('    uv run nexus run-batch -g nexus/extensions/tasks/taiji_analysis/generator.py \\')
    print(f'    -d {output_dir}/taiji_bench.db --max-concurrent 10')


if __name__ == '__main__':
    main()
