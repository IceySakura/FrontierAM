#!/usr/bin/env python3
"""Backfill judge_reasoning into existing results that lack it.

Reads each case's prediction + reference from results.jsonl, calls the judge LLM,
and writes the reasoning back into task_output.metadata.judge_reasoning (in-place).

Usage:
    export TAIJI_JUDGE_API_KEY="$APP_ID:$APP_KEY"
    export TAIJI_JUDGE_BASE_URL="http://llm-api.model-eval.woa.com/v1"
    export TAIJI_JUDGE_MODEL="api_azure_openai_gpt-5.6-sol"
    export TAIJI_JUDGE_API_STYLE="responses"
    export TAIJI_JUDGE_REASONING_EFFORT="xhigh"

    python3 backfill_judge_reasoning.py \
        --results-dir /path/to/hy3_high_results0720_v1 \
        --concurrency 10
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time

from pathlib import Path

import httpx


sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from nexus.extensions.tasks.taiji_analysis.prompt import TAIJI_JUDGE_PROMPT, TAIJI_JUDGE_SYSTEM_PROMPT


DEFAULT_API_KEY = os.environ.get('TAIJI_JUDGE_API_KEY', '')
DEFAULT_API_BASE = os.environ.get('TAIJI_JUDGE_BASE_URL', 'http://llm-api.model-eval.woa.com/v1')
DEFAULT_MODEL = os.environ.get('TAIJI_JUDGE_MODEL', 'api_azure_openai_gpt-5.6-sol')
DEFAULT_API_STYLE = os.environ.get('TAIJI_JUDGE_API_STYLE', 'responses')
DEFAULT_REASONING_EFFORT = os.environ.get('TAIJI_JUDGE_REASONING_EFFORT', 'xhigh')


def parse_judge_output(output: str) -> tuple[float, str]:
    score_match = re.search(r'<is_consistent>\s*([01])\s*</is_consistent>', output)
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', output, re.DOTALL)
    score = float(score_match.group(1)) if score_match else -1.0
    reasoning = reasoning_match.group(1).strip() if reasoning_match else output.strip()[:300]
    return score, reasoning


async def call_judge(
    http_client: httpx.AsyncClient,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    reasoning_effort: str | None = None,
    max_retries: int = 8,
) -> str:
    url = api_base.rstrip('/').removesuffix('/v1') + '/v1/responses'
    body = {
        'model': model,
        'instructions': TAIJI_JUDGE_SYSTEM_PROMPT,
        'input': prompt,
        'max_output_tokens': 1024,
    }
    if reasoning_effort:
        body['reasoning'] = {'effort': reasoning_effort}

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    for attempt in range(max_retries):
        try:
            resp = await http_client.post(url, json=body, headers=headers, timeout=180)
            if resp.status_code == 429:
                wait = min(5 * (attempt + 1), 30)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            for item in data.get('output', []):
                if item.get('type') == 'message':
                    contents = item.get('content', [])
                    if contents:
                        return contents[0].get('text', '')
            return ''
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(3 * (attempt + 1))
            else:
                return f'API Error: {e}'
    return 'API Error: max retries exceeded (429)'


async def backfill_one(
    record: dict,
    question: str,
    *,
    http_client: httpx.AsyncClient,
    api_base: str,
    api_key: str,
    model: str,
    reasoning_effort: str | None,
) -> tuple[str, float, str]:
    """Returns (run_id, score, reasoning)."""
    task_output = record.get('task_output') or {}
    prediction = task_output.get('prediction', '')
    reference = task_output.get('reference', '')

    if not prediction.strip() or not reference.strip():
        return str(record['run_id']), task_output.get('score', 0), ''

    prompt = TAIJI_JUDGE_PROMPT.format(
        question=question,
        report_a=reference,
        report_b=prediction,
    )
    raw_output = await call_judge(http_client, api_base, api_key, model, prompt, reasoning_effort)
    score, reasoning = parse_judge_output(raw_output)
    return str(record['run_id']), score, reasoning


async def run_all(
    records: list[dict],
    questions: dict[str, str],
    *,
    api_base: str,
    api_key: str,
    model: str,
    reasoning_effort: str | None,
    concurrency: int,
) -> dict[str, str]:
    """Returns {run_id: reasoning}."""
    http_client = httpx.AsyncClient()
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, str] = {}

    async def bounded(record):
        async with semaphore:
            rid = str(record['run_id'])
            question = questions.get(rid, '')
            return await backfill_one(
                record,
                question,
                http_client=http_client,
                api_base=api_base,
                api_key=api_key,
                model=model,
                reasoning_effort=reasoning_effort,
            )

    tasks = [bounded(r) for r in records]
    total = len(tasks)
    for done, coro in enumerate(asyncio.as_completed(tasks), 1):
        rid, score, reasoning = await coro
        results[rid] = reasoning
        if done % 10 == 0 or done == total:
            pct = done * 100 // total
            bar = '█' * (pct // 2) + '░' * (50 - pct // 2)
            print(f'\r  [{bar}] {done}/{total} ({pct}%)', end='', flush=True)
    print()

    await http_client.aclose()
    return results


def main():
    parser = argparse.ArgumentParser(description='Backfill judge_reasoning into existing results')
    parser.add_argument('--results-dir', required=True, help='Directory with results.jsonl + taiji_bench*.db')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--api-base', default=DEFAULT_API_BASE)
    parser.add_argument('--api-key', default=DEFAULT_API_KEY)
    parser.add_argument('--api-style', default=DEFAULT_API_STYLE)
    parser.add_argument('--reasoning-effort', default=DEFAULT_REASONING_EFFORT)
    parser.add_argument('--concurrency', type=int, default=10)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_file = results_dir / 'results.jsonl'

    # Load records
    records = []
    with open(results_file) as f:
        for line in f:
            records.append(json.loads(line))
    print(f'Loaded {len(records)} records from {results_file}')

    # Filter to records that lack reasoning
    to_backfill = [r for r in records if not (r.get('task_output') or {}).get('metadata', {}).get('judge_reasoning')]
    print(f'Need backfill: {len(to_backfill)} (already have: {len(records) - len(to_backfill)})')

    if not to_backfill:
        print('Nothing to do.')
        return

    # Load questions from DB
    questions: dict[str, str] = {}
    db_files = list(results_dir.glob('taiji_bench*.db'))
    if db_files:
        conn = sqlite3.connect(str(db_files[0]))
        for row in conn.execute('SELECT run_id, config FROM runs'):
            cfg = json.loads(row[1]) if row[1] else {}
            questions[str(row[0])] = cfg.get('task_config', {}).get('question', '')
        conn.close()
    print(f'Loaded {len(questions)} questions from DB')

    # Run judge
    effort_str = f', reasoning_effort={args.reasoning_effort}' if args.reasoning_effort else ''
    print(f'Judge: model={args.model}, concurrency={args.concurrency}{effort_str}')
    start = time.time()
    reasoning_map = asyncio.run(
        run_all(
            to_backfill,
            questions,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            reasoning_effort=args.reasoning_effort or None,
            concurrency=args.concurrency,
        )
    )
    elapsed = time.time() - start
    print(f'Completed in {elapsed:.1f}s ({len(reasoning_map) / elapsed:.1f} cases/s)')

    # Update results.jsonl in-place
    updated = 0
    for r in records:
        rid = str(r['run_id'])
        if rid in reasoning_map and reasoning_map[rid]:
            if 'task_output' not in r or r['task_output'] is None:
                r['task_output'] = {}
            if 'metadata' not in r['task_output'] or r['task_output']['metadata'] is None:
                r['task_output']['metadata'] = {}
            r['task_output']['metadata']['judge_reasoning'] = reasoning_map[rid]
            updated += 1

    with open(results_file, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Updated {updated} records in {results_file}')

    # Update DB if exists
    if db_files:
        conn = sqlite3.connect(str(db_files[0]))
        for rid, reasoning in reasoning_map.items():
            if not reasoning:
                continue
            row = conn.execute('SELECT task_output FROM runs WHERE run_id = ?', (rid,)).fetchone()
            if row and row[0]:
                to = json.loads(row[0])
                if 'metadata' not in to or to['metadata'] is None:
                    to['metadata'] = {}
                to['metadata']['judge_reasoning'] = reasoning
                conn.execute(
                    'UPDATE runs SET task_output = ? WHERE run_id = ?', (json.dumps(to, ensure_ascii=False), rid)
                )
        conn.commit()
        conn.close()
        print(f'Updated DB: {db_files[0].name}')

    print(f'\nDone: {results_dir}')


if __name__ == '__main__':
    main()
