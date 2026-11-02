#!/usr/bin/env python3
"""Re-judge existing model predictions against a new GT (reference).

Takes results from an existing experiment (with predictions already generated),
replaces the reference with a new GT source, and re-runs the judge LLM to produce
updated scores. This avoids re-running the expensive agent model.

Usage:
    export TAIJI_JUDGE_API_KEY="..."
    export TAIJI_JUDGE_BASE_URL="http://llm-api.model-eval.woa.com/v1"
    export TAIJI_JUDGE_MODEL="openai/api_azure_openai_gpt-5.6-sol"
    export TAIJI_JUDGE_API_STYLE="responses"
    export TAIJI_JUDGE_REASONING_EFFORT="xhigh"

    python3 rejudge_with_new_gt.py \
        --source-results /path/to/opus4.8_low_results0713 \
        --new-gt /path/to/dataset_v1_newrun_0720_390/dataset.parquet \
        --output /path/to/opus4.8_low_results0720_newgt \
        --concurrency 30
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

from pathlib import Path

import httpx
import pandas as pd


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


async def judge_responses(
    http_client: httpx.AsyncClient,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    prompt: str,
    reasoning_effort: str | None = None,
    max_retries: int = 8,
) -> str:
    url = api_base.rstrip('/').removesuffix('/v1') + '/v1/responses'
    body = {
        'model': model,
        'instructions': system_prompt,
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


async def judge_chat(
    client, model: str, system_prompt: str, prompt: str, reasoning_effort: str | None = None, max_retries: int = 3
) -> str:
    for attempt in range(max_retries):
        try:
            kwargs = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                'max_tokens': 512,
                'temperature': 0,
            }
            if reasoning_effort:
                kwargs['reasoning_effort'] = reasoning_effort
            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ''
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
            else:
                return f'API Error: {e}'
    return ''


async def rejudge_one(
    record: dict,
    new_reference: str,
    *,
    http_client: httpx.AsyncClient | None,
    openai_client=None,
    api_base: str,
    api_key: str,
    model: str,
    api_style: str,
    reasoning_effort: str | None,
) -> dict:
    task_output = record.get('task_output') or {}
    prediction = task_output.get('prediction', '')
    question = ''
    config = record.get('config') or {}
    task_config = config.get('task_config', {})
    question = task_config.get('question', '')

    if not prediction.strip():
        new_score = 0.0
        new_reasoning = 'Empty prediction'
        raw_output = ''
    elif not new_reference.strip():
        new_score = 0.0
        new_reasoning = 'Empty reference'
        raw_output = ''
    else:
        prompt = TAIJI_JUDGE_PROMPT.format(
            question=question,
            report_a=new_reference,
            report_b=prediction,
        )
        if api_style == 'responses':
            assert http_client is not None
            raw_output = await judge_responses(
                http_client, api_base, api_key, model, TAIJI_JUDGE_SYSTEM_PROMPT, prompt, reasoning_effort
            )
        else:
            raw_output = await judge_chat(openai_client, model, TAIJI_JUDGE_SYSTEM_PROMPT, prompt, reasoning_effort)

        new_score, new_reasoning = parse_judge_output(raw_output)
        if new_score < 0:
            new_score = 0.0

    new_record = dict(record)
    new_record['task_output'] = {
        'score': new_score,
        'prediction': prediction,
        'reference': new_reference,
        'status': task_output.get('status', 'completed'),
        'metadata': {
            'judge_reasoning': new_reasoning,
            'original_score': task_output.get('score'),
            'original_reference_len': len(task_output.get('reference', '')),
        },
    }
    return new_record


async def run_all(
    records: list[dict],
    new_gt: dict[str, str],
    *,
    api_base: str,
    api_key: str,
    model: str,
    api_style: str,
    reasoning_effort: str | None,
    concurrency: int,
) -> list[dict]:
    http_client = None
    openai_client = None
    if api_style == 'responses':
        http_client = httpx.AsyncClient()
    else:
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(base_url=api_base, api_key=api_key or 'not-needed', timeout=120)

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded(record):
        async with semaphore:
            rid = str(record['run_id'])
            ref = new_gt.get(rid, '')
            return await rejudge_one(
                record,
                ref,
                http_client=http_client,
                openai_client=openai_client,
                api_base=api_base,
                api_key=api_key,
                model=model,
                api_style=api_style,
                reasoning_effort=reasoning_effort,
            )

    tasks = [bounded(r) for r in records]
    total = len(tasks)
    for done, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        results.append(result)
        if done % 10 == 0 or done == total:
            pct = done * 100 // total
            bar = '█' * (pct // 2) + '░' * (50 - pct // 2)
            print(f'\r  [{bar}] {done}/{total} ({pct}%)', end='', flush=True)
    print()

    if http_client:
        await http_client.aclose()

    return results


def load_source_results(source_dir: Path) -> list[dict]:
    results_file = source_dir / 'results.jsonl'
    records = []
    with open(results_file) as f:
        for line in f:
            records.append(json.loads(line))

    # Try to enrich with question from DB
    db_path = source_dir / 'taiji_bench.db'
    if db_path.exists():
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        questions = {}
        for row in conn.execute('SELECT run_id, config FROM runs'):
            cfg = json.loads(row['config']) if row['config'] else {}
            tc = cfg.get('task_config', {})
            questions[str(row['run_id'])] = tc.get('question', '')
        conn.close()

        for r in records:
            rid = str(r['run_id'])
            if rid in questions and not r.get('config', {}).get('task_config', {}).get('question'):
                if 'config' not in r:
                    r['config'] = {}
                if 'task_config' not in r['config']:
                    r['config']['task_config'] = {}
                r['config']['task_config']['question'] = questions[rid]

    return records


def create_db(output_dir: Path, results: list[dict], source_db: Path, new_gt: dict[str, str], db_name: str):
    """Create a new taiji_bench DB with updated task_output (new reference + score)."""
    import shutil
    import sqlite3

    db_path = output_dir / db_name
    # Copy source DB as base (preserves schema, trace events, etc.)
    shutil.copy2(source_db, db_path)

    conn = sqlite3.connect(str(db_path))

    # Update each run's config (task_config.answer) and task_output
    for r in results:
        rid = str(r['run_id'])
        task_output_json = json.dumps(r['task_output'], ensure_ascii=False)

        # Update task_output in runs table
        conn.execute('UPDATE runs SET task_output = ? WHERE run_id = ?', (task_output_json, rid))

        # Update config.task_config.answer to new GT
        row = conn.execute('SELECT config FROM runs WHERE run_id = ?', (rid,)).fetchone()
        if row and row[0]:
            config = json.loads(row[0])
            if 'task_config' in config:
                config['task_config']['answer'] = new_gt.get(rid, '')
                conn.execute(
                    'UPDATE runs SET config = ? WHERE run_id = ?', (json.dumps(config, ensure_ascii=False), rid)
                )

    conn.commit()
    conn.close()
    print(f'{db_name}: {len(results)} runs updated')


def main():
    parser = argparse.ArgumentParser(description='Re-judge predictions with new GT')
    parser.add_argument(
        '--source-results', required=True, help='Directory with existing results (results.jsonl + taiji_bench.db)'
    )
    parser.add_argument('--new-gt', required=True, help='Parquet file with new GT (columns: instance_id, answer)')
    parser.add_argument('--output', required=True, help='Output directory for new results')
    parser.add_argument(
        '--db-name', default='taiji_bench_v4.db', help='Output DB filename (default: taiji_bench_v4.db)'
    )
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--api-base', default=DEFAULT_API_BASE)
    parser.add_argument('--api-key', default=DEFAULT_API_KEY)
    parser.add_argument('--api-style', default=DEFAULT_API_STYLE, choices=['chat', 'responses'])
    parser.add_argument('--reasoning-effort', default=DEFAULT_REASONING_EFFORT)
    parser.add_argument('--concurrency', type=int, default=30)
    args = parser.parse_args()

    source_dir = Path(args.source_results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source predictions
    records = load_source_results(source_dir)
    print(f'Loaded {len(records)} records from {source_dir}')

    # Load new GT
    gt_df = pd.read_parquet(args.new_gt, columns=['instance_id', 'answer'])
    new_gt = {str(row['instance_id']): row['answer'] or '' for _, row in gt_df.iterrows()}
    print(f'Loaded {len(new_gt)} new GT entries from {args.new_gt}')

    # Filter to only records that have new GT
    records = [r for r in records if str(r['run_id']) in new_gt]
    print(f'Matched records: {len(records)}')

    if not records:
        print('ERROR: No matching records found')
        return

    # Run re-judging
    effort_str = f', reasoning_effort={args.reasoning_effort}' if args.reasoning_effort else ''
    print(f'Judge: model={args.model}, api_style={args.api_style}, concurrency={args.concurrency}{effort_str}')
    start = time.time()
    results = asyncio.run(
        run_all(
            records,
            new_gt,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            api_style=args.api_style,
            reasoning_effort=args.reasoning_effort or None,
            concurrency=args.concurrency,
        )
    )
    elapsed = time.time() - start
    print(f'Completed in {elapsed:.1f}s ({len(results) / elapsed:.1f} cases/s)')

    # Sort by run_id
    results.sort(key=lambda r: int(r['run_id']))

    # Write results.jsonl
    results_file = output_dir / 'results.jsonl'
    with open(results_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'results.jsonl: {len(results)} records')

    # Compute and write metrics.json
    scores = [r['task_output']['score'] for r in results]
    passed = sum(1 for s in scores if s >= 1.0)
    pass_rate = round(passed / len(results), 4) if results else 0.0
    metrics = {
        'total': len(results),
        'score': {
            'pass_rate': pass_rate,
            'passed': passed,
            'failed': len(results) - passed,
        },
        'judge_model': args.model,
        'new_gt_source': str(args.new_gt),
        'original_source': str(source_dir),
    }
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'metrics.json: pass_rate={pass_rate * 100:.1f}% ({passed}/{len(results)})')

    # Create DB
    source_db = source_dir / 'taiji_bench.db'
    if source_db.exists():
        create_db(output_dir, results, source_db, new_gt, args.db_name)
    else:
        print(f'WARNING: source DB not found at {source_db}, skipping DB creation')

    print(f'\nDone: {output_dir}')


if __name__ == '__main__':
    main()
