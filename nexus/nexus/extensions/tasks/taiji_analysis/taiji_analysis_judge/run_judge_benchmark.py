#!/usr/bin/env python3
"""Run a judge model on the judge benchmark dataset.

Calls a judge LLM (via OpenAI-compatible API) on each case, producing
(new_score, new_reasoning) to be compared against GT.

Supports two API styles:
- chat: OpenAI Chat Completions API (/v1/chat/completions) — default
- responses: OpenAI Responses API (/v1/responses) — for GPT-5.6-sol etc.

Usage:
    python3 run_judge_benchmark.py --input judge_benchmark.parquet --model gpt-4o
    python3 run_judge_benchmark.py --input judge_benchmark.parquet --model claude-opus-4-20250514 --api-base https://...
    python3 run_judge_benchmark.py --input judge_benchmark.parquet --model hy3-high --prompt-file custom_judge.txt

    # GPT-5.6-sol via Responses API with reasoning effort
    export TAIJI_JUDGE_AGENT_API_STYLE=responses
    export TAIJI_JUDGE_AGENT_MODEL=api_azure_openai_gpt-5.6-sol
    export TAIJI_JUDGE_AGENT_REASONING_EFFORT=xhigh
    python3 run_judge_benchmark.py --input judge_benchmark.parquet
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

from openai import AsyncOpenAI


sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from nexus.extensions.tasks.taiji_analysis.prompt import TAIJI_JUDGE_PROMPT, TAIJI_JUDGE_SYSTEM_PROMPT


DEFAULT_API_BASE = os.environ.get('TAIJI_JUDGE_AGENT_BASE_URL', 'http://llm-api.model-eval.woa.com/v1')
DEFAULT_MODEL = os.environ.get('TAIJI_JUDGE_AGENT_MODEL', 'openai/api_google_claude-opus-4-8')
DEFAULT_API_KEY = os.environ.get('TAIJI_JUDGE_AGENT_API_KEY', '')
DEFAULT_API_STYLE = os.environ.get('TAIJI_JUDGE_AGENT_API_STYLE', 'chat')
DEFAULT_REASONING_EFFORT = os.environ.get('TAIJI_JUDGE_AGENT_REASONING_EFFORT', '')

MODELS_NO_TEMPERATURE = {'fable-5', 'fable5', 'claude-fable-5'}


def _supports_temperature(model: str) -> bool:
    model_lower = model.lower()
    return not any(m in model_lower for m in MODELS_NO_TEMPERATURE)


def parse_judge_output(output: str) -> tuple[float, str]:
    score_match = re.search(r'<is_consistent>\s*([01])\s*</is_consistent>', output)
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', output, re.DOTALL)
    score = float(score_match.group(1)) if score_match else -1.0
    reasoning = reasoning_match.group(1).strip() if reasoning_match else output.strip()[:200]
    return score, reasoning


async def run_judge_chat(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    prompt: str,
    reasoning_effort: str | None = None,
    max_retries: int = 3,
) -> str:
    """Call judge via Chat Completions API."""
    for attempt in range(max_retries):
        try:
            kwargs = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                'max_tokens': 512,
            }
            if _supports_temperature(model):
                kwargs['temperature'] = 0
            if reasoning_effort:
                kwargs['reasoning_effort'] = reasoning_effort
            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ''
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
            else:
                raise
    return ''


async def run_judge_responses(
    http_client: httpx.AsyncClient,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    prompt: str,
    reasoning_effort: str | None = None,
    max_retries: int = 3,
) -> str:
    """Call judge via Responses API (/v1/responses)."""
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
            resp.raise_for_status()
            data = resp.json()
            for item in data.get('output', []):
                if item.get('type') == 'message':
                    contents = item.get('content', [])
                    if contents:
                        return contents[0].get('text', '')
            return ''
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
            else:
                raise
    return ''


async def run_judge(
    case: dict,
    *,
    client: AsyncOpenAI | None,
    http_client: httpx.AsyncClient | None,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    judge_prompt_template: str,
    api_style: str,
    reasoning_effort: str | None = None,
) -> dict:
    prompt = judge_prompt_template.format(
        question=case['question'],
        report_a=case['reference'],
        report_b=case['prediction'],
    )

    try:
        if api_style == 'responses':
            assert http_client is not None
            output = await run_judge_responses(
                http_client, api_base, api_key, model, system_prompt, prompt, reasoning_effort
            )
        else:
            assert client is not None
            output = await run_judge_chat(client, model, system_prompt, prompt, reasoning_effort)

        score, reasoning = parse_judge_output(output)
        return {
            'run_id': case['run_id'],
            'experiment': case['experiment'],
            'new_score': score,
            'new_reasoning': reasoning,
            'raw_output': output,
        }
    except Exception as e:
        return {
            'run_id': case['run_id'],
            'experiment': case['experiment'],
            'new_score': -1.0,
            'new_reasoning': f'ERROR: {e}',
            'raw_output': '',
        }


async def run_all(
    cases: list[dict],
    *,
    client: AsyncOpenAI | None,
    http_client: httpx.AsyncClient | None,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    judge_prompt_template: str,
    api_style: str,
    reasoning_effort: str | None = None,
    concurrency: int = 300,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_run(case):
        async with semaphore:
            return await run_judge(
                case,
                client=client,
                http_client=http_client,
                api_base=api_base,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                judge_prompt_template=judge_prompt_template,
                api_style=api_style,
                reasoning_effort=reasoning_effort,
            )

    tasks = [bounded_run(c) for c in cases]
    total = len(tasks)
    for done, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        results.append(result)
        pct = done * 100 // total
        bar = '█' * (pct // 2) + '░' * (50 - pct // 2)
        print(f'\r  [{bar}] {done}/{total} ({pct}%)', end='', flush=True)
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description='Run judge benchmark')
    parser.add_argument(
        '--input', default='judge_benchmark.parquet', help='Input parquet from build_judge_benchmark.py'
    )
    parser.add_argument('--output', default=None, help='Output jsonl path (default: judge_results_{model}.jsonl)')
    parser.add_argument(
        '--model', default=DEFAULT_MODEL, help='Model name for the judge LLM (default: TAIJI_JUDGE_AGENT_MODEL)'
    )
    parser.add_argument(
        '--api-base',
        default=DEFAULT_API_BASE,
        help='OpenAI-compatible API base URL (default: TAIJI_JUDGE_AGENT_BASE_URL)',
    )
    parser.add_argument('--api-key', default=DEFAULT_API_KEY, help='API key (default: TAIJI_JUDGE_AGENT_API_KEY)')
    parser.add_argument(
        '--api-style',
        default=DEFAULT_API_STYLE,
        choices=['chat', 'responses'],
        help='API style: chat=Chat Completions, responses=Responses API (default: TAIJI_JUDGE_AGENT_API_STYLE)',
    )
    parser.add_argument(
        '--reasoning-effort',
        default=DEFAULT_REASONING_EFFORT or None,
        choices=['none', 'low', 'medium', 'high', 'xhigh', 'max'],
        help='Reasoning effort (default: TAIJI_JUDGE_AGENT_REASONING_EFFORT)',
    )
    parser.add_argument('--system-prompt', default=None, help='Custom system prompt (default: built-in)')
    parser.add_argument('--prompt-file', default=None, help='File containing custom judge prompt template')
    parser.add_argument('--concurrency', type=int, default=300, help='Number of concurrent API calls')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of cases to run (for testing)')
    args = parser.parse_args()

    # Load benchmark
    df = pd.read_parquet(args.input)
    cases = df.to_dict('records')
    print(f'Loaded {len(cases)} cases from {args.input}')

    if args.limit:
        cases = cases[: args.limit]
        print(f'Limited to {len(cases)} cases')

    # Setup prompts
    system_prompt = args.system_prompt or TAIJI_JUDGE_SYSTEM_PROMPT
    judge_prompt_template = Path(args.prompt_file).read_text() if args.prompt_file else TAIJI_JUDGE_PROMPT

    # Setup clients
    client = None
    http_client = None
    if args.api_style == 'responses':
        http_client = httpx.AsyncClient()
    else:
        client = AsyncOpenAI(
            base_url=args.api_base,
            api_key=args.api_key or 'not-needed',
            timeout=120,
        )

    # Run
    effort_str = f', reasoning_effort={args.reasoning_effort}' if args.reasoning_effort else ''
    print(f'Running judge: model={args.model}, api_style={args.api_style}, concurrency={args.concurrency}{effort_str}')
    start = time.time()
    results = asyncio.run(
        run_all(
            cases,
            client=client,
            http_client=http_client,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            system_prompt=system_prompt,
            judge_prompt_template=judge_prompt_template,
            api_style=args.api_style,
            reasoning_effort=args.reasoning_effort,
            concurrency=args.concurrency,
        )
    )
    elapsed = time.time() - start
    print(f'Completed in {elapsed:.1f}s ({len(results) / elapsed:.1f} cases/s)')

    # Save
    output_path = args.output or f'judge_results_{args.model.replace("/", "_")}.jsonl'
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Written to {output_path}')

    # Quick stats
    valid = [r for r in results if r['new_score'] >= 0]
    if valid:
        gt_scores = {(c['run_id'], c['experiment']): c['gt_score'] for c in cases}
        score_match = sum(1 for r in valid if r['new_score'] == gt_scores.get((r['run_id'], r['experiment']), -1))
        print('\nQuick stats (score only, no reasoning check):')
        print(f'  Valid: {len(valid)}/{len(results)}')
        print(f'  Score match: {score_match}/{len(valid)} ({score_match / len(valid) * 100:.1f}%)')


if __name__ == '__main__':
    main()
