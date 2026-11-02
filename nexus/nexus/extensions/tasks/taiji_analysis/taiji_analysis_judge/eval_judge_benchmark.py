#!/usr/bin/env python3
"""Evaluate judge benchmark results by comparing against GT.

For each case:
1. Score match: new_score == gt_score
2. Reasoning match: call judge-judge LLM to compare reasoning consistency

Supports two API styles:
- chat: OpenAI Chat Completions API (/v1/chat/completions) — default
- responses: OpenAI Responses API (/v1/responses) — for GPT-5.6-sol etc.

Usage:
    # Full evaluation (score + reasoning via LLM)
    python3 eval_judge_benchmark.py --benchmark judge_benchmark.parquet --results judge_results_gpt-4o.jsonl --model gpt-4o

    # Score-only (no LLM call needed)
    python3 eval_judge_benchmark.py --benchmark judge_benchmark.parquet --results judge_results_gpt-4o.jsonl --score-only

    # GPT-5.6-sol via Responses API with reasoning effort
    export TAIJI_JUDGE_JUDGE_API_STYLE=responses
    export TAIJI_JUDGE_JUDGE_MODEL=api_azure_openai_gpt-5.6-sol
    export TAIJI_JUDGE_JUDGE_REASONING_EFFORT=xhigh
    python3 eval_judge_benchmark.py --benchmark judge_benchmark.parquet --results judge_results.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import time

import httpx
import pandas as pd

from openai import AsyncOpenAI


DEFAULT_API_BASE = os.environ.get('TAIJI_JUDGE_JUDGE_BASE_URL', 'http://llm-api.model-eval.woa.com/v1')
DEFAULT_MODEL = os.environ.get('TAIJI_JUDGE_JUDGE_MODEL', 'openai/api_google_claude-opus-4-8')
DEFAULT_API_KEY = os.environ.get('TAIJI_JUDGE_JUDGE_API_KEY', '')
DEFAULT_API_STYLE = os.environ.get('TAIJI_JUDGE_JUDGE_API_STYLE', 'chat')
DEFAULT_REASONING_EFFORT = os.environ.get('TAIJI_JUDGE_JUDGE_REASONING_EFFORT', '')

MODELS_NO_TEMPERATURE = {'fable-5', 'fable5', 'claude-fable-5'}


def _supports_temperature(model: str) -> bool:
    model_lower = model.lower()
    return not any(m in model_lower for m in MODELS_NO_TEMPERATURE)


JUDGE_JUDGE_SYSTEM_PROMPT = """你是评审理由的一致性判断专家。
判断两段评审理由是否表达了相同的核心结论。只看核心判断，忽略措辞差异。"""

JUDGE_JUDGE_PROMPT = """判断两段评审理由是否表达相同的核心结论。

## 理由 A（参考）
{reasoning_a}

## 理由 B（待评估）
{reasoning_b}

## 评判标准
- 两段理由指向相同的核心判断（如都认为"根因一致"或"根因不同"）→ 1
- 两段理由的核心结论相反或侧重完全不同的方面 → 0

<result>
<reasoning>一句话理由</reasoning>
<is_consistent>1或0</is_consistent>
</result>"""


def parse_judge_judge_output(output: str) -> tuple[float, str]:
    score_match = re.search(r'<is_consistent>\s*([01])\s*</is_consistent>', output)
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', output, re.DOTALL)
    score = float(score_match.group(1)) if score_match else -1.0
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ''
    return score, reasoning


async def judge_reasoning_chat(
    client: AsyncOpenAI, model: str, prompt: str, reasoning_effort: str | None = None, max_retries: int = 3
) -> str:
    """Call judge-judge via Chat Completions API."""
    for attempt in range(max_retries):
        try:
            kwargs = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': JUDGE_JUDGE_SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                'max_tokens': 256,
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


async def judge_reasoning_responses(
    http_client: httpx.AsyncClient,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    reasoning_effort: str | None = None,
    max_retries: int = 3,
) -> str:
    """Call judge-judge via Responses API (/v1/responses)."""
    url = api_base.rstrip('/').removesuffix('/v1') + '/v1/responses'
    body = {
        'model': model,
        'instructions': JUDGE_JUDGE_SYSTEM_PROMPT,
        'input': prompt,
        'max_output_tokens': 512,
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


async def judge_reasoning_consistency(
    *,
    client: AsyncOpenAI | None,
    http_client: httpx.AsyncClient | None,
    api_base: str,
    api_key: str,
    model: str,
    gt_reasoning: str,
    new_reasoning: str,
    api_style: str,
    reasoning_effort: str | None = None,
    max_retries: int = 3,
) -> tuple[float, str]:
    prompt = JUDGE_JUDGE_PROMPT.format(
        reasoning_a=gt_reasoning,
        reasoning_b=new_reasoning,
    )

    try:
        if api_style == 'responses':
            assert http_client is not None
            output = await judge_reasoning_responses(
                http_client, api_base, api_key, model, prompt, reasoning_effort, max_retries
            )
        else:
            assert client is not None
            output = await judge_reasoning_chat(client, model, prompt, reasoning_effort, max_retries)
        return parse_judge_judge_output(output)
    except Exception as e:
        return -1.0, f'ERROR: {e}'


async def eval_all_reasoning(
    *,
    client: AsyncOpenAI | None,
    http_client: httpx.AsyncClient | None,
    api_base: str,
    api_key: str,
    model: str,
    pairs: list[tuple[str, str]],
    api_style: str,
    reasoning_effort: str | None = None,
    concurrency: int = 300,
) -> list[tuple[float, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded(gt_r, new_r):
        async with semaphore:
            return await judge_reasoning_consistency(
                client=client,
                http_client=http_client,
                api_base=api_base,
                api_key=api_key,
                model=model,
                gt_reasoning=gt_r,
                new_reasoning=new_r,
                api_style=api_style,
                reasoning_effort=reasoning_effort,
            )

    tasks = [bounded(gt_r, new_r) for gt_r, new_r in pairs]
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
    parser = argparse.ArgumentParser(description='Evaluate judge benchmark results')
    parser.add_argument(
        '--benchmark', default='judge_benchmark.parquet', help='Benchmark parquet from build_judge_benchmark.py'
    )
    parser.add_argument('--results', required=True, help='Results jsonl from run_judge_benchmark.py')
    parser.add_argument(
        '--model', default=DEFAULT_MODEL, help='Model for judge-judge LLM (default: TAIJI_JUDGE_JUDGE_MODEL)'
    )
    parser.add_argument(
        '--api-base',
        default=DEFAULT_API_BASE,
        help='OpenAI-compatible API base URL (default: TAIJI_JUDGE_JUDGE_BASE_URL)',
    )
    parser.add_argument('--api-key', default=DEFAULT_API_KEY, help='API key (default: TAIJI_JUDGE_JUDGE_API_KEY)')
    parser.add_argument(
        '--api-style',
        default=DEFAULT_API_STYLE,
        choices=['chat', 'responses'],
        help='API style: chat=Chat Completions, responses=Responses API (default: TAIJI_JUDGE_JUDGE_API_STYLE)',
    )
    parser.add_argument(
        '--reasoning-effort',
        default=DEFAULT_REASONING_EFFORT or None,
        choices=['none', 'low', 'medium', 'high', 'xhigh', 'max'],
        help='Reasoning effort (default: TAIJI_JUDGE_JUDGE_REASONING_EFFORT)',
    )
    parser.add_argument(
        '--score-only', action='store_true', help='Only evaluate score match, skip reasoning comparison'
    )
    parser.add_argument('--concurrency', type=int, default=300, help='Concurrent API calls for reasoning eval')
    parser.add_argument('--output', default=None, help='Output detailed results jsonl')
    args = parser.parse_args()

    if not args.score_only and not args.model and not DEFAULT_MODEL:
        parser.error('--model is required unless --score-only is specified')

    # Load data
    df_benchmark = pd.read_parquet(args.benchmark)
    gt_map = {}
    for _, row in df_benchmark.iterrows():
        key = (row['run_id'], row['experiment'])
        gt_map[key] = {'gt_score': row['gt_score'], 'gt_reasoning': row['gt_reasoning']}

    results = []
    with open(args.results) as f:
        for line in f:
            results.append(json.loads(line))

    print(f'Benchmark: {len(gt_map)} cases')
    print(f'Results: {len(results)} cases')

    # Match results with GT
    eval_items = []
    for r in results:
        key = (r['run_id'], r['experiment'])
        if key not in gt_map:
            continue
        gt = gt_map[key]
        eval_items.append({
            'run_id': r['run_id'],
            'experiment': r['experiment'],
            'gt_score': gt['gt_score'],
            'gt_reasoning': gt['gt_reasoning'],
            'new_score': r['new_score'],
            'new_reasoning': r['new_reasoning'],
        })

    print(f'Matched: {len(eval_items)} cases')

    # Score evaluation
    valid_items = [e for e in eval_items if e['new_score'] >= 0]
    score_matches = sum(1 for e in valid_items if e['new_score'] == e['gt_score'])
    score_accuracy = score_matches / len(valid_items) if valid_items else 0

    print('\n=== Score Evaluation ===')
    print(f'Valid: {len(valid_items)}/{len(eval_items)}')
    print(f'Score accuracy: {score_matches}/{len(valid_items)} ({score_accuracy * 100:.1f}%)')

    # Breakdown
    tp = sum(1 for e in valid_items if e['gt_score'] >= 1 and e['new_score'] >= 1)
    tn = sum(1 for e in valid_items if e['gt_score'] < 1 and e['new_score'] < 1)
    fp = sum(1 for e in valid_items if e['gt_score'] < 1 and e['new_score'] >= 1)
    fn = sum(1 for e in valid_items if e['gt_score'] >= 1 and e['new_score'] < 1)
    print(f'  TP={tp} TN={tn} FP={fp} FN={fn}')
    if tp + fp > 0:
        print(f'  Precision: {tp / (tp + fp) * 100:.1f}%')
    if tp + fn > 0:
        print(f'  Recall: {tp / (tp + fn) * 100:.1f}%')

    # Reasoning evaluation
    if not args.score_only:
        effort_str = f', reasoning_effort={args.reasoning_effort}' if args.reasoning_effort else ''
        print(f'\n=== Reasoning Evaluation (model={args.model}, api_style={args.api_style}{effort_str}) ===')

        client = None
        http_client = None
        if args.api_style == 'responses':
            http_client = httpx.AsyncClient()
        else:
            client = AsyncOpenAI(base_url=args.api_base, api_key=args.api_key or 'not-needed', timeout=120)

        pairs = [(e['gt_reasoning'], e['new_reasoning']) for e in valid_items]
        start = time.time()
        reasoning_results = asyncio.run(
            eval_all_reasoning(
                client=client,
                http_client=http_client,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                pairs=pairs,
                api_style=args.api_style,
                reasoning_effort=args.reasoning_effort,
                concurrency=args.concurrency,
            )
        )
        elapsed = time.time() - start
        print(f'Completed in {elapsed:.1f}s')

        reasoning_valid = [(score, reason) for score, reason in reasoning_results if score >= 0]
        reasoning_matches = sum(1 for score, _ in reasoning_valid if score >= 1)
        reasoning_accuracy = reasoning_matches / len(reasoning_valid) if reasoning_valid else 0

        print(f'Reasoning consistency: {reasoning_matches}/{len(reasoning_valid)} ({reasoning_accuracy * 100:.1f}%)')

        # Overall: both score and reasoning must match
        overall_pass = 0
        for i, e in enumerate(valid_items):
            score_ok = e['new_score'] == e['gt_score']
            reasoning_ok = reasoning_results[i][0] >= 1 if i < len(reasoning_results) else False
            e['score_match'] = score_ok
            e['reasoning_match'] = reasoning_ok
            e['overall_pass'] = score_ok and reasoning_ok
            if score_ok and reasoning_ok:
                overall_pass += 1

        overall_rate = overall_pass / len(valid_items) if valid_items else 0
        print('\n=== Overall ===')
        print(f'Pass (score + reasoning): {overall_pass}/{len(valid_items)} ({overall_rate * 100:.1f}%)')
    else:
        for e in valid_items:
            e['score_match'] = e['new_score'] == e['gt_score']
            e['reasoning_match'] = None
            e['overall_pass'] = None

    # Save detailed results
    if args.output:
        with open(args.output, 'w') as f:
            for e in eval_items:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        print(f'\nDetailed results: {args.output}')


if __name__ == '__main__':
    main()
