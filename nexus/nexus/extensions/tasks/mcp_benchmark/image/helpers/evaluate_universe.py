#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from pathlib import Path


UNIVERSE_ROOT = Path('/opt/mcp-benchmarks/datasets/universe')
if str(UNIVERSE_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIVERSE_ROOT))

from mcpuniverse.common.context import Context  # noqa: E402
from mcpuniverse.evaluator import Evaluator  # noqa: E402


def _build_context(env_values: dict[str, str]) -> Context:
    context = Context()
    context.env.update(env_values)
    os.environ.update(env_values)
    return context


async def _run(input_payload: dict) -> dict:
    response = input_payload.get('response', '')
    evaluators = input_payload.get('evaluators', [])
    env_values = input_payload.get('context_env', {})
    context = _build_context(env_values)

    results = []
    for evaluator_config in evaluators:
        evaluator = Evaluator(evaluator_config, context=context)
        result = await evaluator.evaluate(response)
        results.append(result.model_dump(mode='json'))

    total = len(results)
    passed_count = sum(1 for item in results if item.get('passed'))
    score = float(passed_count / total) if total else 0.0
    return {
        'passed': total > 0 and passed_count == total,
        'score': score,
        'details': {
            'results': results,
            'passed_count': passed_count,
            'total': total,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    result = asyncio.run(_run(payload))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
