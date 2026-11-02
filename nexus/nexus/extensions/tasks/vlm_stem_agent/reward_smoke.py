"""Reward-system-only smoke runner for the multimodal scoring payload.

Scores a canned prediction so the reward router can be exercised without a
model, a sandbox, or a rollout. Must run from inside IDC: a DevCloud machine
gets HTTP 403 from the egress gate before the request reaches the platform.

Prefer ``--jsonl``: it replays a real training row, so the question, the image
and the reference answer are the ones the router actually sees in production.
Without it the runner falls back to a synthetic image that merely proves the
payload is accepted.

Usage::

    NEXUS_REWARD_SYSTEM_TOKEN=... python -m nexus.extensions.tasks.vlm_stem_agent.reward_smoke \\
        --jsonl /path/to/rows.jsonl --row 0
"""

import argparse
import asyncio
import base64
import json
import logging
import os

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageDraw

from nexus.extensions.tasks.search_stateful.evaluator import RewardSystemEvaluator
from nexus.extensions.tasks.vlm_stem_agent.task import (
    DEFAULT_MAX_IMAGE_BYTES,
    normalize_images_to_data_uris,
)


@dataclass
class SmokeCase:
    """One question/image/reference triple plus the router config it belongs to."""

    question: str
    images: list[str]
    reference: str
    router_id: str
    extra: dict


def _image_showing(text: str) -> str:
    """Render the answer into the picture, for the synthetic fallback case."""
    image = Image.new('RGB', (160, 80), color='white')
    ImageDraw.Draw(image).text((40, 30), text, fill='black')
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


async def _case_from_jsonl(args) -> SmokeCase:
    """Replay one training row, normalizing its images exactly as the task does."""
    with open(args.jsonl, encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    task_config = rows[args.row]['nexus']['task_config']
    # Rows carry local image paths; the task stages them as Base64 before the
    # agent runs, and the judge must see those same bytes.
    images = await normalize_images_to_data_uris(
        task_config['images'],
        image_download_timeout=task_config.get('image_download_timeout', 60),
        max_image_bytes=task_config.get('max_image_bytes', DEFAULT_MAX_IMAGE_BYTES),
        min_pixels=task_config.get('min_pixels'),
        max_pixels=task_config.get('max_pixels'),
    )
    reference = task_config.get('ground_truth') or task_config['answer']
    return SmokeCase(
        question=task_config['question'],
        images=images,
        reference=reference,
        router_id=args.router_id or task_config['reward_system_router_id'],
        extra=task_config.get('reward_system_extra') or {},
    )


def _synthetic_case(args) -> SmokeCase:
    return SmokeCase(
        question='What is the number shown in the image? Answer with the integer only.',
        images=[_image_showing('42')],
        reference='42',
        router_id=args.router_id or 'STEMVerifierA20B--2',
        extra={},
    )


async def _score(args, case: SmokeCase, prediction: str, images: list[str] | None, strip) -> dict:
    evaluator = RewardSystemEvaluator(
        user_token=args.token or os.environ.get('NEXUS_REWARD_SYSTEM_TOKEN'),
        exp_id=args.exp_id,
        router_id=case.router_id,
        extra=case.extra,
        images=images,
        strip_multimodal_content=strip,
    )
    score = await evaluator.evaluate(
        prediction=prediction,
        reference=case.reference,
        question=case.question,
    )
    return {'score': score, **(evaluator.get_last_judge_result() or {})}


async def _run(args) -> None:
    case = await _case_from_jsonl(args) if args.jsonl else _synthetic_case(args)
    print(
        json.dumps(
            {
                'router_id': case.router_id,
                'question': case.question[:200],
                'reference': case.reference[:200],
                'image_count': len(case.images),
                'image_uri_lengths': [len(image) for image in case.images],
                'extra_keys': sorted(case.extra),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    wrong = args.wrong_answer or '以上条件不足，无法求解。'
    cases = [
        ('multimodal_correct', case.reference, case.images, None),
        ('multimodal_wrong', wrong, case.images, None),
        ('stripped_by_router_id', case.reference, case.images, [case.router_id]),
        ('stripped_globally', case.reference, case.images, True),
        ('no_images', case.reference, None, None),
    ]
    results = {}
    for name, prediction, images, strip in cases:
        results[name] = await _score(args, case, prediction, images, strip)
    print(json.dumps(results, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', help='Training rows; replays row[--row] instead of the synthetic case')
    parser.add_argument('--row', type=int, default=0)
    parser.add_argument('--router-id', help='Overrides the row router_id; required for the synthetic case')
    parser.add_argument('--token')
    parser.add_argument('--exp-id', default='nexus-reward-smoke')
    parser.add_argument('--wrong-answer')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    logging.basicConfig(level=arguments.log_level, format='%(levelname)s %(name)s %(message)s')
    asyncio.run(_run(arguments))
