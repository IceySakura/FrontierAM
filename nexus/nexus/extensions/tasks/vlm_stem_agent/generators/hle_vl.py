"""Batch generator for the HLE-VL multimodal benchmark.

HLE-VL rows carry no answer-format instructions, so the judge-facing format
block is prepended here. Correctness is left to the official HLE judge unless
the config sets an ``evaluator_type``.

Set ``HLE_VL_LIMIT=0`` for the complete dataset and ``HLE_VL_RUNTIME=gongfeng``
for the remote runtime. For example::

    HLE_VL_LIMIT=20 nexus run-batch \
      -g nexus/extensions/tasks/vlm_stem_agent/generators/hle_vl.py \
      -d /tmp/vlm_stem_hle_vl_smoke.db --max-concurrent 1
"""

from typing import Any

from nexus.extensions.tasks.vlm_stem_agent.generators import _common


DEFAULT_DATASET = (
    '/apdcephfs_cq8/share_1324356/calehu/mm_math_benchmarks/HLE-VL/annotations_hle-vl_342_platform_cos.jsonl'
)
_SYSTEM_PROMPT = (
    'You are a helpful AI assistant solving difficult multimodal questions. Use the supplied image as '
    'primary evidence. You may call search and visit when external factual verification is useful, and '
    'python when calculation or symbolic manipulation is useful. Never fabricate tool results. Tools are '
    'optional: do not call them when they do not help. When you have enough evidence, stop calling tools '
    'and provide the final response in the format requested by the user.'
)


def _build_question(row: dict[str, Any], spec: _common.DatasetSpec) -> str:
    answer_label = 'Answer' if _common.answer_type(row, spec) == 'multipleChoice' else 'Exact Answer'
    return (
        'Your response should be in the following format:\n'
        'Explanation: your explanation for your final answer\n'
        f'{answer_label}: your succinct, final answer\n'
        'Confidence: your confidence score between 0% and 100% for your answer\n\n'
        f'{_common.text_item(row, spec)["value"]}'
    )


SPEC = _common.DatasetSpec(
    name='hle_vl',
    label='HLE-VL',
    env_prefix='HLE_VL',
    default_dataset=DEFAULT_DATASET,
    system_prompt=_SYSTEM_PROMPT,
    build_question=_build_question,
    tools=('search_stateful/search', 'search_stateful/visit', 'jupyter_with_image'),
)


def generate_configs():
    return _common.generate_configs(SPEC)
