"""Batch generator for the MathVision multimodal benchmark.

MathVision rows already include their answer-format Hint and Choices, so the
question is passed through unchanged. Correctness is left to the official
MathVision evaluator unless the config sets an ``evaluator_type``.

Set ``MATHVISION_LIMIT=0`` for the complete dataset and
``MATHVISION_RUNTIME=gongfeng`` for the remote runtime. For example::

    MATHVISION_LIMIT=20 nexus run-batch \
      -g nexus/extensions/tasks/vlm_stem_agent/generators/mathvision.py \
      -d /tmp/vlm_stem_mathvision_smoke.db --max-concurrent 1
"""

from typing import Any

from nexus.extensions.tasks.vlm_stem_agent.generators import _common


DEFAULT_DATASET = '/apdcephfs_cq8/share_1324356/shared_info/rileyrlluo/benchmarks/MathVision/UTF-8MathVision_cos.jsonl'
_SYSTEM_PROMPT = (
    'You are a careful mathematical visual reasoning assistant. You MUST call '
    'the python tool and view every supplied image through its stable sandbox '
    'path before reasoning; do not answer before viewing the images. Continue '
    'using the python tool when cropping, zooming, measuring, counting, or '
    'symbolic/numeric computation would improve reliability. When sufficient '
    'evidence is available, stop calling tools and return the final answer in '
    'the exact format requested by the user.'
)


def _normalize_answer(value: Any) -> str:
    # The source JSONL contains one unescaped ``\frac`` sequence. Python's JSON
    # decoder turns its ``\f`` prefix into a form-feed character; repair that
    # exact command without rewriting unrelated control characters.
    return str(value).replace('\frac', r'\frac').strip()


def _build_question(row: dict[str, Any], spec: _common.DatasetSpec) -> str:
    return _common.text_item(row, spec)['value']


def _extra_metadata(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = row.get('extra_info')
    if not isinstance(extra_info, dict):
        extra_info = {}
    return {
        'level': extra_info.get('level'),
        'category': extra_info.get('category'),
    }


SPEC = _common.DatasetSpec(
    name='mathvision',
    label='MathVision',
    env_prefix='MATHVISION',
    default_dataset=DEFAULT_DATASET,
    system_prompt=_SYSTEM_PROMPT,
    build_question=_build_question,
    normalize_answer=_normalize_answer,
    extra_metadata=_extra_metadata,
)


def generate_configs():
    return _common.generate_configs(SPEC)
