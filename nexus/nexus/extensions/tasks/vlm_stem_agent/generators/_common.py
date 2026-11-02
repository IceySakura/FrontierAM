"""Shared batch-generator machinery for the vlm_stem_agent datasets.

Every supported benchmark ships the same JSONL shape -- one user message whose
content list holds ``image_url`` and ``text`` items -- so only question
assembly, answer normalization, and metadata differ per dataset. Those live in
a :class:`DatasetSpec`; everything else (env parsing, run-input assembly, the
batch loop) is shared.
"""

import json
import os

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.vlm_stem_agent.smoke import (
    DEFAULT_RUNTIME_IMAGE,
    build_run_input,
)
from nexus.extensions.tasks.vlm_stem_agent.task import (
    DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
    DEFAULT_MAX_IMAGE_BYTES,
)


DEFAULT_TIMEOUT = 300
DEFAULT_INFERENCE_TIMEOUT = 10_800
DEFAULT_OBSERVATION_TIMEOUT = 7_200
SEARCH_TOOLS = ('search_stateful/search', 'search_stateful/visit')
# Both search tools read one shared block, so they cannot be configured apart.
SEARCH_TOOL_CONFIG_KEY = 'search_stateful'
_TOOL_IMPORTS = {
    'search_stateful/search': 'nexus.extensions.tools.search_stateful.search',
    'search_stateful/visit': 'nexus.extensions.tools.search_stateful.visit',
    'jupyter_with_image': 'nexus.extensions.tools.code_interpreter.jupyter_with_image',
}
_TOOL_SCHEMA_NAMES = {
    'search_stateful/search': 'search',
    'search_stateful/visit': 'visit',
    'jupyter_with_image': 'python',
}


@dataclass(frozen=True)
class DatasetSpec:
    """Everything that distinguishes one benchmark from another."""

    name: str
    label: str
    env_prefix: str
    default_dataset: str
    system_prompt: str
    build_question: Callable[[dict[str, Any], 'DatasetSpec'], str]
    tools: tuple[str, ...] = ('jupyter_with_image',)
    default_limit: int = 20
    default_max_iterations: int = 100
    default_model: str = 'qwen3.8-max'
    default_provider: str = 'ali'
    default_chat_model: str = 'openai/api_taiji_hy3'
    default_reasoning_effort: str = 'low'
    normalize_answer: Callable[[Any], str] = lambda value: str(value).strip()
    extra_metadata: Callable[[dict[str, Any]], dict[str, Any]] = field(default=lambda row: {})

    def env(self, suffix: str) -> str:
        return f'{self.env_prefix}_{suffix}'


def choice_env(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip()
    if value not in choices:
        allowed = ', '.join(sorted(choices))
        raise ValueError(f'{name} must be one of: {allowed}')
    return value


def int_env(
    name: str,
    *,
    default: int | None = None,
    minimum: int = 1,
) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f'{name} must be an integer') from exc
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def _pixel_bounds(spec: DatasetSpec) -> tuple[int | None, int | None]:
    min_pixels = int_env(spec.env('MIN_PIXELS'))
    max_pixels = int_env(spec.env('MAX_PIXELS'))
    if min_pixels is not None and max_pixels is not None and min_pixels > max_pixels:
        raise ValueError(f'{spec.env("MIN_PIXELS")} must not exceed {spec.env("MAX_PIXELS")}')
    return min_pixels, max_pixels


def _smoke_args(spec: DatasetSpec) -> SimpleNamespace:
    return SimpleNamespace(
        client=choice_env(spec.env('CLIENT'), 'responses', {'chat', 'responses'}),
        runtime=choice_env(spec.env('RUNTIME'), 'local', {'gongfeng', 'local'}),
        force_tool='none',
        provider=os.environ.get(spec.env('PROVIDER'), spec.default_provider),
        litellm_provider=os.environ.get(spec.env('LITELLM_PROVIDER'), 'openai'),
        model=os.environ.get(spec.env('MODEL'), spec.default_model),
        chat_model=os.environ.get(spec.env('CHAT_MODEL'), spec.default_chat_model),
        api_base=os.environ.get(spec.env('API_BASE')) or None,
        reasoning_effort=os.environ.get(spec.env('REASONING_EFFORT'), spec.default_reasoning_effort),
        model_timeout=int_env(spec.env('MODEL_TIMEOUT'), default=DEFAULT_TIMEOUT),
        timeout=int_env(spec.env('TIMEOUT'), default=DEFAULT_TIMEOUT),
        runtime_image=os.environ.get(spec.env('RUNTIME_IMAGE'), DEFAULT_RUNTIME_IMAGE),
        image=None,
        # The generators hand the prediction to their own judge and rebuild
        # task_config from the dataset row, so the smoke evaluator stays off.
        evaluator_type='none',
        reward_router_id='',
        reward_exp_id='',
        reward_strip_multimodal='none',
        dry_run=False,
    )


def last_user_content(row: dict[str, Any], spec: DatasetSpec) -> list[dict[str, Any]]:
    messages = row.get('messages')
    if not isinstance(messages, list):
        raise ValueError(f'{spec.label} row must contain a messages list')

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        content = message.get('content')
        if not isinstance(content, list):
            raise ValueError(f'{spec.label} user message must contain a content list')
        return [item for item in content if isinstance(item, dict)]
    raise ValueError(f'{spec.label} row must contain a user message')


def text_item(row: dict[str, Any], spec: DatasetSpec) -> dict[str, Any]:
    text_items = [item for item in last_user_content(row, spec) if item.get('type') == 'text']
    if len(text_items) != 1:
        raise ValueError(f'{spec.label} user message must contain exactly one text item')
    value = text_items[0].get('value')
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{spec.label} text item must contain a non-empty value')
    return text_items[0]


def _image_source(image_item: dict[str, Any], spec: DatasetSpec) -> str:
    for key in ('cos_object_url', 'public_cos_url', 'cos_url', 'value'):
        source = image_item.get(key)
        if not isinstance(source, str) or not source.strip():
            continue
        source = source.strip()
        if source.startswith('https://'):
            return source
        if source.startswith('data:image/') and ';base64,' in source and '{cos_url}' not in source:
            return source
        # Datasets that ship figures as filesystem paths are staged by the task,
        # which reads the bytes and fails loudly when the path is missing. Only
        # absolute paths qualify, so an unresolved template or a stray label in
        # one of the candidate keys is not mistaken for a file.
        if source.startswith('/'):
            return source
    raise ValueError(f'{spec.label} image item must contain an HTTPS URL, Base64 data URI, or absolute file path')


def image_sources(row: dict[str, Any], spec: DatasetSpec) -> list[str]:
    image_items = [item for item in last_user_content(row, spec) if item.get('type') == 'image_url']
    if not image_items:
        raise ValueError(f'{spec.label} user message must contain at least one image item')
    return [_image_source(item, spec) for item in image_items]


def answer_type(row: dict[str, Any], spec: DatasetSpec) -> str:
    options = text_item(row, spec).get('options')
    if options is None:
        options = []
    if not isinstance(options, list):
        raise ValueError(f'{spec.label} text options must be a list')
    return 'multipleChoice' if options else 'exactMatch'


def _answer(row: dict[str, Any], spec: DatasetSpec) -> str:
    answers = row.get('answer')
    if not isinstance(answers, (list, tuple)) or len(answers) != 1:
        raise ValueError(f'{spec.label} answer must be a single-element list or tuple')
    answer = spec.normalize_answer(answers[0])
    if not answer:
        raise ValueError(f'{spec.label} answer must not be empty')
    return answer


def _task_config(
    row: dict[str, Any],
    spec: DatasetSpec,
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict[str, Any]:
    answer = _answer(row, spec)
    extra_info = row.get('extra_info')
    if not isinstance(extra_info, dict):
        extra_info = {}
    ground_truth = extra_info.get('ground_truth')
    if ground_truth is None:
        ground_truth = answer

    config: dict[str, Any] = {
        'question': spec.build_question(row, spec),
        'images': image_sources(row, spec),
        'answer': answer,
        'ground_truth': spec.normalize_answer(ground_truth),
        'image_download_timeout': DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
        'max_image_bytes': DEFAULT_MAX_IMAGE_BYTES,
    }
    if min_pixels is not None:
        config['min_pixels'] = min_pixels
    if max_pixels is not None:
        config['max_pixels'] = max_pixels
    return config


def _build_base_run_input(spec: DatasetSpec):
    args = _smoke_args(spec)
    base = build_run_input(args)
    max_iterations = int_env(spec.env('MAX_ITERATIONS')) or spec.default_max_iterations
    inference_timeout = int_env(spec.env('INFERENCE_TIMEOUT')) or DEFAULT_INFERENCE_TIMEOUT
    observation_timeout = int_env(spec.env('OBSERVATION_TIMEOUT')) or DEFAULT_OBSERVATION_TIMEOUT
    base.run_id = None
    base.task_name = 'vlm_stem_agent'
    base.task_config = {}
    base.imports = [
        ('nexus.runtime.providers.local' if args.runtime == 'local' else 'nexus.runtime.providers.gongfeng'),
        'nexus.extensions.openai_clients.litellm',
        'nexus.extensions.tasks.vlm_stem_agent.task',
        'nexus.extensions.agents.main.agent',
        *(_TOOL_IMPORTS[tool] for tool in spec.tools),
    ]
    base.agent_config = {
        **base.agent_config,
        'system_prompt': spec.system_prompt,
        'max_iterations': max_iterations,
        'tools': list(spec.tools),
        'parallel_tool_calls': False,
        'tool_call_error_retries': 1,
        'inference_time_timeout': inference_timeout,
        'observation_time_timeout': observation_timeout,
        'tool_overrides': [{'tool_name': tool, 'name': _TOOL_SCHEMA_NAMES[tool]} for tool in spec.tools],
    }
    code_interpreter = base.tool_configs.get('code_interpreter')
    if code_interpreter is None:
        raise ValueError(f'{spec.label} base config is missing code_interpreter')
    tool_configs: dict[str, Any] = {'code_interpreter': code_interpreter}
    if any(tool in spec.tools for tool in SEARCH_TOOLS):
        tool_configs[SEARCH_TOOL_CONFIG_KEY] = base.tool_configs[SEARCH_TOOL_CONFIG_KEY]
    base.tool_configs = tool_configs
    return base


def _sample_id(row: dict[str, Any], spec: DatasetSpec, index: int) -> str:
    value = row.get('sample_id')
    if value is None:
        value = index + 1
    sample_id = str(value).strip()
    if not sample_id:
        raise ValueError(f'{spec.label} sample_id must not be empty')
    return sample_id


def _metadata(
    row: dict[str, Any],
    spec: DatasetSpec,
    *,
    index: int,
    sample_id: str,
) -> dict[str, Any]:
    extra_info = row.get('extra_info')
    if not isinstance(extra_info, dict):
        extra_info = {}
    return {
        'index': extra_info.get('index', index + 1),
        'sample_id': sample_id,
        'answer_type': answer_type(row, spec),
        **spec.extra_metadata(row),
    }


def generate_configs(spec: DatasetSpec) -> Iterator[BatchItem]:
    """Yield one BatchItem per dataset row, honoring the spec's env overrides."""
    dataset_value = os.environ.get(spec.env('DATASET'), spec.default_dataset).strip()
    if not dataset_value:
        raise ValueError(f'{spec.env("DATASET")} must not be empty')
    dataset = Path(dataset_value)
    limit = int_env(spec.env('LIMIT'), default=spec.default_limit, minimum=0)
    assert limit is not None
    min_pixels, max_pixels = _pixel_bounds(spec)
    base = _build_base_run_input(spec)
    seen_sample_ids: set[str] = set()
    record_index = 0

    with dataset.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if limit > 0 and record_index >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid {spec.label} JSON at {dataset}:{line_number}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{spec.label} row at {dataset}:{line_number} must be an object')

            sample_id = _sample_id(row, spec, record_index)
            if sample_id in seen_sample_ids:
                raise ValueError(f'Duplicate {spec.label} sample_id: {sample_id}')
            seen_sample_ids.add(sample_id)

            config = base.model_copy(deep=True)
            config.task_config = _task_config(
                row,
                spec,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            yield BatchItem(
                config=config,
                run_id=f'{spec.name}_{sample_id}',
                metadata=_metadata(
                    row,
                    spec,
                    index=record_index,
                    sample_id=sample_id,
                ),
            )
            record_index += 1
