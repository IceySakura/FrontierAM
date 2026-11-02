#!/usr/bin/env python3
"""WebArena-Verified batch generator for `nexus run-batch`.

This generator reads WebArena-Verified task configurations from webarena-verified.json
and yields `BatchItem`s using the verified dataset format with deterministic evaluation.

Design Decision:
  This generator uses the `webarena_verified` task which includes deterministic evaluation
  logic (type-aware comparison, intelligent text parsing, etc.) for accurate evaluation.
  The webarena_verified task provides significant improvements over the original string_match:
  - Type-aware comparison (numbers, dates, durations, etc.)
  - Intelligent text-to-array parsing (comma/newline/arrow separated)
  - JSON structure comparison
  - Proper NOT_FOUND_ERROR handling

Usage:
  uv run nexus run-batch -g ./nexus/extensions/tasks/webarena_verified/generator.py \\
    -d ./outputs/webarena_verified_batch.db --max-concurrent 5

Environment variables:
  - WEBARENA_VERIFIED_DATASET_PATH: path to webarena-verified.json
    (default: ./workspace/webarena-verified.json)
  - WEBARENA_VERIFIED_TASK_IDS: comma-separated list of specific task_ids to run
  - WEBARENA_VERIFIED_SKIP: number of tasks to skip before starting (default: 0)
  - WEBARENA_VERIFIED_LIMIT: max tasks to yield (default: no limit)
  - WEBARENA_VERIFIED_HARD_ONLY: "1"/"true" to only run hard subset (258 tasks)
  - WEBARENA_BATCH_PREFIX: optional prefix for run_id. If not set, auto-generates
    from OPENAI_MODEL + timestamp (e.g., "HY3_0_A20B_20260409_1621") to ensure
    HAR file isolation for different models and multiple runs of the same model.
  - WEBARENA_AGENT_NAME: agent name (default: browser_gym_agent)
  - WEBARENA_RUNTIME_BASE_URL: runtime base URL
  - WEBARENA_RUNTIME_PROVIDER_NAME: docker/gongfeng/... (default: gongfeng)
  - WEBARENA_RUNTIME_TAG: runtime tag to use (default: latest)
  - WEBARENA_RUNTIME_TIMEOUT: runtime timeout in seconds (default: 1200)
  - WEBARENA_WITH_NA_HINT: "1"/"true" to include N/A hint (default: false)
  - WEBARENA_WITH_HOMEPAGE_HINT: "1"/"true" to include homepage hint (default: false)
  - WEBARENA_USE_DETERMINISTIC_EVAL: "1"/"true" to use deterministic eval (default: true)
  - OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL: used to configure OpenAI client

Note: URL placeholders (__GITLAB__, __SHOPPING__, etc.) in dataset are preserved and
substituted at task runtime based on the webarena_urls configuration.
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.runner.single import RunInput


DEFAULT_WEBARENA_VERIFIED_DATASET_PATH = './workspace/webarena-verified-hard.json'

# WebArena-Verified Hard subset task IDs (258 tasks)
HARD_SUBSET_TASK_IDS = {
    0,
    1,
    2,
    7,
    8,
    9,
    11,
    12,
    16,
    17,
    22,
    23,
    24,
    25,
    26,
    28,
    29,
    32,
    33,
    34,
    35,
    36,
    38,
    39,
    40,
    43,
    44,
    45,
    46,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    66,
    67,
    69,
    71,
    72,
    73,
    74,
    76,
    78,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    88,
    89,
    90,
    91,
    94,
    95,
    96,
    97,
    99,
    100,
    101,
    102,
    104,
    105,
    107,
    108,
    109,
    110,
    111,
    113,
    115,
    117,
    119,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    132,
    133,
    138,
    139,
    140,
    141,
    142,
    144,
    145,
    146,
    147,
    148,
    149,
    152,
    155,
    160,
    161,
    163,
    164,
    165,
    168,
    169,
    173,
    175,
    177,
    178,
    180,
    181,
    182,
    185,
    186,
    188,
    189,
    190,
    191,
    193,
    200,
    201,
    203,
    204,
    205,
    207,
    212,
    213,
    214,
    216,
    217,
    218,
    219,
    222,
    225,
    226,
    227,
    228,
    229,
    231,
    232,
    233,
    234,
    235,
    236,
    239,
    242,
    243,
    244,
    247,
    248,
    249,
    250,
    253,
    256,
    258,
    262,
    263,
    264,
    266,
    268,
    270,
    271,
    274,
    275,
    276,
    279,
    282,
    283,
    287,
    288,
    289,
    291,
    294,
    295,
    296,
    299,
    300,
    302,
    303,
    304,
    305,
    306,
    307,
    308,
    311,
    317,
    318,
    320,
    322,
    325,
    327,
    329,
    332,
    335,
    337,
    338,
    340,
    341,
    342,
    346,
    348,
    351,
    356,
    359,
    361,
    365,
    367,
    372,
    373,
    374,
    375,
    379,
    381,
    382,
    384,
    390,
    391,
    393,
    394,
    397,
    403,
    404,
    405,
    408,
    409,
    411,
    414,
    417,
    418,
    420,
    423,
    430,
    432,
    433,
    435,
    436,
    437,
    438,
    439,
    440,
    441,
    442,
    444,
    447,
    448,
    449,
    450,
}


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str) -> int | None:
    v = os.getenv(name)
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _env_list(name: str) -> set[int] | None:
    """Parse comma-separated list of task IDs from env var, return as set for O(1) lookup."""
    v = os.getenv(name)
    if not v:
        return None
    items = []
    for item in v.split(','):
        item = item.strip()
        if item:
            try:
                items.append(int(item))
            except ValueError:
                continue
    return set(items) if items else None


def _parse_json_field(value: Any) -> Any:
    """Parse a field that may be a JSON string or already parsed.

    HuggingFace datasets often store JSON arrays/objects as strings.
    This function handles both cases gracefully.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _convert_eval_format(eval_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert WebArena-Verified eval format to original WebArena eval format.

    WebArena-Verified uses:
        [{"evaluator": "AgentResponseEvaluator", "expected": {...}}, ...]
        [{"evaluator": "NetworkEventEvaluator", "expected": {"url": [...]}}]

    Original WebArena uses:
        {"eval_types": ["string_match", "url_match", ...], "reference_answers": {...}, ...}

    We convert:
        - AgentResponseEvaluator -> string_match (compare agent answer with expected)
        - NetworkEventEvaluator -> url_match (compare current URL with expected)
    """
    eval_types = []
    reference_answers = None
    reference_url = None

    # Store original evaluators for potential post-processing with deterministic eval
    original_evaluators = eval_list

    for evaluator in eval_list:
        evaluator_type = evaluator.get('evaluator', '')
        expected = evaluator.get('expected', {})

        if evaluator_type == 'AgentResponseEvaluator':
            # AgentResponseEvaluator expects the agent to return a specific answer
            # Map to string_match evaluation
            if 'string_match' not in eval_types:
                eval_types.append('string_match')

            # Extract expected answer from various fields
            # Note: original WebArena expects:
            #   - exact_match: single string
            #   - must_include: list of strings (all must be included)
            retrieved = expected.get('retrieved_data')

            if retrieved is not None and isinstance(retrieved, list) and len(retrieved) > 0:
                # For retrieve tasks with actual data, use must_include
                # The agent's answer should contain all these values
                #
                # NOTE: retrieved_data can be:
                #   - list of strings: ["value1", "value2"]
                #   - list of objects: [{"name": "...", "state": "..."}, ...]
                #
                # For objects, we need to convert to JSON strings for string_match eval
                must_include_values = []
                for item in retrieved:
                    if isinstance(item, dict):
                        # Convert dict to JSON string for matching
                        # The agent's response should contain this JSON representation
                        must_include_values.append(json.dumps(item, sort_keys=True))
                    else:
                        # String or other primitive - use as-is
                        must_include_values.append(str(item) if item is not None else '')

                reference_answers = {
                    'must_include': must_include_values,
                }
            elif 'status' in expected:
                # For tasks with status (SUCCESS, NOT_FOUND_ERROR, ACTION_NOT_ALLOWED_ERROR, etc.)
                # or when retrieved_data is null/empty
                # Use empty must_include - actual verification via deterministic eval
                expected.get('status', '')
                reference_answers = {
                    'must_include': [],  # Empty - actual eval via _original_evaluators
                }

        elif evaluator_type == 'NetworkEventEvaluator':
            # NetworkEventEvaluator checks if the browser navigated to expected URLs
            # Map to url_match evaluation
            if 'url_match' not in eval_types:
                eval_types.append('url_match')

            if 'url' in expected:
                urls = expected['url']
                reference_url = (urls[0] if urls else None) if isinstance(urls, list) else urls

    # Build the converted eval dict
    converted: dict[str, Any] = {
        'eval_types': eval_types if eval_types else ['string_match'],
        # Store original evaluators for deterministic evaluation in post-processing
        '_original_evaluators': original_evaluators,
    }

    if reference_answers:
        converted['reference_answers'] = reference_answers

    if reference_url:
        converted['reference_url'] = reference_url

    return converted


def _normalize_task_config(task_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize task config by parsing JSON string fields and converting formats.

    WebArena-Verified dataset from HuggingFace stores some fields as JSON strings:
    - sites: '["shopping_admin"]' -> ["shopping_admin"]
    - start_urls: '["__SHOPPING_ADMIN__"]' -> ["__SHOPPING_ADMIN__"]
    - eval: '[{...}]' -> [{...}]
    - instantiation_dict: '{...}' -> {...}

    Also converts WebArena-Verified format to original WebArena format:
    - start_urls (array) -> start_url (string with ' |AND| ' separator)
    - eval (list of evaluators) -> eval (dict with eval_types)
    """
    normalized = task_config.copy()

    # Fields that need JSON parsing
    json_fields = ['sites', 'start_urls', 'eval', 'instantiation_dict']
    for field in json_fields:
        if field in normalized:
            normalized[field] = _parse_json_field(normalized[field])

    # Convert start_urls (array) to start_url (string) for original webarena task compatibility
    # Original format uses ' |AND| ' as separator for multiple URLs
    if 'start_urls' in normalized and isinstance(normalized['start_urls'], list):
        normalized['start_url'] = ' |AND| '.join(normalized['start_urls'])

    # Convert eval format from WebArena-Verified (list) to original WebArena (dict)
    if 'eval' in normalized and isinstance(normalized['eval'], list):
        normalized['eval'] = _convert_eval_format(normalized['eval'])

    return normalized


def _load_tasks(dataset_path: str) -> list[dict[str, Any]]:
    """Load all task configurations from webarena-verified.json without URL substitution.

    URL placeholders (__GITLAB__, __SHOPPING__, etc.) are preserved as-is and will be
    substituted at task runtime.

    Also normalizes JSON string fields from HuggingFace dataset format.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

    with path.open('r', encoding='utf-8') as f:
        tasks = json.load(f)

    # Normalize each task config
    return [_normalize_task_config(task) for task in tasks]


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    """Build OpenAI client configurations.

    Note: WebArena-Verified uses deterministic evaluation (no LLM-as-judge),
    so only the main model is needed for the agent.
    """
    main_api_key = os.getenv(
        'OPENAI_API_KEY',
        'xxxxxxxxxxxxxxx?provider=aws_third&model=anthropic.claude-sonnet-4-5-20250929-v1:0',
    )
    main_base_url = os.getenv('OPENAI_BASE_URL', 'http://llm-api.model-eval.woa.com/v1')
    main_model = os.getenv('OPENAI_MODEL', 'api_aws_third_anthropic.claude-sonnet-4-5-20250929-v1:0')

    return {
        'main': {
            'name': 'litellm',
            'config': {
                'client_args': {'base_url': main_base_url, 'api_key': main_api_key},
                'request_args': {
                    'model': 'openai/' + main_model,
                    'max_tokens': 32768,
                    'extra_body': {'reasoning_effort': 'high'},
                    'extra_headers': {'anthropic-beta': 'interleaved-thinking-2025-05-14'},
                },
            },
        },
        # No judge model needed - WebArena-Verified uses deterministic evaluation
    }


def generate_configs() -> Iterator[BatchItem]:
    dataset_path = os.getenv('WEBARENA_VERIFIED_DATASET_PATH', DEFAULT_WEBARENA_VERIFIED_DATASET_PATH)
    skip = _env_int('WEBARENA_VERIFIED_SKIP') or 0
    limit = _env_int('WEBARENA_VERIFIED_LIMIT')
    task_ids_filter = _env_list('WEBARENA_VERIFIED_TASK_IDS')
    hard_only = _env_bool('WEBARENA_VERIFIED_HARD_ONLY', default=False)

    # Get model name for HAR directory naming
    main_model = os.getenv('OPENAI_MODEL', 'HY3.0_A20B_sftexp7_bus0331_from_pre20T_mid256k_2.38T_mtp_n')
    # Sanitize model name: replace special chars with underscore, keep it short
    model_short_name = main_model.replace('/', '_').replace(':', '_').replace('.', '_')[:50]

    # Generate batch prefix: model_name + timestamp (e.g., "hy3_20260409_1621")
    # This ensures isolation for: 1) different models, 2) same model multiple runs
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    # Allow override via env var, otherwise auto-generate from model + timestamp
    batch_prefix = os.getenv('WEBARENA_BATCH_PREFIX') or f'{model_short_name}_{timestamp}'

    agent_name = os.getenv('WEBARENA_AGENT_NAME', 'browser_gym_agent')
    runtime_provider_name = os.getenv('WEBARENA_RUNTIME_PROVIDER_NAME', 'gongfeng')
    runtime_base_url = os.getenv('WEBARENA_RUNTIME_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052')
    os.getenv('WEBARENA_RUNTIME_TAG', 'latest')
    runtime_timeout = _env_int('WEBARENA_RUNTIME_TIMEOUT') or 1200
    runtime_cpu = _env_int('WEBARENA_RUNTIME_CPU') or 8
    runtime_memory = _env_int('WEBARENA_RUNTIME_MEMORY') or 16

    with_na_hint = _env_bool('WEBARENA_WITH_NA_HINT', default=False)
    with_homepage_hint = _env_bool('WEBARENA_WITH_HOMEPAGE_HINT', default=False)

    openai_clients = _build_openai_clients()

    # Load all tasks
    all_tasks = _load_tasks(dataset_path)

    skipped = 0
    produced = 0

    for task_config in all_tasks:
        task_id = task_config.get('task_id')
        if task_id is None:
            continue

        # Filter by hard subset if requested
        if hard_only and task_id not in HARD_SUBSET_TASK_IDS:
            continue

        # Filter by specific task_ids if provided
        if task_ids_filter is not None and task_id not in task_ids_filter:
            continue

        # Skip tasks until we reach the skip threshold
        if skipped < skip:
            skipped += 1
            continue

        # Get deterministic eval setting
        use_deterministic_eval = _env_bool('WEBARENA_USE_DETERMINISTIC_EVAL', default=True)

        config: dict[str, Any] = {
            'imports': [
                # Use webarena_verified task for deterministic evaluation
                'nexus.extensions.tasks.webarena_verified',
                'nexus.extensions.agents.browser_gym',
            ],
            # Use webarena_verified task with deterministic evaluation
            'task_name': 'webarena_verified',
            'task_config': {
                'task_id': task_id,
                'task_config': task_config,
                'runtime_name': 'main',
                'timeout': runtime_timeout,
                'with_na_hint': with_na_hint,
                'with_homepage_hint': with_homepage_hint,
                'use_deterministic_eval': use_deterministic_eval,
                'record_har': True,  # Enable HAR recording
                'har_output_dir': 'outputs/har',  # HAR files stored in outputs/har/<run_id>/network.har
            },
            'agent_name': agent_name,
            'agent_config': {
                'max_iterations': int(os.getenv('WEBARENA_MAX_ITERATIONS', '50')),
                'max_context_tokens': _env_int('WEBARENA_MAX_CONTEXT_TOKENS') or 180 * 1024,
                'max_actions_history': _env_int('WEBARENA_MAX_ACTIONS_HISTORY') or 30,
                'min_keep_iterations': _env_int('WEBARENA_MIN_KEEP_ITERATIONS') or 3,
                'dom_compression_threshold': _env_int('WEBARENA_DOM_COMPRESSION_THRESHOLD') or 15000,
                'enable_dom_compression': os.getenv('WEBARENA_ENABLE_DOM_COMPRESSION', 'true').lower() == 'true',
                'enable_delta_observation': os.getenv('WEBARENA_ENABLE_DELTA_OBSERVATION', 'true').lower() == 'true',
            },
            'runtime_provider_name': runtime_provider_name,
            'runtime_provider_config': {
                'api_base_url': runtime_base_url,
                'token': '6d12fc43adaa15b0f377d83fceeda61d',
                # 'runtime_image': f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{runtime_tag}',
                'runtime_image': 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime-webarenaveri:0409v6',
                'request_timeout': runtime_timeout,
                'resource_spec': {
                    'cpu': runtime_cpu,
                    'memory': runtime_memory,
                },
            },
            'runtime_configs': {
                'main': {'environment': {'no_proxy': '.woa.com,.oa.com,localhost,127.0.0.1,21.183.188.17,.polaris'}}
            },
            'openai_clients': openai_clients,
        }

        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=f'{batch_prefix}_{task_id}',  # Use batch_prefix to distinguish different model runs
            metadata={
                'task_id': task_id,
                'sites': task_config.get('sites', []),
                'intent_template_id': task_config.get('intent_template_id'),
                'is_hard': task_id in HARD_SUBSET_TASK_IDS,
                'batch_prefix': batch_prefix,
            },
        )

        produced += 1
        if limit is not None and produced >= limit:
            break


def get_total_count() -> int:
    dataset_path = os.getenv('WEBARENA_VERIFIED_DATASET_PATH', DEFAULT_WEBARENA_VERIFIED_DATASET_PATH)
    skip = _env_int('WEBARENA_VERIFIED_SKIP') or 0
    limit = _env_int('WEBARENA_VERIFIED_LIMIT')
    task_ids_filter = _env_list('WEBARENA_VERIFIED_TASK_IDS')
    hard_only = _env_bool('WEBARENA_VERIFIED_HARD_ONLY', default=False)

    all_tasks = _load_tasks(dataset_path)

    n = 0
    skipped = 0
    for task_config in all_tasks:
        task_id = task_config.get('task_id')

        # Filter by hard subset if requested
        if hard_only and task_id not in HARD_SUBSET_TASK_IDS:
            continue

        # Filter by specific task_ids if provided
        if task_ids_filter is not None and task_id not in task_ids_filter:
            continue

        # Skip tasks until we reach the skip threshold
        if skipped < skip:
            skipped += 1
            continue

        n += 1
        if limit is not None and n >= limit:
            return limit

    return n


def get_batch_metadata() -> dict[str, Any]:
    dataset_path = os.getenv('WEBARENA_VERIFIED_DATASET_PATH', DEFAULT_WEBARENA_VERIFIED_DATASET_PATH)
    task_ids_filter = _env_list('WEBARENA_VERIFIED_TASK_IDS')
    hard_only = _env_bool('WEBARENA_VERIFIED_HARD_ONLY', default=False)
    use_deterministic_eval = _env_bool('WEBARENA_USE_DETERMINISTIC_EVAL', default=True)

    return {
        'task_name': 'webarena_verified',
        'dataset_path': dataset_path,
        'task_ids_filter': list(task_ids_filter) if task_ids_filter else None,
        'hard_only': hard_only,
        'use_deterministic_eval': use_deterministic_eval,
    }


if __name__ == '__main__':
    """When run as a script, output the first task config as YAML."""
    import sys

    try:
        import yaml
    except ImportError:
        print('Error: PyYAML is required. Install with: pip install pyyaml', file=sys.stderr)
        sys.exit(1)

    # Generate first task
    for batch_item in generate_configs():
        # Convert to dict for YAML output
        config_dict = batch_item.config.model_dump(mode='python', exclude_none=True)

        # Output to stdout
        print('# WebArena-Verified Task Configuration (First Task)')
        print('# Generated from:', os.getenv('WEBARENA_VERIFIED_DATASET_PATH', DEFAULT_WEBARENA_VERIFIED_DATASET_PATH))
        print()
        yaml.dump(config_dict, sys.stdout, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Only output first task
        break
