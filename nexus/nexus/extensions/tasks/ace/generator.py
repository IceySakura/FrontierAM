#!/usr/bin/env python3
"""ACEBench batch generator for `nexus run-batch`.

运行前需要准备官方 ACEBench 源码目录。路径按以下优先级自动发现：
1. 环境变量 `ACE_BENCH_PATH`
2. 本文件同目录下的 `ACEBench/`
3. 仓库根目录下的 `ACEBench-upstream/`

环境变量：
    ACE_BENCH_PATH      官方 ACEBench 源码路径（可选，优先级最高）
    ACE_CATEGORY        normal | special | agent | all（默认 all）
    ACE_LANGUAGE        en | zh（默认 en）
    ACE_TASK_IDS        逗号分隔 task id（可选）
    ACE_SKIP            跳过前 N 条（默认 0）
    ACE_LIMIT           限制条数（可选）
    ACE_AGENT_NAME      Agent name（默认 ace_agent）
    ACE_AGENT_MODEL           Model name（必填）
    ACE_AGENT_MAX_TOKENS      Max tokens（默认 4096）
    ACE_AGENT_TEMPERATURE     Temperature（默认 0.0）
"""

# ============ 导入 ACEBench 官方模块 ============
import importlib
import json
import logging
import os
import sys

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.runner.single import RunInput


# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent.parent  # ace-bench-integration/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


logger = logging.getLogger(__name__)


# ============ 添加 ACEBench 到 Python 路径 ============


def _setup_acebench_path() -> tuple[Path, str | None]:
    """Locate official ACEBench source tree for Nexus integration."""
    candidates = []

    env_path = os.getenv('ACE_BENCH_PATH')
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        Path(__file__).parent / 'ACEBench',
        Path(__file__).resolve().parents[4] / 'ACEBench-upstream',
    ])

    for ace_path in candidates:
        if (ace_path / 'eval_main.py').exists() and (ace_path / 'generate.py').exists():
            if str(ace_path) not in sys.path:
                sys.path.insert(0, str(ace_path))
            logger.debug(f'ACEBench source path: {ace_path}')
            return ace_path, os.getenv('ACE_BENCH_DATASET_PATH')

    searched = ', '.join(str(p) for p in candidates)
    raise RuntimeError(
        f'ACEBench source tree not found. Set ACE_BENCH_PATH or place the official repo at one of: {searched}'
    )


# 初始化
ACE_BENCH_PATH, ACE_BENCH_DATASET_PATH = _setup_acebench_path()

ACE_DATA_CATEGORY = importlib.import_module('category').ACE_DATA_CATEGORY
importlib.import_module('nexus.extensions.tasks.ace.agent')
importlib.import_module('nexus.extensions.tasks.ace.task')


# ============ 数据加载（使用官方格式）============


def _load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load JSONL file (official ACEBench format)"""
    if not file_path.exists():
        logger.warning(f'File not found: {file_path}')
        return []

    data = []
    with open(file_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f'Failed to parse line {line_num} in {file_path}: {e}')

    logger.debug(f'Loaded {len(data)} items from {file_path.name}')
    return data


def _load_all_tasks() -> list[dict[str, Any]]:
    """Load all tasks with ground truth (official ACEBench structure)"""

    # 读取配置
    language = os.getenv('ACE_LANGUAGE', 'en')
    category_arg = os.getenv('ACE_CATEGORY', 'all').lower()

    # 解析分类
    if category_arg == 'all':
        categories = ['normal', 'special', 'agent']
    elif category_arg in ACE_DATA_CATEGORY:
        categories = ACE_DATA_CATEGORY[category_arg]
    else:
        categories = [category_arg]

    # 数据目录（官方结构）
    data_dir = Path(f'{ACE_BENCH_DATASET_PATH}/data_all/data_{language}')
    answer_dir = data_dir / 'possible_answer'

    all_tasks = []

    for category in categories:
        # ACEBench 文件命名: data_{category}*.json
        # 例如: data_normal_atom_bool.json, data_agent_multi_step.json
        pattern = f'data_{category}*.json'
        data_files = sorted(data_dir.glob(pattern))

        if not data_files:
            logger.warning(f"No files found for category '{category}' (pattern: {pattern})")
            continue

        for data_file in data_files:
            # 提取完整的 test_name (e.g., "normal_atom_bool")
            test_name = data_file.stem.replace('data_', '')

            # 加载测试数据
            test_cases = _load_jsonl(data_file)

            # 加载对应的 ground truth
            answer_file = answer_dir / f'data_{test_name}.json'
            ground_truths = _load_jsonl(answer_file)

            # 对齐检查
            if len(test_cases) != len(ground_truths):
                logger.error(
                    f'Data mismatch: {data_file.name} has {len(test_cases)} items, '
                    f'but {answer_file.name} has {len(ground_truths)} items'
                )
                continue

            # 合并数据
            for task_data, gt_data in zip(test_cases, ground_truths, strict=True):
                # 验证 ID 对齐
                task_id = task_data.get('id', '')
                gt_id = gt_data.get('id', '')

                if task_id != gt_id:
                    logger.error(f'ID mismatch: task={task_id}, gt={gt_id}')
                    continue

                task_data['mile_stone'] = gt_data.get('mile_stone', {})
                all_tasks.append({
                    'category': category,
                    'test_name': test_name,
                    'language': language,
                    'task_data': task_data,
                    'api_pool': task_data.get('function', []),
                    'ground_truth': gt_data.get('ground_truth', {}),
                })

    logger.debug(f'Loaded {len(all_tasks)} total tasks across {len(categories)} categories')
    return all_tasks


def _filter_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter tasks by ID, skip, limit"""

    # Filter by IDs
    task_ids_str = os.getenv('ACE_TASK_IDS')
    if task_ids_str:
        id_set = set(task_ids_str.split(','))
        tasks = [t for t in tasks if t['task_data'].get('id', '') in id_set]
        logger.debug(f'Filtered to {len(tasks)} tasks by ACE_TASK_IDS')

    # Skip
    skip = int(os.getenv('ACE_SKIP', '0'))
    if skip > 0:
        tasks = tasks[skip:]
        logger.debug(f'Skipped {skip} tasks')

    # Limit
    limit_str = os.getenv('ACE_LIMIT')
    if limit_str:
        limit = int(limit_str)
        tasks = tasks[:limit]
        logger.debug(f'Limited to {limit} tasks')

    return tasks


# ============ Environment helpers ============


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var (accepts 1/true/yes/y/on, case-insensitive)."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str, default: int | None = None) -> int | None:
    """Parse an integer env var; returns *default* on missing or invalid value."""
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float | None = None) -> float | None:
    """Parse a float env var; returns *default* on missing or invalid value."""
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_list(name: str) -> list[str] | None:
    """Parse a comma-separated list env var (e.g. ``"0,1,5"``)."""
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return items if items else None


def _env_json(name: str, default: dict | None = None) -> dict | None:
    """Parse a JSON-object env var; returns *default* on missing or invalid value."""
    v = os.getenv(name)
    if not v:
        return default
    try:
        parsed = json.loads(v)
        if isinstance(parsed, dict):
            return parsed
        return default
    except (json.JSONDecodeError, TypeError):
        logger.warning(f'Failed to parse JSON from {name}={v!r}')
        return default


# ============ OpenAI Client 配置 ============


def _build_request_args(
    model: str,
    max_tokens: int | None,
    extra_body: dict | None,
    extra_headers: dict | None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    passthrough_args: dict | None = None,
) -> dict[str, Any]:
    """Build ``request_args`` dict for an OpenAI client.

    *passthrough_args* (from ``TAU2_*_REQUEST_ARGS``) serves as the base layer;
    explicit keyword params always override keys in passthrough.
    """
    # Start with passthrough args as the base layer
    args: dict[str, Any] = dict(passthrough_args) if passthrough_args else {}

    # Explicit parameters override passthrough (higher priority)
    args['model'] = model
    if max_tokens is not None:
        args['max_tokens'] = max_tokens
    if extra_body is not None:
        args['extra_body'] = extra_body
    if extra_headers is not None:
        args['extra_headers'] = extra_headers
    if reasoning_effort is not None:
        args['reasoning_effort'] = reasoning_effort
    if temperature is not None:
        args['temperature'] = temperature
    if top_p is not None:
        args['top_p'] = top_p
    return args


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    """Build OpenAI client config from env vars"""
    # =========================
    # Agent (main) client
    # =========================
    api_key = os.getenv('ACE_AGENT_API_KEY', '')
    base_url = os.getenv('ACE_AGENT_BASE_URL', '')
    model = os.getenv('ACE_AGENT_MODEL', '')
    max_tokens = _env_int('ACE_AGENT_MAX_TOKENS')
    extra_body = _env_json('ACE_AGENT_EXTRA_BODY')
    extra_headers = _env_json('ACE_AGENT_EXTRA_HEADERS')
    reasoning_effort = os.getenv('ACE_AGENT_REASONING_EFFORT') or None
    temperature = _env_float('ACE_AGENT_TEMPERATURE')
    top_p = _env_float('ACE_AGENT_TOP_P')

    # =========================
    # User simulator client
    # =========================
    user_api_key = os.getenv('ACE_USER_API_KEY', api_key)
    user_base_url = os.getenv('ACE_USER_BASE_URL', base_url)
    user_model = os.getenv('ACE_USER_MODEL', model)
    user_max_tokens = _env_int('ACE_USER_MAX_TOKENS') or max_tokens
    user_extra_body = _env_json('ACE_USER_EXTRA_BODY') or extra_body
    user_extra_headers = _env_json('ACE_USER_EXTRA_HEADERS') or extra_headers
    user_reasoning_effort = os.getenv('ACE_USER_REASONING_EFFORT') or reasoning_effort
    user_temperature = _env_float('ACE_USER_TEMPERATURE')
    user_top_p = _env_float('ACE_USER_TOP_P')

    clients: dict[str, dict[str, Any]] = {
        'main': {
            'name': 'openai',
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': base_url,
                },
                'request_args': _build_request_args(
                    model, max_tokens, extra_body, extra_headers, reasoning_effort, temperature, top_p
                ),
            },
        },
    }

    clients['user'] = {
        'name': 'openai',
        'config': {
            'client_args': {
                'api_key': user_api_key,
                'base_url': user_base_url,
            },
            'request_args': _build_request_args(
                user_model,
                user_max_tokens,
                user_extra_body,
                user_extra_headers,
                user_reasoning_effort,
                user_temperature,
                user_top_p,
            ),
        },
    }

    return clients


# ============ Generator ============


def _validate_runtime_env() -> tuple[str, str]:
    model_name = os.getenv('ACE_AGENT_MODEL')
    api_key = os.getenv('ACE_AGENT_API_KEY')
    base_url = os.getenv('ACE_AGENT_BASE_URL')
    if not all([model_name, api_key, base_url]):
        raise RuntimeError('Missing required env vars: ACE_AGENT_MODEL, ACE_AGENT_API_KEY, ACE_AGENT_BASE_URL')
    assert model_name is not None
    return model_name, os.getenv('ACE_AGENT_NAME', 'ace_agent')


def get_batch_metadata() -> dict[str, Any]:
    tasks = _filter_tasks(_load_all_tasks())
    return {
        'benchmark': 'ace',
        'task_count': len(tasks),
        'categories': sorted({task['category'] for task in tasks}),
        'language': os.getenv('ACE_LANGUAGE', 'en'),
        'agent_name': os.getenv('ACE_AGENT_NAME', 'ace_agent'),
        'ace_bench_dataset_path': str(ACE_BENCH_DATASET_PATH),
    }


def generate_configs() -> Iterator[BatchItem]:
    """Generate BatchItems for ACEBench evaluation.

    This is the entry point called by nexus run-batch.
    """

    logger.debug('🚀 ACEBench Generator starting...')
    model_name, agent_name = _validate_runtime_env()

    # Load tasks
    tasks = _load_all_tasks()
    tasks = _filter_tasks(tasks)

    if not tasks:
        logger.warning('No tasks to generate')
        return

    # Build config
    openai_clients = _build_openai_clients()

    logger.debug(f'Generating {len(tasks)} BatchItems...')

    for task in tasks:
        task_id = task['task_data']['id']

        # ✅ 将 ground_truth 合并到 task_data 里
        task_data_with_gt = {**task['task_data'], 'ground_truth': task['ground_truth']}

        run_input = RunInput(
            imports=[
                'nexus.extensions.tasks.ace',
            ],
            task_name='ace',
            task_config={
                'category': task['category'],
                'language': task['language'],
                'task_data': task_data_with_gt,  # ✅ 带 ground_truth
                'api_pool': task['api_pool'],
                'max_tokens': int(os.getenv('ACE_AGENT_MAX_TOKENS', '4096')),
                'temperature': float(os.getenv('ACE_AGENT_TEMPERATUREE', '0.0')),
            },
            agent_name=agent_name,
            agent_config={},  # ✅ 空的，参数通过 task 传递
            openai_clients=openai_clients,
        )

        yield BatchItem(
            config=run_input,
            run_id=task_id,
            metadata={
                'category': task['category'],
                'test_name': task['test_name'],
                'language': task['language'],
            },
        )

    logger.debug(f'✅ Generated {len(tasks)} BatchItems')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    print(json.dumps(get_batch_metadata(), ensure_ascii=False, indent=2))
    _validate_runtime_env()
