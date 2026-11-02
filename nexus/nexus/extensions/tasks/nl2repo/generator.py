"""NL2RepoBench 批量任务调度 Generator。

用于 `nexus run-batch` 批量运行 NL2RepoBench 任务。

Usage:
  uv run nexus run-batch -g nexus/extensions/tasks/nl2repo/generator.py --max-concurrent 5

环境变量:
  - NL2REPO_DATASET_PATH: JSONL 数据集文件路径（必需）
  - NL2REPO_AGENT_NAME: Agent 名称（默认 codeact）
  - NL2REPO_MAX_ITERATIONS: Agent 最大迭代次数（默认 500）
  - NL2REPO_EVAL_TIMEOUT: 评估超时秒数（默认 1800）
  - NL2REPO_INFER_IMAGE: 推理阶段镜像（可选）
  - NL2REPO_SKIP: 跳过前 N 个实例（默认 0）
  - NL2REPO_LIMIT: 最多生成 N 个任务（默认无限制）
  - NL2REPO_INSTANCE_IDS: 逗号分隔的实例 ID 列表（可选，用于筛选）
  - NL2REPO_RUNTIME_PROVIDER_NAME: Runtime Provider 名称（默认 gongfeng）
  - NL2REPO_RUNTIME_BASE_URL: Runtime API 地址
  - NL2REPO_RUNTIME_TAG: Runtime 镜像 tag（默认 current）
  - NL2REPO_RUNTIME_TIMEOUT: Runtime 超时秒数（默认 1800）
  - GONGFENG_RUNTIME_TOKEN: Gongfeng sandbox API token
  - OPENAI_CLIENT_NAME / OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL: OpenAI 客户端配置
"""

from __future__ import annotations

import json
import os
import uuid

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.runner.single import RunInput


def _env_int(name: str, default: int | None = None) -> int | None:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_list(name: str) -> set[str] | None:
    """解析逗号分隔的环境变量为 set。"""
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return set(items) if items else None


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    client_name = os.getenv('OPENAI_CLIENT_NAME', 'litellm')
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        raise ValueError('请设置 OPENAI_API_KEY 环境变量')
    base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    model = os.getenv('OPENAI_MODEL', 'gpt-4.1')
    return {
        'main': {
            'name': client_name,
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': base_url,
                },
                'request_args': {
                    'model': model,
                },
            },
        }
    }


def _iter_instances(dataset_path: str) -> Iterator[dict[str, Any]]:
    """从 JSONL 文件中逐行读取实例。"""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'数据集文件不存在: {dataset_path}')

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def generate_configs() -> Iterator[BatchItem]:
    """生成批量任务配置。"""
    dataset_path = os.getenv('NL2REPO_DATASET_PATH', '')
    if not dataset_path:
        raise ValueError('请设置 NL2REPO_DATASET_PATH 环境变量指向 JSONL 数据集文件')

    # 任务参数
    agent_name = os.getenv('NL2REPO_AGENT_NAME', 'codeact')
    max_iterations = _env_int('NL2REPO_MAX_ITERATIONS', 500)
    eval_timeout = _env_int('NL2REPO_EVAL_TIMEOUT', 1800)
    infer_image = os.getenv('NL2REPO_INFER_IMAGE', '')
    skip = _env_int('NL2REPO_SKIP', 0) or 0
    limit = _env_int('NL2REPO_LIMIT')
    instance_ids_filter = _env_list('NL2REPO_INSTANCE_IDS')

    # Runtime 配置
    runtime_provider_name = os.getenv('NL2REPO_RUNTIME_PROVIDER_NAME', 'gongfeng')
    runtime_base_url = os.getenv('NL2REPO_RUNTIME_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052')
    runtime_tag = os.getenv('NL2REPO_RUNTIME_TAG', 'current').strip() or 'current'
    runtime_timeout = _env_int('NL2REPO_RUNTIME_TIMEOUT', 1800)
    runtime_token = os.getenv('GONGFENG_RUNTIME_TOKEN', '')
    runtime_cpu = _env_int('NL2REPO_RUNTIME_CPU', 2) or 2
    runtime_memory = _env_int('NL2REPO_RUNTIME_MEMORY', 10) or 10

    runtime_image = (
        runtime_tag if runtime_tag == 'current' else f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{runtime_tag}'
    )

    openai_clients = _build_openai_clients()

    skipped = 0
    produced = 0

    for instance in _iter_instances(dataset_path):
        instance_id = instance.get('instance_id', '')
        if not instance_id:
            continue

        # 按 ID 过滤
        if instance_ids_filter is not None and instance_id not in instance_ids_filter:
            continue

        # 跳过
        if skipped < skip:
            skipped += 1
            continue

        # 构建 task_config — 直接透传 JSONL 中的所有字段，再补充运行时配置
        task_config: dict[str, Any] = dict(instance)
        task_config.update({
            'max_iterations': max_iterations,
            'eval_timeout_seconds': eval_timeout,
        })
        if infer_image:
            task_config['infer_image'] = infer_image

        config: dict[str, Any] = {
            'imports': [
                'nexus.extensions.tasks.nl2repo',
                'nexus.extensions.tasks.nl2repo.tests.mock_nl2repo_agent',
                'nexus.extensions.agents.codeact',
            ],
            'task_name': 'nl2repo_task',
            'task_config': task_config,
            'runtime_provider_name': runtime_provider_name,
            'runtime_provider_config': {
                'api_base_url': runtime_base_url,
                'token': runtime_token,
                'runtime_image': runtime_image,
                'request_timeout': runtime_timeout,
                'resource_spec': {
                    'cpu': runtime_cpu,
                    'memory': runtime_memory,
                },
            },
            'runtime_configs': {},
            'agent_name': agent_name,
            'agent_config': {
                'max_iterations': max_iterations,
            },
            'openai_clients': openai_clients,
        }

        run_id = f'{instance_id}-{uuid.uuid4().hex[:8]}'

        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=run_id,
            metadata={
                'instance_id': instance_id,
                'eval_image': instance.get('eval_image', ''),
            },
        )

        produced += 1
        if limit is not None and produced >= limit:
            break


def get_total_count() -> int:
    """返回数据集中的实例总数。"""
    dataset_path = os.getenv('NL2REPO_DATASET_PATH', '')
    if not dataset_path:
        return 0

    skip = _env_int('NL2REPO_SKIP', 0) or 0
    limit = _env_int('NL2REPO_LIMIT')
    instance_ids_filter = _env_list('NL2REPO_INSTANCE_IDS')

    n = 0
    skipped = 0

    for instance in _iter_instances(dataset_path):
        instance_id = instance.get('instance_id', '')
        if not instance_id:
            continue
        if instance_ids_filter is not None and instance_id not in instance_ids_filter:
            continue
        if skipped < skip:
            skipped += 1
            continue
        n += 1
        if limit is not None and n >= limit:
            return limit
    return n


def get_batch_metadata() -> dict[str, Any]:
    """返回批量任务的元数据。"""
    dataset_path = os.getenv('NL2REPO_DATASET_PATH', '')
    instance_ids_filter = _env_list('NL2REPO_INSTANCE_IDS')

    return {
        'task_name': 'nl2repo_task',
        'dataset_path': dataset_path,
        'instance_ids_filter': list(instance_ids_filter) if instance_ids_filter else None,
    }
