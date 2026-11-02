#!/usr/bin/env python3
"""
Terminal-Bench2 Batch Generator

用于 nexus run-batch 的 generator 文件，将 terminal-bench2 任务转换为 BatchItem。

Usage:
    # 运行所有任务
    uv run nexus run-batch -g ./nexus/extensions/tasks/terminal/converters/generator.py -d ./batch.db --max-concurrent 5

    # 使用环境变量自定义配置
    TASK_BASE_DIR=/path/to/tasks \
    OPENAI_API_KEY=sk-xxx \
    OPENAI_BASE_URL=https://api.openai.com/v1 \
    OPENAI_MODEL=gpt-4o \
    uv run nexus run-batch -g ./generator.py -d ./batch.db --max-concurrent 5

环境变量:
    TASK_BASE_DIR: task 文件夹的基目录（默认: /apdcephfs_qy4/share_302593112/wxyeszhou/terminal-bench/tasks）
    TASK_NAMES: 逗号分隔的任务名列表（可选，默认运行所有任务）
    RUN_ORACLE: 是否运行 oracle（true/false，默认: false）
    AGENT_NAME: agent 名称（默认: swe_agent）
    MAX_ITERATIONS: 最大迭代次数（默认: 9999）
    OPENAI_CLIENT_NAME: OpenAI 客户端名称（默认: litellm）
    OPENAI_CLIENT_MAIN_NAME: OpenAI 客户端主名称（默认: litellm
    OPENAI_API_KEY: API 密钥
    OPENAI_BASE_URL: API 基础 URL
    OPENAI_MODEL: 模型名称
    TRACE_FILE_PATH: trace 文件路径（默认: workspace/trace.jsonl）
"""

from __future__ import annotations

import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.terminal.converters.config import (
    DEFAULTS,
    TASK2IMAGE,
    get_available_tasks,
    get_env,
    parse_task_toml,
    read_instruction,
)
from nexus.runner.single import RunInput


def _get_workdir(task_name: str) -> str:
    _workdir = {
        'fix-git': '/app/personal-site',
        'prove-plus-comm': '/workspace',
    }
    return _workdir.get(task_name, '/app')


# 特定任务的资源配置覆盖 (这些任务需要更多 CPU/内存)
_TASK_RESOURCE_OVERRIDES: dict[str, dict[str, int]] = {
    'mcmc-sampling-stan': {'cpu': 16, 'memory': 16},
    'mteb-leaderboard': {'cpu': 16, 'memory': 16},
    'sam-cell-seg': {'cpu': 16, 'memory': 16},
    'torch-pipeline-parallelism': {'cpu': 16, 'memory': 16},
    'torch-tensor-parallelism': {'cpu': 16, 'memory': 16},
    'rstan-to-pystan': {'cpu': 16, 'memory': 16},
    'overfull-hbox': {'cpu': 16, 'memory': 16},
    'install-windows-3.11': {'cpu': 16, 'memory': 16},
    'compile-compcert': {'cpu': 16, 'memory': 16},
}


def create_task_config(
    task_folder: Path,
    task_name: str,
    *,
    run_oracle: bool,
    agent_name: str,
    max_iterations: int,
    openai_client_name: str,
    openai_client_main_name: str,
    api_base: str,
    api_key: str,
    model: str,
    trace_file_path: str,
) -> dict[str, Any]:
    """为单个任务创建 RunInput 配置。"""
    # 解析任务配置
    parse_task_toml(task_folder / 'task.toml')
    message = read_instruction(task_folder)
    image = TASK2IMAGE[task_name]

    return {
        'imports': ['nexus.extensions'],
        'task_name': 'terminal_task',
        'task_config': {
            'message': message,
            'run_oracle': run_oracle,
            'task_dir': str(task_folder.absolute()),
            'test_timeout': 60 * 240.0,
            'agent_timeout': 60 * 240.0,
            'docker_timeout': 60 * 240.0,
        },
        'runtime_provider_name': 'docker',
        'runtime_provider_config': {
            'image': image,
            'runtime_image': 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:0.10.16',
            'working_dir': _get_workdir(task_name),
        },
        'agent_name': agent_name,
        'agent_config': {
            'max_iterations': max_iterations,
        },
        'openai_client_name': openai_client_name,
        'openai_clients': {
            'main': {
                'name': openai_client_main_name,
                'config': {
                    'client_args': {
                        'api_key': api_key,
                        'base_url': api_base,
                    },
                    'request_args': {
                        'model': model,
                    },
                },
            },
        },
        'processors': {
            'chat': {
                'name': 'chat_tracker',
            }
            # 'trace': {
            #     'name': 'json_line',
            #     'args': {
            #         'file_path': trace_file_path,
            #     },
            # },
        },
    }


# ============ Generator 必需函数 ============


def generate_configs() -> Iterator[BatchItem]:
    """生成 BatchItem 的迭代器。

    每个 BatchItem 包含：
    - config: RunInput 配置（task、agent、runtime 等）
    - run_id: 任务名称作为 ID
    - metadata: 任务元数据
    """
    # 从环境变量读取配置
    task_base_dir = Path(get_env('TASK_BASE_DIR', DEFAULTS['task_base_dir']))
    task_names_env = os.getenv('TASK_NAMES')
    run_oracle = get_env('RUN_ORACLE', DEFAULTS['run_oracle'])
    agent_name = get_env('AGENT_NAME', DEFAULTS['agent_name'])
    max_iterations = get_env('MAX_ITERATIONS', DEFAULTS['max_iterations'])
    openai_client_name = get_env('OPENAI_CLIENT_NAME', DEFAULTS['openai_client_name'])
    openai_client_main_name = get_env('OPENAI_CLIENT_MAIN_NAME', DEFAULTS['openai_client_main_name'])
    api_base = get_env('OPENAI_BASE_URL', DEFAULTS['api_base'])
    api_key = get_env('OPENAI_API_KEY', DEFAULTS['api_key'])
    model = get_env('OPENAI_MODEL', DEFAULTS['model'])
    trace_file_path = get_env('TRACE_FILE_PATH', DEFAULTS['trace_file_path'])

    # 确定要运行的任务列表
    if task_names_env:
        task_names = [t.strip() for t in task_names_env.split(',') if t.strip()]
    else:
        task_names = get_available_tasks(task_base_dir)

    # 生成每个任务的 BatchItem
    for task_name in task_names:
        task_folder = task_base_dir / task_name

        # 检查任务是否存在
        if not task_folder.exists():
            print(f'⚠ 跳过不存在的任务: {task_name}')
            continue
        if task_name not in TASK2IMAGE:
            print(f'⚠ 跳过未在 task2image.json 中的任务: {task_name}')
            continue

        try:
            config = create_task_config(
                task_folder=task_folder,
                task_name=task_name,
                run_oracle=run_oracle,
                agent_name=agent_name,
                max_iterations=max_iterations,
                openai_client_name=openai_client_name,
                openai_client_main_name=openai_client_main_name,
                api_base=api_base,
                api_key=api_key,
                model=model,
                trace_file_path=trace_file_path,
            )

            yield BatchItem(
                config=RunInput.model_validate(config),
                run_id=task_name,
                metadata={
                    'task_name': task_name,
                    'task_dir': str(task_folder.absolute()),
                    'model': model,
                    'agent_name': agent_name,
                    'run_oracle': run_oracle,
                },
            )
        except Exception as e:
            print(f'✗ 生成任务 {task_name} 配置失败: {e}')
            continue


def get_total_count() -> int:
    """返回总任务数（用于进度显示）。"""
    task_base_dir = Path(get_env('TASK_BASE_DIR', DEFAULTS['task_base_dir']))
    task_names_env = os.getenv('TASK_NAMES')
    if task_names_env:
        return len([t.strip() for t in task_names_env.split(',') if t.strip()])
    return len(get_available_tasks(task_base_dir))


def get_batch_metadata() -> dict:
    """返回 Batch 级别的元数据。"""
    return {
        'description': 'Terminal Bench2 batch evaluation',
        'task_base_dir': get_env('TASK_BASE_DIR', DEFAULTS['task_base_dir']),
        'model': get_env('OPENAI_MODEL', DEFAULTS['model']),
        'agent_name': get_env('AGENT_NAME', DEFAULTS['agent_name']),
        'run_oracle': get_env('RUN_ORACLE', DEFAULTS['run_oracle']),
    }


# ============ 测试入口 ============

if __name__ == '__main__':
    """测试 generator 输出。"""
    print('=' * 60)
    print('Terminal Bench2 Generator Test')
    print('=' * 60)
    print(f'\nBatch Metadata: {get_batch_metadata()}')
    print(f'Total Tasks: {get_total_count()}')
    print('\nGenerating configs...')
    print('-' * 60)

    for i, item in enumerate(generate_configs()):
        print(f'[{i + 1}] run_id={item.run_id}')
        print(f'    task_dir={item.metadata.get("task_dir", "N/A")}')
        print(f'    image={item.config.runtime_configs.get("main", {}).get("image", "N/A")}')
        if i >= 4:  # 只显示前 5 个
            print('    ... (省略剩余任务)')
            break
    print('-' * 60)
    print('Test complete!')
