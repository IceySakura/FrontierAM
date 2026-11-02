#!/usr/bin/env python3
"""
SkillsBench Batch Generator

用于 nexus run-batch 的 generator 文件，将 skillsbench 任务批量转换为 BatchItem。

Usage:
    # 运行所有任务
    uv run nexus run-batch -g ./nexus/extensions/tasks/skills/converters/generator.py -d ./batch.db --max-concurrent 5

    # 自定义配置
    TASK_BASE_DIR=/path/to/skillsbench/tasks \\
    OPENAI_API_KEY=sk-xxx \\
    OPENAI_BASE_URL=https://api.openai.com/v1 \\
    OPENAI_MODEL=claude-opus-4-5-20251101 \\
    uv run nexus run-batch -g ./generator.py -d ./batch.db --max-concurrent 5

环境变量：
    TASK_BASE_DIR：          skillsbench tasks 目录（默认：/Users/yuchen/hy_engineering/skillsbench/tasks）
    TASK_NAMES：             逗号分隔的任务名列表（可选，默认运行所有任务）
    N_ATTEMPTS：             每个任务的运行次数（默认：1），run_id 格式为 <task-name>-pass<N>
    RUN_ORACLE：             是否运行 oracle（true/false，默认：false）
    AGENT_NAME：             agent 名称（默认：terminus2）
    MAX_ITERATIONS：         最大迭代次数（默认：200）
    OPENAI_CLIENT_NAME：     OpenAI 客户端名称（默认：litellm）
    OPENAI_CLIENT_MAIN_NAME：OpenAI 客户端主名称（默认：litellm）
    OPENAI_API_KEY：         API 密钥
    OPENAI_BASE_URL：        API 基础 URL
    OPENAI_MODEL：           模型名称
    IMAGE_PREFIX：           镜像前缀（默认：skillsbench），镜像名为 <prefix>/<task-name>:latest
"""

from __future__ import annotations

import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.skills.converters.config import (
    DEFAULTS,
    get_available_tasks,
    get_env,
    get_task_image,
    parse_task_toml,
    read_instruction,
)
from nexus.runner.single import RunInput


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
    image_prefix: str,
    nexus_runtime_image: str,
) -> dict[str, Any]:
    """为单个任务创建 RunInput 配置字典。"""
    toml_config = parse_task_toml(task_folder / 'task.toml')
    message = read_instruction(task_folder)
    image = get_task_image(task_name, image_prefix)

    return {
        'imports': ['nexus.extensions'],
        'task_name': 'skills_task',
        'task_config': {
            'message': message,
            'run_oracle': run_oracle,
            'task_dir': str(task_folder.absolute()),
            'test_timeout': toml_config['test_timeout'],
            'agent_timeout': toml_config['agent_timeout'],
            'docker_timeout': toml_config['docker_timeout'],
            'verifier_env': toml_config.get('verifier_env', {}),
            'solution_env': toml_config.get('solution_env', {}),
        },
        'runtime_provider_name': 'docker',
        'runtime_provider_config': {
            'image': image,
            'runtime_image': nexus_runtime_image,
            'working_dir': '/root',
        },
        # 将 solution_env/verifier_env 中的环境变量注入到容器启动参数中，
        # 使 agent 模式下也能通过 $ENV_VAR 访问（oracle 模式通过 extra_env 传入，不受影响）
        'runtime_configs': {
            'main': {
                'environment': {
                    **toml_config.get('solution_env', {}),
                    **toml_config.get('verifier_env', {}),
                },
            },
        }
        if toml_config.get('solution_env') or toml_config.get('verifier_env')
        else {},
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
            },
        },
    }


# ============ Generator 必需函数 ============


def generate_configs() -> Iterator[BatchItem]:
    """生成 BatchItem 的迭代器，每个 BatchItem 对应一个 skillsbench task。"""
    task_base_dir = Path(get_env('TASK_BASE_DIR', DEFAULTS['task_base_dir']))
    task_names_env = os.getenv('TASK_NAMES')
    n_attempts = get_env('N_ATTEMPTS', DEFAULTS['n_attempts'])
    run_oracle = get_env('RUN_ORACLE', DEFAULTS['run_oracle'])
    agent_name = get_env('AGENT_NAME', DEFAULTS['agent_name'])
    max_iterations = get_env('MAX_ITERATIONS', DEFAULTS['max_iterations'])
    openai_client_name = get_env('OPENAI_CLIENT_NAME', DEFAULTS['openai_client_name'])
    openai_client_main_name = get_env('OPENAI_CLIENT_MAIN_NAME', DEFAULTS['openai_client_main_name'])
    api_base = get_env('OPENAI_BASE_URL', DEFAULTS['api_base'])
    api_key = get_env('OPENAI_API_KEY', DEFAULTS['api_key'])
    model = get_env('OPENAI_MODEL', DEFAULTS['model'])
    image_prefix = get_env('IMAGE_PREFIX', DEFAULTS['image_prefix'])
    nexus_runtime_image = get_env('NEXUS_RUNTIME_IMAGE', DEFAULTS['nexus_runtime_image'])

    if task_names_env:
        task_names = [t.strip() for t in task_names_env.split(',') if t.strip()]
    else:
        task_names = get_available_tasks(task_base_dir)

    for task_name in task_names:
        task_folder = task_base_dir / task_name

        if not task_folder.exists():
            print(f'Skip non-existent task: {task_name}')
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
                image_prefix=image_prefix,
                nexus_runtime_image=nexus_runtime_image,
            )

            for attempt in range(1, n_attempts + 1):
                run_id = task_name if n_attempts == 1 else f'{task_name}-pass{attempt}'
                yield BatchItem(
                    config=RunInput.model_validate(config),
                    run_id=run_id,
                    metadata={
                        'task_name': task_name,
                        'task_dir': str(task_folder.absolute()),
                        'model': model,
                        'agent_name': agent_name,
                        'run_oracle': run_oracle,
                        'image': get_task_image(task_name, image_prefix),
                        'attempt': attempt,
                    },
                )
        except Exception as e:
            print(f'Failed to generate config for {task_name}: {e}')
            continue


def get_total_count() -> int:
    """返回总任务数（用于进度显示）。"""
    task_base_dir = Path(get_env('TASK_BASE_DIR', DEFAULTS['task_base_dir']))
    task_names_env = os.getenv('TASK_NAMES')
    n_attempts = get_env('N_ATTEMPTS', DEFAULTS['n_attempts'])
    if task_names_env:
        return len([t.strip() for t in task_names_env.split(',') if t.strip()]) * n_attempts
    return len(get_available_tasks(task_base_dir)) * n_attempts


def get_batch_metadata() -> dict:
    """返回 Batch 级别的元数据。"""
    return {
        'description': 'SkillsBench batch evaluation',
        'task_base_dir': get_env('TASK_BASE_DIR', DEFAULTS['task_base_dir']),
        'model': get_env('OPENAI_MODEL', DEFAULTS['model']),
        'agent_name': get_env('AGENT_NAME', DEFAULTS['agent_name']),
        'run_oracle': get_env('RUN_ORACLE', DEFAULTS['run_oracle']),
        'image_prefix': get_env('IMAGE_PREFIX', DEFAULTS['image_prefix']),
        'n_attempts': get_env('N_ATTEMPTS', DEFAULTS['n_attempts']),
    }


# ============ 测试入口 ============

if __name__ == '__main__':
    print('=' * 60)
    print('SkillsBench Generator Test')
    print('=' * 60)
    print(f'\nBatch Metadata: {get_batch_metadata()}')
    print(f'Total Tasks: {get_total_count()}')
    print('\nGenerating configs...')
    print('-' * 60)

    for i, item in enumerate(generate_configs()):
        print(f'[{i + 1}] run_id={item.run_id}')
        print(f'    task_dir={item.metadata.get("task_dir")}')
        print(f'    image={item.metadata.get("image")}')
        if i >= 4:
            print('    ... (省略剩余任务)')
            break
    print('-' * 60)
    print('Test complete!')
