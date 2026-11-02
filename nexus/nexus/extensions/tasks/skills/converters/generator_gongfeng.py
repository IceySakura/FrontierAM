#!/usr/bin/env python3
"""
SkillsBench 工蜂沙箱 Batch Generator

用于 nexus run-batch 的 generator 文件，通过工蜂沙箱批量评测 skillsbench 任务。

Usage:
    export GONGFENG_RUNTIME_TOKEN="xxx"
    export OPENAI_BASE_URL="http://..."
    export OPENAI_API_KEY="..."
    export OPENAI_MODEL="claude-opus-4-5-20251101"
    export TASK_BASE_DIR="/path/to/skillsbench/tasks"
    export IMAGE_REGISTRY="mirrors.tencent.com/hunyuan_yanguan/skillsbench"

    uv run nexus run-batch \\
        -d ./workspace/skillsbench_batch.db \\
        -g ./nexus/extensions/tasks/skills/converters/generator_gongfeng.py \\
        --max-concurrent 12

环境变量:
    TASK_BASE_DIR:              skillsbench tasks 目录
    TASK_NAMES:                 逗号分隔的任务名列表（可选）
    RUN_ORACLE:                 是否运行 oracle（true/false，默认: false）
    AGENT_NAME:                 agent 名称（默认: terminus2）
    MAX_ITERATIONS:             最大迭代次数（默认: 200）
    OPENAI_CLIENT_NAME:         OpenAI 客户端名称（默认: litellm）
    OPENAI_CLIENT_MAIN_NAME:    OpenAI 客户端主名称（默认: litellm）
    OPENAI_API_KEY:             API 密钥
    OPENAI_BASE_URL:            API 基础 URL
    OPENAI_MODEL:               模型名称
    GONGFENG_RUNTIME_TOKEN:     工蜂沙箱 token（必填）
    GONGFENG_API_BASE_URL:      工蜂沙箱请求地址
    GONGFENG_RUNTIME_IMAGE_TAG: nexus-runtime 镜像 tag（默认: latest）
    GONGFENG_REQUEST_TIMEOUT:   请求超时时间
    GONGFENG_RUNTIME_PROVIDER:  沙箱集群，gongfeng 或 suanli（默认: gongfeng）
    GONGFENG_CPU:               每个 POD 分配核数（默认: 2）
    GONGFENG_MEMORY:            每个 POD 分配内存 GB（默认: 10）
    IMAGE_REGISTRY:             任务镜像仓库前缀（默认: mirrors.tencent.com/hunyuan_yanguan/skillsbench）
                                镜像名为 <IMAGE_REGISTRY>/<task-name>:latest
    HY_BASE_URL:                HunyuanLLM proxy 基础 URL
    HY_API_KEY:                 HunyuanLLM proxy API 密钥
    HY_MODEL:                   HunyuanLLM proxy 模型名称
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
    parse_task_toml,
    read_instruction,
)
from nexus.runner.single import RunInput


# 默认镜像仓库前缀（需要将 skillsbench 镜像推送到该仓库）
_DEFAULT_IMAGE_REGISTRY = 'mirrors.tencent.com/skill_and_mcp/skillsbench'

CLAUDE_CODE_VERSION = os.getenv('CLAUDE_CODE_VERSION', '2.1.19')
CLAUDE_CODE_BINARY_URL = os.getenv(
    'CLAUDE_CODE_BINARY_URL',
    'https://hy-rl-agent-test-1258344706.cos-internal.ap-guangzhou.tencentcos.cn/cache/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases/2.1.63/linux-x64/claude',
)


def _get_task_image(task_name: str, image_registry: str) -> str:
    return f'{image_registry}/{task_name}:latest'


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
    temperature: float | None,
    reasoning_effort: str | None,
    hy_api_base: str,
    hy_api_key: str,
    hy_model: str,
    gongfeng_api_base_url: str,
    gongfeng_runtime_token: str,
    gongfeng_runtime_image_tag: str,
    gongfeng_request_timeout: int,
    gongfeng_runtime_provider: str,
    gongfeng_cpu: int,
    gongfeng_memory: int,
    image_registry: str,
) -> dict[str, Any]:
    """为单个任务创建 RunInput 配置字典。"""
    toml_config = parse_task_toml(task_folder / 'task.toml')
    message = read_instruction(task_folder)
    image = _get_task_image(task_name, image_registry)

    request_args: dict[str, Any] = {'model': model}
    if temperature is not None:
        request_args['temperature'] = temperature
    if reasoning_effort is not None:
        request_args['reasoning_effort'] = reasoning_effort

    # 根据 task.toml 的 cpus/memory 设置沙箱资源，受沙箱上限约束
    # 沙箱上限：cpu 0-4，memory 0-12 GB
    gongfeng_max_cpu = 4
    gongfeng_max_memory_gb = 12
    task_cpus = toml_config.get('cpus', gongfeng_cpu)
    task_memory_mb = toml_config.get('memory', gongfeng_memory * 1024)
    # task.toml 的 memory 单位是 MB，沙箱的 memory 单位是 GB
    task_memory_gb = max(1, (task_memory_mb + 1023) // 1024)  # 向上取整到 GB
    actual_cpu = min(max(task_cpus, gongfeng_cpu), gongfeng_max_cpu)
    actual_memory = min(max(task_memory_gb, gongfeng_memory), gongfeng_max_memory_gb)

    config = {
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
            'solution_files': toml_config.get('solution_files', []),
            'verifier_files': toml_config.get('verifier_files', []),
        },
        'runtime_provider_name': 'gongfeng',
        'runtime_provider_config': {
            'api_base_url': gongfeng_api_base_url,
            'token': gongfeng_runtime_token,
            'image': image,
            'runtime_image': f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{gongfeng_runtime_image_tag}',
            'request_timeout': gongfeng_request_timeout,
            'runtime_provider': gongfeng_runtime_provider,
            'rpc_protocol': 'socketio',  # 使用 SocketIO RPC，Engine.IO 心跳保活防止长时间运行的命令被中间代理断开
            'resource_spec': {
                'cpu': actual_cpu,
                'memory': actual_memory,
            },
            'working_dir': '/root',
        },
        # 将 solution_env 中的环境变量注入到容器启动参数中，
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
                        'api_key': hy_api_key,
                        'base_url': hy_api_base,
                    },
                    'request_args': {
                        'model': hy_model,
                        'temperature': 0.9,
                        'top_p': 1.0,
                        'top_k': 50,
                        'chat_template_kwargs': {'reasoning_effort': reasoning_effort, 'interleaved_thinking': True},
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

    if not run_oracle:
        agent_timeout = toml_config['agent_timeout']
        config['agent_name'] = 'claude_code'
        config['agent_config'] = {
            'max_iterations': 250,
            # 'model': 'anthropic.claude-sonnet-4-5-20250929-v1:0',
            # 'subagent_model': 'anthropic.claude-sonnet-4-5-20250929-v1:0',
            'version': CLAUDE_CODE_VERSION,
            'binary_url': CLAUDE_CODE_BINARY_URL,
            'completion_retry_on_tool_error': 15,
            'timeout': agent_timeout,
            # 'extra_env': {
            #     "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            #     "ANTHROPIC_MODEL": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            #     "ANTHROPIC_DEFAULT_HAIKU_MODEL": "anthropic.claude-sonnet-4-5-20250929-v1:0"
            # }
            'extra_env': {
                'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
                'GH_TOKEN': os.getenv('GH_TOKEN', ''),
                'HUGGINGFACE_API_TOKEN': os.getenv('HUGGINGFACE_API_TOKEN', ''),
                'OPENAI_BASE_URL': os.getenv('OPENAI_BASE_URL', ''),
                'MODAL_TOKEN_ID': os.getenv('MODAL_TOKEN_ID', ''),
                'MODAL_TOKEN_SECRET': os.getenv('MODAL_TOKEN_SECRET', ''),
                'OPENAI_WHISPER_BASE_URL': os.getenv('OPENAI_WHISPER_BASE_URL', ''),
                'OPENAI_WHISPER_API_KEY': os.getenv('OPENAI_WHISPER_API_KEY', ''),
                'CLAUDE_CODE_ATTRIBUTION_HEADER': os.getenv('CLAUDE_CODE_ATTRIBUTION_HEADER', '0'),
                'ANTHROPIC_MODEL': os.getenv('ANTHROPIC_MODEL', ''),
                'ANTHROPIC_DEFAULT_HAIKU_MODEL': os.getenv('ANTHROPIC_DEFAULT_HAIKU_MODEL', ''),
                'CLAUDE_CODE_SUBAGENT_MODEL': os.getenv('CLAUDE_CODE_SUBAGENT_MODEL', ''),
                'GONGFENG_RUNTIME_IMAGE_TAG': os.getenv(
                    'GONGFENG_RUNTIME_IMAGE_TAG', DEFAULTS['gongfeng_runtime_image_tag']
                ),
            },
        }

    return config


# ============ Generator 必需函数 ============


def generate_configs() -> Iterator[BatchItem]:
    """生成 BatchItem 的迭代器，每个 BatchItem 对应一个 skillsbench task。"""
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
    temperature = get_env('TEMPERATURE', DEFAULTS['temperature'])
    reasoning_effort = os.getenv('REASONING_EFFORT')  # 不设默认值，None 表示不传该参数
    hy_api_base = get_env(
        'HY_BASE_URL', 'http://stream-server-online-toolcall-10103.turbotke.production.polaris:1081/openapi'
    )
    hy_api_key = get_env('HY_API_KEY')
    hy_model = get_env('HY_MODEL', 'openai/HY3.0_A20B_sftexp7_bus0331_from_pre20T_mid256k_2.38T_HF.gz')
    gongfeng_api_base_url = get_env('GONGFENG_API_BASE_URL', DEFAULTS['gongfeng_api_base_url'])
    gongfeng_runtime_token = os.getenv('GONGFENG_RUNTIME_TOKEN')
    gongfeng_runtime_image_tag = get_env('GONGFENG_RUNTIME_IMAGE_TAG', DEFAULTS['gongfeng_runtime_image_tag'])
    gongfeng_request_timeout = get_env('GONGFENG_REQUEST_TIMEOUT', DEFAULTS['gongfeng_request_timeout'])
    gongfeng_runtime_provider = get_env('GONGFENG_RUNTIME_PROVIDER', DEFAULTS['gongfeng_runtime_provider'])
    gongfeng_cpu = get_env('GONGFENG_CPU', DEFAULTS['gongfeng_cpu'])
    gongfeng_memory = get_env('GONGFENG_MEMORY', DEFAULTS['gongfeng_memory'])
    image_registry = get_env('IMAGE_REGISTRY', _DEFAULT_IMAGE_REGISTRY)

    if not gongfeng_runtime_token:
        raise ValueError('GONGFENG_RUNTIME_TOKEN environment variable is required but not set.')

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
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                hy_api_base=hy_api_base,
                hy_api_key=hy_api_key,
                hy_model=hy_model,
                gongfeng_api_base_url=gongfeng_api_base_url,
                gongfeng_runtime_token=gongfeng_runtime_token,
                gongfeng_runtime_image_tag=gongfeng_runtime_image_tag,
                gongfeng_request_timeout=gongfeng_request_timeout,
                gongfeng_runtime_provider=gongfeng_runtime_provider,
                gongfeng_cpu=gongfeng_cpu,
                gongfeng_memory=gongfeng_memory,
                image_registry=image_registry,
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
                    'image': _get_task_image(task_name, image_registry),
                },
            )
        except Exception as e:
            print(f'Failed to generate config for {task_name}: {e}')
            continue


def get_total_count() -> int:
    task_base_dir = Path(get_env('TASK_BASE_DIR', DEFAULTS['task_base_dir']))
    task_names_env = os.getenv('TASK_NAMES')
    if task_names_env:
        return len([t.strip() for t in task_names_env.split(',') if t.strip()])
    return len(get_available_tasks(task_base_dir))


def get_batch_metadata() -> dict:
    return {
        'description': 'SkillsBench batch evaluation (gongfeng)',
        'task_base_dir': get_env('TASK_BASE_DIR', DEFAULTS['task_base_dir']),
        'model': get_env('OPENAI_MODEL', DEFAULTS['model']),
        'agent_name': get_env('AGENT_NAME', DEFAULTS['agent_name']),
        'run_oracle': get_env('RUN_ORACLE', DEFAULTS['run_oracle']),
        'image_registry': get_env('IMAGE_REGISTRY', _DEFAULT_IMAGE_REGISTRY),
    }


# ============ 测试入口 ============

if __name__ == '__main__':
    print('=' * 60)
    print('SkillsBench Generator (Gongfeng) Test')
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
