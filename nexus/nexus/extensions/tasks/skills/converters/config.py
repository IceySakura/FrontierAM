"""Shared configuration and utilities for SkillsBench task converters."""

from __future__ import annotations

import os
import tomllib

from pathlib import Path
from typing import Any, TypedDict


class DefaultsType(TypedDict):
    task_base_dir: str
    n_attempts: int
    run_oracle: bool
    agent_name: str
    max_iterations: int
    openai_client_name: str
    openai_client_main_name: str
    api_base: str
    api_key: str
    model: str
    temperature: float | None
    trace_file_path: str
    gongfeng_api_base_url: str
    gongfeng_runtime_image_tag: str
    gongfeng_request_timeout: int
    gongfeng_runtime_provider: str
    gongfeng_cpu: int
    gongfeng_memory: int
    timeout_multiplier: float
    image_prefix: str
    nexus_runtime_image: str


DEFAULTS: DefaultsType = {
    'task_base_dir': os.getenv('TASK_BASE_DIR', '/data/home/felixhshi/code/skillsbench/tasks'),
    'n_attempts': 1,
    'run_oracle': False,
    'agent_name': 'terminus2',
    'max_iterations': 200,
    'openai_client_name': 'litellm',
    'openai_client_main_name': 'litellm',
    'api_base': os.getenv('OPENAI_BASE_URL', ''),
    'api_key': os.getenv('OPENAI_API_KEY', 'EMPTY'),
    'model': os.getenv('OPENAI_MODEL', ''),
    'temperature': None,
    'trace_file_path': 'workspace/trace.jsonl',
    'gongfeng_api_base_url': 'http://hyrl-sandbox.prod.woa.com:8052',
    'gongfeng_runtime_image_tag': 'latest',
    'gongfeng_request_timeout': 24000,
    'gongfeng_cpu': 2,
    'gongfeng_memory': 10,
    'gongfeng_runtime_provider': 'gongfeng',
    'timeout_multiplier': 240.0,  # 分钟倍数，用于 test_timeout / agent_timeout / docker_timeout
    # SkillsBench 专属：镜像命名前缀，最终镜像名为 <prefix>/<task-name>:latest
    'image_prefix': 'mirrors.tencent.com/skill_and_mcp/skillsbench',
    # nexus runtime sidecar 镜像
    'nexus_runtime_image': 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:0.10.16',
}

MODULE_DIR = Path(__file__).parent


def get_env(key: str, default: Any = None) -> Any:
    """获取环境变量，支持布尔/整数类型转换。"""
    value = os.getenv(key)
    if value is None:
        return default
    if isinstance(default, bool):
        return value.lower() in ('true', '1', 'yes')
    if isinstance(default, int):
        try:
            return int(value)
        except ValueError:
            return default
    return value


def get_task_image(task_name: str, image_prefix: str) -> str:
    """根据命名约定返回镜像名称。

    SkillsBench 没有统一的 task2image.json，镜像由各 task 的 environment/Dockerfile
    构建而来，命名规则为 <image_prefix>/<task-name>:latest。
    """
    return f'{image_prefix}/{task_name}:latest'


def _expand_env_vars(env_dict: dict[str, str]) -> dict[str, str]:
    """将 env dict 中 ${VAR_NAME} 格式的值替换为实际环境变量值。

    例如: {"OPENAI_API_KEY": "${OPENAI_API_KEY}"} -> {"OPENAI_API_KEY": "sk-xxx..."}
    如果环境变量不存在则保留原值。
    """
    import re

    expanded = {}
    for key, value in env_dict.items():
        if isinstance(value, str):

            def _replacer(m: re.Match) -> str:
                var_name = m.group(1)
                return os.environ.get(var_name, m.group(0))

            expanded[key] = re.sub(r'\$\{(\w+)\}', _replacer, value)
        else:
            expanded[key] = value
    return expanded


def parse_task_toml(toml_file: Path | str) -> dict[str, Any]:
    """解析 task.toml，提取 timeout/cpus/memory 等配置。"""
    try:
        with open(toml_file, 'rb') as f:
            config = tomllib.load(f)
    except OSError as e:
        raise FileNotFoundError(f'无法读取 task.toml: {toml_file}: {e}') from e
    return {
        'test_timeout': config.get('verifier', {}).get('timeout_sec', 600),
        'agent_timeout': config.get('agent', {}).get('timeout_sec', 600),
        'docker_timeout': config.get('environment', {}).get('build_timeout_sec', 600),
        'cpus': config.get('environment', {}).get('cpus', 4),
        'memory': config.get('environment', {}).get('memory_mb', 8192),
        'verifier_env': _expand_env_vars(config.get('verifier', {}).get('env', {})),
        'solution_env': _expand_env_vars(config.get('solution', {}).get('env', {})),
        'solution_files': config.get('solution', {}).get('files', []),
        'verifier_files': config.get('verifier', {}).get('files', []),
    }


def read_instruction(task_folder: Path | str) -> str:
    """读取 instruction.md。"""
    instruction_path = Path(task_folder) / 'instruction.md'
    try:
        with open(instruction_path, encoding='utf-8') as f:
            return f.read().strip()
    except OSError as e:
        raise FileNotFoundError(f'无法读取 instruction.md: {instruction_path}: {e}') from e


def get_available_tasks(task_base_dir: Path) -> list[str]:
    """扫描 task_base_dir，返回所有包含 task.toml 的任务名称列表。"""
    if not task_base_dir.exists():
        raise FileNotFoundError(f'task 目录不存在: {task_base_dir}')
    tasks = []
    for entry in sorted(task_base_dir.iterdir()):
        if entry.is_dir() and (entry / 'task.toml').exists():
            tasks.append(entry.name)
    return tasks
