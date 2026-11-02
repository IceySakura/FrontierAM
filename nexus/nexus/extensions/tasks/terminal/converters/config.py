"""Shared configuration and utilities for terminal task converters."""

from __future__ import annotations

import json
import os
import tomllib

from pathlib import Path
from typing import Any, TypedDict


class DefaultsType(TypedDict):
    """Type definition for DEFAULTS configuration."""

    task_base_dir: str
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


# 默认配置
DEFAULTS: DefaultsType = {
    'task_base_dir': '/apdcephfs_qy4/share_302593112/wxyeszhou/Agent/harbor-main/tasks',
    'run_oracle': False,
    'agent_name': 'terminus2',
    'max_iterations': 200,
    'openai_client_name': 'litellm',
    'openai_client_main_name': 'litellm',
    'api_base': 'http://stream-server-online-toolcall-10103.turbotke.production.polaris:81/openapi',
    'api_key': 'EMPTY',
    'model': 'claude-opus-4-5-20251101',  # 'openai/A30B_MoE_SFT_64K_v260203_tbench_nocot_opus45-parser', #'claude-opus-4-5-20251101',
    'temperature': None,
    'trace_file_path': 'workspace/trace.jsonl',
    'gongfeng_api_base_url': 'http://hyrl-sandbox.prod.woa.com:8052',
    'gongfeng_runtime_image_tag': 'latest',
    'gongfeng_request_timeout': 24000,
    'gongfeng_cpu': 2,
    'gongfeng_memory': 10,
    'gongfeng_runtime_provider': 'gongfeng',
    'timeout_multiplier': 240.0,  # 1分钟的倍数，用于设置test_timeout、agent_timeout、docker_timeout
}

# 模块目录
MODULE_DIR = Path(__file__).parent


def load_task2image() -> dict[str, str]:
    """加载 task2image 映射。 支持通过环境变量 TASK2IMAGE_FILE 指定自定义路径。"""
    task2image_file = os.getenv('TASK2IMAGE_FILE')
    path = Path(task2image_file) if task2image_file else MODULE_DIR / 'task2image.json'
    with open(path, encoding='utf-8') as f:
        return json.load(f)


TASK2IMAGE = load_task2image()


def get_env(key: str, default: Any = None) -> Any:
    """获取环境变量，支持布尔/整数类型转换。"""
    value = os.getenv(key)
    if value is None:
        return default
    if isinstance(default, bool):
        return value.lower() in ('true', '1', 'yes')
    if isinstance(default, int):
        return int(value)
    return value


def parse_task_toml(toml_file: Path | str) -> dict[str, Any]:
    """解析 task.toml，提取 timeout/cpus/memory 等配置。"""
    with open(toml_file, 'rb') as f:
        config = tomllib.load(f)
    return {
        'test_timeout': config.get('verifier', {}).get('timeout_sec', 600),
        'agent_timeout': config.get('agent', {}).get('timeout_sec', 600),
        'docker_timeout': config.get('environment', {}).get('build_timeout_sec', 600),
        'cpus': config.get('environment', {}).get('cpus', '4'),
        'memory': config.get('environment', {}).get('memory', '8g'),
    }


def read_instruction(task_folder: Path) -> str:
    """读取 instruction.md。"""
    instruction_path = task_folder / 'instruction.md'
    with open(instruction_path, encoding='utf-8') as f:
        return f.read().strip()


def get_available_tasks(task_base_dir: Path) -> list[str]:
    """获取所有可用的任务名称（基于 task2image.json）。"""
    available = []
    for task_name in TASK2IMAGE:
        task_folder = task_base_dir / task_name
        if task_folder.exists() and (task_folder / 'task.toml').exists():
            available.append(task_name)
    return sorted(available)
