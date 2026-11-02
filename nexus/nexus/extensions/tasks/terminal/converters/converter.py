#!/usr/bin/env python3
"""
Task YAML 配置文件生成器

将 terminal-bench2 task 转换为 nexus task.yaml 格式。

Usage:
    python converter.py /path/to/task -o output.yaml --run-oracle
"""

from __future__ import annotations

import argparse
import os

from string import Template
from typing import Any

from nexus.extensions.tasks.terminal.converters.config import (
    DEFAULTS,
    MODULE_DIR,
    TASK2IMAGE,
    parse_task_toml,
)


def create_task_yaml(
    task_folder_path: str,
    *,
    run_oracle: bool = DEFAULTS['run_oracle'],
    agent_name: str = DEFAULTS['agent_name'],
    max_iterations: int = DEFAULTS['max_iterations'],
    openai_client_name: str = DEFAULTS['openai_client_name'],
    openai_client_main_name: str = DEFAULTS['openai_client_main_name'],
    api_base: str = DEFAULTS['api_base'],
    api_key: str = DEFAULTS['api_key'],
    model: str = DEFAULTS['model'],
    trace_file_path: str = DEFAULTS['trace_file_path'],
) -> str:
    """根据 task 文件夹和模板生成 task.yaml 配置内容。"""
    task_folder_path = os.path.abspath(task_folder_path)
    task_name = os.path.basename(task_folder_path.rstrip('/'))

    # 加载模板
    with open(MODULE_DIR / 'template.yaml', encoding='utf-8') as f:
        template = Template(f.read())

    # 读取 instruction.md（每行缩进 4 空格用于 YAML 多行格式）
    with open(os.path.join(task_folder_path, 'instruction.md'), encoding='utf-8') as f:
        message = '\n'.join('    ' + line for line in f.read().strip().split('\n'))

    # 读取 task.toml
    toml_config = parse_task_toml(os.path.join(task_folder_path, 'task.toml'))

    # 替换模板变量
    return template.substitute(
        MESSAGE=message,
        TASK_DIR=task_folder_path,
        TEST_TIMEOUT=toml_config['test_timeout'],
        AGENT_TIMEOUT=toml_config['agent_timeout'],
        DOCKER_TIMEOUT=toml_config['docker_timeout'],
        IMAGE_CPU=toml_config['cpus'],
        IMAGE_MEMORY=toml_config['memory'],
        IMAGE=TASK2IMAGE[task_name],
        RUN_ORACLE=str(run_oracle).lower(),
        AGENT_NAME=agent_name,
        MAX_ITERATIONS=max_iterations,
        OPENAI_CLIENT_NAME=openai_client_name,
        OPENAI_CLIENT_MAIN_NAME=openai_client_main_name,
        API_BASE=api_base,
        API_KEY=api_key,
        MODEL=model,
        TRACE_FILE_PATH=trace_file_path,
    )


def save_task_yaml(task_folder_path: str, output_path: str | None = None, **kwargs: Any) -> str:
    """创建并保存 task.yaml 配置文件。"""
    yaml_content = create_task_yaml(task_folder_path, **kwargs)

    if output_path is None:
        task_name = os.path.basename(os.path.abspath(task_folder_path).rstrip('/'))
        output_path = os.path.join(os.path.dirname(os.path.abspath(task_folder_path)), f'{task_name}.yaml')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    return os.path.abspath(output_path)


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description='将 terminal-bench2 task 转换为 nexus task.yaml 格式')
    parser.add_argument('task_folder_path', help='task 文件夹路径')
    parser.add_argument('-o', '--output', dest='output_path', help='输出文件路径')
    # Agent
    parser.add_argument('--run-oracle', action='store_true', help='是否运行 oracle')
    parser.add_argument('--agent-name', default=DEFAULTS['agent_name'])
    parser.add_argument('--max-iterations', type=int, default=DEFAULTS['max_iterations'])
    # OpenAI
    parser.add_argument('--openai-client-name', default=DEFAULTS['openai_client_name'])
    parser.add_argument('--openai-client-main-name', default=DEFAULTS['openai_client_main_name'])
    parser.add_argument('--model', default=DEFAULTS['model'])
    parser.add_argument('--api-base', default=DEFAULTS['api_base'])
    parser.add_argument('--api-key', default=DEFAULTS['api_key'])
    # Processor
    parser.add_argument('--trace-file-path', default=DEFAULTS['trace_file_path'])

    args = parser.parse_args()
    kwargs = {k: getattr(args, k) for k in DEFAULTS if hasattr(args, k)}

    try:
        if args.output_path:
            path = save_task_yaml(args.task_folder_path, args.output_path, **kwargs)
            print(f'✓ 已保存到: {path}')
        else:
            print(create_task_yaml(args.task_folder_path, **kwargs))
    except FileNotFoundError as e:
        print(f'✗ 文件未找到: {e}')
        raise SystemExit(1) from None
    except KeyError as e:
        print(f'✗ task 未在 task2image.json 中找到: {e}')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
