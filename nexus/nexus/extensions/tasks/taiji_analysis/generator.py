"""Taiji Analysis batch generator for `nexus run-batch`.

Reads dataset.parquet and yields BatchItems for the taiji_analysis task.

Usage:
    uv run nexus run-batch \
        -g nexus/extensions/tasks/taiji_analysis/generator.py \
        -d ./results/taiji_bench.db \
        --max-concurrent 100

Environment variables:
    - TAIJI_DATASET_PATH: path to Parquet dataset (default: taiji_analysis/episodes/dataset.parquet)
    - TAIJI_SKIP: number of episodes to skip (default: 0)
    - TAIJI_LIMIT: max episodes to yield (default: no limit)
    - TAIJI_INSTANCE_IDS: comma-separated instIds to filter (default: all)
    - TAIJI_OPENAI_API_KEY / TAIJI_OPENAI_BASE_URL / TAIJI_OPENAI_MODEL: Agent LLM config (required)
    - TAIJI_REASONING_EFFORT: Reasoning mode for agent LLM (optional: no_think/low/high)
    - TAIJI_SKIP_JUDGE: set to "1" to skip judge evaluation (useful for GT generation runs)
    - TAIJI_BEDROCK_MODE: set to "1" to use AWS Bedrock transparent proxy with adaptive thinking
    - TAIJI_BEDROCK_BASE_URL: Bedrock proxy endpoint (default: http://trpc-gpt-eval.production.polaris:8080)
    - TAIJI_JUDGE_API_KEY / TAIJI_JUDGE_BASE_URL / TAIJI_JUDGE_MODEL: Judge LLM config (optional)
    - TAIJI_JUDGE_API_STYLE: chat (default) or responses (for GPT-5.6-sol)
    - TAIJI_JUDGE_REASONING_EFFORT: none/low/medium/high/xhigh/max (optional)
    - GONGFENG_RUNTIME_TOKEN: Sandbox API token (required)
    - GF_TOKEN / TAIJI_API_TOKEN / SWANLAB_API_KEY: Skill tokens injected into container
    - MCPORTER_IWIKI_TOKEN / MCPORTER_GONGFENG_TOKEN / MCPORTER_TAPD_TOKEN: MCP Bearer tokens
"""

from __future__ import annotations

import os


os.environ.setdefault('NEXUS_AUTO_INSTALL', '1')

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from nexus import BatchItem
from nexus.runner.single import RunInput


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable '{name}' is not set.")
    return value


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    return int(value) if value else default


def _escape_oc_interpolations(text: str) -> str:
    """Escape ${oc.*} patterns so OmegaConf treats them as literals.

    Non-oc patterns like ${VAR} are handled by the framework's
    escape_non_oc_interpolations; only ${oc.*} needs explicit escaping
    here because the framework intentionally preserves those.
    """
    return text.replace('${oc.', '\\${oc.') if '${oc.' in text else text


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    """Build LLM client config from environment variables."""
    if os.getenv('TAIJI_BEDROCK_MODE') == '1':
        bedrock_model = os.getenv('TAIJI_BEDROCK_MODEL', 'anthropic.claude-opus-4-8')
        bedrock_api_key = f'{_require_env("TAIJI_OPENAI_API_KEY")}?provider=aws_third&timeout=600&model={bedrock_model}'
        bedrock_base_url = os.getenv(
            'TAIJI_BEDROCK_BASE_URL',
            'http://trpc-gpt-eval.production.polaris:8080',
        )
        effort = os.getenv('TAIJI_REASONING_EFFORT', '')
        thinking_kwargs: dict[str, Any] = (
            {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': effort}}
            if effort and effort != 'none'
            else {}
        )
        clients: dict[str, dict[str, Any]] = {
            'main': {
                'name': 'litellm',
                'config': {
                    'client_args': {
                        'api_key': bedrock_api_key,
                        'base_url': bedrock_base_url,
                    },
                    'request_args': {
                        'model': bedrock_model,
                        **thinking_kwargs,
                    },
                },
            },
        }
    else:
        clients = {
            'main': {
                'name': 'litellm',
                'config': {
                    'client_args': {
                        'api_key': _require_env('TAIJI_OPENAI_API_KEY'),
                        'base_url': _require_env('TAIJI_OPENAI_BASE_URL'),
                    },
                    'request_args': {
                        'model': _require_env('TAIJI_OPENAI_MODEL'),
                        **(
                            {
                                'reasoning_effort': os.environ['TAIJI_REASONING_EFFORT'],
                                'allowed_openai_params': ['reasoning_effort'],
                            }
                            if os.getenv('TAIJI_REASONING_EFFORT')
                            else {}
                        ),
                    },
                },
            },
        }

    judge_model = os.getenv('TAIJI_JUDGE_MODEL', os.getenv('TAIJI_OPENAI_MODEL', ''))
    judge_api_style = os.getenv('TAIJI_JUDGE_API_STYLE', 'chat')
    judge_reasoning_effort = os.getenv('TAIJI_JUDGE_REASONING_EFFORT', '')

    if judge_api_style == 'responses':
        bare_model = judge_model.removeprefix('openai/')
        judge_model_final = f'openai/responses/{bare_model}'
        reasoning_kwargs = (
            {'extra_body': {'reasoning': {'effort': judge_reasoning_effort}}} if judge_reasoning_effort else {}
        )
    else:
        judge_model_final = judge_model
        reasoning_kwargs = (
            {'reasoning_effort': judge_reasoning_effort, 'allowed_openai_params': ['reasoning_effort']}
            if judge_reasoning_effort
            else {}
        )

    clients['judge'] = {
        'name': 'litellm',
        'config': {
            'client_args': {
                'api_key': os.getenv('TAIJI_JUDGE_API_KEY', _require_env('TAIJI_OPENAI_API_KEY')),
                'base_url': os.getenv('TAIJI_JUDGE_BASE_URL', _require_env('TAIJI_OPENAI_BASE_URL')),
            },
            'request_args': {
                'model': judge_model_final,
                **reasoning_kwargs,
            },
        },
    }

    return clients


def _bedrock_env() -> dict[str, str]:
    """Build Bedrock-mode env vars for Claude Code in the container.

    Enables the Bedrock transparent proxy so Claude Code can use
    thinking.type=adaptive with output_config.effort for reasoning control.
    TAIJI_REASONING_EFFORT maps to Bedrock effort levels:
      low/medium/high/xhigh/max → output_config.effort
      none / unset → thinking disabled
    """
    env = {
        'CLAUDE_CODE_USE_BEDROCK': '1',
        'AWS_ACCESS_KEY_ID': 'AKIAIOSFODNN7EXAMPLE',
        'AWS_SECRET_ACCESS_KEY': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
        'AWS_SESSION_TOKEN': (
            f'{_require_env("TAIJI_OPENAI_API_KEY")}?provider=aws_third&timeout=600&model=anthropic.claude-opus-4-8'
        ),
        'AWS_REGION': 'us-east-1',
        'ANTHROPIC_BEDROCK_BASE_URL': os.getenv(
            'TAIJI_BEDROCK_BASE_URL',
            'http://trpc-gpt-eval.production.polaris:8080',
        ),
    }
    effort = os.getenv('TAIJI_REASONING_EFFORT', '')
    if effort and effort != 'none':
        env['CLAUDE_CODE_THINKING'] = 'adaptive'
        env['CLAUDE_CODE_THINKING_EFFORT'] = effort
    return env


RUNTIME_IMAGE = os.getenv('TAIJI_RUNTIME_IMAGE', 'current')
CONTAINER_IMAGE = os.getenv('TAIJI_CONTAINER_IMAGE', 'mirrors.tencent.com/lunahyhuang/nexus-runtime:1.4.8-debug-agent')


def generate_configs() -> Iterator[BatchItem]:
    _default_dataset = str(Path(__file__).resolve().parent / 'episodes' / 'dataset.parquet')
    dataset = pd.read_parquet(os.getenv('TAIJI_DATASET_PATH', _default_dataset))
    openai_clients = _build_openai_clients()

    skip = _env_int('TAIJI_SKIP', 0) or 0
    limit = _env_int('TAIJI_LIMIT')
    instance_ids = os.getenv('TAIJI_INSTANCE_IDS', '').split(',') if os.getenv('TAIJI_INSTANCE_IDS') else None

    for idx, (_, row) in enumerate(dataset.iterrows()):
        if idx < skip:
            continue
        if limit and idx >= skip + limit:
            break
        if instance_ids and row['inst_id'] not in instance_ids:
            continue

        config = RunInput.model_validate({
            'imports': ['nexus.extensions.tasks.taiji_analysis'],
            'task_name': 'taiji_analysis',
            'task_config': {
                'question': _escape_oc_interpolations(row['question']),
                'answer': _escape_oc_interpolations(row['answer']),
                'inst_id': row['inst_id'],
                'task_id': row['task_id'],
                'ws_id': row['ws_id'],
                'url': row['url'],
            },
            'agent_name': 'claude_code',
            'agent_config': {
                'working_dir': '/data/workspace',
                'timeout': _env_int('CLAUDE_CODE_TIMEOUT', 7200),
                'allowed_tools': 'Bash,Read',
                'max_turns': 120,
                'extra_env': {
                    'PATH': '/tmp/nexus/agents/claude-code:/nix/runtime/nix-env/bin:/usr/local/bin:/usr/bin:/bin',
                    'TAIJI_API_TOKEN': os.getenv('TAIJI_API_TOKEN', ''),
                    'GF_TOKEN': os.getenv('GF_TOKEN', ''),
                    'SWANLAB_API_KEY': os.getenv('SWANLAB_API_KEY', ''),
                    'MCPORTER_IWIKI_TOKEN': _require_env('MCPORTER_IWIKI_TOKEN'),
                    'MCPORTER_GONGFENG_TOKEN': _require_env('MCPORTER_GONGFENG_TOKEN'),
                    'MCPORTER_TAPD_TOKEN': _require_env('MCPORTER_TAPD_TOKEN'),
                },
                'extra_cli_args': ['--print'],
                'pre_start_scripts': [],
            },
            'openai_clients': openai_clients,
            'runtime_provider_name': 'gongfeng',
            'runtime_provider_config': {
                'api_base_url': os.getenv('TAIJI_RUNTIME_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052'),
                'token': _require_env('GONGFENG_RUNTIME_TOKEN'),
                'image': CONTAINER_IMAGE,
                'runtime_image': RUNTIME_IMAGE,
                'request_timeout': 7800,
                'resource_spec': {'cpu': 2, 'memory': 4},
                'purpose': 'eval',
            },
        })

        yield BatchItem(
            config=config,
            run_id=str(row['instance_id']),
            metadata={
                'inst_id': row['inst_id'],
                'sender': row['sender'],
                'session_id': row.get('session_id', ''),
                'relay_run_id': row.get('relay_run_id', ''),
            },
        )
