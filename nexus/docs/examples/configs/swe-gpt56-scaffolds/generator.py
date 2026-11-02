#!/usr/bin/env python3
"""SWE-bench generator for codebuddy and workbuddy on gpt-5.6-sol and gpt-6-sol.

Host LLM calls go to AIHub ``/standard/v1/responses``. Chat completions rejects
function tools combined with reasoning_effort for these models. Item ids are
stripped by ``aihub_responses`` so AIHub can route each turn to any Azure
resource. Secrets come from the environment; HyWay injects them from plan
credentials.

``SWE_MODEL=gpt-6-sol`` sends ``reasoning: {effort}`` only (default effort
``xhigh``). gpt-5.6-sol keeps ``summary`` and ``mode``, and defaults effort to
``max``.

Scaffold is ``SWE_AGENT_NAME``, else ``SWE_SCAFFOLD``, else ``SCAFFOLD_NAME``.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.runner.single import RunInput


SUPPORTED_SCAFFOLDS = {'workbuddy', 'codebuddy'}

DEFAULT_RUNTIME_BASE_URL = 'http://hyrl-sandbox.prod.woa.com:8052'
DEFAULT_RUNTIME_IMAGE = 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:feat-cc-support-proxy-convert'
DEFAULT_AIHUB_ENDPOINT = 'http://api.aihub.woa.com/standard/v1/responses'

# Rollout artifacts. The worker does not have the executor's slice file, so the
# instance is inlined; these blobs are not read by swe_task.
_DROP_INSTANCE_FIELDS = frozenset({'main_trajectory', 'instance_with_results'})

_CODEBUDDY_DISALLOWED = (
    '--disallowedTools=WebSearch',
    '--disallowedTools=WebFetch',
    '--disallowedTools=EnterPlanMode',
    '--disallowedTools=ExitPlanMode',
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_list(name: str) -> set[str] | None:
    value = os.getenv(name)
    if not value:
        return None
    items = [item.strip() for item in value.split(',') if item.strip()]
    return set(items) if items else None


def _scaffold() -> str:
    name = (os.getenv('SWE_AGENT_NAME') or os.getenv('SWE_SCAFFOLD') or os.getenv('SCAFFOLD_NAME') or '').strip()
    if name not in SUPPORTED_SCAFFOLDS:
        raise ValueError(f'Unsupported scaffold {name!r}; supported: {sorted(SUPPORTED_SCAFFOLDS)}')
    return name


def _model() -> str:
    return (os.getenv('SWE_MODEL') or 'gpt-5.6-sol').strip() or 'gpt-5.6-sol'


def _is_gpt6_sol(model: str) -> bool:
    return 'gpt-6' in model and model.endswith('sol')


def _effort() -> str:
    explicit = (os.getenv('SWE_REASONING_EFFORT') or os.getenv('SWE_THINKING_EFFORT') or '').strip()
    if explicit:
        return explicit
    if _is_gpt6_sol(_model()):
        return 'xhigh'
    return 'max'


def _iter_instances(dataset_path: str) -> Iterator[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')
    if path.suffix != '.jsonl':
        raise ValueError(f'Dataset must be .jsonl for batch mode: {dataset_path}')
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_request_args(model: str, effort: str, instance_id: str) -> dict[str, Any]:
    if _is_gpt6_sol(model):
        # gpt-6-sol sends reasoning.effort only. ``xhigh`` is the API value
        # and matches ``wb --effort``.
        reasoning: dict[str, Any] = {'effort': effort}
        cache_prefix = 'gpt6sol'
    else:
        summary = (os.getenv('SWE_REASONING_SUMMARY') or 'detailed').strip() or 'detailed'
        mode = (os.getenv('SWE_REASONING_MODE') or 'standard').strip() or 'standard'
        reasoning = {'effort': effort, 'summary': summary, 'mode': mode}
        cache_prefix = 'gpt56'
    return {
        'model': model,
        'max_output_tokens': _env_int('SWE_MAX_TOKENS', 128000) or 128000,
        'parallel_tool_calls': True,
        'prompt_cache_key': f'{cache_prefix}_{hashlib.md5(instance_id.encode()).hexdigest()}',
        'store': False,
        'include': ['reasoning.encrypted_content'],
        'reasoning': reasoning,
    }


def _build_openai_clients(model: str, effort: str, instance_id: str) -> dict[str, Any]:
    api_key = os.environ['OPENAI_API_KEY'].strip().partition('?')[0].strip()
    return {
        'main': {
            'name': 'aihub_responses',
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'endpoint': os.getenv('SWE_AIHUB_ENDPOINT', DEFAULT_AIHUB_ENDPOINT),
                    'timeout': _env_int('SWE_LLM_TIMEOUT', 900) or 900,
                    'max_retries': _env_int('SWE_LLM_MAX_RETRIES', 8) or 8,
                },
                'request_args': _build_request_args(model, effort, instance_id),
            },
        }
    }


def _build_runtime_configs() -> dict[str, Any]:
    worker_venv = {
        'modules': [],
        'extra_requirements': [
            'dataclasses-json>=0.6.7',
            'unidiff>=0.7.5',
        ],
    }
    return {
        'main': {'worker_venv': dict(worker_venv), 'use_new_cache_proxy': True},
        'eval': {'worker_venv': dict(worker_venv), 'use_new_cache_proxy': True},
    }


def _build_runtime_provider_config(runtime_timeout: int, cpu: int, memory: int) -> dict[str, Any]:
    image = os.getenv('SWE_RUNTIME_IMAGE', '').strip()
    tag = os.getenv('SWE_RUNTIME_TAG', '').strip()
    if not image:
        if tag and tag != 'current':
            image = f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{tag}'
        else:
            image = DEFAULT_RUNTIME_IMAGE
    config: dict[str, Any] = {
        'api_base_url': os.getenv('SWE_RUNTIME_BASE_URL', DEFAULT_RUNTIME_BASE_URL),
        'runtime_image': image,
        'request_timeout': runtime_timeout,
        'resource_spec': {'cpu': cpu, 'memory': memory},
        'rpc_protocol': os.getenv('SWE_RPC_PROTOCOL', 'http'),
        'purpose': os.getenv('SWE_RUNTIME_PURPOSE') or os.getenv('GONGFENG_PURPOSE') or 'prod',
    }
    token = os.getenv('GONGFENG_RUNTIME_TOKEN') or os.getenv('GONGFENG_TOKEN')
    if token:
        config['token'] = token
    return config


def _build_agent_config(scaffold: str, timeout: int, effort: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        'working_dir': '/testbed',
        'timeout': timeout,
        'npm_registry': os.getenv('SWE_NPM_REGISTRY', 'http://mirrors.tencent.com/npm/'),
        'extra_env': {'OPENAI_TIMEOUT': str(timeout)},
    }
    if scaffold == 'workbuddy':
        # gpt-5.6-sol: ``wb --effort xhigh`` and Responses effort ``max`` differ.
        # gpt-6-sol: both knobs are ``xhigh``.
        config['effort'] = os.getenv('WORKBUDDY_EFFORT', 'xhigh')
        config['preset'] = os.getenv('WORKBUDDY_PRESET', 'code')
        # The child worker does not inherit the executor environment, so the
        # registry credentials have to travel inside agent_config. npm wins
        # over a tarball path when both are present.
        npm_user = os.getenv('WORKBUDDY_NPM_USERNAME', '').strip()
        npm_token = os.getenv('WORKBUDDY_NPM_TOKEN', '').strip()
        tarball = os.getenv('WORKBUDDY_TARBALL', '').strip()
        if npm_user and npm_token:
            config['npm_username'] = npm_user
            config['npm_token'] = npm_token
        elif tarball:
            config['tarball_path'] = tarball
        else:
            raise RuntimeError('workbuddy needs WORKBUDDY_NPM_USERNAME/WORKBUDDY_NPM_TOKEN or WORKBUDDY_TARBALL')
        # Unset keeps the agent default (``latest``); pin a version for reproducible batches.
        workbuddy_version = os.getenv('WORKBUDDY_VERSION', '').strip()
        if workbuddy_version:
            config['version'] = workbuddy_version
        return config

    version = os.getenv('CODEBUDDY_VERSION', '').strip()
    if version:
        config['version'] = version
    config['extra_cli_args'] = list(_CODEBUDDY_DISALLOWED)
    del effort
    return config


def _build_processors(
    *,
    scaffold: str,
    instance_id: str,
    repo: str,
    language: str,
    dataset_type: str,
    experiment: str,
    model: str,
    effort: str,
) -> dict[str, Any]:
    if _env_bool('SWE_DISABLE_LANGFUSE', False):
        return {'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}}}
    return {
        'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}},
        'langfuse': {
            'name': 'langfuse_tracker',
            'args': {
                'public_key': os.environ['LANGFUSE_PUBLIC_KEY'],
                'secret_key': os.environ['LANGFUSE_SECRET_KEY'],
                'host': os.environ['LANGFUSE_BASE_URL'],
                'session_id': f'{scaffold}__{instance_id}',
                'trace_name': f'{scaffold}__{instance_id}',
                'tags': [scaffold, model, f'effort_{effort}', experiment, language],
                'trace_metadata': {
                    'instance_id': instance_id,
                    'repo': repo,
                    'agent_name': scaffold,
                    'language': language,
                    'dataset_type': dataset_type,
                    'task': 'swe_task',
                    'experiment': experiment,
                    'model': model,
                    'thinking_effort': effort,
                    'run_uuid': uuid.uuid4().hex,
                },
            },
        },
    }


def generate_configs() -> Iterator[BatchItem]:
    scaffold = _scaffold()
    dataset_path = os.environ['SWE_DATASET_PATH']
    skip = _env_int('SWE_SKIP', 0) or 0
    limit = _env_int('SWE_LIMIT')
    instance_ids_filter = _env_list('SWE_INSTANCE_IDS')
    experiment = os.getenv('SWE_EXPERIMENT', f'swe_gpt56_{scaffold}')
    agent_timeout = _env_int('SWE_AGENT_TIMEOUT', 12000) or 12000
    runtime_timeout = _env_int('SWE_RUNTIME_TIMEOUT', agent_timeout + 900) or (agent_timeout + 900)
    cpu = _env_int('SWE_CPU', 4) or 4
    memory = _env_int('SWE_MEMORY', 16) or 16
    model = _model()
    effort = _effort()

    for required in ('OPENAI_API_KEY', 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY', 'LANGFUSE_BASE_URL'):
        if _env_bool('SWE_DISABLE_LANGFUSE', False) and required.startswith('LANGFUSE_'):
            continue
        if not os.getenv(required):
            raise RuntimeError(f'{required} is required')
    if not (os.getenv('GONGFENG_RUNTIME_TOKEN') or os.getenv('GONGFENG_TOKEN')):
        raise RuntimeError('GONGFENG_TOKEN is required')

    runtime_configs = _build_runtime_configs()
    provider_config = _build_runtime_provider_config(runtime_timeout, cpu, memory)
    agent_config = _build_agent_config(scaffold, agent_timeout, effort)
    enable_anti_hack = _env_bool('SWE_ENABLE_ANTI_HACK', True)
    task_config_base = {
        # Workers cannot see the executor slice path. The instance is inlined below.
        'dataset_path': '',
        'run_oracle': False,
        'raise_on_runtime_error': False,
        'enable_anti_hack_prompt': _env_bool('SWE_ENABLE_ANTI_HACK_PROMPT', True),
        'enable_reward_hacking_detection': _env_bool('SWE_ENABLE_REWARD_HACKING_DETECTION', True),
        'enable_anti_hack': enable_anti_hack,
        'hack_reward_override': _env_float('SWE_HACK_REWARD_OVERRIDE', 0.0) if enable_anti_hack else None,
    }

    skipped = 0
    produced = 0
    for inst in _iter_instances(dataset_path):
        instance_id = str(inst.get('instance_id') or '')
        if not instance_id:
            continue
        if instance_ids_filter is not None and instance_id not in instance_ids_filter:
            continue
        if skipped < skip:
            skipped += 1
            continue

        config: dict[str, Any] = {
            'imports': [
                'nexus.extensions.tasks.swe',
                f'nexus.extensions.agents.{scaffold}',
                'nexus.extensions.openai_clients.aihub_responses',
                'nexus.trace.processors.langfuse_tracker',
            ],
            'task_name': 'swe_task',
            'task_config': {
                **task_config_base,
                'instance_id': instance_id,
                'instance': {key: value for key, value in inst.items() if key not in _DROP_INSTANCE_FIELDS},
            },
            'runtime_provider_name': os.getenv('SWE_RUNTIME_PROVIDER_NAME', 'gongfeng'),
            'runtime_provider_config': dict(provider_config),
            'runtime_configs': runtime_configs,
            'processors': _build_processors(
                scaffold=scaffold,
                instance_id=instance_id,
                repo=str(inst.get('repo') or ''),
                language=str(inst.get('language') or 'unknown'),
                dataset_type=str(inst.get('dataset_type') or 'swe'),
                experiment=experiment,
                model=model,
                effort=effort,
            ),
            'agent_name': scaffold,
            'agent_config': dict(agent_config),
            'openai_clients': _build_openai_clients(model, effort, instance_id),
        }
        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=f'{scaffold}__{instance_id}',
            metadata={
                'instance_id': instance_id,
                'agent_name': scaffold,
                'model': model,
                'experiment': experiment,
            },
        )
        produced += 1
        if limit is not None and produced >= limit:
            break


def get_total_count() -> int:
    dataset_path = os.environ['SWE_DATASET_PATH']
    skip = _env_int('SWE_SKIP', 0) or 0
    limit = _env_int('SWE_LIMIT')
    instance_ids_filter = _env_list('SWE_INSTANCE_IDS')
    n = 0
    skipped = 0
    for inst in _iter_instances(dataset_path):
        instance_id = str(inst.get('instance_id') or '')
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
    return {
        'task_name': 'swe_task',
        'scaffold': os.getenv('SWE_AGENT_NAME') or os.getenv('SWE_SCAFFOLD') or os.getenv('SCAFFOLD_NAME'),
        'dataset_path': os.getenv('SWE_DATASET_PATH'),
        'model': _model(),
        'thinking_effort': _effort(),
        'experiment': os.getenv('SWE_EXPERIMENT'),
        'llm_endpoint': os.getenv('SWE_AIHUB_ENDPOINT', DEFAULT_AIHUB_ENDPOINT),
    }
