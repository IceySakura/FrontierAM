#!/usr/bin/env python3
"""HyWay SWE generator for cb-insiders on GPT-6.1 Sol at reasoning effort high.

The CLI effort and the Responses ``reasoning.effort`` are both fixed to
``high``. The AIHub credential must keep ``provider=aws_third`` and
``account_id2``; this generator does not strip that suffix. Secrets stay in
the environment, which HyWay fills from plan credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from nexus import BatchItem
from nexus.runner.single import RunInput


AGENT_NAME = 'cb-insiders'
AGENT_IMPORT = 'nexus.extensions.agents.cb_insiders'
MODEL_NAME = 'us.openai.gpt-6.1-sol'
EFFORT = 'high'
DEFAULT_VERSION = '5.0.0-40236426.983c1e09'
DEFAULT_RUNTIME_BASE_URL = 'http://hyrl-sandbox.prod.woa.com:8052'
DEFAULT_AIHUB_ENDPOINT = 'http://api.aihub.woa.com/openai/v1/responses'
_DROP_INSTANCE_FIELDS = frozenset({'main_trajectory', 'instance_with_results'})


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


def _model() -> str:
    model = (os.getenv('SWE_MODEL') or MODEL_NAME).strip() or MODEL_NAME
    if model != MODEL_NAME:
        raise ValueError(f'cb-insiders HyWay production requires SWE_MODEL={MODEL_NAME}')
    return model


def _effort() -> str:
    reasoning = (os.getenv('SWE_REASONING_EFFORT') or os.getenv('SWE_THINKING_EFFORT') or EFFORT).strip()
    cli_effort = (os.getenv('CB_INSIDERS_EFFORT') or EFFORT).strip()
    if reasoning != EFFORT or cli_effort != EFFORT:
        raise ValueError('cb-insiders HyWay production requires model and CLI effort high')
    return EFFORT


def _aihub_api_key() -> str:
    api_key = os.environ['OPENAI_API_KEY'].strip()
    credential, _separator, query = api_key.partition('?')
    if not credential or credential in {'EMPTY', 'XXX'}:
        raise ValueError('OPENAI_API_KEY must contain an AIHub credential')
    pairs = parse_qsl(query, keep_blank_values=True)
    routing = dict(pairs)
    for field in ('provider', 'account_id2'):
        if sum(key == field for key, _value in pairs) != 1:
            raise ValueError(f'GPT-6.1 Sol requires exactly one {field} in credentials.OPENAI_API_KEY')
    if routing['provider'] != 'aws_third':
        raise ValueError('GPT-6.1 Sol requires provider=aws_third in credentials.OPENAI_API_KEY')
    if not routing['account_id2'].strip():
        raise ValueError('GPT-6.1 Sol requires a nonempty account_id2 in credentials.OPENAI_API_KEY')
    return api_key


def _npm_credentials() -> tuple[str, str]:
    cb_user = os.getenv('CB_INSIDERS_NPM_USERNAME', '').strip()
    cb_token = os.getenv('CB_INSIDERS_NPM_TOKEN', '').strip()
    wb_user = os.getenv('WORKBUDDY_NPM_USERNAME', '').strip()
    wb_token = os.getenv('WORKBUDDY_NPM_TOKEN', '').strip()
    if cb_user or cb_token:
        if not cb_user or not cb_token:
            raise RuntimeError('CB_INSIDERS_NPM_USERNAME and CB_INSIDERS_NPM_TOKEN must both be set')
        return cb_user, cb_token
    if wb_user and wb_token:
        return wb_user, wb_token
    raise RuntimeError('cb-insiders needs CB_INSIDERS_NPM_USERNAME/CB_INSIDERS_NPM_TOKEN')


def _timeouts() -> tuple[int, int, int]:
    agent = _env_int('SWE_AGENT_TIMEOUT', 12000) or 12000
    runtime = _env_int('SWE_RUNTIME_TIMEOUT', 12900) or 12900
    runner = _env_int('SWE_RUNNER_TIMEOUT', 14400) or 14400
    if not 0 < agent < runtime < runner:
        raise ValueError('Timeouts must satisfy 0 < SWE_AGENT_TIMEOUT < SWE_RUNTIME_TIMEOUT < SWE_RUNNER_TIMEOUT')
    return agent, runtime, runner


def _build_request_args(instance_id: str) -> dict[str, Any]:
    return {
        'model': _model(),
        'max_output_tokens': _env_int('SWE_MAX_TOKENS', 128000) or 128000,
        'parallel_tool_calls': True,
        'prompt_cache_key': f'gpt61sol_{hashlib.md5(instance_id.encode()).hexdigest()}',
        'store': False,
        'include': ['reasoning.encrypted_content'],
        'reasoning': {'effort': _effort()},
    }


def _build_openai_clients(instance_id: str) -> dict[str, Any]:
    request_args = _build_request_args(instance_id)
    return {
        'main': {
            'name': 'aihub_responses',
            'config': {
                'client_args': {
                    'api_key': _aihub_api_key(),
                    'endpoint': os.getenv('SWE_AIHUB_ENDPOINT', '').strip() or DEFAULT_AIHUB_ENDPOINT,
                    'timeout': _env_int('SWE_LLM_TIMEOUT', 900) or 900,
                    'max_retries': _env_int('SWE_LLM_MAX_RETRIES', 8) or 8,
                },
                'request_args': request_args,
                'enforced_request_args': {
                    'model': request_args['model'],
                    'reasoning': {'effort': EFFORT},
                    'include': ['reasoning.encrypted_content'],
                    'store': False,
                    'stream': False,
                },
            },
        }
    }


def _build_runtime_configs() -> dict[str, Any]:
    worker_venv = {
        'modules': [],
        'extra_requirements': ['dataclasses-json>=0.6.7', 'unidiff>=0.7.5'],
    }
    return {
        'main': {'worker_venv': dict(worker_venv), 'use_new_cache_proxy': True},
        'eval': {'worker_venv': dict(worker_venv), 'use_new_cache_proxy': True},
    }


def _build_runtime_provider_config(runtime_timeout: int) -> dict[str, Any]:
    image = os.getenv('SWE_RUNTIME_IMAGE', '').strip() or 'current'
    config: dict[str, Any] = {
        'api_base_url': os.getenv('SWE_RUNTIME_BASE_URL', DEFAULT_RUNTIME_BASE_URL),
        'runtime_image': image,
        'request_timeout': runtime_timeout,
        'resource_spec': {
            'cpu': _env_int('SWE_CPU', 4) or 4,
            'memory': _env_int('SWE_MEMORY', 16) or 16,
        },
        'rpc_protocol': os.getenv('SWE_RPC_PROTOCOL', 'http'),
        'purpose': os.getenv('SWE_RUNTIME_PURPOSE') or os.getenv('GONGFENG_PURPOSE') or 'prod',
    }
    token = os.getenv('GONGFENG_RUNTIME_TOKEN', '').strip() or os.getenv('GONGFENG_TOKEN', '').strip()
    if token:
        config['token'] = token
    return config


def _build_agent_config(agent_timeout: int) -> dict[str, Any]:
    username, token = _npm_credentials()
    config: dict[str, Any] = {
        'working_dir': '/testbed',
        'timeout': agent_timeout,
        'version': (os.getenv('CB_INSIDERS_VERSION') or DEFAULT_VERSION).strip() or DEFAULT_VERSION,
        'npm_registry': os.getenv('SWE_NPM_REGISTRY', 'http://mirrors.tencent.com/npm/'),
        'npm_username': username,
        'npm_token': token,
        'model_id': 'nexus',
        'effort': _effort(),
        'permission_mode': 'bypassPermissions',
        'auto_install_system_deps': _env_bool('CB_INSIDERS_AUTO_INSTALL_SYSTEM_DEPS', True),
        'max_input_tokens': _env_int('CB_INSIDERS_MAX_INPUT_TOKENS', 200000) or 200000,
        'max_output_tokens': _env_int('SWE_MAX_TOKENS', 128000) or 128000,
    }
    pre_agent_script = os.getenv('SWE_PRE_AGENT_SCRIPT', '').strip()
    if pre_agent_script:
        config['pre_start_scripts'] = [{'code': pre_agent_script}]
    return config


def _build_processors(instance: dict[str, Any], experiment: str) -> dict[str, Any]:
    instance_id = str(instance.get('instance_id') or '')
    language = str(instance.get('language') or 'unknown')
    metadata = {
        'instance_id': instance_id,
        'repo': str(instance.get('repo') or ''),
        'agent_name': AGENT_NAME,
        'agent_version': (os.getenv('CB_INSIDERS_VERSION') or DEFAULT_VERSION).strip() or DEFAULT_VERSION,
        'language': language,
        'dataset_type': str(instance.get('dataset_type') or ''),
        'task': 'swe_task',
        'experiment': experiment,
        'model': _model(),
        'thinking_effort': EFFORT,
        'cli_effort': EFFORT,
        'run_uuid': uuid.uuid4().hex,
    }
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
                'session_id': f'{AGENT_NAME}__{instance_id}',
                'trace_name': f'{AGENT_NAME}__{instance_id}',
                'tags': [AGENT_NAME, _model(), f'effort_{EFFORT}', experiment, language],
                'trace_metadata': metadata,
            },
        },
    }


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


def _require_environment() -> None:
    if not os.getenv('OPENAI_API_KEY', '').strip():
        raise RuntimeError('OPENAI_API_KEY is required')
    if not any(os.getenv(name, '').strip() for name in ('GONGFENG_RUNTIME_TOKEN', 'GONGFENG_TOKEN')):
        raise RuntimeError('GONGFENG_RUNTIME_TOKEN or GONGFENG_TOKEN is required')
    if not _env_bool('SWE_DISABLE_LANGFUSE', False):
        for name in ('LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY', 'LANGFUSE_BASE_URL'):
            if not os.getenv(name, '').strip():
                raise RuntimeError(f'{name} is required')
    _model()
    _effort()
    _aihub_api_key()
    _npm_credentials()


def generate_configs() -> Iterator[BatchItem]:
    _require_environment()
    dataset_path = os.environ['SWE_DATASET_PATH']
    skip = _env_int('SWE_SKIP', 0) or 0
    limit = _env_int('SWE_LIMIT')
    selected = _env_list('SWE_INSTANCE_IDS')
    experiment = os.getenv('SWE_EXPERIMENT', 'cb-insiders-gpt61-high').strip() or 'cb-insiders-gpt61-high'
    agent_timeout, runtime_timeout, runner_timeout = _timeouts()
    runtime_configs = _build_runtime_configs()
    provider_config = _build_runtime_provider_config(runtime_timeout)
    provider_name = os.getenv('SWE_RUNTIME_PROVIDER_NAME', 'gongfeng')
    agent_config = _build_agent_config(agent_timeout)
    enable_anti_hack = _env_bool('SWE_ENABLE_ANTI_HACK', True)
    task_config_base = {
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
    for instance in _iter_instances(dataset_path):
        instance_id = str(instance.get('instance_id') or '')
        if not instance_id:
            continue
        if selected is not None and instance_id not in selected:
            continue
        if skipped < skip:
            skipped += 1
            continue
        config = {
            'imports': [
                'nexus.extensions.tasks.swe',
                AGENT_IMPORT,
                'nexus.extensions.openai_clients.aihub_responses',
                'nexus.trace.processors.langfuse_tracker',
            ],
            'task_name': 'swe_task',
            'task_config': {
                **task_config_base,
                'instance_id': instance_id,
                'instance': {key: value for key, value in instance.items() if key not in _DROP_INSTANCE_FIELDS},
            },
            'runtime_provider_name': provider_name,
            'runtime_provider_config': dict(provider_config),
            'runtime_configs': runtime_configs,
            'runner': {'timeout': runner_timeout},
            'processors': _build_processors(instance, experiment),
            'agent_name': AGENT_NAME,
            'agent_config': dict(agent_config),
            'openai_clients': _build_openai_clients(instance_id),
        }
        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=f'{AGENT_NAME}__{instance_id}',
            metadata={
                'instance_id': instance_id,
                'agent_name': AGENT_NAME,
                'model': MODEL_NAME,
                'thinking_effort': EFFORT,
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
    selected = _env_list('SWE_INSTANCE_IDS')
    count = 0
    skipped = 0
    for instance in _iter_instances(dataset_path):
        instance_id = str(instance.get('instance_id') or '')
        if not instance_id:
            continue
        if selected is not None and instance_id not in selected:
            continue
        if skipped < skip:
            skipped += 1
            continue
        count += 1
        if limit is not None and count >= limit:
            return limit
    return count


def get_batch_metadata() -> dict[str, Any]:
    return {
        'task_name': 'swe_task',
        'scaffold': AGENT_NAME,
        'dataset_path': os.getenv('SWE_DATASET_PATH'),
        'model': _model(),
        'thinking_effort': _effort(),
        'experiment': os.getenv('SWE_EXPERIMENT'),
        'llm_endpoint': os.getenv('SWE_AIHUB_ENDPOINT', '').strip() or DEFAULT_AIHUB_ENDPOINT,
    }
