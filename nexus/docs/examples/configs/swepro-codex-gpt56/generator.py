#!/usr/bin/env python3
"""SWE-Bench Pro + Codex + GPT-5.6-sol for `nexus run-batch`.

Scores the public SWE-Bench Pro set (731) against the official GPT-5.6 mark (~64.6%).

Gateway (AIHub 2.0 standard Responses — aligned with maxluxchen frontend smoke):

  POST http://api.aihub.woa.com/standard/v1/responses
  Authorization: Bearer <OPENAI_API_KEY>   # bare sk-…; no ?provider= on standard/v1
  client_args.cache_task_id = <uuid4>

  Override via SWE_OPENAI_RESPONSES_ENDPOINT. Legacy Polarist needs APP_ID:APP_KEY?provider=…

Inference knobs:
  store=false, include=[reasoning.encrypted_content],
  reasoning.effort=max (override via SWE_REASONING_EFFORT), summary=detailed,
  reasoning.mode=standard (override via SWE_REASONING_MODE / GPT_REASONING_MODE),
  max_output_tokens=128000, timeout=12000, max_retries=32,
  parallel_tool_calls=false, prompt_cache_key=<session uuid4>

Codex: use_azure_provider=False (colleague CODEX_USE_AZURE=0).
  model_context_window=1000000, auto_compact_token_limit=900000
  remote_compaction_v2=false via extra_config_toml
  model_catalog.json: Codex 0.154 gpt-5.6-sol entry, tool_mode=direct
  Codex version is hard-pinned to 0.154.0 (this file does not read
  SWE_CODEX_VERSION / CODEX_VERSION, so leftover env cannot downgrade).

Langfuse: on by default (SWE_DISABLE_LANGFUSE=1 to skip).

Anti-hack (on by default for fair public-mark scoring; same switches as swe_any):
  SWE_ENABLE_ANTI_HACK_PROMPT=1
  SWE_ENABLE_REWARD_HACKING_DETECTION=1
  SWE_ENABLE_ANTI_HACK=1
  SWE_HACK_REWARD_OVERRIDE=0
  Set any to 0/false to disable. Rerun hacked IDs via SWE_INCLUDE_INSTANCE_IDS.

Usage:
  export OPENAI_API_KEY='sk-…'            # never commit the key; env only
  export GONGFENG_TOKEN='...'
  export LANGFUSE_PUBLIC_KEY='...' LANGFUSE_SECRET_KEY='...' LANGFUSE_BASE_URL='...'
  export SWE_DATASET_PATH=/apdcephfs_qy4/share_302593112/rongbao/datasets/swe/swe_bench_pro_fix0319.jsonl
  uv run nexus run-batch \
    -g docs/examples/configs/swepro-codex-gpt56/generator.py \
    -d ./swepro_gpt56.db --max-concurrent 200
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


_AIHUB_RESPONSES_ENDPOINT = 'http://api.aihub.woa.com/standard/v1/responses'
_RESPONSES_ENDPOINT = _AIHUB_RESPONSES_ENDPOINT
_MODEL = os.getenv('SWE_MODEL', 'gpt-5.6-sol').strip() or 'gpt-5.6-sol'
_GATEWAY = 'aihub2.0'
_TIMEOUT_S = 12000.0
_MAX_RETRIES = 32
_MAX_OUTPUT_TOKENS = 128000
_REASONING_EFFORT = os.getenv('SWE_REASONING_EFFORT', 'max').strip() or 'max'
_REASONING_SUMMARY = 'detailed'
_CONTEXT_WINDOW = 1_000_000
_AUTO_COMPACT_TOKEN_LIMIT = 900_000
_EXTRA_CONFIG_TOML = '[features]\nremote_compaction_v2 = false'
_DEFAULT_DATASET_PATH = (
    '/apdcephfs_qy4/share_302593112/rongbao/datasets/swe/swe_bench_pro_fix0319.jsonl'
)
_DEFAULT_LANGFUSE_HOST = 'http://langfuse-taiji-api.woa.com'
_CODEX_VERSION = '0.154.0'
_MODEL_CATALOG: dict[str, Any] = json.loads(
    Path(__file__).with_name('model_catalog.json').read_text(encoding='utf-8')
)


def _reasoning_mode() -> str:
    """GPT-5.6/GPT-6 Responses reasoning.mode. Default standard."""
    return (
        os.getenv('SWE_REASONING_MODE') or os.getenv('GPT_REASONING_MODE') or 'standard'
    ).strip() or 'standard'


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str) -> set[str] | None:
    raw = os.getenv(name)
    if not raw:
        return None
    items = {part.strip() for part in raw.split(',') if part.strip()}
    return items or None


def _api_key_from_env() -> str:
    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY is required (AIHub sk-…; do not commit it)')
    return api_key.partition('?')[0].strip()


def _iter_instance_ids(dataset_path: str) -> Iterator[str]:
    include = _env_list('SWE_INCLUDE_INSTANCE_IDS')
    with open(dataset_path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            instance_id = str(json.loads(line).get('instance_id') or '')
            if not instance_id:
                continue
            if include is not None and instance_id not in include:
                continue
            yield instance_id


def _build_openai_clients(
    *,
    cache_task_id: str,
    prompt_cache_key: str,
) -> dict[str, Any]:
    return {
        'main': {
            'name': 'openai_responses',
            'config': {
                'client_args': {
                    'api_key': _api_key_from_env(),
                    'endpoint': os.getenv('SWE_OPENAI_RESPONSES_ENDPOINT', _RESPONSES_ENDPOINT),
                    'timeout': _TIMEOUT_S,
                    'max_retries': _MAX_RETRIES,
                    'capture_raw_io': True,
                    'cache_task_id': cache_task_id,
                },
                'request_args': {
                    'model': _MODEL,
                    'store': False,
                    'max_output_tokens': _MAX_OUTPUT_TOKENS,
                    'reasoning': {
                        'effort': _REASONING_EFFORT,
                        'summary': _REASONING_SUMMARY,
                        'mode': _reasoning_mode(),
                    },
                    'prompt_cache_key': prompt_cache_key,
                    'include': ['reasoning.encrypted_content'],
                    'parallel_tool_calls': False,
                    'stream': False,
                },
            },
        }
    }


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _build_processors(
    *,
    instance_id: str,
    cache_task_id: str,
) -> dict[str, Any]:
    """chat_tracker + langfuse_tracker (MR !1267 uploads encrypted reasoning_items)."""
    processors: dict[str, Any] = {
        'chat': {
            'name': 'chat_tracker',
            'args': {'tools_mode': 'branch_last'},
        }
    }
    if _env_bool('SWE_DISABLE_LANGFUSE', False):
        return processors

    public_key = (os.getenv('LANGFUSE_PUBLIC_KEY') or '').strip()
    secret_key = (os.getenv('LANGFUSE_SECRET_KEY') or '').strip()
    if not public_key or not secret_key:
        raise RuntimeError(
            'LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required for trajectory '
            'upload (MR !1267). Set SWE_DISABLE_LANGFUSE=1 to skip.'
        )
    host = (
        os.getenv('LANGFUSE_HOST')
        or os.getenv('LANGFUSE_BASE_URL')
        or _DEFAULT_LANGFUSE_HOST
    ).strip()
    experiment = os.getenv('SWE_EXPERIMENT', 'swepro-codex-gpt56')
    processors['langfuse'] = {
        'name': 'langfuse_tracker',
        'args': {
            'public_key': public_key,
            'secret_key': secret_key,
            'host': host,
            'session_id': instance_id,
            'trace_name': instance_id,
            'tags': [
                'codex',
                'gpt-5.6-sol',
                'swe-bench-pro',
                experiment,
            ],
            'trace_metadata': {
                'instance_id': instance_id,
                'agent_name': 'codex',
                'model': _MODEL,
                'gateway': _GATEWAY,
                'cache_task_id': cache_task_id,
                'use_azure_provider': False,
                'codex_version': _CODEX_VERSION,
                'experiment': experiment,
                'reasoning_effort': _REASONING_EFFORT,
                'reasoning_summary': _REASONING_SUMMARY,
                'reasoning_mode': _reasoning_mode(),
                'model_context_window': _CONTEXT_WINDOW,
                'auto_compact_token_limit': _AUTO_COMPACT_TOKEN_LIMIT,
            },
        },
    }
    return processors


def _build_agent_config() -> dict[str, Any]:
    return {
        'working_dir': os.getenv('CLAUDE_CODE_WORKING_DIR', '/testbed'),
        'timeout': float(_env_int('SWE_RUNTIME_TIMEOUT', 14400)),
        'model': _MODEL,
        'use_azure_provider': False,
        'wrap_plaintext_reasoning_content': False,
        'version': _CODEX_VERSION,
        'model_catalog_json': _MODEL_CATALOG,
        'model_context_window': _CONTEXT_WINDOW,
        'auto_compact_token_limit': _AUTO_COMPACT_TOKEN_LIMIT,
        'extra_config_toml': os.getenv('SWE_CODEX_EXTRA_CONFIG_TOML', _EXTRA_CONFIG_TOML),
        'reasoning_effort': _REASONING_EFFORT,
        'extra_cli_args': ['-c', 'web_search="disabled"'],
        'npm_registry': os.getenv('SWE_NPM_REGISTRY', 'http://mirrors.tencent.com/npm/'),
    }


def generate_configs() -> Iterator[BatchItem]:
    dataset_path = os.environ.get('SWE_DATASET_PATH', _DEFAULT_DATASET_PATH)
    limit = _env_int('SWE_LIMIT')  # None => all instances
    token = os.getenv('GONGFENG_RUNTIME_TOKEN') or os.getenv('GONGFENG_TOKEN') or ''
    if not token:
        raise RuntimeError('GONGFENG_TOKEN (or GONGFENG_RUNTIME_TOKEN) is required')
    enable_anti_hack_prompt = _env_bool('SWE_ENABLE_ANTI_HACK_PROMPT', True)
    enable_reward_hacking_detection = _env_bool('SWE_ENABLE_REWARD_HACKING_DETECTION', True)
    enable_anti_hack = _env_bool('SWE_ENABLE_ANTI_HACK', True)
    hack_reward_override = _env_float('SWE_HACK_REWARD_OVERRIDE', 0.0)
    produced = 0
    for instance_id in _iter_instance_ids(dataset_path):
        cache_task_id = str(uuid.uuid4())
        prompt_cache_key = str(uuid.uuid4())
        config: dict[str, Any] = {
            'imports': [
                'nexus.extensions.tasks.swe',
                'nexus.extensions.agents.codex',
                'nexus.extensions.openai_clients.openai_responses',
                'nexus.trace.processors.langfuse_tracker',
            ],
            'task_name': 'swe_task',
            'task_config': {
                'dataset_path': dataset_path,
                'instance_id': instance_id,
                'run_oracle': False,
                'raise_on_runtime_error': True,
                'eval_timeout_seconds': _env_int('SWE_EVAL_TIMEOUT_SECONDS', 2000),
                'enable_anti_hack_prompt': enable_anti_hack_prompt,
                'enable_reward_hacking_detection': enable_reward_hacking_detection,
                'enable_anti_hack': enable_anti_hack,
                'hack_reward_override': hack_reward_override,
            },
            'runtime_provider_name': 'gongfeng',
            'runtime_provider_config': {
                'api_base_url': os.getenv(
                    'SWE_RUNTIME_BASE_URL',
                    'http://hyrl-sandbox.prod.woa.com:8052',
                ),
                'token': token,
                'runtime_image': os.getenv('SWE_RUNTIME_TAG', 'current'),
                'request_timeout': float(_env_int('SWE_RUNTIME_TIMEOUT', 14400)) + 600,
                'resource_spec': {
                    'cpu': _env_int('SWE_RUNTIME_CPU', 4),
                    'memory': _env_int('SWE_RUNTIME_MEMORY', 16),
                },
                'rpc_protocol': 'http',
                'purpose': os.getenv('GONGFENG_PURPOSE', 'prod'),
            },
            'runtime_configs': {
                'main': {'worker_venv': True},
                'eval': {'worker_venv': True},
            },
            'processors': _build_processors(
                instance_id=instance_id,
                cache_task_id=cache_task_id,
            ),
            'agent_name': 'codex',
            'agent_config': _build_agent_config(),
            'openai_clients': _build_openai_clients(
                cache_task_id=cache_task_id,
                prompt_cache_key=prompt_cache_key,
            ),
        }
        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=instance_id,
            metadata={
                'instance_id': instance_id,
                'agent_name': 'codex',
                'model': _MODEL,
                'use_azure_provider': False,
                'codex_version': _CODEX_VERSION,
                'gateway': _GATEWAY,
                'reasoning_effort': _REASONING_EFFORT,
                'reasoning_mode': _reasoning_mode(),
                'cache_task_id': cache_task_id,
                'prompt_cache_key': prompt_cache_key,
            },
        )
        produced += 1
        if limit is not None and produced >= limit:
            break


def get_total_count() -> int:
    dataset_path = os.environ.get('SWE_DATASET_PATH', _DEFAULT_DATASET_PATH)
    limit = _env_int('SWE_LIMIT')  # None => all instances
    total = sum(1 for _ in _iter_instance_ids(dataset_path))
    return total if limit is None else min(limit, total)


def get_batch_metadata() -> dict[str, Any]:
    return {
        'task_name': 'swe_task',
        'agent_name': 'codex',
        'model': _MODEL,
        'use_azure_provider': False,
        'codex_version': _CODEX_VERSION,
        'endpoint': os.getenv('SWE_OPENAI_RESPONSES_ENDPOINT', _RESPONSES_ENDPOINT),
        'gateway': _GATEWAY,
        'reasoning_effort': _REASONING_EFFORT,
        'reasoning_summary': _REASONING_SUMMARY,
        'reasoning_mode': _reasoning_mode(),
        'model_context_window': _CONTEXT_WINDOW,
        'auto_compact_token_limit': _AUTO_COMPACT_TOKEN_LIMIT,
        'max_output_tokens': _MAX_OUTPUT_TOKENS,
        'timeout_s': int(_TIMEOUT_S),
        'max_retries': _MAX_RETRIES,
    }
