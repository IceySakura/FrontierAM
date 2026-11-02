"""Batch generator: rubric generation (gen_only).

Reads a SWE dataset JSONL → generates rubrics for each instance via agent in runtime.

Environment variables:
  SWE_DATASET_PATH: dataset JSONL (required)
  OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL: LLM config (required)
  RUBRIC_GEN_API_KEY / RUBRIC_GEN_BASE_URL / RUBRIC_GEN_MODEL: LLM overrides (optional)
  SWE_INSTANCE_IDS: comma-separated filter (optional)
  SWE_LIMIT: max count (optional)
  RUBRIC_AGENT_NAME: agent name (default: swe_agent_rubrics)
  SWE_RUNTIME_PROVIDER_NAME: runtime provider (default: gongfeng)
  SWE_RUNTIME_BASE_URL / SWE_RUNTIME_TOKEN / SWE_RUNTIME_IMAGE: runtime config
  SWE_RUNTIME_TIMEOUT / SWE_RUNTIME_CPU / SWE_RUNTIME_MEMORY: runtime resources
  SWE_MAX_ITERATIONS: agent max iterations (default: 50)
"""

import os

from collections.abc import Iterator
from typing import Any

from nexus.extensions.tasks.swe.rubrics.shared.generator_utils import (
    AGENT_IMPORT,
    GEN_TASK_IMPORT,
    LITELLM_IMPORT,
    build_agent_config,
    build_llm_client_from_env,
    build_processors,
    build_runtime_provider_config,
    env_int,
    env_list,
    load_dataset_instances,
    require_path,
)
from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


def generate_configs() -> Iterator[BatchItem]:
    dataset_path = require_path('SWE_DATASET_PATH')
    instances = load_dataset_instances(dataset_path)

    instance_filter = set(env_list('SWE_INSTANCE_IDS'))
    limit = env_int('SWE_LIMIT') or None
    agent_name = os.environ.get('RUBRIC_AGENT_NAME', 'swe_agent_rubrics')
    gen_client_name = os.environ.get('RUBRIC_GEN_CLIENT_NAME', 'main')
    runtime_provider_name = os.environ.get('SWE_RUNTIME_PROVIDER_NAME', 'gongfeng')

    # Validate LLM config early
    gen_client = build_llm_client_from_env('RUBRIC_GEN')
    openai_clients = {gen_client_name: gen_client}

    runtime_provider_config = build_runtime_provider_config()
    agent_config = build_agent_config()
    processors = build_processors()

    count = 0
    for instance in instances:
        instance_id = instance.get('instance_id', '')
        if not instance_id:
            continue
        if instance_filter and instance_id not in instance_filter:
            continue

        yield BatchItem(
            config=RunInput(
                imports=[GEN_TASK_IMPORT, AGENT_IMPORT, LITELLM_IMPORT],
                task_name='swe_rubric_gen_task',
                task_config={
                    'dataset_path': dataset_path,
                    'instance_id': instance_id,
                    'gen_client_name': gen_client_name,
                },
                agent_name=agent_name,
                agent_config=agent_config,
                runtime_provider_name=runtime_provider_name,
                runtime_provider_config=runtime_provider_config,
                runtime_configs={},
                openai_clients=openai_clients,
                processors=processors,
            ),
            run_id=f'{instance_id}_rubric_gen',
            metadata={
                'instance_id': instance_id,
                'dataset_path': dataset_path,
            },
        )

        count += 1
        if limit and count >= limit:
            return


def get_batch_metadata() -> dict[str, Any]:
    return {
        'task_type': 'rubric_gen',
        'dataset_path': os.environ.get('SWE_DATASET_PATH', ''),
        'runtime_provider': os.environ.get('SWE_RUNTIME_PROVIDER_NAME', 'gongfeng'),
    }
