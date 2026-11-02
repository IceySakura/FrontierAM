"""Batch generator: integrated rubric gen + eval (full mode).

Reads two sources:
  1. SWE agent batch DB/JSONL → patch_text per instance
  2. Dataset JSONL → instance info, gold_patch, problem_statement

Produces swe_rubric_task configs (generates rubrics in runtime, then evaluates).

Environment variables:
  SWE_RESULTS_PATH: SWE agent batch DB or JSONL (required, provides patch_text)
  SWE_DATASET_PATH: dataset JSONL (required, provides instance info + gold_patch)
  OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL: LLM config (required)
  RUBRIC_GEN_API_KEY / RUBRIC_GEN_BASE_URL / RUBRIC_GEN_MODEL: gen LLM overrides (optional)
  RUBRIC_EVAL_API_KEY / RUBRIC_EVAL_BASE_URL / RUBRIC_EVAL_MODEL: eval LLM overrides (optional)
  SWE_INSTANCE_IDS: comma-separated filter (optional)
  SWE_LIMIT: max count (optional)
  RUBRIC_AGENT_NAME: agent name (default: swe_agent_rubrics)
  SWE_RUNTIME_PROVIDER_NAME: runtime provider (default: gongfeng)
  SWE_RUNTIME_BASE_URL / SWE_RUNTIME_TOKEN / SWE_RUNTIME_IMAGE: runtime config
  SWE_RUNTIME_TIMEOUT / SWE_RUNTIME_CPU / SWE_RUNTIME_MEMORY: runtime resources
  SWE_MAX_ITERATIONS: agent max iterations (default: 50)
"""

import logging
import os

from collections.abc import Iterator
from typing import Any

from nexus.extensions.tasks.swe.rubrics.shared.generator_utils import (
    AGENT_IMPORT,
    FULL_TASK_IMPORT,
    LITELLM_IMPORT,
    build_agent_config,
    build_llm_client_from_env,
    build_processors,
    build_runtime_provider_config,
    env_int,
    env_list,
    load_runs_from_source,
    require_path,
)
from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)


def generate_configs() -> Iterator[BatchItem]:
    swe_results_path = require_path('SWE_RESULTS_PATH', 'SWE_RESULTS_PATH required (SWE agent batch DB/JSONL)')
    dataset_path = require_path('SWE_DATASET_PATH', 'SWE_DATASET_PATH required (for instance info + gold_patch)')

    instance_filter = set(env_list('SWE_INSTANCE_IDS'))
    limit = env_int('SWE_LIMIT') or None
    agent_name = os.environ.get('RUBRIC_AGENT_NAME', 'swe_agent_rubrics')
    gen_client_name = os.environ.get('RUBRIC_GEN_CLIENT_NAME', 'main')
    eval_client_name = os.environ.get('RUBRIC_EVAL_CLIENT_NAME', 'rubric_eval')
    runtime_provider_name = os.environ.get('SWE_RUNTIME_PROVIDER_NAME', 'gongfeng')

    # Validate both LLM configs early
    gen_client = build_llm_client_from_env('RUBRIC_GEN')
    eval_client = build_llm_client_from_env('RUBRIC_EVAL')
    openai_clients = {
        gen_client_name: gen_client,
        eval_client_name: eval_client,
    }

    runtime_provider_config = build_runtime_provider_config()
    agent_config = build_agent_config()
    processors = build_processors()

    # Load SWE results, index by instance_id
    swe_runs = load_runs_from_source(swe_results_path)
    swe_by_id = {r['instance_id']: r for r in swe_runs if r['instance_id']}
    logger.debug(f'[full_generator] Loaded {len(swe_by_id)} SWE runs from {swe_results_path}')

    count = 0
    for instance_id, swe_run in swe_by_id.items():
        if instance_filter and instance_id not in instance_filter:
            continue

        patch_text = swe_run['task_output'].get('patch_text', '')
        if not patch_text:
            logger.warning(f'{instance_id}: missing patch_text, skipping')
            continue

        task_config: dict[str, Any] = {
            'instance_id': instance_id,
            'dataset_path': dataset_path,
            'patch_text': patch_text,
            'gen_client_name': gen_client_name,
            'eval_client_name': eval_client_name,
            'rubric_agent_name': agent_name,
        }

        yield BatchItem(
            config=RunInput(
                imports=[FULL_TASK_IMPORT, AGENT_IMPORT, LITELLM_IMPORT],
                task_name='swe_rubric_task',
                task_config=task_config,
                agent_name=agent_name,
                agent_config=agent_config,
                runtime_provider_name=runtime_provider_name,
                runtime_provider_config=runtime_provider_config,
                runtime_configs={},
                openai_clients=openai_clients,
                processors=processors,
            ),
            run_id=f'{instance_id}_rubric',
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
        'task_type': 'rubric_full',
        'swe_results_path': os.environ.get('SWE_RESULTS_PATH', ''),
        'dataset_path': os.environ.get('SWE_DATASET_PATH', ''),
        'runtime_provider': os.environ.get('SWE_RUNTIME_PROVIDER_NAME', 'gongfeng'),
    }
