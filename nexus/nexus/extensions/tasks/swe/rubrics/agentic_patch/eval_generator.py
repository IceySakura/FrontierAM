"""Batch generator: rubric evaluation (eval_only).

Reads three sources:
  1. SWE agent batch DB/JSONL → patch_text per instance
  2. Rubric gen batch DB/JSONL → rubrics_yaml per instance
  3. Dataset JSONL → problem_statement per instance

Produces swe_rubric_eval_task configs. No runtime needed.

Environment variables:
  SWE_RESULTS_PATH: SWE agent batch DB or JSONL (required, provides patch_text)
  RUBRIC_GEN_RESULTS_PATH: rubric gen batch DB or JSONL (required, provides rubrics_yaml)
  SWE_DATASET_PATH: dataset JSONL (required, provides problem_statement)
  OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL: LLM config (required)
  RUBRIC_EVAL_API_KEY / RUBRIC_EVAL_BASE_URL / RUBRIC_EVAL_MODEL: LLM overrides (optional)
  SWE_INSTANCE_IDS: comma-separated filter (optional)
  SWE_LIMIT: max count (optional)
"""

import logging
import os

from collections.abc import Iterator
from typing import Any

from nexus.extensions.tasks.swe.rubrics.shared.generator_utils import (
    AGENT_IMPORT,
    EVAL_TASK_IMPORT,
    LITELLM_IMPORT,
    build_llm_client_from_env,
    env_int,
    env_list,
    load_dataset_instances,
    load_runs_from_source,
    require_path,
)
from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)


def generate_configs() -> Iterator[BatchItem]:
    swe_results_path = require_path('SWE_RESULTS_PATH', 'SWE_RESULTS_PATH required (SWE agent batch DB/JSONL)')
    rubric_gen_path = require_path(
        'RUBRIC_GEN_RESULTS_PATH', 'RUBRIC_GEN_RESULTS_PATH required (rubric gen batch DB/JSONL)'
    )
    dataset_path = require_path('SWE_DATASET_PATH', 'SWE_DATASET_PATH required (for problem_statement)')

    instance_filter = set(env_list('SWE_INSTANCE_IDS'))
    limit = env_int('SWE_LIMIT') or None
    eval_client_name = os.environ.get('RUBRIC_EVAL_CLIENT_NAME', 'rubric_eval')

    # Validate LLM config early
    eval_client = build_llm_client_from_env('RUBRIC_EVAL')
    openai_clients = {eval_client_name: eval_client}

    # Load all three sources, index by instance_id
    swe_runs = load_runs_from_source(swe_results_path)
    swe_by_id = {r['instance_id']: r for r in swe_runs if r['instance_id']}

    rubric_runs = load_runs_from_source(rubric_gen_path)
    rubric_by_id = {r['instance_id']: r for r in rubric_runs if r['instance_id']}

    dataset_instances = load_dataset_instances(dataset_path)
    dataset_by_id = {inst.get('instance_id', ''): inst for inst in dataset_instances if inst.get('instance_id')}

    # Join on instance_id: need all three sources
    all_ids = set(swe_by_id.keys()) & set(rubric_by_id.keys()) & set(dataset_by_id.keys())
    logger.debug(
        f'[eval_generator] SWE runs: {len(swe_by_id)}, rubric runs: {len(rubric_by_id)}, '
        f'dataset: {len(dataset_by_id)}, intersection: {len(all_ids)}'
    )

    count = 0
    for instance_id in sorted(all_ids):
        if instance_filter and instance_id not in instance_filter:
            continue

        swe_output = swe_by_id[instance_id]['task_output']
        rubric_output = rubric_by_id[instance_id]['task_output']
        dataset_inst = dataset_by_id[instance_id]

        patch_text = swe_output.get('patch_text', '')
        rubrics_yaml = rubric_output.get('rubrics_yaml', '')
        problem_statement = dataset_inst.get('problem_statement', '')

        if not patch_text:
            logger.warning(f'{instance_id}: missing patch_text, skipping')
            continue
        if not rubrics_yaml:
            logger.warning(f'{instance_id}: missing rubrics_yaml, skipping')
            continue

        yield BatchItem(
            config=RunInput(
                imports=[EVAL_TASK_IMPORT, AGENT_IMPORT, LITELLM_IMPORT],
                task_name='swe_rubric_eval_task',
                task_config={
                    'rubrics_yaml': rubrics_yaml,
                    'patch_text': patch_text,
                    'problem_statement': problem_statement,
                    'eval_client_name': eval_client_name,
                },
                openai_clients=openai_clients,
                # mock agent_name is required for runner to inject openai_clients into ExecutionContext
                agent_name='swe_agent_rubrics',
                agent_config={},
            ),
            run_id=f'{instance_id}_rubric_eval',
            metadata={'instance_id': instance_id},
        )

        count += 1
        if limit and count >= limit:
            return


def get_batch_metadata() -> dict[str, Any]:
    return {
        'task_type': 'rubric_eval',
        'swe_results_path': os.environ.get('SWE_RESULTS_PATH', ''),
        'rubric_gen_results_path': os.environ.get('RUBRIC_GEN_RESULTS_PATH', ''),
        'dataset_path': os.environ.get('SWE_DATASET_PATH', ''),
    }
