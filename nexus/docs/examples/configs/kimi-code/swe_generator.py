"""Run a bounded Kimi Code SWE batch from complete JSONL instances."""

from __future__ import annotations

import copy
import json
import os

from pathlib import Path

from nexus import BatchItem
from nexus.runner.single import RunInput


def generate_configs():
    template = RunInput.from_file(Path(__file__).with_name('smoke.yaml')).model_dump(mode='json')
    limit = int(os.getenv('SWE_LIMIT', '3'))
    if limit <= 0:
        raise ValueError('SWE_LIMIT must be positive')
    requested = {value.strip() for value in os.getenv('SWE_INSTANCE_IDS', '').split(',') if value.strip()}
    agent_timeout = int(os.getenv('SWE_AGENT_TIMEOUT', '12000'))
    runtime_timeout = int(os.getenv('SWE_RUNTIME_TIMEOUT', str(agent_timeout + 900)))
    runner_timeout = int(os.getenv('SWE_RUNNER_TIMEOUT', str(runtime_timeout + 2700)))
    if not 0 < agent_timeout < runtime_timeout < runner_timeout:
        raise ValueError('Timeouts must satisfy agent < runtime < runner')
    produced = 0
    with Path(os.environ['SWE_DATASET_PATH']).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            instance = json.loads(line)
            instance_id = instance.get('instance_id')
            if not instance_id or (requested and instance_id not in requested):
                continue
            config = copy.deepcopy(template)
            config['imports'].append('nexus.extensions.tasks.swe')
            config['task_name'] = 'swe_task'
            config['task_config'] = {
                'instance_id': instance_id,
                'instance': {
                    k: v for k, v in instance.items() if k not in {'main_trajectory', 'instance_with_results'}
                },
                'enable_anti_hack_prompt': True,
                'enable_reward_hacking_detection': True,
                'enable_anti_hack': True,
                'hack_reward_override': 0.0,
                'eval_max_attempts': 1,
                'eval_timeout_seconds': int(os.getenv('SWE_EVAL_TIMEOUT', '2400')),
            }
            main = config['runtime_configs']['main']
            main.pop('image', None)
            requirements = main['worker_venv'].setdefault('extra_requirements', [])
            requirements.extend(
                value for value in ['dataclasses-json>=0.6.7', 'unidiff>=0.7.5'] if value not in requirements
            )
            config['runtime_configs']['eval'] = copy.deepcopy(main)
            config['agent_config'].pop('user_prompt', None)
            config['agent_config'].update(
                working_dir='/testbed', timeout=agent_timeout, max_steps=int(os.getenv('SWE_MAX_STEPS', '300'))
            )
            config['runtime_provider_config']['request_timeout'] = runtime_timeout
            config['runner']['timeout'] = runner_timeout
            run_id = f'kimi_code__{instance_id}'
            langfuse = config['processors']['langfuse']['args']
            langfuse.update(
                trace_name=run_id,
                tags=['kimi-code', '2.3.2', 'swe', 'high'],
                trace_metadata={
                    'instance_id': instance_id,
                    'agent_name': 'kimi_code',
                    'model': os.environ['MODEL_NAME'],
                    'dataset_type': instance.get('dataset_type'),
                },
            )
            yield BatchItem(
                config=RunInput.model_validate(config),
                run_id=run_id,
                metadata={'instance_id': instance_id, 'agent_name': 'kimi_code'},
            )
            produced += 1
            if produced >= limit:
                return
    if produced == 0:
        raise ValueError('No SWE instances matched the requested dataset/filter')
