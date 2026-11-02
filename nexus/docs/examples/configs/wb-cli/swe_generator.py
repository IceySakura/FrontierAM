"""SWE smoke generator for the private npm WorkBuddy agent.

``smoke.yaml`` is the tool-smoke template. ``SWE_DATASET_PATH`` supplies a JSONL
of complete instances. Credentials stay in the environment.
"""

from __future__ import annotations

import copy
import json
import os

from pathlib import Path

from nexus import BatchItem
from nexus.runner.single import RunInput


def generate_configs():
    template = RunInput.from_file(Path(__file__).with_name('smoke.yaml')).model_dump(mode='json')
    path = Path(os.environ['SWE_DATASET_PATH'])
    limit = int(os.getenv('SWE_LIMIT', '1'))
    if limit <= 0:
        raise ValueError('SWE_LIMIT must be positive')
    agent_timeout = int(os.getenv('SWE_AGENT_TIMEOUT', '1800'))
    runtime_timeout = int(os.getenv('SWE_RUNTIME_TIMEOUT', str(agent_timeout + 900)))
    runner_timeout = int(os.getenv('SWE_RUNNER_TIMEOUT', str(agent_timeout + 3600)))
    if not 0 < agent_timeout < runtime_timeout < runner_timeout:
        raise ValueError('Timeouts must satisfy 0 < SWE_AGENT_TIMEOUT < SWE_RUNTIME_TIMEOUT < SWE_RUNNER_TIMEOUT')
    requested = {item.strip() for item in os.getenv('SWE_INSTANCE_IDS', '').split(',') if item.strip()}
    count = 0
    with path.open(encoding='utf-8') as handle:
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
                    key: value
                    for key, value in instance.items()
                    if key not in {'main_trajectory', 'instance_with_results'}
                },
                'enable_anti_hack': True,
                'enable_anti_hack_prompt': True,
                'enable_reward_hacking_detection': True,
                'hack_reward_override': 0.0,
                'eval_max_attempts': 1,
                'eval_timeout_seconds': 900,
            }
            main = config['runtime_configs']['main']
            main.pop('image', None)
            main['worker_venv']['extra_requirements'] = ['dataclasses-json>=0.6.7', 'unidiff>=0.7.5']
            config['runtime_configs']['eval'] = copy.deepcopy(main)
            config['agent_config'].update(working_dir='/testbed', timeout=agent_timeout)
            if os.getenv('WB_THINKING', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
                config['agent_config']['thinking'] = True
            config['runtime_provider_config']['request_timeout'] = runtime_timeout
            config['runner']['timeout'] = runner_timeout
            yield BatchItem(config=RunInput.model_validate(config), run_id=f'workbuddy__{instance_id}')
            count += 1
            if count >= limit:
                break
    if count == 0:
        raise ValueError(f'no instances selected from {path}')
