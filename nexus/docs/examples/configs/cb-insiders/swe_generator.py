"""SWE smoke generator for the private npm cb-insiders agent.

SWE_DATASET_PATH supplies a JSONL of complete instances. Defaults to two cases;
all credentials are read from the environment and never printed.
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
    agent_name = template['agent_name']
    client = template['openai_clients']['main']['config']
    model = {**client['request_args'], **client.get('enforced_request_args', {})}['model']
    path = Path(os.environ['SWE_DATASET_PATH'])
    limit = int(os.getenv('SWE_LIMIT', '2'))
    if limit <= 0:
        raise ValueError('SWE_LIMIT must be positive')
    agent_timeout = int(os.getenv('SWE_AGENT_TIMEOUT', '1800'))
    runtime_timeout = int(os.getenv('SWE_RUNTIME_TIMEOUT', str(agent_timeout + 900)))
    runner_timeout = int(os.getenv('SWE_RUNNER_TIMEOUT', str(agent_timeout + 3600)))
    if not 0 < agent_timeout < runtime_timeout < runner_timeout:
        raise ValueError('Timeouts must satisfy 0 < SWE_AGENT_TIMEOUT < SWE_RUNTIME_TIMEOUT < SWE_RUNNER_TIMEOUT')
    pre_agent_script = os.getenv('SWE_PRE_AGENT_SCRIPT', '').strip()
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
            main.pop('image')  # SWE picks each case's own image.
            requirements = main['worker_venv'].setdefault('extra_requirements', [])
            requirements.extend(
                requirement
                for requirement in ['dataclasses-json>=0.6.7', 'unidiff>=0.7.5']
                if requirement not in requirements
            )
            config['runtime_configs']['eval'] = copy.deepcopy(main)
            config['agent_config'].pop('user_prompt')
            config['agent_config'].update(working_dir='/testbed', timeout=agent_timeout)
            if pre_agent_script:
                config['agent_config']['pre_start_scripts'] = [{'code': pre_agent_script}]
            config['runtime_provider_config']['request_timeout'] = runtime_timeout
            config['runner']['timeout'] = runner_timeout
            langfuse = config['processors']['langfuse']['args']
            langfuse['trace_name'] = f'{agent_name}__{instance_id}'
            langfuse['tags'] = [agent_name, 'swe-smoke', model]
            langfuse['trace_metadata'] = {
                'instance_id': instance_id,
                'agent_name': agent_name,
                'model': model,
                'agent_timeout_seconds': agent_timeout,
                'runtime_timeout_seconds': runtime_timeout,
                'runner_timeout_seconds': runner_timeout,
                'pre_agent_script_enabled': bool(pre_agent_script),
            }
            yield BatchItem(config=RunInput.model_validate(config), run_id=f'{agent_name}__{instance_id}')
            count += 1
            if count >= limit:
                break
