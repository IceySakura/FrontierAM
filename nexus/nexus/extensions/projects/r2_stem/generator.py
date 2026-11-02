"""Batch generator for R2 STEM question/ref_answer/image JSONL samples."""

import copy
import json
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus.extensions.projects.r2_stem.evaluator.common import verifier_registry_network_target
from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


TASK_CONFIG_ROW_KEYS = (
    'evaluation_method',
    'task_dir',
    'test_timeout',
    'agent_timeout',
    'agent_wall_time_reserve_seconds',
    'stop_at_trajectory_budget',
    'verifier_score_truncated',
    'working_dir',
    'docker_timeout',
    'verifier_runtime_name',
    'verifier_image',
    'verifier_dind',
    'verifier_network_access',
    'verifier_artifacts_read_only',
    'verifier_mounts',
    'verifier_allowed_protected_mounts',
    'verifier_env',
    'verifier_env_refs',
)


# Headroom between the agent budget and the transport ceiling, covering sandbox
# acquisition and image pull before the invoke clock starts.
REQUEST_TIMEOUT_HEADROOM_SECONDS = 600.0


def _config_bool(value: Any, *, sample_id: str, field: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise TypeError(f'Sample {sample_id} {field} must be a boolean')


def _resolve_agent_name(config: dict[str, Any], *, sample_id: str) -> str:
    """Mirror the runner-level agent into task_config so the task never depends on ExecutionContext."""
    runner_agent = str(config.get('agent_name') or '').strip()
    task_agent = str(config['task_config'].get('agent_name') or '').strip()
    if runner_agent and task_agent and runner_agent != task_agent:
        raise ValueError(
            f'Sample {sample_id} task_config.agent_name ({task_agent}) does not match agent_name ({runner_agent})'
        )
    agent_name = task_agent or runner_agent
    if not agent_name:
        raise KeyError(f'Sample {sample_id} missing required key: agent_name')
    return agent_name


def _raise_request_timeout(config: dict[str, Any]) -> None:
    """Keep the provider transport ceiling above the agent budget.

    ``call_runtime_agent`` sends no per-request timeout, so the invoke falls back to
    ``runtime_provider_config.request_timeout``. When that is the smaller of the two,
    the transport deadline fires first and surfaces as a bogus agent timeout.
    """
    from nexus.extensions.projects.r2_stem.task import ScienceAgentTaskInput

    default_agent_timeout = ScienceAgentTaskInput.model_fields['agent_timeout'].default
    agent_timeout = float(config['task_config'].get('agent_timeout') or default_agent_timeout)
    agent_config_timeout = config.get('agent_config', {}).get('timeout')
    if agent_config_timeout is not None:
        agent_timeout = max(agent_timeout, float(agent_config_timeout))

    provider_config = config.setdefault('runtime_provider_config', {})
    provider_config['request_timeout'] = max(
        float(provider_config.get('request_timeout') or 0.0),
        agent_timeout + REQUEST_TIMEOUT_HEADROOM_SECONDS,
    )


def _iter_rows(path: str) -> Iterator[tuple[int, dict[str, Any]]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f'Terminus JSONL input not found: {path}')
    with jsonl_path.open(encoding='utf-8', errors='replace') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f'Line {line_no} must be a JSON object')
            yield line_no, row


def _build_config(base_config: RunInput, row: dict[str, Any], sample_id: str) -> RunInput:
    if 'question' not in row:
        raise KeyError(f'Sample {sample_id} missing required key: question')
    if 'image' not in row:
        print(f'Sample {sample_id} missing image, using default image: python:3.11-slim')
        row['image'] = 'python:3.11-slim'

    config = copy.deepcopy(base_config.model_dump())
    config.setdefault('task_config', {})
    config['task_config']['question'] = row['question']
    config['task_config']['ref_answer'] = row.get('ref_answer')
    config['task_config']['image'] = row['image']
    config['task_config']['sample_id'] = sample_id
    config['task_config']['agent_name'] = _resolve_agent_name(config, sample_id=sample_id)
    for key in TASK_CONFIG_ROW_KEYS:
        if key in row:
            config['task_config'][key] = row[key]
    _raise_request_timeout(config)

    if config['task_config'].get('evaluation_method') == 'verifier_image':
        if row.get('compose_enabled') is True:
            raise NotImplementedError(
                f'Sample {sample_id} requires Compose lifecycle support; verifier_image is post-agent only'
            )
        verifier_image = str(config['task_config'].get('verifier_image') or '').strip()
        if not verifier_image:
            raise KeyError(f'Sample {sample_id} missing required key for verifier_image: verifier_image')
        verifier_runtime_name = str(config['task_config'].get('verifier_runtime_name') or 'verifier')
        runtime_configs = config.setdefault('runtime_configs', {})
        verifier_runtime_config = copy.deepcopy(runtime_configs.get(verifier_runtime_name) or {})
        row_runtime_config = row.get('verifier_runtime_config') or {}
        if not isinstance(row_runtime_config, dict):
            raise TypeError(f'Sample {sample_id} verifier_runtime_config must be an object')
        verifier_runtime_config.update(copy.deepcopy(row_runtime_config))
        verifier_runtime_config['image'] = verifier_image
        verifier_dind = _config_bool(
            config['task_config'].get('verifier_dind'), sample_id=sample_id, field='verifier_dind'
        )
        verifier_runtime_config['enable_dind'] = verifier_dind
        verifier_runtime_config.setdefault('working_dir', str(row.get('verifier_working_dir') or '/app'))
        network_policy = verifier_runtime_config.setdefault('network_policy', {'default_action': 'deny', 'egress': []})
        if verifier_dind:
            # Only the nested runner pulls; the inner daemon starts with an empty image store.
            registry_rule = {'action': 'allow', 'target': verifier_registry_network_target(verifier_image)}
            egress = network_policy.setdefault('egress', [])
            if registry_rule not in egress:
                egress.append(registry_rule)
        if 'verifier_resource_spec' in row:
            verifier_runtime_config['resource_spec'] = copy.deepcopy(row['verifier_resource_spec'])
        runtime_configs[verifier_runtime_name] = verifier_runtime_config
    return RunInput.model_validate(config)


def generate_configs() -> Iterator[BatchItem]:
    input_path = os.getenv('MINI_SWE_INPUT', '').strip()
    if not input_path:
        raise ValueError('MINI_SWE_INPUT is required')
    base_config_path = os.getenv('MINI_SWE_BASE_CONFIG', '').strip()
    if not base_config_path:
        raise ValueError('MINI_SWE_BASE_CONFIG is required')
    repeat = int(os.getenv('MINI_SWE_REPEAT', '').strip() or 1)

    base_config = RunInput.from_file(base_config_path)

    for line_no, row in _iter_rows(input_path):
        sample_id = str(row.get('id') or row.get('sample_id') or f'line-{line_no:06d}')
        for repeat_index in range(repeat):
            run_id = sample_id if repeat == 1 else f'{sample_id}-pass{repeat_index + 1:02d}'
            yield BatchItem(
                config=_build_config(base_config, row, sample_id),
                run_id=run_id,
                metadata={
                    'sample_id': sample_id,
                    'repeat_index': repeat_index,
                    'repeat_count': repeat,
                    'line_no': line_no,
                    'image': row.get('image', ''),
                },
            )


def get_total_count() -> int:
    input_path = os.getenv('MINI_SWE_INPUT', '').strip()
    if not input_path:
        raise ValueError('MINI_SWE_INPUT is required')
    repeat = int(os.getenv('MINI_SWE_REPEAT', '').strip() or 1)
    total = 0
    for _line_no, _row in _iter_rows(input_path):
        total += 1
    return total * repeat


def get_batch_metadata() -> dict[str, Any]:
    return {
        'task_name': 'science_agent_task',
        'input_path': os.getenv('MINI_SWE_INPUT', ''),
        'base_config': os.getenv('MINI_SWE_BASE_CONFIG', ''),
        'repeat': int(os.getenv('MINI_SWE_REPEAT', '').strip() or 1),
    }


__all__ = ['generate_configs', 'get_total_count', 'get_batch_metadata']
