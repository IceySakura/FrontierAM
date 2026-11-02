"""Minimal HLE batch generator for run-batch."""

from __future__ import annotations

import os

from pathlib import Path

from omegaconf import OmegaConf

from nexus import BatchItem
from nexus.extensions.tasks.search_stateful.benchmark.eval.formats import load_records


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').exists():
            return parent
    return Path(__file__).resolve().parents[-1]


def _config_path(base_config_path: str | Path | None) -> Path:
    if base_config_path:
        path = Path(base_config_path)
        if path.exists():
            return path
        raise FileNotFoundError(f'Base config not found: {path}')

    env_path = os.getenv('SCIENCE_CONFIG')
    if env_path and Path(env_path).exists():
        return Path(env_path)

    default_path = Path(__file__).resolve().parent / 'config' / 'base.yaml'
    if default_path.exists():
        return default_path

    raise FileNotFoundError('Base config not found. Provide --config or edit eval/config.')


def _input_path(input_path: str | None) -> Path:
    if input_path:
        return Path(input_path)

    env_input = os.getenv('SCIENCE_INPUT_PATH') or os.getenv('HLE_DATASET_PATH')
    if env_input:
        return Path(env_input)

    raise FileNotFoundError(
        'Dataset path not specified. Set SCIENCE_INPUT_PATH/HLE_DATASET_PATH or pass --dataset when invoking run-batch.'
    )


def _load_base_config(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def _task_config(record, index: int, base_task_config: dict) -> dict:
    category = record.category or record.raw_subject or 'general'
    task_id = record.id or f'hle_{index}'

    task_config: dict = {
        'task_id': task_id,
        'category': category,
        'question': record.question,
        'ground_truth': record.answer,  # for hle task
        'answer': record.answer,  # for search_stateful task (uses LLM judge)
        'evaluation_type': base_task_config.get('evaluation_type', 'fuzzy'),
        'timeout': base_task_config.get('timeout', 600),
    }

    metadata = {}
    for key in ['answer_type', 'raw_subject', 'author_name', 'canary']:
        value = getattr(record, key, None)
        if value is not None:
            metadata[key] = value
    if metadata:
        task_config['metadata'] = metadata

    if record.file_names and record.file_paths:
        task_config['file_names'] = record.file_names
        task_config['file_paths'] = record.file_paths

    # Propagate evaluator_type / reward_system_* from base_task_config (yaml-level defaults).
    # Record-level values still take precedence below.
    reward_system_keys = (
        'evaluator_type',
        'evaluator_none_score',
        'reward_system_user_token',
        'reward_system_exp_id',
        'reward_system_router_id',
        'reward_system_extra',
        'reward_system_public_test',
        'reward_system_ci_test',
    )
    for key in reward_system_keys:
        if key in base_task_config and base_task_config[key] is not None:
            task_config[key] = base_task_config[key]

    # Pass through evaluator_type / reward_system_* from data record (if present), overriding base config.
    for key in reward_system_keys:
        value = getattr(record, key, None)
        if value is not None:
            task_config[key] = value

    return task_config


class HLEBatchGenerator:
    def __init__(self, input_path: str | None = None, base_config_path: str | Path | None = None) -> None:
        self.input_path = _input_path(input_path)
        self.base_config = _load_base_config(_config_path(base_config_path))
        self.base_task_config = self.base_config.get('task_config', {})

    def __iter__(self):
        records = load_records(self.input_path)
        for i, record in enumerate(records):
            config = dict(self.base_config)
            config['task_config'] = _task_config(record, i, self.base_task_config)
            yield BatchItem(
                config=config,  # type: ignore[arg-type]
                run_id=f'hle_{i}',
                metadata={'index': i},
            )

    @staticmethod
    def total_count() -> int:
        return len(load_records(_input_path(None)))

    @staticmethod
    def metadata() -> dict:
        dataset_name = os.getenv('HLE_DATASET_NAME', 'hle_200')
        return {
            'description': f'HLE evaluation ({dataset_name})',
            'experiment': f'{dataset_name}_eval_v1',
        }


def generate_configs(input_path: str | None = None, base_config_path: str | Path | None = None):
    yield from HLEBatchGenerator(input_path=input_path, base_config_path=base_config_path)


def get_total_count() -> int:
    return HLEBatchGenerator.total_count()


def get_batch_metadata() -> dict:
    return HLEBatchGenerator.metadata()


__all__ = ['generate_configs', 'get_total_count', 'get_batch_metadata']
