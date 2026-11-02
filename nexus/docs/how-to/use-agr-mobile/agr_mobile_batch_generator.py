from __future__ import annotations

import copy
import json
import os

from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


DEFAULT_CONFIG_PATH = Path(__file__).with_name('agr_mobile_batch_config.yaml')


def _get_run_suffix() -> str:
    return os.getenv('AGR_BATCH_RUN_ID', '').strip() or datetime.now().strftime('%Y%m%d_%H%M%S')


def _resolve_output_dir(output_dir: Any, mode: str) -> str | None:
    if output_dir is None:
        return None

    raw = str(output_dir).strip()
    if not raw:
        return None

    resolved = Path(raw).expanduser() / mode / _get_run_suffix()
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def _normalize_task_config(config: dict[str, Any], mode: str) -> None:
    task_config = config.get('task_config')
    if not isinstance(task_config, dict):
        return

    output_dir = _resolve_output_dir(task_config.get('output_dir'), mode)
    if output_dir:
        task_config['output_dir'] = output_dir


def _load_batch_config() -> DictConfig | None:
    config_path = os.getenv('AGR_BATCH_CONFIG_PATH', '').strip()
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        return None
    return OmegaConf.load(path)


_BATCH_CONFIG = _load_batch_config()


def _require_config() -> dict[str, Any]:
    if _BATCH_CONFIG is None:
        raise FileNotFoundError(
            f'AGR mobile batch config not found. Create {DEFAULT_CONFIG_PATH} or set AGR_BATCH_CONFIG_PATH.'
        )
    data = OmegaConf.to_container(_BATCH_CONFIG, resolve=True)
    if not isinstance(data, dict):
        raise TypeError('AGR mobile batch config must resolve to a mapping')
    return data


def _parse_sandbox_ids(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []

    if raw.startswith('['):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError('batch.sandbox_ids JSON value must be a list')
        ids = [str(item).strip() for item in data if str(item).strip()]
    else:
        normalized = raw.replace('\n', ',')
        ids = [item.strip() for item in normalized.split(',') if item.strip()]

    seen: set[str] = set()
    ordered: list[str] = []
    for sandbox_id in ids:
        if sandbox_id not in seen:
            seen.add(sandbox_id)
            ordered.append(sandbox_id)
    return ordered


def _load_sandbox_ids(batch: dict[str, Any]) -> list[str]:
    value = batch.get('sandbox_ids', [])
    if isinstance(value, ListConfig | list | tuple):
        ids = [str(item).strip() for item in value if str(item).strip()]
        return list(dict.fromkeys(ids))
    if isinstance(value, str):
        return _parse_sandbox_ids(value)
    raise TypeError('batch.sandbox_ids must be a list or string')


def _build_metadata(mode: str, sandbox_id: str | None, sandbox_template: str | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {'mode': mode}
    if sandbox_id:
        metadata['sandbox_id'] = sandbox_id
        metadata['conflict_groups'] = [f'sandbox:{sandbox_id}']
    if sandbox_template:
        metadata['sandbox_template'] = sandbox_template
    return metadata


def _build_run_input(base_run_input: dict[str, Any], runtime_config: dict[str, Any], run_id: str) -> RunInput:
    config = copy.deepcopy(base_run_input)
    config['runtime_configs'] = {'android': runtime_config}
    config['run_id'] = run_id
    return RunInput.from_dict(config)


def generate_configs():
    config = _require_config()
    batch = config.get('batch')
    if not isinstance(batch, dict):
        raise ValueError('batch config is required and must be a mapping')

    mode = str(batch.get('mode', '')).strip().lower()
    if mode not in {'template_create', 'existing_sandbox_ids'}:
        raise ValueError("batch.mode must be 'template_create' or 'existing_sandbox_ids'")

    base_run_input = copy.deepcopy(config)
    base_run_input.pop('batch', None)
    _normalize_task_config(base_run_input, mode)
    if base_run_input.get('runtime_provider_name') != 'agr_mobile':
        raise ValueError("runtime_provider_name must be 'agr_mobile'")

    provider_config = base_run_input.get('runtime_provider_config', {})
    if not isinstance(provider_config, dict):
        raise ValueError('runtime_provider_config must be a mapping')

    if mode == 'existing_sandbox_ids':
        sandbox_ids = _load_sandbox_ids(batch)
        if not sandbox_ids:
            raise ValueError('batch.sandbox_ids is required when batch.mode=existing_sandbox_ids')
        for sandbox_id in sandbox_ids:
            run_id = f'agr_mobile_{sandbox_id}'
            yield BatchItem(
                config=_build_run_input(base_run_input, {'sandbox_id': sandbox_id}, run_id),
                run_id=run_id,
                metadata=_build_metadata(mode, sandbox_id, None),
            )
        return

    batch_count = int(batch.get('count', 0))
    if batch_count <= 0:
        raise ValueError('batch.count must be > 0 when batch.mode=template_create')

    sandbox_template = str(provider_config.get('sandbox_template', '')).strip()
    if not sandbox_template:
        raise ValueError('runtime_provider_config.sandbox_template is required in template_create mode')

    for index in range(batch_count):
        run_id = f'agr_mobile_template_{index:03d}'
        yield BatchItem(
            config=_build_run_input(base_run_input, {}, run_id),
            run_id=run_id,
            metadata=_build_metadata(mode, None, sandbox_template),
        )


def get_total_count() -> int:
    config = _require_config()
    batch = config.get('batch', {})
    mode = str(batch.get('mode', '')).strip().lower()
    if mode == 'existing_sandbox_ids':
        return len(_load_sandbox_ids(batch))
    return int(batch.get('count', 0))


def get_batch_metadata() -> dict[str, Any]:
    config = _require_config()
    batch = config.get('batch', {})
    mode = str(batch.get('mode', '')).strip().lower()
    return {
        'runtime_provider_name': 'agr_mobile',
        'mode': mode,
        'sandbox_count': get_total_count(),
    }
