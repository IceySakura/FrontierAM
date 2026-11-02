from __future__ import annotations

import argparse
import asyncio
import os
import sys

from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

import pytest
import yaml

from nexus.runner.batch import BatchConfig, BatchItem
from nexus.runner.batch.runner import run_batch
from nexus.runner.single import RunInput


REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_CONFIG_PATH = REPO_ROOT / 'tests' / 'e2e' / 'config' / 'agr_mobile_e2e.yaml'
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'tests' / 'e2e' / 'agr_mobile_output'
RUN_SUFFIX = os.environ.get('AGR_E2E_RUN_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')


def _ensure_repo_venv_site_packages() -> None:
    candidates = glob(str(REPO_ROOT / '.venv' / 'lib' / 'python*' / 'site-packages'))
    for candidate in sorted(candidates):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_ensure_repo_venv_site_packages()


def _load_e2e_config() -> dict[str, Any]:
    if not E2E_CONFIG_PATH.is_file():
        example_path = REPO_ROOT / 'tests' / 'e2e' / 'config' / 'agr_mobile_e2e_example.yaml'
        pytest.skip(
            f'agr mobile e2e config not found: {E2E_CONFIG_PATH}. '
            f'Create it from {example_path.name} first.'
        )

    with E2E_CONFIG_PATH.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f'agr mobile e2e config must be a mapping: {E2E_CONFIG_PATH}')
    return data


def _base_runtime_provider_config() -> dict:
    config = _load_e2e_config()
    provider_config = config.get('runtime_provider_config')
    if not isinstance(provider_config, dict):
        raise ValueError('agr mobile e2e runtime_provider_config must be a mapping')
    return provider_config


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _prepare_output_dir(raw_path: str) -> Path:
    base_dir = _resolve_repo_path(raw_path)
    output_dir = base_dir / RUN_SUFFIX
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _build_run_input(
    *,
    run_id: str,
    runtime_config: dict,
    output_root: Path,
) -> RunInput:
    return RunInput.model_validate(
        {
            'imports': [
                'nexus.extensions.projects.android_demo.tools',
                'nexus.extensions.projects.android_demo',
            ],
            'task_name': 'agr_mobile_api_e2e',
            'task_config': {
                'output_dir': str(output_root),
                'apk_dir': str(REPO_ROOT / 'tests' / 'e2e' / 'apk'),
            },
            'runtime_provider_name': 'agr_mobile',
            'runtime_provider_config': _base_runtime_provider_config(),
            'runtime_configs': {
                'android': runtime_config,
            },
            'run_id': run_id,
        }
    )


async def _run_items(items: list[BatchItem], db_path: Path) -> None:
    from nexus.runner.batch.loader import GeneratorLoader

    generator_path = db_path.parent / f'{db_path.stem}_generator.py'
    body_lines = [
        'from nexus.runner.batch import BatchItem',
        'from nexus.runner.single import RunInput',
        '',
        'def generate_configs():',
    ]
    for item in items:
        payload = repr(item.config.model_dump())
        metadata = repr(item.metadata)
        run_id = repr(item.run_id)
        body_lines.append(
            f"    yield BatchItem(config=RunInput.model_validate({payload}), run_id={run_id}, metadata={metadata})"
        )
    generator_path.write_text('\n'.join(body_lines) + '\n')

    loader = GeneratorLoader(generator_path)
    generated = list(loader.generate())
    assert len(generated) == len(items)

    batch_config = BatchConfig(max_concurrent=len(items), enable_conflict_groups=True, fail_fast=True)
    summary = await run_batch(generator_path=generator_path, db_path=db_path, batch_config=batch_config)
    assert summary.total == len(items)
    assert summary.failed == 0
    assert summary.success == len(items)


def _assert_outputs(
    db_path: Path,
    expected_count: int,
    release_expected: bool,
    expected_sandbox_ids: dict[str, str] | None = None,
) -> None:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute('SELECT run_id, status, task_output, metadata FROM runs ORDER BY run_id').fetchall()
    finally:
        conn.close()

    assert len(rows) == expected_count
    for run_id, status, task_output_json, metadata_json in rows:
        assert status == 'success'
        assert task_output_json, run_id
        import json

        task_output = json.loads(task_output_json)
        metadata = json.loads(metadata_json)
        connected_sandbox_id = task_output['connected_sandbox_id']
        assert connected_sandbox_id
        assert task_output['device_info']['success'] is True
        assert task_output['device_info_path']
        assert Path(task_output['device_info_path']).is_file()
        assert task_output['apk_local_path']
        assert Path(task_output['apk_local_path']).is_file()
        assert task_output['upload_result']['success'] is True
        assert task_output['install_result']['success'] is True
        assert task_output['grant_result']['success'] is True
        assert task_output['launch_result']['success'] is True
        assert task_output['reset_result']['success'] is True
        screenshot_path = Path(task_output['screenshot_path'])
        assert screenshot_path.is_file()
        assert run_id in str(screenshot_path.parent)
        logcat_path = Path(task_output['logcat_path'])
        assert logcat_path.is_file()
        assert run_id in str(logcat_path.parent)
        if release_expected:
            assert 'sandbox_template' in metadata
            assert connected_sandbox_id not in expected_sandbox_ids.values() if expected_sandbox_ids else True
        else:
            assert 'sandbox_id' in metadata
            assert connected_sandbox_id == metadata['sandbox_id']
            if expected_sandbox_ids is not None:
                assert connected_sandbox_id == expected_sandbox_ids[run_id]


async def _run_template_parallel_e2e() -> None:
    config = _load_e2e_config()
    scenarios = config.get('scenarios')
    if not isinstance(scenarios, dict):
        raise ValueError('agr mobile e2e scenarios must be a mapping')

    scenario = scenarios.get('template_parallel_e2e')
    if not isinstance(scenario, dict):
        pytest.skip('template_parallel_e2e scenario is missing in agr mobile e2e config')

    sandbox_template = str(scenario.get('sandbox_template', '')).strip()
    if not sandbox_template:
        pytest.skip('template_parallel_e2e.sandbox_template is empty')

    parallel_count = int(scenario.get('parallel_count', 5))
    output_root = str(scenario.get('output_dir', DEFAULT_OUTPUT_ROOT / 'template_parallel_e2e'))
    output_dir = _prepare_output_dir(output_root)
    items: list[BatchItem] = []
    for index in range(parallel_count):
        run_id = f'agr_template_e2e_{index:02d}'
        items.append(
            BatchItem(
                config=_build_run_input(
                    run_id=run_id,
                    runtime_config={'sandbox_template': sandbox_template},
                    output_root=output_dir,
                ),
                run_id=run_id,
                metadata={'sandbox_template': sandbox_template},
            )
        )

    db_path = output_dir / 'agr_template_e2e.db'
    await _run_items(items, db_path)
    _assert_outputs(db_path, expected_count=parallel_count, release_expected=True)


@pytest.mark.e2e
def test_agr_mobile_template_parallel_e2e() -> None:
    asyncio.run(_run_template_parallel_e2e())


async def _run_existing_sandbox_parallel_e2e() -> None:
    config = _load_e2e_config()
    scenarios = config.get('scenarios')
    if not isinstance(scenarios, dict):
        raise ValueError('agr mobile e2e scenarios must be a mapping')

    scenario = scenarios.get('existing_sandbox_parallel_e2e')
    if not isinstance(scenario, dict):
        pytest.skip('existing_sandbox_parallel_e2e scenario is missing in agr mobile e2e config')

    sandbox_ids_raw = scenario.get('sandbox_ids', [])
    if not isinstance(sandbox_ids_raw, list):
        raise ValueError('existing_sandbox_parallel_e2e.sandbox_ids must be a list')

    sandbox_ids = [str(item).strip() for item in sandbox_ids_raw if str(item).strip()]
    if not sandbox_ids:
        pytest.skip('existing_sandbox_parallel_e2e.sandbox_ids is empty')

    output_root = str(scenario.get('output_dir', DEFAULT_OUTPUT_ROOT / 'existing_sandbox_parallel_e2e'))
    output_dir = _prepare_output_dir(output_root)
    items: list[BatchItem] = []
    expected_sandbox_ids: dict[str, str] = {}
    for sandbox_id in sandbox_ids:
        run_id = f'agr_existing_{sandbox_id}'
        expected_sandbox_ids[run_id] = sandbox_id
        items.append(
            BatchItem(
                config=_build_run_input(
                    run_id=run_id,
                    runtime_config={'sandbox_id': sandbox_id},
                    output_root=output_dir,
                ),
                run_id=run_id,
                metadata={
                    'sandbox_id': sandbox_id,
                    'conflict_groups': [f'sandbox:{sandbox_id}'],
                },
            )
        )

    db_path = output_dir / 'agr_existing_e2e.db'
    await _run_items(items, db_path)
    _assert_outputs(
        db_path,
        expected_count=len(items),
        release_expected=False,
        expected_sandbox_ids=expected_sandbox_ids,
    )


@pytest.mark.e2e
def test_agr_mobile_existing_sandbox_parallel_e2e() -> None:
    asyncio.run(_run_existing_sandbox_parallel_e2e())


def _parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run AGR mobile E2E scenarios directly.')
    parser.add_argument(
        'scenario',
        nargs='?',
        choices=['template_parallel_e2e', 'existing_sandbox_parallel_e2e', 'all'],
        default='all',
        help='Scenario to run. Defaults to all.',
    )
    parser.add_argument(
        '-k',
        dest='keyword',
        choices=['template_parallel_e2e', 'existing_sandbox_parallel_e2e'],
        help='Backward-compatible selector matching the old pytest -k usage.',
    )
    return parser.parse_args(argv)


async def _run_selected_scenarios(selection: str) -> None:
    if selection in ('template_parallel_e2e', 'all'):
        await _run_template_parallel_e2e()
    if selection in ('existing_sandbox_parallel_e2e', 'all'):
        await _run_existing_sandbox_parallel_e2e()


if __name__ == '__main__':
    args = _parse_cli_args(sys.argv[1:])
    selection = args.keyword or args.scenario
    try:
        asyncio.run(_run_selected_scenarios(selection))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
