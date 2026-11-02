"""Pytest entrypoints for CUA-Gym task integration flows."""

from __future__ import annotations

import json
import os
import subprocess

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_CUA_GYM_ENV = 'NEXUS_RUN_CUA_GYM_INTEGRATION'
RUNTIME_PREBUILT_ENV = 'NEXUS_GUI_RUNTIME_IMAGE_PREBUILT'
CUA_GYM_COMMAND_LINE_CONFIGS = [
    'cua_gym_docker.yaml',
    'cua_gym_gongfeng.yaml',
]


def _requires_cua_gym_integration() -> None:
    if os.getenv(RUN_CUA_GYM_ENV) != '1':
        pytest.skip(f'set {RUN_CUA_GYM_ENV}=1 to run CUA-Gym integration tests')
    if not (
        os.getenv('OPENAI_API_KEY')
        or (os.getenv('MODEL_SERVICE_API_APP_ID') and os.getenv('MODEL_SERVICE_API_APP_KEY'))
    ):
        pytest.skip('OPENAI_API_KEY or MODEL_SERVICE_API_APP_ID/MODEL_SERVICE_API_APP_KEY is required')


async def _ensure_local_images() -> None:
    from tests.integration.gui_agent.runtime_image import ensure_runtime_image

    from nexus.extensions.tasks.cua_gym.cua_gym_session_image import ensure_cua_gym_session_image

    await ensure_runtime_image()
    ensure_cua_gym_session_image()


@pytest.mark.integration
@pytest.mark.gui
@pytest.mark.llm
@pytest.mark.parametrize('config_name', CUA_GYM_COMMAND_LINE_CONFIGS)
async def test_cua_gym_command_line_matrix(config_name, monkeypatch, tmp_path):
    """Run CUA-Gym smoke configs through nexus run-single."""
    _requires_cua_gym_integration()
    if 'gongfeng' in config_name and not os.getenv('GONGFENG_TOKEN'):
        pytest.skip('GONGFENG_TOKEN is required for Gongfeng CUA-Gym integration tests')
    monkeypatch.setenv(RUNTIME_PREBUILT_ENV, '1')
    if 'docker' in config_name:
        await _ensure_local_images()
    output_path = tmp_path / f'{Path(config_name).stem}.result.json'

    completed = subprocess.run(
        [
            'uv',
            'run',
            '--extra',
            'runtime',
            'nexus',
            'run-single',
            f'tests/integration/cua_gym/configs/{config_name}',
            '--output',
            str(output_path),
            '--log-trim',
            '80',
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, RUNTIME_PREBUILT_ENV: '1'},
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert output_path.is_file()
    result = json.loads(output_path.read_text(encoding='utf-8'))
    task_output = result['task_output']
    assert result['success'] is True
    assert task_output['completed'] is True
    assert task_output['score'] == 1.0
    assert task_output['evaluator_result']['supported'] is True
    assert task_output['agent_output']['exit_status'] != 'tool_error'
