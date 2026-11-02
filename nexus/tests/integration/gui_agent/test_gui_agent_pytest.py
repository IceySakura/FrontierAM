"""Pytest entrypoints for desktop agent integration flows."""

from __future__ import annotations

import json
import os
import subprocess

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_AGENT_ENV = 'NEXUS_RUN_GUI_AGENT_INTEGRATION'
RUNTIME_PREBUILT_ENV = 'NEXUS_GUI_RUNTIME_IMAGE_PREBUILT'
GUI_AGENT_CONFIGS = [
    'local_gui_agent_gpt54mini.yaml',
    'local_gui_apple_agent_gpt54mini.yaml',
    'gongfeng_gui_agent_gpt54mini.yaml',
    'gongfeng_gui_apple_agent_gpt54mini.yaml',
]


def _requires_gui_agent_integration() -> None:
    if os.getenv(RUN_AGENT_ENV) != '1':
        pytest.skip(f'set {RUN_AGENT_ENV}=1 to run desktop agent integration tests')
    if not (
        os.getenv('OPENAI_API_KEY')
        or (os.getenv('MODEL_SERVICE_API_APP_ID') and os.getenv('MODEL_SERVICE_API_APP_KEY'))
    ):
        pytest.skip('OPENAI_API_KEY or MODEL_SERVICE_API_APP_ID/MODEL_SERVICE_API_APP_KEY is required')


@pytest.mark.integration
@pytest.mark.gui
@pytest.mark.llm
@pytest.mark.parametrize('config_name', GUI_AGENT_CONFIGS)
def test_gui_agent_command_line_matrix(config_name, tmp_path):
    """Run every desktop agent branch through the run-single YAML command line."""
    _requires_gui_agent_integration()
    if config_name.startswith('gongfeng_') and not os.getenv('GONGFENG_TOKEN'):
        pytest.skip('GONGFENG_TOKEN is required for Gongfeng GUI agent integration tests')
    output_path = tmp_path / f'{Path(config_name).stem}.result.json'
    env = {
        **os.environ,
        'NEXUS_GUI_OBSERVE': '0',
        RUNTIME_PREBUILT_ENV: '1',
    }

    completed = subprocess.run(
        [
            'uv',
            'run',
            'nexus',
            'run-single',
            f'tests/integration/gui_agent/configs/{config_name}',
            '--output',
            str(output_path),
            '--log-trim',
            '80',
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert output_path.is_file()
    result = json.loads(output_path.read_text(encoding='utf-8'))
    assert result['success'] is True
    assert result['agent_output']['exit_status'] == 'finished'
