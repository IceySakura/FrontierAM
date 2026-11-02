"""Pytest entrypoints for local desktop GUI integration smoke flows."""

from __future__ import annotations

import os
import subprocess
import sys

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_GUI_ENV = 'NEXUS_RUN_GUI_INTEGRATION'
RUNTIME_PREBUILT_ENV = 'NEXUS_GUI_RUNTIME_IMAGE_PREBUILT'


def _requires_gui_integration() -> None:
    if os.getenv(RUN_GUI_ENV) != '1':
        pytest.skip(f'set {RUN_GUI_ENV}=1 to run Docker GUI integration tests')


@pytest.mark.integration
@pytest.mark.gui
async def test_local_gui_runtime_smoke_direct_api(monkeypatch):
    """Run the deterministic local GUI smoke through the Python API."""
    _requires_gui_integration()
    monkeypatch.setenv(RUNTIME_PREBUILT_ENV, '1')
    from tests.integration.gui_agent.run_local_gui_runtime_smoke import run_local_gui_runtime_smoke

    result = await run_local_gui_runtime_smoke()

    assert result['success'] is True
    assert {step['name'] for step in result['steps']} >= {
        'gui_start',
        'gui_get_screen_size',
        'gui_launch_app_xterm',
        'run_command_verify_file',
        'gui_screenshot_final',
    }


@pytest.mark.integration
@pytest.mark.gui
def test_local_gui_runtime_smoke_command_line():
    """Run the deterministic local GUI smoke through its command-line Python entrypoint."""
    _requires_gui_integration()
    env = {
        **os.environ,
        RUNTIME_PREBUILT_ENV: '1',
    }

    completed = subprocess.run(
        [sys.executable, 'tests/integration/gui_agent/run_local_gui_runtime_smoke.py'],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert '"success": true' in completed.stdout
    assert '"name": "run_command_verify_file"' in completed.stdout


@pytest.mark.integration
@pytest.mark.gui
async def test_ubuntu_session_smoke_direct_api(monkeypatch):
    """Run the heavier Ubuntu desktop session smoke through the Python API."""
    _requires_gui_integration()
    monkeypatch.setenv(RUNTIME_PREBUILT_ENV, '1')
    from tests.integration.gui_agent.run_local_ubuntu_session_runtime_smoke import run_ubuntu_session_smoke

    result = await run_ubuntu_session_smoke()

    assert result['success'] is True
    assert {step['name'] for step in result['steps']} >= {
        'gui_start_gnome_session',
        'gnome_shell_process_running',
        'gui_launch_app_libreoffice_writer',
        'libreoffice_writer_window_visible',
    }
