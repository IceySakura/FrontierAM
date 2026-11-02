from pathlib import Path

import pytest

from nexus.runtime.server import platform as runtime_platform


def test_windows_venv_paths(monkeypatch):
    monkeypatch.setattr(runtime_platform, 'is_windows', lambda: True)
    root = Path('runtime-venv')

    assert runtime_platform.venv_python_path(root) == root / 'Scripts' / 'python.exe'
    assert runtime_platform.venv_site_packages_path(root) == root / 'Lib' / 'site-packages'


def test_posix_venv_paths(monkeypatch):
    monkeypatch.setattr(runtime_platform, 'is_windows', lambda: False)
    root = Path('runtime-venv')

    assert runtime_platform.venv_python_path(root) == root / 'bin' / 'python'
    assert runtime_platform.venv_site_packages_path(root).parts[-1] == 'site-packages'


@pytest.mark.parametrize('windows', [True, False])
def test_subprocess_flags_follow_platform(monkeypatch, windows):
    import subprocess

    monkeypatch.setattr(runtime_platform, 'is_windows', lambda: windows)
    monkeypatch.setattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 512, raising=False)
    assert runtime_platform.subprocess_creation_flags() == (512 if windows else 0)
