import sys

import pytest

from nexus.core import dep_checker


def test_installer_targets_the_running_interpreter_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """uv otherwise picks the shell's activated env, which need not be the one running Nexus.

    On a venv-on-conda layout an activated conda env wins over the project ``.venv``, so
    the install reports success while the import keeps failing.
    """
    monkeypatch.setattr(dep_checker.shutil, 'which', lambda name: '/usr/local/bin/uv')

    assert dep_checker._find_installer() == ('/usr/local/bin/uv', 'pip', 'install', '--python', sys.executable)


def test_installer_falls_back_to_the_running_interpreter_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dep_checker.shutil, 'which', lambda name: None)

    assert dep_checker._find_installer() == (sys.executable, '-m', 'pip', 'install')
