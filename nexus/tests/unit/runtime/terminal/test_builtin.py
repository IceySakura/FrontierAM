import pytest

from nexus.runtime.executors.terminal.builtin import BuiltinUnavailableError, locate_patched_bash


def test_env_var_takes_priority(tmp_path, monkeypatch):
    bash = tmp_path / 'bash'
    bash.write_bytes(b'\x00')
    bash.chmod(0o755)
    monkeypatch.setenv('NEXUS_BASH_BIN', str(bash))
    assert locate_patched_bash(fixed_path=str(tmp_path / 'nope')) == str(bash)


def test_env_var_missing_file_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv('NEXUS_BASH_BIN', str(tmp_path / 'nope'))
    with pytest.raises(BuiltinUnavailableError):
        locate_patched_bash(fixed_path=str(tmp_path / 'also-nope'))


def test_fixed_path_used_when_present(tmp_path, monkeypatch):
    monkeypatch.delenv('NEXUS_BASH_BIN', raising=False)
    fixed = tmp_path / 'bash'
    fixed.write_bytes(b'\x00')
    fixed.chmod(0o755)
    assert locate_patched_bash(fixed_path=str(fixed)) == str(fixed)


def test_raises_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.delenv('NEXUS_BASH_BIN', raising=False)
    with pytest.raises(BuiltinUnavailableError):
        locate_patched_bash(fixed_path=str(tmp_path / 'nope'))
