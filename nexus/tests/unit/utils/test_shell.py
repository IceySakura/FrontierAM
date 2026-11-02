"""Tests for nexus.utils.shell.get_clean_sandbox_env PATH handling."""

from __future__ import annotations

import nexus.utils.shell as shell


def test_strips_nix_bin_and_appends_agent_bin_when_present(monkeypatch):
    """nix bin is removed; agent-bin is appended to the END of PATH."""
    monkeypatch.setattr(
        shell.os,
        'environ',
        {'PATH': f'/usr/local/bin:/usr/bin:{shell._NIX_BIN}:/bin'},
    )
    monkeypatch.setattr(shell.os.path, 'isdir', lambda p: p == shell._NIX_AGENT_BIN)

    env = shell.get_clean_sandbox_env()
    parts = env['PATH'].split(':')

    # nix bin gone, agent-bin appended at the end
    assert shell._NIX_BIN not in parts
    assert parts[-1] == shell._NIX_AGENT_BIN
    # user dirs preserved and still ahead of agent-bin
    assert parts.index('/usr/bin') < parts.index(shell._NIX_AGENT_BIN)
    assert parts[:3] == ['/usr/local/bin', '/usr/bin', '/bin']


def test_does_not_append_agent_bin_when_missing(monkeypatch):
    """If agent-bin dir does not exist, PATH is only stripped, nothing appended."""
    monkeypatch.setattr(
        shell.os,
        'environ',
        {'PATH': f'/usr/bin:{shell._NIX_BIN}'},
    )
    monkeypatch.setattr(shell.os.path, 'isdir', lambda p: False)

    env = shell.get_clean_sandbox_env()

    assert env['PATH'] == '/usr/bin'
    assert shell._NIX_AGENT_BIN not in env['PATH']


def test_agent_bin_alone_when_path_becomes_empty(monkeypatch):
    """If PATH held only the nix bin, agent-bin becomes the sole entry (no leading colon)."""
    monkeypatch.setattr(shell.os, 'environ', {'PATH': shell._NIX_BIN})
    monkeypatch.setattr(shell.os.path, 'isdir', lambda p: p == shell._NIX_AGENT_BIN)

    env = shell.get_clean_sandbox_env()

    assert env['PATH'] == shell._NIX_AGENT_BIN


def test_no_path_key_is_left_untouched(monkeypatch):
    """No PATH in environ -> no PATH synthesised."""
    monkeypatch.setattr(shell.os, 'environ', {'HOME': '/root'})
    monkeypatch.setattr(shell.os.path, 'isdir', lambda p: True)

    env = shell.get_clean_sandbox_env()

    assert 'PATH' not in env
