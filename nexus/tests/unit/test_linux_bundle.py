import json
import os
import stat
import subprocess
import tarfile

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


@pytest.fixture
def linux_builder(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / 'deployment/bootstrap/linux'
    monkeypatch.syspath_prepend(str(directory))
    spec = spec_from_file_location('linux_bundle_build', directory / 'main.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_preserves_executable_and_internal_nix_links(linux_builder, tmp_path):
    payload = tmp_path / 'payload'
    tool = payload / 'nix/store/tool/bin/run'
    tool.parent.mkdir(parents=True)
    tool.write_text('#!/bin/sh\nprintf RELOCATED_OK')
    tool.chmod(0o755)
    profile = payload / 'nix/runtime/nix-env/bin'
    profile.mkdir(parents=True)
    (profile / 'run').symlink_to('../../../store/tool/bin/run')
    (payload / 'entrypoint.py').write_text('#!/usr/bin/env python3\n')
    (payload / 'entrypoint.py').chmod(0o755)
    linux_builder.record_links(payload)
    output = tmp_path / 'output'
    output.mkdir()
    linux_builder.pack_release(payload, output, '1.2.3')
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['os'] == 'linux'
    extracted = tmp_path / 'different-location'
    extracted.mkdir()
    with tarfile.open(output / 'bundles/1.2.3.tar.gz') as tar:
        tar.extractall(extracted, filter='data')
    from entrypoint import restore_links

    restore_links(extracted)
    restore_links(extracted)  # Restart is idempotent.
    relocated = extracted / 'nix/runtime/nix-env/bin/run'
    assert relocated.resolve() == extracted / 'nix/store/tool/bin/run'
    assert relocated.stat().st_mode & stat.S_IXUSR
    assert subprocess.check_output([str(relocated)], text=True) == 'RELOCATED_OK'
    assert linux_builder.digest(output / 'bundles/1.2.3.tar.gz') == manifest['files'][0]['sha256']


def test_link_metadata_cannot_write_through_external_parent(linux_builder, tmp_path):
    from entrypoint import restore_links

    (tmp_path / 'nix').mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (tmp_path / 'nix/escape').symlink_to(outside)
    (tmp_path / 'links.json').write_text(json.dumps([{'path': 'nix/escape/file', 'target': '/etc/passwd'}]))
    with pytest.raises(ValueError, match='parent is a symlink'):
        restore_links(tmp_path)
    assert not (outside / 'file').exists()


def test_absolute_and_dangling_link_targets_are_preserved(linux_builder, tmp_path):
    from entrypoint import restore_links

    (tmp_path / 'nix').mkdir()
    links = [{'path': 'nix/link', 'target': '/etc/nonexistent-nexus-test'}]
    (tmp_path / 'links.json').write_text(json.dumps(links))
    restore_links(tmp_path)
    assert os.readlink(tmp_path / 'nix/link') == links[0]['target']


def test_namespace_preserves_host_paths_and_replaces_only_nix(linux_builder, tmp_path):
    host = tmp_path / 'host'
    for name in ('dev', 'usr', 'home', 'nix'):
        (host / name).mkdir(parents=True)
    (host / 'bin').symlink_to('usr/bin')
    release = tmp_path / 'release/nix'
    command = linux_builder.namespace_command(release, host_root=host)
    assert ['--ro-bind', str(release), '/nix'] == command[-3:]
    assert str(host / 'nix') not in command
    assert '--dev-bind' in command
    assert ['--symlink', os.readlink(host / 'bin'), '/bin'] == command[2:5]
    assert '--unshare-net' not in command


def test_launcher_keeps_temporary_paths_short_for_chromium(linux_builder, monkeypatch, tmp_path):
    import sys

    import entrypoint

    state = tmp_path / ('long-state-directory-' * 8)
    workspace = tmp_path / 'workspace'
    monkeypatch.setattr(sys, 'argv', ['entrypoint.py', '--state-root', str(state), '--workspace', str(workspace)])
    monkeypatch.setattr(entrypoint, 'restore_links', lambda release: None)
    monkeypatch.setattr(entrypoint.shutil, 'which', lambda name: '/usr/bin/bwrap')
    monkeypatch.setattr(entrypoint, 'namespace_command', lambda *args, **kwargs: ['/usr/bin/bwrap'])
    monkeypatch.setattr(entrypoint.os, 'environ', dict(os.environ, TMPDIR=str(state / 'tmp')))
    monkeypatch.chdir(tmp_path)

    def launch(executable, command):
        assert os.environ['TMPDIR'] == '/tmp'
        assert command[-2:] == ['--workspace', str(workspace)]
        raise SystemExit(0)

    monkeypatch.setattr(entrypoint.os, 'execv', launch)
    with pytest.raises(SystemExit) as result:
        entrypoint.main()
    assert result.value.code == 0
