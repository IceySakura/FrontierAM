#!/usr/bin/env python3
"""Expose a release's Nix environment without changing the host filesystem."""

import argparse
import json
import os
import shutil

from pathlib import Path


def namespace_command(nix_root, *, readonly=True, host_root=Path('/'), bwrap='bwrap'):
    # Binding / wholesale prevents an unprivileged user from adding a missing /nix.
    # Reproduce its top-level entries in bwrap's private root instead.
    command = [bwrap, '--die-with-parent']
    for path in sorted(host_root.iterdir()):
        if path.name == 'nix':
            continue
        target = '/' + path.name
        if path.is_symlink():
            command += ['--symlink', os.readlink(path), target]
        else:
            command += ['--dev-bind' if path.name == 'dev' else '--bind', str(path), target]
    command += ['--ro-bind' if readonly else '--bind', str(nix_root), '/nix']
    return command


def restore_links(release):
    """Restore the archive's hash-verified link metadata without writing through links."""
    nix_root = (release / 'nix').resolve(strict=True)
    for item in json.loads((release / 'links.json').read_text()):
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != ('nix',):
            raise ValueError(f'Invalid Nix link path: {relative}')
        path = release / relative
        # No stored link may act as a parent for another write.
        for parent in path.relative_to(release).parents:
            if (release / parent).is_symlink():
                raise ValueError(f'Link parent is a symlink: {relative}')
        if not path.parent.resolve(strict=True).is_relative_to(nix_root):
            raise ValueError(f'Link parent escapes Nix payload: {relative}')
        if path.is_symlink():
            if os.readlink(path) != item['target']:
                raise ValueError(f'Unexpected existing link: {relative}')
        elif path.exists():
            raise ValueError(f'Link would replace an existing file: {relative}')
        else:
            path.symlink_to(item['target'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    bwrap = shutil.which('bwrap') or ''
    if not bwrap:
        parser.error('Install bubblewrap in the base image; bwrap is required')
    release = Path(__file__).resolve().parent
    restore_links(release)
    state = args.state_root.resolve()
    workspace = args.workspace.resolve()
    for path in (state, workspace, state / 'uv-cache'):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        NEXUS_HOME=str(state / 'nexus-home'),
        NEXUS_WORKSPACE=str(workspace),
        NEXUS_AUTO_INSTALL='0',
        NEXUS_BASH_BIN='/nix/runtime/nix-env/bin/bash',
        UV_CACHE_DIR=str(state / 'uv-cache'),
        UV_PYTHON_DOWNLOADS='never',
        # Chromium needs short temporary paths for its Unix-domain sockets.
        TMPDIR='/tmp',
        FONTCONFIG_FILE='/nix/runtime/nix-env/etc/fonts/fonts.conf',
    )
    os.environ['PATH'] = '/nix/runtime/nix-env/bin:' + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = '/nix/runtime/nix-env/lib' + (
        ':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''
    )
    # Keep the template's DISPLAY/XAUTHORITY and device/network access.
    os.chdir(workspace)
    command = namespace_command(release / 'nix', bwrap=bwrap)
    command += [
        '/nix/runtime/nexus/.venv/bin/python',
        '-I',
        '-B',
        '-u',
        '/nix/runtime/server.py',
        '--workspace',
        str(workspace),
    ]
    os.execv(bwrap, command)


if __name__ == '__main__':
    main()
