#!/usr/bin/env python3
"""Build private npm carrier packages with the exact upstream deb bytes.

Run from the repository root with ``uv run --no-sync python ...``. Publication
is explicit; registry credentials come only from the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import urllib.request

from pathlib import Path

from nexus.extensions.agents.cb_insiders.installer import (
    DEB_SHA256,
    DEFAULT_VERSION,
    PACKAGE_PREFIX,
    UPSTREAM_BUILD,
    UPSTREAM_COMMIT,
    private_npmrc,
    sha256,
    source_url,
)


def build_package(archive: Path, arch: str, destination: Path, registry: str) -> dict:
    expected = DEB_SHA256[arch]
    actual = sha256(archive)
    if actual != expected:
        raise ValueError(f'{arch} deb SHA256 mismatch: {actual}')
    name = PACKAGE_PREFIX + arch
    manifest = {
        'name': name,
        'version': DEFAULT_VERSION,
        'description': f'Unmodified CodeBuddy CN Insiders Linux {arch} artifact for Nexus',
        'license': 'UNLICENSED',
        'os': ['linux'],
        'cpu': [arch],
        'files': ['payload/codebuddy-insiders.deb', 'upstream.json', 'README.md'],
        'publishConfig': {'registry': registry},
    }
    provenance = {
        'build': UPSTREAM_BUILD,
        'commit': UPSTREAM_COMMIT,
        'architecture': arch,
        'source_url': source_url(arch),
        'sha256': expected,
        'bytes': archive.stat().st_size,
    }
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f'nexus-cb-insiders-linux-{arch}-{DEFAULT_VERSION}.tgz'
    # The deb is already xz-compressed; gzip level 1 avoids recompressing 230 MB
    # at high CPU cost. No npm lifecycle scripts or external downloads are used.
    with tarfile.open(output, 'w:gz', compresslevel=1, format=tarfile.PAX_FORMAT) as tar:
        for filename, value in {
            'package.json': json.dumps(manifest, indent=2) + '\n',
            'upstream.json': json.dumps(provenance, indent=2) + '\n',
            'README.md': (
                f'# {name}\n\nOriginal CodeBuddy CN Insiders {UPSTREAM_BUILD} deb.\n\n'
                'Nexus verifies the payload checksum and extracts it with dpkg-deb.\n'
                'The original CLI, craft prompt, plugins and tools are retained.\n'
                'Requires a compatible Linux runtime with Electron libraries and Xvfb.\n'
            ),
        }.items():
            content = value.encode()
            info = tarfile.TarInfo('package/' + filename)
            info.size = len(content)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(content))
        info = tarfile.TarInfo('package/payload/codebuddy-insiders.deb')
        info.size = archive.stat().st_size
        info.mode = 0o644
        with archive.open('rb') as source:
            tar.addfile(info, source)
    # Inspect the actual publishable tar, including the entire embedded payload.
    with tarfile.open(output) as tar:
        members = tar.getmembers()
        expected_members = {
            'package/package.json',
            'package/upstream.json',
            'package/README.md',
            'package/payload/codebuddy-insiders.deb',
        }
        if (
            len(members) != len(expected_members)
            or {member.name for member in members} != expected_members
            or any(not member.isfile() for member in members)
        ):
            raise ValueError('npm archive must contain exactly the expected regular files')
        manifest_file = tar.extractfile('package/package.json')
        payload = tar.extractfile('package/payload/codebuddy-insiders.deb')
        if manifest_file is None or payload is None:
            raise ValueError('npm archive is missing its manifest or deb payload')
        actual_manifest = json.load(manifest_file)
        digest = hashlib.sha256()
        for chunk in iter(lambda: payload.read(1024 * 1024), b''):
            digest.update(chunk)
        if actual_manifest != manifest:
            raise ValueError('npm archive manifest differs from the expected package')
        if digest.hexdigest() != expected:
            raise ValueError(f'{arch} packaged deb SHA256 mismatch: {digest.hexdigest()}')
    return {
        'package': name,
        'version': DEFAULT_VERSION,
        'path': str(output),
        'sha256': sha256(output),
        'upstream': provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deb-dir', type=Path, required=True, help='Directory with cb-insiders-linux-{arch}.deb')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--arch', choices=['x64', 'arm64', 'all'], default='all')
    parser.add_argument('--registry', default=os.getenv('CB_INSIDERS_NPM_REGISTRY', 'http://mirrors.tencent.com/npm/'))
    parser.add_argument('--download', action='store_true', help='Download missing original deb artifacts')
    parser.add_argument('--publish', action='store_true', help='Publish the verified packages to the private registry')
    args = parser.parse_args()
    private_npmrc(args.registry, None, None)  # Validate before building anything.
    records = []
    for arch in ['x64', 'arm64'] if args.arch == 'all' else [args.arch]:
        archive = args.deb_dir / f'cb-insiders-linux-{arch}.deb'
        if not archive.is_file() and args.download:
            archive.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(source_url(arch), archive)
        record = build_package(archive, arch, args.output_dir, args.registry)
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.output_dir / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    if args.publish:
        username = os.environ['CB_INSIDERS_NPM_PUBLISH_USERNAME']
        token = os.environ['CB_INSIDERS_NPM_PUBLISH_TOKEN']
        with tempfile.TemporaryDirectory(prefix='cb-insiders-publish-') as tmp:
            npmrc = Path(tmp) / 'npmrc'
            npmrc.write_text(private_npmrc(args.registry, username, token))
            npmrc.chmod(0o600)
            env = {**os.environ, 'NPM_CONFIG_USERCONFIG': str(npmrc)}
            for record in records:
                subprocess.run(
                    [
                        'npm',
                        'publish',
                        record['path'],
                        '--registry',
                        args.registry,
                        '--tag',
                        'insiders',
                        '--ignore-scripts',
                    ],
                    env=env,
                    check=True,
                )


if __name__ == '__main__':
    main()
