#!/usr/bin/env python3
"""Build a versioned prebuilt DeepSeek Harness npm installation archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile

from pathlib import Path


MANIFEST_VERSION = 1
ARCHITECTURES = {'aarch64': 'arm64', 'amd64': 'x64', 'arm64': 'arm64', 'x86_64': 'x64'}
ARCHIVE_ROOTS = ('package.json', 'package-lock.json', 'node_modules')
EXACT_PACKAGE_VERSION_PATTERN = re.compile(r'^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$')


def exact_package_version(value: str) -> str:
    if not EXACT_PACKAGE_VERSION_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(f'expected an exact package version, got {value!r}')
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dsh-version', type=exact_package_version, required=True)
    parser.add_argument('--acp-version', type=exact_package_version, required=True)
    parser.add_argument('--source-install', type=Path)
    parser.add_argument('--node', default=shutil.which('node') or 'node')
    parser.add_argument('--npm', default=shutil.which('npm') or 'npm')
    parser.add_argument('--registry', default='http://mirrors.tencent.com/npm/')
    parser.add_argument(
        '--minimum-libc-version',
        help='Minimum target libc version; defaults conservatively to the build host version.',
    )
    return parser.parse_args()


def package_version(install: Path, package_name: str) -> str:
    path = install / 'node_modules' / package_name / 'package.json'
    data = json.loads(path.read_text())
    version = data.get('version')
    if not isinstance(version, str):
        raise RuntimeError(f'{path} has no string version')
    return version


def node_major(node: str) -> int:
    raw = subprocess.check_output([node, '--version'], text=True).strip().removeprefix('v')
    try:
        return int(raw.split('.', 1)[0])
    except ValueError as exc:
        raise RuntimeError(f'cannot parse Node.js version: {raw!r}') from exc


def target_tag(node: str) -> tuple[str, str | None]:
    architecture = ARCHITECTURES.get(platform.machine().lower())
    if architecture is None:
        raise RuntimeError(f'unsupported architecture: {platform.machine()}')
    major = node_major(node)
    if platform.system() == 'Linux':
        libc, libc_version = platform.libc_ver()
        libc = libc.lower().replace('gnu libc', 'glibc')
        if libc not in {'glibc', 'musl'}:
            raise RuntimeError(f'unsupported Linux libc: {(libc, libc_version)!r}')
        return f'linux-{architecture}-{libc}-node{major}', libc_version or None
    if platform.system() == 'Darwin':
        return f'macos-{architecture}-node{major}', None
    raise RuntimeError(f'unsupported build platform: {platform.system()}')


def validate_install(install: Path, node: str) -> tuple[str, str]:
    missing = [name for name in ARCHIVE_ROOTS if not (install / name).exists()]
    if missing:
        raise RuntimeError(f'installation is missing archive roots: {missing}')
    dsh_version = package_version(install, '@deepseek-ai/dsh')
    acp_version = package_version(install, '@deepseek-ai/dsh-acp')
    node_pty = install / 'node_modules' / 'node-pty'
    subprocess.run([node, '-e', f'require({str(node_pty)!r})'], cwd=install, check=True)
    return dsh_version, acp_version


def build_install(
    destination: Path,
    *,
    dsh_version: str,
    acp_version: str,
    npm: str,
    registry: str,
) -> None:
    destination.mkdir(parents=True)
    package = {
        'name': 'nexus-dsh-runtime',
        'private': True,
        'dependencies': {
            '@deepseek-ai/dsh': dsh_version,
            '@deepseek-ai/dsh-acp': acp_version,
        },
    }
    (destination / 'package.json').write_text(json.dumps(package, indent=2) + '\n')
    env = os.environ.copy()
    env.update({
        'CI': '1',
        'NO_COLOR': '1',
        'NPM_CONFIG_AUDIT': 'false',
        'NPM_CONFIG_FUND': 'false',
        'NPM_CONFIG_REGISTRY': registry,
        'npm_config_build_from_source': 'true',
    })
    subprocess.run(
        [npm, 'install', '--prefix', str(destination), '--no-audit', '--no-fund', '--registry', registry],
        cwd=destination,
        env=env,
        check=True,
    )
    # node-pty 1.1.0's npm tarball carries a linux-x64 prebuild directory but
    # no pty.node inside it. Its install script checks only for the directory,
    # so force the native build instead of trusting that false positive.
    subprocess.run(
        [
            npm,
            'rebuild',
            'node-pty',
            '--build-from-source=true',
            '--foreground-scripts',
            '--no-audit',
            '--no-fund',
            '--registry',
            registry,
        ],
        cwd=destination,
        env=env,
        check=True,
    )


def normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ''
    info.gname = ''
    info.mtime = 0
    return info


def archive_install(install: Path, archive_path: Path) -> None:
    with archive_path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        with tarfile.open(mode='w|', fileobj=compressed, format=tarfile.PAX_FORMAT) as archive:
            for root_name in ARCHIVE_ROOTS:
                source = install / root_name
                archive.add(source, arcname=root_name, recursive=False, filter=normalized_tar_info)
                if not source.is_dir():
                    continue
                for path in sorted(source.rglob('*'), key=lambda item: item.relative_to(install).as_posix()):
                    archive.add(
                        path,
                        arcname=path.relative_to(install).as_posix(),
                        recursive=False,
                        filter=normalized_tar_info,
                    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> Path:
    target, detected_libc_version = target_tag(args.node)
    with tempfile.TemporaryDirectory(prefix='nexus-dsh-prebuilt-') as temporary:
        if args.source_install is None:
            install = Path(temporary) / 'install'
            build_install(
                install,
                dsh_version=args.dsh_version,
                acp_version=args.acp_version,
                npm=args.npm,
                registry=args.registry,
            )
        else:
            install = args.source_install.resolve()
        dsh_version, acp_version = validate_install(install, args.node)
        expected_versions = (args.dsh_version, args.acp_version)
        if (dsh_version, acp_version) != expected_versions:
            raise RuntimeError(
                f'source installation versions {(dsh_version, acp_version)!r} do not match '
                f'arguments {expected_versions!r}'
            )
        destination = args.output_dir.resolve() / f'dsh-{dsh_version}' / f'acp-{acp_version}' / target
        destination.mkdir(parents=True, exist_ok=True)
        archive_name = f'nexus-dsh-runtime-{dsh_version}-{target}.tar.gz'
        archive_path = destination / archive_name
        archive_install(install, archive_path)
        manifest = {
            'manifest_version': MANIFEST_VERSION,
            'dsh_version': dsh_version,
            'acp_version': acp_version,
            'target': target,
            'archive': archive_name,
            'archive_sha256': sha256(archive_path),
            'archive_size': archive_path.stat().st_size,
            'package_lock_sha256': sha256(install / 'package-lock.json'),
            'minimum_libc_version': args.minimum_libc_version or detected_libc_version,
        }
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        return destination


def main() -> None:
    destination = build(parse_args())
    print(destination)


if __name__ == '__main__':
    main()
