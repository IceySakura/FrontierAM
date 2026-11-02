"""Install an architecture-specific private npm artifact containing the original deb."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import platform
import re
import shlex
import shutil
import tempfile

from pathlib import Path
from urllib.parse import urlsplit

from filelock import FileLock

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    find_runtime_binary,
    resolve_npm_registry,
)


logger = logging.getLogger(__name__)

DEFAULT_VERSION = '5.0.0-40236426.983c1e09'
UPSTREAM_BUILD = '5.0.0.40236426-983c1e09'
UPSTREAM_COMMIT = '983c1e090fc9863cc5b0aa66a361766bac083b2e'
DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/cb-insiders'
PACKAGE_PREFIX = '@tencent/nexus-cb-insiders-linux-'
DEB_SHA256 = {
    'x64': '4b2534b57fd9d3aa3f867c893d5c2691f9a3286085ec67defb2a6233626fc436',
    'arm64': '8ffe79bafc3e1818df449e21b1558147b39197ea35755628d15b74a7b5b4412b',
}
APP_RELATIVE_PATH = Path('usr/share/buddycn-insiders')
_VERSION_RE = re.compile(r'^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$')


def runtime_architecture() -> str:
    if platform.system() != 'Linux':
        raise RuntimeError('cb-insiders supports Linux runtimes only')
    machine = platform.machine().lower()
    arch = {'x86_64': 'x64', 'amd64': 'x64', 'aarch64': 'arm64', 'arm64': 'arm64'}.get(machine)
    if arch is None:
        raise RuntimeError(f'Unsupported cb-insiders architecture: {machine}')
    return arch


def validate_version(version: str) -> str:
    if not _VERSION_RE.fullmatch(version):
        raise ValueError('cb-insiders version must be an exact npm version')
    return version


def source_url(arch: str) -> str:
    if arch not in DEB_SHA256:
        raise ValueError(f'Unsupported cb-insiders architecture: {arch}')
    return (
        f'https://acc-1258344699.cos.ap-guangzhou.myqcloud.com/aiide/insiders/linux-{arch}/'
        f'CodeBuddy-CN-Insiders-linux-{arch}-{UPSTREAM_BUILD}.deb'
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def private_npmrc(registry: str, username: str | None, token: str | None) -> str:
    parsed = urlsplit(registry)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('npm_registry must be an HTTP(S) URL without embedded credentials')
    if any(char in registry for char in '\r\n') or parsed.query or parsed.fragment:
        raise ValueError('npm_registry must be a registry base URL')
    url = registry.rstrip('/') + '/'
    config = f'registry={url}\n@tencent:registry={url}\n'
    if bool(username) != bool(token):
        raise ValueError('npm_username and npm_token must both be set')
    if username and token:
        auth = base64.b64encode(f'{username}:{token}'.encode()).decode('ascii')
        config += f'//{parsed.netloc}{parsed.path.rstrip("/")}/:_auth={auth}\nalways-auth=true\n'
    return config


async def _command(command: list[str], *, timeout: float, env: dict[str, str] | None = None) -> str:
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    result = await run_command(
        RunCommandRequest(command=shlex.join(command), cwd='/tmp', timeout=timeout, extra_env=env or {})
    )
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        raise RuntimeError(result.stderr or result.stdout or f'Command failed: {result.status}')
    return result.stdout or ''


def _validated_app(root: Path, version: str, arch: str) -> Path | None:
    app = root / 'app' / APP_RELATIVE_PATH
    try:
        marker = json.loads((root / 'installed.json').read_text())
        product = json.loads((app / 'resources/app/product.json').read_text())
        valid = marker == {'version': version, 'architecture': arch, 'sha256': DEB_SHA256[arch]}
        valid = valid and product.get('commit') == UPSTREAM_COMMIT
        if valid and all(
            (app / path).is_file()
            for path in (
                'bin/buddycn-insiders',
                'buddycn-insiders',
                'resources/app/extensions/genie/ide-cli/ide-cli.mjs',
            )
        ):
            return app
    except (OSError, ValueError):
        pass
    return None


async def install_cb_insiders(
    install_dir: str = DEFAULT_INSTALL_DIR,
    *,
    version: str = DEFAULT_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    npm_username: str | None = None,
    npm_token: str | None = None,
    package_path: str | None = None,
) -> Path:
    """Install via private npm; package_path is an explicit runtime-local dev fallback."""
    validate_version(version)
    if version != DEFAULT_VERSION:
        raise ValueError(f'No verified deb checksums for cb-insiders {version}')
    arch = runtime_architecture()
    registry = resolve_npm_registry(npm_registry)
    auth = private_npmrc(registry, npm_username, npm_token)
    root = Path(install_dir) / f'{version}-{arch}'
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root / '.install.lock'), timeout=900)
    # FileLock is blocking; polling avoids blocking the Runtime event loop.
    from filelock import Timeout

    for _ in range(3600):
        try:
            lock.acquire(timeout=0)
            break
        except Timeout:
            await asyncio.sleep(0.25)
    else:
        raise RuntimeError('Timed out waiting for the cb-insiders install lock')
    try:
        if existing := _validated_app(root, version, arch):
            return existing
        package_name = PACKAGE_PREFIX + arch
        if package_path:
            archive = Path(package_path)
            if not archive.is_absolute() or not archive.is_file():
                raise ValueError('package_path must be an existing absolute path in the runtime')
        else:
            npm = find_runtime_binary('npm')
            if not npm:
                raise RuntimeError('npm is required to install the cb-insiders private package')
            with tempfile.TemporaryDirectory(prefix='.npm-auth-', dir=root) as tmp:
                npmrc = Path(tmp) / 'npmrc'
                npmrc.write_text(auth, encoding='utf-8')
                npmrc.chmod(0o600)
                env = build_npm_env(npm_registry=registry)
                env['NPM_CONFIG_USERCONFIG'] = str(npmrc)
                logger.info('Installing %s@%s', package_name, version)
                await _command(
                    [
                        npm,
                        'install',
                        '--prefix',
                        str(root / 'npm'),
                        '--no-audit',
                        '--no-fund',
                        '--ignore-scripts',
                        '--no-package-lock',
                        '--registry',
                        registry,
                        f'{package_name}@{version}',
                    ],
                    timeout=900,
                    env=env,
                )
            package = root / 'npm/node_modules' / package_name
            manifest = json.loads((package / 'package.json').read_text())
            if manifest.get('name') != package_name or manifest.get('version') != version:
                raise RuntimeError('cb-insiders npm package name/version mismatch')
            archive = package / 'payload/codebuddy-insiders.deb'
        actual = await asyncio.to_thread(sha256, archive)
        if actual != DEB_SHA256[arch]:
            raise RuntimeError(f'cb-insiders deb SHA256 mismatch for {arch}: {actual}')
        dpkg = find_runtime_binary('dpkg-deb')
        if not dpkg:
            raise RuntimeError('dpkg-deb is required to unpack cb-insiders (use an Ubuntu/Debian runtime)')
        deb_arch = (await _command([dpkg, '-f', str(archive), 'Architecture'], timeout=30)).strip()
        if deb_arch != {'x64': 'amd64', 'arm64': 'arm64'}[arch]:
            raise RuntimeError(f'cb-insiders deb architecture mismatch: {deb_arch}')
        with tempfile.TemporaryDirectory(prefix='.unpack-', dir=root) as tmp:
            staged = Path(tmp) / 'app'
            await _command([dpkg, '-x', str(archive), str(staged)], timeout=180)
            app = staged / APP_RELATIVE_PATH
            product = json.loads((app / 'resources/app/product.json').read_text())
            if product.get('commit') != UPSTREAM_COMMIT:
                raise RuntimeError('cb-insiders upstream build identity mismatch')
            if (root / 'app').exists():
                await asyncio.to_thread(shutil.rmtree, root / 'app')
            staged.rename(root / 'app')
        (root / 'installed.json').write_text(
            json.dumps({'version': version, 'architecture': arch, 'sha256': actual}), encoding='utf-8'
        )
        result = _validated_app(root, version, arch)
        if result is None:
            raise RuntimeError('cb-insiders installation is incomplete')
        return result
    finally:
        lock.release()


async def ensure_system_dependencies(app: Path, *, auto_install: bool = True) -> None:
    """The original IDE CLI uses Electron even in CLI mode."""
    libc, libc_version = platform.libc_ver()
    if libc == 'glibc' and tuple(int(x) for x in libc_version.split('.')[:2]) < (2, 29):
        raise RuntimeError('This cb-insiders build needs glibc >= 2.29; use Ubuntu 20.04 or newer')
    ldd = find_runtime_binary('ldd')
    missing = not find_runtime_binary('Xvfb')
    if ldd:
        output = await _command([ldd, str(app / 'buddycn-insiders')], timeout=30)
        missing = missing or 'not found' in output
    if not missing:
        return
    apt = find_runtime_binary('apt-get')
    if not auto_install or not apt or os.geteuid() != 0:
        raise RuntimeError(
            'cb-insiders needs Xvfb and Electron system libraries; preinstall them or enable auto_install_system_deps on a root Ubuntu runtime'
        )
    env = {'DEBIAN_FRONTEND': 'noninteractive'}
    await _command([apt, 'update', '-qq'], timeout=180, env=env)
    release = Path('/etc/os-release').read_text()
    sound = 'libasound2t64' if 'VERSION_ID="24.' in release else 'libasound2'
    await _command(
        [
            apt,
            'install',
            '-y',
            '-qq',
            'xvfb',
            'xauth',
            'libgtk-3-0',
            'libgbm1',
            'libnss3',
            sound,
            'libxss1',
            'libxkbfile1',
        ],
        timeout=600,
        env=env,
    )
