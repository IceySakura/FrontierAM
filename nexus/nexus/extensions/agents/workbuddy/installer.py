"""Install WorkBuddy CLI from npm, a Generic artifact, or a host tarball.

The Nexus-private ``wb`` pack is ``@tencent/nexus-wb-cli``: a platform-specific
Node package, not a standalone binary. Registry and Generic installs happen
inside the runtime. Host upload is retained only as a development fallback;
Socket.IO / HTTP RPC cap a single message at 100 MiB, so that path splits the
~200 MiB uncompressed tarball and reassembles it here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import shlex
import urllib.parse

from collections.abc import Iterator
from pathlib import Path

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    find_runtime_binary,
    resolve_npm_registry,
)


logger = logging.getLogger(__name__)

DEFAULT_WORKBUDDY_VERSION = 'latest'
NPM_PACKAGE = '@tencent/nexus-wb-cli'
DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/workbuddy'
REMOTE_TARBALL = '/tmp/nexus/workbuddy/wb-linux-x64.tar'
REMOTE_PACKAGE = '/tmp/nexus/workbuddy/workbuddy-cli.tgz'
REMOTE_NPMRC = '/tmp/nexus/workbuddy/npmrc'
SQLITE_DEP = 'better-sqlite3@12.8.0'
# 32 MiB raw → ~43 MiB base64, safely under the 100 MiB runtime message cap.
UPLOAD_CHUNK_BYTES = 32 * 1024 * 1024
# The host-upload fallback names a concrete build; a dist-tag has no tarball.
DEFAULT_TARBALL_NAME = 'wb-5.9.0-linux-x64.tar'


def resolve_tarball_path(explicit: str | None = None) -> Path:
    """Locate the linux-x64 WorkBuddy tarball on the host."""
    candidates: list[Path] = []
    if explicit and explicit.strip():
        candidates.append(Path(explicit).expanduser())
    env_path = os.getenv('WORKBUDDY_TARBALL', '').strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    # wbcli-nexus/ sits two levels above the nexus git root.
    module_dir = Path(__file__).resolve()
    repo_root = module_dir.parents[4]
    parent_root = module_dir.parents[5] if len(module_dir.parents) > 5 else repo_root.parent
    candidates.extend([
        Path.cwd() / DEFAULT_TARBALL_NAME,
        parent_root / DEFAULT_TARBALL_NAME,
        repo_root / DEFAULT_TARBALL_NAME,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        f'WorkBuddy linux-x64 tarball not found. Set WORKBUDDY_TARBALL or tarball_path. Searched: {searched}'
    )


def iter_file_chunks(path: Path, chunk_bytes: int = UPLOAD_CHUNK_BYTES) -> Iterator[bytes]:
    """Yield successive slices of *path* without loading the whole file."""
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return
            yield chunk


def remote_part_path(index: int, dest: str = REMOTE_TARBALL) -> str:
    return f'{dest}.part{index:03d}'


def _package_dirs(install_dir: str) -> tuple[Path, ...]:
    root = Path(install_dir)
    return (
        root / 'node_modules' / Path(*NPM_PACKAGE.split('/')),
        root / 'package',
    )


def _read_package_version(package_dir: Path) -> str | None:
    package_json = package_dir / 'package.json'
    if not package_json.is_file():
        return None
    try:
        data = json.loads(package_json.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    version = data.get('version')
    return version if isinstance(version, str) else None


def pinned_version(version: str) -> str | None:
    """``version`` when it names one release; None for a dist-tag such as ``latest``.

    A dist-tag resolves at install time, so it can neither reuse a cached install
    nor be compared with ``package.json``.
    """
    return version if version[:1].isdigit() else None


def resolve_wb_js(install_dir: str, expected_version: str | None = None) -> Path | None:
    """Return the entrypoint from an npm install or legacy extracted tarball."""
    for package_dir in _package_dirs(install_dir):
        path = package_dir / 'bin' / 'wb.js'
        if path.is_file() and (expected_version is None or _read_package_version(package_dir) == expected_version):
            return path
    return None


def sqlite_addon_present(install_dir: str) -> bool:
    """True when ``better-sqlite3`` is available to either install layout."""
    root = Path(install_dir)
    return any(
        path.is_dir()
        for path in (
            root / 'node_modules' / 'better-sqlite3',
            root / 'package' / 'node_modules' / 'better-sqlite3',
        )
    )


def get_installed_version(install_dir: str) -> str | None:
    for package_dir in _package_dirs(install_dir):
        if (package_dir / 'bin' / 'wb.js').is_file():
            version = _read_package_version(package_dir)
            if version is not None:
                return version
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_package_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('WorkBuddy package_url must be an HTTP(S) URL without embedded credentials')


def build_private_npmrc(registry: str, username: str, token: str) -> str:
    """Build npm Basic Auth config without returning the raw token in logs."""
    if not token.strip() or not (username or '').strip():
        raise ValueError('npm username and token are required for private WorkBuddy installs')
    parsed = urllib.parse.urlsplit(registry)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('WorkBuddy npm_registry must be an HTTP(S) URL without embedded credentials')
    registry_url = registry.rstrip('/') + '/'
    registry_path = parsed.path.rstrip('/') + '/'
    auth_key = f'//{parsed.netloc}{registry_path}'
    encoded = base64.b64encode(f'{username.strip()}:{token.strip()}'.encode()).decode('ascii')
    return f'registry={registry_url}\n@tencent:registry={registry_url}\n{auth_key}:_auth={encoded}\nalways-auth=true\n'


def _write_npmrc(registry: str, token: str, username: str | None = None) -> Path:
    """Create the short-lived npm auth file used only by the install command."""
    npmrc = Path(REMOTE_NPMRC)
    npmrc.parent.mkdir(parents=True, exist_ok=True)
    npmrc.write_text(build_private_npmrc(registry, username or '', token), encoding='utf-8')
    npmrc.chmod(0o600)
    return npmrc


async def install_workbuddy_via_npm(
    install_dir: str,
    *,
    version: str = DEFAULT_WORKBUDDY_VERSION,
    package_url: str | None = None,
    package_sha256: str | None = None,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    npm_token: str | None = None,
    npm_username: str | None = None,
) -> Path:
    """Install the package from a private npm registry or Generic artifact URL."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    pinned = pinned_version(version)
    existing = resolve_wb_js(install_dir, pinned) if pinned else None
    if package_url is None and existing is not None and sqlite_addon_present(install_dir):
        logger.debug('WorkBuddy already installed at %s (version %s)', existing, version)
        return existing

    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)
    registry = resolve_npm_registry(npm_registry)
    package_spec = f'{NPM_PACKAGE}@{version}'

    if package_url is None and (not npm_username or not npm_token):
        raise ValueError('WORKBUDDY_NPM_USERNAME and WORKBUDDY_NPM_TOKEN are required for private npm installs')

    if package_url:
        _validate_package_url(package_url)
        archive = Path(REMOTE_PACKAGE)
        archive.parent.mkdir(parents=True, exist_ok=True)
        curl = find_runtime_binary('curl') or 'curl'
        download = (
            f'{shlex.quote(curl)} -fL --retry 3 --connect-timeout 15 '
            f'--max-time 600 -o {shlex.quote(str(archive))} {shlex.quote(package_url)}'
        )
        logger.debug('Downloading WorkBuddy %s from Generic artifact storage', version)
        result = await run_command(RunCommandRequest(command=download, cwd='/tmp', timeout=900))
        if result.status != CommandRunStatus.Finished or (result.return_code or 0) != 0:
            raise RuntimeError(
                f'Failed to download WorkBuddy package: {result.stderr or result.stdout or result.status}'
            )
        actual_sha256 = _sha256(archive)
        if package_sha256 and actual_sha256.lower() != package_sha256.lower():
            raise RuntimeError(
                f'WorkBuddy package SHA256 mismatch: expected {package_sha256.lower()}, got {actual_sha256}'
            )
        if not package_sha256:
            logger.warning('WorkBuddy package_url has no package_sha256; downloaded bytes were not pinned')
        package_spec = str(archive)

    npm_executable = find_runtime_binary('npm') or 'npm'
    npm_install = (
        f'{shlex.quote(npm_executable)} install --prefix {shlex.quote(str(install_path))} '
        f'--no-audit --no-fund --strict-ssl=false --registry {shlex.quote(registry)}'
    )
    command = f'{npm_install} {shlex.quote(package_spec)}'
    source = package_url or f'{NPM_PACKAGE}@{version} via {registry}'
    logger.debug('Installing WorkBuddy from %s', source)
    npmrc: Path | None = None
    npm_env = build_npm_env(npm_registry=registry)
    if npm_token:
        npmrc = _write_npmrc(registry, npm_token, npm_username)
        npm_env.update({
            'NPM_CONFIG_USERCONFIG': str(npmrc),
            'npm_config_userconfig': str(npmrc),
        })
    try:
        result = await run_command(
            RunCommandRequest(
                command=command,
                cwd=str(install_path),
                timeout=1800,
                extra_env=npm_env,
            )
        )
    finally:
        if npmrc is not None:
            npmrc.unlink(missing_ok=True)
    if result.status != CommandRunStatus.Finished or (result.return_code or 0) != 0:
        raise RuntimeError(f'Failed to install WorkBuddy: {result.stderr or result.stdout or result.status}')

    # 5.9.0 no longer vendors better-sqlite3 (5.7.0 does); add it at the prefix.
    if not sqlite_addon_present(install_dir):
        logger.info('Installing %s into %s via %s', SQLITE_DEP, install_path, registry)
        result = await run_command(
            RunCommandRequest(
                command=f'{npm_install} {shlex.quote(SQLITE_DEP)}',
                cwd=str(install_path),
                timeout=1800,
                extra_env=npm_env,
            )
        )
        if result.status != CommandRunStatus.Finished or (result.return_code or 0) != 0:
            raise RuntimeError(f'Failed to install {SQLITE_DEP}: {result.stderr or result.stdout or result.status}')

    wb_js = resolve_wb_js(install_dir, pinned)
    if wb_js is None:
        raise RuntimeError(f'WorkBuddy npm install succeeded but {NPM_PACKAGE}@{version} has no bin/wb.js')
    if not sqlite_addon_present(install_dir):
        raise RuntimeError('WorkBuddy npm install succeeded but better-sqlite3 is missing')
    wb_js.chmod(wb_js.stat().st_mode | 0o111)
    if pinned:
        logger.debug('WorkBuddy installed at %s (version %s)', wb_js, version)
    else:
        # The only record of which release a dist-tag resolved to.
        logger.info(
            'WorkBuddy installed at %s (version %s, requested %s)', wb_js, get_installed_version(install_dir), version
        )
    return wb_js


def build_assemble_command(part_paths: list[str], assembled_tar: str, install_dir: str) -> str:
    """Build the shell command that concatenates parts and extracts the tarball."""
    if not part_paths:
        raise ValueError('part_paths must not be empty')
    quoted_parts = ' '.join(shlex.quote(part) for part in part_paths)
    return (
        f'mkdir -p {shlex.quote(str(Path(assembled_tar).parent))} {shlex.quote(install_dir)} && '
        f'cat {quoted_parts} > {shlex.quote(assembled_tar)} && '
        f'tar -xf {shlex.quote(assembled_tar)} -C {shlex.quote(install_dir)}'
    )


async def install_workbuddy_from_parts(
    install_dir: str,
    part_paths: list[str],
    *,
    assembled_tar: str = REMOTE_TARBALL,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
) -> Path:
    """Reassemble the uploaded tarball, extract it, and install better-sqlite3."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)

    wb_js = resolve_wb_js(install_dir)
    if wb_js is None:
        assemble = build_assemble_command(part_paths, assembled_tar, install_dir)
        logger.debug('Assembling WorkBuddy tarball from %d parts', len(part_paths))
        result = await run_command(RunCommandRequest(command=assemble, cwd='/', timeout=600))
        if result.status != CommandRunStatus.Finished or (result.return_code or 0) != 0:
            raise RuntimeError(
                f'Failed to extract WorkBuddy tarball: {result.stderr or result.stdout or result.status}'
            )
        wb_js = resolve_wb_js(install_dir)
        if wb_js is None:
            raise RuntimeError(f'WorkBuddy extract succeeded but package/bin/wb.js is missing under {install_dir}')
        wb_js.chmod(wb_js.stat().st_mode | 0o111)
    else:
        logger.debug('WorkBuddy already extracted at %s (version %s)', wb_js, get_installed_version(install_dir))

    if not sqlite_addon_present(install_dir):
        package_dir = wb_js.parent.parent
        npm_executable = find_runtime_binary('npm') or 'npm'
        registry = resolve_npm_registry(npm_registry)
        npm_cmd = (
            f'{shlex.quote(npm_executable)} install --no-audit --no-fund --strict-ssl=false '
            f'--registry {shlex.quote(registry)} {shlex.quote(SQLITE_DEP)}'
        )
        logger.debug('Installing %s into %s via %s', SQLITE_DEP, package_dir, registry)
        npm_result = await run_command(
            RunCommandRequest(
                command=npm_cmd,
                cwd=str(package_dir),
                timeout=1800,
                extra_env=build_npm_env(npm_registry=registry),
            )
        )
        if npm_result.status != CommandRunStatus.Finished or (npm_result.return_code or 0) != 0:
            raise RuntimeError(
                f'Failed to install {SQLITE_DEP} for WorkBuddy: '
                f'{npm_result.stderr or npm_result.stdout or npm_result.status}'
            )
    else:
        logger.debug('better-sqlite3 already present under %s', install_dir)

    logger.debug('WorkBuddy installed at %s (version %s)', wb_js, get_installed_version(install_dir))
    return wb_js


async def install_workbuddy(
    install_dir: str = DEFAULT_INSTALL_DIR,
    *,
    version: str = DEFAULT_WORKBUDDY_VERSION,
    package_url: str | None = None,
    package_sha256: str | None = None,
    remote_part_paths: list[str] | None = None,
    assembled_tar: str = REMOTE_TARBALL,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    npm_token: str | None = None,
    npm_username: str | None = None,
) -> Path:
    """Install by priority: cached exact version, Generic/npm, host upload fallback.

    A configured Generic URL is immutable and preferred over registry lookup.
    Without a URL, omitted host parts mean that the private npm registry is the
    source.  Supplying host parts explicitly selects the legacy development path.
    """
    pinned = pinned_version(version)
    existing = resolve_wb_js(install_dir, pinned) if pinned else None
    if package_url is None and existing is not None and sqlite_addon_present(install_dir):
        logger.debug('WorkBuddy already installed at %s (version %s)', existing, version)
        return existing
    if package_url or not remote_part_paths:
        return await install_workbuddy_via_npm(
            install_dir,
            version=version,
            package_url=package_url,
            package_sha256=package_sha256,
            npm_registry=npm_registry,
            npm_token=npm_token,
            npm_username=npm_username,
        )
    wb_js = await install_workbuddy_from_parts(
        install_dir,
        remote_part_paths,
        assembled_tar=assembled_tar,
        npm_registry=npm_registry,
    )
    installed_version = get_installed_version(install_dir)
    if pinned and installed_version != pinned:
        raise RuntimeError(f'WorkBuddy host tarball version is {installed_version!r}, expected {version!r}')
    return wb_js


__all__ = [
    'DEFAULT_INSTALL_DIR',
    'DEFAULT_TARBALL_NAME',
    'DEFAULT_WORKBUDDY_VERSION',
    'NPM_PACKAGE',
    'REMOTE_PACKAGE',
    'REMOTE_NPMRC',
    'REMOTE_TARBALL',
    'SQLITE_DEP',
    'UPLOAD_CHUNK_BYTES',
    'build_assemble_command',
    'build_private_npmrc',
    'get_installed_version',
    'install_workbuddy',
    'install_workbuddy_from_parts',
    'install_workbuddy_via_npm',
    'iter_file_chunks',
    'pinned_version',
    'remote_part_path',
    'resolve_tarball_path',
    'resolve_wb_js',
    'sqlite_addon_present',
]
