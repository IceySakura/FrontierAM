"""Install, configure, and launch WorkBuddy ``wb --acp`` for the CodeBuddy session runtime.

``acp_cli='workbuddy'`` reuses the CodeBuddy session runtime for WorkBuddy's ACP v2 server.
This module holds what differs: where the ``wb`` package comes from, its data directory,
environment, and command line. The Host uploads a linux-x64 package tarball once per sandbox
(:func:`upload_workbuddy_package`); the runtime unpacks it once per version
(:func:`ensure_workbuddy_cli`). The single-turn ``workbuddy`` agent and its 5.x installer are
reused, never changed: their npm path installs ``better-sqlite3``, which 5.11 no longer needs
because it uses ``node:sqlite``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import uuid
import weakref

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nexus.extensions.agents.installed_common import DEFAULT_AGENT_INSTALL_ROOT, find_runtime_binary
from nexus.extensions.agents.workbuddy.installer import (
    DEFAULT_INSTALL_DIR as WORKBUDDY_DEFAULT_INSTALL_DIR,
)
from nexus.extensions.agents.workbuddy.installer import (
    UPLOAD_CHUNK_BYTES,
    iter_file_chunks,
    remote_part_path,
    resolve_wb_js,
)
from nexus.extensions.agents.workbuddy.runtime_setup import _PROTECTED_ENV, build_workbuddy_env, cli_model_flag

from .runtime_setup import _apply_json_patch, _write_models_json


logger = logging.getLogger(__name__)

DEFAULT_WORKBUDDY_ACP_VERSION = '5.11.0'
WORKBUDDY_PREINSTALLED_ROOT: Path = Path('/nix/runtime/workbuddy')
"""Image install root, accepting an extracted package or an npm install, optionally under its version."""
WORKBUDDY_ACP_INSTALL_ROOT = Path(DEFAULT_AGENT_INSTALL_ROOT) / 'workbuddy-acp'
"""Runtime-side root; each package unpacks into ``<root>/<version>[-<sha256 prefix>]/package``."""
REMOTE_PACKAGE_DIR = '/tmp/nexus/workbuddy-acp'
"""Runtime-side directory the Host uploads package parts into."""
WorkBuddyMode = Literal['craft', 'ask', 'plan', 'quick']
WorkBuddyPermissionMode = Literal['default', 'acceptEdits', 'bypassPermissions', 'fullAccess']
DEFAULT_WORKBUDDY_FEATURES: dict[str, Any] = {'ecosystem': True, 'tencentDocsEngine': True}
"""Plugins, skills, and connectors, plus the local Tencent Docs engine the bundled office skills (workbuddy-pptx)
drive documents through, as on WorkBuddy Desktop. ``wb`` leaves every feature off unless this is passed."""
DEFAULT_WORKBUDDY_PERMISSION_MODE: WorkBuddyPermissionMode = 'bypassPermissions'
"""Skips permission prompts, but wb's security review still asks before dangerous commands such as ``rm -rf``."""

# NOTE(wallyshi): 5.11.0 package.json declares engines "^22.16.0 || >=24"; node:sqlite needs it.
_SUPPORTED_NODE_RANGES = (((22, 16, 0), (23, 0, 0)), ((24, 0, 0), None))
_SUPPORTED_NODE_TEXT = '^22.16.0 || >=24'
_NODE_VERSION = re.compile(r'v?(\d+)\.(\d+)\.(\d+)')
_SHA256 = re.compile(r'[0-9a-f]{64}')
_SHA256_PREFIX_CHARS = 12
_HASH_BLOCK_BYTES = 1024 * 1024
_UPLOAD_CHUNK_BYTES = UPLOAD_CHUNK_BYTES
_NODE_VERSION_TIMEOUT = 30.0
_EXTRACT_TIMEOUT = 600.0
_INSTALL_MARKER = '.package-sha256'
_INTERNET_ENVIRONMENT_ENV = 'CODEBUDDY_INTERNET_ENVIRONMENT'
# NOTE(wallyshi): The session runtime always sets these four itself; only a caller override of the
# other protected keys is worth a warning.
_RUNTIME_MANAGED_ENV = frozenset({'HOME', 'PATH', 'IS_SANDBOX', 'CODEBUDDY_API_KEY'})

_INSTALL_LOCK = asyncio.Lock()
_UPLOADS: weakref.WeakKeyDictionary[Any, dict[tuple[str, str], asyncio.Task[list[str]]]] = weakref.WeakKeyDictionary()
_LOCAL_SHA256: dict[tuple[str, int, int], str] = {}


class WorkBuddyAcpOptions(BaseModel):
    """WorkBuddy settings of a session started with ``acp_cli='workbuddy'``.

    Args:
        version (str): Exact ``wb`` package version; the unpacked ``package.json`` must match it.
        cli_path (str | None): Runtime path of a preinstalled CLI entrypoint. None searches
            /nix/runtime/workbuddy before the runtime's install cache.
        package_path (str | None): Host path of the linux-x64 package tarball, uploaded once per
            sandbox. None expects the version to be installed in the runtime image already.
        package_sha256 (str | None): Expected SHA-256 of the tarball. The Host-computed digest is
            sent when this is None, so the runtime always verifies what it unpacks.
        preset (str | None): ``--preset`` default of new sessions; None keeps the CLI default.
        mode (str | None): ``--mode`` default of new sessions; None keeps ``craft``.
        effort (str | None): ``--effort`` default of new sessions.
        features (dict | None): ``--features`` switches. Defaults to ``ecosystem`` and ``tencentDocsEngine``.
            ``None`` or ``{}`` leaves the blank WorkBuddy. A value replaces the default, so
            another switch has to repeat both to keep them.
        fork_mode (str | None): Mode a ``session/fork`` follow-up switches to; ``quick`` answers
            without tools, ``ask`` allows read-only tools, so neither writes files next to the turn
            it asks about. None keeps the source mode.
        permission_mode (str | None): ``permission_mode`` set after ``session/new`` and ``session/fork``.
            Defaults to ``bypassPermissions`` whatever ``skip_permissions`` says. None derives it from
            ``skip_permissions``: ``fullAccess`` when skipping, ``default`` otherwise.
    """

    model_config = ConfigDict(extra='forbid')

    version: str = DEFAULT_WORKBUDDY_ACP_VERSION
    cli_path: str | None = None
    package_path: str | None = None
    package_sha256: str | None = None
    preset: Literal['code', 'work', 'design'] | None = None
    mode: WorkBuddyMode | None = None
    effort: Literal['minimal', 'low', 'medium', 'high', 'xhigh', 'max'] | None = None
    features: dict[str, Any] | None = Field(default_factory=lambda: dict(DEFAULT_WORKBUDDY_FEATURES))
    fork_mode: WorkBuddyMode | None = 'quick'
    permission_mode: WorkBuddyPermissionMode | None = DEFAULT_WORKBUDDY_PERMISSION_MODE

    @field_validator('package_sha256')
    @classmethod
    def _normalize_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        digest = value.strip().lower()
        if not _SHA256.fullmatch(digest):
            raise ValueError(f'package_sha256 must be 64 hex characters, got {value!r}')
        return digest

    def runtime_options(self, package_parts: list[str] | None, *, package_sha256: str | None = None) -> dict[str, Any]:
        """Build the ``workbuddy_options`` the runtime session receives; the Host path stays here.

        Args:
            package_parts (list[str] | None): Runtime paths of the uploaded tarball parts, in order.
            package_sha256 (str | None): Digest to verify instead of ``self.package_sha256``, such as
                the one the Host computed while uploading.
        """
        return {
            'version': self.version,
            'cli_path': self.cli_path,
            'package_parts': list(package_parts) if package_parts else None,
            'package_sha256': package_sha256 or self.package_sha256,
            'preset': self.preset,
            'mode': self.mode,
            'effort': self.effort,
            'features': dict(self.features) if self.features is not None else None,
            'fork_mode': self.fork_mode,
            'permission_mode': self.permission_mode,
        }


async def local_package_sha256(package_path: str) -> str:
    """Return the SHA-256 of a Host-side package, computed once per file size and mtime."""
    path = Path(package_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'WorkBuddy package {package_path!r} is not a file; set workbuddy.package_path')
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    digest = _LOCAL_SHA256.get(key)
    if digest is None:
        digest = await asyncio.to_thread(_sha256_of, [path])
        _LOCAL_SHA256[key] = digest
    return digest


async def upload_workbuddy_package(runtime: Any, options: WorkBuddyAcpOptions) -> list[str] | None:
    """Upload ``options.package_path`` into the sandbox once and return the runtime paths of its parts.

    Concurrent sessions of one sandbox share a single upload; a failed upload is retried by the next
    caller. Socket.IO and HTTP RPC cap one message at 100 MiB, so the tarball goes in slices.

    Returns:
        The part paths in order, or None when no Host package is configured.

    Raises:
        FileNotFoundError: ``package_path`` is not a file.
        RuntimeError: The package digest differs from ``package_sha256``, or a part failed to upload.
    """
    if not options.package_path:
        return None
    local = Path(options.package_path).expanduser().resolve()
    digest = await local_package_sha256(str(local))
    if options.package_sha256 is not None and digest != options.package_sha256:
        raise RuntimeError(
            f'WorkBuddy package {local} has SHA-256 {digest}, expected {options.package_sha256}; '
            'fix workbuddy.package_sha256 or point package_path at the pinned tarball'
        )
    uploads = _UPLOADS.setdefault(runtime, {})
    key = (str(local), digest)
    upload = uploads.get(key)
    if upload is None or (upload.done() and (upload.cancelled() or upload.exception() is not None)):
        destination = f'{REMOTE_PACKAGE_DIR}/wb-{options.version}-{digest[:_SHA256_PREFIX_CHARS]}.tgz'
        upload = uploads[key] = asyncio.ensure_future(_upload_parts(runtime, local, destination))
    return list(await asyncio.shield(upload))


async def _upload_parts(runtime: Any, local: Path, destination: str) -> list[str]:
    # NOTE(wallyshi): The executor models are only needed when a Host actually uploads a package.
    from nexus.runtime.executors.models.file import FileItem

    logger.info('Uploading WorkBuddy package %s (%d bytes) to %s', local, local.stat().st_size, destination)
    part_paths: list[str] = []
    for index, chunk in enumerate(iter_file_chunks(local, _UPLOAD_CHUNK_BYTES)):
        remote = remote_part_path(index, destination)
        response = await runtime.upload_file(
            [FileItem(path=remote, content=base64.b64encode(chunk).decode('ascii'), encoding='base64')],
            overwrite=True,
        )
        for result in getattr(response, 'results', None) or []:
            if not getattr(result, 'success', True):
                raise RuntimeError(f'Failed to upload WorkBuddy package part {remote}: {result.message}')
        part_paths.append(remote)
    if not part_paths:
        raise RuntimeError(f'WorkBuddy package {local} is empty')
    logger.info('Uploaded WorkBuddy package in %d part(s)', len(part_paths))
    return part_paths


async def ensure_workbuddy_cli(options: dict[str, Any], *, node_path: str) -> Path:
    """Return the ``wb.js`` of ``options['version']``, unpacking the uploaded package once if needed.

    An explicit cli_path takes precedence. Otherwise the version is looked up under
    /nix/runtime/workbuddy and the existing install caches before uploaded package parts are
    unpacked.

    Args:
        options (dict[str, Any]): The session's ``workbuddy_options``.
        node_path (str): Node binary that will run ``wb.js``; its version is checked.

    Raises:
        RuntimeError: Node is too old, the package is missing or does not match its digest or
            version, or unpacking failed.
    """
    version = str(options.get('version') or DEFAULT_WORKBUDDY_ACP_VERSION)
    raw_parts = options.get('package_parts') or []
    parts = [Path(part) for part in raw_parts]
    digest = options.get('package_sha256')
    await check_node_version(node_path)
    if options.get('cli_path'):
        entry = Path(options['cli_path']).resolve()
        if not entry.is_file() or _package_version(entry.parent.parent) != version:
            raise RuntimeError(
                f'workbuddy.cli_path {entry} must point to a WorkBuddy {version} entrypoint; '
                'set cli_path to the installed package/bin/wb.js'
            )
        return entry
    async with _INSTALL_LOCK:
        for directory in (WORKBUDDY_PREINSTALLED_ROOT / version, WORKBUDDY_PREINSTALLED_ROOT):
            preinstalled = resolve_wb_js(str(directory), version)
            if preinstalled is None:
                for binary in (
                    directory / 'bin' / 'wb.js',
                    directory / 'node_modules' / '.bin' / 'wb',
                    directory / 'bin' / 'wb',
                ):
                    entry = binary.resolve()
                    if entry.is_file() and _package_version(entry.parent.parent) == version:
                        preinstalled = entry
                        break
            if preinstalled is not None:
                logger.info('Using preinstalled WorkBuddy %s at %s', version, preinstalled)
                return preinstalled
        preinstalled = resolve_wb_js(WORKBUDDY_DEFAULT_INSTALL_DIR, version)
        if preinstalled is not None:
            logger.info('Using preinstalled WorkBuddy %s at %s', version, preinstalled)
            return preinstalled
        install_dir = _install_dir(version, digest)
        installed = resolve_wb_js(str(install_dir), version)
        if installed is not None:
            return installed
        if not parts:
            raise RuntimeError(
                f'WorkBuddy {version} is not installed under {WORKBUDDY_PREINSTALLED_ROOT}, '
                f'{WORKBUDDY_DEFAULT_INSTALL_DIR} or {install_dir} and no package was uploaded; '
                'set workbuddy.cli_path to a preinstalled entrypoint or workbuddy.package_path to '
                f'the Host path of wb-{version}-linux-x64.tgz'
            )
        return await _install_from_parts(parts, install_dir=install_dir, version=version, digest=digest)


async def check_node_version(node_path: str) -> tuple[int, int, int]:
    """Return the version of ``node_path``, or raise when ``wb`` cannot run on it."""
    process = await asyncio.create_subprocess_exec(
        node_path,
        '--version',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=_NODE_VERSION_TIMEOUT)
    text = stdout.decode('utf-8', errors='replace').strip()
    match = _NODE_VERSION.match(text)
    if process.returncode != 0 or match is None:
        raise RuntimeError(
            f'Cannot read the Node.js version of {node_path} (exit {process.returncode}): '
            f'{text or stderr.decode("utf-8", errors="replace").strip()}'
        )
    found = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if not any(low <= found and (high is None or found < high) for low, high in _SUPPORTED_NODE_RANGES):
        raise RuntimeError(
            f'WorkBuddy needs Node.js {_SUPPORTED_NODE_TEXT}, but {node_path} is {text}; '
            'use a runtime image whose node satisfies it'
        )
    return found


def _install_dir(version: str, digest: object) -> Path:
    if isinstance(digest, str) and digest:
        return WORKBUDDY_ACP_INSTALL_ROOT / f'{version}-{digest[:_SHA256_PREFIX_CHARS]}'
    return WORKBUDDY_ACP_INSTALL_ROOT / version


async def _install_from_parts(parts: list[Path], *, install_dir: Path, version: str, digest: object) -> Path:
    missing = [str(part) for part in parts if not part.is_file()]
    if missing:
        raise RuntimeError(
            f'WorkBuddy package parts are missing in the runtime: {missing}; the sandbox lost its uploads, '
            'start a new runtime so the Host uploads them again'
        )
    WORKBUDDY_ACP_INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    archive = WORKBUDDY_ACP_INSTALL_ROOT / f'.download-{token}.tgz'
    staging = WORKBUDDY_ACP_INSTALL_ROOT / f'.staging-{token}'
    try:
        actual = await asyncio.to_thread(_join_parts, parts, archive)
        if isinstance(digest, str) and digest and actual != digest:
            raise RuntimeError(
                f'WorkBuddy package assembled from {len(parts)} part(s) has SHA-256 {actual}, expected {digest}; '
                'the upload was corrupted or package_sha256 is wrong'
            )
        staging.mkdir(parents=True)
        await _extract_archive(archive, staging)
        unpacked = resolve_wb_js(str(staging))
        if unpacked is None:
            raise RuntimeError(f'WorkBuddy package has no package/bin/wb.js; is {parts[0]} a wb linux-x64 tarball?')
        found = _package_version(unpacked.parent.parent)
        if found != version:
            raise RuntimeError(
                f'WorkBuddy package is version {found!r}, expected {version!r}; set workbuddy.version to match'
            )
        (staging / _INSTALL_MARKER).write_text(actual, encoding='utf-8')
        try:
            os.rename(staging, install_dir)
        except OSError as error:
            # NOTE(wallyshi): Another runtime process finished the same install first; keep its copy.
            if resolve_wb_js(str(install_dir), version) is None:
                raise RuntimeError(
                    f'Cannot install WorkBuddy {version} at {install_dir}: {error}; '
                    'remove that directory if it is a broken install'
                ) from error
            logger.info('WorkBuddy %s was installed concurrently at %s', version, install_dir)
    finally:
        archive.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    wb_js = resolve_wb_js(str(install_dir), version)
    if wb_js is None:
        raise RuntimeError(f'WorkBuddy install at {install_dir} has no package/bin/wb.js for {version}')
    logger.info('WorkBuddy %s installed at %s', version, wb_js)
    return wb_js


def _join_parts(parts: list[Path], archive: Path) -> str:
    digest = hashlib.sha256()
    with archive.open('wb') as output:
        for part in parts:
            with part.open('rb') as handle:
                while block := handle.read(_HASH_BLOCK_BYTES):
                    digest.update(block)
                    output.write(block)
    return digest.hexdigest()


def _sha256_of(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open('rb') as handle:
            while block := handle.read(_HASH_BLOCK_BYTES):
                digest.update(block)
    return digest.hexdigest()


async def _extract_archive(archive: Path, destination: Path) -> None:
    tar = find_runtime_binary('tar')
    if tar is None:
        raise RuntimeError('tar is required to unpack the WorkBuddy package but is not on PATH')
    process = await asyncio.create_subprocess_exec(
        tar,
        '-xzf',
        str(archive),
        '-C',
        str(destination),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=_EXTRACT_TIMEOUT)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(f'Unpacking the WorkBuddy package took longer than {_EXTRACT_TIMEOUT:g}s') from None
    if process.returncode != 0:
        raise RuntimeError(
            f'Failed to unpack the WorkBuddy package (tar exit {process.returncode}): '
            f'{stderr.decode("utf-8", errors="replace").strip()}'
        )


def _package_version(package_dir: Path) -> str | None:
    try:
        data = json.loads((package_dir / 'package.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    version = data.get('version') if isinstance(data, dict) else None
    return version if isinstance(version, str) else None


def prepare_workbuddy_config_dir(
    config_dir: Path,
    *,
    proxy_url: str,
    model_id: str,
    api_key: str,
    max_output_tokens: int = 65536,
) -> None:
    """Write the session's ``models.json`` and turn off WorkBuddy's cloud memory.

    ``models.json`` names the sandbox proxy, which ``--model custom-local:<id>`` selects.
    ``app/app-config.json`` is read from ``WORKBUDDY_USER_DATA_DIR`` (see
    :func:`build_workbuddy_acp_env`); without ``generateMemoryEnabled: false`` wb polls the
    cloud memory service with the proxy's placeholder key and logs a 401 every 20 seconds.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_models_json(config_dir, proxy_url, model_id, api_key, max_output_tokens=max_output_tokens)
    _apply_json_patch(str(config_dir / 'app' / 'app-config.json'), {'generateMemoryEnabled': False})
    _link_image_skills(config_dir / 'skills')


def _link_image_skills(skills_dir: Path) -> None:
    # NOTE(wallyshi): wb ignores CODEBUDDY_SESSION_SKILL_DIRS; only <config-dir>/skills reaches the agent.
    image_skills = WORKBUDDY_PREINSTALLED_ROOT / 'skills'
    if not image_skills.is_dir():
        return
    linked = []
    for skill in sorted(image_skills.iterdir()):
        target = skills_dir / skill.name
        if not (skill / 'SKILL.md').is_file() or target.exists() or target.is_symlink():
            continue
        skills_dir.mkdir(parents=True, exist_ok=True)
        target.symlink_to(skill, target_is_directory=True)
        linked.append(skill.name)
    if linked:
        logger.info('Linked image WorkBuddy skills into %s: %s', skills_dir, ', '.join(linked))


def build_workbuddy_acp_env(
    *,
    base_env: dict[str, str],
    home: str,
    config_dir: str,
    proxy_base_url: str,
    api_key: str,
) -> dict[str, str]:
    """Build the ``wb --acp`` environment on top of the session's managed and caller environment.

    The single-turn agent's rules apply: model traffic, ``HOME``, and the data directories are
    pinned to this session, and the CA variables are defaulted. ``api_key`` authenticates web tools;
    the separate ``models.json`` keeps the proxy credential. ``CODEBUDDY_INTERNET_ENVIRONMENT``
    is ignored because WorkBuddy selects its bundled product configuration.
    """
    overridden = sorted(key for key in base_env if key in _PROTECTED_ENV and key not in _RUNTIME_MANAGED_ENV)
    if overridden:
        logger.warning('Ignoring protected WorkBuddy environment overrides: %s', ', '.join(overridden))
    allowed = {key: value for key, value in base_env.items() if key not in _PROTECTED_ENV}
    if _INTERNET_ENVIRONMENT_ENV in allowed:
        logger.warning('Ignoring %s: WorkBuddy uses its bundled product configuration', _INTERNET_ENVIRONMENT_ENV)
        allowed.pop(_INTERNET_ENVIRONMENT_ENV)
    env = build_workbuddy_env(
        home=home,
        install_dir=str(WORKBUDDY_ACP_INSTALL_ROOT),
        proxy_base_url=proxy_base_url,
        api_key=api_key,
        config_dir=config_dir,
        extra_env=allowed,
    )
    _add_image_toolchain(env, Path(home))
    return env


def _add_image_toolchain(env: dict[str, str], home: Path) -> None:
    """Expose the image's skill toolchain (python, node, soffice, ...) behind the workspace's own tools."""
    toolchain = WORKBUDDY_PREINSTALLED_ROOT / 'toolchain'
    if not (toolchain / 'bin').is_dir():
        return
    env['PATH'] = os.pathsep.join(part for part in (env.get('PATH'), str(toolchain / 'bin')) if part)
    node_modules = toolchain / 'node' / 'node_modules'
    if node_modules.is_dir():
        env['NODE_PATH'] = os.pathsep.join(part for part in (env.get('NODE_PATH'), str(node_modules)) if part)
    fonts_conf = toolchain / 'etc' / 'fonts' / 'fonts.conf'
    if fonts_conf.is_file():
        env.setdefault('FONTCONFIG_FILE', str(fonts_conf))
    _append_toolchain_profile(home, toolchain / 'bin', node_modules if node_modules.is_dir() else None)


_PROFILE_MARKER = '# nexus: WorkBuddy image toolchain'


def _append_toolchain_profile(home: Path, bin_dir: Path, node_modules: Path | None) -> None:
    # NOTE(wallyshi): wb's Bash tool takes PATH from a snapshot of $HOME/.profile (.zshrc for zsh), taken after
    # Alpine's /etc/profile resets PATH, and drops NODE_PATH; the environment alone never reaches its commands.
    lines = [_PROFILE_MARKER, f'case ":$PATH:" in *":{bin_dir}:"*) ;; *) export PATH="$PATH:{bin_dir}" ;; esac']
    if node_modules is not None:
        lines.append(
            f'case ":$NODE_PATH:" in *":{node_modules}:"*) ;; '
            f'*) export NODE_PATH="${{NODE_PATH:+$NODE_PATH:}}{node_modules}" ;; esac'
        )
    block = '\n'.join(lines) + '\n'
    home.mkdir(parents=True, exist_ok=True)
    for name in ('.profile', '.zshrc'):
        profile = home / name
        text = profile.read_text(encoding='utf-8') if profile.is_file() else ''
        if _PROFILE_MARKER in text:
            continue
        separator = '\n' if text and not text.endswith('\n') else ''
        profile.write_text(f'{text}{separator}{block}', encoding='utf-8')


async def stop_workbuddy_process_group(pid: int) -> None:
    """Stop the ``wb`` process group ``pid`` and everything it spawned, TERM before KILL."""
    # NOTE(wallyshi): The single-turn agent already owns the grace timings that reap wb's sidecars.
    from nexus.extensions.agents.workbuddy.runtime_setup import _stop_process_group

    await _stop_process_group(pid)


def build_workbuddy_acp_argv(
    node_path: str,
    wb_js: str,
    *,
    model_id: str,
    version: str,
    preset: str | None = None,
    mode: str | None = None,
    effort: str | None = None,
    features: dict[str, Any] | None = None,
    extra_cli_args: Sequence[str] = (),
) -> list[str]:
    """Assemble ``node wb.js --acp``; the CLI flags only set defaults of the sessions it creates."""
    cmd = [node_path, wb_js, '--acp', '--model', cli_model_flag(model_id, version)]
    if preset:
        cmd.extend(['--preset', preset])
    if mode:
        cmd.extend(['--mode', mode])
    if effort:
        cmd.extend(['--effort', effort])
    if features:
        cmd.extend(['--features', json.dumps(features, ensure_ascii=False, separators=(',', ':'))])
    cmd.extend(extra_cli_args)
    return cmd


__all__ = [
    'DEFAULT_WORKBUDDY_ACP_VERSION',
    'REMOTE_PACKAGE_DIR',
    'WORKBUDDY_ACP_INSTALL_ROOT',
    'WORKBUDDY_PREINSTALLED_ROOT',
    'WorkBuddyAcpOptions',
    'build_workbuddy_acp_argv',
    'build_workbuddy_acp_env',
    'check_node_version',
    'ensure_workbuddy_cli',
    'local_package_sha256',
    'prepare_workbuddy_config_dir',
    'stop_workbuddy_process_group',
    'upload_workbuddy_package',
]
