"""Cline CLI installer.

Two-phase install to avoid npm's slow dependency resolution:

Phase 1 – Tarball download (fast, ~5 s):
  Download the cline npm tarball (a ~17 KB launcher; the bulk lives in its
  dependencies) and extract it into ``node_modules/cline/``.  This skips npm's
  registry-metadata round-trips for the package itself.

Phase 2 – Dependency install (npm, ~1–2 min):
  Run ``npm install --ignore-scripts`` from *inside* the extracted cline
  directory.  Because the package is already present locally, npm only needs
  to install its dependencies.  A per-run ``.npmrc`` forces all scoped
  packages (``@jrichman``, ``@vscode``, ``@agentclientprotocol``) to use
  the same registry, preventing fallback to the slow official registry for
  packages like ``@jrichman/ink``.

Phase 3 – ripgrep binary:
  Provide the ripgrep binary via system copy, apt-get, or CDN download
  (same multi-strategy approach as before).

Falls back to ``npm install cline`` if Phase 1 fails.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import stat

from pathlib import Path

from nexus.extensions.agents.installed_common import (
    build_npm_env,
    find_runtime_binary,
)


logger = logging.getLogger(__name__)


# Pinned for SWE-bench experiment reproducibility.
#
# Do not go back to 3.0.3: it encodes --reasoning-effort as Anthropic-style
# `thinking` and `effort` request fields even on an openai provider, and
# chat.completions.create has an explicit signature that rejects both.
#
# 3.0.4+ splits cline into @cline/* sub-packages that transitively pull
# string-width@8, whose ES2024 regex `v` flag needs Node 20+; the sandbox
# runtime ships nodejs_22 (deployment/runtime/flake.nix), so that is fine.
# Also update SCAFFOLD_VERSION_PIN in ceph_configs/swe_generator_nj.py when
# bumping this.
DEFAULT_CLINE_VERSION = '3.0.62'
NPM_PACKAGE = 'cline'

# npmmirror (Taobao) mirrors all of npmjs.org, including scoped packages.
_CLINE_NPM_REGISTRY = 'https://registry.npmmirror.com'
_CLINE_CDN_BASE = 'https://cdn.npmmirror.com/packages'

# ripgrep prebuilt release matching @vscode/ripgrep's expected binary.
_RG_VERSION = 'v14.1.0'
_RG_ARCHIVE_NAME = f'ripgrep-{_RG_VERSION}-x86_64-unknown-linux-musl.tar.gz'
_RG_GITHUB_PATH = f'microsoft/ripgrep-prebuilt/releases/download/{_RG_VERSION}/{_RG_ARCHIVE_NAME}'
_RG_DOWNLOAD_URLS = [
    # GitHub proxy mirrors (CN-accessible).
    f'https://mirror.ghproxy.com/https://github.com/{_RG_GITHUB_PATH}',
    f'https://ghproxy.com/https://github.com/{_RG_GITHUB_PATH}',
    f'https://kkgithub.com/{_RG_GITHUB_PATH}',
    f'https://download.fastgit.org/{_RG_GITHUB_PATH}',
    # Direct GitHub releases CDN.
    f'https://github.com/{_RG_GITHUB_PATH}',
]

# .npmrc content that forces all scoped registries used by cline to the Tencent
# internal mirror.  All cline dependencies are present there and the internal
# mirror is significantly faster inside the Gongfeng sandbox than external CDNs.
# Without explicit scope overrides npm may fall back to the slow official registry.
_TENCENT_REGISTRY = 'http://mirrors.tencent.com/npm/'
_NPMRC_CONTENT = f"""\
registry={_TENCENT_REGISTRY}
@jrichman:registry={_TENCENT_REGISTRY}
@vscode:registry={_TENCENT_REGISTRY}
@agentclientprotocol:registry={_TENCENT_REGISTRY}
strict-ssl=false
fund=false
audit=false
update-notifier=false
legacy-peer-deps=true
"""

# npm failures worth another attempt. ETARGET dominates: the mirror is load balanced and
# a node whose packument has not caught up yet reports a freshly published version as
# nonexistent, so the same install succeeds against a sibling node seconds later.
_TRANSIENT_NPM_MARKERS = (
    'ETARGET',
    'ENOTFOUND',
    'ECONNRESET',
    'ETIMEDOUT',
    'EAI_AGAIN',
    'socket hang up',
    '429 Too Many Requests',
)


def _tarball_urls(version: str) -> list[str]:
    """Candidate tarball URLs for the cline package, internal mirror first.

    The sandbox routes external HTTPS through an egress proxy that answers 403 for
    cdn.npmmirror.com, so that host is only useful where the public internet is
    reachable directly. Note the two use different layouts: the registry serves
    ``<name>/-/<name>-<version>.tgz``, the CDN ``<name>/<version>/<name>-<version>.tgz``.
    """
    return [
        f'{_TENCENT_REGISTRY.rstrip("/")}/{NPM_PACKAGE}/-/{NPM_PACKAGE}-{version}.tgz',
        f'{_CLINE_CDN_BASE}/{NPM_PACKAGE}/{version}/{NPM_PACKAGE}-{version}.tgz',
    ]


def get_installed_version(install_dir: str) -> str | None:
    """Return the installed Cline CLI version if present."""
    package_json = Path(install_dir) / 'node_modules' / 'cline' / 'package.json'
    if not package_json.exists():
        return None
    try:
        data = json.loads(package_json.read_text())
    except Exception:
        return None
    version = data.get('version')
    return version if isinstance(version, str) else None


def _resolve_cline_package_entry(cline_pkg_dir: Path) -> Path | None:
    """Return the entry script shipped inside the cline package directory.

    Resolved from the package's own ``bin`` field, which is what npm uses to
    create ``.bin`` links. Upstream moves this between releases (3.0.x ships
    ``bin/cline``, pre-3.0 shipped the ``dist/cli.mjs`` ESM bundle), so reading
    the manifest tracks those moves instead of needing a new hardcoded guess
    each time. The literal paths remain only as a fallback for a malformed or
    missing manifest.
    """
    declared: list[Path] = []
    try:
        meta = json.loads((cline_pkg_dir / 'package.json').read_text())
    except Exception:
        meta = {}
    spec = meta.get('bin')
    rel = spec if isinstance(spec, str) else spec.get(NPM_PACKAGE) if isinstance(spec, dict) else None
    if isinstance(rel, str) and rel:
        declared.append(cline_pkg_dir / rel)

    for candidate in (*declared, cline_pkg_dir / 'bin' / 'cline', cline_pkg_dir / 'dist' / 'cli.mjs'):
        if candidate.exists():
            return candidate
    return None


def _resolve_cline_executable(install_dir: str) -> Path | None:
    """Return the Cline CLI executable path, if present.

    Prefers the entry script inside the package itself: npm does not bin-link
    cline under ``--prefix`` installs, so ``.bin/cline`` is normally absent, and
    when it does exist it is only a symlink to that same script. Going straight
    to the real file also sidesteps a stale link left by an earlier install.
    """
    base = Path(install_dir) / 'node_modules'
    entry = _resolve_cline_package_entry(base / 'cline')
    if entry is not None:
        return entry
    bin_link = base / '.bin' / 'cline'
    return bin_link if bin_link.exists() else None


def _rg_bin_path(install_dir: str) -> Path:
    """Return the expected ripgrep binary path inside @vscode/ripgrep."""
    return Path(install_dir) / 'node_modules' / '@vscode' / 'ripgrep' / 'bin' / 'rg'


async def _fetch_json(url: str, timeout: int = 30) -> dict | None:
    """Download and parse a JSON response via curl."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    curl = find_runtime_binary('curl') or 'curl'
    result = await run_command(
        RunCommandRequest(
            command=f'{shlex.quote(curl)} -fsSL --connect-timeout 15 {shlex.quote(url)}',
            cwd='/tmp',
            timeout=timeout,
        )
    )
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        return None
    try:
        return json.loads(result.stdout or '')
    except Exception:
        return None


async def _download_and_extract_tarball(
    url: str,
    dest_dir: Path,
    timeout: int = 120,
) -> bool:
    """Download a .tgz and extract it into dest_dir, stripping the top-level 'package/' prefix."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    dest_dir.mkdir(parents=True, exist_ok=True)
    curl = find_runtime_binary('curl') or 'curl'
    tar = find_runtime_binary('tar') or 'tar'

    cmd = (
        f'{shlex.quote(curl)} -fsSL --connect-timeout 15 {shlex.quote(url)} '
        f'| {shlex.quote(tar)} xz --strip-components=1 -C {shlex.quote(str(dest_dir))}'
    )
    result = await run_command(RunCommandRequest(command=cmd, cwd='/tmp', timeout=timeout))
    if result.status == CommandRunStatus.Finished and result.return_code == 0:
        return True
    # curl's own message is the only thing that distinguishes an unreachable host from a
    # proxy refusing the URL; without it a failure here is indistinguishable from any other.
    logger.warning(
        'Tarball download failed (status=%s rc=%s): %s',
        result.status,
        result.return_code,
        (result.stderr or result.stdout or '').strip()[:300] or '<no output>',
    )
    return False


async def _run_npm_install(command: str, cwd: str, registry: str, attempts: int = 3) -> tuple[bool, str]:
    """Run an npm install, retrying while the failure looks transient.

    Returns (ok, output) where output is whatever npm said on the last attempt.
    """
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    output = ''
    for attempt in range(1, attempts + 1):
        result = await run_command(
            RunCommandRequest(
                command=command,
                cwd=cwd,
                timeout=1800,
                extra_env=build_npm_env(npm_registry=registry),
            )
        )
        if result.status == CommandRunStatus.Finished and result.return_code == 0:
            return True, result.stdout or ''

        output = (result.stderr or result.stdout or str(result.status)).strip()
        if attempt == attempts or not any(marker in output for marker in _TRANSIENT_NPM_MARKERS):
            return False, output

        delay = 10 * attempt
        logger.warning(
            'npm install failed transiently (attempt %d/%d), retrying in %ds: %s',
            attempt,
            attempts,
            delay,
            output[:300],
        )
        await asyncio.sleep(delay)

    return False, output


async def _install_cline_two_phase(install_dir: str, version: str = DEFAULT_CLINE_VERSION) -> Path | None:
    """Two-phase install: tarball download + local npm install."""

    install_path = Path(install_dir)

    # Phase 1: resolve version and download the cline tarball.
    if version == 'latest':
        meta = await _fetch_json(f'{_CLINE_NPM_REGISTRY}/cline/latest', timeout=30)
        if not meta:
            logger.warning('Could not fetch cline metadata from npmmirror')
            return None
        version = meta.get('version', '')
        if not version:
            return None

    cline_pkg_dir = install_path / 'node_modules' / 'cline'
    if not (cline_pkg_dir / 'package.json').exists():
        for tarball_url in _tarball_urls(version):
            logger.debug(f'Downloading cline {version} tarball from {tarball_url}')
            if await _download_and_extract_tarball(tarball_url, cline_pkg_dir, timeout=120):
                logger.debug(f'Extracted cline {version} into {cline_pkg_dir}')
                break
        else:
            logger.warning(f'No source served the cline {version} tarball')
            return None

    # Phase 2: install cline's dependencies using npm from within the package dir.
    # Write .npmrc to force all scoped registries to npmmirror.
    npmrc_path = cline_pkg_dir / '.npmrc'
    npmrc_path.write_text(_NPMRC_CONTENT)
    logger.debug(f'Wrote .npmrc to {npmrc_path}')

    npm_executable = find_runtime_binary('npm') or 'npm'
    # Install into the parent node_modules (install_path/node_modules/) so
    # Node.js module resolution finds them when running the cline entry script.
    # --legacy-peer-deps: skip the slow peer-dependency algorithm.
    dep_install_cmd = (
        f'{shlex.quote(npm_executable)} install '
        f'--prefix {shlex.quote(str(install_path))} '
        f'--no-audit --no-fund --strict-ssl=false --ignore-scripts '
        f'--legacy-peer-deps '
        f'--registry {shlex.quote(_TENCENT_REGISTRY)}'
    )
    logger.debug(f'Installing cline dependencies: {dep_install_cmd}')

    ok, output = await _run_npm_install(dep_install_cmd, str(cline_pkg_dir), _TENCENT_REGISTRY)
    if not ok:
        logger.warning(f'npm dep install failed: {output[:300]}')
        return None

    # Phase 2b: ensure the .bin/cline symlink exists. The tarball was unpacked by
    # hand, so npm never ran its bin-linking step for the cline package itself.
    bin_dir = install_path / 'node_modules' / '.bin'
    bin_dir.mkdir(parents=True, exist_ok=True)
    cline_bin_link = bin_dir / 'cline'
    cli_target = _resolve_cline_package_entry(cline_pkg_dir)

    if cli_target is None:
        logger.warning(f'No cline entry script under {cline_pkg_dir} (looked for bin/cline and dist/cli.mjs)')
        return None

    if not cline_bin_link.exists() and not cline_bin_link.is_symlink():
        cline_bin_link.symlink_to(cli_target)
    elif cline_bin_link.is_symlink() and not cline_bin_link.exists():
        cline_bin_link.unlink()
        cline_bin_link.symlink_to(cli_target)

    cli_target.chmod(cli_target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    logger.debug(f'Cline {version} installed at {cline_bin_link}')
    return cline_bin_link


async def _ensure_ripgrep_binary(install_dir: str) -> None:
    """Place a ripgrep binary where @vscode/ripgrep expects it.

    Tries in order:
      1. Copy the system ``rg`` binary (if available on PATH or in the Nix runtime).
      2. Install via ``apt-get install ripgrep`` if apt is available.
      3. Download from a GitHub proxy mirror (mirror.ghproxy.com / ghproxy.com).
      4. Download directly from the GitHub releases CDN.
    """
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    rg_bin = _rg_bin_path(install_dir)
    if rg_bin.exists():
        return

    rg_bin.parent.mkdir(parents=True, exist_ok=True)

    # Strategy 1: use system ripgrep.
    system_rg = find_runtime_binary('rg') or find_runtime_binary('ripgrep')
    if system_rg:
        import shutil

        shutil.copy2(system_rg, rg_bin)
        rg_bin.chmod(rg_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        logger.debug(f'Copied system ripgrep from {system_rg} to {rg_bin}')
        return

    # Strategy 2: apt-get install ripgrep.
    apt = find_runtime_binary('apt-get') or find_runtime_binary('apt')
    if apt:
        apt_result = await run_command(
            RunCommandRequest(
                command=f'{shlex.quote(apt)} install -y ripgrep',
                cwd='/tmp',
                timeout=120,
                extra_env={'DEBIAN_FRONTEND': 'noninteractive'},
            )
        )
        if apt_result.status == CommandRunStatus.Finished and apt_result.return_code == 0:
            system_rg = find_runtime_binary('rg')
            if system_rg:
                import shutil

                shutil.copy2(system_rg, rg_bin)
                rg_bin.chmod(rg_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
                logger.debug(f'Installed ripgrep via apt and copied to {rg_bin}')
                return

    # Strategy 3 & 4: download prebuilt binary.
    curl = find_runtime_binary('curl') or 'curl'
    archive_path = rg_bin.parent / 'rg.tar.gz'

    for url in _RG_DOWNLOAD_URLS:
        logger.debug(f'Downloading ripgrep from {url}')
        download_cmd = (
            f'{shlex.quote(curl)} -fsSL --connect-timeout 15 {shlex.quote(url)} -o {shlex.quote(str(archive_path))}'
        )
        dl_result = await run_command(RunCommandRequest(command=download_cmd, cwd=str(rg_bin.parent), timeout=60))
        if dl_result.status != CommandRunStatus.Finished or dl_result.return_code != 0 or not archive_path.exists():
            archive_path.unlink(missing_ok=True)
            logger.warning(f'Download failed from {url}')
            continue

        tar = find_runtime_binary('tar') or 'tar'
        extract_cmd = f'{shlex.quote(tar)} xzf {shlex.quote(str(archive_path))} -C {shlex.quote(str(rg_bin.parent))}'
        extract_result = await run_command(RunCommandRequest(command=extract_cmd, cwd=str(rg_bin.parent), timeout=30))
        archive_path.unlink(missing_ok=True)

        if extract_result.status != CommandRunStatus.Finished or extract_result.return_code != 0:
            continue

        extracted_rg = rg_bin.parent / 'rg'
        if not extracted_rg.exists():
            for candidate in rg_bin.parent.rglob('rg'):
                if candidate.is_file():
                    extracted_rg = candidate
                    break

        if extracted_rg.exists():
            extracted_rg.rename(rg_bin)
            rg_bin.chmod(rg_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            logger.debug(f'ripgrep binary installed at {rg_bin} (from {url})')
            return

    logger.warning('Could not obtain ripgrep binary; cline file-search may not work')


async def _find_global_cline_js() -> Path | None:
    """Return the dist/cli.mjs entry point of a globally installed cline, if found.

    The nexus-runtime Docker image pre-installs cline at a known location:
    /usr/local/lib/node_modules/cline  (Alpine npm --prefix /usr/local install)

    Falls back to checking the output of ``npm root -g`` so it also works if
    cline was installed via the default Nix npm global prefix.

    When found, we use the pre-installed cline to skip a slow per-task npm install.
    """
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    # Check the known Docker-image path first (avoids an npm subprocess).
    known_paths = [
        Path('/usr/local/lib/node_modules/cline/dist/cli.mjs'),
    ]
    for p in known_paths:
        if p.exists():
            logger.debug(f'Found pre-installed cline at {p}')
            return p

    # Fallback: ask npm where the global root is.
    npm_executable = find_runtime_binary('npm') or 'npm'
    result = await run_command(
        RunCommandRequest(command=f'{shlex.quote(npm_executable)} root -g', cwd='/tmp', timeout=15)
    )
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        return None

    global_root = (result.stdout or '').strip()
    if not global_root:
        return None

    cli_js = Path(global_root) / 'cline' / 'dist' / 'cli.mjs'
    if cli_js.exists():
        logger.debug(f'Found globally installed cline at {cli_js}')
        return cli_js
    return None


async def install_cline_via_npm(
    install_dir: str,
    version: str = DEFAULT_CLINE_VERSION,
    npm_registry: str | None = None,
) -> Path:
    """Install Cline CLI and return the executable path.

    Priority:
    1. Global install in the Docker runtime image (fastest — no install needed).
    2. Previously installed local cline binary (cache hit).
    3. Two-phase: tarball download + ``npm install`` of deps.
    4. Fallback: direct ``npm install cline``.
    """
    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)

    # When the caller pins an explicit version (version != 'latest'), we must NOT
    # silently fall through to the Docker-image global install — it ignores the
    # version arg and may carry an incompatible cline (e.g. 3.0.4+ pulls
    # @cline/* sub-packages that transitively require string-width@8, which uses
    # ES2024 regex `v` flag and crashes on Node 18).  In that case, only accept
    # the global install if its package.json version actually matches the pin.

    def _version_matches(pkg_json_path: Path) -> bool:
        try:
            data = json.loads(pkg_json_path.read_text())
            return data.get('version') == version
        except Exception:
            return False

    # Priority 1: use the Docker-image-level global install (only if version matches).
    global_js = await _find_global_cline_js()
    if global_js is not None:
        global_cline_dir = global_js.parent.parent  # .../node_modules/cline/
        accept_global = version == 'latest' or _version_matches(global_cline_dir / 'package.json')
        if accept_global:
            # Check both nested (per-package) and hoisted (top-level) ripgrep locations.
            nested_rg = global_cline_dir / 'node_modules' / '@vscode' / 'ripgrep' / 'bin' / 'rg'
            hoisted_rg = global_cline_dir.parent / '@vscode' / 'ripgrep' / 'bin' / 'rg'
            if not nested_rg.exists() and not hoisted_rg.exists():
                # Provision ripgrep under the global cline package directory.
                nested_rg.parent.mkdir(parents=True, exist_ok=True)
                system_rg = find_runtime_binary('rg') or find_runtime_binary('ripgrep')
                if system_rg:
                    import shutil

                    shutil.copy2(system_rg, nested_rg)
                    nested_rg.chmod(nested_rg.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
                    logger.debug(f'Copied system ripgrep to global cline at {nested_rg}')
            return global_js
        logger.debug(
            f'Skipping global cline at {global_js} because version pin requires {version!r} '
            f'(global is a different version).'
        )

    existing = _resolve_cline_executable(install_dir)
    if existing is not None:
        installed_version = get_installed_version(install_dir)
        if version == 'latest' or installed_version == version:
            logger.debug(f'Cline already installed at {existing} (version {installed_version})')
            await _ensure_ripgrep_binary(install_dir)
            return existing
        # Stale cline from a previous run with a different version pin — purge and reinstall.
        import shutil as _shutil

        stale_pkg = install_path / 'node_modules' / 'cline'
        logger.debug(f'Purging stale cline {installed_version} at {stale_pkg} (pin requires {version}).')
        try:
            _shutil.rmtree(stale_pkg)
        except Exception as exc:
            logger.error('Failed to purge stale cline at %s: %s', stale_pkg, exc)
        # A surviving package.json makes the two-phase install skip its download, so the run would silently keep the stale version while reporting the pinned one.
        if stale_pkg.exists():
            raise RuntimeError(
                f'Could not purge stale cline {installed_version} at {stale_pkg}; refusing to continue because the install would keep that version instead of the pinned {version}.'
            )

    # --- Fast two-phase path ---
    cline_bin = await _install_cline_two_phase(install_dir, version)
    if cline_bin is not None:
        await _ensure_ripgrep_binary(install_dir)
        return cline_bin

    # --- Fallback: plain npm install ---
    logger.warning('Two-phase install failed, falling back to npm install cline')

    npm_executable = find_runtime_binary('npm') or 'npm'
    registry = npm_registry or _TENCENT_REGISTRY
    package_spec = NPM_PACKAGE if version == 'latest' else f'{NPM_PACKAGE}@{version}'

    # Write .npmrc to steer all scoped packages to the Tencent mirror.
    npmrc_path = install_path / '.npmrc'
    npmrc_path.write_text(_NPMRC_CONTENT)

    command = (
        f'{shlex.quote(npm_executable)} install '
        f'--prefix {shlex.quote(str(install_path))} --no-audit --no-fund --strict-ssl=false '
        f'--ignore-scripts --legacy-peer-deps '
        f'--registry {shlex.quote(registry)} {shlex.quote(package_spec)}'
    )
    logger.info(f'npm fallback: installing {package_spec} via {registry}')

    ok, output = await _run_npm_install(command, str(install_path), registry)
    if not ok:
        raise RuntimeError(f'Failed to install Cline: {output}')

    cline_bin = _resolve_cline_executable(install_dir)
    if cline_bin is None:
        raise RuntimeError(f'Cline install succeeded but binary not found under {install_path}/node_modules/.bin/')

    logger.debug(f'Cline installed via npm at {cline_bin} (version {get_installed_version(install_dir)})')
    await _ensure_ripgrep_binary(install_dir)
    return cline_bin


__all__ = [
    'DEFAULT_CLINE_VERSION',
    'NPM_PACKAGE',
    'get_installed_version',
    'install_cline_via_npm',
]
