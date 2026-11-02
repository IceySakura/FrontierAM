"""OpenCC installer.

Installs OpenCC from pre-built standalone binaries (bun-compiled).
Ships both glibc and musl variants; the runtime-side installer detects
the C library at install time and picks the matching binary.
"""

from __future__ import annotations

import logging
import shutil
import stat
import subprocess

from pathlib import Path

from nexus.extensions.agents.installed_common import DEFAULT_AGENT_INSTALL_ROOT


logger = logging.getLogger(__name__)

BIN_GLIBC = 'opencc-bin'
BIN_MUSL = 'opencc-bin-musl'
BIN_INSTALLED = 'opencc-bin'

_MODULE_DIR = Path(__file__).resolve().parent
BIN_PATH_GLIBC = _MODULE_DIR / BIN_GLIBC
BIN_PATH_MUSL = _MODULE_DIR / BIN_MUSL

DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/opencc'


def _is_musl() -> bool:
    """Detect whether the current system uses musl libc."""
    try:
        result = subprocess.run(['ldd', '--version'], capture_output=True, text=True, timeout=5)
        combined = (result.stdout + result.stderr).lower()
        if 'musl' in combined:
            return True
    except Exception:
        pass
    ld_musl = Path('/lib/ld-musl-x86_64.so.1')
    return ld_musl.exists()


async def install_opencc_binary(
    install_dir: str,
    remote_glibc_path: str,
    remote_musl_path: str,
) -> Path:
    """Pick the correct binary variant, copy to *install_dir*, and make it executable."""
    install_path = Path(install_dir)
    target = install_path / BIN_INSTALLED

    if target.exists() and target.stat().st_size > 0:
        logger.debug('OpenCC binary already present at %s', target)
        return target

    install_path.mkdir(parents=True, exist_ok=True)

    musl = _is_musl()
    src_path = remote_musl_path if musl else remote_glibc_path
    variant = 'musl' if musl else 'glibc'
    logger.debug('Detected %s libc, using %s binary from %s', variant, variant, src_path)

    src = Path(src_path)
    if not src.exists():
        raise RuntimeError(f'OpenCC {variant} binary not found at {src_path}')

    shutil.copy2(str(src), str(target))
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    logger.debug('OpenCC binary installed at %s (%d bytes, %s)', target, target.stat().st_size, variant)
    return target


def resolve_opencc_launch_argv(install_dir: str) -> list[str]:
    """Return the argv prefix to launch OpenCC (standalone binary)."""
    binary = Path(install_dir) / BIN_INSTALLED
    if not binary.exists():
        raise RuntimeError(f'OpenCC binary not found at {binary}')
    return [str(binary)]


__all__ = [
    'BIN_GLIBC',
    'BIN_INSTALLED',
    'BIN_MUSL',
    'BIN_PATH_GLIBC',
    'BIN_PATH_MUSL',
    'DEFAULT_INSTALL_DIR',
    'install_opencc_binary',
    'resolve_opencc_launch_argv',
]
