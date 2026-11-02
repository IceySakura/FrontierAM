"""Cross-platform helpers shared by the runtime server and worker manager."""

from __future__ import annotations

import sys

from pathlib import Path

from nexus.utils.platform import is_windows


def venv_python_path(venv_dir: Path) -> Path:
    """Return the Python executable path for a virtual environment."""
    if is_windows():
        return venv_dir / 'Scripts' / 'python.exe'
    return venv_dir / 'bin' / 'python'


def venv_site_packages_path(venv_dir: Path) -> Path:
    """Return the site-packages path for a virtual environment."""
    if is_windows():
        return venv_dir / 'Lib' / 'site-packages'

    lib = venv_dir / 'lib'
    if lib.exists():
        for pydir in lib.iterdir():
            site_packages = pydir / 'site-packages'
            if site_packages.exists():
                return site_packages
    return lib / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'


def subprocess_creation_flags() -> int:
    """Return flags that place Windows children in their own process group."""
    if not is_windows():
        return 0
    import subprocess

    return subprocess.CREATE_NEW_PROCESS_GROUP  # ty: ignore[unresolved-attribute] -- Windows-only API


__all__ = [
    'subprocess_creation_flags',
    'venv_python_path',
    'venv_site_packages_path',
]
