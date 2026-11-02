"""Resolve browser executables without installing or launching software."""

import os

from pathlib import Path

from nexus.utils.platform import is_windows


def _windows_browser_candidates():
    import winreg

    for executable, suffix in (
        ('chrome.exe', 'Google/Chrome/Application/chrome.exe'),
        ('msedge.exe', 'Microsoft/Edge/Application/msedge.exe'),
    ):
        key_path = rf'SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{executable}'
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):  # ty: ignore[unresolved-attribute] -- Windows-only API
            for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):  # ty: ignore[unresolved-attribute] -- Windows-only API
                try:
                    with winreg.OpenKey(hive, key_path, 0, winreg.KEY_READ | view) as key:  # ty: ignore[unresolved-attribute] -- Windows-only API
                        value, _ = winreg.QueryValueEx(key, '')  # ty: ignore[unresolved-attribute] -- Windows-only API
                    if isinstance(value, str) and value.strip():
                        yield Path(os.path.expandvars(value.strip().strip('"')))
                except OSError:
                    continue
        for variable in ('ProgramFiles', 'ProgramFiles(x86)', 'LOCALAPPDATA'):
            root = os.environ.get(variable)
            if root:
                yield Path(root) / suffix


def resolve_browser_executable(explicit: str | None, playwright_path: str) -> str:
    if explicit is not None:
        if not Path(explicit).is_file():
            raise FileNotFoundError(f'Browser executable does not exist: {explicit}')
        return explicit

    if is_windows():
        if Path(playwright_path).is_file():
            return playwright_path
        for candidate in _windows_browser_candidates():
            if candidate.is_file():
                return str(candidate)
    else:
        for candidate in ('/nix/runtime/nix-env/bin/chromium', playwright_path):
            if Path(candidate).is_file():
                return candidate

    raise FileNotFoundError(
        'No browser executable found. Set executable_path to an installed Chromium-compatible browser. '
        'On Windows, Playwright Chromium, Chrome and Edge are searched; no browser is installed automatically.'
    )
