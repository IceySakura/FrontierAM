"""Microsoft Edge getters for WAA evaluation (msedge.py subset).

These getters handle Edge-specific features that differ from the chrome-like
getters in edge.py: home page (from Secure Preferences), PWA validation,
and default download folder.

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import json
import logging

from typing import Any

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.msedge')

# Edge file paths on Windows
_EDGE_SECURE_PREFERENCES = (
    r"import os; print(os.path.join(os.getenv('LOCALAPPDATA'), 'Microsoft/Edge/User Data/Default/Secure Preferences'))"
)
_EDGE_PREFERENCES = (
    r"import os; print(os.path.join(os.getenv('LOCALAPPDATA'), 'Microsoft/Edge/User Data/Default/Preferences'))"
)


@getter_registry.register()
async def get_edge_home_page(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get the Edge home page from Secure Preferences."""
    result = await runtime.execute(['python', '-c', _EDGE_SECURE_PREFERENCES], shell=False)
    preference_file_path = result['output'].strip()

    try:
        content = await runtime.get_file(preference_file_path)
        data = json.loads(content)

        home_page = data.get('homepage', {})
        if home_page:
            return home_page.replace('https://', '').replace('http://', '').replace('/', '')
    except Exception:
        return 'error'
    return 'False'


@getter_registry.register()
async def get_validate_pwa_installed(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Validate if a PWA (pwabuilder) app is installed via Get-AppxPackage."""
    try:
        cmd = 'powershell -Command "Get-AppxPackage | Select-Object Name,PackageFullName | ConvertTo-Json"'
        result = await runtime.execute(cmd)
        if result.get('output'):
            data = json.loads(result['output'])
            pwa_packages = [pkg for pkg in data if 'pwabuilder' in pkg['Name'].lower()]
            if pwa_packages:
                return 'app_installed'
    except Exception:
        return 'error'
    return 'app_not_installed'


@getter_registry.register()
async def get_edge_default_download_folder(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get Edge default download folder from Preferences."""
    result = await runtime.execute(['python', '-c', _EDGE_PREFERENCES], shell=False)
    preference_file_path = result['output'].strip()

    try:
        content = await runtime.get_file(preference_file_path)
        data = json.loads(content)

        folder = data.get('download', {}).get('default_directory', {})
        if folder:
            return folder
    except Exception:
        return 'error'
    return 'False'
