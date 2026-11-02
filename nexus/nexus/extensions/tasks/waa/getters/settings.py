"""Windows Settings getters for WAA evaluation.

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import json
import logging

from typing import Any

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.settings')


async def _get_registry_key(runtime: AGSRawWindowsAgentArenaRuntime, path: str, value: str) -> dict:
    """Fetch a Windows registry key value via PowerShell."""
    command = f"powershell -Command \"Get-ItemPropertyValue -Path '{path}' -Name '{value}'\""
    return await runtime.execute(command)


def _generate_time(hour: int, minute: int) -> str:
    if hour < 12:
        am_pm = 'AM'
    else:
        am_pm = 'PM'
        hour -= 12
    return f'{hour:02d}:{minute:02d} {am_pm}'


@getter_registry.register()
async def get_night_light_state(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get night light state and schedule from registry."""
    key_path = r'HKCU:\Software\Microsoft\Windows\CurrentVersion\CloudStore\Store\DefaultAccount\Current\default$windows.data.bluelightreduction.settings\windows.data.bluelightreduction.settings'
    setting_path = 'Data'
    result = await _get_registry_key(runtime, key_path, setting_path)

    output = result.get('output')
    if not output:
        logger.warning(f'get_night_light_state: no output from registry query, result={result}')
        return False

    decoded_result = output.split('\n')
    if decoded_result[24] == '1':
        starting_hour = int(decoded_result[28])
        if starting_hour != 202:
            starting_minute = int(decoded_result[30] if decoded_result[29] != '0' else '0')
            ending_index = 35 if starting_minute != 0 else 33
            ending_hour = int(decoded_result[ending_index])
            ending_minute = int(decoded_result[ending_index + 2] if decoded_result[ending_index + 1] != '0' else '0')
            start = _generate_time(starting_hour, starting_minute)
            end = _generate_time(ending_hour, ending_minute)
            return [True, start, end]
        return [True]
    return False


@getter_registry.register()
async def get_storage_sense_run_frequency(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get Storage Sense run frequency from registry."""
    key_path = r'HKCU:\Software\Microsoft\Windows\CurrentVersion\StorageSense\Parameters\StoragePolicy'

    is_enabled = await _get_registry_key(runtime, key_path, '01')
    if is_enabled.get('status') == 'success' and is_enabled.get('output') == '1\n':
        frequency = await _get_registry_key(runtime, key_path, '2048')
        if frequency.get('status') == 'success':
            return frequency['output'].split('\n')[0]

    return None


@getter_registry.register()
async def get_system_timezone(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get system timezone via PowerShell."""
    command = 'powershell -Command "Get-TimeZone | Select-Object -Property DisplayName, StandardName | ConvertTo-Json"'
    try:
        result = await runtime.execute(command)
        data = json.loads(result['output'])
        display_name = data.get('DisplayName', None)
        standard_name = data.get('StandardName', None)
        if display_name and standard_name:
            return display_name
    except Exception:
        return 'Error getting the timezone'
    return 'Unable to get time zone'


@getter_registry.register()
async def get_desktop_background(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a custom desktop background is set (via registry)."""
    key_path = r'HKCU:Control Panel\Desktop'
    setting_path = 'WallPaper'
    result = await _get_registry_key(runtime, key_path, setting_path)
    output = result.get('output')
    if output is None:
        logger.warning(f'get_desktop_background: no output from registry query, result={result}')
        return 'error'
    wallpaper_path = output.replace('\n', '')
    if wallpaper_path == '':
        return 'True'
    return 'False'


@getter_registry.register()
async def get_system_notifications(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if system notifications are disabled (ToastEnabled=0)."""
    key_path = r'HKCU:Software\Microsoft\Windows\CurrentVersion\PushNotifications'
    setting_path = 'ToastEnabled'
    is_notification_enabled = await _get_registry_key(runtime, key_path, setting_path)
    try:
        if is_notification_enabled is None or is_notification_enabled.get('output') == '':
            return 'False'
        if is_notification_enabled.get('output'):
            toast_enabled = is_notification_enabled['output'].replace('\n', '')
            if toast_enabled == '0':
                return 'True'
        return 'False'
    except Exception:
        return 'Unable to get the notifications settings state'
