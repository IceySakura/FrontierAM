"""Edge browser getters for WAA evaluation.

These getters retrieve Microsoft Edge state from the Windows VM — preference files
and sqlite databases.

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.edge')

# Edge User Data paths on Windows
_EDGE_PREFERENCES = (
    r"import os; print(os.path.join(os.getenv('LOCALAPPDATA'), 'Microsoft/Edge/User Data/Default/Preferences'))"
)
_EDGE_BOOKMARKS = (
    r"import os; print(os.path.join(os.getenv('LOCALAPPDATA'), 'Microsoft/Edge/User Data/Default/Bookmarks'))"
)
_EDGE_COOKIES = (
    r"import os; print(os.path.join(os.getenv('LOCALAPPDATA'), 'Microsoft/Edge/User Data/Default/Network/Cookies'))"
)
_EDGE_HISTORY = r"""import os; print(os.path.join(os.getenv('USERPROFILE'), "AppData", "Local", "Microsoft", "Edge", "User Data", "Default", "History"))"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _load_edge_preferences(runtime: AGSRawWindowsAgentArenaRuntime) -> dict:
    """Download and parse Edge Preferences JSON from the VM."""
    result = await runtime.execute(['python', '-c', _EDGE_PREFERENCES], shell=False)
    pref_path = result['output'].strip()
    content = await runtime.get_file(pref_path)
    return json.loads(content)


async def _read_edge_policy_dword(runtime: AGSRawWindowsAgentArenaRuntime, value_name: str) -> int | None:
    """Read an Edge policy DWORD from HKCU/HKLM."""

    script = rf"""
import winreg

for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
    try:
        with winreg.OpenKey(hive, r"Software\Policies\Microsoft\Edge") as key:
            value, _ = winreg.QueryValueEx(key, {value_name!r})
            print(int(value))
            break
    except FileNotFoundError:
        continue
    except OSError:
        continue
else:
    print("")
"""
    result = await runtime.execute(['python', '-c', script], shell=False)
    output = str(result.get('output', '')).strip()
    if not output:
        return None
    try:
        return int(output, 0)
    except ValueError:
        logger.warning('Unexpected Edge policy value for %s: %r', value_name, output)
        return None


# ---------------------------------------------------------------------------
# Preference-based getters
# ---------------------------------------------------------------------------


@getter_registry.register()
async def get_profile_name_from_edge(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get the Edge profile name from Preferences."""
    try:
        data = await _load_edge_preferences(runtime)
        return data.get('profile', {}).get('name', None)
    except Exception as e:
        logger.error(f'Error: {e}')
        return None


@getter_registry.register()
async def get_favorites(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get Edge favorites (bookmarks) JSON."""
    result = await runtime.execute(['python', '-c', _EDGE_BOOKMARKS], shell=False)
    bookmarks_path = result['output'].strip()
    content = await runtime.get_file(bookmarks_path)
    if not content:
        return []
    data = json.loads(content)
    return data.get('roots', {})


@getter_registry.register()
async def get_cookie_data_for_edge(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get cookies from the Edge Cookies sqlite database."""
    from nexus.extensions.tasks.waa.setup import cache_dir

    try:
        if 'user_data_dir' in config:
            assert '\\' not in config['user_data_dir'], 'user_data_dir cannot contain backslash'
            assert "'" not in config['user_data_dir'], 'user_data_dir cannot contain single backticks'
            cmd = f"import os; print(os.path.join(os.path.expandvars('{config['user_data_dir']}'), 'Default/Network/Cookies'))"
        else:
            cmd = _EDGE_COOKIES

        # Kill msedge first, otherwise we can't access the cookies
        await runtime.execute(['python', '-c', "import os; os.system('taskkill /F /IM msedge.exe /T')"], shell=False)

        result = await runtime.execute(['python', '-c', cmd], shell=False)
        cookie_file_path = result['output'].strip()
        content = await runtime.get_file(cookie_file_path)
        local_path = cache_dir / config['dest']
        await asyncio.to_thread(local_path.write_bytes, content)

        def _query_cookies(path: Path):
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM cookies')
            cookies = cursor.fetchall()
            conn.close()
            return cookies

        return await asyncio.to_thread(_query_cookies, local_path)
    except Exception as e:
        logger.error(f'Error: {e}')
        return None


@getter_registry.register()
async def get_history_for_edge(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get browsing history from the Edge History sqlite database."""
    from nexus.extensions.tasks.waa.setup import cache_dir

    try:
        result = await runtime.execute(['python', '-c', _EDGE_HISTORY], shell=False)
        history_path = result['output'].strip()
        content = await runtime.get_file(history_path)
        local_path = cache_dir / config['dest']
        await asyncio.to_thread(local_path.write_bytes, content)

        def _query_history(path: Path):
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()
            cursor.execute('SELECT url, title, last_visit_time FROM urls')
            items = cursor.fetchall()
            conn.close()
            return items

        return await asyncio.to_thread(_query_history, local_path)
    except Exception as e:
        logger.error(f'Error: {e}')
        return None


@getter_registry.register()
async def get_enable_do_not_track_from_edge(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if Do Not Track is enabled in Edge."""
    try:
        data = await _load_edge_preferences(runtime)
        enabled = data.get('enable_do_not_track', False)
        return 1.0 if enabled else 0.0
    except Exception as e:
        logger.error(f'Error: {e}')
        return 0.0


@getter_registry.register()
async def get_enable_enhanced_safety_browsing_from_edge(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check whether Edge unsafe-site warning protection is enabled.

    The task asks Edge to warn on potentially harmful or unsafe sites, which is
    controlled by Microsoft Defender SmartScreen. Older WAA evaluators checked
    only ``edge.super_duper_secure_mode.enabled`` in Preferences; keep that
    compatibility path, but honor current Edge SmartScreen policy first.
    """
    try:
        # 当前 Edge 的“有害站点警告”实际由 SmartScreen 策略控制；只查旧 Preferences 字段会误判。
        # 因此优先读策略注册表，读不到再走旧字段兼容历史任务。
        smart_screen_policy = await _read_edge_policy_dword(runtime, 'SmartScreenEnabled')
        if smart_screen_policy is not None:
            return 1.0 if smart_screen_policy == 1 else 0.0

        data = await _load_edge_preferences(runtime)
        is_enabled = data.get('edge', {}).get('super_duper_secure_mode', {}).get('enabled', False)
        return 1.0 if is_enabled else 0.0
    except Exception as e:
        logger.error(f'Error: {e}')
        return 0.0


@getter_registry.register()
async def get_data_delete_automacally_from_edge(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if Edge is set to auto-delete data on exit."""
    try:
        data = await _load_edge_preferences(runtime)
        data_delete_state = data['browser'].get('clear_data_on_exit', None)
        target_state = {
            'browsing_history': True,
            'cache': True,
            'cookies': True,
            'download_history': True,
            'form_data': True,
            'hosted_apps_data': False,
            'passwords': True,
            'site_settings': True,
        }
        return 1.0 if data_delete_state == target_state else 0.0
    except Exception as e:
        logger.error(f'Error: {e}')
        return 0.0


@getter_registry.register()
async def get_edge_font_size(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get Edge font size settings from Preferences."""
    try:
        data = await _load_edge_preferences(runtime)
        return data.get('webkit', {}).get(
            'webprefs',
            {
                'default_fixed_font_size': 13,
                'default_font_size': 16,
            },
        )
    except Exception as e:
        logger.error(f'Error: {e}')
        return {
            'default_fixed_font_size': 13,
            'default_font_size': 16,
        }


@getter_registry.register()
async def get_default_search_engine_from_edge(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get the default search engine name from Edge Preferences."""
    try:
        data = await _load_edge_preferences(runtime)
        engine_guids = {
            '485bf7d3-0215-45af-87dc-538868000001': 'Bing',
            '485bf7d3-0215-45af-87dc-538868000002': 'Yahoo!',
            '485bf7d3-0215-45af-87dc-538868000003': 'Google',
            '485bf7d3-0215-45af-87dc-538868000092': 'DuckDuckGo',
            '485bf7d3-0215-45af-87dc-538868000015': 'Yandex',
        }
        cur_guid = data.get('default_search_provider', {}).get('synced_guid')
        search_engine = engine_guids.get(cur_guid, '')
        if not search_engine:
            search_engine = 'Unknown'
        return search_engine
    except Exception as e:
        logger.error(f'Error: {e}')
        return 'Unknown'


@getter_registry.register()
async def get_url_shortcuts_on_desktop(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get Edge URL shortcut (.url) files from the desktop."""
    desktop_path_resp = await runtime.desktop_path()
    desktop_path = desktop_path_resp['desktop_path']
    dir_tree_resp = await runtime.list_directory(desktop_path)
    dir_tree = dir_tree_resp.get('directory_tree', {})

    shortcuts_paths = [f['name'] for f in dir_tree.get('children', []) if f['name'].endswith('.url')]

    short_cuts = {}
    for shortcut_name in shortcuts_paths:
        shortcut_full_path = os.path.join(desktop_path, shortcut_name)
        shortcut_data = await runtime.get_file(shortcut_full_path)
        short_cuts[shortcut_name] = shortcut_data.decode('utf-8')

    return short_cuts
