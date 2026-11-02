"""Chrome browser getters for WAA evaluation.

These getters retrieve Chrome state from the Windows VM — preference files,
sqlite databases, and live page info via CDP (Playwright).

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.chrome')

# Chrome User Data paths on Windows
_CHROME_PREFERENCES = (
    r"import os; print(os.path.join(os.getenv('LOCALAPPDATA'), 'Google/Chrome/User Data/Default/Preferences'))"
)
_CHROME_LOCAL_STATE = (
    r"import os; print(os.path.join(os.getenv('LOCALAPPDATA'), 'Google/Chrome/User Data/Local State'))"
)
_CHROME_BOOKMARKS = (
    r"import os; print(os.path.join(os.getenv('LOCALAPPDATA'), 'Google/Chrome/User Data/Default/Bookmarks'))"
)
_CHROME_COOKIES = (
    r"import os; print(os.path.join(os.getenv('LOCALAPPDATA'), 'Google/Chrome/User Data/Default/Network/Cookies'))"
)
_CHROME_HISTORY = r"""import os; print(os.path.join(os.getenv('USERPROFILE'), "AppData", "Local", "Google", "Chrome", "User Data", "Default", "History"))"""

_DEFAULT_NAVIGATION_TIMEOUT_MS = 15000
_DEFAULT_LOAD_STATE_TIMEOUT_MS = 8000
_DEFAULT_PAGE_SNAPSHOT_TIMEOUT_MS = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_chrome_pref_path(runtime: AGSRawWindowsAgentArenaRuntime) -> str:
    """Get the Chrome Preferences file path on the VM."""
    result = await runtime.execute(['python', '-c', _CHROME_PREFERENCES], shell=False)
    return result['output'].strip()


async def _get_chrome_local_state_path(runtime: AGSRawWindowsAgentArenaRuntime) -> str:
    """Get the Chrome Local State file path on the VM."""
    result = await runtime.execute(['python', '-c', _CHROME_LOCAL_STATE], shell=False)
    return result['output'].strip()


async def _load_chrome_preferences(runtime: AGSRawWindowsAgentArenaRuntime) -> dict:
    """Download and parse Chrome Preferences JSON from the VM."""
    pref_path = await _get_chrome_pref_path(runtime)
    content = await runtime.get_file(pref_path)
    return json.loads(content)


async def _load_chrome_local_state(runtime: AGSRawWindowsAgentArenaRuntime) -> dict:
    """Download and parse Chrome Local State JSON from the VM."""
    state_path = await _get_chrome_local_state_path(runtime)
    content = await runtime.get_file(state_path)
    return json.loads(content)


@asynccontextmanager
async def _connect_cdp(runtime: AGSRawWindowsAgentArenaRuntime) -> AsyncIterator:
    """Connect to the Chrome instance on the VM via CDP.

    Usage:
        async with _connect_cdp(runtime) as browser:
            page = await browser.new_page()
            ...
    """
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.connect_over_cdp(await runtime.chromium_cdp_url(), headers=runtime.handle.headers)
    except Exception:
        await pw.stop()
        raise
    try:
        yield browser
    finally:
        await browser.close()
        await pw.stop()


async def _get_active_url_from_access_tree(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> str | None:
    """Get the active tab URL from the Windows accessibility tree.

    Uses the UIA backend to find the Chrome address bar value.
    """
    resp = await runtime.accessibility_tree(backend='uia')
    at_xml = resp.get('AT')
    if not at_xml:
        raise RuntimeError('Failed to get accessibility tree: response missing "AT" field')

    # The accessibility tree response is XML; parse it to find the address bar
    import lxml.etree

    from lxml.cssselect import CSSSelector

    ns_map = {
        'st': 'uri:deskat:state.at-spi.gnome.org',
        'attr': 'uri:deskat:attributes.at-spi.gnome.org',
        'cp': 'uri:deskat:component.at-spi.gnome.org',
        'doc': 'uri:deskat:document.at-spi.gnome.org',
        'docattr': 'uri:deskat:attributes.document.at-spi.gnome.org',
        'txt': 'uri:deskat:text.at-spi.gnome.org',
        'val': 'uri:deskat:value.at-spi.gnome.org',
        'act': 'uri:deskat:action.at-spi.gnome.org',
    }

    try:
        at = lxml.etree.fromstring(at_xml.encode() if isinstance(at_xml, str) else at_xml)
    except Exception as e:
        raise RuntimeError(f'Error parsing accessibility tree: {e}') from e

    selector_string = 'chrome_widgetwin_1 [name=Address\\ and\\ search\\ bar]'
    try:
        selector = CSSSelector(selector_string, namespaces=ns_map)
    except Exception as e:
        raise RuntimeError(f'Failed to parse CSS selector: {e}') from e

    elements = selector(at)
    if not elements:
        logger.warning('No address bar element found in accessibility tree')
        return None

    text = elements[-1].attrib.get('{uri:deskat:value.at-spi.gnome.org}value')
    if not text:
        logger.warning('No text value found in address bar element')
        return None

    goto_prefix = config.get('goto_prefix', 'https://')
    return f'{goto_prefix}{text}'


async def _get_active_url_from_cdp(runtime: AGSRawWindowsAgentArenaRuntime) -> str | None:
    """Best-effort fallback for active-tab URL lookup via Chrome DevTools."""
    try:
        async with _connect_cdp(runtime) as browser:
            urls: list[str] = []
            for context in browser.contexts:
                for page in context.pages:
                    url = page.url
                    if url and url not in {'about:blank', 'chrome://newtab/'}:
                        urls.append(url)
            if urls:
                return urls[-1]
            logger.warning('No non-blank Chrome pages found via CDP')
            return None
    except Exception as e:
        logger.warning('Failed to get active tab URL via CDP fallback: %s', e)
        return None


async def _get_active_url(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> str | None:
    """Get the active tab URL, falling back to CDP when UIA is incomplete."""
    # Chrome 地址栏在 UIA 树中偶发缺失或 value 为空；这不代表浏览器状态不可评估，
    # 所以先读 UIA，失败再用 CDP 枚举页面 URL，避免把可恢复的观测缺口记成 ERROR。
    try:
        active_url = await _get_active_url_from_access_tree(runtime, config)
    except Exception as e:
        logger.warning('Failed to get active URL from accessibility tree: %s', e)
        active_url = None
    if active_url:
        return active_url

    logger.warning('Falling back to CDP for active Chrome URL')
    return await _get_active_url_from_cdp(runtime)


def _page_timeouts(config: dict) -> tuple[int, int, int]:
    navigation_timeout = int(config.get('navigation_timeout_ms', _DEFAULT_NAVIGATION_TIMEOUT_MS))
    load_state_timeout = int(config.get('load_state_timeout_ms', _DEFAULT_LOAD_STATE_TIMEOUT_MS))
    snapshot_timeout = int(config.get('snapshot_timeout_ms', _DEFAULT_PAGE_SNAPSHOT_TIMEOUT_MS))
    return navigation_timeout, load_state_timeout, snapshot_timeout


async def _get_page_snapshot(page, *, snapshot_timeout_ms: int) -> dict[str, str]:
    """Return best-effort page title, URL, and HTML without letting one field block evaluation."""

    async def _with_timeout(coro, default: str) -> str:
        try:
            return await asyncio.wait_for(coro, timeout=snapshot_timeout_ms / 1000)
        except Exception as exc:
            logger.warning('Chrome page snapshot field unavailable: %s', exc)
            return default

    return {
        'title': await _with_timeout(page.title(), ''),
        'url': page.url,
        'content': await _with_timeout(page.content(), ''),
    }


async def _wait_for_page_state(page, load_state: str, *, timeout_ms: int) -> None:
    try:
        await page.wait_for_load_state(load_state, timeout=timeout_ms)
    except Exception as exc:
        logger.warning('Chrome page did not reach load_state=%s within %sms: %s', load_state, timeout_ms, exc)


async def _generate_pdf_with_vm_chrome(runtime: AGSRawWindowsAgentArenaRuntime, url: str, local_path: str) -> str:
    remote_path = rf'C:\Users\Docker\AppData\Local\Temp\waa_expected_{uuid.uuid4().hex}.pdf'
    script = f"""
import os
import shutil
import subprocess
import time
from pathlib import Path

url = {json.dumps(url)}
out = Path({json.dumps(remote_path)})
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    out.unlink()

candidates = [
    Path(os.environ.get('PROGRAMFILES', '')) / 'Google' / 'Chrome' / 'Application' / 'chrome.exe',
    Path(os.environ.get('PROGRAMFILES(X86)', '')) / 'Google' / 'Chrome' / 'Application' / 'chrome.exe',
    Path(os.environ.get('LOCALAPPDATA', '')) / 'Google' / 'Chrome' / 'Application' / 'chrome.exe',
]
chrome = next((str(path) for path in candidates if path.exists()), shutil.which('chrome') or shutil.which('chrome.exe'))
if not chrome:
    raise FileNotFoundError('Could not find Google Chrome executable')

cmd = [
    chrome,
    '--headless=new',
    '--disable-gpu',
    '--no-pdf-header-footer',
    f'--print-to-pdf={{out}}',
    url,
]
proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
for _ in range(30):
    if out.exists() and out.stat().st_size > 1000:
        break
    time.sleep(1)
if proc.returncode != 0:
    raise RuntimeError(f'Chrome print-to-pdf failed rc={{proc.returncode}} stdout={{proc.stdout[:500]}} stderr={{proc.stderr[:500]}}')
if not out.exists() or out.stat().st_size <= 1000:
    raise RuntimeError(f'Chrome print-to-pdf did not create a usable PDF at {{out}}')
print(str(out))
"""
    result = await runtime.execute(['python', '-c', script], shell=False)
    if result.get('returncode') != 0:
        raise RuntimeError(
            'VM Chrome PDF generation failed: '
            f'returncode={result.get("returncode")} output={str(result.get("output"))[:500]!r} '
            f'error={str(result.get("error"))[:500]!r}'
        )

    data = await runtime.get_file(remote_path)
    if not data.startswith(b'%PDF'):
        raise RuntimeError(f'VM Chrome PDF generation returned invalid PDF bytes: len={len(data)} prefix={data[:16]!r}')
    await asyncio.to_thread(Path(local_path).write_bytes, data)
    return local_path


# ---------------------------------------------------------------------------
# Preference-based getters (no CDP needed)
# ---------------------------------------------------------------------------


@getter_registry.register()
async def get_default_search_engine(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get the default search engine name from Chrome Preferences."""
    try:
        data = await _load_chrome_preferences(runtime)

        # GUID reference:
        # https://chromium.googlesource.com/chromium/src/+/8aca664983590de953a8785d88151bddc91c170c/components/test/data/web_database/version_82.sql
        engine_guids = {
            '485bf7d3-0215-45af-87dc-538868000001': 'Google',
            '485bf7d3-0215-45af-87dc-538868000003': 'Bing',
            '485bf7d3-0215-45af-87dc-538868000002': 'Yahoo!',
            '485bf7d3-0215-45af-87dc-538868000092': 'DuckDuckGo',
        }
        search_engine = engine_guids.get(data.get('default_search_provider', {}).get('guid'))
        if not search_engine:  # fallback
            search_engine = (
                data.get('default_search_provider_data', {}).get('template_url_data', {}).get('short_name', 'Google')
            )
        return search_engine
    except Exception as e:
        logger.error(f'Error: {e}')
        return 'Google'


@getter_registry.register()
async def get_cookie_data(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get cookies from the Chrome Cookies sqlite database."""
    from nexus.extensions.tasks.waa.setup import cache_dir

    try:
        if 'user_data_dir' in config:
            assert '\\' not in config['user_data_dir'], 'user_data_dir cannot contain backslash'
            assert "'" not in config['user_data_dir'], 'user_data_dir cannot contain single backticks'
            cmd = f"import os; print(os.path.join(os.path.expandvars('{config['user_data_dir']}'), 'Default/Network/Cookies'))"
        else:
            cmd = _CHROME_COOKIES

        result = await runtime.execute(['python', '-c', cmd], shell=False)
        cookie_file_path = result['output'].strip()
        content = await runtime.get_file(cookie_file_path)
        local_path = cache_dir / config['dest']
        await asyncio.to_thread(local_path.write_bytes, content)

        def _query_cookies(path: Path):
            conn = sqlite3.connect(str(path))
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM cookies')
                return cursor.fetchall()
            finally:
                conn.close()

        return await asyncio.to_thread(_query_cookies, local_path)
    except Exception as e:
        logger.warning('Unable to read Chrome Cookies database; cookie state is unavailable: %s', e)
        return None


@getter_registry.register()
async def get_history(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get browsing history from the Chrome History sqlite database."""
    from nexus.extensions.tasks.waa.setup import cache_dir

    try:
        result = await runtime.execute(['python', '-c', _CHROME_HISTORY], shell=False)
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
async def get_enabled_experiments(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get enabled Chrome flags/experiments from Local State."""
    try:
        data = await _load_chrome_local_state(runtime)
        return data.get('browser', {}).get('enabled_labs_experiments', [])
    except Exception as e:
        logger.error(f'Error: {e}')
        return []


@getter_registry.register()
async def get_profile_name(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get the Chrome profile name from Preferences."""
    try:
        data = await _load_chrome_preferences(runtime)
        return data.get('profile', {}).get('name', None)
    except Exception as e:
        logger.error(f'Error: {e}')
        return None


@getter_registry.register()
async def get_chrome_font_size(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get Chrome font size settings from Preferences."""
    try:
        data = await _load_chrome_preferences(runtime)
        return data.get('webkit', {}).get(
            'webprefs',
            {
                'default_fixed_font_size': 13,
                'default_font_size': 16,
                'minimum_font_size': 13,
            },
        )
    except Exception as e:
        logger.error(f'Error: {e}')
        return {
            'default_fixed_font_size': 13,
            'default_font_size': 16,
        }


@getter_registry.register()
async def get_bookmarks(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get Chrome bookmarks JSON."""
    result = await runtime.execute(['python', '-c', _CHROME_BOOKMARKS], shell=False)
    bookmarks_path = result['output'].strip()
    content = await runtime.get_file(bookmarks_path)
    if not content:
        return []
    data = json.loads(content)
    return data.get('roots', {})


@getter_registry.register()
async def get_enable_do_not_track(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if Do Not Track is enabled in Chrome."""
    try:
        data = await _load_chrome_preferences(runtime)
        enabled = data.get('enable_do_not_track', False)
        return 'true' if enabled else 'false'
    except Exception as e:
        logger.error(f'Error: {e}')
        return 'false'


@getter_registry.register()
async def get_enable_enhanced_safety_browsing(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if Enhanced Safe Browsing is enabled in Chrome."""
    try:
        data = await _load_chrome_preferences(runtime)
        enabled = data.get('safebrowsing', {}).get('enhanced', False)
        return 'true' if enabled else 'false'
    except Exception as e:
        logger.error(f'Error: {e}')
        return 'false'


@getter_registry.register()
async def get_data_delete_automacally(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if Chrome is set to auto-delete data on exit."""
    try:
        data = await _load_chrome_preferences(runtime)
        data_delete_state = data.get('profile', {}).get('default_content_setting_values', None)
        return 'true' if data_delete_state is not None else 'false'
    except Exception as e:
        logger.error(f'Error: {e}')
        return 'false'


@getter_registry.register()
async def get_shortcuts_on_desktop(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get Chrome shortcut (.lnk) files from the desktop.

    Downloads each .lnk file and parses it using LnkParse3.
    """
    from contextlib import redirect_stdout
    from io import BytesIO, StringIO

    import LnkParse3

    desktop_path_resp = await runtime.desktop_path()
    desktop_path = desktop_path_resp['desktop_path']
    dir_tree_resp = await runtime.list_directory(desktop_path)
    dir_tree = dir_tree_resp.get('directory_tree', {})

    shortcuts_paths = [f['name'] for f in dir_tree.get('children', []) if f['name'].endswith('.lnk')]

    short_cuts = {}
    for shortcut_name in shortcuts_paths:
        import os

        shortcut_full_path = os.path.join(desktop_path, shortcut_name)
        shortcut_data = await runtime.get_file(shortcut_full_path)
        with BytesIO(shortcut_data) as indata:
            lnk = LnkParse3.lnk_file(indata)
        mock_stdout = StringIO()
        with redirect_stdout(mock_stdout):
            lnk.print_json(print_all=True)
        short_cuts[shortcut_name] = mock_stdout.getvalue()

    return short_cuts


# ---------------------------------------------------------------------------
# CDP-based getters (Playwright)
# ---------------------------------------------------------------------------


@getter_registry.register()
async def get_page_info(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Navigate to a URL and return page title, url, and HTML content."""
    url = config['url']
    load_state = config.get('load_state', 'load')
    navigation_timeout, load_state_timeout, snapshot_timeout = _page_timeouts(config)

    async with _connect_cdp(runtime) as browser:
        page = await browser.contexts[0].new_page()
        try:
            # setup/evaluator 取页面状态时只要求页面开始渲染；等待完整 load 会被 live-web 长连接拖死。
            await page.goto(url, wait_until='domcontentloaded', timeout=navigation_timeout)
        except Exception as e:
            logger.warning('Chrome page navigation did not complete for %s within %sms: %s', url, navigation_timeout, e)
        await _wait_for_page_state(page, load_state, timeout_ms=load_state_timeout)
        return await _get_page_snapshot(page, snapshot_timeout_ms=snapshot_timeout)


@getter_registry.register()
async def get_open_tabs_info(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get info (title, url) for all open tabs via CDP."""
    async with _connect_cdp(runtime) as browser:
        tabs_info = []
        for context in browser.contexts:
            for page in context.pages:
                try:
                    await page.wait_for_load_state('networkidle')
                    title = await page.title()
                    url = page.url
                    tabs_info.append({'title': title, 'url': url})
                except TimeoutError:
                    tabs_info.append({'title': 'Load timeout', 'url': page.url})
                except Exception as e:
                    logger.error(f'Error: {e}')
                    tabs_info.append({'title': 'Error encountered', 'url': page.url})
        return tabs_info


@getter_registry.register()
async def get_active_tab_info(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get title, url, and HTML content of the active Chrome tab."""
    active_tab_url = await _get_active_url(runtime, config)
    if active_tab_url is None:
        return None

    _navigation_timeout, _load_state_timeout, snapshot_timeout = _page_timeouts(config)
    try:
        async with _connect_cdp(runtime) as browser:
            for context in browser.contexts:
                for page in context.pages:
                    if page.url == active_tab_url:
                        return await _get_page_snapshot(page, snapshot_timeout_ms=snapshot_timeout)
    except Exception as exc:
        logger.warning('Failed to inspect active Chrome tab via CDP: %s', exc)

    # URL 已经从 UIA/CDP 拿到时，HTML/title 采集失败按空内容返回，避免把观测失败升级为 runner failed。
    return {'title': '', 'url': active_tab_url, 'content': ''}


@getter_registry.register()
async def get_active_tab_url_parse(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Parse URL query parameters from the active tab URL.

    config:
        'parse_keys': list of query param keys to extract
        'replace': optional dict mapping old_key -> new_key
        'or': optional dict mapping key -> fallback_key
    """
    active_tab_url = await _get_active_url(runtime, config)
    if active_tab_url is None:
        return None

    parsed_url = urlparse(active_tab_url)
    query_params = parse_qs(parsed_url.query)

    keys_of_interest = list(config['parse_keys'])
    extracted_params = {key: query_params.get(key, [''])[0] for key in keys_of_interest}

    if 'replace' in config:
        for key in config['replace']:
            value = extracted_params.pop(key)
            extracted_params[config['replace'][key]] = value
    if 'or' in config:
        for key in config['or']:
            extracted_params[key] = extracted_params[key] or extracted_params[config['or'][key]]

    return extracted_params


@getter_registry.register()
async def get_url_dashPart(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:  # noqa: N802
    """Extract a dash-separated segment from the active tab URL.

    config:
        'partIndex': index of the '/' separated part
        'needDeleteId': bool, strip '?...' suffix if True
        'returnType': 'string' or 'json'
        'key': (for json returnType) the key name
    """
    active_tab_url = await _get_active_url(runtime, config)
    if active_tab_url is None:
        return None

    dash_part = active_tab_url.split('/')[config['partIndex']]
    if config.get('needDeleteId'):
        dash_part = dash_part.split('?')[0]

    if config['returnType'] == 'string':
        return dash_part
    if config['returnType'] == 'json':
        return {config['key']: dash_part}
    return dash_part


@getter_registry.register()
async def get_pdf_from_url(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Navigate to a URL via CDP and export the page as PDF."""
    from nexus.extensions.tasks.waa.setup import cache_dir

    url = config['path']
    local_path = str(cache_dir / config['dest'])

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            async with _connect_cdp(runtime) as browser:
                page = await browser.new_page()
                await page.goto(url, timeout=60000, wait_until='domcontentloaded')
                await page.pdf(path=local_path)
            return local_path
        except Exception as exc:
            last_error = exc
            logger.warning('PDF expected-state generation failed on attempt %s/3: %s', attempt + 1, exc)
            await asyncio.sleep(2)

    # CDP 生成 expected PDF 时遇到过 Target closed；改用 VM 内 Chrome headless 兜底，避免 expected 侧失败污染 agent 分数。
    logger.warning('Falling back to VM Chrome PDF generation for %s after CDP failure: %s', url, last_error)
    try:
        return await _generate_pdf_with_vm_chrome(runtime, url, local_path)
    except Exception as exc:
        raise RuntimeError(f'Failed to generate expected PDF from {url}: {last_error}; fallback failed: {exc}') from exc
