"""Async setup orchestrator for WAA tasks.

Replaces WAA's SetupController.setup() — parses the config list from task JSON
and executes each setup action via the runtime.
"""

import asyncio
import json
import logging
import os
import tempfile

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from nexus.extensions.tasks.waa.registry import setup_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime
from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)

# Local cache directory for storing intermediate files (stdout/stderr captures, downloads, etc.)
# Mirrors WAA's env.cache_dir concept.
cache_dir: Path = Path(tempfile.mkdtemp(prefix='waa_cache_'))
download_cache_dir: Path = Path(
    os.environ.get('WAA_DOWNLOAD_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'nexus_waa_download_cache'))
)
download_cache_dir.mkdir(parents=True, exist_ok=True)


def _find_legacy_download_cache(cache_filename: str) -> Path | None:
    # 兼容旧版每次运行一个临时 waa_cache_* 目录的下载缓存，避免升级到持久缓存后重复拉取大文件。
    candidates: list[Path] = []
    for directory in Path(tempfile.gettempdir()).glob('waa_cache_*'):
        candidate = directory / cache_filename
        if candidate.is_file():
            candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _navigation_started(current_url: str, target_url: str) -> bool:
    if not current_url or current_url == 'about:blank':
        return False
    current = urlparse(current_url)
    target = urlparse(target_url)
    return bool(current.scheme and current.netloc) and (
        current.netloc == target.netloc or current_url.startswith(target_url)
    )


async def setup_environment(runtime: AGSRawWindowsAgentArenaRuntime, config_list: list[dict]) -> None:
    """Execute task environment initialization actions.

    Args:
        runtime: AGSRawWindowsAgentArenaRuntime instance.
        config_list: List of setup actions from task config JSON's "config" field.
            Each action has format: {"type": "launch", "parameters": {...}}
    """
    for cfg in config_list:
        config_type = cfg.get('type', '')
        parameters = cfg.get('parameters', {})

        handler = setup_registry.get(config_type)
        if handler is None:
            logger.warning(f'Unknown setup type: {config_type}, skipping')
            continue

        logger.debug(f'Setup: {config_type}({parameters})')
        await handler(runtime, **parameters)


async def close_all_and_cleanup(runtime: AGSRawWindowsAgentArenaRuntime) -> None:
    """Close all applications and clean up task files before a new task.

    This corresponds to WAA's env.reset() pre-cleanup:
    - setup_controller._close_all_setup()
    - setup_controller._clean_up_files()
    """
    try:
        await runtime.setup_close_all()
    except Exception:
        logger.warning('Failed to close all windows during cleanup', exc_info=True)

    try:
        await runtime.setup_clear_task_files()
    except Exception:
        logger.warning('Failed to clear task files during cleanup', exc_info=True)

    # Wait for cleanup to take effect
    await asyncio.sleep(2)


# ---- Individual setup handlers ----
# NOTE: All handlers accept **kwargs to silently ignore extra fields from task config
# JSON's "parameters" dict (e.g. WAA's _execute_setup originally had stdout/stderr/until
# fields that we don't use). This prevents TypeError on unexpected keyword arguments.


@setup_registry.register('set_default_browser')
async def _set_default_browser(runtime: AGSRawWindowsAgentArenaRuntime, browser: str, **kwargs: Any) -> None:
    """Set the default browser."""
    # Comment: this function doesn't work. It does set the registry key correctly, but looks like there's some security mechanism that prevents the key by itself from setting the default browser
    key_path = r'HKCU:Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice'
    setting_path = r'ProgId'

    browser_prog_ids = {
        'chrome': 'ChromeHTML',
        'edge': 'MSEdgeHTM',
    }
    assert browser.lower() in browser_prog_ids, (
        f"Invalid browser: {browser}. Choose from 'chrome', 'edge', 'firefox', 'ie'."
    )
    browser_prog_id = browser_prog_ids[browser.lower()]

    await runtime.execute(
        f"powershell -Command \"Set-ItemProperty -Path '{key_path}' -Name '{setting_path}' -Value '{browser_prog_id}'\""
    )


@setup_registry.register('download')
async def _download_setup(
    runtime: AGSRawWindowsAgentArenaRuntime, files: list[dict[str, str]] | None = None, **kwargs: Any
) -> None:
    """Download multiple files from URLs and upload to VM paths.

    Downloads locally first (with cache) then uploads to VM to avoid remote VM network issues.

    files format: [{"url": "...", "path": "..."}]
    """
    import os
    import uuid

    if not files:
        return
    for f in files:
        url = f.get('url', '')
        path = f.get('path', '')
        if not url or not path:
            raise ValueError(f'Setup Download - Invalid URL ({url}) or path ({path}).')

        # Cache locally: same URL+basename won't re-download
        # GitHub raw/CDN 在全量并发时出现过短时 400/5xx；先落本地持久缓存，再上传 VM，
        # 可以把外部网络波动和 VM 内部下载能力从任务 setup 中解耦。
        cache_filename = f'{uuid.uuid5(uuid.NAMESPACE_URL, url)}_{os.path.basename(path)}'
        cache_path = download_cache_dir / cache_filename

        if not cache_path.exists():
            try:
                await _download_to_local_then_upload(runtime, url, path, local_cache_path=cache_path)
                continue
            except Exception:
                fallback = _find_legacy_download_cache(cache_filename)
                if fallback is None:
                    raise
                logger.warning(
                    'Download failed for %s; using legacy cached file %s and seeding %s',
                    url,
                    fallback,
                    cache_path,
                    exc_info=True,
                )
                await asyncio.to_thread(cache_path.write_bytes, fallback.read_bytes())

        logger.debug(f'Using cached file: {cache_path}, uploading to VM at {path}')
        await runtime.setup_upload(path, cache_path.read_bytes())


@setup_registry.register('upload_file')
async def _upload_file_setup(
    runtime: AGSRawWindowsAgentArenaRuntime, files: list[dict[str, str]] | None = None, **kwargs: Any
) -> None:
    """Upload files to the VM.

    files format: [{"local_path": "...", "path": "..."}]
    """
    import os

    if not files:
        return
    for f in files:
        local_path = f.get('local_path', '')
        path = f.get('path', '')
        if local_path and path:
            if not os.path.exists(local_path):
                raise ValueError(f'Setup Upload - Invalid Local path {local_path}.')
            with open(local_path, 'rb') as fp:
                file_data = fp.read()
            await runtime.setup_upload(path, file_data)


@setup_registry.register('change_wallpaper')
async def _change_wallpaper_setup(runtime: AGSRawWindowsAgentArenaRuntime, path: str = '', **kwargs: Any) -> None:
    """Change the desktop wallpaper."""
    if not path:
        raise ValueError(f'Setup Wallpaper - Invalid path ({path})')
    await runtime.setup_change_wallpaper(path)


@setup_registry.register('open')
async def _open_setup(runtime: AGSRawWindowsAgentArenaRuntime, path: str = '', **kwargs: Any) -> None:
    """Open a file with the default application."""
    if not path:
        raise ValueError(f'Setup Open - Invalid path ({path}).')
    await runtime.setup_open_file(path)
    await asyncio.sleep(2)


@setup_registry.register('launch')
async def _launch_setup(
    runtime: AGSRawWindowsAgentArenaRuntime, command: str | list[str] = '', shell: bool = False, **kwargs: Any
) -> None:
    """Launch an application (non-blocking)."""
    if not command:
        raise ValueError('Empty command to launch')
    if not shell and isinstance(command, str) and len(command.split()) > 1:
        logger.warning('Command should be a list of strings. Now it is a string. Will split it by space.')
        command = command.split()

    await runtime.setup_launch(command, shell=shell)


@setup_registry.register('execute')
async def _execute_setup(
    runtime: AGSRawWindowsAgentArenaRuntime,
    command: list[str] | None = None,
    stdout: str = '',
    stderr: str = '',
    shell: bool = False,
    until: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Execute a command (blocking, waits for completion).

    Supports retry-until semantics: if `until` is provided, the command is
    re-executed until the condition is met or 5 consecutive failures occur.

    until format: {"returncode": 0} or {"stdout": "expected"} or {"stderr": "expected"}
    """
    if not command:
        raise ValueError('Empty command to execute')

    until = until or {}
    nb_failings = 0

    while True:
        try:
            results = await runtime.setup_execute(command, shell=shell)
            if isinstance(results, dict):
                if stdout:
                    await asyncio.to_thread((cache_dir / stdout).write_text, results.get('output', ''))
                if stderr:
                    await asyncio.to_thread((cache_dir / stderr).write_text, results.get('error', ''))
            else:
                results = None
                nb_failings += 1
        except Exception:
            logger.error('execute_setup failed for command: %s', command, exc_info=True)
            results = None
            nb_failings += 1

        # Check termination conditions
        if not until:
            break

        if results is not None:
            if 'returncode' in until and results.get('returncode') == until['returncode']:
                break
            if 'stdout' in until and until['stdout'] in results.get('output', ''):
                break
            if 'stderr' in until and until['stderr'] in results.get('error', ''):
                break

        if nb_failings >= 5:
            break

        await asyncio.sleep(0.3)


@setup_registry.register('command')
async def _command_setup(runtime: AGSRawWindowsAgentArenaRuntime, command: list[str], **kwargs: Any) -> None:
    """Execute a command (alias for execute)."""
    await _execute_setup(runtime, command=command, **kwargs)


@setup_registry.register('sleep')
async def _sleep_setup(runtime: AGSRawWindowsAgentArenaRuntime, seconds: float, **kwargs: Any) -> None:
    """Wait for a specified duration (seconds)."""
    await asyncio.sleep(seconds)


@setup_registry.register('activate_window')
async def _activate_window_setup(
    runtime: AGSRawWindowsAgentArenaRuntime,
    window_name: str,
    strict: bool = False,
    by_class: bool = False,
    **kwargs: Any,
) -> None:
    """Activate (bring to front) a specific window."""
    if not window_name:
        raise ValueError(f'Setup Open - Invalid path ({window_name}).')
    await runtime.setup_activate_window(window_name, strict=strict, by_class=by_class)


@setup_registry.register('close_all')
async def _close_all_setup(runtime: AGSRawWindowsAgentArenaRuntime, **kwargs: Any) -> None:
    """Close all applications."""
    await runtime.setup_close_all()


@setup_registry.register('close_window')
async def _close_window_setup(
    runtime: AGSRawWindowsAgentArenaRuntime,
    window_name: str = '',
    strict: bool = False,
    by_class: bool = False,
    **kwargs: Any,
) -> None:
    """Close a specific window."""
    if not window_name:
        raise ValueError(f'Setup Open - Invalid path ({window_name}).')

    await runtime.setup_close_window(window_name, strict=strict, by_class=by_class)


@setup_registry.register('create_folder')
async def _create_folder_setup(runtime: AGSRawWindowsAgentArenaRuntime, path: str = '', **kwargs: Any) -> None:
    """Create a folder on the VM."""
    if not path:
        raise ValueError(f'Setup Create Folder - Invalid path ({path}).')
    await runtime.setup_create_folder(path)


@setup_registry.register('create_file')
async def _create_file_setup(
    runtime: AGSRawWindowsAgentArenaRuntime, path: str = '', content: str = '', **kwargs: Any
) -> None:
    """Create a file on the VM."""
    if not path:
        raise ValueError(f'Setup Create File - Invalid path ({path}).')
    await runtime.setup_create_file(path, content)


@setup_registry.register('recycle_file')
async def _recycle_file_setup(runtime: AGSRawWindowsAgentArenaRuntime, path: str = '', **kwargs: Any) -> None:
    """Move a file to the Recycle Bin (alias for recycle with 'path' parameter)."""
    if not path:
        raise ValueError(f'Setup recycle file - Invalid path ({path}).')
    await runtime.setup_recycle(path)


@setup_registry.register('chrome_open_tabs')
async def _chrome_open_tabs_setup(
    runtime: AGSRawWindowsAgentArenaRuntime, urls_to_open: list[str] | None = None, **kwargs: Any
) -> None:
    """Open tabs in Chrome via Playwright CDP connection.

    Connects to the existing Chrome instance via remote debugging port and opens
    new tabs, matching the original WAA setup_controller behavior.
    """
    if not urls_to_open:
        return

    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright

    cdp_url = await runtime.chromium_cdp_url()
    logger.debug('Connect to Chrome @: %s', cdp_url)

    async with async_playwright() as pw:
        browser = None
        for attempt in range(15):
            if attempt > 0:
                await asyncio.sleep(5)
            try:
                browser = await pw.chromium.connect_over_cdp(cdp_url, headers=runtime.handle.headers)
                break
            except Exception as e:
                if attempt < 14:
                    logger.error(f'Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}')
                    continue
                logger.error(f'Failed to connect after multiple attempts: {e}')
                raise

        if not browser:
            return

        logger.debug('Opening %s...', urls_to_open)
        context = browser.contexts[0]
        for i, url in enumerate(urls_to_open):
            page = await context.new_page()
            try:
                # live-web 页面在全量并发时经常卡在 load/networkidle；setup 只需要确认导航已开始，
                # 后续页面内容由 agent/evaluator 处理，因此用 commit + 短 domcontentloaded 等待。
                await page.goto(url, timeout=30000, wait_until='commit')
                try:
                    await page.wait_for_load_state('domcontentloaded', timeout=15000)
                except PlaywrightTimeoutError:
                    logger.warning(
                        'Continuing after Chrome navigation commit without domcontentloaded: url=%s current_url=%s',
                        url,
                        page.url,
                    )
            except PlaywrightTimeoutError as e:
                if _navigation_started(page.url, url):
                    logger.warning(
                        'Continuing after Chrome navigation timeout because page left about:blank: url=%s current_url=%s',
                        url,
                        page.url,
                    )
                else:
                    raise RuntimeError(f'Failed to open {url}: {e}') from e
            except Exception as e:
                raise RuntimeError(f'Failed to open {url}: {e}') from e
            logger.debug(f'Opened tab {i + 1}: {url}')

            if i == 0:
                default_page = context.pages[0]
                await default_page.close()


@setup_registry.register('edge_open_tabs')
async def _edge_open_tabs_setup(
    runtime: AGSRawWindowsAgentArenaRuntime, urls_to_open: list[str] | None = None, **kwargs: Any
) -> None:
    """Open tabs in Edge via Playwright CDP (same as Chrome - Chromium-based)."""
    await _chrome_open_tabs_setup(runtime, urls_to_open=urls_to_open, **kwargs)


@setup_registry.register('chrome_close_tabs')
async def _chrome_close_tabs_setup(
    runtime: AGSRawWindowsAgentArenaRuntime, urls_to_close: list[str] | None = None, **kwargs: Any
) -> None:
    """Close specific Chrome tabs via Playwright CDP connection.

    Connects to Chrome's remote debugging port, iterates pages in the context,
    and closes those matching the target URLs.
    """
    if not urls_to_close:
        return

    await asyncio.sleep(5)  # Wait for Chrome to finish launching

    from urllib.parse import urlparse

    from playwright.async_api import async_playwright

    def _urls_match(url1: str, url2: str) -> bool:
        """Compare URLs ignoring trailing slashes and scheme differences."""
        p1 = urlparse(url1)
        p2 = urlparse(url2)
        return p1.netloc == p2.netloc and p1.path.rstrip('/') == p2.path.rstrip('/')

    cdp_url = await runtime.chromium_cdp_url()
    async with async_playwright() as pw:
        browser = None
        for attempt in range(15):
            try:
                browser = await pw.chromium.connect_over_cdp(cdp_url, headers=runtime.handle.headers)
                break
            except Exception as e:
                if attempt < 14:
                    logger.error(f'Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}')
                    await asyncio.sleep(5)
                else:
                    logger.error(f'Failed to connect after multiple attempts: {e}')
                    raise

        if not browser:
            return

        context = browser.contexts[0]
        for i, url in enumerate(urls_to_close):
            for page in context.pages:
                # if two urls are the same, close the tab
                if _urls_match(page.url, url):
                    await page.close()
                    logger.debug(f'Closed tab {i + 1}: {url}')
                    break


async def _download_to_local_then_upload(
    runtime: AGSRawWindowsAgentArenaRuntime,
    url: str,
    remote_path: str,
    *,
    local_cache_path: Path | None = None,
) -> None:
    """Download a file locally, then upload to the VM.

    Downloads via nexus host network to avoid remote VM network issues.
    If local_cache_path is provided, saves the downloaded file there for caching.
    """
    import aiohttp

    logger.debug(f'Downloading {url} ...')
    max_retries = 3
    data: bytes | None = None
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await http_client.get(url, timeout=aiohttp.ClientTimeout(total=120))
            response.raise_for_status()
            data = response.content
            break
        except Exception as e:
            last_error = e
            logger.error(f'Failed to download {url}: {e}. Retrying... ({max_retries - attempt - 1} left)')
    if data is None:
        raise RuntimeError(f'Failed to download {url}. No retries left. Error: {last_error}')

    # Save to local cache if requested
    if local_cache_path is not None:
        await asyncio.to_thread(local_cache_path.write_bytes, data)

    logger.debug(f'Downloaded {len(data)} bytes, uploading to VM at {remote_path}')
    await runtime.setup_upload(remote_path, data)


@setup_registry.register('login')
async def _login_setup(runtime: AGSRawWindowsAgentArenaRuntime, **kwargs: Any) -> None:
    """Login to a website.
    @args:
            config(Dict[str, Any]): contain keys
                settings_file(str): path to the settings file
                platform(str): platform to login, implemented platforms include:
                    googledrive: https://drive.google.com/drive/my-drive
    """
    from playwright.async_api import async_playwright

    cdp_url = await runtime.chromium_cdp_url()
    async with async_playwright() as pw:
        browser = None
        for attempt in range(15):
            try:
                browser = await pw.chromium.connect_over_cdp(cdp_url, headers=runtime.handle.headers)
                break
            except Exception as e:
                if attempt < 14:
                    logger.error(f'Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}')
                    await asyncio.sleep(5)
                else:
                    logger.error(f'Failed to connect after multiple attempts: {e}')
                    raise
        if not browser:
            return

        context = browser.contexts[0]
        platform = kwargs['platform']

        if platform == 'googledrive':
            url = 'https://drive.google.com/drive/my-drive'
            page = await context.new_page()
            try:
                await page.goto(url, timeout=60000)
            except Exception as e:
                raise RuntimeError(f'Failed to open {url}: {e}') from e
            logger.debug(f'Opened new page: {url}')
            with open(kwargs['settings_file']) as f:
                settings = json.load(f)
            email, password = settings['email'], settings['password']

            try:
                await page.wait_for_selector('input[type="email"]', state='visible', timeout=3000)
                await page.fill('input[type="email"]', email)
                await page.click('#identifierNext > div > button')
                await page.wait_for_selector('input[type="password"]', state='visible', timeout=5000)
                await page.fill('input[type="password"]', password)
                await page.click('#passwordNext > div > button')
                await page.wait_for_load_state('load', timeout=5000)
            except TimeoutError as e:
                raise TimeoutError('Timeout waiting for Google Drive login page to load') from e
        else:
            raise NotImplementedError(
                f'Login platform {platform!r} is not supported. Only "googledrive" is implemented.'
            )


async def _build_history_db(history: list[dict]) -> Path:
    """Build a Chrome/Edge History SQLite DB locally from a template.

    Creates a copy of the empty history template in cache_dir, inserts all
    history items with proper Chrome timestamps (microseconds since 1601-01-01).

    Returns the path to the populated DB file.
    """
    import shutil
    import sqlite3

    from datetime import datetime, timedelta

    def _build_sync() -> Path:
        # Use the bundled empty history template
        template_path = Path(__file__).parent / 'assets' / 'history_empty.sqlite'
        db_path = cache_dir / 'history_new.sqlite'
        shutil.copyfile(template_path, db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for item in history:
            url = item['url']
            title = item['title']
            visit_time = datetime.now() - timedelta(seconds=item['visit_time_from_now_in_seconds'])

            # Chrome uses microseconds from 1601-01-01 as timestamp
            epoch_start = datetime(1601, 1, 1)
            chrome_timestamp = int((visit_time - epoch_start).total_seconds() * 1000000)

            cursor.execute(
                'INSERT INTO urls (url, title, visit_count, typed_count, last_visit_time, hidden) VALUES (?, ?, ?, ?, ?, ?)',
                (url, title, 1, 0, chrome_timestamp, 0),
            )
            url_id = cursor.lastrowid
            cursor.execute(
                'INSERT INTO visits (url, visit_time, from_visit, transition, segment_id, visit_duration) VALUES (?, ?, ?, ?, ?, ?)',
                (url_id, chrome_timestamp, 0, 805306368, 0, 0),
            )

        conn.commit()
        conn.close()
        logger.debug('Fake browsing history added successfully.')
        return db_path

    return await asyncio.to_thread(_build_sync)


@setup_registry.register('update_browse_history')
async def _update_browse_history_setup(
    runtime: AGSRawWindowsAgentArenaRuntime, history: list[dict] | None = None, **kwargs: Any
) -> None:
    """Insert fake browsing history into Chrome's History database.

    Builds a SQLite History DB locally from a template, then uploads to the VM.
    """
    if not history:
        return

    db_path = await _build_history_db(history)

    # Resolve Chrome History path on VM
    result = await runtime.execute(
        [
            'python',
            '-c',
            'import os; print(os.path.join(os.getenv("USERPROFILE"), "AppData", "Local", "Google", "Chrome", "User Data", "Default", "History"))',
        ],
        shell=False,
    )
    if isinstance(result, dict):
        chrome_history_path = result.get('output', '').strip().strip('"')
    else:
        chrome_history_path = r'C:\Users\Docker\AppData\Local\Google\Chrome\User Data\Default\History'

    db_data = await asyncio.to_thread(db_path.read_bytes)
    await runtime.setup_upload(chrome_history_path, db_data)


@setup_registry.register('update_browse_history_edge')
async def _update_browse_history_edge_setup(
    runtime: AGSRawWindowsAgentArenaRuntime, history: list[dict] | None = None, **kwargs: Any
) -> None:
    """Insert fake browsing history into Edge's History database.

    Builds a SQLite History DB locally from a template, then uploads to the VM.
    """
    if not history:
        return

    db_path = await _build_history_db(history)

    # Resolve Edge History path on VM
    result = await runtime.execute(
        [
            'python',
            '-c',
            'import os; print(os.path.join(os.getenv("USERPROFILE"), "AppData", "Local", "Microsoft", "Edge", "User Data", "Default", "History"))',
        ],
        shell=False,
    )
    if isinstance(result, dict):
        edge_history_path = result.get('output', '').strip().strip('"')
    else:
        edge_history_path = r'C:\Users\Docker\AppData\Local\Microsoft\Edge\User Data\Default\History'

    db_data = await asyncio.to_thread(db_path.read_bytes)
    await runtime.setup_upload(edge_history_path, db_data)
