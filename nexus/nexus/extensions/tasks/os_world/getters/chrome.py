"""Chrome and browser-state getters for OSWorld evaluation."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import re
import sqlite3

from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp

from nexus.extensions.tasks.os_world.evaluation.errors import EvaluatorInfrastructureError
from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry
from nexus.extensions.tasks.os_world.execution.cdp import chromium_cdp_connection, connect_chromium
from nexus.extensions.tasks.os_world.execution.commands import command_to_shell
from nexus.extensions.tasks.os_world.execution.setup import add_chrome_proxy_arg, osworld_runtime_use_proxy
from nexus.extensions.tasks.os_world.getters.misc import get_accessibility_tree
from nexus.extensions.tasks.os_world.getters.runtime import (
    cache_dir_from_config,
    download_url,
    get_file_bytes,
    response_stdout,
    run_bash,
    write_bytes,
)
from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)

_ACCESSIBILITY_NS = {
    'st': 'https://accessibility.ubuntu.example.org/ns/state',
    'attr': 'https://accessibility.ubuntu.example.org/ns/attributes',
    'cp': 'https://accessibility.ubuntu.example.org/ns/component',
    'doc': 'https://accessibility.ubuntu.example.org/ns/document',
    'docattr': 'https://accessibility.ubuntu.example.org/ns/document/attributes',
    'txt': 'https://accessibility.ubuntu.example.org/ns/text',
    'val': 'https://accessibility.ubuntu.example.org/ns/value',
    'act': 'https://accessibility.ubuntu.example.org/ns/action',
}


def _chrome_profile_candidates(file_name: str) -> list[str]:
    return [
        f'/home/user/.config/google-chrome/Default/{file_name}',
        f'/home/user/.config/google-chrome/Profile 1/{file_name}',
        f'/home/user/snap/chromium/common/chromium/Default/{file_name}',
        f'/home/ubuntu/.config/google-chrome/Default/{file_name}',
    ]


def _chrome_local_state_candidates() -> list[str]:
    return [
        '/home/user/.config/google-chrome/Local State',
        '/home/user/snap/chromium/common/chromium/Local State',
        '/home/ubuntu/.config/google-chrome/Local State',
    ]


async def _read_first_json(runtime: Any, paths: list[str]) -> dict[str, Any]:
    for path in paths:
        content = await get_file_bytes(runtime, path)
        if not content:
            continue
        try:
            return json.loads(content.decode('utf-8'))
        except Exception:
            continue
    return {}


async def _download_first_file(runtime: Any, paths: list[str], dest: Path) -> str | None:
    for path in paths:
        content = await get_file_bytes(runtime, path)
        if content is None:
            continue
        await write_bytes(dest, content)
        return str(dest)
    return None


async def _cdp_connection(runtime: Any) -> tuple[str, dict[str, str] | None] | None:
    return await chromium_cdp_connection(runtime)


async def _launch_chrome_for_cdp(runtime: Any) -> None:
    use_proxy = osworld_runtime_use_proxy(runtime)
    google_chrome = command_to_shell(
        add_chrome_proxy_arg(['google-chrome', '--remote-debugging-port=1337'], use_proxy=use_proxy)
    )
    chromium = command_to_shell(add_chrome_proxy_arg(['chromium', '--remote-debugging-port=1337'], use_proxy=use_proxy))
    command = f'command -v google-chrome >/dev/null && {google_chrome} || {chromium}'
    await run_bash(runtime, f'nohup bash -lc {command!r} >/tmp/osworld-chrome-cdp.log 2>&1 &', timeout=10)
    await asyncio.sleep(3)


async def _browser_page_infos_via_playwright(runtime: Any) -> list[dict[str, str]]:
    async with connect_chromium(runtime) as browser:
        infos: list[dict[str, str]] = []
        for context in browser.contexts:
            for page in context.pages:
                if page.is_closed():
                    continue
                infos.append({'title': await page.title(), 'url': page.url, 'content': await page.content()})
        return infos


async def _browser_page_infos(runtime: Any) -> list[dict[str, str]]:
    async def _collect_once() -> list[dict[str, str]]:
        connection = await _cdp_connection(runtime)
        if not connection:
            return []
        endpoint, headers = connection
        if urlparse(endpoint).scheme not in {'http', 'https'}:
            return await _browser_page_infos_via_playwright(runtime)

        response = await http_client.get(
            f'{endpoint.rstrip("/")}/json/list',
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        )
        response.raise_for_status()
        targets = response.json()
        if not isinstance(targets, list):
            return []
        return [
            {
                'title': str(target.get('title') or ''),
                'url': str(target.get('url') or ''),
                'content': '',
            }
            for target in targets
            if isinstance(target, dict) and target.get('type') == 'page'
        ]

    try:
        return await _collect_once()
    except Exception:
        await _launch_chrome_for_cdp(runtime)
        try:
            return await _collect_once()
        except Exception:
            return []


async def _page_info(runtime: Any, url: str) -> dict[str, str]:
    async def _load_once() -> dict[str, str]:
        async with connect_chromium(runtime) as browser:
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = await context.new_page()
            # Use the standard getter's load event boundary.
            await page.goto(url, wait_until='load', timeout=60000)
            return {'title': await page.title(), 'url': page.url, 'content': await page.content()}

    try:
        return await _load_once()
    except Exception:
        await _launch_chrome_for_cdp(runtime)
        try:
            return await _load_once()
        except Exception:
            return {'title': '', 'url': url, 'content': ''}


def _normalize_url_text(text: str, goto_prefix: str) -> str:
    stripped = text.strip()
    if re.match(r'^[a-zA-Z][a-zA-Z0-9+\-.]*:', stripped):
        return stripped
    if goto_prefix.endswith('www.') and stripped.lower().startswith('www.'):
        stripped = stripped[4:]
    return f'{goto_prefix}{stripped}'


async def _active_url_from_frame_title(runtime: Any, root: Any) -> str | None:
    from lxml.cssselect import CSSSelector

    selectors = [
        'application[name=Google\\ Chrome] frame',
        'application[name=Chromium] frame',
    ]
    frames = []
    for selector_text in selectors:
        selector = CSSSelector(selector_text, namespaces=_ACCESSIBILITY_NS)
        frames = selector(root)
        if frames:
            break
    if not frames:
        return None

    active = [item for item in frames if item.get(f'{{{_ACCESSIBILITY_NS["st"]}}}active') == 'true']
    frame = active[-1] if active else frames[-1]
    frame_titles = [line.strip() for line in str(frame.get('name') or '').splitlines() if line.strip()]
    if not frame_titles:
        return None

    page_infos = await _browser_page_infos(runtime)
    for frame_title in reversed(frame_titles):
        for info in page_infos:
            page_title = str(info.get('title') or '').strip()
            if page_title and (page_title == frame_title or page_title in frame_title or frame_title in page_title):
                return str(info.get('url') or '') or None
    return None


@getter_registry.register('active_url_from_accessTree')
async def get_active_url_from_access_tree(runtime: Any, config: dict[str, Any]) -> str | None:
    tree = await get_accessibility_tree(runtime, config)
    if not tree:
        return None
    try:
        import lxml.etree

        from lxml.cssselect import CSSSelector

        root = lxml.etree.fromstring(tree.encode('utf-8') if isinstance(tree, str) else tree)
        selectors = [
            'application[name=Google\\ Chrome] entry[name=Address\\ and\\ search\\ bar]',
            'application[name=Chromium] entry[name=Address\\ and\\ search\\ bar]',
            'entry[name=Address\\ and\\ search\\ bar]',
        ]
        elements = []
        for selector_text in selectors:
            selector = CSSSelector(selector_text, namespaces=_ACCESSIBILITY_NS)
            elements = [element for element in selector(root) if element.text]
            if elements:
                break
        if elements:
            focused = [item for item in elements if item.get(f'{{{_ACCESSIBILITY_NS["st"]}}}focused') == 'true']
            active = focused[-1] if focused else elements[-1]
            return _normalize_url_text(str(active.text), str(config.get('goto_prefix', 'https://') or ''))
        frame_url = await _active_url_from_frame_title(runtime, root)
        return frame_url or None
    except Exception:
        return None


@getter_registry.register()
async def get_active_tab_info(runtime: Any, config: dict[str, Any]) -> dict[str, str] | None:
    active_url = await get_active_url_from_access_tree(runtime, config)
    if not active_url:
        return None
    infos = await _browser_page_infos(runtime)
    normalized_active = unquote(active_url).rstrip('/')
    for info in infos:
        if unquote(info.get('url', '')).rstrip('/') == normalized_active:
            return info
    return await _page_info(runtime, active_url)


async def _active_tab_body_text(runtime: Any, active_url: str) -> str:
    def _normalize(url: str) -> str:
        return unquote(url or '').rstrip('/')

    async with connect_chromium(runtime) as browser:
        target_page = None
        for context in getattr(browser, 'contexts', []) or []:
            for page in getattr(context, 'pages', []) or []:
                if _normalize(getattr(page, 'url', '')) == _normalize(active_url):
                    target_page = page
                    break
            if target_page is not None:
                break

        page_to_close = None
        if target_page is None:
            if hasattr(browser, 'new_page'):
                target_page = await browser.new_page()
            else:
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                target_page = await context.new_page()
            page_to_close = target_page
            target_page.set_default_timeout(60000)
            await target_page.goto(active_url, wait_until='load', timeout=60000)

        try:
            return await target_page.evaluate("document.body ? document.body.innerText : ''") or ''
        finally:
            if page_to_close is not None:
                await page_to_close.close()


async def _active_tab_live_input_values(runtime: Any, active_url: str, config: dict[str, Any]) -> dict[str, str]:
    def _normalize(url: str) -> str:
        return unquote(url or '').rstrip('/')

    async with connect_chromium(runtime) as browser:
        target_page = None
        for context in getattr(browser, 'contexts', []) or []:
            for page in getattr(context, 'pages', []) or []:
                if _normalize(getattr(page, 'url', '')) == _normalize(active_url):
                    target_page = page
                    break
            if target_page is not None:
                break

        page_to_close = None
        if target_page is None:
            if hasattr(browser, 'new_page'):
                target_page = await browser.new_page()
            else:
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                target_page = await context.new_page()
            page_to_close = target_page
            target_page.set_default_timeout(60000)
            await target_page.goto(active_url, wait_until='load', timeout=60000)

        try:
            return await _active_tab_input_values_from_page(target_page, config)
        finally:
            if page_to_close is not None:
                await page_to_close.close()


async def _active_tab_input_values_from_page(page: Any, config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for xpath, key in config.get('inputObject', {}).items():
        try:
            if hasattr(page, 'is_closed') and page.is_closed():
                result[key] = ''
                continue
            inputs = page.locator(f'xpath={xpath}')
            if await inputs.count() <= 0:
                result[key] = ''
                continue
            first = inputs.first
            if callable(first):
                first = first()
            value = await first.input_value()
            result[key] = value.strip() if value else ''
        except Exception:
            result[key] = ''
    return result


@getter_registry.register()
async def get_active_tab_url_and_body_text(runtime: Any, config: dict[str, Any]) -> dict[str, str]:
    active_url = await get_active_url_from_access_tree(runtime, config)
    if not isinstance(active_url, str):
        return {'url': '', 'body': ''}

    try:
        body = await _active_tab_body_text(runtime, active_url)
    except Exception:
        await _launch_chrome_for_cdp(runtime)
        try:
            body = await _active_tab_body_text(runtime, active_url)
        except Exception:
            body = ''

    return {'url': active_url, 'body': body}


@getter_registry.register()
async def get_open_tabs_info(runtime: Any, config: dict[str, Any]) -> list[dict[str, str]]:
    infos = await _browser_page_infos(runtime)
    return [{'title': item.get('title', ''), 'url': item.get('url', '')} for item in infos]


@getter_registry.register()
async def get_page_info(runtime: Any, config: dict[str, Any]) -> dict[str, str]:
    return await _page_info(runtime, str(config['url']))


@getter_registry.register()
async def get_active_tab_html_parse(runtime: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    info = await get_active_tab_info(runtime, config)
    if not info:
        return None
    if config.get('category') == 'input' and info.get('url'):
        live_input = await _active_tab_input_values(runtime, info.get('url', ''), config)
        if live_input is not None:
            return live_input
    if not info.get('content'):
        normalized_active = unquote(info.get('url', '')).rstrip('/')
        try:
            page_infos = await _browser_page_infos_via_playwright(runtime)
        except Exception:
            page_infos = []
        matching_info = next(
            (
                page_info
                for page_info in page_infos
                if unquote(page_info.get('url', '')).rstrip('/') == normalized_active
            ),
            None,
        )
        info = matching_info or await _page_info(runtime, info.get('url', ''))
    return await asyncio.to_thread(_parse_html_config, info.get('content', ''), info.get('url', ''), config)


async def _active_tab_input_values(runtime: Any, active_url: str, config: dict[str, Any]) -> dict[str, str] | None:
    try:
        return await _active_tab_live_input_values(runtime, active_url, config)
    except Exception:
        await _launch_chrome_for_cdp(runtime)
        try:
            return await _active_tab_live_input_values(runtime, active_url, config)
        except Exception:
            return None


def _html_text_content(element: Any) -> str:
    return element.get_text().strip()


def _parse_html_config(html: str, page_url: str, config: dict[str, Any]) -> dict[str, Any]:
    from bs4 import BeautifulSoup
    from lxml import html as lxml_html

    soup = BeautifulSoup(html, 'html.parser')
    result: dict[str, Any] = {}
    category = config.get('category')

    if category in {'class', 'class&url'}:
        for class_name, object_dict in config.get('class_multiObject', {}).items():
            texts = [_html_text_content(item) for item in soup.select(f'.{class_name}')]
            if isinstance(object_dict, dict):
                for order_key, key in object_dict.items():
                    idx = int(order_key)
                    result[key] = texts[idx] if idx < len(texts) else ''
            else:
                lowered = [text.lower() for text in texts]
                for key in object_dict:
                    result[str(key).lower()] = str(key).lower() in lowered
                result['is_other_exist'] = any(
                    all(text.lower() not in str(key).lower() for key in object_dict) for text in texts
                )

        for class_name, object_dict in config.get('class_multiObject_child', {}).items():
            elements = soup.select(f'.{class_name}')
            texts = []
            for element in elements:
                direct = [
                    str(child).strip()
                    for child in element.children
                    if getattr(child, 'name', None) is None and str(child).strip()
                ]
                texts.extend(direct)
            for order_key, key in object_dict.items():
                idx = int(order_key)
                result[key] = texts[idx] if idx < len(texts) else ''

        for class_name, object_dict in config.get('class_multiObject_only_child', {}).items():
            texts = [_html_text_content(item) for item in soup.select(f'.{class_name} h3')]
            for order_key, key in object_dict.items():
                idx = int(order_key)
                result[key] = texts[idx] if idx < len(texts) else ''

        for class_name, object_list in config.get('class_multiObject_search_exist', {}).items():
            texts = [_html_text_content(item) for item in soup.select(f'.{class_name}')]
            for key in object_list:
                if key != 'is_other_exist':
                    result[key] = key in texts
            if 'is_other_exist' in object_list:
                result['is_other_exist'] = any(text not in object_list for text in texts)

        for class_name, key in config.get('class_singleObject', {}).items():
            elements = soup.select(f'.{class_name}')
            result[key] = _html_text_content(elements[0]) if elements else ''

        for class_name, object_list in config.get('class_multiObject_li', {}).items():
            texts = [_html_text_content(item) for item in soup.select(f'.{class_name} li.catAllProducts span')]
            lowered = [text.lower() for text in texts]
            for key in object_list:
                result[str(key).lower()] = str(key).lower() in lowered
            result.setdefault(
                'is_other_exist',
                any(all(text.lower() not in str(key).lower() for key in object_list) for text in texts),
            )

        for key in config.get('url_include_expected', []):
            result.setdefault(str(key).lower(), str(key).lower() in page_url.lower())
        for key, value in config.get('url_include_expected_multichoice', {}).items():
            result.setdefault(str(value).lower(), str(key).lower() in page_url.lower())

    elif category == 'xpath':
        doc = lxml_html.fromstring(html)
        for xpath, key in config.get('xpathObject', {}).items():
            matches = doc.xpath(xpath)
            result[key] = matches[0].text_content().strip() if matches else ''
    elif category == 'input':
        doc = lxml_html.fromstring(html)
        for xpath, key in config.get('inputObject', {}).items():
            matches = doc.xpath(xpath)
            result[key] = matches[0].get('value', '').strip() if matches else ''
    elif category == 'label':
        for label, key in config.get('labelObject', {}).items():
            match = soup.find(string=re.compile(re.escape(label)))
            result[key] = str(match).strip() if match else ''
    return result


@getter_registry.register()
async def get_active_tab_url_parse(runtime: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    active_url = await get_active_url_from_access_tree(runtime, config)
    if not active_url:
        return None
    query = parse_qs(urlparse(active_url).query)
    extracted = {key: query.get(key, [''])[0] for key in config['parse_keys']}
    for old_key, new_key in config.get('replace', {}).items():
        extracted[new_key] = extracted.pop(old_key)
    if config.get('split_list', False):
        extracted = {key: value.split(',') for key, value in extracted.items()}
    return extracted


@getter_registry.register('url_dashPart')
async def get_url_dash_part(runtime: Any, config: dict[str, Any]) -> str | dict[str, str] | None:
    active_url = await get_active_url_from_access_tree(runtime, config)
    if not active_url:
        return None
    parts = active_url.split('/')
    index = int(config['partIndex'])
    if abs(index) > len(parts):
        return None
    value = parts[index]
    if config.get('needDeleteId', False):
        value = value.split('?', 1)[0]
    if config['returnType'] == 'json':
        return {config['key']: value}
    return value


@getter_registry.register()
async def get_url_path_parse(runtime: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    active_url = await get_active_url_from_access_tree(runtime, config)
    if not active_url:
        return None
    return _parse_macys_product_url(active_url, config['parse_keys'])


def _parse_macys_product_url(active_url: str, parse_keys: list[str]) -> dict[str, Any]:
    path = unquote(urlparse(active_url).path)
    parts = path.strip('/').split('/')
    normalized_parts = [part.lower() for part in parts]
    normalized_path = '/'.join(normalized_parts)

    shirts_flag = 'shirts' in normalized_parts or 'shirts' in normalized_path
    short_sleeve_flag = 'short-sleeve' in normalized_path or 'short sleeve' in normalized_path
    result: dict[str, Any] = {
        'mens_clothing': True
        if (
            'mens-clothing' in normalized_path
            or "men's" in normalized_path
            or 'men-' in normalized_path
            or ('mens' in normalized_parts and 'clothing' in normalized_parts)
            or 'men' in normalized_parts
        )
        else None,
    }
    key_values: dict[str, Any] = {}
    for left, right in zip(parts, parts[1:], strict=False):
        if ',' not in left or ',' not in right:
            continue
        for key, value in zip(left.split(','), right.split(','), strict=False):
            parsed_key = key.strip()
            parsed_value = value.strip()
            if parsed_key == 'Price_discount_range':
                key_values[parsed_key] = [item.strip() for item in parsed_value.split('|')] if parsed_value else None
            else:
                key_values[parsed_key] = parsed_value if parsed_value else None
            if parsed_key == 'Product_department' and parsed_value.lower() in {'shirts', 'shirt'}:
                shirts_flag = True
            if parsed_key == 'Sleeve_length' and parsed_value.lower() in {'short-sleeve', 'short sleeve'}:
                short_sleeve_flag = True
        break

    key_values.setdefault('Men_regular_size_t', None)
    key_values.setdefault('Price_discount_range', None)
    result['shirts'] = shirts_flag if shirts_flag else None
    result['short_sleeve'] = short_sleeve_flag if short_sleeve_flag else None

    for key in parse_keys:
        if key not in key_values:
            continue
        if key == 'Price_discount_range':
            discounts = key_values[key]
            result[key] = (
                '50_PERCENT_ off & more'
                if discounts is not None
                and '50_PERCENT_ off & more' in discounts
                and '30_PERCENT_ off & more' not in discounts
                and '20_PERCENT_ off & more' not in discounts
                else 'not_50_PERCENT_ off & more'
            )
        else:
            result[key] = key_values[key]
    return result


@getter_registry.register()
async def get_default_search_engine(runtime: Any, config: dict[str, Any]) -> str:
    data = await _read_first_json(runtime, _chrome_profile_candidates('Preferences'))
    return data.get('default_search_provider_data', {}).get('template_url_data', {}).get('short_name', 'Google')


@getter_registry.register()
async def get_enable_do_not_track(runtime: Any, config: dict[str, Any]) -> str:
    data = await _read_first_json(runtime, _chrome_profile_candidates('Preferences'))
    return 'true' if data.get('enable_do_not_track') else 'false'


@getter_registry.register()
async def get_enable_enhanced_safety_browsing(runtime: Any, config: dict[str, Any]) -> str:
    data = await _read_first_json(runtime, _chrome_profile_candidates('Preferences'))
    return 'true' if data.get('safebrowsing', {}).get('enhanced') else 'false'


@getter_registry.register()
async def get_enable_safe_browsing(runtime: Any, config: dict[str, Any]) -> str:
    data = await _read_first_json(runtime, _chrome_profile_candidates('Preferences'))
    safe_browsing = data.get('safebrowsing', {})
    return 'true' if safe_browsing.get('enhanced') or safe_browsing.get('enabled') else 'false'


@getter_registry.register()
async def get_new_startup_page(runtime: Any, config: dict[str, Any]) -> str:
    data = await _read_first_json(runtime, _chrome_profile_candidates('Preferences'))
    if 'session' not in data:
        return 'true'
    return 'true' if data.get('session', {}).get('restore_on_startup') == 5 else 'false'


@getter_registry.register()
async def get_profile_name(runtime: Any, config: dict[str, Any]) -> str | None:
    data = await _read_first_json(runtime, _chrome_profile_candidates('Preferences'))
    return data.get('profile', {}).get('name')


@getter_registry.register()
async def get_chrome_font_size(runtime: Any, config: dict[str, Any]) -> dict[str, Any]:
    data = await _read_first_json(runtime, _chrome_profile_candidates('Preferences'))
    return data.get('webkit', {}).get('webprefs', {'default_fixed_font_size': 13, 'default_font_size': 16})


@getter_registry.register()
async def get_chrome_appearance_mode_ui(runtime: Any, config: dict[str, Any]) -> str:
    return await get_chrome_color_scheme(runtime, config)


@getter_registry.register()
async def get_chrome_color_scheme(runtime: Any, config: dict[str, Any]) -> str:
    data = await _read_first_json(runtime, _chrome_profile_candidates('Preferences'))
    theme = data.get('browser', {}).get('theme', {})
    raw = theme.get('color_scheme2', theme.get('color_scheme'))
    if raw in (1, '1', 'light'):
        return 'light'
    if raw in (2, '2', 'dark'):
        return 'dark'
    system_theme = data.get('extensions', {}).get('theme', {}).get('system_theme')
    if system_theme in (0, '0', False):
        return 'light'
    return 'system'


@getter_registry.register()
async def get_bookmarks(runtime: Any, config: dict[str, Any]) -> dict[str, Any]:
    return await _read_first_json(runtime, _chrome_profile_candidates('Bookmarks'))


@getter_registry.register()
async def get_cookie_data(runtime: Any, config: dict[str, Any]) -> list[Any] | None:
    cache_path = cache_dir_from_config(config) / str(config.get('dest', 'Cookies'))
    downloaded = await _download_first_file(runtime, _chrome_profile_candidates('Cookies'), cache_path)
    if not downloaded:
        return None
    return await asyncio.to_thread(_sqlite_rows, cache_path, 'SELECT * FROM cookies')


@getter_registry.register()
async def get_history(runtime: Any, config: dict[str, Any]) -> list[Any] | None:
    cache_path = cache_dir_from_config(config) / str(config.get('dest', 'History'))
    downloaded = await _download_first_file(runtime, _chrome_profile_candidates('History'), cache_path)
    if not downloaded:
        return None
    return await asyncio.to_thread(_sqlite_rows, cache_path, 'SELECT url, title, last_visit_time FROM urls')


def _sqlite_rows(path: Path, query: str) -> list[Any]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(query).fetchall()
    finally:
        conn.close()


@getter_registry.register()
async def get_find_unpacked_extension_path(runtime: Any, config: dict[str, Any]) -> list[str]:
    data = await _read_first_json(runtime, _chrome_profile_candidates('Preferences'))
    return [
        str(item.get('path')) for item in data.get('extensions', {}).get('settings', {}).values() if item.get('path')
    ]


@getter_registry.register()
async def get_find_installed_extension_name(runtime: Any, config: dict[str, Any]) -> list[str]:
    data = await _read_first_json(runtime, _chrome_profile_candidates('Preferences'))
    names = []
    for item in data.get('extensions', {}).get('settings', {}).values():
        manifest = item.get('manifest', {})
        if manifest.get('name'):
            names.append(str(manifest['name']))
    return names


@getter_registry.register()
async def get_data_delete_automacally(runtime: Any, config: dict[str, Any]) -> str:
    data = await _read_first_json(runtime, _chrome_profile_candidates('Preferences'))
    return 'true' if data.get('profile', {}).get('default_content_setting_values') is not None else 'false'


@getter_registry.register()
async def get_shortcuts_on_desktop(runtime: Any, config: dict[str, Any]) -> dict[str, str]:
    """Map each ``~/Desktop/*.desktop`` file name to its **text content**.

    The consuming metric (``is_shortcut_on_desktop``) searches ``'Name=' + expected`` inside the
    dict *values*, matching upstream OSWorld, which pairs each desktop entry with the result of
    reading the file. An earlier version of this getter stored the file *path* as the value, so
    the substring search could never match and the check returned 0 for every launcher that was
    in fact created correctly (observed on chrome/35253b65, whose ``actual`` was
    ``{'chrome-Play_Puzzle_Game_2048.desktop': '/home/user/Desktop/chrome-...desktop'}``).

    Contents are base64-framed per file so newlines, tabs and non-UTF-8 bytes in a ``.desktop``
    entry cannot corrupt the framing, and everything is fetched in one round trip to the VM.
    """
    result = await run_bash(
        runtime,
        'find /home/user/Desktop -maxdepth 1 -name "*.desktop" -print0 2>/dev/null | '
        "while IFS= read -r -d '' f; do "
        'printf "%s\\t%s\\n" "$(basename "$f")" "$(base64 -w0 "$f")"; '
        'done',
        timeout=30,
    )
    shortcuts: dict[str, str] = {}
    for line in response_stdout(result).splitlines():
        name, sep, blob = line.partition('\t')
        if not name or not sep:
            continue
        try:
            shortcuts[name] = base64.b64decode(blob).decode('utf-8', errors='replace')
        except (ValueError, binascii.Error):
            logger.warning('Could not decode desktop entry %s while reading shortcuts', name)
            shortcuts[name] = ''
    return shortcuts


@getter_registry.register()
async def get_pdf_from_url(runtime: Any, config: dict[str, Any]) -> str:
    url = str(config['path'])
    path = cache_dir_from_config(config) / str(config['dest'])
    if await _cdp_connection(runtime):
        async with connect_chromium(runtime) as browser:
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = await context.new_page()
            await page.goto(url, wait_until='load', timeout=60000)
            content = await page.pdf(format='A4')
        if content:
            await write_bytes(path, content)
            return str(path)
    await download_url(url, path)
    return str(path)


@getter_registry.register()
async def get_info_from_website(runtime: Any, config: dict[str, Any]) -> list[str]:
    # This expected getter is only used with explicit backups in official OSWorld.
    backups = config.get('backups')
    if backups:
        return backups
    page = await _page_info(runtime, str(config['url']))
    return [page.get('content', '')]


@getter_registry.register('gotoRecreationPage_and_get_html_content')
async def get_goto_recreation_page_and_get_html_content(runtime: Any, config: dict[str, Any]) -> dict[str, Any]:
    try:
        return await _load_recreation_expected_content(runtime, config)
    except Exception:
        try:
            await _launch_chrome_for_cdp(runtime)
            return await _load_recreation_expected_content(runtime, config)
        except Exception as second_error:
            raise EvaluatorInfrastructureError(
                'Recreation.gov expected-value getter failed after its existing retry'
            ) from second_error


async def _load_recreation_expected_content(runtime: Any, config: dict[str, Any]) -> dict[str, Any]:
    async with connect_chromium(runtime) as browser:
        # Close only the evaluator-owned context before disconnecting CDP.
        async with contextlib.AsyncExitStack() as cleanup:
            context = await browser.new_context()
            cleanup.push_async_callback(context.close)
            page = await context.new_page()
            page.set_default_timeout(60000)
            await page.set_extra_http_headers({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'DNT': '1',
                'Upgrade-Insecure-Requests': '1',
            })
            detail_page = await _goto_recreation_diamond_page(page)
            await _click_next_available_and_wait_for_header(
                detail_page,
                header_selector=f'.{config["class"]}',
                header_index=int(config.get('order', 2)),
            )
            return await _recreation_expected_from_page(detail_page, config)


async def _goto_recreation_diamond_page(page: Any) -> Any:
    await page.goto('https://www.recreation.gov/', wait_until='domcontentloaded', timeout=60000)
    await page.wait_for_selector('input#hero-search-input', state='visible', timeout=60000)
    await page.fill('input#hero-search-input', 'Diamond')
    await page.wait_for_selector('button.nav-search-button', state='visible', timeout=60000)
    await page.click('button.nav-search-button')
    await page.wait_for_url(re.compile(r'recreation\.gov/search'), timeout=20000)
    await page.wait_for_load_state('domcontentloaded', timeout=15000)
    link = (
        page
        .locator('a[href*="/camping/campgrounds/233869"]')
        .filter(has_text=re.compile(r'\bDiamond\b', re.IGNORECASE))
        .first
    )
    await link.wait_for(state='visible', timeout=25000)
    detail_page = await _click_and_resolve_navigation_page(page, link)
    await detail_page.wait_for_load_state('load', timeout=20000)
    return detail_page


_DIAMOND_CAMPGROUND_URL_PATTERN = re.compile(
    r'/camping/campgrounds/233869(?:[/?#]|$)',
    re.IGNORECASE,
)


async def _click_and_resolve_navigation_page(
    source_page,
    click_target,
    url_pattern=_DIAMOND_CAMPGROUND_URL_PATTERN,
    *,
    timeout_ms=30000,
    poll_interval_ms=250,
):
    """Return the source page or a newly opened page after one click.

    Recreation.gov can open a campground either in the current tab or in a
    new tab. Only the source page and pages created by this click are eligible,
    so a stale campground tab cannot be mistaken for the navigation result.
    """
    if timeout_ms <= 0:
        raise ValueError('timeout_ms must be positive')
    if poll_interval_ms <= 0:
        raise ValueError('poll_interval_ms must be positive')

    context = source_page.context
    page_ids_before_click = {id(page) for page in context.pages}
    await click_target.click(no_wait_after=True)

    remaining_ms = timeout_ms
    observed_urls = []
    while True:
        candidates = [source_page]
        candidates.extend(
            page for page in context.pages if id(page) not in page_ids_before_click and page is not source_page
        )

        observed_urls = []
        for candidate in candidates:
            if candidate.is_closed():
                continue
            try:
                candidate_url = candidate.url
            except Exception:
                continue
            observed_urls.append(candidate_url)
            if url_pattern.search(candidate_url):
                return candidate

        if remaining_ms <= 0:
            break

        wait_ms = min(poll_interval_ms, remaining_ms)
        await source_page.wait_for_timeout(wait_ms)
        remaining_ms -= wait_ms

    raise TimeoutError(f'Timed out waiting for Diamond campground navigation; observed URLs: {observed_urls!r}')


def _normalize_recreation_calendar_header(text):
    return re.sub(r'\s+', '', text or '').title()


async def _click_next_available_and_wait_for_header(
    page,
    *,
    header_selector='.camp-sortable-column-header',
    header_index=2,
    timeout_ms=30000,
    poll_interval_ms=250,
):
    """Click the exact Next Available button and return the changed header."""
    if timeout_ms <= 0:
        raise ValueError('timeout_ms must be positive')
    if poll_interval_ms <= 0:
        raise ValueError('poll_interval_ms must be positive')

    header = page.locator(header_selector).nth(header_index)
    before = _normalize_recreation_calendar_header(await header.inner_text(timeout=max(1, timeout_ms)))
    if not before:
        raise RuntimeError('Initial Recreation.gov calendar header is empty')

    button = page.get_by_role(
        'button',
        name='Next Available',
        exact=True,
    )
    await button.wait_for(state='visible', timeout=timeout_ms)
    await button.click(timeout=timeout_ms)

    remaining_ms = timeout_ms
    current = before
    while True:
        current = _normalize_recreation_calendar_header(await header.inner_text(timeout=max(1, remaining_ms)))
        if current and current != before:
            return current

        if remaining_ms <= 0:
            break

        wait_ms = min(poll_interval_ms, remaining_ms)
        await page.wait_for_timeout(wait_ms)
        remaining_ms -= wait_ms

    raise TimeoutError(f'Next Available did not update the Recreation.gov calendar header; header remained {current!r}')


async def _recreation_expected_from_page(page: Any, config: dict[str, Any]) -> dict[str, Any]:
    if config.get('selector') != 'class':
        raise EvaluatorInfrastructureError('Unsupported Recreation.gov selector')

    class_name = str(config.get('class', ''))
    if not class_name:
        raise EvaluatorInfrastructureError('Missing Recreation.gov class selector')

    try:
        elements = await page.query_selector_all(f'.{class_name}')
        order = int(config.get('order', 0))
    except Exception as exc:
        raise EvaluatorInfrastructureError('Cannot read Recreation.gov expected header') from exc

    if order < 0 or len(elements) <= order:
        raise EvaluatorInfrastructureError('Missing Recreation.gov expected header')

    text = (await elements[order].text_content() or '').strip()
    if not text:
        raise EvaluatorInfrastructureError('Empty Recreation.gov expected header')
    return {'expected': {class_name: text}}


@getter_registry.register()
async def get_googledrive_file(runtime: Any, config: dict[str, Any]) -> str | list[str | None] | None:
    try:
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
    except Exception as exc:
        raise RuntimeError('PyDrive is required for googledrive_file getter') from exc

    cache_dir = cache_dir_from_config(config)
    auth = GoogleAuth(
        settings_file=str(config.get('settings_file', 'evaluation_examples/settings/googledrive/settings.yml'))
    )
    drive = GoogleDrive(auth)

    def _single(query: list[str], dest: str) -> str | None:
        parent_id = 'root'
        try:
            for item in query:
                file_list = drive.ListFile({'q': f'( {item} ) and "{parent_id}" in parents'}).GetList()
                if not file_list:
                    return None
                file = file_list[0]
                parent_id = file['id']
            out = cache_dir / dest
            file.GetContentFile(str(out), mimetype=file['mimeType'])
            return str(out)
        except Exception:
            return None

    if 'query' in config:
        return await asyncio.to_thread(_single, config['query'], config['dest'])
    if 'path' in config:
        query = _google_path_query(config['path'])
        return await asyncio.to_thread(_single, query, config['dest'])

    results = []
    items = config.get('query_list')
    if items is not None:
        for query, dest in zip(items, config['dest'], strict=False):
            results.append(await asyncio.to_thread(_single, query, dest))
        return results
    for path, dest in zip(config.get('path_list', []), config['dest'], strict=False):
        results.append(await asyncio.to_thread(_single, _google_path_query(path), dest))
    return results


def _google_path_query(path: list[str]) -> list[str]:
    return [
        f"title = '{part}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        if idx < len(path) - 1
        else f"title = '{part}' and trashed = false"
        for idx, part in enumerate(path)
    ]


@getter_registry.register()
async def get_default_search_engine_name(runtime: Any, config: dict[str, Any]) -> str:
    return await get_default_search_engine(runtime, config)
