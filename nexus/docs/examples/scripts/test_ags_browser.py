"""End-to-end demo for the AGS Browser provider.

模拟一个真实的网页浏览任务：打开百度、用关键词搜索、读取页面可观测信息、
在搜索框里换词重新搜索、滚动翻页、开新标签页，全程截图存档。 不接 LLM，
纯脚本驱动，演示 browser_* executor（nexus interface）和原始 Playwright
（raw interface）在真实任务里怎么串起来用。

用百度（境内可达、稳定）而非境外站点，避免沙盒访问外网超时。

Usage:
    # template 模式（E2B 模板）：
    uv run python docs/examples/scripts/test_ags_browser.py \
        --api-key <key> --template <browser-template> \
        [--domain ap-guangzhou.tencentags.com] \
        [--interface nexus|raw] \
        [--query 沙盒]

    # control_plane 模式（腾讯云控制面，复用已有内置 browser 工具）：
    uv run python docs/examples/scripts/test_ags_browser.py \
        --provision control_plane \
        --secret-id <SecretId> --secret-key <SecretKey> --tool-id <ToolId> \
        [--region ap-guangzhou] [--interface nexus|raw]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys

from pathlib import Path
from typing import Any

from nexus.utils.log import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'screenshots'

# 沙盒访问外网可能较慢，goto 用较宽松的超时（毫秒）
GOTO_TIMEOUT_MS = 30000


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='AGS Browser end-to-end demo')
    p.add_argument(
        '--provision',
        choices=['template', 'control_plane'],
        default='template',
        help='template=E2B 模板（需 --template + --api-key）; '
        'control_plane=腾讯云控制面（需 --secret-id + --secret-key + --tool-id，复用已有内置 browser 工具）',
    )
    p.add_argument('--template', default=None, help='AGS browser template name (template 模式必填)')
    p.add_argument('--domain', default='ap-guangzhou.tencentags.com')
    p.add_argument('--region', default='ap-guangzhou', help='Tencent Cloud region (control_plane 模式)')
    p.add_argument('--interface', choices=['nexus', 'raw'], default='nexus')
    p.add_argument('--api-key', default=None, help='AGS API key (template 模式必填)')
    p.add_argument('--secret-id', default=None, help='Tencent Cloud SecretId (control_plane 模式必填)')
    p.add_argument('--secret-key', default=None, help='Tencent Cloud SecretKey (control_plane 模式必填)')
    p.add_argument('--tool-id', default=None, help='已有内置 browser 工具的 ToolId (control_plane 模式必填)')
    p.add_argument('--query', default='腾讯云', help='Baidu search term')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Directory to save screenshots')
    return p.parse_args()


def _save_png(data: bytes, out_dir: Path, name: str) -> None:
    """Save PNG bytes to <out_dir>/<name> and print the absolute path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_bytes(data)
    print(f'    saved screenshot -> {path.resolve()}')


def _save_b64_png(b64: str, out_dir: Path, name: str) -> None:
    if b64:
        _save_png(base64.b64decode(b64), out_dir, name)


async def _shot(runtime: Any, session_id: str, out_dir: Path, name: str) -> None:
    """Take a full-page screenshot via browser_screenshot and save it."""
    resp = await runtime.call_executor(
        'browser_screenshot',
        {
            'session_id': session_id,
            'full_page': True,
        },
    )
    # browser_screenshot 返回字段是 'screenshot'（base64），不是 'image'。
    _save_b64_png(resp.get('screenshot') or '', out_dir, name)


async def _test_nexus(runtime: Any, out_dir: Path, query: str) -> None:
    """真实任务：打开百度首页 -> 搜索框输入回车 -> 读结果 -> 滚动 -> 多标签。"""
    # 1. 开 session，1440x900 视口贴近真实桌面浏览器
    session = await runtime.call_executor(
        'create_browser_session',
        {
            'viewport_width': 1440,
            'viewport_height': 900,
        },
    )
    session_id = session['session_id']
    print(f'  [1] session created: {session_id}')

    try:
        # 2. 打开百度首页（首页搜索框稳定可见，比直达结果页更适合演示表单交互）
        goto = await runtime.call_executor(
            'browser_goto',
            {
                'session_id': session_id,
                'url': 'https://www.baidu.com',
                'timeout': GOTO_TIMEOUT_MS,
            },
        )
        if not goto.get('success'):
            raise RuntimeError(f'browser_goto failed: {goto}')
        landed = goto.get('url')
        print(f'  [2] opened baidu home, landed url: {landed}')
        if not landed or landed == 'about:blank':
            raise RuntimeError(f'navigation did not land on a real page (got {landed!r}); 沙盒可能访问外网受限')
        await _shot(runtime, session_id, out_dir, 'ags_browser_1_home.png')

        # 3. 在首页大搜索框输入关键词并回车，演示真实表单交互
        #    百度搜索框是 input#kw
        await runtime.call_executor(
            'browser_fill',
            {
                'session_id': session_id,
                'locator': {'type': 'selector', 'selector': '#kw'},
                'value': query,
            },
        )
        await runtime.call_executor(
            'browser_press_key',
            {
                'session_id': session_id,
                'key': 'Enter',
            },
        )
        await asyncio.sleep(1.5)
        print(f'  [3] searched for {query!r} via the home search box')

        # 4. 读取结果页可观测信息：标题 + accessibility tree（axtree 是字符串）
        obs = await runtime.call_executor(
            'browser_get_observation',
            {
                'session_id': session_id,
                'viewport_only': True,
            },
        )
        axtree = obs.get('axtree') or ''
        print(f'  [4] result page title: {obs.get("title")!r} url: {obs.get("url")}')
        print(f'      axtree length: {len(axtree)} chars, focused_bid={obs.get("focused_element_bid")!r}')
        # 打印 axtree 前几行，演示如何从中挑选可点击元素
        preview = '\n'.join(f'        {line}' for line in axtree.splitlines()[:8])
        if preview:
            print('      axtree preview:')
            print(preview)
        await _shot(runtime, session_id, out_dir, 'ags_browser_2_results.png')

        # 5. 向下滚动，模拟阅读结果列表
        await runtime.call_executor(
            'browser_scroll',
            {
                'session_id': session_id,
                'direction': 'down',
                'amount': 1200,
            },
        )
        print('  [5] scrolled down 1200px')
        await _shot(runtime, session_id, out_dir, 'ags_browser_3_scrolled.png')

        # 6. 开新标签页访问另一个站点，演示多标签管理
        newtab = await runtime.call_executor(
            'browser_new_tab',
            {
                'session_id': session_id,
                'url': 'https://www.qq.com',
            },
        )
        print(f'  [6] opened new tab index={newtab.get("tab_index")}')
        tabs = await runtime.call_executor('browser_list_tabs', {'session_id': session_id})
        for tab in tabs.get('tabs', []):
            marker = '*' if tab.get('is_active') else ' '
            print(f'      {marker} tab[{tab.get("index")}] {tab.get("title")!r} {tab.get("url")}')
    finally:
        await runtime.call_executor('destroy_browser_session', {'session_id': session_id})
        print('  [done] session destroyed')


async def _test_raw(runtime: Any, out_dir: Path, query: str) -> None:
    """raw interface：直接用 Playwright 对象模型完成同一类任务。"""
    version = await runtime.json_version()
    print(f'  [1] browser version: {version.get("Browser") if isinstance(version, dict) else version}')

    browser, playwright = await runtime.connect_playwright()
    try:
        page = await browser.new_page()
        await page.set_viewport_size({'width': 1440, 'height': 900})
        page.set_default_timeout(GOTO_TIMEOUT_MS)

        # 1. 打开百度首页（首页搜索框稳定可见）
        await page.goto('https://www.baidu.com', timeout=GOTO_TIMEOUT_MS)
        print(f'  [2] home title: {await page.title()!r}')
        _save_png(await page.screenshot(full_page=True), out_dir, 'ags_browser_raw_1_home.png')

        # 2. 真实表单交互：在首页大搜索框输入关键词并回车
        search_box = page.locator('#kw').first
        await search_box.fill(query)
        await search_box.press('Enter')
        await page.wait_for_load_state('domcontentloaded')
        await asyncio.sleep(1.0)
        print(f'  [3] searched for {query!r}, result url: {page.url}')
        _save_png(await page.screenshot(full_page=True), out_dir, 'ags_browser_raw_2_results.png')

        # 3. 用 Playwright 选择器读取第一条搜索结果标题
        first_result = page.locator('div.result h3, div.c-container h3').first
        try:
            text = (await first_result.inner_text(timeout=5000)).strip()
            print(f'  [4] first result title: {text[:80]!r}')
        except Exception:
            print('  [4] first result title not found (页面结构可能有变)')

        # 4. 统计页面上的链接数量，演示任意 DOM 查询
        link_count = await page.locator('a').count()
        print(f'  [5] anchor tags on page: {link_count}')
    finally:
        await playwright.stop()


async def main() -> int:
    setup_logging()
    args = _parse_args()

    from nexus.runtime.providers.ags import AGSRuntimeProvider

    if args.provision == 'control_plane':
        if not (args.secret_id and args.secret_key):
            print('ERROR: control_plane 模式需要 --secret-id 和 --secret-key', file=sys.stderr)
            return 1
        if not args.tool_id:
            print('ERROR: control_plane 模式需要 --tool-id（已有内置 browser 工具的 ToolId）', file=sys.stderr)
            return 1
        provider = AGSRuntimeProvider(
            secret_id=args.secret_id,
            secret_key=args.secret_key,
            region=args.region,
            domain=args.domain,
            interface_mode=args.interface,
            request_timeout=300.0,
            instance_ready_timeout=300.0,
        )
        # control_plane：用 tool_id 直接复用已有内置 browser 工具，provider 不按名查找、不新建、不删除。
        acquire_params: dict[str, Any] = {
            'runtime_kind': 'browser',
            'tool_type': 'browser',
            'tool_id': args.tool_id,
        }
        print(
            f'[ags-browser] provision=control_plane interface={args.interface} region={args.region} tool_id={args.tool_id}'
        )
    else:
        api_key = args.api_key
        if not api_key:
            print('ERROR: template 模式需要 --api-key', file=sys.stderr)
            return 1
        if not args.template:
            print('ERROR: template 模式需要 --template', file=sys.stderr)
            return 1
        provider = AGSRuntimeProvider(
            api_key=api_key,
            domain=args.domain,
            interface_mode=args.interface,
            request_timeout=300.0,
            instance_ready_timeout=300.0,
        )
        acquire_params = {
            'runtime_kind': 'browser',
            'template': args.template,
        }
        print(
            f'[ags-browser] provision=template interface={args.interface} template={args.template} query={args.query!r}'
        )

    out_dir = Path(args.output_dir)
    runtime = None
    try:
        runtime = await provider.acquire('main', **acquire_params)
        print('[ags-browser] runtime acquired')

        if args.interface == 'nexus':
            await _test_nexus(runtime, out_dir, args.query)
        else:
            await _test_raw(runtime, out_dir, args.query)

        print('[ags-browser] ALL PASSED')
        return 0
    except Exception as e:
        print(f'[ags-browser] FAILED: {e}', file=sys.stderr)
        return 1
    finally:
        await provider.release_all()


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
