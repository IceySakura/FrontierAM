"""End-to-end demo for the AGR Mobile provider.

模拟一个真实的安卓操作任务：连接设备、回到桌面、启动设置 App、上下滑动浏览、
点击进入子页面、返回，全程截图存档。 不接 LLM，纯脚本驱动，演示 gui_* executor
（点击 / 滑动 / 按键 / 应用管理）在真实移动任务里怎么串起来用。

Usage:
    # template 模式（新建沙盒）：
    uv run python docs/examples/scripts/test_agr_mobile.py \
        --api-key <key> --template <android-template> \
        [--domain ap-guangzhou.tencentags.com] \
        [--app com.android.settings]

    # existing 模式（复用已有沙盒）：
    uv run python docs/examples/scripts/test_agr_mobile.py \
        --api-key <key> --sandbox-id <sandbox-id>
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='AGR Mobile end-to-end demo')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--template', default=None, help='Android sandbox template name')
    group.add_argument('--sandbox-id', default=None, help='Existing sandbox ID to reuse')
    p.add_argument('--domain', default='ap-guangzhou.tencentags.com')
    p.add_argument('--api-key', default=None, help='AGS/E2B API key (必填)')
    p.add_argument('--app', default='com.android.settings', help='Android package to launch')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Directory to save screenshots')
    return p.parse_args()


def _save_png(data: bytes, out_dir: Path, name: str) -> None:
    """Save PNG bytes to <out_dir>/<name> and print the absolute path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_bytes(data)
    print(f'    saved screenshot -> {path.resolve()}')


async def _shot(runtime: Any, out_dir: Path, name: str) -> None:
    """gui_screenshot 返回 base64（字段 'image'），解码后保存。"""
    shot = await runtime.call_executor('gui_screenshot', {})
    img = shot.get('image') or ''
    if img:
        _save_png(base64.b64decode(img), out_dir, name)


async def _run_task(runtime: Any, out_dir: Path, app: str) -> None:
    """真实移动任务：回桌面 -> 启动 App -> 滑动浏览 -> 点击 -> 返回。"""
    # 1. 设备信息（确认连上了什么设备）
    info = await runtime.call_executor('gui_get_device_info', {})
    print(f'  [1] device: {info.get("device_name")} android={info.get("platform_version")}')
    print(f'      current foreground: {info.get("current_package")}/{info.get("current_activity")}')

    # 2. 屏幕尺寸，后面滑动/点击坐标基于它
    size = await runtime.call_executor('gui_get_screen_size', {})
    w, h = size.get('width', 1080), size.get('height', 1920)
    print(f'  [2] screen size: {w}x{h}')

    # 3. 先回到桌面（Home），保证起点干净
    await runtime.call_executor('gui_press_key', {'key': 'Home'})
    await asyncio.sleep(1.0)
    print('  [3] pressed Home')
    await _shot(runtime, out_dir, 'agr_mobile_1_home.png')

    # 4. 启动目标 App
    await runtime.call_executor('gui_launch_app', {'app': app})
    await asyncio.sleep(2.5)
    info2 = await runtime.call_executor('gui_get_device_info', {})
    print(f'  [4] launched {app}, now foreground: {info2.get("current_package")}')
    await _shot(runtime, out_dir, 'agr_mobile_2_app_opened.png')

    # 5. 上滑浏览列表（从屏幕下方往上滑 = 内容向上滚）
    await runtime.call_executor('gui_drag', {
        'x1': w // 2, 'y1': int(h * 0.75),
        'x2': w // 2, 'y2': int(h * 0.25),
        'duration': 400,
    })
    await asyncio.sleep(1.0)
    print('  [5] swiped up to scroll the list')
    await _shot(runtime, out_dir, 'agr_mobile_3_scrolled.png')

    # 6. 点击列表中部的一项，进入子页面
    await runtime.call_executor('gui_click', {'x': w // 2, 'y': int(h * 0.4)})
    await asyncio.sleep(1.5)
    info3 = await runtime.call_executor('gui_get_device_info', {})
    print(f'  [6] tapped a list item, activity now: {info3.get("current_activity")}')
    await _shot(runtime, out_dir, 'agr_mobile_4_detail.png')

    # 7. 按返回键退回上一级
    await runtime.call_executor('gui_press_key', {'key': 'Back'})
    await asyncio.sleep(1.0)
    print('  [7] pressed Back')
    await _shot(runtime, out_dir, 'agr_mobile_5_back.png')

    # 8. 收尾：关闭 App + 回桌面，恢复干净状态
    await runtime.call_executor('gui_close_app', {'app': app})
    await runtime.call_executor('gui_press_key', {'key': 'Home'})
    print(f'  [done] closed {app} and returned home')


async def main() -> int:
    setup_logging()
    args = _parse_args()

    api_key = args.api_key
    if not api_key:
        print('ERROR: 需要 --api-key', file=sys.stderr)
        return 1

    from nexus.runtime.providers.agr_mobile import AGRMobileProvider

    provider = AGRMobileProvider(
        api_key=api_key,
        domain=args.domain,
        request_timeout=1800.0,
        ready_timeout=120,
    )

    mode = 'template' if args.template else 'existing'
    label = args.template or args.sandbox_id
    print(f'[agr-mobile] mode={mode} ({label}) app={args.app}')

    out_dir = Path(args.output_dir)
    runtime = None
    try:
        if args.template:
            runtime = await provider.acquire('android', sandbox_template=args.template)
        else:
            runtime = await provider.acquire('android', sandbox_id=args.sandbox_id)

        print(f'[agr-mobile] runtime acquired, sandbox_id={runtime.sandbox_id}')
        await _run_task(runtime, out_dir, args.app)
        print('[agr-mobile] ALL PASSED')
        return 0
    except Exception as e:
        print(f'[agr-mobile] FAILED: {e}', file=sys.stderr)
        return 1
    finally:
        await provider.release_all()


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
