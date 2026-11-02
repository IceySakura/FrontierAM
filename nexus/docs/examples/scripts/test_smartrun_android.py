"""SmartRun Android Runtime 使用示例。

通过真实 Android 设备验证 Nexus SmartRun Runtime 的核心 GUI executor：
设备申请、截图、启动设置、OCR 定位、点击、输入、返回、关闭应用和设备释放。

运行方式：
  cd nexus
  source ../nexus-workspace/secret.env
  uv run python docs/examples/scripts/test_smartrun_android.py

环境变量：
  SMARTRUN_TOKEN   SmartRun API token（必填）
  SMARTRUN_DEVICE  指定 Android device_id（可选）

输出目录：docs/examples/scripts/output/android/<timestamp>/
"""

from __future__ import annotations

import asyncio
import base64
import os

from datetime import datetime
from pathlib import Path
from typing import Any

from nexus.utils.log import setup_logging


SMARTRUN_TOKEN = os.environ.get('SMARTRUN_TOKEN', '')
SMARTRUN_DEVICE = os.environ.get('SMARTRUN_DEVICE', '')

_step_no = 0


def step(desc: str) -> None:
    global _step_no
    _step_no += 1
    print(f'\n[{_step_no:02d}] {desc}')


async def snap(runtime: Any, run_dir: Path, name: str) -> None:
    shot = await runtime.gui_screenshot()
    if not shot.success or not shot.image:
        raise RuntimeError(f'截图失败: {shot}')
    path = run_dir / f'{_step_no:02d}_{name}.jpg'
    path.write_bytes(base64.b64decode(shot.image))
    print(f'     截图已保存: {path.name}')


async def locate(runtime: Any, keywords: tuple[str, ...], min_conf: float = 0.4) -> tuple[int, int, str] | None:
    """通过 SmartRun 服务端截图分析定位文本，避免依赖固定坐标。"""
    for keyword in keywords:
        result = await runtime._call(
            'screenshot_with_analysis',
            {
                'device_id': runtime.device_id,
                'session_id': runtime.session_id,
                'target_text': keyword,
                'json_output': True,
                'timeout': 120,
            },
        )
        for item in result.get('ocr_results', []):
            if keyword.lower() in item.get('text', '').lower() and item.get('confidence', 0) >= min_conf:
                cx, cy = item['center']
                return int(cx), int(cy), item.get('text', keyword)
    return None


async def demo(runtime: Any, run_dir: Path) -> None:
    step('gui_get_screen_size —— 读取 Android 屏幕分辨率')
    size = await runtime.gui_get_screen_size()
    print(f'     屏幕分辨率: {size.width} x {size.height}')

    step("gui_press_key('Home') —— 返回 Android 桌面")
    result = await runtime.gui_press_key('Home')
    if not result.success:
        raise RuntimeError(f'Home 操作失败: {result}')
    await asyncio.sleep(2)
    await snap(runtime, run_dir, 'home')

    step("gui_launch_app('com.android.settings') —— 启动系统设置")
    await runtime.gui_close_app('com.android.settings')
    launched = await runtime.gui_launch_app('com.android.settings')
    if not launched.success:
        raise RuntimeError(f'启动设置失败: {launched}')
    await asyncio.sleep(3)
    await snap(runtime, run_dir, 'settings')

    step('screenshot_with_analysis + gui_click —— 定位并打开网络设置')
    target = await locate(runtime, ('网络和互联网', '网络与互联网', 'WLAN', 'Wi-Fi', '无线网络'))
    if target is None:
        print('     当前页面未定位到网络入口，跳过点击并继续验证搜索')
    else:
        x, y, text = target
        print(f'     命中「{text}」: ({x}, {y})')
        clicked = await runtime.gui_click(x, y)
        if not clicked.success:
            raise RuntimeError(f'点击网络入口失败: {clicked}')
        await asyncio.sleep(2)
        await snap(runtime, run_dir, 'network')
        backed = await runtime.gui_press_key('Back')
        if not backed.success:
            raise RuntimeError(f'返回设置首页失败: {backed}')
        await asyncio.sleep(2)

    step('screenshot_with_analysis + gui_click + gui_type —— 搜索 Bluetooth')
    search = await locate(runtime, ('搜索设置', '搜索', 'Search settings', 'Search'), min_conf=0.2)
    if search is None:
        print('     未定位到可点击的搜索入口，跳过输入验证')
    else:
        x, y, text = search
        print(f'     命中「{text}」: ({x}, {y})')
        clicked = await runtime.gui_click(x, y)
        if not clicked.success:
            raise RuntimeError(f'点击搜索入口失败: {clicked}')
        await asyncio.sleep(1)
        typed = await runtime.gui_type('Bluetooth')
        if not typed.success:
            raise RuntimeError(f'输入失败: {typed}')
        await asyncio.sleep(2)
        await snap(runtime, run_dir, 'searched')
        await runtime.gui_press_key('Back')
        await asyncio.sleep(1)

    step("gui_close_app + gui_press_key('Home') —— 清理并结束")
    closed = await runtime.gui_close_app('com.android.settings')
    if not closed.success:
        raise RuntimeError(f'关闭设置失败: {closed}')
    await runtime.gui_press_key('Home')
    await asyncio.sleep(2)
    await snap(runtime, run_dir, 'done')


async def main() -> None:
    setup_logging()
    from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider

    if not SMARTRUN_TOKEN:
        raise SystemExit('请先 source nexus-workspace/secret.env，提供 SMARTRUN_TOKEN')

    provider = SmartRunRuntimeProvider(
        token=SMARTRUN_TOKEN,
        platform='android',
        duration=600,
        request_timeout=120,
    )
    acquire_kwargs: dict[str, Any] = {}
    if SMARTRUN_DEVICE:
        acquire_kwargs['device_id'] = SMARTRUN_DEVICE

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(__file__).parent / 'output' / 'android' / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('SmartRun Android Runtime 使用示例')
    print(f'输出目录: {run_dir}')
    print('=' * 60)

    try:
        runtime = await provider.acquire('main', purpose='nexus android runtime demo', **acquire_kwargs)
        print(
            f'设备: {provider.get_device_id("main")} '
            f'session={provider.get_runtime_id("main")} platform={runtime.platform}'
        )
        await demo(runtime, run_dir)
        print(f'\n测试完成，共 {_step_no} 步，截图见 {run_dir}')
    finally:
        await provider.release_all()
        print('设备已释放')


if __name__ == '__main__':
    asyncio.run(main())
