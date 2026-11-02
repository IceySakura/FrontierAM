"""SmartRun iOS Runtime 使用示例。

通过一个真实任务演示如何使用 SmartRunRuntime 的 GUI executor：

    任务：在一台 iOS 真机上，启动「设置」App，搜索 Wi-Fi、进入「无线局域网」详情，
          长按已连接的网络呼出菜单，最后返回并关闭设置、回到主屏幕。

这一个流程自然地用到了 Runtime 暴露的这些 gui_*() 方法：

    gui_get_screen_size   获取屏幕分辨率
    gui_screenshot        截图
    gui_launch_app        启动 App
    gui_close_app         关闭 App
    gui_press_key         按键（Home / Back）
    gui_click             单击
    gui_type              输入文字
    gui_drag              滑动
    gui_long_click        长按

这是一个「怎么用」的 demo，不是覆盖测试：每一步只是调用对应方法、
打印它做了什么、并把截图存下来方便回看。

运行方式：
  cd nexus
  SMARTRUN_TOKEN=<your_token> uv run python docs/examples/scripts/test_smartrun_ios.py

环境变量：
  SMARTRUN_TOKEN   SmartRun API token（必填）
  SMARTRUN_DEVICE  指定 device_id（可选，不填则随机分配一台 iOS 设备）

输出目录：docs/examples/scripts/output/<timestamp>/  （依次编号的步骤截图）
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


# ---------------------------------------------------------------------------
# 小工具：编号、保存截图、按比例算坐标、OCR 定位元素
# ---------------------------------------------------------------------------


_step_no = 0


def step(desc: str) -> None:
    """打印一个带编号的步骤标题。"""
    global _step_no
    _step_no += 1
    print(f'\n[{_step_no:02d}] {desc}')


async def snap(runtime: Any, run_dir: Path, name: str) -> None:
    """用 gui_screenshot() 截图并保存到输出目录。"""
    shot = await runtime.gui_screenshot()
    if shot.success and shot.image:
        path = run_dir / f'{_step_no:02d}_{name}.jpg'
        path.write_bytes(base64.b64decode(shot.image))
        print(f'     截图已保存: {path.name}')


async def locate(runtime: Any, text: str, min_conf: float = 0.5) -> tuple[int, int] | None:
    """用 SmartRun 的 OCR 能力定位屏幕上的文字，返回其中心坐标。

    真实场景里界面元素的位置会变，写死坐标不可靠，
    所以这里用 screenshot_with_analysis 先「看」一眼屏幕再决定点哪。

    min_conf 可调：灰色占位符（如搜索栏的「搜索」）OCR 置信度天然偏低，
    定位这类元素时要把阈值调低。
    """
    result = await runtime._call(
        'screenshot_with_analysis',
        {
            'device_id': runtime.device_id,
            'session_id': runtime.session_id,
            'target_text': text,
            'json_output': True,
            'timeout': 60,
        },
    )
    for item in result.get('ocr_results', []):
        if text in item.get('text', '') and item.get('confidence', 0) > min_conf:
            cx, cy = item['center']
            return int(cx), int(cy)
    return None


# ---------------------------------------------------------------------------
# 真实任务流程
# ---------------------------------------------------------------------------


async def demo(runtime: Any, run_dir: Path) -> None:
    """用一个完整任务串起所有 gui_*() executor。"""

    # 1) gui_get_screen_size —— 拿到屏幕尺寸，后面算坐标要用
    step('gui_get_screen_size —— 读取屏幕分辨率')
    size = await runtime.gui_get_screen_size()
    w, h = size.width, size.height
    print(f'     屏幕分辨率: {w} x {h}')

    # 2) gui_press_key('Home') —— 回到主屏幕，保证起点干净
    step("gui_press_key('Home') —— 返回主屏幕")
    await runtime.gui_press_key('Home')
    await asyncio.sleep(2.0)

    # 3) gui_screenshot —— 看看当前主屏幕长什么样
    step('gui_screenshot —— 截取主屏幕')
    await snap(runtime, run_dir, 'home')

    # 4) gui_launch_app —— 直接启动「设置」App
    #    （SmartRun MCP 已支持启动系统 App；启动前先 close 一次，保证从首页干净进入）
    step("gui_launch_app('com.apple.Preferences') —— 启动设置 App")
    await runtime.gui_close_app('com.apple.Preferences')
    await asyncio.sleep(1.0)
    launched = await runtime.gui_launch_app('com.apple.Preferences')
    print(f'     启动结果: {launched.success}')
    await asyncio.sleep(3.0)
    await snap(runtime, run_dir, 'settings_opened')

    # 5) gui_drag —— 在设置列表里上滑浏览一下
    step('gui_drag —— 在设置列表中向上滑动')
    cx = int(0.5 * w)
    await runtime.gui_drag(cx, int(0.7 * h), cx, int(0.35 * h), duration=400)
    await asyncio.sleep(1.5)
    await snap(runtime, run_dir, 'scrolled')

    # 滑回顶部，方便点搜索栏
    await runtime.gui_drag(cx, int(0.35 * h), cx, int(0.7 * h), duration=400)
    await asyncio.sleep(1.5)

    # 6) gui_click + gui_type —— 点搜索栏并输入「Wi-Fi」
    step('gui_click + gui_type —— 在搜索栏输入「Wi-Fi」')
    # 搜索栏占位符「搜索」是灰字，OCR 置信度低，用低阈值定位它的 y；
    # 搜索框横跨整行，x 用屏幕中线点击最稳。
    search_xy = await locate(runtime, '搜索', min_conf=0.2)
    search_x = int(0.5 * w)
    search_y = search_xy[1] if search_xy else int(0.16 * h)
    print(f'     搜索栏坐标: ({search_x}, {search_y})')
    await runtime.gui_click(search_x, search_y)
    await asyncio.sleep(2.0)  # 等键盘弹出、输入框获得焦点
    typed = await runtime.gui_type('Wi-Fi')
    print(f'     输入结果: {typed.success}')
    await asyncio.sleep(1.5)
    await snap(runtime, run_dir, 'searched')

    # 7) gui_click —— 点搜索结果里的 WiFi 设置项进入详情
    #    不同系统语言/版本下，这一项可能叫「无线局域网」或「WLAN」，依次尝试。
    step('gui_click —— 点击 WiFi 设置项进入详情')
    wifi_xy = None
    for kw in ('无线局域网', 'WLAN'):
        wifi_xy = await locate(runtime, kw, min_conf=0.5)
        if wifi_xy:
            print(f'     命中「{kw}」: {wifi_xy}')
            break
    if wifi_xy is None:
        wifi_xy = (int(0.5 * w), int(0.25 * h))  # 兜底：搜索结果第一项大致位置
        print(f'     未命中关键词，用兜底坐标: {wifi_xy}')
    await runtime.gui_click(*wifi_xy)
    await asyncio.sleep(2.5)
    await snap(runtime, run_dir, 'wifi_page')

    # 8) gui_long_click —— 长按已连接的网络行，iOS 弹出上下文菜单（复制/忽略此网络）
    step('gui_long_click —— 长按已连接的网络行，呼出上下文菜单')
    net_xy = await locate(runtime, 'Tencent-WiFi', min_conf=0.4)
    if net_xy is None:
        net_xy = (int(0.4 * w), int(0.4 * h))  # 兜底：已连网络一般在 WiFi 列表顶部
    print(f'     网络行坐标: {net_xy}')
    await runtime.gui_long_click(net_xy[0], net_xy[1], duration=1200)
    await asyncio.sleep(1.5)
    await snap(runtime, run_dir, 'long_pressed')

    # 9) gui_press_key('Back') —— 返回上一页
    step("gui_press_key('Back') —— 返回上一页")
    await runtime.gui_press_key('Back')
    await asyncio.sleep(1.5)
    await snap(runtime, run_dir, 'back')

    # 10) gui_close_app —— 关闭设置 App
    step("gui_close_app('com.apple.Preferences') —— 关闭设置 App")
    closed = await runtime.gui_close_app('com.apple.Preferences')
    print(f'     关闭结果: {closed.success}')
    await asyncio.sleep(1.5)

    # 11) gui_press_key('Home') —— 任务结束，回主屏幕
    step("gui_press_key('Home') —— 返回主屏幕，结束任务")
    await runtime.gui_press_key('Home')
    await asyncio.sleep(1.5)
    await snap(runtime, run_dir, 'done')


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


async def main() -> None:
    setup_logging()
    from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider

    if not SMARTRUN_TOKEN:
        raise SystemExit('请通过环境变量 SMARTRUN_TOKEN 提供 SmartRun API token')

    # 1. 创建 provider（指定平台为 iOS）
    provider = SmartRunRuntimeProvider(token=SMARTRUN_TOKEN, platform='ios', duration=600)

    acquire_kwargs: dict[str, Any] = {}
    if SMARTRUN_DEVICE:
        acquire_kwargs['device_id'] = SMARTRUN_DEVICE

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(__file__).parent / 'output' / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('SmartRun iOS Runtime 使用示例')
    print(f'输出目录: {run_dir}')
    print('=' * 60)

    try:
        # 2. acquire 一台设备，拿到 runtime（已 connect）
        runtime = await provider.acquire('main', purpose='ios usage demo', **acquire_kwargs)
        print(f'设备: {provider.get_device_id("main")}  platform={runtime.platform}')

        # 3. 跑完整任务流程
        await demo(runtime, run_dir)

        print('\n' + '=' * 60)
        print(f'✓ 任务完成，共 {_step_no} 步，截图见 {run_dir}')
        print('=' * 60)
    finally:
        # 4. 释放设备
        await provider.release_all()
        print('设备已释放')


if __name__ == '__main__':
    asyncio.run(main())
