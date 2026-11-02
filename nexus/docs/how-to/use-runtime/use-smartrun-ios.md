# 使用 SmartRun iOS

## 概述

`SmartRunRuntimeProvider` 把 SmartRun 真机平台（[smartrun.woa.com](https://smartrun.woa.com)）接进 Nexus，让 agent 操控 iOS / Android 真机。 它底层走 SmartRun MCP（JSON-RPC over SSE）：平台维护一批真实设备，Nexus 通过 `assign_device` 租用一台，再把点击、滑动、输入、截图、应用启停这些操作翻译成一组 `gui_*` executor。 本文以 iOS 为例，介绍 `smartrun` provider 的配置字段、可用工具，并用一个端到端示例脚本演示完整用法。

## 前提条件

- 一个 SmartRun API token（在 SmartRun 平台申请），在代码或 YAML 里通过 `${SMARTRUN_TOKEN}` 引用。
- 目标平台（`ios` / `android` / `harmony`）。 不指定具体 `device_id` 时，provider 会按平台自动分配一台空闲设备。

## 快速上手

`smartrun` provider 在 `acquire()` 时租用一台设备，返回一个已连接的 `SmartRunRuntime`，agent 直接使用 `gui_*` executor；`release()` 时归还设备。

```yaml
runtime_provider_name: smartrun
runtime_provider_config:
  # SmartRun API token
  token: "${SMARTRUN_TOKEN}"
  # SmartRun 服务域名
  api_base_url: "https://smartrun.woa.com"
  # 默认平台过滤：ios / android / harmony
  platform: ios
  # 设备租约时长（秒），配合 auto_renew 自动续租
  duration: 1800
  # HTTP 请求超时（秒）
  request_timeout: 60.0

runtime_configs:
  main:
    # 指定具体设备（可选，不填则按 platform 自动分配一台空闲设备）
    device_id: "00008020-000824562151002E"
    purpose: "示例任务"
```

也可以在代码里直接创建 provider：

```python
from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider

provider = SmartRunRuntimeProvider(token='mcp_xxx', platform='ios', duration=600)
runtime = await provider.acquire('main', purpose='demo')   # 已 connect
await runtime.gui_screenshot()
await provider.release_all()
```

### Provider 配置字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `token` | — | SmartRun API token（也可用 `SMARTRUN_TOKEN` 环境变量） |
| `api_base_url` | `https://smartrun.woa.com` | SmartRun 服务域名 |
| `platform` | `None` | 默认平台过滤：`ios` / `android` / `harmony` |
| `brand` | `None` | 默认品牌过滤（如 `Apple`） |
| `model` | `None` | 默认型号过滤（如 `iPhone11,2`） |
| `duration` | `1800` | 设备租约时长（秒），配合 `auto_renew` 自动续租 |
| `request_timeout` | `60.0` | HTTP 请求超时（秒） |
| `startup_jitter_max` | `0.0` | acquire 前的随机抖动上限（秒），用于错峰批量启动 |

`acquire()` 还支持 `device_id`、`platform`、`brand`、`model`、`duration`、`purpose` 等参数，覆盖 provider 默认值。

## 可用工具

以下 executor 通过 `call_executor()` 调用，也可绑定到 agent 的工具集。 高层调用接口是 `SmartRunRuntime` 上同名的 `gui_*()` 方法。

| Executor | 参数 | 说明 |
|----------|------|------|
| `gui_get_screen_size` | — | 获取屏幕分辨率 |
| `gui_screenshot` | — | 截图，返回 base64 编码 JPG |
| `gui_click` | `x`, `y` | 点击坐标 |
| `gui_double_click` | `x`, `y` | 双击坐标 |
| `gui_long_click` | `x`, `y`, `duration` | 长按（毫秒） |
| `gui_drag` | `x1`, `y1`, `x2`, `y2`, `duration` | 滑动手势 |
| `gui_type` | `text` | 输入文本（需先点中输入框、键盘弹出后调用） |
| `gui_press_key` | `key` | 按键，支持 `Home`、`Back` |
| `gui_launch_app` | `app` | 启动 App（bundle id） |
| `gui_close_app` | `app` | 关闭 App |

除 `gui_*` executor 外，`runtime._call(tool, args)` 还可直接调用 SmartRun MCP 的底层工具，例如 `screenshot_with_analysis`（截图 + OCR，用于定位屏幕元素）、`stop_app` 等。

## 示例脚本与运行效果

仓库提供了一个端到端示例脚本 [`docs/examples/scripts/test_smartrun_ios.py`](https://git.woa.com/nexus-agent/nexus/blob/main/docs/examples/scripts/test_smartrun_ios.py)，用一个真实任务把上面的 `gui_*` executor 串起来：启动「设置」App、搜索 Wi-Fi、进入「无线局域网」详情、长按已连接的网络，最后返回并关闭设置、回到主屏幕。 它是一个「怎么用」的 demo，每一步只调用对应方法、打印它做了什么、并把截图存下来方便回看。

### 完整脚本

```python
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
```

### 运行

```bash
cd nexus
SMARTRUN_TOKEN=<your_token> uv run python docs/examples/scripts/test_smartrun_ios.py
```

指定设备（可选）：加上 `SMARTRUN_DEVICE=<device_id>`，不填则按平台自动分配一台空闲 iOS 设备。

### 运行输出

```text
============================================================
SmartRun iOS Runtime 使用示例
输出目录: docs/examples/scripts/output/20260626_152052
============================================================
设备: 00008020-000824562151002E  platform=ios

[01] gui_get_screen_size —— 读取屏幕分辨率
     屏幕分辨率: 1125 x 2436

[02] gui_press_key('Home') —— 返回主屏幕

[03] gui_screenshot —— 截取主屏幕
     截图已保存: 03_home.jpg

[04] gui_launch_app('com.apple.Preferences') —— 启动设置 App
     启动结果: True
     截图已保存: 04_settings_opened.jpg

[05] gui_drag —— 在设置列表中向上滑动
     截图已保存: 05_scrolled.jpg

[06] gui_click + gui_type —— 在搜索栏输入「Wi-Fi」
     搜索栏坐标: (562, 478)
     输入结果: True
     截图已保存: 06_searched.jpg

[07] gui_click —— 点击 WiFi 设置项进入详情
     命中「WLAN」: (246, 357)
     截图已保存: 07_wifi_page.jpg

[08] gui_long_click —— 长按已连接的网络行，呼出上下文菜单
     网络行坐标: (320, 1162)
     截图已保存: 08_long_pressed.jpg

[09] gui_press_key('Back') —— 返回上一页
     截图已保存: 09_back.jpg

[10] gui_close_app('com.apple.Preferences') —— 关闭设置 App
     关闭结果: True

[11] gui_press_key('Home') —— 返回主屏幕，结束任务
     截图已保存: 11_done.jpg

============================================================
✓ 任务完成，共 11 步，截图见 docs/examples/scripts/output/20260626_152052
============================================================
设备已释放
```

### 运行截图

脚本每步都会截图，下面挑几张关键节点：

<figure markdown="span">
  ![启动设置 App](/assets/images/use-smartrun-ios/1-settings.jpg){ width="280" }
  <figcaption>① gui_launch_app('com.apple.Preferences') 启动设置 App，首页稳定可见</figcaption>
</figure>

<figure markdown="span">
  ![搜索 Wi-Fi](/assets/images/use-smartrun-ios/2-searched.jpg){ width="280" }
  <figcaption>② gui_click 点中搜索栏、gui_type 输入「Wi-Fi」，键盘弹出、出现 WLAN 结果</figcaption>
</figure>

<figure markdown="span">
  ![进入无线局域网详情](/assets/images/use-smartrun-ios/3-wifi.jpg){ width="280" }
  <figcaption>③ gui_click 点搜索结果进入「无线局域网」详情，看到已连的 Tencent-WiFi</figcaption>
</figure>

<figure markdown="span">
  ![回到主屏幕](/assets/images/use-smartrun-ios/4-home.jpg){ width="280" }
  <figcaption>④ gui_close_app 关闭设置、gui_press_key('Home') 回到主屏幕，任务结束</figcaption>
</figure>
