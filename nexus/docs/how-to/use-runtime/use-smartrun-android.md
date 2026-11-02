# 使用 SmartRun Android

## 概述

`SmartRunRuntimeProvider` 把 SmartRun 真机平台（[smartrun.woa.com](https://smartrun.woa.com)）接入 Nexus，让 agent 操控 iOS / Android 真机。它底层使用 SmartRun MCP（JSON-RPC over SSE）：平台维护一批真实设备，Nexus 通过 `assign_device` 租用一台，再把点击、滑动、输入、截图、应用启停等操作转换成一组 `gui_*` executor。

本文以 Android 为例，介绍 `smartrun` provider 的配置字段、可用工具，并通过一个端到端脚本演示完整用法。

## 前提条件

- 一个 SmartRun API token（在 SmartRun 平台申请），推荐通过 `SMARTRUN_TOKEN` 环境变量提供。
- SmartRun 平台上有空闲的 Android 设备。不指定 `device_id` 时，provider 会自动分配一台符合过滤条件的设备。

## 快速上手

`smartrun` provider 在 `acquire()` 时租用设备并返回已经连接的 `SmartRunRuntime`；使用完成后，通过 `release()` 或 `release_all()` 归还设备。

```yaml
runtime_provider_name: smartrun
runtime_provider_config:
  # SmartRun API token
  token: "${SMARTRUN_TOKEN}"
  # SmartRun 服务域名
  api_base_url: "https://smartrun.woa.com"
  # 默认申请 Android 设备
  platform: android
  # 设备租约时长（秒），配合 auto_renew 自动续租
  duration: 1800
  # HTTP 请求超时（秒）；设备申请和 OCR 分析建议适当放宽
  request_timeout: 120.0

runtime_configs:
  main:
    # 指定具体设备（可选，不填则自动分配空闲 Android 设备）
    device_id: "<android-device-id>"
    purpose: "示例任务"
```

也可以在代码里直接创建 provider：

```python
from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider

provider = SmartRunRuntimeProvider(
    token='mcp_xxx',
    platform='android',
    duration=600,
    request_timeout=120,
)
runtime = await provider.acquire('main', purpose='android demo')
await runtime.gui_screenshot()
await provider.release_all()
```

### Provider 配置字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `token` | — | SmartRun API token（也可用 `SMARTRUN_TOKEN` 环境变量） |
| `api_base_url` | `https://smartrun.woa.com` | SmartRun 服务域名 |
| `platform` | `None` | 默认平台过滤；Android 场景设置为 `android` |
| `brand` | `None` | 默认品牌过滤 |
| `model` | `None` | 默认型号过滤 |
| `duration` | `1800` | 设备租约时长（秒），配合 `auto_renew` 自动续租 |
| `request_timeout` | `60.0` | HTTP 请求超时（秒） |
| `startup_jitter_max` | `0.0` | `acquire()` 前的随机抖动上限（秒），用于批量并发申请时错峰启动 |

`acquire()` 还支持 `device_id`、`platform`、`brand`、`model`、`duration`、`purpose` 等参数，用来覆盖 provider 的默认配置。

## 可用工具

以下 executor 可以通过 `call_executor()` 调用，也可以绑定到 agent 的工具集。高层接口是 `SmartRunRuntime` 上同名的 `gui_*()` 方法。

| Executor | 参数 | 说明 |
|----------|------|------|
| `gui_get_screen_size` | — | 获取屏幕分辨率 |
| `gui_screenshot` | — | 截图，返回 base64 编码 JPG |
| `gui_click` | `x`, `y` | 点击坐标 |
| `gui_double_click` | `x`, `y` | 双击坐标 |
| `gui_long_click` | `x`, `y`, `duration` | 长按（毫秒） |
| `gui_drag` | `x1`, `y1`, `x2`, `y2`, `duration` | 滑动手势 |
| `gui_type` | `text` | 向当前获得焦点的输入框输入文本 |
| `gui_press_key` | `key` | 按键，支持 `Home`、`Back` |
| `gui_launch_app` | `app` | 按 Android package name 启动应用 |
| `gui_close_app` | `app` | 按 Android package name 关闭应用 |

除 `gui_*` executor 外，`runtime._call(tool, args)` 还可调用 SmartRun MCP 的底层工具。例如 `screenshot_with_analysis` 会同时截图并进行 OCR，可用于在不同品牌、分辨率和系统语言的设备上定位元素，减少对固定坐标的依赖。

## Android 使用要点

- Android 应用使用 package name，例如系统设置通常是 `com.android.settings`；这不同于 iOS 的 bundle id。
- `gui_press_key('Back')` 对应 Android 原生返回键，适合页面返回和退出搜索状态；`Home` 返回桌面。
- 不同厂商 ROM 的设置项名称可能不同。示例会依次尝试“网络和互联网”“网络与互联网”“WLAN”“Wi-Fi”“无线网络”等关键词。
- 真机申请和 OCR 分析可能需要等待，建议将 `request_timeout` 设置为 `120` 秒，并为页面切换预留短暂等待时间。
- 必须使用 `try/finally` 释放设备，避免脚本异常后设备一直占用。

## 示例脚本与运行效果

仓库提供了端到端示例脚本 [`docs/examples/scripts/test_smartrun_android.py`](../../examples/scripts/test_smartrun_android.py)。脚本会申请一台真实 Android 设备，回到桌面，启动系统设置，通过 OCR 查找并打开网络设置，搜索 Bluetooth，最后关闭设置、回到桌面并释放设备。

这个流程覆盖设备申请、`gui_get_screen_size`、`gui_screenshot`、`gui_launch_app`、`gui_close_app`、`gui_press_key`、`gui_click`、`gui_type`、OCR 定位和设备释放。

### 完整脚本

```python
"""SmartRun Android Runtime 使用示例。

运行方式：
  cd nexus
  source ../nexus-workspace/secret.env
  uv run python docs/examples/scripts/test_smartrun_android.py

环境变量：
  SMARTRUN_TOKEN   SmartRun API token（必填）
  SMARTRUN_DEVICE  指定 Android device_id（可选）
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


async def locate(
    runtime: Any,
    keywords: tuple[str, ...],
    min_conf: float = 0.4,
) -> tuple[int, int, str] | None:
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
            if (
                keyword.lower() in item.get('text', '').lower()
                and item.get('confidence', 0) >= min_conf
            ):
                cx, cy = item['center']
                return int(cx), int(cy), item.get('text', keyword)
    return None


async def demo(runtime: Any, run_dir: Path) -> None:
    step('gui_get_screen_size —— 读取 Android 屏幕分辨率')
    size = await runtime.gui_get_screen_size()
    print(f'     屏幕分辨率: {size.width} x {size.height}')

    step("gui_press_key('Home') —— 返回 Android 桌面")
    await runtime.gui_press_key('Home')
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
    target = await locate(
        runtime,
        ('网络和互联网', '网络与互联网', 'WLAN', 'Wi-Fi', '无线网络'),
    )
    if target is not None:
        x, y, text = target
        print(f'     命中「{text}」: ({x}, {y})')
        await runtime.gui_click(x, y)
        await asyncio.sleep(2)
        await snap(runtime, run_dir, 'network')
        await runtime.gui_press_key('Back')
        await asyncio.sleep(2)

    step('screenshot_with_analysis + gui_click + gui_type —— 搜索 Bluetooth')
    search = await locate(
        runtime,
        ('搜索设置', '搜索', 'Search settings', 'Search'),
        min_conf=0.2,
    )
    if search is not None:
        x, y, text = search
        print(f'     命中「{text}」: ({x}, {y})')
        await runtime.gui_click(x, y)
        await asyncio.sleep(1)
        await runtime.gui_type('Bluetooth')
        await asyncio.sleep(2)
        await snap(runtime, run_dir, 'searched')
        await runtime.gui_press_key('Back')

    step("gui_close_app + gui_press_key('Home') —— 清理并结束")
    await runtime.gui_close_app('com.android.settings')
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

    try:
        runtime = await provider.acquire(
            'main',
            purpose='nexus android runtime demo',
            **acquire_kwargs,
        )
        print(
            f'设备: {provider.get_device_id("main")} '
            f'session={provider.get_runtime_id("main")} platform={runtime.platform}'
        )
        await demo(runtime, run_dir)
    finally:
        await provider.release_all()
        print('设备已释放')


if __name__ == '__main__':
    asyncio.run(main())
```

仓库中的脚本是可运行的完整版本；后续以脚本源码为准。

### 运行

如果密钥保存在与 `nexus` 同级的 `nexus-workspace/secret.env`：

```bash
cd nexus
source ../nexus-workspace/secret.env
uv run python docs/examples/scripts/test_smartrun_android.py
```

也可以仅为本次命令提供 token：

```bash
SMARTRUN_TOKEN=<your_token> uv run python docs/examples/scripts/test_smartrun_android.py
```

指定设备时增加 `SMARTRUN_DEVICE=<device_id>`；不设置则自动申请一台空闲 Android 设备。

### 运行输出

```text
============================================================
SmartRun Android Runtime 使用示例
输出目录: docs/examples/scripts/output/android/20260716_164103
============================================================
设备: <device-id> session=<session-id> platform=android

[01] gui_get_screen_size —— 读取 Android 屏幕分辨率
     屏幕分辨率: 1080 x 2400

[02] gui_press_key('Home') —— 返回 Android 桌面
     截图已保存: 02_home.jpg

[03] gui_launch_app('com.android.settings') —— 启动系统设置
     截图已保存: 03_settings.jpg

[04] screenshot_with_analysis + gui_click —— 定位并打开网络设置
     截图已保存: 04_network.jpg

[05] screenshot_with_analysis + gui_click + gui_type —— 搜索 Bluetooth
     截图已保存: 05_searched.jpg

[06] gui_close_app + gui_press_key('Home') —— 清理并结束
     截图已保存: 06_done.jpg

测试完成，共 6 步
设备已释放
```

具体分辨率、OCR 命中文案和坐标会随设备型号、ROM 和系统语言变化。

### 运行截图

<figure markdown="span">
  ![启动 Android 系统设置](/assets/images/use-smartrun-android/1-settings.jpg){ width="280" }
  <figcaption>① gui_launch_app('com.android.settings') 启动系统设置</figcaption>
</figure>

<figure markdown="span">
  ![打开网络设置](/assets/images/use-smartrun-android/2-network.jpg){ width="280" }
  <figcaption>② screenshot_with_analysis 定位入口，gui_click 打开网络设置</figcaption>
</figure>

<figure markdown="span">
  ![搜索 Bluetooth](/assets/images/use-smartrun-android/3-searched.jpg){ width="280" }
  <figcaption>③ gui_click 点中搜索入口，gui_type 输入 Bluetooth</figcaption>
</figure>

<figure markdown="span">
  ![回到 Android 桌面](/assets/images/use-smartrun-android/4-home.jpg){ width="280" }
  <figcaption>④ gui_close_app 关闭设置，gui_press_key('Home') 回到桌面</figcaption>
</figure>

## 并发申请

每次 `acquire()` 对应一台独立设备。批量任务可以并发申请多个 runtime，并通过不同 runtime 独立操作设备。建议设置 `startup_jitter_max` 或在任务启动时增加随机延迟，避免大量请求在同一时刻到达平台。

如果没有足够的空闲设备，`acquire()` 会失败并抛出异常；调用方应捕获异常，记录失败任务，并确保已经成功申请的设备最终得到释放。

```python
provider = SmartRunRuntimeProvider(
    token=SMARTRUN_TOKEN,
    platform='android',
    duration=600,
    request_timeout=120,
    startup_jitter_max=3.0,
)

try:
    runtime = await provider.acquire('worker-0', purpose='android concurrent test')
    await runtime.gui_screenshot()
except Exception as exc:
    print(f'设备申请或执行失败: {exc}')
finally:
    await provider.release_all()
```
