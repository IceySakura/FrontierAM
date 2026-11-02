"""并发验证 Nexus SmartRun Android Runtime。

默认同时申请 7 台 Android 设备，并在每台设备上并行执行：
Home、截图、启动系统设置、截图、关闭设置、返回 Home，最后释放全部设备。

运行方式：
  cd nexus
  source ../nexus-workspace/secret.env
  uv run python docs/examples/scripts/test_smartrun_android_concurrent.py

环境变量：
  SMARTRUN_TOKEN        SmartRun API token（必填）
  SMARTRUN_CONCURRENCY  并发设备数（默认 7）

输出目录：docs/examples/scripts/output/android-concurrent/<timestamp>/
"""

from __future__ import annotations

import asyncio
import base64
import os
import time

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nexus.utils.log import setup_logging


SMARTRUN_TOKEN = os.environ.get('SMARTRUN_TOKEN', '')
SMARTRUN_CONCURRENCY = int(os.environ.get('SMARTRUN_CONCURRENCY', '7'))
SETTINGS_PACKAGE = 'com.android.settings'


@dataclass
class DeviceResult:
    name: str
    device_id: str | None
    session_id: str | None
    elapsed: float
    success: bool
    error: str = ''


async def save_screenshot(runtime: Any, path: Path) -> None:
    shot = await runtime.gui_screenshot()
    if not shot.success or not shot.image:
        raise RuntimeError(f'截图失败: {shot}')
    path.write_bytes(base64.b64decode(shot.image))


async def exercise_device(provider: Any, index: int, output_dir: Path) -> DeviceResult:
    name = f'android-{index:02d}'
    started = time.monotonic()
    device_id: str | None = None
    session_id: str | None = None

    try:
        runtime = await provider.acquire(name, purpose=f'nexus android concurrency test {index}/{SMARTRUN_CONCURRENCY}')
        device_id = provider.get_device_id(name)
        session_id = provider.get_runtime_id(name)
        device_dir = output_dir / name
        device_dir.mkdir(parents=True, exist_ok=True)
        print(f'[{name}] 已申请 device={device_id} session={session_id}')

        home = await runtime.gui_press_key('Home')
        if not home.success:
            raise RuntimeError(f'Home 操作失败: {home}')
        await asyncio.sleep(1)
        await save_screenshot(runtime, device_dir / '01_home.jpg')

        await runtime.gui_close_app(SETTINGS_PACKAGE)
        launched = await runtime.gui_launch_app(SETTINGS_PACKAGE)
        if not launched.success:
            raise RuntimeError(f'启动设置失败: {launched}')
        await asyncio.sleep(2)
        await save_screenshot(runtime, device_dir / '02_settings.jpg')

        closed = await runtime.gui_close_app(SETTINGS_PACKAGE)
        if not closed.success:
            raise RuntimeError(f'关闭设置失败: {closed}')
        final_home = await runtime.gui_press_key('Home')
        if not final_home.success:
            raise RuntimeError(f'最终 Home 操作失败: {final_home}')

        elapsed = time.monotonic() - started
        print(f'[{name}] 通过，耗时 {elapsed:.1f}s')
        return DeviceResult(name, device_id, session_id, elapsed, True)
    except Exception as exc:
        elapsed = time.monotonic() - started
        print(f'[{name}] 失败，耗时 {elapsed:.1f}s: {exc}')
        return DeviceResult(name, device_id, session_id, elapsed, False, str(exc))
    finally:
        await provider.release(name)
        print(f'[{name}] 已释放')


def print_summary(results: list[DeviceResult], total_elapsed: float, output_dir: Path) -> None:
    passed = [result for result in results if result.success]
    failed = [result for result in results if not result.success]

    print('\n' + '=' * 88)
    print(f'并发测试完成：{len(passed)}/{len(results)} 通过，总耗时 {total_elapsed:.1f}s')
    print(f'输出目录：{output_dir}')
    print('-' * 88)
    print(f'{"名称":<14} {"设备 ID":<28} {"结果":<8} {"耗时":>8}  错误')
    for result in results:
        status = 'PASS' if result.success else 'FAIL'
        print(f'{result.name:<14} {(result.device_id or "-"):<28} {status:<8} {result.elapsed:>7.1f}s  {result.error}')
    print('=' * 88)

    if failed:
        raise SystemExit(1)


async def main() -> None:
    setup_logging()
    from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider

    if not SMARTRUN_TOKEN:
        raise SystemExit('请先 source nexus-workspace/secret.env，提供 SMARTRUN_TOKEN')
    if SMARTRUN_CONCURRENCY < 1:
        raise SystemExit('SMARTRUN_CONCURRENCY 必须大于 0')

    provider = SmartRunRuntimeProvider(
        token=SMARTRUN_TOKEN,
        platform='android',
        duration=600,
        request_timeout=120,
    )
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'output' / 'android-concurrent' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 88)
    print(f'SmartRun Android 并发测试：同时申请 {SMARTRUN_CONCURRENCY} 台设备')
    print(f'输出目录：{output_dir}')
    print('=' * 88)

    started = time.monotonic()
    try:
        results = await asyncio.gather(
            *(exercise_device(provider, index, output_dir) for index in range(1, SMARTRUN_CONCURRENCY + 1))
        )
    finally:
        await provider.release_all()

    print_summary(results, time.monotonic() - started, output_dir)


if __name__ == '__main__':
    asyncio.run(main())
