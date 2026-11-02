"""CloudTest Android Runtime 使用示例。

通过已有 Android 设备验证连接、截图、按键、应用管理、滑动和点击。
需要安装 provider 依赖：uv run nexus modules install android_cloudtest

运行方式：
  export CLOUDTEST_USER_ID='<your-user-id>'
  export CLOUDTEST_USER_KEY='<your-user-key>'
  export CLOUDTEST_DEVICE_ID='12345'  # 替换为实际设备 ID
  uv run python docs/examples/scripts/test_cloudtest_android.py

CLI 参数优先于对应环境变量；必填项可通过任一方式提供。
CLI 参数：--user-id、--user-key、--device-id、--host、--is-public-cloud / --no-is-public-cloud、--project。

环境变量：
  CLOUDTEST_USER_ID          CloudTest 用户 ID（必填）
  CLOUDTEST_USER_KEY         CloudTest 用户密钥（必填）
  CLOUDTEST_DEVICE_ID        已有 Android 设备 ID（必填，整数）
  CLOUDTEST_HOST             API 地址（可选，默认 http://api.cloudtest.woa.com/）
  CLOUDTEST_IS_PUBLIC_CLOUD  是否公有云（可选，默认 false）
  CLOUDTEST_PROJECT          公有云项目名（可选）

输出目录：docs/examples/scripts/output/cloudtest_android/<timestamp>/
结束时关闭设置并返回桌面，release_all 断开本地连接，不销毁或归还云端设备。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os

from datetime import datetime
from pathlib import Path
from typing import Any

from nexus.utils.log import setup_logging


_step_no = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CloudTest Android Runtime 使用示例（CLI 参数优先于环境变量）')
    parser.add_argument('--user-id', default=os.environ.get('CLOUDTEST_USER_ID'), help='CloudTest 用户 ID')
    parser.add_argument('--user-key', default=os.environ.get('CLOUDTEST_USER_KEY'), help='CloudTest 用户密钥')
    parser.add_argument('--device-id', type=int, default=os.environ.get('CLOUDTEST_DEVICE_ID'), help='Android 设备 ID')
    parser.add_argument(
        '--host', default=os.environ.get('CLOUDTEST_HOST', 'http://api.cloudtest.woa.com/'), help='CloudTest API 地址'
    )
    parser.add_argument(
        '--is-public-cloud', action=argparse.BooleanOptionalAction, default=None, help='是否公有云（默认 false）'
    )
    parser.add_argument('--project', default=os.environ.get('CLOUDTEST_PROJECT'), help='公有云项目名')
    args = parser.parse_args()
    for name in ('user_id', 'user_key', 'device_id'):
        if getattr(args, name) in (None, ''):
            parser.error(f'请提供 --{name.replace("_", "-")} 或 CLOUDTEST_{name.upper()} 环境变量')
    if args.is_public_cloud is None:
        value = os.environ.get('CLOUDTEST_IS_PUBLIC_CLOUD', 'false').lower()
        if value not in ('true', 'false', '1', '0'):
            parser.error('CLOUDTEST_IS_PUBLIC_CLOUD 必须是 true/false 或 1/0')
        args.is_public_cloud = value in ('true', '1')
    return args


def step(desc: str) -> None:
    global _step_no
    _step_no += 1
    print(f'\n[{_step_no:02d}] {desc}')


async def execute(runtime: Any, name: str, **params: Any) -> dict[str, Any]:
    result = await runtime.call_executor(name, params)
    if not result.get('success'):
        raise RuntimeError(f'{name} 失败: {result.get("error", "未返回成功状态")}')
    return result


async def snap(runtime: Any, run_dir: Path, name: str) -> None:
    shot = await execute(runtime, 'gui_screenshot')
    if not shot.get('image'):
        raise RuntimeError('截图失败: 图像为空')
    path = run_dir / f'{_step_no:02d}_{name}.png'
    path.write_bytes(base64.b64decode(shot['image'], validate=True))
    print(f'     截图已保存: {path.name}')


async def demo(runtime: Any, run_dir: Path) -> None:
    step('gui_get_screen_size —— 读取 Android 屏幕分辨率')
    size = await execute(runtime, 'gui_get_screen_size')
    width, height = size.get('width'), size.get('height')
    if not width or not height or width <= 0 or height <= 0:
        raise RuntimeError(f'无效屏幕分辨率: {width} x {height}')
    print(f'     屏幕分辨率: {width} x {height}')

    try:
        step("gui_press_key('Home') —— 返回 Android 桌面")
        await execute(runtime, 'gui_press_key', key='Home')
        await asyncio.sleep(2)
        await snap(runtime, run_dir, 'home')

        step("gui_launch_app('com.android.settings') —— 启动系统设置")
        await execute(runtime, 'gui_launch_app', app='com.android.settings')
        await asyncio.sleep(3)
        await snap(runtime, run_dir, 'settings')

        step('gui_drag —— 上滑浏览设置列表')
        # executor 接收像素坐标，CloudTest runtime 内部转换为 0–1000 坐标。
        await execute(
            runtime,
            'gui_drag',
            x1=width // 2,
            y1=int(height * 0.75),
            x2=width // 2,
            y2=int(height * 0.25),
            duration=400,
        )
        await asyncio.sleep(2)
        await snap(runtime, run_dir, 'scrolled')

        step('gui_click —— 点击设置列表中部，截图记录实际页面')
        await execute(runtime, 'gui_click', x=width // 2, y=int(height * 0.4))
        await asyncio.sleep(2)
        await snap(runtime, run_dir, 'clicked')

        step("gui_press_key('Back') —— 返回")
        await execute(runtime, 'gui_press_key', key='Back')
        await asyncio.sleep(1)
        await snap(runtime, run_dir, 'back')
    finally:
        step('gui_close_app + gui_press_key —— 关闭设置并返回桌面')
        try:
            await execute(runtime, 'gui_close_app', app='com.android.settings')
        finally:
            await execute(runtime, 'gui_press_key', key='Home')

    await asyncio.sleep(1)
    await snap(runtime, run_dir, 'done')


async def main() -> None:
    setup_logging()
    args = parse_args()

    from nexus.runtime.providers.android_cloudtest.provider import CloudTestAndroidProvider

    provider = CloudTestAndroidProvider(
        user_id=args.user_id,
        user_key=args.user_key,
        host=args.host,
        is_public_cloud=args.is_public_cloud,
        project=args.project or None,
    )
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(__file__).resolve().parent / 'output' / 'cloudtest_android' / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('CloudTest Android Runtime 使用示例')
    print(f'输出目录: {run_dir}')
    print('=' * 60)

    try:
        runtime = await provider.acquire('main', device_id=args.device_id)
        await demo(runtime, run_dir)
        print(f'\n测试完成，共 {_step_no} 步，截图见 {run_dir}')
    finally:
        await provider.release_all()
        print('本地设备连接已断开')


if __name__ == '__main__':
    asyncio.run(main())
