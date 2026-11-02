"""End-to-end demo for the AGS OSWorld provider.

模拟一个真实的桌面任务：在桌面环境里用 shell 创建并写入文件、用 GUI 打开文件、
做鼠标键盘操作、读取 accessibility tree、全程截图存档。 不接 LLM，纯脚本驱动，
演示 run_command + gui_* executor（nexus interface）和原始 OSWorld HTTP 表面
（raw interface）在真实桌面任务里怎么配合。

Usage:
    # template 模式（E2B 模板）：
    uv run python docs/examples/scripts/test_ags_osworld.py \
        --api-key <key> --template <osworld-template> \
        [--domain ap-guangzhou.tencentags.com] \
        [--interface nexus|raw]

    # control_plane 模式（腾讯云控制面，复用已有内置 osworld 工具）：
    uv run python docs/examples/scripts/test_ags_osworld.py \
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

# 桌面任务里要操作的目标文件（用 /tmp，桌面环境里全局可写；/root 通常不可写）
TARGET_FILE = '/tmp/nexus_osworld_demo.txt'
FILE_BODY = 'Hello from the Nexus OSWorld demo.\nLine 2: GUI + shell working together.\n'


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='AGS OSWorld end-to-end demo')
    p.add_argument(
        '--provision',
        choices=['template', 'control_plane'],
        default='template',
        help='template=E2B 模板（需 --template + --api-key）; '
        'control_plane=腾讯云控制面（需 --secret-id + --secret-key + --tool-id，复用已有内置 osworld 工具）',
    )
    p.add_argument('--template', default=None, help='AGS OSWorld template name (template 模式必填)')
    p.add_argument('--domain', default='ap-guangzhou.tencentags.com')
    p.add_argument('--region', default='ap-guangzhou', help='Tencent Cloud region (control_plane 模式)')
    p.add_argument('--interface', choices=['nexus', 'raw'], default='nexus')
    p.add_argument('--api-key', default=None, help='AGS API key (template 模式必填)')
    p.add_argument('--secret-id', default=None, help='Tencent Cloud SecretId (control_plane 模式必填)')
    p.add_argument('--secret-key', default=None, help='Tencent Cloud SecretKey (control_plane 模式必填)')
    p.add_argument('--tool-id', default=None, help='已有内置 osworld 工具的 ToolId (control_plane 模式必填)')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Directory to save screenshots')
    return p.parse_args()


def _save_png(data: bytes, out_dir: Path, name: str) -> None:
    """Save PNG bytes to <out_dir>/<name> and print the absolute path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_bytes(data)
    print(f'    saved screenshot -> {path.resolve()}')


async def _gui_shot(runtime: Any, out_dir: Path, name: str) -> None:
    """gui_screenshot 返回 base64（字段 'image'），解码后保存。"""
    shot = await runtime.call_executor('gui_screenshot', {})
    img = shot.get('image') or ''
    if img:
        _save_png(base64.b64decode(img), out_dir, name)


async def _test_nexus(runtime: Any, out_dir: Path) -> None:
    """真实桌面任务：shell 造文件 -> GUI 打开 -> 鼠标键盘操作 -> 截图。"""
    # 1. 屏幕尺寸，后面的点击坐标都基于它
    size = await runtime.call_executor('gui_get_screen_size', {})
    w, h = size.get('width', 1920), size.get('height', 1080)
    print(f'  [1] screen size: {w}x{h}')

    # 2. 用 shell 创建一个文本文件（真实任务常见的环境准备）
    write_cmd = f'cat > {TARGET_FILE} <<\'EOF\'\n{FILE_BODY}EOF\necho "bytes:" $(wc -c < {TARGET_FILE})'
    res = await runtime.call_executor('run_command', {'command': write_cmd, 'timeout': 30.0})
    print(f'  [2] wrote {TARGET_FILE}: {res.get("stdout", "").strip()!r} (status={res.get("status")})')
    assert res.get('return_code') == 0, f'write failed: {res}'

    # 3. 用 GUI 打开这个文件（gui_launch_app 在 OSWorld 里映射到 open_file）
    await runtime.call_executor('gui_launch_app', {'app': TARGET_FILE})
    await asyncio.sleep(2.0)
    print(f'  [3] opened {TARGET_FILE} in default GUI app')
    await _gui_shot(runtime, out_dir, 'ags_osworld_1_file_opened.png')

    # 4. 鼠标移动到屏幕中央并点击，模拟聚焦窗口
    await runtime.call_executor('gui_click', {'x': w // 2, 'y': h // 2})
    print(f'  [4] clicked center ({w // 2}, {h // 2})')

    # 5. 键盘输入一行文字（演示 gui_type），再按 End/Enter
    await runtime.call_executor('gui_type', {'text': 'appended by gui_type'})
    await runtime.call_executor('gui_press_key', {'key': 'enter'})
    print('  [5] typed text + pressed Enter')
    await _gui_shot(runtime, out_dir, 'ags_osworld_2_after_type.png')

    # 6. 再用 shell 确认文件还在、读回内容（GUI 与 shell 共享同一文件系统）
    read = await runtime.call_executor('run_command', {'command': f'cat {TARGET_FILE}', 'timeout': 15.0})
    print('  [6] file content via shell:\n      ' + read.get('stdout', '').strip().replace('\n', '\n      '))

    # 7. 清理
    await runtime.call_executor('run_command', {'command': f'rm -f {TARGET_FILE}', 'timeout': 15.0})
    print('  [done] cleaned up target file')


async def _test_raw(runtime: Any, out_dir: Path) -> None:
    """raw interface：直接打 OSWorld 原始 HTTP 表面完成同类任务。"""
    # 1. 屏幕尺寸
    size = await runtime.screen_size()
    print(f'  [1] screen_size: {size}')

    # 2. run_bash_script 造文件
    script = f"cat > {TARGET_FILE} <<'EOF'\n{FILE_BODY}EOF\nwc -c {TARGET_FILE}"
    res = await runtime.run_bash_script(script, timeout=30)
    print(f'  [2] run_bash_script result: {res}')

    # 3. run_python 在桌面环境里跑 Python，读回文件
    py = await runtime.run_python(f'print(open({TARGET_FILE!r}).read())')
    print(f'  [3] run_python read-back: {py}')

    # 4. 用 setup_open_file 打开文件（原始端点，返回原始 HTTPResponse；成功时 body 为空）
    resp = await runtime.setup_open_file(TARGET_FILE)
    print(f'  [4] setup_open_file {TARGET_FILE}: HTTP {resp.status_code}')
    await asyncio.sleep(2.0)

    # 5. 截图（原始 /screenshot 直接返回 bytes）
    screenshot_bytes = await runtime.screenshot()
    print(f'  [5] screenshot: {len(screenshot_bytes)} bytes')
    if screenshot_bytes:
        _save_png(screenshot_bytes, out_dir, 'ags_osworld_raw_1_opened.png')

    # 6. 读 accessibility tree，演示原始观测能力
    tree = await runtime.accessibility_tree()
    print(f'  [6] accessibility_tree: {len(str(tree))} chars')

    # 7. 清理
    await runtime.execute(f'rm -f {TARGET_FILE}', shell=True)
    print('  [done] cleaned up target file')


async def main() -> int:
    setup_logging()
    args = _parse_args()

    from nexus.runtime.providers.ags import AGSRuntimeProvider

    if args.provision == 'control_plane':
        if not (args.secret_id and args.secret_key):
            print('ERROR: control_plane 模式需要 --secret-id 和 --secret-key', file=sys.stderr)
            return 1
        if not args.tool_id:
            print('ERROR: control_plane 模式需要 --tool-id（已有内置 osworld 工具的 ToolId）', file=sys.stderr)
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
        # control_plane：用 tool_id 直接复用已有内置 osworld 工具，provider 不按名查找、不新建、不删除。
        acquire_params: dict[str, Any] = {
            'runtime_kind': 'osworld',
            'tool_type': 'osworld',
            'tool_id': args.tool_id,
        }
        print(
            f'[ags-osworld] provision=control_plane interface={args.interface} region={args.region} tool_id={args.tool_id}'
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
            'runtime_kind': 'osworld',
            'template': args.template,
        }
        print(f'[ags-osworld] provision=template interface={args.interface} template={args.template}')

    out_dir = Path(args.output_dir)
    runtime = None
    try:
        runtime = await provider.acquire('main', **acquire_params)
        print('[ags-osworld] runtime acquired')

        if args.interface == 'nexus':
            await _test_nexus(runtime, out_dir)
        else:
            await _test_raw(runtime, out_dir)

        print('[ags-osworld] ALL PASSED')
        return 0
    except Exception as e:
        print(f'[ags-osworld] FAILED: {e}', file=sys.stderr)
        return 1
    finally:
        await provider.release_all()


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
