"""End-to-end demo for the AGS WindowsAgentArena (WAA) provider.

模拟一个真实的 Windows 桌面任务：用 shell 创建文件、截图、鼠标点击、键盘输入、
读取屏幕尺寸，全程截图存档。不接 LLM，纯脚本驱动，演示 run_command + gui_*
executor（nexus interface）和原始 WAA HTTP 表面（raw interface）在真实
Windows 桌面任务里怎么配合。

Usage:
    # template 模式（E2B 模板）：
    uv run python docs/examples/scripts/test_ags_waa.py \
        --api-key <key> --template <waa-template> \
        [--domain ap-guangzhou.tencentags.com] \
        [--interface nexus|raw]

    # control_plane 模式（腾讯云控制面，复用已有内置 waa 工具）：
    uv run python docs/examples/scripts/test_ags_waa.py \
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

# Windows 桌面任务里要操作的目标文件（用 C:\\Users\\Docker\\Desktop，WAA VM 默认用户目录）
TARGET_FILE = r'C:\Users\Docker\Desktop\nexus_waa_demo.txt'
FILE_BODY = 'Hello from the Nexus WAA demo.\nLine 2: GUI + shell working together.\n'
GUI_TEXT = 'Appended by Nexus gui_type.'


def _write_file_command(path: str, content: str) -> str:
    """Build a quoting-safe PowerShell command that writes UTF-8 text."""
    escaped_path = path.replace("'", "''")
    escaped_content = content.replace("'", "''")
    script = (
        '$utf8 = New-Object System.Text.UTF8Encoding($false); '
        f"[System.IO.File]::WriteAllText('{escaped_path}', '{escaped_content}', $utf8)"
    )
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    return f'powershell.exe -NoProfile -NonInteractive -EncodedCommand {encoded}'


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='AGS WindowsAgentArena end-to-end demo')
    p.add_argument(
        '--provision',
        choices=['template', 'control_plane'],
        default='template',
        help='template=E2B 模板（需 --template + --api-key）; '
        'control_plane=腾讯云控制面（需 --secret-id + --secret-key + --tool-id，复用已有内置 waa 工具）',
    )
    p.add_argument('--template', default=None, help='AGS WAA template name (template 模式必填)')
    p.add_argument('--domain', default='ap-guangzhou.tencentags.com')
    p.add_argument('--region', default='ap-guangzhou', help='Tencent Cloud region (control_plane 模式)')
    p.add_argument('--interface', choices=['nexus', 'raw'], default='nexus')
    p.add_argument('--api-key', default=None, help='AGS API key (template 模式必填)')
    p.add_argument('--secret-id', default=None, help='Tencent Cloud SecretId (control_plane 模式必填)')
    p.add_argument('--secret-key', default=None, help='Tencent Cloud SecretKey (control_plane 模式必填)')
    p.add_argument('--tool-id', default=None, help='已有内置 waa 工具的 ToolId (control_plane 模式必填)')
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


async def _gui_shot(runtime: Any, out_dir: Path, name: str) -> None:
    """gui_screenshot 返回 base64（字段 'image'），解码后保存。"""
    shot = await runtime.call_executor('gui_screenshot', {})
    img = shot.get('image') or ''
    if img:
        _save_png(base64.b64decode(img), out_dir, name)


async def _test_nexus(runtime: Any, out_dir: Path) -> None:
    """真实 Windows 桌面任务：shell 造文件 -> 截图 -> 鼠标键盘操作 -> 读回文件。"""
    # 1. 设备信息（确认连上了 WindowsAgentArena VM）
    info = await runtime.call_executor('gui_get_device_info', {})
    print(f'  [1] device: {info.get("device_name")} size={info.get("window_width")}x{info.get("window_height")}')

    # 2. 屏幕尺寸，后面的点击坐标都基于它
    size = await runtime.call_executor('gui_get_screen_size', {})
    w, h = size.get('width', 1920), size.get('height', 1080)
    print(f'  [2] screen size: {w}x{h}')

    # 3. 用 PowerShell EncodedCommand 可靠创建 UTF-8 多行文本文件。
    write_cmd = _write_file_command(TARGET_FILE, FILE_BODY)
    res = await runtime.call_executor('run_command', {'command': write_cmd, 'timeout': 30.0})
    print(f'  [3] wrote {TARGET_FILE} (status={res.get("status")})')

    # 4. 用 GUI 打开文件，等待记事本窗口完成渲染。
    await runtime.call_executor('gui_launch_app', {'app': TARGET_FILE})
    await asyncio.sleep(3.0)
    print(f'  [4] opened {TARGET_FILE} in the default editor')
    await _gui_shot(runtime, out_dir, 'ags_waa_1_file_opened.png')

    # 5. 点击编辑区，跳到文件末尾并通过 GUI 追加一行。
    edit_x, edit_y = w // 2, h // 2
    await runtime.call_executor('gui_click', {'x': edit_x, 'y': edit_y})
    await runtime.call_executor('gui_press_key', {'key': 'end'})
    await runtime.call_executor('gui_press_key', {'key': 'enter'})
    await runtime.call_executor('gui_type', {'text': GUI_TEXT})
    print(f'  [5] appended text through GUI at ({edit_x}, {edit_y})')
    await _gui_shot(runtime, out_dir, 'ags_waa_2_after_type.png')

    # 6. Ctrl+S 保存；gui_press_key 只支持单键，组合键通过 Windows pyautogui 执行。
    await runtime.call_executor(
        'run_command',
        {'command': "python -c \"import pyautogui; pyautogui.hotkey('ctrl', 's')\"", 'timeout': 15.0},
    )
    await asyncio.sleep(1.0)

    # 7. 用 shell 读回文件内容，确认 GUI 与 shell 共享同一文件系统。
    read = await runtime.call_executor('run_command', {'command': f'type "{TARGET_FILE}"', 'timeout': 15.0})
    content = read.get('stdout') or ''
    print('  [7] file content via shell:\n      ' + content.strip().replace('\n', '\n      '))

    # 8. 在编辑器中从追加文本末尾向行首拖拽，截图应能看到高亮选区。
    text_y = int(h * 0.285)
    drag_start_x = int(w * 0.325)
    drag_end_x = int(w * 0.112)
    await runtime.call_executor(
        'gui_drag',
        {'x1': drag_start_x, 'y1': text_y, 'x2': drag_end_x, 'y2': text_y, 'duration': 700},
    )
    print(f'  [8] dragged across the appended text from x={drag_start_x} to x={drag_end_x} at y={text_y}')
    await _gui_shot(runtime, out_dir, 'ags_waa_3_text_selected.png')

    # 9. 关闭窗口并清理。
    await runtime.call_executor('gui_close_app', {'app': 'Notepad'})
    await runtime.call_executor('run_command', {'command': f'del /f "{TARGET_FILE}"', 'timeout': 15.0})
    print('  [done] closed editor and cleaned up target file')


async def _test_raw(runtime: Any, out_dir: Path) -> None:
    """raw interface：直接打 WAA 原始 HTTP 表面完成同类任务。"""
    # 1. 屏幕尺寸
    size = await runtime.screen_size()
    print(f'  [1] screen_size: {size}')

    # 2. 用 execute 在 Windows 上可靠创建 UTF-8 多行文件
    res = await runtime.execute(_write_file_command(TARGET_FILE, FILE_BODY), shell=True)
    print(f'  [2] execute result: {res}')

    # 3. 用 setup_open_file 打开文件（原始端点）
    resp_text = await runtime.setup_open_file(TARGET_FILE)
    print(f'  [3] setup_open_file {TARGET_FILE}: {resp_text[:100]!r}')
    await asyncio.sleep(2.0)

    # 4. 截图（原始 /screenshot 直接返回 bytes）
    screenshot_bytes = await runtime.screenshot()
    print(f'  [4] screenshot: {len(screenshot_bytes)} bytes')
    if screenshot_bytes:
        _save_png(screenshot_bytes, out_dir, 'ags_waa_raw_1_opened.png')

    # 5. 读 accessibility tree，演示原始观测能力
    tree = await runtime.accessibility_tree(backend='uia')
    print(f'  [5] accessibility_tree: {len(str(tree))} chars')

    # 6. 列出桌面目录，确认文件存在
    listing = await runtime.list_directory(r'C:\Users\Docker\Desktop')
    print(f'  [6] list_directory Desktop: {str(listing)[:200]}')

    # 7. 用 setup_close_all 关闭所有窗口
    close_resp = await runtime.setup_close_all()
    print(f'  [7] setup_close_all: {close_resp[:100]!r}')

    # 8. 清理
    await runtime.execute(f'del /f "{TARGET_FILE}"', shell=True)
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
            print('ERROR: control_plane 模式需要 --tool-id（已有内置 waa 工具的 ToolId）', file=sys.stderr)
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
        # control_plane：用 tool_id 直接复用已有内置 waa 工具，provider 不按名查找、不新建、不删除。
        acquire_params: dict[str, Any] = {
            'runtime_kind': 'waa',
            'tool_type': 'waa',
            'tool_id': args.tool_id,
        }
        print(
            f'[ags-waa] provision=control_plane interface={args.interface} region={args.region} tool_id={args.tool_id}'
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
            'runtime_kind': 'waa',
            'template': args.template,
        }
        print(f'[ags-waa] provision=template interface={args.interface} template={args.template}')

    out_dir = Path(args.output_dir)
    runtime = None
    try:
        runtime = await provider.acquire('main', **acquire_params)
        print('[ags-waa] runtime acquired')

        if args.interface == 'nexus':
            await _test_nexus(runtime, out_dir)
        else:
            await _test_raw(runtime, out_dir)

        print('[ags-waa] ALL PASSED')
        return 0
    except Exception as e:
        print(f'[ags-waa] FAILED: {e}', file=sys.stderr)
        return 1
    finally:
        await provider.release_all()


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
