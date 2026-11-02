"""DevCloud macOS Runtime 使用示例。

运行方式：
  cd nexus
  uv run python docs/examples/scripts/test_devcloud_macos.py \
    --app-id <your_app_id> --app-token <your_app_token>

也可以通过 DEVCLOUD_APP_ID / DEVCLOUD_APP_TOKEN 提供凭据；其他参数都有默认值，
可按需通过命令行覆盖。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys

from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nexus.runtime.executors.models.file import FileItem  # noqa: E402
from nexus.runtime.providers.devcloud.provider import DevCloudRuntimeProvider  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / 'output'


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='DevCloud macOS Runtime end-to-end demo')
    p.add_argument(
        '--app-id', default=os.getenv('DEVCLOUD_APP_ID'), help='DevCloud APPID（必填，可用 DEVCLOUD_APP_ID）'
    )
    p.add_argument(
        '--app-token',
        default=os.getenv('DEVCLOUD_APP_TOKEN'),
        help='DevCloud appToken（必填，可用 DEVCLOUD_APP_TOKEN）',
    )
    p.add_argument(
        '--sandbox-api-base-url',
        default=os.getenv('DEVCLOUD_SANDBOX_API_BASE_URL', 'http://21.6.221.82:18800'),
        help='DevCloud sandbox lifecycle API base URL',
    )
    p.add_argument(
        '--lease-time',
        type=int,
        default=int(os.getenv('DEVCLOUD_LEASE_TIME', '300')),
        help='Sandbox lease time in seconds',
    )
    p.add_argument(
        '--server-port',
        type=int,
        default=int(os.getenv('DEVCLOUD_SERVER_PORT', '5000')),
        help='OSWorld HTTP server port',
    )
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_ROOT), help='Directory to save screenshots')
    args = p.parse_args()
    if not args.app_id:
        p.error('--app-id is required (or set DEVCLOUD_APP_ID)')
    if not args.app_token:
        p.error('--app-token is required (or set DEVCLOUD_APP_TOKEN)')
    return args


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: ('***REDACTED***' if 'password' in k.lower() or 'token' in k.lower() else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _ensure_success(label: str, value: Any) -> None:
    success = getattr(value, 'success', None)
    error = getattr(value, 'error', None)
    if success is False:
        raise RuntimeError(f'{label} failed: {error}')


def _save_screenshot_image(image: str, output_dir: Path, name: str) -> Path:
    path = output_dir / name
    path.write_bytes(base64.b64decode(image))
    return path


async def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = DevCloudRuntimeProvider(
        app_id=args.app_id,
        app_token=args.app_token,
        sandbox_api_base_url=args.sandbox_api_base_url,
        lease_time=args.lease_time,
        server_port=args.server_port,
    )

    try:
        print('Acquiring DevCloud macOS runtime...')
        runtime = await provider.acquire('main')
        print('Runtime metadata:', _redact(getattr(runtime, 'metadata', {})))
        print('Runtime base_url:', runtime.base_url)

        print('Running command...')
        command = await runtime.run_command('uname -a && sw_vers || true', timeout=60)
        print('status:', command.status)
        print('stdout:\n', command.stdout)
        print('stderr:\n', command.stderr)

        print('Getting screen size...')
        screen = await runtime.gui_get_screen_size()
        _ensure_success('gui_get_screen_size', screen)
        print('screen:', screen.model_dump())

        print('Opening TextEdit for observable GUI actions...')
        textedit_setup = await runtime.run_command(
            "rm -f /tmp/nexus-devcloud-gui-demo.txt && touch /tmp/nexus-devcloud-gui-demo.txt && "
            "open -a TextEdit /tmp/nexus-devcloud-gui-demo.txt && "
            "osascript -e 'tell application \"TextEdit\" to activate' "
            "-e 'tell application \"TextEdit\" to set bounds of front window to {220, 120, 1120, 720}'",
            timeout=30,
        )
        print('TextEdit setup status:', textedit_setup.status)
        if textedit_setup.return_code not in (0, None):
            print('WARNING: TextEdit setup stderr:', textedit_setup.stderr)
        await asyncio.sleep(2)

        print('TextEdit GUI action demo: click, type, press key, double click, long click, and drag...')
        text_left = 230
        first_line_y = 155
        second_line_y = 172
        typing_actions = [
            ('gui_click_textedit_body', lambda: runtime.gui_click(text_left, first_line_y)),
            ('gui_type_first_line', lambda: runtime.gui_type('Nexus DevCloud GUI smoke test')),
            ('gui_press_key_enter', lambda: runtime.gui_press_key('enter')),
            ('gui_type_second_line', lambda: runtime.gui_type('Visible click type enter drag demo')),
        ]
        for label, action in typing_actions:
            result = await action()
            _ensure_success(label, result)
            print(f'{label}:', result.model_dump())
            await asyncio.sleep(0.2)

        print('Taking screenshot after TextEdit typing actions...')
        typed_screenshot = await runtime.gui_screenshot()
        if not typed_screenshot.success or not typed_screenshot.image:
            raise RuntimeError(f'TextEdit typed screenshot failed: {typed_screenshot.error}')
        typed_screenshot_path = _save_screenshot_image(
            typed_screenshot.image, output_dir, 'devcloud_macos_textedit_typed.png'
        )
        print('Saved TextEdit typed screenshot:', typed_screenshot_path)

        selection_actions = [
            ('gui_double_click_first_word', lambda: runtime.gui_double_click(text_left + 18, first_line_y)),
            ('gui_long_click_second_line', lambda: runtime.gui_long_click(text_left + 40, second_line_y, duration=300)),
            (
                'gui_drag_select_second_line',
                lambda: runtime.gui_drag(text_left, second_line_y, text_left + 285, second_line_y, duration=700),
            ),
        ]
        for label, action in selection_actions:
            result = await action()
            _ensure_success(label, result)
            print(f'{label}:', result.model_dump())
            await asyncio.sleep(0.3)

        print('Taking screenshot after TextEdit drag selection...')
        selection_screenshot = await runtime.gui_screenshot()
        if not selection_screenshot.success or not selection_screenshot.image:
            raise RuntimeError(f'TextEdit drag selection screenshot failed: {selection_screenshot.error}')
        selection_screenshot_path = _save_screenshot_image(
            selection_screenshot.image, output_dir, 'devcloud_macos_textedit_drag_selection.png'
        )
        print('Saved TextEdit drag selection screenshot:', selection_screenshot_path)

        print('Launching and closing System Settings...')
        launch = await runtime.gui_launch_app('/System/Applications/System Settings.app')
        _ensure_success('gui_launch_app', launch)
        print('gui_launch_app:', launch.model_dump())
        activate = await runtime.run_command(
            'open -a "System Settings" && osascript -e \'tell application "System Settings" to activate\'',
            timeout=30,
        )
        print('activate System Settings status:', activate.status)
        if activate.return_code not in (0, None):
            print('WARNING: System Settings activate stderr:', activate.stderr)
        await asyncio.sleep(3)

        print('Taking screenshot after launch...')
        launch_screenshot = await runtime.gui_screenshot()
        if not launch_screenshot.success or not launch_screenshot.image:
            raise RuntimeError(f'launch screenshot failed: {launch_screenshot.error}')
        launch_screenshot_path = _save_screenshot_image(
            launch_screenshot.image, output_dir, 'devcloud_macos_after_launch.png'
        )
        print('Saved launch screenshot:', launch_screenshot_path)

        try:
            close = await asyncio.wait_for(runtime.gui_close_app('System Settings'), timeout=15)
            print('gui_close_app:', close.model_dump())
            if not close.success:
                print(f'WARNING: gui_close_app failed but continuing demo: {close.error}')
        except TimeoutError:
            print('WARNING: gui_close_app timed out after 15s; continuing demo')
        await asyncio.sleep(1)

        print('Taking screenshot after close...')
        close_screenshot = await runtime.gui_screenshot()
        if not close_screenshot.success or not close_screenshot.image:
            raise RuntimeError(f'close screenshot failed: {close_screenshot.error}')
        close_screenshot_path = _save_screenshot_image(
            close_screenshot.image, output_dir, 'devcloud_macos_after_close.png'
        )
        print('Saved close screenshot:', close_screenshot_path)

        print('Taking final screenshot...')
        screenshot = await runtime.gui_screenshot()
        if not screenshot.success or not screenshot.image:
            raise RuntimeError(f'screenshot failed: {screenshot.error}')
        screenshot_path = _save_screenshot_image(screenshot.image, output_dir, 'devcloud_macos_screenshot.png')
        print('Saved screenshot:', screenshot_path)

        content = 'hello from Nexus DevCloud example\n'
        remote_path = '/tmp/hello-devcloud.txt'
        print('Uploading file...')
        upload = await runtime.upload_file(files=[FileItem(path=remote_path, content=content, encoding='utf8')])
        print('upload:', upload.model_dump())
        failed_uploads = [item for item in upload.results if not item.success]
        if failed_uploads:
            raise RuntimeError(f'upload failed: {[item.model_dump() for item in failed_uploads]}')

        print('Verifying remote file with cat...')
        cat_result = await runtime.run_command(f'cat {remote_path}', timeout=30)
        print('cat stdout:', cat_result.stdout)

        print('Downloading file...')
        download = await runtime.download_file(files=[FileItem(path=remote_path, encoding='utf8')])
        print('download:', download.model_dump())
        if download.errors:
            raise RuntimeError(f'download failed: {[item.model_dump() for item in download.errors]}')
        downloaded = download.files[0].content if download.files else None
        if downloaded != content:
            raise RuntimeError(f'downloaded content mismatch: {downloaded!r}')
        print('File round-trip OK')
    finally:
        print('Releasing DevCloud runtime...')
        await provider.release('main')


if __name__ == '__main__':
    asyncio.run(main(_parse_args()))
