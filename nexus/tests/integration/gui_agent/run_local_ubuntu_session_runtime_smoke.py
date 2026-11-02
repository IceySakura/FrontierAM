"""Validate a multi-window Ubuntu session on the local Docker GUI runtime."""

from __future__ import annotations

import asyncio
import base64
import json
import sys

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RUNTIME_NAME = 'ubuntu-session-smoke'
DEFAULT_OUTPUT_DIR = Path('outputs/gui_agent/ubuntu_session')
DEFAULT_OUTPUT_FILE = '/tmp/nexus_ubuntu_session_multi_window.txt'
DEFAULT_OUTPUT_TEXT = 'hello from nexus ubuntu session multi window'


@dataclass(frozen=True)
class StepResult:
    """Outcome for a single Ubuntu session smoke-test step."""

    name: str
    success: bool
    details: dict[str, Any]


def _output_dir() -> Path:
    """Resolve and create the artifact output directory."""
    import os

    output_dir = Path(os.getenv('NEXUS_GUI_OUTPUT_DIR', str(DEFAULT_OUTPUT_DIR))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _record(step_results: list[StepResult], step: StepResult) -> None:
    """Record and stream a compact JSON line for one smoke step."""
    step_results.append(step)
    print(json.dumps({'step': step.name, 'success': step.success}, default=str), flush=True)


def _command_step(name: str, command_result: Any, success: bool | None = None) -> StepResult:
    """Convert a runtime command result into a smoke-test step."""
    details = command_result.model_dump()
    passed = command_result.return_code == 0 if success is None else success
    if not passed:
        raise RuntimeError(f'{name} failed: {details}')
    return StepResult(name=name, success=True, details=details)


def _gui_step(name: str, response: Any) -> StepResult:
    """Convert a GUI runtime response into a strictly validated smoke-test step."""
    details = response.model_dump()
    if not response.success:
        raise RuntimeError(f'{name} failed: {details}')
    return StepResult(name=name, success=True, details=details)


async def _wait_for_command(
    runtime: Any,
    command: str,
    *,
    timeout_seconds: float = 60.0,
    interval_seconds: float = 2.0,
) -> Any:
    """Poll a runtime command until it succeeds or return the final failed result."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_result = None
    while True:
        last_result = await runtime.run_command(command)
        if last_result.return_code == 0:
            return last_result
        if asyncio.get_running_loop().time() >= deadline:
            return last_result
        await asyncio.sleep(interval_seconds)


async def run_ubuntu_session_smoke() -> dict[str, Any]:
    """Run the Ubuntu session validation flow through DockerRuntimeProvider."""
    from nexus.runtime.providers.docker import DockerRuntimeProvider
    from tests.integration.gui_agent.runtime_image import ensure_runtime_image, require_desktop_image

    image = require_desktop_image()
    await ensure_runtime_image()

    provider = DockerRuntimeProvider(
        runtime_image='current',
        force_rebuild=False,
        exposed_ports=[5900, 6080],
        print_port_mappings=True,
        auto_connect_ws=False,
    )
    runtime = None
    step_results: list[StepResult] = []
    output_dir = _output_dir()

    try:
        runtime = await provider.acquire(
            DEFAULT_RUNTIME_NAME,
            image=image,
            extra_args=[
                '--tmpfs',
                '/run',
                '--tmpfs',
                '/run/lock',
                '-e',
                'GTK_A11Y=none',
                '-e',
                'NEXUS_GUI_REQUIRE_SYSTEM_DBUS=1',
            ],
        )

        gui_start = await runtime.gui_start(desktop_session='gnome-session', gnome_session_name='ubuntu')
        _record(step_results, _gui_step('gui_start_gnome_session', gui_start))

        await asyncio.sleep(4.0)

        session_process = await runtime.run_command("ps -ef | grep '[g]nome-session-binary'")
        _record(step_results, _command_step('gnome_session_process_running', session_process))

        failed_session = await runtime.run_command("if ps -ef | grep '[g]nome-session-failed'; then exit 1; fi")
        _record(step_results, _command_step('gnome_session_failed_absent', failed_session))

        logind_process = await runtime.run_command("ps -ef | grep '[s]ystemd-logind'")
        _record(step_results, _command_step('systemd_logind_process_running', logind_process))

        shell_process = await runtime.run_command("ps -ef | grep '[g]nome-shell'")
        _record(step_results, _command_step('gnome_shell_process_running', shell_process))

        openbox_process = await runtime.run_command("if ps -ef | grep '[o]penbox'; then exit 1; fi")
        _record(step_results, _command_step('openbox_process_absent', openbox_process))

        screen_size = await runtime.gui_get_screen_size()
        if not screen_size.success:
            raise RuntimeError(f'gui_get_screen_size failed: {screen_size.model_dump()}')
        _record(step_results, StepResult('gui_get_screen_size', True, screen_size.model_dump()))

        escape_overview = await runtime.gui_press_key('Escape')
        _record(step_results, _gui_step('gui_press_key_escape_overview', escape_overview))
        await asyncio.sleep(0.5)

        screenshot = await runtime.gui_screenshot()
        if not screenshot.success or not screenshot.image:
            raise RuntimeError(f'gui_screenshot failed: {screenshot.model_dump()}')
        screenshot_path = output_dir / 'ubuntu-session.png'
        screenshot_path.write_bytes(base64.b64decode(screenshot.image))
        if screenshot.width < 1000 or screenshot.height < 700 or screenshot_path.stat().st_size < 5_000:
            raise RuntimeError(f'Ubuntu session screenshot is unexpectedly small: {screenshot_path.stat().st_size}')
        _record(
            step_results,
            StepResult(
                'gui_screenshot_ubuntu_session',
                True,
                {'path': str(screenshot_path), 'bytes': screenshot_path.stat().st_size},
            ),
        )

        output_file = DEFAULT_OUTPUT_FILE
        output_text = DEFAULT_OUTPUT_TEXT
        xterm_command = 'xterm -title NexusUbuntuSessionXterm -geometry 82x22+80+90 -fa Monospace -fs 12'
        launch_xterm = await runtime.gui_launch_app(xterm_command)
        _record(step_results, _gui_step('gui_launch_app_xterm', launch_xterm))

        await asyncio.sleep(1.0)
        _record(step_results, _gui_step('gui_click_xterm', await runtime.gui_click(220, 180)))

        type_command = f'printf %s {output_text!r} > {output_file}'
        type_result = await runtime.gui_type(type_command)
        _record(step_results, _gui_step('gui_type_xterm_command', type_result))

        enter_result = await runtime.gui_press_key('Enter')
        _record(step_results, _gui_step('gui_press_key_enter', enter_result))

        verify_file = await runtime.run_command(f'cat {output_file}')
        if verify_file.return_code != 0 or verify_file.stdout.strip() != output_text:
            raise RuntimeError(f'run_command_verify_file failed: {verify_file.model_dump()}')
        _record(step_results, _command_step('run_command_verify_file', verify_file))

        dialog_a_command = (
            "xmessage -title 'Nexus Ubuntu Dialog A' -buttons Close:0 "
            "-geometry 320x140+420+300 'First Ubuntu session window'"
        )
        dialog_b_command = (
            "xmessage -title 'Nexus Ubuntu Dialog B' -buttons Close:0 "
            "-geometry 320x140+760+340 'Second Ubuntu session window'"
        )
        launch_a = await runtime.gui_launch_app(dialog_a_command)
        _record(step_results, _gui_step('gui_launch_app_dialog_a', launch_a))

        launch_b = await runtime.gui_launch_app(dialog_b_command)
        _record(step_results, _gui_step('gui_launch_app_dialog_b', launch_b))

        await asyncio.sleep(2.0)
        xterm_window = await runtime.run_command(
            'DISPLAY=:99 /nix/runtime/nix-env/bin/xdotool search --class xterm '
            "|| ps -ef | grep '[x]term -title NexusUbuntuSessionXterm'"
        )
        _record(step_results, _command_step('xterm_window_present', xterm_window))

        dialog_a_window = await runtime.run_command(
            "DISPLAY=:99 /nix/runtime/nix-env/bin/xdotool search --onlyvisible --name 'Nexus Ubuntu Dialog A'"
        )
        _record(step_results, _command_step('dialog_a_window_visible', dialog_a_window))

        dialog_b_window = await runtime.run_command(
            "DISPLAY=:99 /nix/runtime/nix-env/bin/xdotool search --onlyvisible --name 'Nexus Ubuntu Dialog B'"
        )
        _record(step_results, _command_step('dialog_b_window_visible', dialog_b_window))

        dialog_screenshot = await runtime.gui_screenshot()
        dialog_path = output_dir / 'ubuntu-session-multi-window.png'
        dialog_path.write_bytes(base64.b64decode(dialog_screenshot.image or ''))
        if dialog_path.stat().st_size < 10_000:
            raise RuntimeError(
                f'Ubuntu session multi-window screenshot is unexpectedly small: {dialog_path.stat().st_size}'
            )
        _record(
            step_results,
            StepResult(
                'gui_screenshot_multi_window',
                True,
                {'path': str(dialog_path), 'bytes': dialog_path.stat().st_size},
            ),
        )

        close_dialogs = await runtime.gui_close_app('xmessage')
        _record(step_results, _gui_step('gui_close_app_xmessage_dialogs', close_dialogs))

        close_xterm = await runtime.gui_close_app('NexusUbuntuSessionXterm')
        _record(step_results, _gui_step('gui_close_app_xterm', close_xterm))

        libreoffice_command = (
            'SAL_USE_VCLPLUGIN=gtk3 HOME=/tmp/nexus-libreoffice-smoke '
            'libreoffice --writer --norestore --nofirststartwizard --nolockcheck'
        )
        launch_libreoffice = await runtime.gui_launch_app(libreoffice_command)
        _record(step_results, _gui_step('gui_launch_app_libreoffice_writer', launch_libreoffice))

        libreoffice_window = await _wait_for_command(
            runtime,
            "DISPLAY=:99 wmctrl -l -G -p | grep 'LibreOffice Writer' || ps -ef | grep '[s]office.bin.*--writer'",
            timeout_seconds=60.0,
        )
        _record(step_results, _command_step('libreoffice_writer_window_present', libreoffice_window))

        libreoffice_visible = await _wait_for_command(
            runtime,
            "DISPLAY=:99 /nix/runtime/nix-env/bin/xdotool search --onlyvisible --name 'LibreOffice Writer'",
            timeout_seconds=60.0,
        )
        _record(step_results, _command_step('libreoffice_writer_window_visible', libreoffice_visible))

        libreoffice_normal = await _wait_for_command(
            runtime,
            "for id in $(DISPLAY=:99 wmctrl -l | awk '/LibreOffice Writer/{print $1}'); do "
            'DISPLAY=:99 xprop -id "$id" WM_STATE; '
            "done | grep 'window state: Normal'",
            timeout_seconds=60.0,
        )
        _record(step_results, _command_step('libreoffice_writer_window_normal', libreoffice_normal))

        libreoffice_screenshot = await runtime.gui_screenshot()
        libreoffice_path = output_dir / 'ubuntu-session-libreoffice-writer.png'
        libreoffice_path.write_bytes(base64.b64decode(libreoffice_screenshot.image or ''))
        if libreoffice_path.stat().st_size < 10_000:
            raise RuntimeError(
                f'Ubuntu session LibreOffice screenshot is unexpectedly small: {libreoffice_path.stat().st_size}'
            )
        _record(
            step_results,
            StepResult(
                'gui_screenshot_libreoffice_writer',
                True,
                {'path': str(libreoffice_path), 'bytes': libreoffice_path.stat().st_size},
            ),
        )

        close_libreoffice = await runtime.run_command(
            'for pid in $(pgrep -f \'[s]office.bin|[o]osplash\'); do kill -TERM "$pid"; done'
        )
        _record(step_results, _command_step('run_command_close_libreoffice_writer', close_libreoffice))

        return {
            'success': True,
            'runtime_name': DEFAULT_RUNTIME_NAME,
            'image': image,
            'steps': [asdict(step) for step in step_results],
        }
    finally:
        if runtime is not None:
            await provider.release(DEFAULT_RUNTIME_NAME)
        await provider.release_all()
        await provider.cleanup_runtime_container()


async def _run_smoke() -> int:
    """Run the smoke test and print its structured result."""
    result = await run_ubuntu_session_smoke()
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get('success') is True else 1


def main() -> int:
    """CLI entrypoint."""
    return asyncio.run(_run_smoke())


if __name__ == '__main__':
    raise SystemExit(main())
