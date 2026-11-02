"""Run a deterministic GUI runtime smoke test against the Docker provider."""

import asyncio
import json
import os
import sys

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RUNTIME_NAME = 'gui-smoke'
DEFAULT_BASE_IMAGE = 'ubuntu:24.04'
DEFAULT_OUTPUT_FILE = '/tmp/nexus_gui_runtime_smoke.txt'
DEFAULT_OUTPUT_TEXT = 'hello from nexus gui runtime smoke'


@dataclass(frozen=True)
class StepResult:
    """Outcome for a single smoke-test step."""

    name: str
    success: bool
    details: dict[str, Any]


def _base_image() -> str:
    """Resolve the base image used for the smoke container."""
    return os.getenv('NEXUS_GUI_BASE_IMAGE', os.getenv('NEXUS_GUI_RUNTIME_IMAGE', DEFAULT_BASE_IMAGE))


def _assert_success(name: str, response: Any, *, details: dict[str, Any] | None = None) -> StepResult:
    """Convert a runtime response into a validated step result."""
    payload = details or {}
    if hasattr(response, 'model_dump'):
        payload = {**payload, **response.model_dump()}
    success = bool(getattr(response, 'success', False))
    if not success:
        raise RuntimeError(f'{name} failed: {payload}')
    return StepResult(name=name, success=True, details=payload)


def _record(step_results: list[StepResult], step: StepResult) -> None:
    """Record and stream a compact JSON line for one smoke step."""
    step_results.append(step)
    print(json.dumps({'step': step.name, 'success': step.success}, default=str), flush=True)


async def run_local_gui_runtime_smoke() -> dict[str, Any]:
    """Execute the deterministic GUI smoke flow end to end."""
    from nexus.runtime.providers.docker import DockerRuntimeProvider
    from tests.integration.gui_agent.runtime_image import ensure_runtime_image

    await ensure_runtime_image()
    provider = DockerRuntimeProvider(
        runtime_image='current',
        force_rebuild=False,
        exposed_ports=[5900, 6080],
        auto_connect_ws=False,
    )
    runtime = None
    step_results: list[StepResult] = []
    output_file = os.getenv('NEXUS_GUI_OUTPUT_FILE', DEFAULT_OUTPUT_FILE)
    output_text = os.getenv('NEXUS_GUI_OUTPUT_TEXT', DEFAULT_OUTPUT_TEXT)

    try:
        runtime = await provider.acquire(
            DEFAULT_RUNTIME_NAME,
            image=_base_image(),
        )

        gui_start = await runtime.gui_start()
        _record(step_results, _assert_success('gui_start', gui_start))

        try:
            device_info = await runtime.gui_get_device_info()
            _record(step_results, _assert_success('gui_get_device_info', device_info))
        except Exception as exc:
            print(json.dumps({'step': 'gui_get_device_info', 'success': False, 'error': str(exc)}, default=str), flush=True)

        screen_size = await runtime.gui_get_screen_size()
        _record(step_results, _assert_success('gui_get_screen_size', screen_size))

        initial_screenshot = await runtime.gui_screenshot()
        _record(
            step_results,
            _assert_success(
                'gui_screenshot_initial',
                initial_screenshot,
                details={'image_bytes': len(initial_screenshot.image or '')},
            ),
        )

        xterm_command = 'xterm -geometry 100x28+120+80 -fa Monospace -fs 12'
        _record(step_results, _assert_success('gui_launch_app_xterm', await runtime.gui_launch_app(xterm_command)))

        await asyncio.sleep(1.0)
        _record(step_results, _assert_success('gui_click', await runtime.gui_click(260, 180)))
        _record(step_results, _assert_success('gui_double_click', await runtime.gui_double_click(260, 180)))
        _record(step_results, _assert_success('gui_long_click', await runtime.gui_long_click(300, 220, duration=200)))
        _record(step_results, _assert_success('gui_drag', await runtime.gui_drag(320, 260, 440, 260, duration=250)))

        type_command = f"printf %s {output_text!r} > {output_file}"
        _record(step_results, _assert_success('gui_type', await runtime.gui_type(type_command)))
        _record(step_results, _assert_success('gui_press_key_enter', await runtime.gui_press_key('Enter')))

        command_result = await runtime.run_command(f'cat {output_file}')
        command_details = command_result.model_dump()
        if command_result.return_code != 0:
            raise RuntimeError(f'run_command failed: {command_details}')
        if command_result.stdout.strip() != output_text:
            raise RuntimeError(
                f'Unexpected file content: expected {output_text!r}, got {command_result.stdout.strip()!r}'
            )
        _record(step_results, StepResult(name='run_command_verify_file', success=True, details=command_details))

        dialog_command = "xmessage -buttons Close:0 -geometry 320x120+430+220 'Nexus GUI dialog'"
        _record(
            step_results,
            _assert_success('gui_launch_app_xmessage', await runtime.gui_launch_app(dialog_command)),
        )
        await asyncio.sleep(0.75)

        post_dialog_screenshot = await runtime.gui_screenshot()
        _record(
            step_results,
            _assert_success(
                'gui_screenshot_dialog',
                post_dialog_screenshot,
                details={'image_bytes': len(post_dialog_screenshot.image or '')},
            ),
        )

        _record(step_results, _assert_success('gui_close_app_xmessage', await runtime.gui_close_app('xmessage')))
        _record(step_results, _assert_success('gui_close_app_xterm', await runtime.gui_close_app('xterm')))

        reset_apps = await runtime.gui_reset_apps()
        _record(step_results, _assert_success('gui_reset_apps', reset_apps))

        final_screenshot = await runtime.gui_screenshot()
        _record(
            step_results,
            _assert_success(
                'gui_screenshot_final',
                final_screenshot,
                details={'image_bytes': len(final_screenshot.image or '')},
            ),
        )

        return {
            'success': True,
            'runtime_name': DEFAULT_RUNTIME_NAME,
            'steps': [asdict(step) for step in step_results],
        }
    finally:
        if runtime is not None:
            await provider.release(DEFAULT_RUNTIME_NAME)
        await provider.release_all()
        await provider.cleanup_runtime_container()


async def _run_smoke() -> int:
    """Run the smoke test and print its structured result."""
    result = await run_local_gui_runtime_smoke()
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get('success') is True else 1


def main() -> int:
    """CLI entrypoint."""
    return asyncio.run(_run_smoke())


if __name__ == '__main__':
    raise SystemExit(main())
