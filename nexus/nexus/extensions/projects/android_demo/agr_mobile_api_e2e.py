"""AGR mobile API-only E2E task for the Android demo project."""

import logging

from nexus.core import TaskRunInput, TaskRunOutput, require_context, task_registry
from nexus.extensions.projects.android_demo.agr_mobile_shared import (
    APP_CONFIGS,
    build_run_dir,
    ensure_apk_local,
    save_base64_png,
    write_json,
    write_text,
)
from nexus.runtime.providers.agr_mobile.client import AGRMobileRuntime


logger = logging.getLogger(__name__)


class AGRMobileApiE2ETaskInput(TaskRunInput):
    """Input for AGR mobile API-only E2E validation."""

    output_dir: str | None = None
    """Optional directory used to persist artifacts for this run."""

    apk_dir: str | None = None
    """Optional local cache directory for APK files."""


class AGRMobileApiE2ETaskOutput(TaskRunOutput):
    """Output for AGR mobile API-only E2E validation."""

    connected_sandbox_id: str | None = None
    device_info: dict | None = None
    device_info_path: str | None = None
    apk_local_path: str | None = None
    apk_downloaded: bool = False
    upload_result: dict | None = None
    install_result: dict | None = None
    grant_result: dict | None = None
    launch_result: dict | None = None
    screenshot_path: str | None = None
    reset_result: dict | None = None
    logcat_path: str | None = None


@task_registry.register('agr_mobile_api_e2e')
async def agr_mobile_api_e2e_task(task_input: AGRMobileApiE2ETaskInput) -> AGRMobileApiE2ETaskOutput:
    """Run API-only AGR mobile validation without any model dependency."""

    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    assert isinstance(runtime, AGRMobileRuntime)
    app_config = APP_CONFIGS['yyb']

    connected_sandbox_id = getattr(runtime, 'sandbox_id', None)
    run_dir = build_run_dir(task_input.output_dir, ctx.run_id)

    logger.info('AGR mobile API E2E started: run_id=%s sandbox_id=%s', ctx.run_id, connected_sandbox_id)
    apk_local_path, apk_downloaded = await ensure_apk_local(task_input.apk_dir, app_config)
    logger.info('AGR mobile API E2E using APK: %s (downloaded=%s)', apk_local_path, apk_downloaded)

    device_info = await runtime.gui_get_device_info()
    device_info_path = None
    if run_dir is not None:
        write_json(
            run_dir / 'sandbox_info.json',
            {
                'run_id': str(ctx.run_id or 'unknown_run'),
                'connected_sandbox_id': connected_sandbox_id,
            },
        )
        device_info_path = write_json(run_dir / 'device_info.json', device_info.model_dump())

    logger.info('AGR mobile API E2E preparing App Store in sandbox %s', connected_sandbox_id)
    upload_result = await runtime.upload_apk(str(apk_local_path), str(app_config['remote_path']))
    install_result = await runtime.install_apk(str(app_config['package']), str(app_config['remote_path']))
    grant_result = await runtime.grant_permissions(str(app_config['package']), list(app_config['permissions']))
    launch_result = await runtime.launch_app_with_fallback(str(app_config['package']), str(app_config['activity']))

    screenshot = await runtime.gui_screenshot()
    reset_result = await runtime.gui_reset_apps()
    logcat_text = await runtime.dump_logcat()

    screenshot_path = None
    logcat_path = None
    if run_dir is not None and screenshot.success and screenshot.image:
        screenshot_path = save_base64_png(screenshot.image, run_dir / 'app_store_after_launch.png')
    if run_dir is not None:
        logcat_path = write_text(run_dir / 'logcat.txt', logcat_text)
        write_json(run_dir / 'upload_result.json', upload_result)
        write_json(run_dir / 'install_result.json', install_result)
        write_json(run_dir / 'grant_result.json', grant_result)
        write_json(run_dir / 'launch_result.json', launch_result)
        write_json(run_dir / 'reset_result.json', reset_result.model_dump())

    logger.info(
        'AGR mobile API E2E finished: run_id=%s sandbox_id=%s stopped_packages=%s',
        ctx.run_id,
        connected_sandbox_id,
        reset_result.stopped_packages,
    )

    return AGRMobileApiE2ETaskOutput(
        connected_sandbox_id=connected_sandbox_id,
        device_info=device_info.model_dump(),
        device_info_path=device_info_path,
        apk_local_path=str(apk_local_path),
        apk_downloaded=apk_downloaded,
        upload_result=upload_result,
        install_result=install_result,
        grant_result=grant_result,
        launch_result=launch_result,
        screenshot_path=screenshot_path,
        reset_result=reset_result.model_dump(),
        logcat_path=logcat_path,
    )
