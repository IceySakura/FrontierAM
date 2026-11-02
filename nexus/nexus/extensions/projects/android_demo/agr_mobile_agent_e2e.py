"""AGR mobile agent-assisted E2E task for the Android demo project."""

import logging

from typing import Any

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, require_context, task_registry
from nexus.extensions.projects.android_demo.agr_mobile_shared import (
    APP_CONFIGS,
    build_run_dir,
    ensure_apk_local,
    save_base64_png,
    write_json,
    write_text,
)
from nexus.runtime.providers.agr_mobile.client import AGRMobileRuntime


DEFAULT_AGENT_SYSTEM_PROMPT = """You are an Android GUI verification agent.
Your only job is to judge the current Android screen state after the task has already launched App Store (应用宝).
Do not try to recover the app state. Do not navigate. Do not relaunch the app.
Base your conclusion primarily on the screenshot you capture at the beginning.
"""

DEFAULT_AGENT_USER_PROMPT = """The task has already finished the fixed preparation steps for you:
- connected to the sandbox
- uploaded and installed App Store (应用宝)
- granted permissions
- launched App Store

Now you must only verify whether App Store is visibly open.

Required workflow:
1. Call `android/screenshot` first.
2. Optionally call `android/device_info` once if you need extra confirmation.
3. Do not call any other tool.
4. Then stop and answer whether App Store is successfully open on screen.

Judgement rules:
- If the screenshot clearly shows App Store main UI, answer success.
- If the screenshot shows the welcome page, onboarding page, privacy agreement page, or any page containing text such as `欢迎使用应用宝`, that already counts as App Store successfully opened. In that case answer success.
- If the screenshot shows an agreement dialog with buttons such as `同意` or `不同意`, but the page background/header/body is clearly App Store, that still counts as success because the app itself has already opened.
- If the screenshot only shows a pure splash/loading/blank transitional page and no App Store content is visible yet, answer that App Store is not fully open yet.
- If the screenshot shows home screen or another app, answer failure.

Decision priority:
- Visible App Store content beats activity name.
- Do not rely only on `SplashActivity` from device info if the screenshot already shows App Store welcome or agreement content.

Final answer format:
- `visible: yes|no`
- one short sentence describing the strongest visual evidence
"""

DEFAULT_AGENT_TOOLS = [
    'android/screenshot',
    'android/device_info',
]


logger = logging.getLogger(__name__)


class AGRMobileAgentE2ETaskInput(TaskRunInput, WithAgentRunner):
    output_dir: str | None = None
    apk_dir: str | None = None
    agent_user_prompt: str | None = None
    agent_system_prompt: str | None = None
    agent_max_iterations: int = 3
    agent_inference_timeout: float = 60.0
    agent_observation_timeout: float = 30.0


class AGRMobileAgentE2ETaskOutput(TaskRunOutput):
    connected_sandbox_id: str | None = None
    device_info: dict | None = None
    device_info_path: str | None = None
    apk_local_path: str | None = None
    apk_downloaded: bool = False
    upload_result: dict | None = None
    install_result: dict | None = None
    grant_result: dict | None = None
    launch_result: dict | None = None
    pre_agent_screenshot_path: str | None = None
    agent_output: dict | None = None
    agent_final_answer: str | None = None
    screenshot_path: str | None = None
    post_reset_screenshot_path: str | None = None
    reset_result: dict | None = None
    logcat_path: str | None = None


def _extract_final_answer(agent_output: Any) -> str | None:
    final_answer = getattr(agent_output, 'final_answer', None)
    if isinstance(final_answer, str) and final_answer.strip():
        return final_answer.strip()
    return None


@task_registry.register('agr_mobile_agent_e2e')
async def agr_mobile_agent_e2e_task(task_input: AGRMobileAgentE2ETaskInput) -> AGRMobileAgentE2ETaskOutput:
    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    assert isinstance(runtime, AGRMobileRuntime)
    app_config = APP_CONFIGS['yyb']

    connected_sandbox_id = getattr(runtime, 'sandbox_id', None)
    run_dir = build_run_dir(task_input.output_dir, ctx.run_id)

    logger.info('AGR mobile agent E2E started: run_id=%s sandbox_id=%s', ctx.run_id, connected_sandbox_id)
    apk_local_path, apk_downloaded = await ensure_apk_local(task_input.apk_dir, app_config)
    logger.info('AGR mobile agent E2E using APK: %s (downloaded=%s)', apk_local_path, apk_downloaded)

    device_info = await runtime.gui_get_device_info()
    device_info_payload = device_info.model_dump()
    device_info_path = None
    if run_dir is not None:
        write_json(
            run_dir / 'sandbox_info.json',
            {
                'run_id': str(ctx.run_id or 'unknown_run'),
                'connected_sandbox_id': connected_sandbox_id,
            },
        )
        device_info_path = write_json(run_dir / 'device_info.json', device_info_payload)

    logger.info('AGR mobile agent E2E preparing App Store in sandbox %s', connected_sandbox_id)
    upload_result = await runtime.upload_apk(str(apk_local_path), str(app_config['remote_path']))
    install_result = await runtime.install_apk(str(app_config['package']), str(app_config['remote_path']))
    grant_result = await runtime.grant_permissions(str(app_config['package']), list(app_config['permissions']))
    launch_result = await runtime.launch_app_with_fallback(str(app_config['package']), str(app_config['activity']))

    pre_agent_screenshot = await runtime.gui_screenshot()
    pre_agent_screenshot_path = None

    if run_dir is not None:
        write_json(run_dir / 'upload_result.json', upload_result)
        write_json(run_dir / 'install_result.json', install_result)
        write_json(run_dir / 'grant_result.json', grant_result)
        write_json(run_dir / 'launch_result.json', launch_result)
        if pre_agent_screenshot.success and pre_agent_screenshot.image:
            pre_agent_screenshot_path = save_base64_png(
                pre_agent_screenshot.image, run_dir / 'app_store_before_agent.png'
            )

    ctx.tool_state['agr_mobile_prepared_app'] = {
        'name': app_config['name'],
        'package': app_config['package'],
        'activity': app_config['activity'],
    }

    logger.info('AGR mobile agent E2E handing off screen verification to agent for run_id=%s', ctx.run_id)
    agent_output = await task_input.run_agent(
        system_prompt=task_input.agent_system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT,
        user_prompt=task_input.agent_user_prompt or DEFAULT_AGENT_USER_PROMPT,
        tools=list(DEFAULT_AGENT_TOOLS),
        max_iterations=task_input.agent_max_iterations,
        parallel_tool_calls=False,
        invalid_tool_args_retries=1,
        tool_call_error_retries=1,
        tool_output_truncation=4000,
        inference_time_timeout=task_input.agent_inference_timeout,
        observation_time_timeout=task_input.agent_observation_timeout,
    )
    agent_output_payload = agent_output.model_dump() if hasattr(agent_output, 'model_dump') else None
    agent_final_answer = _extract_final_answer(agent_output)

    screenshot = await runtime.gui_screenshot()
    reset_result = await runtime.gui_reset_apps()
    post_reset_screenshot = await runtime.gui_screenshot()
    logcat_text = await runtime.dump_logcat()
    logger.info(
        'AGR mobile agent E2E finished: run_id=%s visible=%s stopped_packages=%s',
        ctx.run_id,
        agent_final_answer,
        reset_result.stopped_packages,
    )

    screenshot_path = None
    post_reset_screenshot_path = None
    logcat_path = None
    if run_dir is not None:
        if screenshot.success and screenshot.image:
            screenshot_path = save_base64_png(screenshot.image, run_dir / 'app_store_after_agent.png')
        write_json(run_dir / 'reset_result.json', reset_result.model_dump())
        if post_reset_screenshot.success and post_reset_screenshot.image:
            post_reset_screenshot_path = save_base64_png(post_reset_screenshot.image, run_dir / 'after_reset.png')
        logcat_path = write_text(run_dir / 'logcat.txt', logcat_text)
        if agent_output_payload is not None:
            write_json(run_dir / 'agent_output.json', agent_output_payload)
        if agent_final_answer:
            write_text(run_dir / 'agent_final_answer.txt', agent_final_answer)
        write_json(
            run_dir / 'task_summary.json',
            {
                'connected_sandbox_id': connected_sandbox_id,
                'apk_downloaded': apk_downloaded,
                'apk_local_path': str(apk_local_path),
                'pre_agent_screenshot_path': pre_agent_screenshot_path,
                'agent_final_answer': agent_final_answer,
                'device_info': device_info_payload,
                'launch_result': launch_result,
                'reset_result': reset_result.model_dump(),
                'post_reset_screenshot_path': post_reset_screenshot_path,
            },
        )

    return AGRMobileAgentE2ETaskOutput(
        connected_sandbox_id=connected_sandbox_id,
        device_info=device_info_payload,
        device_info_path=device_info_path,
        apk_local_path=str(apk_local_path),
        apk_downloaded=apk_downloaded,
        upload_result=upload_result,
        install_result=install_result,
        grant_result=grant_result,
        launch_result=launch_result,
        pre_agent_screenshot_path=pre_agent_screenshot_path,
        agent_output=agent_output_payload,
        agent_final_answer=agent_final_answer,
        screenshot_path=screenshot_path,
        post_reset_screenshot_path=post_reset_screenshot_path,
        reset_result=reset_result.model_dump(),
        logcat_path=logcat_path,
    )
