"""AGR mobile identity task for the Android demo project."""

import logging

from nexus.core import require_context, task_registry
from nexus.extensions.projects.android_demo.agr_mobile_shared import build_run_dir, save_base64_png
from nexus.extensions.tasks.identity.task import IdentityTaskInput, IdentityTaskOutput
from nexus.runtime.providers.agr_mobile.client import AGRMobileRuntime


logger = logging.getLogger(__name__)


class AGRMobileIdentityTaskInput(IdentityTaskInput):
    """AGR mobile identity task input."""

    screenshot_dir: str | None = None
    """Optional directory used to persist screenshots for this run."""


class AGRMobileIdentityTaskOutput(IdentityTaskOutput):
    """AGR mobile identity task output."""

    device_info: dict | None = None
    screenshot_path: str | None = None
    reset_result: dict | None = None
    agent_output: dict | None = None


@task_registry.register('agr_mobile_identity')
async def agr_mobile_identity_task(task_input: AGRMobileIdentityTaskInput) -> AGRMobileIdentityTaskOutput:
    """Identity task with Android runtime teardown: runs the agent then captures state."""

    agent_result = await task_input.run_agent()

    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    assert isinstance(runtime, AGRMobileRuntime)

    device_info = await runtime.gui_get_device_info()
    screenshot = await runtime.gui_screenshot()
    reset_result = await runtime.gui_reset_apps()

    screenshot_path = None
    if task_input.screenshot_dir and screenshot.success and screenshot.image:
        run_dir = build_run_dir(task_input.screenshot_dir, ctx.run_id)
        if run_dir is not None:
            screenshot_path = save_base64_png(screenshot.image, run_dir / 'final_screenshot.png')

    agent_output = None
    if hasattr(agent_result, 'model_dump'):
        agent_output = agent_result.model_dump()

    return AGRMobileIdentityTaskOutput(
        device_info=device_info.model_dump() if hasattr(device_info, 'model_dump') else None,
        screenshot_path=screenshot_path,
        reset_result=reset_result.model_dump() if hasattr(reset_result, 'model_dump') else None,
        agent_output=agent_output,
    )
