"""Nexus task adapter for official CUA-Gym public dataset bundles."""

import json
import re
import tempfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore, require_context, task_registry
from nexus.extensions.tasks.cua_gym.evaluator import evaluate_cua_gym
from nexus.extensions.tasks.cua_gym.materializer import load_env_replacements, materialize_task_bundle
from nexus.extensions.tasks.cua_gym.mock_hub import MockHubMode, MockHubResult, prepare_mock_hub
from nexus.extensions.tasks.cua_gym.setup import UnsupportedSetupActionError, run_setup_actions
from nexus.extensions.tasks.os_world.task import (
    build_os_world_agent_tools,
    build_os_world_system_prompt,
    build_os_world_user_prompt,
)
from nexus.extensions.tools.desktop.observation import (
    DESKTOP_OBSERVATION_RUNTIME_NAME_STATE,
    DESKTOP_OBSERVATION_STEP_PAUSE_STATE,
    LEGACY_OS_WORLD_RUNTIME_NAME_STATE,
    LEGACY_OS_WORLD_STEP_PAUSE_STATE,
)


ActionSpace = Literal['computer_13', 'pyautogui']
_X11_DISPLAY_PATTERN = re.compile(r'^:?(?P<number>\d+)(?:\.\d+)?$')
CUA_GYM_HOME_DIR = '/home/user'
CUA_GYM_ENVIRONMENT_PROMPT = (
    'Environment information:\n'
    f'- The VM user home directory is {CUA_GYM_HOME_DIR}.\n'
    f'- The Desktop directory is {CUA_GYM_HOME_DIR}/Desktop.'
)


@dataclass(frozen=True)
class LoadedCuaGymTask:
    path: Path
    bundle_dir: Path
    data: dict[str, Any]
    task_id: str
    app_type: str
    instruction: str


class CuaGymTaskInput(TaskRunInput, WithAgentRunner):
    """Input for a single CUA-Gym task bundle run."""

    task_file: str | None = None
    task_root: str = ''
    task_id: str = ''
    instruction: str | None = None
    runtime_name: str = 'main'
    run_setup: bool = True
    run_evaluator: bool = True
    unsupported_setup_policy: Literal['skip', 'fail'] = 'fail'
    require_done: bool = True
    fail_on_setup_error: bool = True
    desktop_session: str = 'gnome-session'
    gnome_session_name: str = 'ubuntu'
    screen_width: int = 1920
    screen_height: int = 1080
    step_pause: float = 2.0
    client_password: str = 'password'
    action_space: ActionSpace = 'pyautogui'
    guidance: str = ''
    mock_hub_mode: MockHubMode = 'local'
    mock_hub_root: str = '/opt/cua-gym-hub'
    mock_hub_start_port: int = 18000
    url_env_file: str | None = None
    strict_placeholders: bool = True
    setup_timeout: int = 600
    evaluator_timeout: int = 600
    display: str = ':99'


class CuaGymTaskOutput(TaskRunOutput, WithScore):
    completed: bool
    supported: bool = True
    task_id: str
    app_type: str
    instruction: str
    setup_results: list[dict[str, Any]] = Field(default_factory=list)
    evaluator_result: dict[str, Any] | None = None
    agent_output: Any | None = None
    mock_hub_result: dict[str, Any] | None = None
    error: str | None = None


def _task_path(task_file: str | None, task_root: str, task_id: str) -> Path:
    if task_file:
        path = Path(task_file)
        return path / 'task.json' if path.is_dir() else path
    if not task_root or not task_id:
        raise ValueError('CUA-Gym task requires task_file or task_root + task_id')
    return Path(task_root) / task_id / 'task.json'


def load_cua_gym_task(
    *,
    task_file: str | None = None,
    task_root: str = '',
    task_id: str = '',
) -> LoadedCuaGymTask:
    """Load one official CUA-Gym public dataset task bundle."""
    path = _task_path(task_file, task_root, task_id)
    data = json.loads(path.read_text(encoding='utf-8'))
    bundle_dir = path.parent
    resolved_task_id = str(data.get('task_id') or data.get('id') or task_id or bundle_dir.name)
    app_type = str(data.get('app_type') or data.get('application') or data.get('app') or '')
    instruction = str(data.get('instruction') or data.get('task') or '')
    return LoadedCuaGymTask(
        path=path,
        bundle_dir=bundle_dir,
        data=data,
        task_id=resolved_task_id,
        app_type=app_type,
        instruction=instruction,
    )


def build_cua_gym_agent_tools(action_space: ActionSpace = 'pyautogui') -> list[str]:
    """Build the agent tool list for the selected CUA-Gym action space."""
    try:
        return build_os_world_agent_tools(action_space)
    except ValueError as exc:
        raise ValueError(f'Unsupported CUA-Gym action_space: {action_space}') from exc


def build_cua_gym_system_prompt(*, action_space: ActionSpace = 'pyautogui', client_password: str = 'password') -> str:
    """Build the fixed CUA-Gym system prompt."""
    try:
        return build_os_world_system_prompt(action_space=action_space, client_password=client_password)
    except ValueError as exc:
        raise ValueError(f'Unsupported CUA-Gym action_space: {action_space}') from exc


def build_cua_gym_user_prompt(
    instruction: str,
    *,
    guidance: str = '',
    screenshot_response: Any | None = None,
) -> str | list[dict[str, Any]]:
    """Build the per-task CUA-Gym user prompt."""
    prompt = f'{instruction}\n\n{CUA_GYM_ENVIRONMENT_PROMPT}'
    if guidance.strip():
        prompt += f'\n\nAdditional guidance:\n{guidance.strip()}'
    return build_os_world_user_prompt(prompt, screenshot_response)


def _last_action_type(action: str | dict[str, Any] | None) -> str | None:
    if isinstance(action, dict):
        value = action.get('action_type') or action.get('marker')
        return str(value) if value is not None else None
    if action is None:
        return None
    return str(action)


def _setup_failed(results: list[dict[str, Any]]) -> bool:
    return any(item.get('success') is False for item in results)


def _mock_hub_dump(result: MockHubResult) -> dict[str, Any]:
    return {
        'replacements': result.replacements,
        'started_apps': result.started_apps,
        'uploaded_apps': result.uploaded_apps,
        'missing_placeholders': result.missing_placeholders,
    }


async def _prepare_bundle(
    runtime: Any,
    loaded: LoadedCuaGymTask,
    task_input: CuaGymTaskInput,
    materialized_root: Path,
    display: str,
) -> tuple[Path, MockHubResult]:
    env_replacements = load_env_replacements(task_input.url_env_file)
    mock_result = await prepare_mock_hub(
        runtime,
        bundle_dir=loaded.bundle_dir,
        mode=task_input.mock_hub_mode,
        hub_root=task_input.mock_hub_root,
        start_port=task_input.mock_hub_start_port,
        env_file=task_input.url_env_file,
        timeout=60,
        display=display,
    )
    replacements = {**env_replacements, **mock_result.replacements}
    if task_input.strict_placeholders and mock_result.missing_placeholders:
        missing = ', '.join(mock_result.missing_placeholders)
        raise ValueError(f'CUA-Gym mock URL placeholders were not resolved: {missing}')
    if not replacements:
        return loaded.bundle_dir, mock_result
    output_dir = materialized_root / loaded.task_id
    return materialize_task_bundle(loaded.bundle_dir, replacements, output_dir=output_dir), mock_result


def _effective_runtime_display(start_response: Any, fallback: str) -> str:
    display = getattr(start_response, 'display', None)
    return str(display) if display else fallback


def _x11_display_alias_command(display: str) -> str | None:
    match = _X11_DISPLAY_PATTERN.match(display.strip())
    if not match:
        return None
    display_number = match.group('number')
    if display_number == '0':
        return None
    source = f'/tmp/.X11-unix/X{display_number}'
    target = '/tmp/.X11-unix/X0'
    return (
        f'if [ -S {source} ]; then '
        f'if [ ! -e {target} ] || [ -L {target} ]; then ln -sfn X{display_number} {target}; fi; '
        'fi'
    )


async def _ensure_x11_display_alias(runtime: Any, display: str) -> Any | None:
    command = _x11_display_alias_command(display)
    if command is None:
        return None
    return await runtime.run_command(command=command, timeout=30)


@task_registry.register('cua_gym')
async def run_cua_gym(task_input: CuaGymTaskInput) -> CuaGymTaskOutput:
    """Run one CUA-Gym task bundle through a Nexus agent and reward.py evaluator."""
    ctx = require_context()
    loaded = load_cua_gym_task(
        task_file=task_input.task_file,
        task_root=task_input.task_root,
        task_id=task_input.task_id,
    )
    instruction = task_input.instruction or loaded.instruction
    runtime = await ctx.get_runtime(task_input.runtime_name)
    start_response = await runtime.gui_start(
        desktop_session=task_input.desktop_session,
        gnome_session_name=task_input.gnome_session_name,
        screen_width=task_input.screen_width,
        screen_height=task_input.screen_height,
    )
    if not getattr(start_response, 'success', False):
        return CuaGymTaskOutput(
            score=0.0,
            completed=False,
            task_id=loaded.task_id,
            app_type=loaded.app_type,
            instruction=instruction,
            error=getattr(start_response, 'error', None) or 'Failed to start CUA-Gym desktop session',
        )
    runtime_display = _effective_runtime_display(start_response, task_input.display)
    alias_response = await _ensure_x11_display_alias(runtime, runtime_display)
    if alias_response is not None and getattr(alias_response, 'return_code', None) != 0:
        return CuaGymTaskOutput(
            score=0.0,
            completed=False,
            task_id=loaded.task_id,
            app_type=loaded.app_type,
            instruction=instruction,
            error=getattr(alias_response, 'stderr', '') or 'Failed to prepare CUA-Gym X11 display alias',
        )

    ctx.tool_state['cua_gym_runtime_name'] = task_input.runtime_name
    ctx.tool_state[DESKTOP_OBSERVATION_RUNTIME_NAME_STATE] = task_input.runtime_name
    ctx.tool_state[DESKTOP_OBSERVATION_STEP_PAUSE_STATE] = task_input.step_pause
    ctx.tool_state.pop(LEGACY_OS_WORLD_RUNTIME_NAME_STATE, None)
    ctx.tool_state.pop(LEGACY_OS_WORLD_STEP_PAUSE_STATE, None)
    ctx.tool_state.pop('desktop_last_tool', None)
    ctx.tool_state.pop('os_world_last_action', None)

    with tempfile.TemporaryDirectory(prefix='nexus-cua-gym-') as tmp:
        setup_results: list[dict[str, Any]] = []
        try:
            bundle_dir, mock_result = await _prepare_bundle(runtime, loaded, task_input, Path(tmp), runtime_display)
        except Exception as exc:
            return CuaGymTaskOutput(
                score=0.0,
                completed=False,
                task_id=loaded.task_id,
                app_type=loaded.app_type,
                instruction=instruction,
                setup_results=setup_results,
                supported=False,
                error=str(exc),
            )

        if task_input.run_setup:
            try:
                setup_results.extend(
                    await run_setup_actions(
                        runtime,
                        loaded.data.get('config') or [],
                        bundle_dir=bundle_dir,
                        unsupported_policy=task_input.unsupported_setup_policy,
                        execute_timeout=task_input.setup_timeout,
                        display=runtime_display,
                    )
                )
            except UnsupportedSetupActionError as exc:
                return CuaGymTaskOutput(
                    score=0.0,
                    completed=False,
                    task_id=loaded.task_id,
                    app_type=loaded.app_type,
                    instruction=instruction,
                    supported=False,
                    mock_hub_result=_mock_hub_dump(mock_result),
                    error=str(exc),
                )
            except Exception as exc:
                return CuaGymTaskOutput(
                    score=0.0,
                    completed=False,
                    task_id=loaded.task_id,
                    app_type=loaded.app_type,
                    instruction=instruction,
                    mock_hub_result=_mock_hub_dump(mock_result),
                    error=f'CUA-Gym setup failed: {exc}',
                )
            if task_input.fail_on_setup_error and _setup_failed(setup_results):
                return CuaGymTaskOutput(
                    score=0.0,
                    completed=False,
                    task_id=loaded.task_id,
                    app_type=loaded.app_type,
                    instruction=instruction,
                    setup_results=setup_results,
                    supported=all(item.get('supported', True) is not False for item in setup_results),
                    mock_hub_result=_mock_hub_dump(mock_result),
                    error='CUA-Gym setup failed',
                )

        initial_screenshot = await runtime.gui_screenshot()
        agent_output = await task_input.run_agent(
            system_prompt=build_cua_gym_system_prompt(
                action_space=task_input.action_space,
                client_password=task_input.client_password,
            ),
            user_prompt=build_cua_gym_user_prompt(
                instruction,
                guidance=task_input.guidance,
                screenshot_response=initial_screenshot,
            ),
            tools=build_cua_gym_agent_tools(task_input.action_space),
            post_iteration_hook='nexus.extensions.tools.desktop.observation::desktop_observation',
        )
        last_action = _last_action_type(ctx.tool_state.get('os_world_last_action'))
        if task_input.require_done and last_action != 'DONE':
            return CuaGymTaskOutput(
                score=0.0,
                completed=False,
                task_id=loaded.task_id,
                app_type=loaded.app_type,
                instruction=instruction,
                setup_results=setup_results,
                agent_output=agent_output,
                mock_hub_result=_mock_hub_dump(mock_result),
                error='CUA-Gym task did not end with desktop/fixed_marker marker DONE',
            )

        evaluator_result: dict[str, Any] | None = None
        score = 0.0
        if task_input.run_evaluator:
            evaluator_result = await evaluate_cua_gym(
                runtime,
                loaded.data.get('evaluator') or {'type': 'python', 'path': 'reward.py'},
                bundle_dir=bundle_dir,
                task_id=loaded.task_id,
                timeout=task_input.evaluator_timeout,
                display=runtime_display,
            )
            score = float(evaluator_result.get('score', 0.0))
        completed = score >= 1.0 if task_input.run_evaluator else True
        error = None if completed else 'CUA-Gym evaluator score below success threshold'

        return CuaGymTaskOutput(
            score=score,
            completed=completed,
            task_id=loaded.task_id,
            app_type=loaded.app_type,
            instruction=instruction,
            setup_results=setup_results,
            evaluator_result=evaluator_result,
            agent_output=agent_output,
            mock_hub_result=_mock_hub_dump(mock_result),
            error=error,
        )


__all__ = [
    'CuaGymTaskInput',
    'CuaGymTaskOutput',
    'LoadedCuaGymTask',
    'build_cua_gym_agent_tools',
    'build_cua_gym_system_prompt',
    'build_cua_gym_user_prompt',
    'load_cua_gym_task',
    'run_cua_gym',
]
