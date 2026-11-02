"""Nexus task adapter for official OSWorld task JSON files."""

import asyncio
import json
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, SecretStr

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore, require_context, task_registry
from nexus.extensions.tasks.os_world.data import prefetch
from nexus.extensions.tasks.os_world.data.paths import (
    CACHE_DIR_ENV_VAR,
    resolve_evaluator_cache_dir,
    resolve_setup_cache_dir,
)
from nexus.extensions.tasks.os_world.evaluation.evaluator import evaluate_os_world
from nexus.extensions.tasks.os_world.execution.commands import osworld_user_command
from nexus.extensions.tasks.os_world.execution.guest_network import proxy_setup_failure_message
from nexus.extensions.tasks.os_world.execution.prompts import (
    screenshot_computer_13_tool_prompt,
    screenshot_pyautogui_tool_prompt,
)
from nexus.extensions.tasks.os_world.execution.runtime import get_osworld_runtime
from nexus.extensions.tasks.os_world.execution.setup import run_proxy_setup, run_setup_actions
from nexus.extensions.tasks.os_world.outputs import (
    OSWorldJudgeOutput,
    classify_exception,
    normalize_agent_output,
    run_task_agent,
    summarize_task_output,
    wrap_inner_timeout,
)
from nexus.extensions.tools.desktop.observation import (
    DESKTOP_OBSERVATION_RUNTIME_NAME_STATE,
    DESKTOP_OBSERVATION_STEP_PAUSE_STATE,
    LEGACY_OS_WORLD_RUNTIME_NAME_STATE,
    LEGACY_OS_WORLD_STEP_PAUSE_STATE,
    desktop_observation,
    desktop_screenshot_content_part,
)


ActionSpace = Literal['computer_13', 'pyautogui']


@dataclass(frozen=True)
class LoadedOSWorldTask:
    path: Path
    data: dict[str, Any]
    domain: str


class OSWorldTaskInput(TaskRunInput, WithAgentRunner):
    """Input for a single OSWorld task run without full benchmark orchestration."""

    task_file: str | None = None
    task_root: str = ''
    domain: str = ''
    task_id: str = ''
    instruction: str | None = None
    runtime_name: str = 'main'
    run_setup: bool = True
    run_evaluator: bool = True
    unsupported_evaluator_policy: Literal['skip', 'fail'] = 'skip'
    desktop_session: str = 'gnome-session'
    gnome_session_name: str = 'ubuntu'
    screen_width: int = 1920
    screen_height: int = 1080
    step_pause: float = 2.0
    client_password: str = 'password'
    skip_gui_start: bool = False
    command_as_user: bool = True
    setup_wait_time: float = 60.0
    enable_proxy: bool = False
    force_proxy: bool = False
    proxy_config_file: str | None = None
    observation_type: Literal['screenshot'] = 'screenshot'
    action_space: ActionSpace = 'computer_13'
    # False -> text-only prompt. The CC/codex bridge screenshots the desktop itself, so the
    # seed image would only push a base64 blob into the prompt.
    include_initial_screenshot: bool = True
    # Setup fixtures are downloaded once per task and reused across retries; a shared root
    # keeps them off the per-run scratch dir. Only consulted when cache_dir is unset.
    setup_cache_dir: str = '.cache/osworld'
    # Host-side cache root for downloaded setup assets and evaluator reference
    # files. Set this (rather than NEXUS_OSWORLD_CACHE_DIR) for platform runs:
    # run_single must be self-contained and may not read process env. Unset =>
    # env var, then the packaged ``os_world/cache`` directory.
    cache_dir: str | None = None
    # Wedata key JSON used to sign remote ``task_files`` reads. Needed on the
    # platform, where the host has no ``~/.nexus/auth/wedata/key``: the dataset
    # layer resolves the key from the process environment, so the value has to
    # arrive through the config and be published to os.environ before any
    # download. Inject it as ``"${oc.env:WEDATA_KEY}"``.
    #
    # Accepts a mapping as well as a string. The 七彩石 value is a JSON object,
    # and the platform resolves ``${oc.env:WEDATA_KEY}`` into a parsed dict
    # before it reaches the task -- a ``str``-only annotation made pydantic
    # reject every run before ``run_os_world`` was even entered.
    wedata_key: str | dict[str, Any] | None = None
    artifact_taihu_pat: SecretStr | None = Field(default=None, exclude=True, repr=False)
    """Per-run host credential; absent uses ARTIFACT_TAIHU_PAT or artifact login."""
    artifact_service_url: str | None = None
    """Per-run artifact endpoint; absent uses ARTIFACT_SERVICE_URL or the SDK default."""


class OSWorldTaskOutput(TaskRunOutput, WithScore):
    status: Literal['success', 'agent_error', 'judge_error'] = 'success'
    completed: bool
    task_id: str
    domain: str
    instruction: str
    setup_results: list[dict[str, Any]] = Field(default_factory=list)
    evaluator_result: dict[str, Any] | None = None
    agent_output: Any | None = None
    judge_output: OSWorldJudgeOutput | None = None
    error: str | None = None
    artifacts: dict[str, Any] | None = None


def _task_path(task_file: str | None, task_root: str, domain: str, task_id: str) -> Path:
    if task_file:
        return Path(task_file)
    if not task_root or not domain or not task_id:
        raise ValueError('OSWorld task requires task_file or task_root + domain + task_id')
    return Path(task_root) / domain / f'{task_id}.json'


def load_os_world_task(
    *,
    task_file: str | None = None,
    task_root: str = '',
    domain: str = '',
    task_id: str = '',
) -> LoadedOSWorldTask:
    """Load one official OSWorld task JSON file."""
    path = _task_path(task_file, task_root, domain, task_id)
    data = json.loads(path.read_text(encoding='utf-8'))
    resolved_domain = domain or str(data.get('snapshot') or path.parent.name)
    return LoadedOSWorldTask(path=path, data=data, domain=resolved_domain)


def build_os_world_agent_tools(action_space: ActionSpace = 'computer_13') -> list[str]:
    """Build the agent tool list for the selected OSWorld action space."""
    if action_space == 'computer_13':
        return ['desktop/computer_13', 'desktop/fixed_marker']
    if action_space == 'pyautogui':
        return ['desktop/pyautogui', 'desktop/fixed_marker']
    raise ValueError(f'Unsupported OSWorld action_space: {action_space}')


def build_os_world_system_prompt(
    *,
    observation_type: Literal['screenshot'] = 'screenshot',
    action_space: ActionSpace = 'computer_13',
    client_password: str = 'password',
) -> str:
    """Build the fixed OSWorld system prompt for one observation/action-space mode."""
    if observation_type != 'screenshot':
        raise ValueError(f'Unsupported OSWorld observation_type: {observation_type}')
    if action_space == 'computer_13':
        return screenshot_computer_13_tool_prompt(client_password=client_password)
    if action_space == 'pyautogui':
        return screenshot_pyautogui_tool_prompt(client_password=client_password)
    raise ValueError(f'Unsupported OSWorld action_space: {action_space}')


def build_os_world_user_prompt(
    instruction: str,
    screenshot_response: Any | None = None,
) -> str | list[dict[str, Any]]:
    """Build the per-task OSWorld user prompt with instruction and optional initial screenshot."""
    prompt = f'You are asked to complete the following task: {instruction}'
    if screenshot_response is None:
        return prompt
    return _build_user_prompt_with_observation(prompt, screenshot_response)


def build_os_world_prompt(
    loaded: LoadedOSWorldTask,
    *,
    instruction_override: str | None = None,
    observation_type: Literal['screenshot'] = 'screenshot',
    action_space: ActionSpace = 'computer_13',
) -> str:
    """Combine the OSWorld system prompt and task instruction."""
    instruction = instruction_override or str(loaded.data.get('instruction', ''))
    system_prompt = build_os_world_system_prompt(observation_type=observation_type, action_space=action_space)
    return f'{system_prompt}\nYou are asked to complete the following task: {instruction}'


def _should_use_proxy(task_data: dict[str, Any], *, enable_proxy: bool, force_proxy: bool) -> bool:
    if not enable_proxy:
        return False
    if force_proxy:
        return True
    return bool(task_data.get('proxy', False))


def _screenshot_content_part(response: Any) -> dict[str, Any] | None:
    return desktop_screenshot_content_part(response)


def _build_user_prompt_with_observation(prompt: str, screenshot_response: Any) -> str | list[dict[str, Any]]:
    screenshot_part = _screenshot_content_part(screenshot_response)
    if screenshot_part is None:
        return prompt
    return [{'type': 'text', 'text': prompt}, screenshot_part]


async def os_world_observation(ctx: Any, hook_context: Any) -> Any:
    """Delegate OSWorld observation processing to the desktop observation hook."""
    return await desktop_observation(ctx, hook_context)


# Re-export the shared dataset layout and prefetch helpers.
TASK_FILES_SETUP_PATTERN = prefetch.TASK_FILES_SETUP_PATTERN
TASK_FILES_EVALUATOR_PATTERN = prefetch.TASK_FILES_EVALUATOR_PATTERN
TASK_JSON_NAME = prefetch.TASK_JSON_NAME
_apply_wedata_key = prefetch.apply_wedata_key
_prefetch_task_files = prefetch.prefetch_task_files
_prefetch_task_json = prefetch.prefetch_task_json


def _cache_dir(task_input: OSWorldTaskInput) -> str | None:
    """The cache root to hand ``paths.resolve_*``, or None to let them resolve it.

    ``cache_dir`` is the platform's route, since ``run_single`` reads no environment.
    ``setup_cache_dir`` is the older field name for the same root and speaks only when
    neither ``cache_dir`` nor the env var is set, so it can shadow neither of them.
    """
    if task_input.cache_dir and task_input.cache_dir.strip():
        return task_input.cache_dir
    if os.environ.get(CACHE_DIR_ENV_VAR, '').strip():
        return None
    return task_input.setup_cache_dir or None


def _marker_from_agent_output(agent_output: Any) -> str | None:
    """Translate a bridge agent's exit_status into an os_world_last_action marker.

    The native CU agents write the marker themselves through ``desktop/fixed_marker``.
    A CC/codex bridge has no such action — it reports ``finished``/``infeasible`` on the
    agent output instead, and the evaluator needs the marker for the tasks whose correct
    answer is "this cannot be done". Matches the native OSWorld execution profile.
    """
    exit_status = getattr(agent_output, 'exit_status', None)
    if getattr(agent_output, 'conclusion', None) == 'infeasible':
        return 'FAIL'
    if exit_status == 'infeasible':
        return 'FAIL'
    if exit_status == 'finished':
        return 'DONE'
    return None


@task_registry.register('os_world')
async def run_os_world(task_input: OSWorldTaskInput) -> OSWorldTaskOutput:
    from nexus.extensions.tasks.os_world.data.artifacts import RunArtifacts

    ctx = require_context()
    ctx.tool_state.pop('osworld_partial_task_output', None)
    artifacts = RunArtifacts(
        taihu_pat=task_input.artifact_taihu_pat.get_secret_value() if task_input.artifact_taihu_pat else None,
        service_url=task_input.artifact_service_url,
    )
    from nexus.extensions.tasks.os_world.execution.release import install_release

    release_records = []
    ctx.tool_state['osworld_release_records'] = release_records
    artifacts.reference['release'] = release_records
    install_release(ctx.runtime_provider, task_input.runtime_name, release_records, artifacts.root)
    ctx.tool_state['osworld_artifact_directory'] = str(artifacts.files)
    ctx.tool_state['osworld_artifact_publisher'] = artifacts
    result = None
    primary = None
    ctx.tool_state['osworld_output_stage'] = 'prepare'
    ctx.tool_state.pop('osworld_agent_output', None)
    ctx.tool_state.pop('osworld_judge_output', None)
    try:
        result = await _run_os_world(task_input)
        return result  # noqa: RET504 - finally publishes and annotates this result.
    except BaseException as exc:
        primary = wrap_inner_timeout(
            exc,
            stage=ctx.tool_state.get('osworld_output_stage', 'prepare'),
            timeout_handle=getattr(ctx, 'timeout_handle', None),
        )
        if primary is exc:
            raise
        raise primary from exc
    finally:
        try:
            if result is None:
                # Failed profiles may raise after recording their full transcript.
                agent = ctx.tool_state.get('osworld_agent_output')
                if agent is not None:
                    (artifacts.files / 'agent-output.json').write_text(
                        agent.model_dump_json(indent=2), encoding='utf-8'
                    )
            artifacts.finish(task_id=task_input.task_id, run_id=ctx.run_id, result=result, error=primary)
            publication = asyncio.create_task(artifacts.publish())
            try:
                reference = await asyncio.shield(publication)
            except asyncio.CancelledError:
                await publication
                raise
            if result is not None:
                result.artifacts = reference
                result.artifacts['release'] = release_records
            elif primary is not None:
                primary.add_note(
                    f'OSWorld artifacts: status={reference["status"]}, '
                    f'uri={reference["uri"]}, local_directory={reference["local_directory"]}'
                )
        except asyncio.CancelledError:
            if primary is None:
                raise
            # Preserve the original exception if cancellation occurs while reporting it.
        except Exception as upload_error:
            # Publication/packaging is ancillary to evaluation. Never discard a
            # completed score because the artifact service is unavailable.
            import logging

            artifacts.reference.update(status='upload_failed', upload_failed=True)
            if result is not None:
                result.artifacts = artifacts.reference
            logging.getLogger(__name__).warning(
                'Artifact publication failed (%s); task outcome preserved; staging=%s',
                type(upload_error).__name__,
                artifacts.root,
            )
            if primary is not None:
                primary.add_note(f'OSWorld artifact publication failed; local_directory={artifacts.root}')
        finally:
            # Retain only the publisher until the explicit output hook has archived
            # processor results. No additional sandbox work or release ownership.
            ctx.tool_state['osworld_output_publisher'] = artifacts
            if result is None:
                stage = ctx.tool_state.get('osworld_output_stage', 'prepare')
                agent = ctx.tool_state.get('osworld_agent_output')
                judge = ctx.tool_state.get('osworld_judge_output')
                detail = (
                    f'{type(primary).__name__}: {primary}' if primary is not None else 'Task did not return a result'
                )
                if isinstance(primary, Exception) and (agent is None or (stage != 'judge' and agent.exit_code == 0)):
                    agent = classify_exception(primary, stage=stage)
                if stage == 'judge' and judge is None:
                    judge = OSWorldJudgeOutput(exit_status='judge_error', exit_code=-1, error=detail)
                partial_output = {
                    'status': 'judge_error' if stage == 'judge' else 'agent_error',
                    'completed': False,
                    'task_id': task_input.task_id,
                    'score': None,
                    'agent_output': agent.model_dump(mode='json') if agent is not None else None,
                    'judge_output': judge.model_dump(mode='json') if judge is not None else None,
                    'error': (
                        judge.error if stage == 'judge' and judge is not None else agent.error if agent else detail
                    ),
                    'artifacts': artifacts.reference,
                }
                ctx.tool_state['osworld_partial_task_output'] = summarize_task_output(partial_output)
            else:
                summarize_task_output(result)
            for key in ('osworld_output_stage', 'osworld_agent_output', 'osworld_judge_output'):
                ctx.tool_state.pop(key, None)
            ctx.tool_state.pop('osworld_artifact_directory', None)
            ctx.tool_state.pop('osworld_artifact_publisher', None)


async def _run_os_world(task_input: OSWorldTaskInput) -> OSWorldTaskOutput:
    """Run one OSWorld task JSON through a Nexus agent and supported evaluator subset."""
    ctx = require_context()

    # Must precede every task_files download: the dataset layer signs remote
    # reads with whatever credential the environment exposes.
    _apply_wedata_key(task_input.wedata_key)

    # Identify the task before loading so the caches (and the packaged-JSON
    # staging dir) resolve without depending on task_root existing.
    cache_dir = _cache_dir(task_input)
    pre_load_id = task_input.task_id or (Path(task_input.task_file).stem if task_input.task_file else '')
    setup_cache_dir = resolve_setup_cache_dir(cache_dir)
    evaluator_cache_dir = resolve_evaluator_cache_dir(pre_load_id, cache_dir) if pre_load_id else None

    # Attached task_files are authoritative: failures must not fall back to a
    # stale checkout. Only plain local runs without task_files use task_root.
    packaged_json: Path | None = None
    if task_input.task_file is None and pre_load_id:
        packaged_json = await _prefetch_task_json(evaluator_cache_dir or setup_cache_dir)

    loaded = load_os_world_task(
        task_file=str(packaged_json) if packaged_json is not None else task_input.task_file,
        task_root=task_input.task_root,
        domain=task_input.domain,
        task_id=task_input.task_id,
    )
    if packaged_json is not None and task_input.task_id and loaded.data.get('id') != task_input.task_id:
        raise ValueError('OSWorld task_config.task_id does not match the attached task_files/task.json id')
    # The id is only authoritative after the load, so the evaluator dir is recomputed.
    osworld_cache_task_id = str(loaded.data.get('id') or loaded.path.stem)
    evaluator_cache_dir = resolve_evaluator_cache_dir(osworld_cache_task_id, cache_dir)

    # Prime both caches from the dataset when one is attached, so neither setup
    # nor the evaluator has to reach the public internet. Downloads still run
    # normally for anything the dataset does not carry, which is what keeps a
    # plain local run (no task_files) working unchanged.
    await _prefetch_task_files(setup_cache_dir, evaluator_cache_dir, task_data=loaded.data)
    # OSWorld providers carry task-scoped hints beyond the base Runtime API.
    ctx.tool_state['osworld_output_stage'] = 'runtime'
    runtime = cast(Any, await get_osworld_runtime(ctx, task_input.runtime_name))
    ctx.tool_state['osworld_output_stage'] = 'prepare'
    runtime.osworld_command_as_user = task_input.command_as_user
    # The evaluator's local_file getter resolves relative fixtures against this; it runs far
    # from any task config, so the value travels on the runtime like the two hints below.
    runtime.osworld_examples_root = task_input.task_root or None
    use_proxy = _should_use_proxy(
        loaded.data,
        enable_proxy=task_input.enable_proxy,
        force_proxy=task_input.force_proxy,
    )
    runtime.osworld_use_proxy = use_proxy
    if not task_input.skip_gui_start:
        start_response = await runtime.gui_start(
            desktop_session=task_input.desktop_session,
            gnome_session_name=task_input.gnome_session_name,
            screen_width=task_input.screen_width,
            screen_height=task_input.screen_height,
        )
        if not getattr(start_response, 'success', False):
            raise RuntimeError(getattr(start_response, 'error', None) or 'Failed to start OSWorld desktop session')

    setup_results: list[dict[str, Any]] = []
    if task_input.run_setup:
        setup_user_command_fn = (lambda command: command) if not task_input.command_as_user else osworld_user_command
        if use_proxy:
            if not task_input.proxy_config_file:
                raise ValueError('OSWorld proxy is enabled for this task but proxy_config_file is not configured')
            proxy_result = await run_proxy_setup(
                runtime,
                proxy_config_file=task_input.proxy_config_file,
                user_command_fn=setup_user_command_fn,
                client_password=task_input.client_password,
            )
            setup_results.append(proxy_result)
            if proxy_result.get('success') is not True:
                raise RuntimeError(proxy_setup_failure_message(proxy_result))
        setup_results.extend(
            await run_setup_actions(
                runtime,
                loaded.data.get('config') or [],
                user_command_fn=setup_user_command_fn,
                cache_dir=setup_cache_dir,
                client_password=task_input.client_password,
                use_proxy=use_proxy,
            )
        )
        await asyncio.sleep(task_input.setup_wait_time)

    ctx.tool_state[DESKTOP_OBSERVATION_RUNTIME_NAME_STATE] = task_input.runtime_name
    ctx.tool_state[DESKTOP_OBSERVATION_STEP_PAUSE_STATE] = task_input.step_pause
    ctx.tool_state.pop(LEGACY_OS_WORLD_RUNTIME_NAME_STATE, None)
    ctx.tool_state.pop(LEGACY_OS_WORLD_STEP_PAUSE_STATE, None)
    ctx.tool_state.pop('os_world_last_action', None)

    _osworld_task_id = osworld_cache_task_id
    _osworld_domain = loaded.domain
    _related_apps_raw = loaded.data.get('related_apps')
    _osworld_related_apps = (
        tuple(value.strip() for value in _related_apps_raw if isinstance(value, str) and value.strip())
        if isinstance(_related_apps_raw, list)
        else ()
    )

    system_prompt = build_os_world_system_prompt(
        observation_type=task_input.observation_type,
        action_space=task_input.action_space,
        client_password=task_input.client_password,
    )
    instruction = task_input.instruction or str(loaded.data.get('instruction', ''))
    initial_screenshot = await runtime.gui_screenshot() if task_input.include_initial_screenshot else None
    user_prompt = build_os_world_user_prompt(instruction, initial_screenshot)
    trajectory_dir = ctx.tool_state['osworld_artifact_directory']
    ctx.tool_state['osworld_output_stage'] = 'agent'
    agent_output = await run_task_agent(
        task_input.run_agent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        post_iteration_hook='nexus.extensions.tools.desktop.observation::desktop_observation',
        tools=build_os_world_agent_tools(task_input.action_space),
        osworld_task_id=_osworld_task_id,
        osworld_domain=_osworld_domain,
        osworld_related_apps=_osworld_related_apps,
        trajectory_dir=trajectory_dir,
        task_context={
            'osworld_task_id': _osworld_task_id,
            'osworld_domain': _osworld_domain,
            'osworld_related_apps': _osworld_related_apps,
            'trajectory_dir': trajectory_dir,
        },
    )

    agent_output = ctx.tool_state.get('osworld_agent_output') or normalize_agent_output(
        agent_output, harness='cc' if ctx.agent_name == 'claude_code_native_bridge' else None
    )
    ctx.tool_state['osworld_agent_output'] = agent_output
    if agent_output.exit_code != 0:
        raise RuntimeError(f'OSWorld agent failed: {agent_output.error}')
    bridge_marker = _marker_from_agent_output(agent_output)
    if bridge_marker is not None:
        ctx.tool_state['os_world_last_action'] = bridge_marker
    evaluator_result: dict[str, Any] | None = None
    score = 0.0
    ctx.tool_state['osworld_output_stage'] = 'judge'
    if task_input.run_evaluator:
        evaluator_result = await evaluate_os_world(
            runtime,
            loaded.data.get('evaluator'),
            unsupported_policy=task_input.unsupported_evaluator_policy,
            last_action=ctx.tool_state.get('os_world_last_action'),
            client_password=task_input.client_password,
            cache_dir=evaluator_cache_dir,
        )
        score = float(evaluator_result.get('score', 0.0))
        if evaluator_result.get('supported') is False:
            raise RuntimeError(evaluator_result.get('error') or 'OSWorld evaluator is unsupported')
    judge_output = (
        OSWorldJudgeOutput(exit_status='finished', exit_code=0, result=evaluator_result)
        if task_input.run_evaluator
        else OSWorldJudgeOutput(
            exit_status='judge_error',
            exit_code=-1,
            error='OSWorld evaluation was disabled; no judge result is available',
        )
    )
    ctx.tool_state['osworld_judge_output'] = judge_output

    if trajectory_dir:
        _traj_task_dir = trajectory_dir
        if os.path.isdir(_traj_task_dir):
            from nexus.extensions.agents.cua.cua_support import _save_result

            _save_result(
                _traj_task_dir,
                {
                    'task_id': _osworld_task_id,
                    'domain': _osworld_domain,
                    'instruction': task_input.instruction or str(loaded.data.get('instruction', '')),
                    'score': score,
                    'evaluator_result': evaluator_result,
                    'exit_status': getattr(agent_output, 'exit_status', None),
                    'iterations': getattr(agent_output, 'iterations', None),
                },
            )

    return OSWorldTaskOutput(
        status='success' if judge_output.exit_code == 0 else 'judge_error',
        score=score,
        completed=True,
        task_id=str(loaded.data.get('id')),
        domain=loaded.domain,
        instruction=task_input.instruction or str(loaded.data.get('instruction', '')),
        setup_results=setup_results,
        evaluator_result=evaluator_result,
        agent_output=agent_output,
        judge_output=judge_output,
        error=judge_output.error,
    )


__all__ = [
    'LoadedOSWorldTask',
    'OSWorldTaskInput',
    'OSWorldTaskOutput',
    'build_os_world_agent_tools',
    'build_os_world_prompt',
    'build_os_world_system_prompt',
    'build_os_world_user_prompt',
    'load_os_world_task',
    'os_world_observation',
    'run_os_world',
]
