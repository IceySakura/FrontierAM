"""OSWorld task regressions."""

from __future__ import annotations

import json
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _no_external_artifact_service(monkeypatch):
    from unittest.mock import AsyncMock

    from nexus.extensions.tasks.os_world.data.artifacts import RunArtifacts

    monkeypatch.setattr(
        RunArtifacts,
        'publish',
        AsyncMock(return_value={'status': 'published', 'uri': 'artifact://test/run/', 'uris': []}),
    )


def _write_task(tmp_path: Path, evaluator: dict | None = None, *, proxy: bool = False) -> Path:
    task_path = tmp_path / 'chrome' / 'task-1.json'
    task_path.parent.mkdir()
    task_path.write_text(
        json.dumps({
            'id': 'task-1',
            'snapshot': 'chrome',
            'instruction': 'Enable do not track.',
            'config': [{'type': 'sleep', 'parameters': {'seconds': 0}}],
            'trajectory': 'trajectories/',
            'related_apps': ['chrome'],
            'evaluator': evaluator
            or {
                'func': 'exact_match',
                'result': {'type': 'vm_command_line', 'command': 'printf true', 'shell': True},
                'expected': {'type': 'rule', 'rules': {'expected': 'true'}},
            },
            'proxy': proxy,
            'fixed_ip': False,
            'possibility_of_env_change': 'low',
        }),
        encoding='utf-8',
    )
    return task_path


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['zero', 'infeasible', 'judge_error', 'unsupported', 'disabled', 'upload_failed'])
async def test_standard_stages_keep_scoring_and_nonfatal_upload(tmp_path, monkeypatch, outcome):
    from unittest.mock import AsyncMock

    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.data.artifacts import RunArtifacts
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    async def agent(**kwargs):
        return {'exit_status': 'infeasible' if outcome == 'infeasible' else 'finished'}

    evaluation = {'score': 1 if outcome == 'infeasible' else 0, 'supported': outcome != 'unsupported'}
    judge_error = RuntimeError('getter unavailable')
    evaluator = AsyncMock(return_value=evaluation, side_effect=judge_error if outcome == 'judge_error' else None)
    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.evaluate_os_world', evaluator)
    if outcome == 'upload_failed':
        monkeypatch.setattr(RunArtifacts, 'publish', AsyncMock(side_effect=ConnectionError('COS unavailable')))
    ctx = ExecutionContext(task_name='os_world')
    ctx._runtimes['main'] = SimpleNamespace(drain_trace=AsyncMock())
    async with ctx:
        inp = OSWorldTaskInput(
            task_file=str(_write_task(tmp_path)),
            run_agent=agent,
            run_setup=False,
            run_evaluator=outcome != 'disabled',
            include_initial_screenshot=False,
            skip_gui_start=True,
            setup_wait_time=0,
        )
        if outcome in {'judge_error', 'unsupported'}:
            with pytest.raises(RuntimeError) as raised:
                await run_os_world(inp)
            if outcome == 'judge_error':
                assert raised.value is judge_error
            partial = ctx.tool_state['osworld_partial_task_output']
            assert partial['status'] == 'judge_error'
            assert partial['agent_output']['exit_code'] == 0
            assert partial['judge_output']['exit_code'] == -1
            assert partial['error'] == partial['judge_output']['error']
            assert partial['score'] is None
        else:
            output = await run_os_world(inp)
            assert output.score == evaluation['score']
            assert output.status == ('judge_error' if outcome == 'disabled' else 'success')
            assert output.agent_output.exit_code == 0
            assert output.judge_output.exit_code == (-1 if outcome == 'disabled' else 0)
            if outcome == 'infeasible':
                assert evaluator.call_args.kwargs['last_action'] == 'FAIL'
                assert output.agent_output.conclusion == 'infeasible'
            if outcome == 'upload_failed':
                assert output.artifacts['upload_failed'] is True
    assert 'osworld_agent_output' not in ctx.tool_state


@pytest.mark.asyncio
async def test_os_world_rejects_failed_standard_agent_before_scoring(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    async def failed_agent(**kwargs):
        return SimpleNamespace(success=False, stderr='standard runner failed')

    evaluator = AsyncMock()
    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.evaluate_os_world', evaluator)
    ctx = ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = SimpleNamespace(drain_trace=AsyncMock())
    async with ctx:
        with pytest.raises(RuntimeError, match='OSWorld agent failed: standard runner failed'):
            await run_os_world(
                OSWorldTaskInput(
                    task_file=str(_write_task(tmp_path)),
                    run_agent=failed_agent,
                    run_setup=False,
                    run_evaluator=True,
                    include_initial_screenshot=False,
                    skip_gui_start=True,
                    setup_wait_time=0,
                )
            )
    evaluator.assert_not_awaited()
    assert ctx.tool_state['osworld_partial_task_output']['status'] == 'agent_error'
    assert ctx.tool_state['osworld_partial_task_output']['agent_output']['exit_code'] == -1
    assert ctx.tool_state['osworld_partial_task_output']['score'] is None
    assert ctx.tool_state['osworld_partial_task_output']['error'] == 'standard runner failed'


@pytest.mark.asyncio
@pytest.mark.parametrize('cc_return_code', [1, -1])
async def test_os_world_scores_explicit_turn_budget_outcome(tmp_path, monkeypatch, cc_return_code):
    from unittest.mock import AsyncMock

    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    async def budget_agent(**kwargs):
        return SimpleNamespace(success=True, exit_status='max_iterations', cc_return_code=cc_return_code)

    evaluator = AsyncMock(return_value={'score': 0.5})
    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.evaluate_os_world', evaluator)
    ctx = ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = SimpleNamespace(drain_trace=AsyncMock())
    async with ctx:
        result = await run_os_world(
            OSWorldTaskInput(
                task_file=str(_write_task(tmp_path)),
                run_agent=budget_agent,
                run_setup=False,
                run_evaluator=True,
                include_initial_screenshot=False,
                skip_gui_start=True,
                setup_wait_time=0,
            )
        )
    assert result.score == 0.5
    evaluator.assert_awaited_once()
    assert evaluator.call_args.kwargs['last_action'] is None


def test_os_world_loads_official_json_by_domain_and_id(tmp_path):
    from nexus.extensions.tasks.os_world.task import load_os_world_task

    task_path = _write_task(tmp_path)

    loaded = load_os_world_task(task_root=str(tmp_path), domain='chrome', task_id='task-1')

    assert loaded.path == task_path
    assert loaded.data['id'] == 'task-1'
    assert loaded.domain == 'chrome'


def test_os_world_system_prompt_is_fixed_for_screenshot_computer_13():
    from nexus.extensions.tasks.os_world.task import build_os_world_agent_tools, build_os_world_system_prompt

    prompt = build_os_world_system_prompt()

    assert 'Task id: task-1' not in prompt
    assert 'Domain: chrome' not in prompt
    assert 'You are asked to complete the following task:' not in prompt
    assert 'desktop/computer_13' in prompt
    assert 'desktop/fixed_marker' in prompt
    assert 'For each step, you will get an observation of an image' in prompt
    assert 'osworld/' not in prompt
    for action_type in [
        'MOVE_TO',
        'CLICK',
        'MOUSE_DOWN',
        'MOUSE_UP',
        'RIGHT_CLICK',
        'DOUBLE_CLICK',
        'DRAG_TO',
        'SCROLL',
        'TYPING',
        'PRESS',
        'KEY_DOWN',
        'KEY_UP',
        'HOTKEY',
    ]:
        assert action_type in prompt
    assert 'WAIT' in prompt
    assert 'DONE' in prompt
    assert 'FAIL' in prompt
    assert '- DRAG_TO: {"x": number, "y": number}' in prompt
    assert 'duration' not in prompt
    assert 'call desktop/fixed_marker with marker DONE' in prompt
    assert 'call desktop/fixed_marker with marker FAIL' in prompt
    assert 'RETURN ME THE action_type' not in prompt
    assert build_os_world_agent_tools('computer_13') == ['desktop/computer_13', 'desktop/fixed_marker']


def test_os_world_user_prompt_wraps_instruction_and_initial_screenshot():
    from nexus.extensions.tasks.os_world.task import build_os_world_user_prompt

    prompt = build_os_world_user_prompt(
        'Enable do not track.',
        SimpleNamespace(success=True, image='abc123', format='png'),
    )

    assert prompt == [
        {'type': 'text', 'text': 'You are asked to complete the following task: Enable do not track.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc123'}},
    ]


def test_os_world_system_prompt_can_describe_pyautogui_action_space():
    from nexus.extensions.tasks.os_world.task import build_os_world_agent_tools, build_os_world_system_prompt

    prompt = build_os_world_system_prompt(action_space='pyautogui')

    assert 'You are asked to complete the following task:' not in prompt
    assert 'desktop/pyautogui' in prompt
    assert 'Return one line or multiple lines of python code' in prompt
    assert 'DONOT use the `pyautogui.locateCenterOnScreen`' in prompt
    assert 'DONOT USE `pyautogui.screenshot()`' in prompt
    assert 'Only direct pyautogui.* calls, time.sleep(...), and read-only platform helpers are available' in prompt
    assert 'Do not import modules' in prompt
    assert 'Do not read or write files' in prompt
    assert 'Do not call shell commands, network APIs, or mock-service APIs' in prompt
    assert 'Do not directly modify application state outside the visible GUI' in prompt
    assert 'Do not type javascript: or data:text/html URLs' in prompt
    assert 'no variables or function can be shared from history' in prompt
    assert 'call desktop/fixed_marker with marker DONE' in prompt
    assert 'call desktop/fixed_marker with marker FAIL' in prompt
    assert build_os_world_agent_tools('pyautogui') == ['desktop/pyautogui', 'desktop/fixed_marker']


@pytest.mark.asyncio
async def test_os_world_does_not_need_project_specific_pyautogui_user_config(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    class FakeRuntime:
        def __init__(self):
            self.start_kwargs = None

        async def gui_start(self, **kwargs):
            self.start_kwargs = kwargs
            return SimpleNamespace(success=True)

        async def gui_screenshot(self):
            return SimpleNamespace(success=False, image=None, format='png', width=0, height=0, error='no screenshot')

    async def fake_run_setup_actions(runtime, actions, **kwargs):
        return []

    async def fake_evaluate_os_world(runtime, evaluator, *, unsupported_policy='skip', last_action=None, **kwargs):
        assert last_action == 'DONE'
        return {'score': 1.0, 'supported': True}

    tool_configs_seen: list[dict[str, object]] = []

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        from nexus.core import require_context

        tool_configs_seen.append(dict(require_context().tool_configs))
        assert 'desktop/computer_13' in system_prompt
        assert 'You are asked to complete the following task:' not in system_prompt
        assert user_prompt == 'You are asked to complete the following task: Enable do not track.'
        assert kwargs['post_iteration_hook'] == 'nexus.extensions.tools.desktop.observation::desktop_observation'
        assert kwargs['tools'] == ['desktop/computer_13', 'desktop/fixed_marker']
        return {'exit_status': 'finished'}

    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.run_setup_actions', fake_run_setup_actions)
    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.evaluate_os_world', fake_evaluate_os_world)

    task_file = _write_task(tmp_path)
    ctx = ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={})
    runtime = FakeRuntime()
    ctx._runtimes['main'] = runtime

    async with ctx:
        result = await run_os_world(
            OSWorldTaskInput(
                setup_wait_time=0, task_file=str(task_file), run_agent=fake_agent, run_setup=True, run_evaluator=True
            )
        )

    assert result.score == 1.0
    assert tool_configs_seen == [{}]
    assert runtime.start_kwargs['screen_width'] == 1920
    assert runtime.start_kwargs['screen_height'] == 1080


@pytest.mark.asyncio
async def test_os_world_can_skip_gui_start_for_prestarted_runtime(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    class FakeRuntime:
        async def gui_start(self, **kwargs):
            raise AssertionError('gui_start should be skipped')

        async def gui_screenshot(self):
            return SimpleNamespace(success=True, image='abc123', format='png', width=1280, height=800, error=None)

    setup_seen: dict[str, object] = {}

    async def fake_run_setup_actions(runtime, actions, **kwargs):
        setup_seen['actions'] = actions
        setup_seen['user_command_fn'] = kwargs['user_command_fn']
        setup_seen['client_password'] = kwargs['client_password']
        return [{'type': 'sleep', 'success': True}]

    async def fake_evaluate_os_world(runtime, evaluator, *, unsupported_policy='skip', last_action=None, **kwargs):
        return {'score': 1.0, 'supported': True}

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        assert isinstance(user_prompt, list)
        assert user_prompt[1] == {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc123'}}
        return {'exit_status': 'finished'}

    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.run_setup_actions', fake_run_setup_actions)
    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.evaluate_os_world', fake_evaluate_os_world)

    task_file = _write_task(tmp_path)
    ctx = ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = FakeRuntime()

    async with ctx:
        result = await run_os_world(
            OSWorldTaskInput(
                setup_wait_time=0,
                task_file=str(task_file),
                run_agent=fake_agent,
                skip_gui_start=True,
                command_as_user=False,
                client_password='task-input-password',
                run_setup=True,
                run_evaluator=True,
            )
        )

    assert result.completed is True
    assert result.score == 1.0
    assert setup_seen['actions'] == [{'type': 'sleep', 'parameters': {'seconds': 0}}]
    assert setup_seen['user_command_fn']('printf hello') == 'printf hello'
    assert setup_seen['client_password'] == 'task-input-password'
    assert ctx._runtimes['main'].osworld_command_as_user is False


@pytest.mark.asyncio
async def test_os_world_proxy_flag_is_ignored_without_enable_proxy(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    class FakeRuntime:
        async def gui_screenshot(self):
            return SimpleNamespace(success=False, image=None, format='png', width=0, height=0, error='no screenshot')

    setup_seen: dict[str, object] = {}

    async def fake_run_proxy_setup(runtime, **kwargs):
        raise AssertionError('proxy setup should not run when enable_proxy is false')

    async def fake_run_setup_actions(runtime, actions, **kwargs):
        setup_seen['use_proxy'] = kwargs['use_proxy']
        setup_seen['runtime_use_proxy'] = runtime.osworld_use_proxy
        return []

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        return {'exit_status': 'finished'}

    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.run_proxy_setup', fake_run_proxy_setup)
    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.run_setup_actions', fake_run_setup_actions)

    task_file = _write_task(tmp_path, proxy=True)
    ctx = ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = FakeRuntime()

    async with ctx:
        await run_os_world(
            OSWorldTaskInput(
                setup_wait_time=0,
                task_file=str(task_file),
                run_agent=fake_agent,
                run_evaluator=False,
                skip_gui_start=True,
                enable_proxy=False,
                force_proxy=True,
                proxy_config_file=str(tmp_path / 'proxy.json'),
            )
        )

    assert setup_seen == {'use_proxy': False, 'runtime_use_proxy': False}


@pytest.mark.asyncio
async def test_os_world_force_proxy_runs_for_non_proxy_task(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    class FakeRuntime:
        async def gui_screenshot(self):
            return SimpleNamespace(success=False, image=None, format='png', width=0, height=0, error='no screenshot')

    proxy_calls = 0

    async def fake_run_proxy_setup(runtime, **kwargs):
        nonlocal proxy_calls
        proxy_calls += 1
        return {'type': 'proxy', 'success': True}

    async def fake_run_setup_actions(runtime, actions, **kwargs):
        assert kwargs['use_proxy'] is True
        return []

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        return {'exit_status': 'finished'}

    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.run_proxy_setup', fake_run_proxy_setup)
    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.run_setup_actions', fake_run_setup_actions)

    task_file = _write_task(tmp_path, proxy=False)
    ctx = ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = FakeRuntime()

    async with ctx:
        await run_os_world(
            OSWorldTaskInput(
                setup_wait_time=0,
                task_file=str(task_file),
                run_agent=fake_agent,
                run_evaluator=False,
                skip_gui_start=True,
                enable_proxy=True,
                force_proxy=True,
                proxy_config_file=str(tmp_path / 'proxy.json'),
            )
        )

    assert proxy_calls == 1


@pytest.mark.asyncio
async def test_os_world_proxy_task_requires_proxy_config_file(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    class FakeRuntime:
        async def gui_screenshot(self):
            return SimpleNamespace(success=False, image=None, format='png', width=0, height=0, error='no screenshot')

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        return {'exit_status': 'finished'}

    task_file = _write_task(tmp_path, proxy=True)
    ctx = ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = FakeRuntime()

    async with ctx:
        with pytest.raises(ValueError, match='proxy_config_file'):
            await run_os_world(
                OSWorldTaskInput(
                    setup_wait_time=0,
                    task_file=str(task_file),
                    run_agent=fake_agent,
                    run_evaluator=False,
                    skip_gui_start=True,
                    enable_proxy=True,
                )
            )


@pytest.mark.asyncio
async def test_os_world_passes_initial_screenshot_and_last_action_to_evaluator(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    class FakeRuntime:
        async def gui_start(self, **kwargs):
            return SimpleNamespace(success=True)

        async def gui_screenshot(self):
            return SimpleNamespace(success=True, image='abc123', format='png', width=1280, height=800, error=None)

    async def fake_run_setup_actions(runtime, actions, **kwargs):
        return []

    seen: dict[str, object] = {}

    async def fake_evaluate_os_world(runtime, evaluator, *, unsupported_policy='skip', last_action=None, **kwargs):
        seen['last_action'] = last_action
        return {'score': 0.0, 'supported': True}

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        seen['system_prompt'] = system_prompt
        seen['user_prompt'] = user_prompt
        seen['post_iteration_hook'] = kwargs['post_iteration_hook']
        seen['tools'] = kwargs['tools']
        from nexus.core import require_context

        require_context().tool_state['os_world_last_action'] = 'FAIL'
        return {'exit_status': 'infeasible'}

    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.run_setup_actions', fake_run_setup_actions)
    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.evaluate_os_world', fake_evaluate_os_world)

    task_file = _write_task(tmp_path)
    ctx = ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = FakeRuntime()

    async with ctx:
        result = await run_os_world(OSWorldTaskInput(setup_wait_time=0, task_file=str(task_file), run_agent=fake_agent))

    assert result.completed is True
    assert seen['last_action'] == 'FAIL'
    assert 'desktop/computer_13' in seen['system_prompt']
    assert seen['tools'] == ['desktop/computer_13', 'desktop/fixed_marker']
    assert 'You are asked to complete the following task:' not in seen['system_prompt']
    user_prompt = seen['user_prompt']
    assert isinstance(user_prompt, list)
    assert user_prompt[0]['type'] == 'text'
    assert user_prompt[0]['text'] == 'You are asked to complete the following task: Enable do not track.'
    assert user_prompt[1] == {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc123'}}
    assert seen['post_iteration_hook'] == 'nexus.extensions.tools.desktop.observation::desktop_observation'


@pytest.mark.asyncio
async def test_desktop_observation_hook_appends_screenshot_message():
    from nexus.extensions.agents.main.agent import PostIterationHookContext
    from nexus.extensions.tools.desktop.observation import desktop_observation

    class FakeRuntime:
        async def gui_screenshot(self):
            return SimpleNamespace(success=True, image='after-action', format='png', width=1280, height=800, error=None)

    class FakeContext:
        tool_state = {'desktop_observation_runtime_name': 'main', 'desktop_observation_step_pause': 0}

        async def get_runtime(self, name='main'):
            assert name == 'main'
            return FakeRuntime()

    hook_context = PostIterationHookContext(
        iteration=1,
        max_iterations=10,
        history=(),
        tool_messages=({'role': 'tool', 'content': 'ok'},),
    )
    result = await desktop_observation(FakeContext(), hook_context)

    assert result is not None
    assert result.extra_messages == [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Current desktop screenshot.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,after-action'}},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_os_world_observation_hook_applies_official_step_pause(monkeypatch):
    from nexus.extensions.agents.main.agent import PostIterationHookContext
    from nexus.extensions.tools.desktop import observation as desktop_observation

    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    class FakeRuntime:
        async def gui_screenshot(self):
            return SimpleNamespace(success=True, image='after-wait', format='png', width=1920, height=1080, error=None)

    class FakeContext:
        tool_state = {'desktop_observation_runtime_name': 'main', 'desktop_observation_step_pause': 0.25}

        async def get_runtime(self, name='main'):
            assert name == 'main'
            return FakeRuntime()

    monkeypatch.setattr(desktop_observation.asyncio, 'sleep', fake_sleep)

    hook_context = PostIterationHookContext(
        iteration=1,
        max_iterations=10,
        history=(),
        tool_messages=({'role': 'tool', 'content': [{'type': 'text', 'text': 'WAIT'}]},),
    )
    result = await desktop_observation.desktop_observation(FakeContext(), hook_context)

    assert result is not None
    assert sleeps == [0.25, 0.25]


@pytest.mark.parametrize('marker', ['WAIT', 'DONE', 'FAIL'])
@pytest.mark.asyncio
async def test_os_world_fixed_marker_tool_returns_official_markers(marker):
    from nexus.extensions.tasks.os_world.execution.tools import FixedMarkerInput, fixed_marker_tool

    response = await fixed_marker_tool(FixedMarkerInput(marker=marker), context=None)

    assert response.output == [{'type': 'text', 'text': marker}]
    assert response.meta['marker'] == marker
    assert response.meta['termination_requested'] is (marker in {'DONE', 'FAIL'})
    assert 'termination_marker' not in response.meta
    assert 'done' not in response.meta
    assert 'fail' not in response.meta
    assert response.context_updates == {'os_world_last_action': marker, 'desktop_last_tool': 'desktop/fixed_marker'}


def test_os_world_fixed_marker_schema_describes_official_control_markers():
    from nexus.core import get_tool_schema
    from nexus.extensions.tasks.os_world.execution.tools import fixed_marker_tool

    schema = get_tool_schema(fixed_marker_tool)
    description = schema['function']['description']
    marker_description = schema['function']['parameters']['properties']['marker']['description']

    assert 'official OSWorld fixed markers' in description
    assert 'WAIT' in marker_description
    assert 'DONE' in marker_description
    assert 'FAIL' in marker_description
    assert 'task is complete' in marker_description
    assert 'cannot be completed' in marker_description


def test_os_world_manifest_registers_task_and_tools():
    manifest = Path('nexus/extensions/tasks/os_world/pyproject.toml').read_text(encoding='utf-8')

    assert '"os_world" = "nexus.extensions.tasks.os_world.task"' in manifest
    assert '"desktop/fixed_marker" = "nexus.extensions.tasks.os_world.execution.tools"' in manifest
    assert 'build-ubuntu-session-image' not in manifest
    assert 'osworld/' not in manifest
    assert '"desktop/bash"' not in manifest


def test_os_world_import_registers_required_agent_tools():
    code = """\
from nexus.core import Registry, task_registry, tool_registry
Registry.import_module('nexus.extensions.tasks.os_world')
task_registry.get_entry('os_world')
tool_registry.get_entry('desktop/computer_13')
tool_registry.get_entry('desktop/pyautogui')
tool_registry.get_entry('desktop/fixed_marker')
print('ok')
"""
    completed = subprocess.run([sys.executable, '-c', code], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert completed.stdout.strip() == 'ok'


def test_os_world_configs_resolve_with_environment(monkeypatch):
    from nexus.runner.single import RunInput

    monkeypatch.setenv('MODEL_SERVICE_API_APP_ID', 'app-test')
    monkeypatch.setenv('MODEL_SERVICE_API_APP_KEY', 'key-test')
    monkeypatch.setenv('MODEL_SERVICE_API_BASE_URL', 'http://model-service.test/openai/v1')
    monkeypatch.setenv('GONGFENG_TOKEN', 'gf-test')
    monkeypatch.delenv('OSWORLD_SMOKE_TASK_FILE', raising=False)
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('NEXUS_MODEL', raising=False)
    monkeypatch.setenv('NEXUS_MODEL', 'test-model')
    monkeypatch.setenv('NEXUS_BASE_URL', 'http://model.test/v1')
    monkeypatch.setenv('NEXUS_API_KEY', 'test-key')
    monkeypatch.setenv('AGS_SECRET_ID', 'test-id')
    monkeypatch.setenv('AGS_SECRET_KEY', 'test-secret')
    monkeypatch.setenv('AGS_OSWORLD_TOOL_ID', 'test-tool')
    monkeypatch.setenv('WEDATA_KEY', '{"id":1,"subject":"test","key":"test"}')

    smoke = RunInput.from_file('tests/integration/os_world/configs/os_world_smoke.yaml')
    pyautogui_smoke = RunInput.from_file('tests/integration/os_world/configs/os_world_smoke_pyautogui.yaml')
    docker = RunInput.from_file('tests/integration/os_world/configs/os_world_docker.yaml')
    gongfeng = RunInput.from_file('tests/integration/os_world/configs/os_world_gongfeng.yaml')

    assert smoke.task_name == 'os_world'
    assert smoke.task_config['task_file'] == 'tests/integration/os_world/fixtures/smoke_task.json'
    assert smoke.task_config['observation_type'] == 'screenshot'
    assert smoke.task_config['action_space'] == 'computer_13'
    assert smoke.agent_name == 'main_agent'
    assert 'tools' not in smoke.agent_config
    assert 'system_prompt' not in smoke.agent_config
    assert 'post_iteration_hook' not in smoke.agent_config
    assert pyautogui_smoke.task_name == 'os_world'
    assert pyautogui_smoke.task_config['task_file'] == 'tests/integration/os_world/fixtures/pyautogui_smoke_task.json'
    assert pyautogui_smoke.task_config['observation_type'] == 'screenshot'
    assert pyautogui_smoke.task_config['action_space'] == 'pyautogui'
    assert 'tools' not in pyautogui_smoke.agent_config
    assert 'system_prompt' not in pyautogui_smoke.agent_config
    assert 'post_iteration_hook' not in pyautogui_smoke.agent_config
    assert docker.runtime_provider_name == 'docker'
    assert gongfeng.runtime_provider_name == 'gongfeng'
    assert gongfeng.task_config['action_space'] == 'pyautogui'
    assert 'tools' not in gongfeng.agent_config


@pytest.mark.asyncio
async def test_os_world_proxy_setup_runs_only_for_proxy_task(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    class FakeRuntime:
        async def gui_screenshot(self):
            return SimpleNamespace(success=False, image=None, format='png', width=0, height=0, error='no screenshot')

    calls: dict[str, object] = {}

    async def fake_run_proxy_setup(runtime, **kwargs):
        calls['proxy_kwargs'] = kwargs
        return {'type': 'proxy', 'success': True, 'command': 'proxy setup {CLIENT_PASSWORD}'}

    async def fake_run_setup_actions(runtime, actions, **kwargs):
        calls['setup_use_proxy'] = kwargs['use_proxy']
        calls['runtime_use_proxy'] = runtime.osworld_use_proxy
        return [{'type': 'sleep', 'success': True}]

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        return {'exit_status': 'finished'}

    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.run_proxy_setup', fake_run_proxy_setup)
    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.run_setup_actions', fake_run_setup_actions)

    task_file = _write_task(tmp_path, proxy=True)
    ctx = ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = FakeRuntime()

    async with ctx:
        result = await run_os_world(
            OSWorldTaskInput(
                setup_wait_time=0,
                task_file=str(task_file),
                run_agent=fake_agent,
                run_evaluator=False,
                skip_gui_start=True,
                enable_proxy=True,
                force_proxy=False,
                proxy_config_file=str(tmp_path / 'proxy.json'),
            )
        )

    assert result.completed is True
    assert result.setup_results == []
    saved = json.loads((ctx.tool_state['osworld_output_publisher'].files / 'run-result.json').read_text())
    assert [item['type'] for item in saved['result']['setup_results']] == ['proxy', 'sleep']
    assert calls['proxy_kwargs']['proxy_config_file'] == str(tmp_path / 'proxy.json')
    assert calls['setup_use_proxy'] is True
    assert calls['runtime_use_proxy'] is True
    assert 'password' not in json.dumps(result.setup_results)


@pytest.mark.asyncio
async def test_os_world_proxy_setup_failure_reports_sanitized_details(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world

    class FakeRuntime:
        pass

    async def fake_run_proxy_setup(runtime, **kwargs):
        return {
            'type': 'proxy',
            'success': False,
            'command': 'proxy setup {CLIENT_PASSWORD}',
            'upload': {'success': True, 'status_code': 200},
            'return_code': 127,
            'stdout': 'install output',
            'stderr': 'tinyproxy: not found',
        }

    async def fake_run_setup_actions(runtime, actions, **kwargs):
        raise AssertionError('setup actions should not run after proxy setup failure')

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        return {'exit_status': 'finished'}

    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.run_proxy_setup', fake_run_proxy_setup)
    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.run_setup_actions', fake_run_setup_actions)

    task_file = _write_task(tmp_path, proxy=True)
    ctx = ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = FakeRuntime()

    async with ctx:
        with pytest.raises(RuntimeError) as exc_info:
            await run_os_world(
                OSWorldTaskInput(
                    setup_wait_time=0,
                    task_file=str(task_file),
                    run_agent=fake_agent,
                    run_evaluator=False,
                    skip_gui_start=True,
                    enable_proxy=True,
                    proxy_config_file=str(tmp_path / 'proxy.json'),
                    client_password='real-client-password',
                )
            )

    message = str(exc_info.value)
    assert 'upload_success=True' in message
    assert 'upload_status_code=200' in message
    assert 'command_return_code=127' in message
    assert 'tinyproxy: not found' in message
    assert 'real-client-password' not in message
