from types import SimpleNamespace

import pytest


class _FakeRuntime:
    """Minimal Android World runtime double with the lifecycle + GUI surface."""

    def __init__(self, *, screenshot=None, score=None, init_error=None):
        self.calls: list[str] = []
        self._screenshot = screenshot or SimpleNamespace(
            success=True, image='shot0', format='jpeg', width=1080, height=2400, error=None
        )
        self._score = score if score is not None else {'score': 1.0}
        self._init_error = init_error

    async def task_initialize(self, task_type, seed=0, task_idx=0):
        self.calls.append('task_initialize')
        if self._init_error is not None:
            raise RuntimeError(self._init_error)
        return f'Goal for {task_type}'

    async def gui_get_screen_size(self):
        self.calls.append('gui_get_screen_size')
        return SimpleNamespace(success=True, width=1080, height=2400)

    async def gui_screenshot(self):
        self.calls.append('gui_screenshot')
        return self._screenshot

    async def task_score(self):
        self.calls.append('task_score')
        return self._score

    async def task_tear_down(self):
        self.calls.append('task_tear_down')
        return True


def test_android_world_agent_tools_are_fixed():
    from nexus.extensions.tasks.android_world.task import build_android_world_agent_tools
    from nexus.extensions.tools.mobile.tools import MOBILE_TOOL_NAMES

    tools = build_android_world_agent_tools()
    assert tools == list(MOBILE_TOOL_NAMES)
    assert 'mobile/click' in tools
    assert 'mobile/swipe' in tools
    assert len(tools) == 11
    assert 'mobile/computer' not in tools


def test_android_world_native_system_prompt_describes_action_space():
    from nexus.extensions.tasks.android_world.task import build_android_world_system_prompt

    # 'native' is the only style: HyData-aligned mobile/* tools, normalized [0, 1] coords.
    prompt = build_android_world_system_prompt(screen_width=1080, screen_height=2400)

    assert '[0, 1]' in prompt
    assert '(1080, 2400)' not in prompt
    assert 'mobile/computer' not in prompt
    assert 'You are asked to complete the following task:' not in prompt
    for action in ['click', 'type', 'mobile.swipe', 'mobile.long_press', 'mobile.open_app', 'answer', 'terminate']:
        assert action in prompt


def test_android_world_user_prompt_wraps_instruction_and_initial_screenshot():
    from nexus.extensions.tasks.android_world.task import build_android_world_user_prompt

    prompt = build_android_world_user_prompt(
        'Add a contact.',
        SimpleNamespace(success=True, image='abc123', format='jpeg'),
    )

    assert prompt == [
        {'type': 'text', 'text': 'You are asked to complete the following task: Add a contact.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,abc123'}},
    ]


def test_android_world_user_prompt_without_screenshot_is_text_only():
    from nexus.extensions.tasks.android_world.task import build_android_world_user_prompt

    prompt = build_android_world_user_prompt('Add a contact.', None)

    assert prompt == 'You are asked to complete the following task: Add a contact.'


@pytest.mark.asyncio
async def test_android_world_runs_lifecycle_in_order_and_scores():
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.android_world.task import AndroidWorldTaskInput, run_android_world

    seen: dict[str, object] = {}

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        from nexus.core import require_context

        seen['system_prompt'] = system_prompt
        seen['user_prompt'] = user_prompt
        seen['tools'] = kwargs['tools']
        seen['post_iteration_hook'] = kwargs['post_iteration_hook']
        # Simulate the mobile/computer answer action writing the answer to state.
        require_context().tool_state['android_world_answer'] = 'John Doe'
        return {'ok': True}

    ctx = ExecutionContext(task_name='android_world', runtime_configs={}, tool_configs={})
    runtime = _FakeRuntime()
    ctx._runtimes['android'] = runtime

    async with ctx:
        result = await run_android_world(
            AndroidWorldTaskInput(task_type='ContactsAddContact', seed=42, run_agent=fake_agent)
        )

    # Lifecycle ordering: init -> screen size -> initial screenshot -> (agent) -> score -> tear down.
    assert runtime.calls == [
        'task_initialize',
        'gui_get_screen_size',
        'gui_screenshot',
        'task_score',
        'task_tear_down',
    ]
    assert result.completed is True
    assert result.score == 1.0
    assert result.task_goal == 'Goal for ContactsAddContact'
    assert result.agent_answer == 'John Doe'

    assert 'mobile/click' in seen['tools']
    assert 'mobile/swipe' in seen['tools']
    assert 'mobile/computer' not in seen['tools']
    assert seen['post_iteration_hook'] == 'nexus.extensions.tools.mobile.observation::mobile_observation'
    # Native prompt: HyData-aligned mobile tools, normalized [0, 1] coords, no raw resolution.
    assert '[0, 1]' in seen['system_prompt']
    assert 'mobile/computer' not in seen['system_prompt']
    user_prompt = seen['user_prompt']
    assert isinstance(user_prompt, list)
    assert user_prompt[0]['text'] == 'You are asked to complete the following task: Goal for ContactsAddContact'
    assert user_prompt[1] == {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,shot0'}}


@pytest.mark.asyncio
async def test_android_world_sets_observation_state():
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.android_world.task import AndroidWorldTaskInput, run_android_world

    captured: dict[str, object] = {}

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        from nexus.core import require_context

        state = require_context().tool_state
        captured['runtime_name'] = state.get('mobile_observation_runtime_name')
        captured['step_pause'] = state.get('mobile_observation_step_pause')
        captured['screen_width'] = state.get('mobile_observation_screen_width')
        captured['screen_height'] = state.get('mobile_observation_screen_height')
        return {'ok': True}

    ctx = ExecutionContext(task_name='android_world', runtime_configs={}, tool_configs={})
    ctx._runtimes['android'] = _FakeRuntime()

    async with ctx:
        await run_android_world(
            AndroidWorldTaskInput(task_type='ContactsAddContact', step_pause=1.5, run_agent=fake_agent)
        )

    assert captured['runtime_name'] == 'android'
    assert captured['step_pause'] == 1.5
    # The task publishes the real screen size so the bridge client can scale coordinates.
    assert captured['screen_width'] == 1080
    assert captured['screen_height'] == 2400


@pytest.mark.asyncio
async def test_android_world_returns_zero_score_when_initialization_fails():
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.android_world.task import AndroidWorldTaskInput, run_android_world

    agent_called = False

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        nonlocal agent_called
        agent_called = True
        return {'ok': True}

    ctx = ExecutionContext(task_name='android_world', runtime_configs={}, tool_configs={})
    runtime = _FakeRuntime(init_error='device offline')
    ctx._runtimes['android'] = runtime

    async with ctx:
        result = await run_android_world(AndroidWorldTaskInput(task_type='ContactsAddContact', run_agent=fake_agent))

    assert result.completed is False
    assert result.score == 0.0
    assert 'device offline' in result.task_goal
    assert result.error == 'device offline'
    assert agent_called is False
    # Only the failed initialize should have been attempted.
    assert runtime.calls == ['task_initialize']


@pytest.mark.asyncio
async def test_android_world_skips_scoring_when_run_evaluator_false():
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.android_world.task import AndroidWorldTaskInput, run_android_world

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        return {'ok': True}

    ctx = ExecutionContext(task_name='android_world', runtime_configs={}, tool_configs={})
    runtime = _FakeRuntime()
    ctx._runtimes['android'] = runtime

    async with ctx:
        result = await run_android_world(
            AndroidWorldTaskInput(task_type='ContactsAddContact', run_evaluator=False, run_agent=fake_agent)
        )

    assert 'task_score' not in runtime.calls
    assert 'task_tear_down' in runtime.calls
    assert result.score == 0.0
    assert result.completed is True
