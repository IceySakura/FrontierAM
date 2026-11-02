import base64
import inspect
import json
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_cua_bundle(
    tmp_path: Path,
    *,
    task_id: str = 'task-1',
    config: list[dict] | None = None,
    evaluator: dict | None = None,
) -> Path:
    bundle = tmp_path / task_id
    bundle.mkdir()
    (bundle / 'initial_setup.py').write_text(
        "from pathlib import Path\nPath('/home/user/cua_gym_smoke.txt').write_text('setup\\n')\n",
        encoding='utf-8',
    )
    (bundle / 'reward.py').write_text("print('REWARD: 1.0')\n", encoding='utf-8')
    (bundle / 'task.json').write_text(
        json.dumps({
            'task_id': task_id,
            'app_type': 'libreoffice_calc',
            'instruction': 'Append done to /home/user/cua_gym_smoke.txt.',
            'config': config
            if config is not None
            else [
                {
                    'type': 'download',
                    'parameters': {'files': [{'url': 'initial_setup.py', 'path': '/home/user/initial_setup.py'}]},
                },
                {
                    'type': 'execute',
                    'parameters': {'command': ['python3', '/home/user/initial_setup.py']},
                },
            ],
            'evaluator': evaluator if evaluator is not None else {'type': 'python', 'path': 'reward.py'},
        }),
        encoding='utf-8',
    )
    return bundle


def test_cua_gym_loads_public_dataset_bundle_by_root_and_id(tmp_path):
    from nexus.extensions.tasks.cua_gym.task import load_cua_gym_task

    bundle = _write_cua_bundle(tmp_path)

    loaded = load_cua_gym_task(task_root=str(tmp_path), task_id='task-1')

    assert loaded.path == bundle / 'task.json'
    assert loaded.bundle_dir == bundle
    assert loaded.task_id == 'task-1'
    assert loaded.app_type == 'libreoffice_calc'
    assert loaded.instruction == 'Append done to /home/user/cua_gym_smoke.txt.'


def test_cua_gym_prompt_defaults_to_pyautogui_and_uses_osworld_observations():
    from nexus.extensions.tasks.cua_gym.task import (
        CuaGymTaskInput,
        build_cua_gym_agent_tools,
        build_cua_gym_system_prompt,
        build_cua_gym_user_prompt,
    )

    async def run_agent(**kwargs):
        return kwargs

    task_input = CuaGymTaskInput(task_file='task.json', run_agent=run_agent)
    system_prompt = build_cua_gym_system_prompt()
    user_prompt = build_cua_gym_user_prompt('Open the spreadsheet and finish the edit.')

    assert task_input.action_space == 'pyautogui'
    assert task_input.screen_width == 1920
    assert task_input.screen_height == 1080
    assert 'desktop/screenshot' not in system_prompt
    assert 'desktop/pyautogui' in system_prompt
    assert 'desktop/fixed_marker' in system_prompt
    assert 'desktop/computer_13' not in system_prompt
    assert 'marker DONE' in system_prompt
    assert 'desktop/bash' not in system_prompt
    assert 'browser-address JavaScript' not in system_prompt
    assert (
        'Only direct pyautogui.* calls, time.sleep(...), and read-only platform helpers are available' in system_prompt
    )
    assert 'Do not import modules' in system_prompt
    assert 'Do not read or write files' in system_prompt
    assert 'Do not call shell commands, network APIs, or mock-service APIs' in system_prompt
    assert 'Do not directly modify application state outside the visible GUI' in system_prompt
    assert 'Do not type javascript: or data:text/html URLs' in system_prompt
    assert 'localhost:' not in system_prompt
    assert 'Do not use OS shell or launcher shortcuts' not in system_prompt
    assert 'browser toolbar and address bar are not the task UI' not in system_prompt
    assert 'DONOT USE `pyautogui.screenshot()`' in system_prompt
    assert 'DONOT use the `pyautogui.locateCenterOnScreen`' in system_prompt
    assert 'no variables or function can be shared from history' in system_prompt
    assert 'Do not call desktop/screenshot repeatedly without taking a GUI action' not in system_prompt
    assert 'Use the application, page, or window already opened by setup' not in system_prompt
    assert 'Never finish by replying in natural language only' in system_prompt
    assert 'For each step, you will get an observation of an image' in system_prompt
    assert 'pyautogui.position()' not in system_prompt
    assert user_prompt == (
        'You are asked to complete the following task: Open the spreadsheet and finish the edit.\n\n'
        'Environment information:\n'
        '- The VM user home directory is /home/user.\n'
        '- The Desktop directory is /home/user/Desktop.'
    )
    assert build_cua_gym_agent_tools('pyautogui') == [
        'desktop/pyautogui',
        'desktop/fixed_marker',
    ]


def test_cua_gym_computer_13_prompt_and_tools_are_selected_together():
    from nexus.extensions.tasks.cua_gym.task import build_cua_gym_agent_tools, build_cua_gym_system_prompt

    system_prompt = build_cua_gym_system_prompt(action_space='computer_13')

    assert 'desktop/computer_13' in system_prompt
    assert 'desktop/pyautogui' not in system_prompt
    assert 'right bottom corner of the screen is (1920, 1080)' in system_prompt
    assert '- DRAG_TO: {"x": number, "y": number}' in system_prompt
    assert 'duration' not in system_prompt
    assert build_cua_gym_agent_tools('computer_13') == [
        'desktop/computer_13',
        'desktop/fixed_marker',
    ]


def test_cua_gym_user_prompt_appends_optional_guidance_only_when_present():
    from nexus.extensions.tasks.cua_gym.task import build_cua_gym_user_prompt

    assert build_cua_gym_user_prompt('Fill the visible form.') == (
        'You are asked to complete the following task: Fill the visible form.\n\n'
        'Environment information:\n'
        '- The VM user home directory is /home/user.\n'
        '- The Desktop directory is /home/user/Desktop.'
    )
    assert build_cua_gym_user_prompt('Fill the visible form.', guidance='Prefer the visible Answer input.') == (
        'You are asked to complete the following task: Fill the visible form.\n\n'
        'Environment information:\n'
        '- The VM user home directory is /home/user.\n'
        '- The Desktop directory is /home/user/Desktop.\n\n'
        'Additional guidance:\nPrefer the visible Answer input.'
    )


def test_cua_gym_rejects_bash_as_action_space():
    from pydantic import ValidationError

    from nexus.extensions.tasks.cua_gym.task import CuaGymTaskInput, build_cua_gym_system_prompt

    with pytest.raises(ValidationError):
        CuaGymTaskInput(task_file='task.json', action_space='bash')
    with pytest.raises(ValueError, match='Unsupported CUA-Gym action_space'):
        build_cua_gym_system_prompt(action_space='bash')


def test_cua_gym_x11_display_alias_command_maps_zero_to_runtime_display():
    from nexus.extensions.tasks.cua_gym.task import _x11_display_alias_command

    command = _x11_display_alias_command(':99')

    assert command is not None
    assert '/tmp/.X11-unix/X99' in command
    assert 'ln -sfn X99 /tmp/.X11-unix/X0' in command
    assert _x11_display_alias_command(':0') is None
    assert _x11_display_alias_command('') is None


def test_cua_gym_manifest_registers_task_tools_and_image_command():
    manifest = Path('nexus/extensions/tasks/cua_gym/pyproject.toml').read_text(encoding='utf-8')

    assert '"cua_gym" = "nexus.extensions.tasks.cua_gym.task"' in manifest
    assert '"desktop/screenshot" = "nexus.extensions.tools.desktop.tools"' in manifest
    assert '"desktop/bash"' not in manifest
    assert '"desktop/computer_13" = "nexus.extensions.tools.desktop.tools"' in manifest
    assert '"desktop/pyautogui" = "nexus.extensions.tools.desktop.tools"' in manifest
    assert '"desktop/fixed_marker" = "nexus.extensions.tasks.os_world.execution.tools"' in manifest
    assert '"build-cua-gym-session-image" = "nexus.extensions.tasks.cua_gym.cua_gym_session_image"' in manifest


def test_cua_gym_import_registers_required_agent_tools():
    code = """\
from nexus.core import Registry, task_registry, tool_registry
Registry.import_module('nexus.extensions.tasks.cua_gym')
task_registry.get_entry('cua_gym')
tool_registry.get_entry('desktop/screenshot')
tool_registry.get_entry('desktop/computer_13')
tool_registry.get_entry('desktop/pyautogui')
tool_registry.get_entry('desktop/fixed_marker')
print('ok')
"""
    completed = subprocess.run([sys.executable, '-c', code], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert completed.stdout.strip() == 'ok'


def test_cua_gym_configs_resolve_with_environment(monkeypatch):
    from nexus.runner.single import RunInput

    monkeypatch.setenv('MODEL_SERVICE_API_APP_ID', 'app-test')
    monkeypatch.setenv('MODEL_SERVICE_API_APP_KEY', 'key-test')
    monkeypatch.setenv('MODEL_SERVICE_API_BASE_URL', 'http://model-service.test/openai/v1')
    monkeypatch.setenv('GONGFENG_TOKEN', 'gf-test')
    monkeypatch.delenv('NEXUS_MODEL', raising=False)

    docker = RunInput.from_file('tests/integration/cua_gym/configs/cua_gym_docker.yaml')
    gongfeng = RunInput.from_file('tests/integration/cua_gym/configs/cua_gym_gongfeng.yaml')
    example_gongfeng = RunInput.from_file('docs/examples/configs/cua_gym_gongfeng.yaml')

    assert docker.task_name == 'cua_gym'
    assert docker.runtime_provider_name == 'docker'
    assert docker.task_config['task_file'] == 'tests/integration/cua_gym/fixtures/gui_web_form_task/task.json'
    assert 'guidance' not in docker.task_config
    assert docker.task_config['action_space'] == 'pyautogui'
    assert 'bootstrap_dependencies' not in docker.task_config
    assert 'mock_hub_source' not in docker.task_config
    assert docker.runtime_configs['main']['image'] == (
        'mirrors.tencent.com/hunyuan_yanguan/nexus-cua-gym-session:dev-integration'
    )
    assert 'tools' not in docker.agent_config
    assert 'post_iteration_hook' not in docker.agent_config
    assert gongfeng.runtime_provider_name == 'gongfeng'
    assert gongfeng.task_config['action_space'] == 'pyautogui'
    assert 'guidance' not in gongfeng.task_config
    assert 'bootstrap_dependencies' not in gongfeng.task_config
    assert 'mock_hub_source' not in gongfeng.task_config
    assert gongfeng.runtime_configs['main']['image'] == (
        'mirrors.tencent.com/hunyuan_yanguan/nexus-cua-gym-session:dev-integration'
    )
    assert 'tools' not in gongfeng.agent_config
    assert example_gongfeng.runtime_provider_name == 'gongfeng'
    assert example_gongfeng.runtime_configs['main']['image'] == (
        'mirrors.tencent.com/hunyuan_yanguan/nexus-cua-gym-session:latest'
    )
    assert example_gongfeng.task_config['mock_hub_mode'] == 'local'
    assert example_gongfeng.task_config['action_space'] == 'pyautogui'
    assert 'guidance' not in example_gongfeng.task_config
    assert 'tools' not in example_gongfeng.agent_config


def test_cua_gym_materializer_replaces_text_placeholders_without_touching_source(tmp_path):
    from nexus.extensions.tasks.cua_gym.materializer import materialize_task_bundle

    source = _write_cua_bundle(tmp_path)
    (source / 'mock.txt').write_text('__CUA_GYM_INSTAGRAM_URL__\n', encoding='utf-8')
    (source / 'binary.bin').write_bytes(b'\xff\x00__CUA_GYM_INSTAGRAM_URL__')

    materialized = materialize_task_bundle(
        source,
        {'__CUA_GYM_INSTAGRAM_URL__': 'http://127.0.0.1:18000'},
        output_dir=tmp_path / 'materialized',
    )

    assert (materialized / 'mock.txt').read_text(encoding='utf-8') == 'http://127.0.0.1:18000\n'
    assert (materialized / 'binary.bin').read_bytes() == b'\xff\x00__CUA_GYM_INSTAGRAM_URL__'
    assert (source / 'mock.txt').read_text(encoding='utf-8') == '__CUA_GYM_INSTAGRAM_URL__\n'


def test_cua_gym_materializer_rejects_output_inside_source_bundle(tmp_path):
    from nexus.extensions.tasks.cua_gym.materializer import materialize_task_bundle

    source = _write_cua_bundle(tmp_path)

    with pytest.raises(ValueError, match='separate'):
        materialize_task_bundle(source, {}, output_dir=source / 'materialized')


@pytest.mark.asyncio
async def test_cua_gym_mock_hub_external_resolves_env_placeholders(tmp_path):
    from nexus.extensions.tasks.cua_gym.mock_hub import prepare_mock_hub

    bundle = _write_cua_bundle(tmp_path)
    (bundle / 'reward.py').write_text("BASE='__CUA_GYM_INSTAGRAM_URL__'\n", encoding='utf-8')
    env_file = tmp_path / 'urls.env'
    env_file.write_text('CUA_GYM_INSTAGRAM_URL=http://mock.example.test\n', encoding='utf-8')

    result = await prepare_mock_hub(object(), bundle_dir=bundle, mode='external', env_file=env_file)

    assert result.replacements == {'__CUA_GYM_INSTAGRAM_URL__': 'http://mock.example.test'}
    assert result.missing_placeholders == []
    assert result.started_apps == {}


@pytest.mark.asyncio
async def test_cua_gym_mock_hub_local_starts_each_required_app_once(tmp_path):
    from nexus.extensions.tasks.cua_gym.mock_hub import prepare_mock_hub

    bundle = _write_cua_bundle(tmp_path)
    (bundle / 'reward.py').write_text(
        "URL='__CUA_GYM_INSTAGRAM_URL__'\nHOST='__CUA_GYM_INSTAGRAM_HOST__'\n",
        encoding='utf-8',
    )
    commands = []

    class FakeRuntime:
        async def run_command(self, command, timeout=60):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

    result = await prepare_mock_hub(
        FakeRuntime(),
        bundle_dir=bundle,
        mode='local',
        hub_root='/opt/cua-gym-hub',
        start_port=18100,
    )

    assert result.started_apps == {'instagram_mock': 'http://127.0.0.1:18100'}
    assert result.replacements['__CUA_GYM_INSTAGRAM_URL__'] == 'http://127.0.0.1:18100'
    assert result.replacements['__CUA_GYM_INSTAGRAM_HOST__'] == '127.0.0.1:18100'
    assert sum('npm run preview' in command for command in commands) == 1
    assert all('npm install' not in command for command in commands)
    assert all('npm run build' not in command for command in commands)
    assert any('test -d node_modules' in command for command in commands)
    assert any('test -d dist' in command for command in commands)
    assert any('&& (nohup npm run preview' in command for command in commands)
    assert any('/go?sid=nexus-health' in command for command in commands)
    assert any('curl -fsS http://127.0.0.1:18100/ >/dev/null' in command for command in commands)


@pytest.mark.asyncio
async def test_cua_gym_mock_hub_has_no_host_source_upload_escape_hatch():
    from nexus.extensions.tasks.cua_gym.mock_hub import prepare_mock_hub

    assert 'host_hub_source' not in inspect.signature(prepare_mock_hub).parameters


@pytest.mark.asyncio
async def test_cua_gym_setup_download_uploads_relative_bundle_file_and_execute_runs_as_user(tmp_path):
    from nexus.extensions.tasks.cua_gym.setup import run_setup_actions

    bundle = _write_cua_bundle(tmp_path)
    uploads = []
    commands = []

    class FakeRuntime:
        async def upload_file(self, files, overwrite=True):
            uploads.extend(files)
            return SimpleNamespace(
                results=[SimpleNamespace(path=file.path, success=True, message='ok') for file in files]
            )

        async def run_command(self, command, timeout=60):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

        async def gui_get_screen_size(self):
            return SimpleNamespace(width=1280, height=800)

    results = await run_setup_actions(
        FakeRuntime(),
        json.loads((bundle / 'task.json').read_text(encoding='utf-8'))['config'],
        bundle_dir=bundle,
    )

    assert [item['type'] for item in results] == ['download', 'execute']
    assert uploads[0].path == '/home/user/initial_setup.py'
    assert uploads[0].encoding == 'base64'
    assert 'runuser -u user' in commands[-1]
    assert 'DISPLAY="${DISPLAY:-:0}"' in commands[-1]
    assert 'cd /home/user' in commands[-1]
    assert 'python3 /home/user/initial_setup.py' in commands[-1]


@pytest.mark.asyncio
async def test_cua_gym_setup_stops_before_execute_after_failed_download(tmp_path):
    from nexus.extensions.tasks.cua_gym.setup import run_setup_actions

    bundle = _write_cua_bundle(tmp_path)
    commands = []

    class FakeRuntime:
        async def upload_file(self, files, overwrite=True):
            return SimpleNamespace(results=[SimpleNamespace(path=files[0].path, success=False, message='denied')])

        async def run_command(self, command, timeout=60):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

    results = await run_setup_actions(
        FakeRuntime(),
        json.loads((bundle / 'task.json').read_text(encoding='utf-8'))['config'],
        bundle_dir=bundle,
    )

    assert [item['type'] for item in results] == ['download']
    assert results[0]['success'] is False
    assert all('python3 /home/user/initial_setup.py' not in command for command in commands)


@pytest.mark.asyncio
async def test_cua_gym_setup_supports_open_via_osworld_dispatch(tmp_path):
    """open/launch/sleep 等原子动作复用 OSWorld dispatch（DISPLAY=:0 注入）。"""
    from nexus.extensions.tasks.cua_gym.setup import run_setup_actions

    commands = []

    class FakeRuntime:
        async def run_command(self, command, timeout=60):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

        async def gui_get_screen_size(self):
            return SimpleNamespace(width=1280, height=800)

    results = await run_setup_actions(
        FakeRuntime(),
        [{'type': 'open', 'parameters': {'path': '/home/user/file.txt'}}],
        bundle_dir=_write_cua_bundle(tmp_path),
        unsupported_policy='fail',
    )

    assert [item['type'] for item in results] == ['open']
    assert results[0]['success'] is True
    assert 'xdg-open' in commands[-1]
    assert '/home/user/file.txt' in commands[-1]
    assert ':0' in commands[-1]


@pytest.mark.asyncio
async def test_cua_gym_setup_still_rejects_truly_unsupported_action(tmp_path):
    """不在支持集合内的动作仍按 unsupported_policy 处理。"""
    from nexus.extensions.tasks.cua_gym.setup import UnsupportedSetupActionError, run_setup_actions

    with pytest.raises(UnsupportedSetupActionError):
        await run_setup_actions(
            object(),
            [{'type': 'chrome_open_tabs', 'parameters': {'urls_to_open': ['http://x']}}],
            bundle_dir=_write_cua_bundle(tmp_path),
            unsupported_policy='fail',
        )


@pytest.mark.asyncio
async def test_cua_gym_evaluator_uploads_reward_and_parses_last_reward_line(tmp_path):
    from nexus.extensions.tasks.cua_gym.evaluator import evaluate_cua_gym

    bundle = _write_cua_bundle(tmp_path)
    uploads = []
    commands = []

    class FakeRuntime:
        async def upload_file(self, files, overwrite=True):
            uploads.extend(files)
            return SimpleNamespace(
                results=[SimpleNamespace(path=file.path, success=True, message='ok') for file in files]
            )

        async def run_command(self, command, timeout=60):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='noise\nREWARD: 0.25\nREWARD: 0.75\n', stderr='')

    result = await evaluate_cua_gym(
        FakeRuntime(),
        {'type': 'python', 'path': 'reward.py'},
        bundle_dir=bundle,
        task_id='task-1',
    )

    assert result['score'] == 0.75
    assert result['supported'] is True
    upload_paths = {file.path for file in uploads}
    assert '/tmp/cua-gym/task-1/reward.py' in upload_paths
    assert '/tmp/cua-gym/task-1/task.json' in upload_paths
    assert all(file.encoding == 'base64' for file in uploads)
    assert 'python3 /tmp/cua-gym/task-1/reward.py' in commands[0]
    assert 'runuser -u user' in commands[0]


@pytest.mark.asyncio
async def test_cua_gym_evaluator_scores_zero_when_reward_exits_nonzero(tmp_path):
    from nexus.extensions.tasks.cua_gym.evaluator import evaluate_cua_gym

    bundle = _write_cua_bundle(tmp_path)

    class FakeRuntime:
        async def upload_file(self, files, overwrite=True):
            return SimpleNamespace(
                results=[SimpleNamespace(path=file.path, success=True, message='ok') for file in files]
            )

        async def run_command(self, command, timeout=60):
            return SimpleNamespace(return_code=2, stdout='REWARD: 1.0\n', stderr='boom')

    result = await evaluate_cua_gym(
        FakeRuntime(),
        {'type': 'python', 'path': 'reward.py'},
        bundle_dir=bundle,
        task_id='task-1',
    )

    assert result['score'] == 0.0
    assert result['supported'] is True
    assert result['return_code'] == 2
    assert 'non-zero' in result['error']


@pytest.mark.asyncio
async def test_cua_gym_evaluator_uploads_bundle_support_files(tmp_path):
    from nexus.extensions.tasks.cua_gym.evaluator import evaluate_cua_gym

    bundle = _write_cua_bundle(tmp_path)
    (bundle / 'helpers').mkdir()
    (bundle / 'helpers' / 'score_helper.py').write_text('VALUE = 1.0\n', encoding='utf-8')
    (bundle / 'data.txt').write_text('expected\n', encoding='utf-8')
    (bundle / 'reward.py').write_text(
        'from helpers.score_helper import VALUE\n'
        'from pathlib import Path\n'
        "assert Path('data.txt').read_text().strip() == 'expected'\n"
        "print(f'REWARD: {VALUE}')\n",
        encoding='utf-8',
    )
    uploads = []

    class FakeRuntime:
        async def upload_file(self, files, overwrite=True):
            uploads.extend(files)
            return SimpleNamespace(
                results=[SimpleNamespace(path=file.path, success=True, message='ok') for file in files]
            )

        async def run_command(self, command, timeout=60):
            return SimpleNamespace(return_code=0, stdout='REWARD: 1.0\n', stderr='')

    result = await evaluate_cua_gym(
        FakeRuntime(),
        {'type': 'python', 'path': 'reward.py'},
        bundle_dir=bundle,
        task_id='task-1',
    )

    assert result['score'] == 1.0
    assert {file.path for file in uploads} >= {
        '/tmp/cua-gym/task-1/reward.py',
        '/tmp/cua-gym/task-1/helpers/score_helper.py',
        '/tmp/cua-gym/task-1/data.txt',
    }


@pytest.mark.asyncio
async def test_cua_gym_evaluator_uploads_packaged_reward_judge_when_reward_imports_it(tmp_path):
    from nexus.extensions.tasks.cua_gym.evaluator import PACKAGED_REWARD_JUDGE_PATH, evaluate_cua_gym

    bundle = _write_cua_bundle(tmp_path)
    (bundle / 'reward.py').write_text(
        "import sys\nsys.path.insert(0, '/tmp')\nfrom reward_judge import call_llm_judge\nprint('REWARD: 1.0')\n",
        encoding='utf-8',
    )
    uploads = []

    class FakeRuntime:
        async def upload_file(self, files, overwrite=True):
            uploads.extend(files)
            return SimpleNamespace(
                results=[SimpleNamespace(path=file.path, success=True, message='ok') for file in files]
            )

        async def run_command(self, command, timeout=60):
            return SimpleNamespace(return_code=0, stdout='REWARD: 1.0\n', stderr='')

    result = await evaluate_cua_gym(
        FakeRuntime(),
        {'type': 'python', 'path': 'reward.py'},
        bundle_dir=bundle,
        task_id='task-1',
    )

    assert result['score'] == 1.0
    upload_by_path = {file.path: file for file in uploads}
    assert '/tmp/cua-gym/task-1/reward.py' in upload_by_path
    assert '/tmp/reward_judge.py' in upload_by_path
    assert base64.b64decode(upload_by_path['/tmp/reward_judge.py'].content).decode(
        'utf-8'
    ) == PACKAGED_REWARD_JUDGE_PATH.read_text(encoding='utf-8')


@pytest.mark.asyncio
async def test_cua_gym_task_uses_osworld_observation_loop_and_requires_done(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.cua_gym.task import CuaGymTaskInput, run_cua_gym

    task_dir = _write_cua_bundle(tmp_path)
    seen: dict[str, object] = {}

    class FakeRuntime:
        async def gui_start(self, **kwargs):
            seen['gui_start'] = kwargs
            return SimpleNamespace(success=True, display=':99')

        async def run_command(self, command, timeout=60):
            seen.setdefault('run_commands', []).append(command)
            seen.setdefault('run_command_timeouts', []).append(timeout)
            return SimpleNamespace(return_code=0, stdout='', stderr='')

        async def gui_screenshot(self):
            seen['screenshot_calls'] = seen.get('screenshot_calls', 0) + 1
            return SimpleNamespace(success=True, image='initial-screen', format='png')

    async def fake_run_setup_actions(runtime, actions, **kwargs):
        seen['setup_actions'] = actions
        seen['setup_kwargs'] = kwargs
        return [{'type': 'download', 'success': True}, {'type': 'execute', 'success': True}]

    async def fake_evaluate_cua_gym(runtime, evaluator, **kwargs):
        seen['evaluator'] = evaluator
        seen['evaluator_kwargs'] = kwargs
        return {'score': 1.0, 'supported': True}

    async def fake_prepare_mock_hub(runtime, **kwargs):
        from nexus.extensions.tasks.cua_gym.mock_hub import MockHubResult

        seen['mock_hub_kwargs'] = kwargs
        return MockHubResult()

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        from nexus.core import require_context

        seen['system_prompt'] = system_prompt
        seen['user_prompt'] = user_prompt
        seen['agent_kwargs'] = kwargs
        seen['cua_gym_block_keys'] = sorted(
            key for key in require_context().tool_state if key.startswith('cua_gym_block_')
        )
        seen['last_desktop_tool'] = require_context().tool_state.get('desktop_last_tool')
        seen['desktop_observation_runtime_name'] = require_context().tool_state.get('desktop_observation_runtime_name')
        seen['desktop_observation_step_pause'] = require_context().tool_state.get('desktop_observation_step_pause')
        seen['legacy_os_world_runtime_name'] = require_context().tool_state.get('os_world_runtime_name')
        seen['legacy_os_world_step_pause'] = require_context().tool_state.get('os_world_step_pause')
        require_context().tool_state['os_world_last_action'] = 'DONE'
        return SimpleNamespace(exit_status='finished')

    monkeypatch.setattr('nexus.extensions.tasks.cua_gym.task.run_setup_actions', fake_run_setup_actions)
    monkeypatch.setattr('nexus.extensions.tasks.cua_gym.task.evaluate_cua_gym', fake_evaluate_cua_gym)
    monkeypatch.setattr('nexus.extensions.tasks.cua_gym.task.prepare_mock_hub', fake_prepare_mock_hub)

    ctx = ExecutionContext(task_name='cua_gym', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = FakeRuntime()

    async with ctx:
        result = await run_cua_gym(
            CuaGymTaskInput(task_file=str(task_dir / 'task.json'), run_agent=fake_agent, require_done=True)
        )

    assert result.completed is True
    assert result.score == 1.0
    assert (
        seen['agent_kwargs'].get('post_iteration_hook')
        == 'nexus.extensions.tools.desktop.observation::desktop_observation'
    )
    assert seen['agent_kwargs']['tools'] == ['desktop/pyautogui', 'desktop/fixed_marker']
    assert seen['cua_gym_block_keys'] == []
    assert seen['last_desktop_tool'] is None
    assert seen['desktop_observation_runtime_name'] == 'main'
    assert seen['desktop_observation_step_pause'] == 2.0
    assert seen['legacy_os_world_runtime_name'] is None
    assert seen['legacy_os_world_step_pause'] is None
    assert 'desktop/screenshot' not in seen['system_prompt']
    assert seen['user_prompt'][0] == {
        'type': 'text',
        'text': (
            'You are asked to complete the following task: Append done to /home/user/cua_gym_smoke.txt.\n\n'
            'Environment information:\n'
            '- The VM user home directory is /home/user.\n'
            '- The Desktop directory is /home/user/Desktop.'
        ),
    }
    assert seen['user_prompt'][1] == {
        'type': 'image_url',
        'image_url': {'url': 'data:image/png;base64,initial-screen'},
    }
    assert seen['screenshot_calls'] == 1
    assert seen['gui_start'] == {
        'desktop_session': 'gnome-session',
        'gnome_session_name': 'ubuntu',
        'screen_width': 1920,
        'screen_height': 1080,
    }
    assert seen['mock_hub_kwargs']['display'] == ':99'
    assert seen['setup_kwargs']['display'] == ':99'
    assert seen['evaluator_kwargs']['display'] == ':99'
    assert any('ln -sfn X99 /tmp/.X11-unix/X0' in command for command in seen['run_commands'])
    assert not any('DefaultBrowserSettingEnabled' in command for command in seen['run_commands'])
    assert not any('ShowTipOfTheDay' in command for command in seen['run_commands'])
    assert not any('WarnAlienFormat' in command for command in seen['run_commands'])
    assert seen['run_command_timeouts'] == [30]


@pytest.mark.asyncio
async def test_cua_gym_task_marks_reward_failure_incomplete(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.cua_gym.task import CuaGymTaskInput, run_cua_gym

    task_dir = _write_cua_bundle(tmp_path)

    class FakeRuntime:
        async def gui_start(self, **kwargs):
            return SimpleNamespace(success=True, display=':99')

        async def run_command(self, command, timeout=60):
            return SimpleNamespace(return_code=0, stdout='', stderr='')

        async def gui_screenshot(self):
            return SimpleNamespace(success=True, image='initial-screen', format='png')

    async def fake_run_setup_actions(runtime, actions, **kwargs):
        return []

    async def fake_evaluate_cua_gym(runtime, evaluator, **kwargs):
        return {'score': 0.0, 'supported': True}

    async def fake_prepare_mock_hub(runtime, **kwargs):
        from nexus.extensions.tasks.cua_gym.mock_hub import MockHubResult

        return MockHubResult()

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        from nexus.core import require_context

        require_context().tool_state['os_world_last_action'] = 'DONE'
        return SimpleNamespace(exit_status='finished')

    monkeypatch.setattr('nexus.extensions.tasks.cua_gym.task.run_setup_actions', fake_run_setup_actions)
    monkeypatch.setattr('nexus.extensions.tasks.cua_gym.task.evaluate_cua_gym', fake_evaluate_cua_gym)
    monkeypatch.setattr('nexus.extensions.tasks.cua_gym.task.prepare_mock_hub', fake_prepare_mock_hub)

    ctx = ExecutionContext(task_name='cua_gym', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = FakeRuntime()

    async with ctx:
        result = await run_cua_gym(
            CuaGymTaskInput(task_file=str(task_dir / 'task.json'), run_agent=fake_agent, require_done=True)
        )

    assert result.completed is False
    assert result.score == 0.0
    assert 'score below success threshold' in result.error


@pytest.mark.asyncio
async def test_cua_gym_task_returns_unsupported_for_setup_action_outside_subset(tmp_path):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.cua_gym.task import CuaGymTaskInput, run_cua_gym

    task_dir = _write_cua_bundle(
        tmp_path,
        config=[{'type': 'chrome_open_tabs', 'parameters': {'urls_to_open': ['http://x']}}],
    )

    class FakeRuntime:
        async def gui_start(self, **kwargs):
            return SimpleNamespace(success=True)

        async def run_command(self, command, timeout=60):
            return SimpleNamespace(return_code=0, stdout='', stderr='')

        async def gui_screenshot(self):
            return SimpleNamespace(success=True, image='initial-screen', format='png')

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        raise AssertionError('agent should not run for unsupported setup actions')

    ctx = ExecutionContext(task_name='cua_gym', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = FakeRuntime()

    async with ctx:
        result = await run_cua_gym(
            CuaGymTaskInput(
                task_file=str(task_dir / 'task.json'), run_agent=fake_agent, unsupported_setup_policy='fail'
            )
        )

    assert result.completed is False
    assert result.supported is False
    assert result.score == 0.0
    assert 'Unsupported CUA-Gym setup action: chrome_open_tabs' in result.error


@pytest.mark.asyncio
async def test_cua_gym_task_skips_reward_when_done_marker_is_missing(tmp_path, monkeypatch):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.cua_gym.task import CuaGymTaskInput, run_cua_gym

    task_dir = _write_cua_bundle(tmp_path)

    class FakeRuntime:
        async def gui_start(self, **kwargs):
            return SimpleNamespace(success=True)

        async def run_command(self, command, timeout=60):
            return SimpleNamespace(return_code=0, stdout='', stderr='')

        async def gui_screenshot(self):
            return SimpleNamespace(success=True, image='initial-screen', format='png')

    async def fake_run_setup_actions(runtime, actions, **kwargs):
        return []

    async def fake_evaluate_cua_gym(runtime, evaluator, **kwargs):
        raise AssertionError('reward should not run before DONE when require_done is enabled')

    async def fake_agent(system_prompt, user_prompt, **kwargs):
        return SimpleNamespace(exit_status='finished')

    monkeypatch.setattr('nexus.extensions.tasks.cua_gym.task.run_setup_actions', fake_run_setup_actions)
    monkeypatch.setattr('nexus.extensions.tasks.cua_gym.task.evaluate_cua_gym', fake_evaluate_cua_gym)

    ctx = ExecutionContext(task_name='cua_gym', runtime_configs={}, tool_configs={})
    ctx._runtimes['main'] = FakeRuntime()

    async with ctx:
        result = await run_cua_gym(
            CuaGymTaskInput(task_file=str(task_dir / 'task.json'), run_agent=fake_agent, require_done=True)
        )

    assert result.completed is False
    assert result.score == 0.0
    assert result.evaluator_result is None
    assert 'DONE' in result.error


def _cache_breakpoints(messages):
    return [
        i
        for i, m in enumerate(messages)
        if isinstance(m.get('content'), list)
        and any(isinstance(p, dict) and p.get('cache_control') for p in m['content'])
    ]


def test_cua_gym_image_window_cache_breakpoint_is_before_first_image():
    """滑动窗口缓存策略：断点必须打在首张图之前，否则旧图改写会让缓存前缀分叉。"""
    from nexus.extensions.tasks.cua_gym.cache import image_window

    def img(tag):
        return {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,IMG_{tag}'}}

    def txt(text):
        return {'type': 'text', 'text': text}

    messages = [
        {'role': 'system', 'content': 'SYS'},
        {'role': 'user', 'content': [txt('obs 1'), txt('[previous image omitted]')]},
        {'role': 'assistant', 'content': 'act 1'},
        {'role': 'user', 'content': [txt('obs 2'), img(2)]},
        {'role': 'user', 'content': [txt('obs 3'), img(3)]},
    ]
    original = [dict(m) for m in messages]

    tagged = image_window(messages)
    breakpoints = _cache_breakpoints(tagged)
    first_image = next(
        i
        for i, m in enumerate(tagged)
        if isinstance(m.get('content'), list) and any(p.get('type') in {'image_url', 'image'} for p in m['content'])
    )

    assert breakpoints, 'expected a cache breakpoint'
    assert all(bp < first_image for bp in breakpoints)
    # 不改原消息内容（图片原样保留）——保证 reward 不受影响
    assert messages == original
    assert tagged[3]['content'][1] == img(2)


def test_cua_gym_image_window_no_images_tags_last_message():
    from nexus.extensions.tasks.cua_gym.cache import image_window

    messages = [
        {'role': 'system', 'content': 'SYS'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'obs 1'}]},
        {'role': 'assistant', 'content': 'act 1'},
    ]

    assert _cache_breakpoints(image_window(messages)) == [2]
