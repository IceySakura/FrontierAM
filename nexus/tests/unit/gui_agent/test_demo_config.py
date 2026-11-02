"""Unit coverage for local GUI integration config helpers."""

import asyncio
import importlib.util
import os

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.runner.single import RunInput


@pytest.mark.parametrize('available', [False, True])
def test_desktop_preflight_requires_prebuilt_image(monkeypatch, available):
    from tests.integration.gui_agent.runtime_image import require_desktop_image

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0 if available else 1)

    monkeypatch.setattr('subprocess.run', run)
    if available:
        assert require_desktop_image() == 'nexus-ubuntu-session:latest'
    else:
        with pytest.raises(RuntimeError, match='Prebuilt desktop image'):
            require_desktop_image()
    assert calls == [['docker', 'image', 'inspect', 'nexus-ubuntu-session:latest']]


def test_resolve_openai_settings_requires_model_service_credentials():
    from tests.integration.gui_agent.demo_config import resolve_openai_settings

    with pytest.raises(RuntimeError, match='MODEL_SERVICE_API_APP_ID'):
        resolve_openai_settings({})


def test_resolve_openai_settings_builds_distillation_platform_key():
    from tests.integration.gui_agent.demo_config import resolve_openai_settings

    settings = resolve_openai_settings(
        {
            'MODEL_SERVICE_API_APP_ID': 'app-id',
            'MODEL_SERVICE_API_APP_KEY': 'app-key',
            'MODEL_SERVICE_API_BASE_URL': 'http://model-service.test/openai/v1',
        }
    )

    assert settings.base_url == 'http://model-service.test/openai/v1'
    assert settings.model == 'gpt-5.5'
    assert settings.api_key == 'app-id:app-key?provider=azure_openai&model=gpt-5.5&timeout=60'


def test_build_run_config_produces_gui_main_agent_flow():
    from tests.integration.gui_agent.demo_config import (
        DEFAULT_BASE_IMAGE,
        DEFAULT_MODEL,
        DEFAULT_RUNTIME_EXTRA_ARGS,
        build_run_config,
    )

    config = build_run_config(api_key='sk-test')
    prompt = config['agent_config']['user_prompt']

    assert config['task_name'] == 'identity'
    assert config['agent_name'] == 'main_agent'
    assert config['runtime_provider_name'] == 'docker'
    assert config['runtime_configs']['main']['image'] == DEFAULT_BASE_IMAGE
    assert config['runtime_configs']['main']['image'] == 'nexus-ubuntu-session:latest'
    assert config['runtime_configs']['main']['exposed_ports'] == [5900, 6080]
    assert config['runtime_configs']['main']['extra_args'] == DEFAULT_RUNTIME_EXTRA_ARGS
    assert config['openai_clients']['main']['config']['request_args']['model'] == DEFAULT_MODEL
    assert 'desktop/start' in config['agent_config']['tools']
    assert 'desktop/screenshot' in config['agent_config']['tools']
    assert 'desktop/launch_app' in config['agent_config']['tools']
    assert 'desktop/close_app' in config['agent_config']['tools']
    assert 'desktop/computer_13' in config['agent_config']['tools']
    assert 'desktop/bash' in config['agent_config']['tools']
    assert 'desktop/click' not in config['agent_config']['tools']
    assert 'desktop/type_text' not in config['agent_config']['tools']
    assert 'desktop/press_key' not in config['agent_config']['tools']
    assert config['agent_config']['parallel_tool_calls'] is False
    assert config['runtime_provider_config']['print_port_mappings'] is True
    assert config['agent_config']['max_iterations'] == 30
    assert '"desktop_session":"gnome-session","session_options":{"gnome_session_name":"ubuntu"}' in prompt
    assert 'NexusPrimary' in prompt
    assert 'NexusSecondary' in prompt
    assert 'Nexus GUI dialog A' in prompt
    assert 'Nexus GUI dialog B' in prompt
    assert '/tmp/nexus_gui_agent_secondary_demo.txt' in prompt


def test_runtime_extra_args_pass_environment_variables_as_docker_flags():
    from tests.integration.gui_agent.demo_config import DEFAULT_RUNTIME_EXTRA_ARGS

    env_args = ['GTK_A11Y=none', 'NEXUS_GUI_REQUIRE_SYSTEM_DBUS=1']

    for env_arg in env_args:
        idx = DEFAULT_RUNTIME_EXTRA_ARGS.index(env_arg)
        assert DEFAULT_RUNTIME_EXTRA_ARGS[idx - 1] == '-e'


def test_ubuntu_session_smoke_extra_args_pass_environment_variables_as_docker_flags():
    source = Path('tests/integration/gui_agent/run_local_ubuntu_session_runtime_smoke.py').read_text()

    assert (
        "'-e',\n                'GTK_A11Y=none',\n                '-e',\n                'NEXUS_GUI_REQUIRE_SYSTEM_DBUS=1'"
        in source
    )


def test_desktop_manifest_registers_start_tool():
    manifest = Path('nexus/extensions/tools/desktop/pyproject.toml').read_text()

    assert '"desktop/start" = "nexus.extensions.tools.desktop.tools"' in manifest


def test_desktop_start_tool_schema_does_not_expose_window_manager():
    from nexus.extensions.tools.desktop.tools import StartInput

    properties = StartInput.model_json_schema()['properties']

    assert 'desktop_session' in properties
    assert 'session_options' in properties
    assert 'gnome_session_name' not in properties
    assert 'window_manager' not in properties
    assert properties['desktop_session']['anyOf'][0]['enum'] == ['gnome-session', 'gnome-shell']


@pytest.mark.asyncio
async def test_desktop_start_tool_defaults_gnome_session_to_ubuntu(monkeypatch):
    from nexus.extensions.tools.desktop import tools
    from nexus.extensions.tools.desktop.tools import StartInput, start_tool

    calls: list[dict[str, object]] = []

    class FakeRuntime:
        async def gui_start(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                success=True,
                display=':99',
                vnc_port=5900,
                novnc_port=6080,
                desktop_session='gnome-session',
                stderr='',
                error=None,
            )

    async def fake_runtime():
        return FakeRuntime()

    monkeypatch.setattr(tools, '_runtime', fake_runtime)

    response = await start_tool(StartInput(desktop_session='gnome-session'), context=None)

    assert response.meta['success'] is True
    assert calls == [{'desktop_session': 'gnome-session', 'gnome_session_name': 'ubuntu'}]


def test_build_run_config_observation_mode_adds_pause_steps():
    from tests.integration.gui_agent.demo_config import build_run_config, build_user_prompt

    prompt = build_user_prompt(observe=True, after_launch_seconds=9, before_close_seconds=4)
    config = build_run_config(api_key='sk-test', observe=True, after_launch_seconds=9, before_close_seconds=4)

    assert 'Call desktop/bash with command: sleep 9' in prompt
    assert 'Call desktop/bash with command: sleep 4' in prompt
    assert prompt.index('Call desktop/bash with command: sleep 9') < prompt.index('5. Call desktop/screenshot.')
    assert prompt.index('17. Call desktop/bash with command: sleep 4') < prompt.index(
        '18. Call desktop/screenshot again'
    )
    assert 'NexusSecondary' in prompt
    assert 'Nexus GUI dialog B' in prompt
    assert config['agent_config']['user_prompt'] == prompt


def test_resolve_observation_settings_reads_env(monkeypatch):
    from tests.integration.gui_agent.demo_config import resolve_observation_settings

    monkeypatch.setenv('NEXUS_GUI_OBSERVE', '1')
    monkeypatch.setenv('NEXUS_GUI_OBSERVE_AFTER_LAUNCH_SECONDS', '9')
    monkeypatch.setenv('NEXUS_GUI_OBSERVE_BEFORE_CLOSE_SECONDS', '4')

    settings = resolve_observation_settings()

    assert settings.observe is True
    assert settings.after_launch_seconds == 9
    assert settings.before_close_seconds == 4


def test_build_run_config_is_accepted_by_run_input():
    from tests.integration.gui_agent.demo_config import build_run_config

    run_input = RunInput.from_dict(build_run_config(api_key='sk-test'))

    assert run_input.task_name == 'identity'
    assert run_input.agent_name == 'main_agent'
    assert run_input.runtime_provider_name == 'docker'
    assert run_input.runtime_provider_config['force_rebuild'] is False


def test_example_yaml_resolves_with_environment(monkeypatch):
    monkeypatch.setenv('MODEL_SERVICE_API_APP_ID', 'app-test')
    monkeypatch.setenv('MODEL_SERVICE_API_APP_KEY', 'key-test')
    monkeypatch.setenv('MODEL_SERVICE_API_BASE_URL', 'http://model-service.test/openai/v1')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('NEXUS_MODEL', raising=False)

    run_input = RunInput.from_file('tests/integration/gui_agent/configs/local_gui_agent_gpt54mini.yaml')

    assert run_input.task_name == 'identity'
    assert run_input.agent_name == 'main_agent'
    assert run_input.runtime_provider_name == 'docker'
    assert (
        run_input.openai_clients['main']['config']['client_args']['base_url'] == 'http://model-service.test/openai/v1'
    )
    assert run_input.openai_clients['main']['config']['client_args']['api_key'].startswith(
        'app-test:key-test?provider=azure_openai&model=gpt-5.5&timeout=60'
    )


def test_gongfeng_example_yaml_uses_remote_images_and_fixed_gui_ports(monkeypatch):
    monkeypatch.setenv('MODEL_SERVICE_API_APP_ID', 'app-test')
    monkeypatch.setenv('MODEL_SERVICE_API_APP_KEY', 'key-test')
    monkeypatch.setenv('MODEL_SERVICE_API_BASE_URL', 'http://model-service.test/openai/v1')
    monkeypatch.setenv('GONGFENG_TOKEN', 'gf-test')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('NEXUS_MODEL', raising=False)
    monkeypatch.delenv('NEXUS_GONGFENG_RUNTIME_IMAGE', raising=False)
    monkeypatch.delenv('NEXUS_GONGFENG_DESKTOP_IMAGE', raising=False)

    run_input = RunInput.from_file('tests/integration/gui_agent/configs/gongfeng_gui_agent_gpt54mini.yaml')

    assert run_input.task_name == 'identity'
    assert run_input.agent_name == 'main_agent'
    assert run_input.runtime_provider_name == 'gongfeng'
    assert run_input.runtime_provider_config['print_port_mappings'] is True
    assert (
        run_input.runtime_provider_config['runtime_image']
        == 'current'
    )
    assert (
        run_input.runtime_configs['main']['image']
        == 'mirrors.tencent.com/hunyuan_yanguan/nexus-ubuntu-session:latest'
    )
    assert run_input.runtime_configs['main']['exposed_ports'] == [5900, 6080]
    assert run_input.runtime_configs['main']['environment']['NEXUS_GUI_REQUIRE_SYSTEM_DBUS'] == '1'


def test_example_yaml_keeps_multi_window_flow_in_sync(monkeypatch):
    from tests.integration.gui_agent.demo_config import build_user_prompt

    monkeypatch.setenv('MODEL_SERVICE_API_APP_ID', 'app-test')
    monkeypatch.setenv('MODEL_SERVICE_API_APP_KEY', 'key-test')
    monkeypatch.setenv('MODEL_SERVICE_API_BASE_URL', 'http://model-service.test/openai/v1')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('NEXUS_MODEL', raising=False)

    run_input = RunInput.from_file('tests/integration/gui_agent/configs/local_gui_agent_gpt54mini.yaml')
    yaml_prompt = run_input.agent_config['user_prompt']
    helper_prompt = build_user_prompt()
    expected_fragments = [
        'desktop/start tool',
        '"desktop_session":"gnome-session","session_options":{"gnome_session_name":"ubuntu"}',
        'xterm -title NexusPrimary',
        'xterm -title NexusSecondary',
        '/tmp/nexus_gui_agent_secondary_demo.txt',
        'Nexus GUI dialog A',
        'Nexus GUI dialog B',
        'Close the secondary xterm',
        'Close the primary xterm',
    ]

    for fragment in expected_fragments:
        assert fragment in yaml_prompt
        assert fragment in helper_prompt
    assert run_input.agent_config['max_iterations'] == 30


def test_runtime_smoke_entrypoints_exist():
    assert Path('tests/integration/gui_agent/run_local_gui_runtime_smoke.py').is_file()


def test_runtime_smoke_python_entrypoint_ensures_runtime_image_before_acquire():
    script = Path('tests/integration/gui_agent/run_local_gui_runtime_smoke.py').read_text()

    assert 'ensure_runtime_image' in script
    assert script.index('await ensure_runtime_image()') < script.index('DockerRuntimeProvider(')


def test_runtime_smoke_entrypoint_exports_main():
    spec = importlib.util.spec_from_file_location(
        'run_local_gui_runtime_smoke', 'tests/integration/gui_agent/run_local_gui_runtime_smoke.py'
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'main')


def test_runtime_smoke_provider_reuses_prebuilt_runtime_image():
    spec = importlib.util.spec_from_file_location(
        'run_local_gui_runtime_smoke', 'tests/integration/gui_agent/run_local_gui_runtime_smoke.py'
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = Path('tests/integration/gui_agent/run_local_gui_runtime_smoke.py').read_text()

    assert 'await ensure_runtime_image()' in source
    assert 'force_rebuild=False' in source
    assert 'await runtime.gui_start()' in source


def test_ensure_runtime_image_force_builds_current_tag_once(monkeypatch):
    from tests.integration.gui_agent import runtime_image

    calls = []

    async def fake_build_runtime_image(image_name: str, *, force_rebuild: bool) -> None:
        calls.append((image_name, force_rebuild))

    monkeypatch.setattr(runtime_image, 'get_package_version', lambda: '9.9.9')
    monkeypatch.setattr(runtime_image, 'build_runtime_image', fake_build_runtime_image)
    monkeypatch.delenv(runtime_image.RUNTIME_IMAGE_PREBUILT_ENV, raising=False)

    asyncio.run(runtime_image.ensure_runtime_image())

    assert calls == [('nexus-runtime:9.9.9', True)]
    assert os.environ[runtime_image.RUNTIME_IMAGE_PREBUILT_ENV] == '1'


def test_ensure_runtime_image_skips_when_shell_prebuilt_marker_is_set(monkeypatch):
    from tests.integration.gui_agent import runtime_image

    calls = []

    async def fake_build_runtime_image(image_name: str, *, force_rebuild: bool) -> None:
        calls.append((image_name, force_rebuild))

    monkeypatch.setattr(runtime_image, 'build_runtime_image', fake_build_runtime_image)
    monkeypatch.setenv(runtime_image.RUNTIME_IMAGE_PREBUILT_ENV, '1')

    asyncio.run(runtime_image.ensure_runtime_image())

    assert calls == []
