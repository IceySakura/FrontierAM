"""Unit coverage for Apple-specific GUI integration config helpers."""

from pathlib import Path

import pytest

from nexus.runner.single import RunInput


def test_runtime_flake_keeps_chromium_without_google_chrome():
    flake = Path('deployment/runtime/flake.nix').read_text()

    assert 'chromium' in flake
    assert 'google-chrome' not in flake
    assert 'allowUnfree = true' not in flake


def test_build_run_config_produces_apple_gui_flow():
    from tests.integration.gui_agent.apple_demo_config import (
        DEFAULT_BASE_IMAGE,
        DEFAULT_MODEL,
        build_run_config,
        build_user_prompt,
    )

    prompt = build_user_prompt()
    config = build_run_config(api_key='sk-test')

    assert 'apple.com' in prompt
    assert 'Learn more' in prompt
    assert config['runtime_configs']['main']['image'] == DEFAULT_BASE_IMAGE
    assert config['runtime_configs']['main']['image'] == 'nexus-ubuntu-session:latest'
    assert config['runtime_configs']['main']['exposed_ports'] == [5900, 6080]
    assert 'NEXUS_GUI_REQUIRE_SYSTEM_DBUS=1' in config['runtime_configs']['main']['extra_args']
    assert 'desktop/start' in config['agent_config']['tools']
    assert 'desktop/computer_13' in config['agent_config']['tools']
    assert 'desktop/bash' in config['agent_config']['tools']
    assert 'desktop/click' not in config['agent_config']['tools']
    assert 'desktop/screenshot' in config['agent_config']['tools']
    assert 'google-chrome' in config['agent_config']['user_prompt']
    assert 'chromium' not in config['agent_config']['user_prompt']
    assert '--enable-unsafe-swiftshader' not in config['agent_config']['user_prompt']
    assert 'rm -rf /tmp/google-chrome-profile' in config['agent_config']['user_prompt']
    assert '--window-position=0,0' in config['agent_config']['user_prompt']
    assert '--window-size=1280,800' in config['agent_config']['user_prompt']
    assert '--start-maximized' in config['agent_config']['user_prompt']
    assert '--start-fullscreen' not in config['agent_config']['user_prompt']
    assert (
        '"desktop_session":"gnome-session","screen_width":1280,"screen_height":800,'
        '"session_options":{"gnome_session_name":"ubuntu"}'
    ) in prompt
    assert 'apt-get' not in config['agent_config']['user_prompt']
    assert config['openai_clients']['main']['config']['request_args']['model'] == DEFAULT_MODEL
    assert config['runtime_provider_config']['print_port_mappings'] is True


def test_build_run_config_observation_mode_adds_sleep_and_bash():
    from tests.integration.gui_agent.apple_demo_config import build_run_config, build_user_prompt

    prompt = build_user_prompt(observe=True, before_click_seconds=12, after_click_seconds=7)
    config = build_run_config(api_key='sk-test', observe=True, before_click_seconds=12, after_click_seconds=7)

    assert 'Do not use the bash tool.' not in prompt
    assert 'Call desktop/bash with command: sleep 12' in prompt
    assert 'Call desktop/bash with command: sleep 7' in prompt
    assert prompt.index('Call desktop/bash with command: sleep 12') < prompt.index('5. Call desktop/screenshot.')
    assert 'desktop/bash' in config['agent_config']['tools']
    assert config['agent_config']['user_prompt'] == prompt


def test_build_run_config_default_mode_waits_for_page_rendering_with_bash():
    from tests.integration.gui_agent.apple_demo_config import DEFAULT_OBSERVE_BEFORE_CLICK_SECONDS, build_run_config

    config = build_run_config(api_key='sk-test')
    prompt = config['agent_config']['user_prompt']

    assert 'desktop/bash' in config['agent_config']['tools']
    wait_step = f'Call desktop/bash with command: sleep {DEFAULT_OBSERVE_BEFORE_CLICK_SECONDS}'
    assert wait_step in prompt
    assert prompt.index(wait_step) < prompt.index('5. Call desktop/screenshot.')
    assert 'desktop/computer_13 with action_type CLICK' in prompt
    assert 'Do not use the bash tool.' not in prompt


def test_resolve_observation_settings_reads_env(monkeypatch):
    from tests.integration.gui_agent.apple_demo_config import resolve_observation_settings

    monkeypatch.setenv('NEXUS_GUI_OBSERVE', '1')
    monkeypatch.setenv('NEXUS_GUI_OBSERVE_BEFORE_CLICK_SECONDS', '12')
    monkeypatch.setenv('NEXUS_GUI_OBSERVE_AFTER_CLICK_SECONDS', '7')

    settings = resolve_observation_settings()

    assert settings.observe is True
    assert settings.before_click_seconds == 12
    assert settings.after_click_seconds == 7


def test_build_run_config_is_accepted_by_run_input():
    from tests.integration.gui_agent.apple_demo_config import build_run_config

    run_input = RunInput.from_dict(build_run_config(api_key='sk-test'))

    assert run_input.task_name == 'identity'
    assert run_input.agent_name == 'main_agent'
    assert run_input.runtime_provider_name == 'docker'
    assert run_input.runtime_provider_config['force_rebuild'] is False


def test_example_yaml_resolves_with_environment(monkeypatch):
    from tests.integration.gui_agent.apple_demo_config import DEFAULT_CHROME_COMMAND

    monkeypatch.setenv('MODEL_SERVICE_API_APP_ID', 'app-test')
    monkeypatch.setenv('MODEL_SERVICE_API_APP_KEY', 'key-test')
    monkeypatch.setenv('MODEL_SERVICE_API_BASE_URL', 'http://model-service.test/openai/v1')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('NEXUS_MODEL', raising=False)

    run_input = RunInput.from_file('tests/integration/gui_agent/configs/local_gui_apple_agent_gpt54mini.yaml')

    assert run_input.task_name == 'identity'
    assert run_input.agent_name == 'main_agent'
    assert run_input.runtime_provider_name == 'docker'
    assert DEFAULT_CHROME_COMMAND in run_input.agent_config['user_prompt']


def test_gongfeng_apple_yaml_uses_remote_images_and_fixed_gui_ports(monkeypatch):
    from tests.integration.gui_agent.apple_demo_config import DEFAULT_CHROME_COMMAND

    monkeypatch.setenv('MODEL_SERVICE_API_APP_ID', 'app-test')
    monkeypatch.setenv('MODEL_SERVICE_API_APP_KEY', 'key-test')
    monkeypatch.setenv('MODEL_SERVICE_API_BASE_URL', 'http://model-service.test/openai/v1')
    monkeypatch.setenv('GONGFENG_TOKEN', 'gf-test')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('NEXUS_MODEL', raising=False)
    monkeypatch.delenv('NEXUS_GONGFENG_RUNTIME_IMAGE', raising=False)
    monkeypatch.delenv('NEXUS_GONGFENG_DESKTOP_IMAGE', raising=False)

    run_input = RunInput.from_file('tests/integration/gui_agent/configs/gongfeng_gui_apple_agent_gpt54mini.yaml')

    assert run_input.task_name == 'identity'
    assert run_input.agent_name == 'main_agent'
    assert run_input.runtime_provider_name == 'gongfeng'
    assert run_input.runtime_provider_config['print_port_mappings'] is True
    assert run_input.runtime_provider_config['runtime_image'] == 'current'
    assert (
        run_input.runtime_configs['main']['image'] == 'mirrors.tencent.com/hunyuan_yanguan/nexus-ubuntu-session:latest'
    )
    assert run_input.runtime_configs['main']['exposed_ports'] == [5900, 6080]
    assert DEFAULT_CHROME_COMMAND in run_input.agent_config['user_prompt']


def test_resolve_openai_settings_requires_model_service_credentials():
    from tests.integration.gui_agent.apple_demo_config import resolve_openai_settings

    with pytest.raises(RuntimeError, match='MODEL_SERVICE_API_APP_ID'):
        resolve_openai_settings({})
