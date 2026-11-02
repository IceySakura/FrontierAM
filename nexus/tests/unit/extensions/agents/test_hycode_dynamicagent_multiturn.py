import pytest

from nexus.extensions.agents.hycode_dynamicagent import runtime_setup


@pytest.mark.asyncio
@pytest.mark.parametrize('enable_user_input_hooks', [False, True])
async def test_dynamicagent_user_input_hooks_control_session_initialize(
    monkeypatch,
    enable_user_input_hooks: bool,
):
    captured: dict[str, object] = {}

    async def fake_run_installed_cli_session_turn(**kwargs):
        captured.update(kwargs)
        return {'success': True}

    monkeypatch.setattr(
        'nexus.extensions.agents.installed_common.multiturn.run_installed_cli_session_turn',
        fake_run_installed_cli_session_turn,
    )

    await runtime_setup.run_hycode_dynamicagent_session_turn(
        session_id='session-1',
        prompt='hello',
        enable_user_input_hooks=enable_user_input_hooks,
        npm_registry='https://registry.example.test/',
        pre_start_scripts=[{'interpreter': 'bash', 'code': 'echo custom'}],
    )

    assert captured['enable_user_input_hooks'] is enable_user_input_hooks
    assert 'initialize_session' not in captured
    assert captured['pre_start_scripts'][0]['interpreter'] == 'bash'
    assert 'link_binary bash /bin/bash' in captured['pre_start_scripts'][0]['code']
    assert 'link_binary node' in captured['pre_start_scripts'][0]['code']
    assert 'link_binary npm' in captured['pre_start_scripts'][0]['code']
    assert 'link_binary npx' in captured['pre_start_scripts'][0]['code']
    assert captured['pre_start_scripts'][1]['code'] == 'echo custom'
    assert captured['extra_env']['NPM_CONFIG_REGISTRY'] == 'https://registry.example.test/'
    assert captured['extra_env']['npm_config_registry'] == 'https://registry.example.test/'
    assert captured['extra_env']['NPM_CONFIG_STRICT_SSL'] == 'false'
