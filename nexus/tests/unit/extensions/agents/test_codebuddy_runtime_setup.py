from __future__ import annotations

import inspect
import json

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.codebuddy import runtime_setup


def test_setup_preserves_ten_positional_arguments() -> None:
    patches = [{'path': '~/.codebuddy/settings.json', 'content': {'test': True}}]
    scripts = [{'code': 'true'}]
    signature = inspect.signature(runtime_setup.setup_and_run_codebuddy)
    bound = signature.bind('work', '/workspace', 3600, 'latest', None, 'model', None, None, patches, scripts)
    bound.apply_defaults()
    assert bound.arguments['config_file_patches'] is patches, 'old config patches must keep their position'
    assert bound.arguments['pre_start_scripts'] is scripts, 'old scripts must keep their position'
    assert bound.arguments['skip_permissions'] is True, 'one-shot default must remain unchanged'
    assert signature.parameters['skip_permissions'].kind is inspect.Parameter.KEYWORD_ONLY, 'flag must be keyword-only'


@pytest.mark.parametrize('skip_permissions', [False, True])
def test_permission_cli_args_deduplicate_and_enforce_explicit_setting(skip_permissions: bool) -> None:
    extras = ['--verbose', '--dangerously-skip-permissions', '--dangerously-skip-permissions']
    result = runtime_setup.build_codebuddy_permission_cli_args(skip_permissions=skip_permissions, extra_cli_args=extras)
    assert result.count('--dangerously-skip-permissions') == int(skip_permissions), 'dedicated setting must win'
    assert '--verbose' in result, 'unrelated CLI flags must survive'
    assert len(extras) == 3, 'caller arguments must not be mutated'


def test_config_patch_respects_explicit_home_and_keeps_absolute_paths(tmp_path: Path) -> None:
    home = tmp_path / 'branch-home'
    runtime_setup._apply_json_patch('~/.codebuddy/settings.json', {'nested': {'first': 1}}, home=home)
    runtime_setup._apply_json_patch('~/.codebuddy/settings.json', {'nested': {'second': 2}}, home=home)
    result = json.loads((home / '.codebuddy/settings.json').read_text())
    assert result == {'nested': {'first': 1, 'second': 2}}, 'branch home patch must preserve existing values'
    absolute = tmp_path / 'absolute.json'
    assert runtime_setup._resolve_config_path(str(absolute), home=home) == absolute, 'absolute paths must not move'
    assert runtime_setup._resolve_config_path('~', home=home) == home, 'bare tilde must use explicit home'


@pytest.mark.asyncio
@pytest.mark.parametrize('skip_permissions', [False, True])
async def test_setup_preserves_one_shot_home_and_controls_permission_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skip_permissions: bool
) -> None:
    # NOTE(wallyshi): Patch runtime-only dependencies at their import sites without starting real services.
    import nexus.extensions.agents.codebuddy.installer as installer
    import nexus.runtime.context as runtime_context
    import nexus.runtime.server.services.openai_proxy as proxy_module

    home = tmp_path / 'process-home'
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setattr(installer, 'install_codebuddy_via_npm', AsyncMock(return_value=Path('/installed/codebuddy.js')))
    monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: SimpleNamespace(sio=object()))
    proxy = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(proxy_module, 'OpenAIProxy', lambda **kwargs: proxy)
    monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 12345)
    monkeypatch.setattr(runtime_setup, 'find_nexus_runtime_binary', lambda name: '/installed/node')
    script = AsyncMock()
    run = AsyncMock(return_value=(0, 'answer', ''))
    monkeypatch.setattr(runtime_setup, '_run_pre_start_script', script)
    monkeypatch.setattr(runtime_setup, '_run_codebuddy_streaming', run)
    result = await runtime_setup.setup_and_run_codebuddy(
        'hello',
        working_dir=str(tmp_path / 'workspace'),
        extra_env={'CUSTOM_OPTION': 'value'},
        extra_cli_args=['--verbose', '--dangerously-skip-permissions'],
        config_file_patches=[{'path': '~/.codebuddy/settings.json', 'content': {'test': True}}],
        pre_start_scripts=[{'code': 'true'}],
        skip_permissions=skip_permissions,
    )
    assert result['success'], 'one-shot execution must retain result contract'
    assert (home / '.codebuddy/models.json').is_file(), 'models must keep the one-shot HOME path'
    assert json.loads((home / '.codebuddy/settings.json').read_text()) == {'test': True}, 'patch must keep HOME path'
    assert script.await_args is not None, 'pre-start script must be awaited'
    assert script.await_args.args[2]['HOME'] == str(home), 'scripts must inherit HOME'
    assert run.await_args is not None, 'CLI must be awaited'
    arguments: dict[str, Any] = dict(run.await_args.kwargs)
    assert arguments['env']['HOME'] == str(home), 'CLI must inherit child HOME'
    assert arguments['cmd'].count('--dangerously-skip-permissions') == int(skip_permissions), 'CLI must enforce flag'
    assert arguments['cmd'][-1] == 'hello', 'prompt must remain the last CLI argument'
    assert arguments['env']['CUSTOM_OPTION'] == 'value', 'one-shot execution must preserve extra environment'
    proxy.stop.assert_awaited_once()


def test_models_json_keeps_the_codebuddy_output_cap_unless_raised(tmp_path: Path) -> None:
    signature = inspect.signature(runtime_setup._write_models_json)
    assert signature.parameters['max_output_tokens'].kind is inspect.Parameter.KEYWORD_ONLY
    runtime_setup._write_models_json(tmp_path, 'http://proxy/v1/chat/completions', 'm', 'k')
    assert json.loads((tmp_path / 'models.json').read_text())['models'][0]['maxOutputTokens'] == 16384
    runtime_setup._write_models_json(tmp_path, 'http://proxy/v1/chat/completions', 'm', 'k', max_output_tokens=65536)
    assert json.loads((tmp_path / 'models.json').read_text())['models'][0]['maxOutputTokens'] == 65536
