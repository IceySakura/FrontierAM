"""Shared registrations keep their contracts even while a macOSWorld run is active."""

from __future__ import annotations

import asyncio
import base64
import io

from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from PIL import Image

from nexus.extensions.agents.cua.cua_anthropic_agent import agent as shared
from nexus.extensions.agents.cua.cua_gpt_agent import agent as shared_gpt
from nexus.extensions.agents.cua.macosworld_cua import anthropic as mac
from nexus.extensions.agents.cua.macosworld_cua import gpt as mac_gpt
from nexus.extensions.tasks.mac_os_world import task
from nexus.runtime.providers.devcloud.client import DevCloudRuntime
from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime


def test_platform_name_alone_does_not_enable_agent_adaptations():
    ordinary = shared.CuaAnthropicAdapter()
    special = mac.MacOSWorldAnthropicAdapter()
    params = mac.MacOSWorldAnthropicInput(user_prompt='task', platform_name='macos')
    block = {'input': {'action': 'key', 'text': 'super_l+a'}}
    assert "keyDown('win')" in ordinary.parse_action(block, (1, 1), params)
    assert "keyDown('command')" in special.parse_action(block, (1, 1), params)
    assert 'Ubuntu' in ordinary.prompt(params, '')
    assert 'macOS' in special.prompt(params, '')
    assert ordinary.profile('claude-3-7-sonnet') is shared.DEFAULT_CLAUDE_PROFILE
    assert special.profile('claude-3-7-sonnet') is mac.CLAUDE_37_PROFILE


def test_gpt_common_entry_ignores_macosworld_paper_settings():
    params = mac_gpt.MacOSWorldGptInput(
        user_prompt='task',
        platform_name='macos',
        prompt_style='macosworld_paper',
        home_dir='/Users/tester',
    )
    ordinary = shared_gpt.CuaGptAdapter()
    special = mac_gpt.MacOSWorldGptAdapter()
    assert '/home/user' in ordinary.instructions(params)
    assert '`tester`' in special.instructions(params)
    args = {'x': 10, 'y': 20, 'button': 'back'}
    assert "hotkey('alt', 'left')" in ordinary.parse_action('click', args, params)
    assert "hotkey('command', '[')" in special.parse_action('click', args, params)


@pytest.mark.asyncio
async def test_claude_adapters_are_isolated_across_concurrent_runs(monkeypatch):
    png = io.BytesIO()
    Image.new('RGB', (4, 4)).save(png, format='PNG')
    image = base64.b64encode(png.getvalue()).decode()
    current = ContextVar('isolation_context')
    monkeypatch.setattr(shared, 'require_context', current.get)

    async def run(entry, input_type):
        runtime = SimpleNamespace(
            gui_screenshot=AsyncMock(return_value={'success': True, 'image': image}),
            run_command=AsyncMock(return_value={'status': 'Finished', 'return_code': 0}),
        )
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason='tool_calls',
                    message=SimpleNamespace(
                        content='I will type the literal marker [DONE].',
                        thinking_blocks=[],
                        tool_calls=[
                            SimpleNamespace(
                                id='c1',
                                function=SimpleNamespace(
                                    name='computer', arguments='{"action":"left_click","coordinate":[1,1]}'
                                ),
                            )
                        ],
                    ),
                )
            ]
        )
        client = SimpleNamespace(
            api_style='messages',
            get_effective_request_args=lambda: {'model': 'claude-opus-4-8'},
            create_completion=AsyncMock(return_value=completion),
        )
        ctx = SimpleNamespace(
            get_runtime=AsyncMock(return_value=runtime),
            get_openai_client=lambda _: client,
            tool_state={},
        )
        token = current.set(ctx)
        try:
            await asyncio.sleep(0)
            result = await entry(
                input_type(
                    user_prompt='task',
                    max_iterations=1,
                    transport='client',
                    platform_name='macos',
                    pyautogui_execution_mode='run_command',
                    step_pause=0,
                )
            )
            return result, runtime.run_command.await_count
        finally:
            current.reset(token)

    (legacy, legacy_actions), (adapted, adapted_actions) = await asyncio.gather(
        run(shared.run_cua_anthropic_agent, shared.CuaAnthropicAgentInput),
        run(mac.run_macosworld_cua_anthropic, mac.MacOSWorldAnthropicInput),
    )
    assert legacy.exit_status == 'finished'
    assert legacy_actions == 0
    assert adapted.exit_status == 'max_iterations'
    assert adapted_actions == 1
    assert 'screenshots' not in legacy.model_dump()
    assert adapted.screenshots >= 1


def test_only_macosworld_runtime_selects_private_protocol_overlay():
    from nexus.extensions.agents.installed_common.native_bridge import _build_adapter
    from nexus.runtime.bridge.adapters.devcloud_macos import DevCloudMacOSBridgeAdapter, DevCloudMacOSBridgeLaunchResult
    from nexus.runtime.bridge.host_client import BridgeHostClient
    from nexus.runtime.providers.macosworld_devcloud.bridge import MacOSWorldBridgeAdapter

    legacy = DevCloudRuntime(base_url='http://fixture:5000')
    adapted = MacOSWorldDevCloudRuntime(base_url='http://fixture:5000', proxy_url='http://proxy:8080')
    assert type(_build_adapter(legacy)) is DevCloudMacOSBridgeAdapter
    adapter = _build_adapter(adapted)
    assert type(adapter) is MacOSWorldBridgeAdapter
    assert adapter._health_request_kwargs(None)['proxy'] == 'http://proxy:8080'
    client = adapter.create_host_client(
        DevCloudMacOSBridgeLaunchResult(
            base_url='http://fixture:8765',
            local_base_url='http://localhost:8765',
            proxy_url='http://proxy:8080',
        )
    )
    assert isinstance(client, BridgeHostClient)


@pytest.mark.asyncio
@pytest.mark.parametrize('expired', [False, True])
async def test_operation_timeout_diagnostics_are_local_to_macosworld(monkeypatch, expired):
    monkeypatch.setattr(task, '_run_mac_os_world', AsyncMock(side_effect=TimeoutError('screenshot deadline')))
    monkeypatch.setattr(
        task, 'require_context', lambda: SimpleNamespace(timeout_handle=SimpleNamespace(expired=lambda: expired))
    )
    with pytest.raises(TimeoutError if expired else RuntimeError, match='screenshot deadline'):
        await task.run_mac_os_world(task.MacOSWorldTaskInput(task_id='test', category='test', run_agent=AsyncMock()))


def test_default_anthropic_tool_validation_is_not_expanded():
    from nexus.extensions.openai_clients.vertex import convert_tools_to_anthropic

    with pytest.raises(ValueError, match='Unsupported'):
        convert_tools_to_anthropic([{'type': 'computer_toolset_20260801'}])


def test_new_registrations_do_not_replace_shared_names():
    from nexus.core.module_meta import discover_all_modules

    modules = discover_all_modules(include_compiled_manifest=False)
    providers = {k: v for module in modules for k, v in module.objects.get('runtime_providers', {}).items()}
    agents = {k: v for module in modules for k, v in module.objects.get('agents', {}).items()}
    clients = {k: v for module in modules for k, v in module.objects.get('openai_clients', {}).items()}
    assert providers['devcloud'] == 'nexus.runtime.providers.devcloud.provider'
    assert 'macosworld_devcloud' not in providers
    assert agents['cua_anthropic_agent'] == shared.__name__
    assert agents['macosworld_cua_anthropic'] == mac.__name__
    assert clients['infer_proxy'] == 'nexus.extensions.openai_clients.infer_proxy'
    assert clients['macosworld_infer_proxy'] == 'nexus.extensions.tasks.mac_os_world.clients'


def test_shared_agent_schemas_do_not_advertise_macosworld_settings():
    from nexus.extensions.agents.codex.agent import CodexNativeBridgeInput
    from nexus.extensions.agents.codex.macosworld.agent import MacOSWorldCodexInput

    assert 'max_screenshots' not in shared.CuaAnthropicAgentInput.model_fields
    assert 'computer_only' not in shared.CuaAnthropicAgentInput.model_fields
    assert 'max_recent_screenshots' not in shared_gpt.CuaGptAgentInput.model_fields
    assert 'prompt_style' not in shared_gpt.CuaGptAgentInput.model_fields
    assert 'api_protocol' not in CodexNativeBridgeInput.model_fields
    assert MacOSWorldCodexInput.model_fields['api_protocol'].default == 'responses'
    assert 'screenshots' not in shared.CuaAnthropicAgentOutput.model_fields
    assert 'screenshots_sent' not in shared_gpt.CuaGptAgentOutput.model_fields


@pytest.mark.asyncio
async def test_bridge_factory_preserves_task_runtime_wrapper(monkeypatch):
    from nexus.extensions.agents.installed_common.native_bridge import _build_adapter
    from nexus.extensions.tasks.mac_os_world.runtime_adapter import MacOSWorldRuntimeAdapter

    runtime = MacOSWorldDevCloudRuntime(base_url='http://fixture:5000')
    execute = AsyncMock(return_value={'status': 'Finished', 'return_code': 0})
    monkeypatch.setattr(runtime, 'run_command', execute)
    wrapped = MacOSWorldRuntimeAdapter(runtime, screenshot_timeout=30)
    adapter = _build_adapter(wrapped)
    assert adapter.runtime is wrapped
    await adapter.runtime.run_command('echo bridge', timeout=30)
    assert execute.await_args.kwargs['command'].startswith('export PATH="/usr/local/bin:$PATH"')
