import copy
import json

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from nexus.extensions.agents.workbuddy.prompt_identity import (
    WORKBUDDY_TASK_MARKER,
    PromptIdentityError,
    WorkBuddyPromptGuard,
    prepare_builtin_prompt_plugins,
)
from nexus.extensions.agents.workbuddy.proxy import WorkBuddyProxy
from nexus.extensions.agents.workbuddy.runtime_setup import build_workbuddy_argv


def package(tmp_path: Path) -> tuple[Path, Path]:
    package_root = tmp_path / 'node_modules' / '@tencent' / 'nexus-wb-cli'
    entry = package_root / 'bin' / 'wb.js'
    entry.parent.mkdir(parents=True)
    entry.touch()
    root = package_root / 'resources' / 'plugins' / 'workbuddy-builtin'
    entries = []
    for name, source in (
        ('welcomemode-code', './welcomemode/code'),
        ('interactionmode-craft', './interactionmode/craft'),
        ('prompt-common', './prompt-common'),
        ('welcomemode-work', './welcomemode/work'),
    ):
        entries.append({'name': name, 'source': source})
        plugin = root / source
        (plugin / '.workbuddy-plugin').mkdir(parents=True)
        manifest = {'name': name}
        if name.startswith('welcomemode-'):
            (plugin / 'agents').mkdir()
            (plugin / 'agents' / 'code.md').write_text('root agent')
            (plugin / 'prompt.tpl').write_text('builtin template')
            manifest['agents'] = ['agents/code.md']
        (plugin / '.workbuddy-plugin' / 'plugin.json').write_text(json.dumps(manifest))
    (root / '.codebuddy-plugin').mkdir()
    (root / '.codebuddy-plugin' / 'marketplace.json').write_text(
        json.dumps({'name': 'workbuddy-builtin', 'plugins': entries})
    )
    return entry, root


def test_builtin_settings_keep_user_values_and_select_exact_preset(tmp_path: Path) -> None:
    wb_js, root = package(tmp_path)
    config = tmp_path / 'config'
    config.mkdir()
    path = config / 'settings.json'
    path.write_text(
        json.dumps(
            {
                'permissions': {'defaultMode': 'plan'},
                'enabledPlugins': {'my-plugin@custom': True, 'welcomemode-work@workbuddy-builtin': True},
                'extraKnownMarketplaces': {'custom': {'source': {'source': 'directory', 'path': '/custom'}}},
            }
        )
    )
    prepare_builtin_prompt_plugins(wb_js, config, 'code', 'craft')
    settings = json.loads(path.read_text())
    assert settings['permissions'] == {'defaultMode': 'plan'}
    assert settings['enabledPlugins']['my-plugin@custom'] is True
    assert settings['enabledPlugins']['welcomemode-code@workbuddy-builtin'] is True
    assert settings['enabledPlugins']['welcomemode-work@workbuddy-builtin'] is False
    assert settings['extraKnownMarketplaces']['workbuddy-builtin']['source']['path'] == str(root)
    assert settings['extraKnownMarketplaces']['custom']['source']['path'] == '/custom'
    assert path.stat().st_mode & 0o777 == 0o600


def test_missing_root_agent_fails_before_writing_settings(tmp_path: Path) -> None:
    wb_js, root = package(tmp_path)
    (root / 'welcomemode/code/agents/code.md').unlink()
    config = tmp_path / 'config'
    with pytest.raises(ValueError, match='root agent is missing'):
        prepare_builtin_prompt_plugins(wb_js, config, 'code', 'craft')
    assert not (config / 'settings.json').exists()


def test_malformed_settings_are_not_silently_overwritten(tmp_path: Path) -> None:
    wb_js, _ = package(tmp_path)
    config = tmp_path / 'config'
    config.mkdir()
    path = config / 'settings.json'
    path.write_text('{not-json')
    with pytest.raises(json.JSONDecodeError, match='Expecting property name'):
        prepare_builtin_prompt_plugins(wb_js, config, 'code', 'craft')
    assert path.read_text() == '{not-json'


def request(system: str, user: str = 'Fix the failing test') -> dict:
    return {
        'model': 'general-model',
        'messages': [
            {'role': 'system', 'content': [{'type': 'text', 'text': system}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': user}]},
        ],
    }


MAIN = 'This conversation is powered by general-model\n\n' + WORKBUDDY_TASK_MARKER
CODEBUDDY = 'You are CodeBuddy Code.You are an interactive CLI tool.'


@pytest.mark.asyncio
async def test_codebuddy_fallback_never_reaches_the_model_bridge() -> None:
    proxy = WorkBuddyProxy(sio=Mock(), verify_prompt=True)
    bridge = Mock()
    bridge.request = AsyncMock()
    proxy.bridge = bridge
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy._create_app()), base_url='http://test'
    ) as client:
        response = await client.post('/v1/chat/completions', json=request(CODEBUDDY))
    assert response.status_code == 400
    bridge.request.assert_not_awaited()
    assert proxy.prompt_guard is not None
    assert proxy.prompt_guard.report()['verified'] is False


def test_verified_root_and_linked_codebuddy_child_keep_actual_prompts() -> None:
    guard = WorkBuddyPromptGuard()
    root = request(MAIN)
    original = copy.deepcopy(root)
    guard.check(root)
    guard.inspect_response(
        {
            'choices': [
                {
                    'message': {
                        'tool_calls': [
                            {
                                'function': {
                                    'name': 'Agent',
                                    'arguments': json.dumps({'prompt': 'Inspect parser.py for regressions'}),
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    child = request(CODEBUDDY, '<user_query>Inspect parser.py for regressions</user_query>')
    guard.check(child)
    assert root == original
    assert child['messages'][0]['content'][0]['text'] == CODEBUDDY
    assert guard.report()['verified'] is True
    assert guard.report()['sub_agent_requests'] == 1


def test_a_later_unlinked_codebuddy_request_invalidates_the_run() -> None:
    guard = WorkBuddyPromptGuard()
    guard.check(request(MAIN))
    with pytest.raises(PromptIdentityError, match='no WorkBuddy main template'):
        guard.check(request(CODEBUDDY))
    assert guard.report()['verified'] is False
    with pytest.raises(PromptIdentityError, match='previous request failed'):
        guard.check(request(MAIN))


def test_title_is_not_treated_as_main_or_forwarded() -> None:
    guard = WorkBuddyPromptGuard()
    with pytest.raises(PromptIdentityError, match='session-title'):
        guard.check(request('Generate a concise, sentence-case title for this session'))


def test_rollout_flag_is_explicit_and_cannot_be_disabled_by_extra_args() -> None:
    argv = build_workbuddy_argv('node', '/installed/bin/wb.js', 'task', effort='high', rollout_mode=True)
    assert argv[argv.index('--features') + 1] == '{"rollout":true}'
    assert argv[-1] == 'task'
    with pytest.raises(ValueError, match='owns --features'):
        build_workbuddy_argv(
            'node', '/installed/bin/wb.js', 'task', rollout_mode=True, extra_cli_args=['--features={}']
        )


class _RecordingBridge:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    async def request(self, body: dict, timeout: float) -> dict:
        self.bodies.append(body)
        return {'id': 'response', 'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]}


@pytest.mark.asyncio
async def test_rollout_proxy_keeps_the_gateway_identity_and_template_kwargs(tmp_path: Path) -> None:
    from nexus.extensions.agents.workbuddy.proxy import WorkBuddyMetadataCaptureProxy

    headers = {
        'X-Agent-Type': 'main',
        'X-Agent-Purpose': 'main',
        'X-Conversation-Id': 'conv-1',
        'X-Request-Id': 'request-1',
    }
    for proxy in (
        WorkBuddyProxy(sio=None, verify_prompt=True, chat_template_kwargs={'reasoning_effort': 'no_think'}),
        WorkBuddyMetadataCaptureProxy(
            sio=None,
            verify_prompt=True,
            chat_template_kwargs={'reasoning_effort': 'no_think'},
            capture_dir=str(tmp_path),
        ),
    ):
        bridge = _RecordingBridge()
        proxy.bridge = bridge
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy._create_app()), base_url='http://test'
        ) as client:
            ok = await client.post('/v1/chat/completions', json=request(MAIN), headers=headers)
            rejected = await client.post('/v1/chat/completions', json=request(CODEBUDDY), headers=headers)

        assert ok.status_code == 200
        assert rejected.status_code == 400
        assert len(bridge.bodies) == 1
        forwarded = bridge.bodies[0]
        assert forwarded['request_meta']['workbuddy']['conversation_id'] == 'conv-1'
        assert forwarded['extra_body']['chat_template_kwargs'] == {'reasoning_effort': 'no_think'}
        assert proxy.prompt_guard is not None
    assert list(tmp_path.glob('*.request.json'))
