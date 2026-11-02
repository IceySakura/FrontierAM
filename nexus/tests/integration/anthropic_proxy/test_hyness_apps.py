"""Installed Hyness Apps through the real Nexus launcher, proxy and result adapter.

Set HYNESS_TEST_PREFIX to a clean npm installation and HYNESS_TEST_NODE to its
supported Node binary. Package installation and the host model backend are the
only substitutes; no real model, sandbox service or GPU is used.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import uuid

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from PIL import Image
from omegaconf import OmegaConf

from nexus.extensions.agents.hyness import HynessInput, agent, runtime_setup
from nexus.extensions.agents.hyness.agent import encode_user_prompt
from nexus.extensions.configs.load_config import load_config
from nexus.runtime import context
from nexus.runtime.server.services.anthropic_proxy import server


class ScriptedBridge:
    def __init__(
        self,
        tool: str,
        *,
        actions: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        self.actions = actions if actions is not None else [(tool, {'command': "printf 'after\\n' > value.txt"})]
        self.requests: list[dict[str, Any]] = []

    async def request(self, params: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        del timeout
        self.requests.append(deepcopy(params))
        number = len(self.requests)
        message: dict[str, Any] = {'role': 'assistant', 'content': 'DONE'}
        if number <= len(self.actions):
            name, arguments = self.actions[number - 1]
            message = {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': f'call-{number}',
                        'type': 'function',
                        'function': {'name': name, 'arguments': json.dumps(arguments)},
                    }
                ],
            }
        return {
            'id': f'completion-{number}',
            'object': 'chat.completion',
            'created': 1,
            'model': params['model'],
            'choices': [
                {'index': 0, 'message': message, 'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}
            ],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20},
        }


@pytest.fixture
def installed_hyness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    prefix_value = os.environ.get('HYNESS_TEST_PREFIX')
    node = shutil.which(os.environ.get('HYNESS_TEST_NODE', 'node'))
    if not prefix_value or not node:
        raise pytest.skip.Exception('set HYNESS_TEST_PREFIX and a supported HYNESS_TEST_NODE')
    prefix = Path(prefix_value).resolve()
    engine = prefix / 'node_modules/@tencent/hyness'
    entry = engine / 'apps/cli/dist/hyness.js'
    assert entry.is_file(), 'HYNESS_TEST_PREFIX must contain an installed Hyness engine'
    version = json.loads((engine / 'package.json').read_text())['version']
    installed = {'prefix': str(prefix), 'entry': str(entry), 'node': node, 'version': version}

    async def install(_transport, packages, _registry):
        assert f'@tencent/hyness@{version}' in packages
        return installed

    monkeypatch.setattr(runtime_setup, 'install_hyness', install)
    monkeypatch.setattr(context, 'require_runtime_control', lambda: context.RuntimeControl(sio=object()))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    for name in list(os.environ):
        if name.startswith(('ANTHROPIC_', 'CLAUDE_', 'HYNESS_')) or name.lower() in {
            'http_proxy',
            'https_proxy',
            'all_proxy',
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('NO_PROXY', '*')
    monkeypatch.setenv('no_proxy', '*')
    return installed


async def launch(config: HynessInput, bridge: ScriptedBridge, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(server, 'LLMBridge', lambda _sio: bridge)
    payload = config.model_dump(
        exclude={
            'user_prompt',
            'version',
            'disable_invoke_wait_timeout',
            'mcp_servers',
            'no_thinking_request_args',
            'no_thinking_request_args_mode',
            'haiku_model',
        }
    )
    return await runtime_setup.setup_and_run_hyness(
        task=encode_user_prompt(config.user_prompt, config.input_format), **payload
    )


@pytest.mark.integration
async def test_input_reaches_first_model_request_without_tools(
    installed_hyness: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    buffer = io.BytesIO()
    Image.new('RGB', (64, 48), (220, 30, 60)).save(buffer, format='PNG')
    data = base64.b64encode(buffer.getvalue()).decode()
    prompt = [
        {'type': 'text', 'text': 'Describe this input without calling tools.'},
        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{data}'}},
    ]
    config = HynessInput(
        user_prompt=prompt,
        input_format='stream-json',
        working_dir=str(tmp_path),
        packages=[f'@tencent/hyness@{installed_hyness["version"]}'],
        app_config='@tencent/hyness/app-definitions/hyness/app.yaml',
        harness_id=None,
        app_config_overlay=None,
        max_turns=1,
        timeout=60,
        extra_cli_args=['--setting-sources', ''],
        extra_env=process_options(),
    )
    bridge = ScriptedBridge('unused', actions=[])
    result = await launch(config, bridge, monkeypatch)
    assert result['success'], result
    assert result['process_exit_code'] == 0
    assert len(bridge.requests) == 1
    messages = bridge.requests[0]['messages']
    images = [
        part
        for message in messages
        if message['role'] == 'user' and isinstance(message['content'], list)
        for part in message['content']
        if part['type'] == 'image_url'
    ]
    assert len(images) == 1
    encoded = images[0]['image_url']['url'].split(',', 1)[1]
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert decoded.size == (64, 48)
    assert decoded.convert('RGB').getpixel((0, 0)) == (220, 30, 60)
    assert not any(message['role'] == 'tool' for message in messages)
    assert agent.classify_hyness_result(result).termination_reason == 'ENV_DONE'


def process_options() -> dict[str, str]:
    return {
        'CLAUDE_CODE_DISABLE_BACKGROUND_TASKS': '1',
        'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
        'CLAUDE_CODE_MAX_RETRIES': '0',
        'CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK': '1',
        'ENABLE_TOOL_SEARCH': 'false',
        'DISABLE_COMPACT': '1',
        'DISABLE_AUTO_COMPACT': '1',
    }


@pytest.mark.integration
@pytest.mark.parametrize('subagent_type', ['general-purpose', 'Explore'])
async def test_workbuddy_agent_reads_through_host_model_without_context_override(
    installed_hyness: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    subagent_type: str,
) -> None:
    package = Path(installed_hyness['prefix']) / 'node_modules/@tencent/hyness-workbuddy'
    if not package.is_dir():
        pytest.skip('Install the matching WorkBuddy App and prepare its native addon')
    app_version = json.loads((package / 'package.json').read_text())['version']
    parent_system = (package / 'apps/workbuddy/prompts/system.md').read_text().strip()
    work = tmp_path / 'work'
    work.mkdir()
    token = uuid.uuid4().hex
    target = work / 'child-only.txt'
    target.write_text(token + '\n')

    def text(content: Any) -> str:
        return content if isinstance(content, str) else '\n'.join(part.get('text', '') for part in content or [])

    class AgentBridge(ScriptedBridge):
        parent_calls = 0
        child_calls = 0

        async def request(self, params, timeout):
            system = '\n\n'.join(text(m['content']) for m in params['messages'] if m['role'] == 'system')
            parent = system == parent_system
            if parent:
                self.parent_calls += 1
                action = (
                    'Agent',
                    {
                        'description': 'Read a local fixture',
                        'prompt': f'Read {target} using Read and return its exact contents. Do not modify files.',
                        'subagent_type': subagent_type,
                        'max_turns': 3,
                        'run_in_background': False,
                    },
                )
                call_number = self.parent_calls
            else:
                self.child_calls += 1
                action = ('Read', {'file_path': str(target)})
                call_number = self.child_calls
            if call_number == 1:
                self.actions.append(action)
            response = await super().request(params, timeout)
            if not parent and call_number > 1:
                observed = [text(m['content']) for m in params['messages'] if m['role'] == 'tool']
                response['choices'][0]['message'] = {'role': 'assistant', 'content': '\n'.join(observed)}
            return response

    bridge = AgentBridge('Agent', actions=[])
    monkeypatch.delenv('RUNTIME_CONFIG', raising=False)
    monkeypatch.delenv('LLM_CONFIG', raising=False)
    for key, value in {
        'AGENT_CONFIG': 'hyness_workbuddy',
        'HYNESS_PACKAGES': json.dumps(
            [f'@tencent/hyness@{installed_hyness["version"]}', f'@tencent/hyness-workbuddy@{app_version}']
        ),
        'AGENT_WORKING_DIR': str(work),
        'AGENT_MAX_TURNS': '4',
        'AGENT_TIMEOUT': '120',
    }.items():
        monkeypatch.setenv(key, value)
    loaded = OmegaConf.to_container(OmegaConf.create(load_config()), resolve=True)
    assert isinstance(loaded, dict)
    config = HynessInput.model_validate({'user_prompt': 'Delegate reading child-only.txt.', **loaded['agent_config']})
    assert 'ANTHROPIC_CONTEXT_WINDOW' not in config.extra_env
    result = await launch(config, bridge, monkeypatch)
    assert result['success'], result['stderr']
    assert bridge.parent_calls == bridge.child_calls == 2, bridge.requests
    parent_result = next(m for m in bridge.requests[-1]['messages'] if m.get('tool_call_id') == 'call-1')
    assert not parent_result.get('is_error'), parent_result
    assert token in text(parent_result['content'])
    assert target.read_text() == token + '\n'
    assert [len(r['tools']) for r in [bridge.requests[0], bridge.requests[-1]]] == [20, 20]
    assert len({r['request_meta']['session_id'] for r in bridge.requests}) == 1
    assert len({r['model'] for r in bridge.requests}) == 1
    assert list((tmp_path / 'home/.hyness/nexus').iterdir()) == []


@pytest.mark.integration
@pytest.mark.parametrize(
    ('profile', 'app', 'tools'),
    [
        ('hyness_mini_swe_agent', 'mini-swe-agent', ['bash']),
        ('hyness_swe_agent', 'swe-agent', ['bash', 'str_replace_editor', 'submit']),
    ],
)
async def test_installed_app_profile_uses_own_tools_and_submission(
    installed_hyness: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str,
    app: str,
    tools: list[str],
) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'value.txt').write_text('before\n')
    package = Path(installed_hyness['prefix']) / f'node_modules/@tencent/hyness-{app}'
    assert (package / 'package.json').is_file(), 'Install the matching Hyness App package for this profile'
    app_version = json.loads((package / 'package.json').read_text())['version']
    actions = (
        [
            ('bash', {'command': "printf 'after\\n' > value.txt"}),
            ('bash', {'command': 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'}),
        ]
        if app == 'mini-swe-agent'
        else [
            ('str_replace_editor', {'command': 'view', 'path': 'value.txt'}),
            (
                'str_replace_editor',
                {'command': 'str_replace', 'path': 'value.txt', 'old_str': 'before', 'new_str': 'after'},
            ),
            ('submit', {}),
            ('submit', {}),
        ]
    )
    bridge = ScriptedBridge(tools[0], actions=actions)
    if app == 'swe-agent':
        for args in [
            ['init', '-q'],
            ['config', 'user.name', 'Fixture'],
            ['config', 'user.email', 'fixture@example.test'],
        ]:
            subprocess.run(['git', *args], cwd=workspace, check=True, capture_output=True)
        subprocess.run(['git', 'add', 'value.txt'], cwd=workspace, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-qm', 'base'], cwd=workspace, check=True, capture_output=True)
    # Long input exercises the complete CLI path beyond the per-argument OS limit.
    prompt = 'Change value.txt to after.\n' + ('long task context\n' * 10_000)
    monkeypatch.delenv('RUNTIME_CONFIG', raising=False)
    monkeypatch.delenv('LLM_CONFIG', raising=False)
    for key, value in {
        'AGENT_CONFIG': profile,
        'HYNESS_PACKAGES': json.dumps(
            [
                f'@tencent/hyness@{installed_hyness["version"]}',
                f'@tencent/hyness-{app}@{app_version}',
            ]
        ),
        'AGENT_WORKING_DIR': str(workspace),
        'AGENT_MAX_TURNS': '5',
        'AGENT_TIMEOUT': '120',
    }.items():
        monkeypatch.setenv(key, value)
    loaded = OmegaConf.to_container(OmegaConf.create(load_config()), resolve=True)
    assert isinstance(loaded, dict)
    config = HynessInput.model_validate({'user_prompt': prompt, **loaded['agent_config']})
    assert config.app_config_overlay is None
    assert config.app_config == f'@tencent/hyness-{app}/apps/{app}/app.yaml'
    assert config.preserve_tool_schema
    result = await launch(config, bridge, monkeypatch)
    assert result['success'], result['stderr']
    assert result['exit_code'] == result['process_exit_code'] == 0
    assert result['execution_termination']['reason'] == 'hook_stopped'
    assert agent.classify_hyness_result(result).termination_reason == 'ENV_DONE'
    assert (workspace / 'value.txt').read_text() == 'after\n'
    assert len(bridge.requests) == len(actions)
    assert [t['function']['name'] for t in bridge.requests[0]['tools']] == tools
    assert prompt.rstrip() in json.dumps(bridge.requests[0]['messages'], ensure_ascii=False).replace('\\n', '\n')
    assert list((tmp_path / 'home/.hyness/nexus').iterdir()) == []


@pytest.mark.integration
@pytest.mark.parametrize('persist_session', [False, True])
async def test_session_persistence_is_explicit_and_uses_ephemeral_state(
    installed_hyness: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    persist_session: bool,
) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    captured = []
    original = runtime_setup._capture_process

    async def capture(command, cwd, env, timeout, control_id, **kwargs):
        captured.append((command, env['CLAUDE_CONFIG_DIR']))
        return await original(command, cwd, env, timeout, control_id, **kwargs)

    monkeypatch.setattr(runtime_setup, '_capture_process', capture)
    package = Path(installed_hyness['prefix']) / 'node_modules/@tencent/hyness-mini-swe-agent'
    app_version = json.loads((package / 'package.json').read_text())['version']
    config = HynessInput(
        user_prompt='Change value.txt to after.',
        working_dir=str(workspace),
        packages=[
            f'@tencent/hyness@{installed_hyness["version"]}',
            f'@tencent/hyness-mini-swe-agent@{app_version}',
        ],
        app_config='@tencent/hyness-mini-swe-agent/apps/mini-swe-agent/app.yaml',
        harness_id='mini-swe-agent',
        preserve_tool_schema=True,
        session_persistence=persist_session,
        max_turns=3,
        timeout=45,
        extra_env=process_options(),
    )
    bridge = ScriptedBridge(
        'bash',
        actions=[
            ('bash', {'command': "printf 'after\\n' > value.txt"}),
            ('bash', {'command': 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'}),
        ],
    )
    result = await launch(config, bridge, monkeypatch)
    assert result['success'], result['stderr']
    assert ('--no-session-persistence' in captured[0][0]) is not persist_session
    assert (workspace / 'value.txt').read_text() == 'after\n'
    assert not Path(captured[0][1]).exists()


@pytest.mark.integration
@pytest.mark.parametrize('config_source', ['direct', 'shared'])
async def test_app_default_and_overlay_survive_stale_process_environment(
    installed_hyness: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_source: str,
) -> None:
    engine = Path(installed_hyness['prefix']) / 'node_modules/@tencent/hyness'
    app = tmp_path / 'custom-app'
    (app / 'harnesses').mkdir(parents=True)
    (app / 'node_modules').symlink_to(Path(installed_hyness['prefix']) / 'node_modules', target_is_directory=True)
    (app / 'prompt.md').write_text('CUSTOM_APP_PROMPT: edit the file using Bash.')
    definition = {'prompt': './prompt.md', 'model': 'inherit', 'tools': ['Bash'], 'permissionMode': 'default'}
    (app / 'app.yaml').write_text(
        yaml.safe_dump(
            {
                'version': 1,
                'id': 'fixture',
                'displayName': 'Fixture',
                'mainAgent': {'name': 'fixture', 'description': 'A local test agent.', **definition},
                'harnesses': {'root': './harnesses', 'default': 'selected'},
            }
        )
    )
    (app / 'harnesses/selected.yaml').write_text(
        yaml.safe_dump(
            {
                'version': 1,
                'id': 'selected',
                'name': 'Selected',
                'description': 'The App default Harness.',
                'agent': {**definition, 'prompt': '../prompt.md'},
                'plugins': ['hyness-core'],
            }
        )
    )
    overlay = app / 'policy.yaml'
    overlay.write_text('runtime:\n  retry:\n    NonStreamingFallback: false\n')
    before = {p: p.read_bytes() for p in [app / 'app.yaml', overlay]}
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    bridge = ScriptedBridge('Bash')
    if config_source == 'shared':
        monkeypatch.delenv('RUNTIME_CONFIG', raising=False)
        monkeypatch.delenv('LLM_CONFIG', raising=False)
        for key, value in {
            'AGENT_CONFIG': 'hyness',
            'HYNESS_PACKAGES': json.dumps([f'@tencent/hyness@{installed_hyness["version"]}']),
            'HYNESS_APP_CONFIG': str(app / 'app.yaml'),
            'HYNESS_APP_CONFIG_OVERLAY': str(overlay),
            'HYNESS_HARNESS_ID': 'null',
            'AGENT_WORKING_DIR': str(workspace),
            'AGENT_MAX_TURNS': '3',
            'AGENT_TIMEOUT': '45',
        }.items():
            monkeypatch.setenv(key, value)
        loaded = OmegaConf.to_container(OmegaConf.create(load_config()), resolve=True)
        assert isinstance(loaded, dict)
        config = HynessInput.model_validate({'user_prompt': 'Change value.txt.', **loaded['agent_config']})
    else:
        config = HynessInput(
            user_prompt='Change value.txt.',
            working_dir=str(workspace),
            packages=[f'@tencent/hyness@{installed_hyness["version"]}'],
            app_config=str(app / 'app.yaml'),
            app_config_overlay=str(overlay),
            max_turns=3,
            timeout=45,
            extra_env=process_options(),
        )
    # Values from another process must not override the already selected App.
    monkeypatch.setenv('HYNESS_HARNESS_ID', 'opencc')
    monkeypatch.setenv('HYNESS_DEFAULT_APP_CONFIG', str(engine / 'app-definitions/hyness/app.yaml'))
    config.extra_env = {**(config.extra_env or {}), 'HYNESS_HARNESS_ID': 'opencc'}
    assert config.harness_id is None
    result = await launch(config, bridge, monkeypatch)
    assert result['success'], result['stderr']
    assert len(bridge.requests) == 2
    assert [t['function']['name'] for t in bridge.requests[0]['tools']] == ['Bash']
    assert 'CUSTOM_APP_PROMPT' in json.dumps(bridge.requests[0]['messages'])
    assert (workspace / 'value.txt').read_text() == 'after\n'
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.integration
async def test_missing_app_package_fails_before_model_call(
    installed_hyness: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    bridge = ScriptedBridge('Bash')
    config = HynessInput(
        user_prompt='Change value.txt.',
        working_dir=str(workspace),
        packages=[f'@tencent/hyness@{installed_hyness["version"]}'],
        app_config='@example/missing-app/apps/missing/app.yaml',
        timeout=45,
        extra_env=process_options(),
    )
    result = await launch(config, bridge, monkeypatch)
    assert not result['success']
    assert result['process_exit_code'] is None
    assert result['failure_stage'] == 'install'
    assert 'Hyness App configuration is missing' in result['stderr']
    assert not bridge.requests
    assert agent.classify_hyness_result(result).failure_kind == 'infrastructure'


@pytest.mark.integration
@pytest.mark.parametrize('config_source', ['direct', 'shared'])
async def test_opencc_defaults_and_shared_tool_restrictions_run(
    installed_hyness: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_source: str,
) -> None:
    package = Path(installed_hyness['prefix']) / 'node_modules/@tencent/hyness-opencc'
    if not (package / 'package.json').is_file():
        raise pytest.skip.Exception('install the matching OpenCC package to verify legacy defaults')
    app_version = json.loads((package / 'package.json').read_text())['version']
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    bridge = ScriptedBridge('Bash')
    packages = [f'@tencent/hyness@{installed_hyness["version"]}', f'@tencent/hyness-opencc@{app_version}']
    denied = {'WebSearch', 'WebFetch', 'NotebookEdit', 'AskUserQuestion', 'EnterPlanMode', 'ExitPlanMode'}
    if config_source == 'shared':
        monkeypatch.delenv('RUNTIME_CONFIG', raising=False)
        monkeypatch.delenv('LLM_CONFIG', raising=False)
        for key, value in {
            'AGENT_CONFIG': 'hyness_opencc',
            'HYNESS_PACKAGES': json.dumps(packages),
            'AGENT_WORKING_DIR': str(workspace),
            'AGENT_MAX_TURNS': '3',
            'AGENT_TIMEOUT': '45',
        }.items():
            monkeypatch.setenv(key, value)
        loaded = OmegaConf.to_container(OmegaConf.create(load_config()), resolve=True)
        assert isinstance(loaded, dict)
        config = HynessInput.model_validate({'user_prompt': 'Change value.txt.', **loaded['agent_config']})
        args = config.extra_cli_args
        assert denied == set(args[args.index('--disallowedTools') + 1].split(','))
    else:
        config = HynessInput(
            user_prompt='Change value.txt.',
            working_dir=str(workspace),
            packages=packages,
            max_turns=3,
            timeout=45,
            extra_env=process_options(),
        )
    assert config.harness_id == 'opencc'
    assert config.app_config_overlay is not None
    result = await launch(config, bridge, monkeypatch)
    assert result['success'], result['stderr']
    assert result['exit_code'] == result['process_exit_code'] == 0
    assert result['execution_termination']['reason'] == 'completed'
    assert agent.classify_hyness_result(result).termination_reason == 'ENV_DONE'
    assert len(bridge.requests) == 2
    assert (workspace / 'value.txt').read_text() == 'after\n'
    if config_source == 'shared':
        for request in bridge.requests:
            names = {tool['function']['name'] for tool in request['tools']}
            assert 'Bash' in names
            assert not names & denied
    event = json.loads(result['process_output'].splitlines()[-1])
    assert event['protocol'] == 'hyness.execution-result'
    assert event['schemaVersion'] == 1
    assert set(event['usageBreakdown']) == {'streaming', 'nonStreaming'}
    mapped = json.loads(result['output'].splitlines()[-1])
    assert mapped['usage'] == event['usageBreakdown']['streaming']
