import json
import sys

from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.agents.codex import runtime_setup
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class _RuntimeControl:
    sio = object()


class _ResponsesProxy:
    def __init__(self, captured: dict[str, Any], **kwargs: Any) -> None:
        captured['proxy_kwargs'] = kwargs

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


INLINE_MODEL_CATALOG = {'models': [{'slug': 'gpt-5', 'apply_patch_tool_type': 'freeform'}]}


async def test_tool_reference_override_rejects_a_custom_request_converter() -> None:
    with pytest.raises(ValueError, match='requires the default Responses converter'):
        await runtime_setup.setup_and_run_codex(
            task='test capability ownership',
            supports_tool_references=False,
            responses_to_openai_converter=lambda request: request,
        )


async def test_inline_schema_format_rejects_a_custom_request_converter() -> None:
    with pytest.raises(ValueError, match='tool_search_output_format requires the default Responses converter'):
        await runtime_setup.setup_and_run_codex(
            task='test inline schemas',
            tool_search_output_format='tool_search_output',
            responses_to_openai_converter=lambda request: request,
        )


async def test_invalid_tool_search_output_format_raises_before_install() -> None:
    with pytest.raises(ValueError, match='choose tool_reference or tool_search_output'):
        await runtime_setup.setup_and_run_codex(task='test invalid format', tool_search_output_format='invalid')


@pytest.mark.parametrize('supports_tool_references', [None, False])
async def test_inline_schema_format_requires_deferred_loading(supports_tool_references: bool | None) -> None:
    with pytest.raises(ValueError, match='requires supports_tool_references=True'):
        await runtime_setup.setup_and_run_codex(
            task='test inline schemas',
            tool_search_output_format='tool_search_output',
            supports_tool_references=supports_tool_references,
        )


async def test_load_bundled_model_catalog_uses_agent_execution_transport(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class _Transport:
        async def exec_as_agent(self, command: list[str], **kwargs: Any) -> RunCommandResponse:
            captured['command'] = command
            captured['kwargs'] = kwargs
            return RunCommandResponse(
                status=CommandRunStatus.Finished,
                return_code=0,
                stdout='{"models": [{"slug": "gpt-5"}]}',
            )

    codex_home = tmp_path / 'codex-home'
    catalog = await runtime_setup._load_bundled_model_catalog(
        command_argv=['/agent/codex'],
        codex_home=codex_home,
        working_dir='/workspace',
        transport=_Transport(),
        extra_env={'NEXUS_TEST': '1'},
    )

    assert catalog == {'models': [{'slug': 'gpt-5'}]}
    assert captured['command'] == ['/agent/codex', 'debug', 'models', '--bundled']
    assert captured['kwargs']['working_dir'] == '/workspace'
    assert captured['kwargs']['timeout'] == 60
    assert captured['kwargs']['env']['NEXUS_TEST'] == '1'
    assert captured['kwargs']['env']['CODEX_HOME'] == str(codex_home)


def test_model_catalog_overrides_clone_unknown_model_entry() -> None:
    catalog = {
        'models': [
            {
                'slug': 'gpt-5',
                'priority': 3,
                'multi_agent_version': 'v2',
                'use_responses_lite': True,
            }
        ]
    }

    updated = runtime_setup._override_model_catalog_entry(
        catalog,
        'deepseek-flash',
        {'multi_agent_version': 'v1', 'use_responses_lite': False},
    )

    assert updated['models'][0] == catalog['models'][0]
    assert updated['models'][1] == {
        'slug': 'deepseek-flash',
        'priority': 3,
        'multi_agent_version': 'v1',
        'use_responses_lite': False,
    }


def test_model_catalog_overrides_update_known_model_entry() -> None:
    catalog = {'models': [{'slug': 'deepseek-flash', 'multi_agent_version': 'v2'}]}

    updated = runtime_setup._override_model_catalog_entry(catalog, 'deepseek-flash', {'multi_agent_version': 'v1'})

    assert updated == {'models': [{'slug': 'deepseek-flash', 'multi_agent_version': 'v1'}]}


def test_shared_config_builder_enables_interactive_input_only_when_requested(tmp_path: Path) -> None:
    base = {
        'codex_home': tmp_path,
        'proxy_base_url': 'http://127.0.0.1:8099/v1',
        'version': '0.155.1',
    }
    batch = '\n'.join(runtime_setup.build_codex_config_lines(**base))
    interactive = '\n'.join(runtime_setup.build_codex_config_lines(**base, default_mode_request_user_input=True))
    assert 'features.default_mode_request_user_input' not in batch
    assert 'features.default_mode_request_user_input = true' in interactive


def test_shared_config_builder_marks_only_the_interactive_working_dir_as_trusted(tmp_path: Path) -> None:
    working_dir = tmp_path / '工作目录'
    config = runtime_setup.build_codex_config_lines(
        codex_home=tmp_path,
        proxy_base_url='http://127.0.0.1:8099/v1',
        version='0.155.1',
        trusted_project=str(working_dir),
    )

    assert config[-3:] == [
        '',
        f'[projects.{json.dumps(str(working_dir))}]',
        'trust_level = "trusted"',
    ]


def test_shared_config_builder_places_extra_root_settings_before_tables(tmp_path: Path) -> None:
    config = runtime_setup.build_codex_config_lines(
        codex_home=tmp_path,
        proxy_base_url='http://127.0.0.1:8099/v1',
        version='0.155.1',
        extra_config_toml='model_reasoning_effort = "high"',
    )

    assert config.index('model_reasoning_effort = "high"') < config.index('[model_providers.nexus]')


def test_shared_config_builder_emits_model_instructions_file_as_a_root_key(tmp_path: Path) -> None:
    config = runtime_setup.build_codex_config_lines(
        codex_home=tmp_path,
        proxy_base_url='http://127.0.0.1:8099/v1',
        version='0.155.1',
        extra_config_toml='model_reasoning_effort = "high"',
        model_instructions_file='/tmp/ppt_sysp_staged.txt',
    )
    text = '\n'.join(config)

    assert 'model_instructions_file = "/tmp/ppt_sysp_staged.txt"' in config
    assert config.index('model_instructions_file = "/tmp/ppt_sysp_staged.txt"') < config.index(
        '[model_providers.nexus]'
    )
    assert 'model_reasoning_effort = "high"' in text
    assert tomllib.loads(text)


def test_shared_config_builder_replaces_a_same_named_extra_config_key(tmp_path: Path) -> None:
    config = runtime_setup.build_codex_config_lines(
        codex_home=tmp_path,
        proxy_base_url='http://127.0.0.1:8099/v1',
        version='0.155.1',
        extra_config_toml='model_instructions_file = "/tmp/stale.txt"\nmodel_reasoning_effort = "high"',
        model_instructions_file='/tmp/ppt_sysp_staged.txt',
    )
    text = '\n'.join(config)

    # Two identical keys -- however equal their values -- make the TOML invalid,
    # so the staged file's line must be the only authority for the key.
    assert 'model_instructions_file = "/tmp/ppt_sysp_staged.txt"' in config
    assert '/tmp/stale.txt' not in text
    assert 'model_reasoning_effort = "high"' in text
    assert tomllib.loads(text)


@pytest.mark.parametrize(
    ('stale_key_line', 'leaks'),
    [
        ('"model_instructions_file" = "/tmp/stale.txt"', ('/tmp/stale.txt',)),
        ("'model_instructions_file' = '/tmp/stale.txt'", ('/tmp/stale.txt',)),
        ('model_instructions_file = """\n/tmp/stale.txt\n"""', ('/tmp/stale.txt',)),
        ("model_instructions_file = '''\n/tmp/stale.txt\n'''", ('/tmp/stale.txt',)),
    ],
    ids=('quoted-key', 'single-quoted-key', 'multiline-basic-string', 'multiline-literal-string'),
)
def test_shared_config_builder_replaces_quoted_and_multiline_stale_keys(
    tmp_path: Path, stale_key_line: str, leaks: tuple[str, ...]
) -> None:
    """Review follow-up: the replacement must also consume quoted keys
    (double- and single-quoted, both legal TOML) and multi-line string values,
    whose continuation lines a line-only pattern would leave behind as stray,
    invalid TOML."""
    extra = f'{stale_key_line}\nmodel_reasoning_effort = "high"'
    config = runtime_setup.build_codex_config_lines(
        codex_home=tmp_path,
        proxy_base_url='http://127.0.0.1:8099/v1',
        version='0.155.1',
        extra_config_toml=extra,
        model_instructions_file='/tmp/ppt_sysp_staged.txt',
    )
    text = '\n'.join(config)

    assert 'model_instructions_file = "/tmp/ppt_sysp_staged.txt"' in config
    for leak in leaks:
        assert leak not in text
    assert 'model_reasoning_effort = "high"' in text
    assert tomllib.loads(text)


def test_shared_config_builder_keeps_example_line_inside_multiline_string(tmp_path: Path) -> None:
    """Review follow-up (round 3): a line-anchored strip cannot tell a root
    key from a continuation line of a multi-line string value. Same-named
    example lines inside a multi-line developer_instructions value must
    survive; only the root-level key is replaced."""
    extra = (
        'model_instructions_file = "/tmp/stale.txt"\n'
        'developer_instructions = """\n'
        'Example: model_instructions_file = "/tmp/example.txt"\n'
        '"""\n'
        'model_reasoning_effort = "high"'
    )
    config = runtime_setup.build_codex_config_lines(
        codex_home=tmp_path,
        proxy_base_url='http://127.0.0.1:8099/v1',
        version='0.155.1',
        extra_config_toml=extra,
        model_instructions_file='/tmp/ppt_sysp_staged.txt',
    )
    text = '\n'.join(config)

    assert 'model_instructions_file = "/tmp/ppt_sysp_staged.txt"' in config
    assert '/tmp/stale.txt' not in text
    # The example line is string content, not a root key: it must survive.
    assert 'model_instructions_file = "/tmp/example.txt"' in text
    assert 'model_reasoning_effort = "high"' in text
    assert tomllib.loads(text)


@pytest.mark.parametrize(
    (
        'version_kwargs',
        'model',
        'experimental_use_freeform_apply_patch',
        'model_catalog_json',
        'expected_legacy_apply_patch_config',
    ),
    [
        ({}, None, None, None, True),
        ({'version': None}, None, None, None, True),
        ({'version': 'latest'}, None, None, None, False),
        ({'version': '0.114.0'}, None, None, None, True),
        ({'version': '0.114.0'}, None, False, None, False),
        ({'version': '0.147.0'}, 'gpt-5', True, None, True),
        ({'version': '0.147.0'}, 'gpt-5', None, None, False),
        ({'version': '0.147.0'}, 'gpt-5', None, INLINE_MODEL_CATALOG, False),
    ],
)
@pytest.mark.parametrize(
    ('supports_tool_references', 'tool_search_output_format'),
    [(None, 'tool_reference'), (False, 'tool_reference'), (True, 'tool_reference'), (True, 'tool_search_output')],
)
async def test_codex_runtime_config_uses_active_command_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    version_kwargs: dict[str, str | None],
    model: str | None,
    experimental_use_freeform_apply_patch: bool | None,
    model_catalog_json: dict[str, Any] | None,
    expected_legacy_apply_patch_config: bool,
    supports_tool_references: bool | None,
    tool_search_output_format: str,
) -> None:
    js_entry = tmp_path / 'codex.js'
    js_entry.write_text('')
    captured: dict[str, Any] = {}

    async def install_codex_cli(**kwargs: Any) -> Path:
        captured['install_kwargs'] = kwargs
        binary = 'codex.cmd' if runtime_setup.is_windows() else 'codex'
        return Path(kwargs['install_dir']) / 'node_modules' / '.bin' / binary

    async def run_command(request: Any) -> RunCommandResponse:
        captured['request'] = request
        return RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='codex-cli 0.158.0\n',
            stderr='',
        )

    def responses_proxy(**kwargs: Any) -> _ResponsesProxy:
        return _ResponsesProxy(captured, **kwargs)

    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    monkeypatch.setattr('nexus.extensions.agents.codex.installer.install_codex_cli', install_codex_cli)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: _RuntimeControl())
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', run_command)
    monkeypatch.setattr('nexus.runtime.server.services.responses_proxy.ResponsesProxy', responses_proxy)
    monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 12345)
    monkeypatch.setattr(runtime_setup, 'find_nexus_runtime_binary', lambda _name: '/bin/node')
    monkeypatch.setattr(runtime_setup, 'resolve_npm_bin_entry', lambda *_args: js_entry)
    monkeypatch.setattr(runtime_setup, 'get_current_span_id', lambda: 'codex-runtime-timeout', raising=False)

    await runtime_setup.setup_and_run_codex(
        task='inspect model configuration',
        working_dir=str(tmp_path),
        timeout=10,
        **version_kwargs,
        model=model,
        experimental_use_freeform_apply_patch=experimental_use_freeform_apply_patch,
        model_catalog_json=model_catalog_json,
        supports_tool_references=supports_tool_references,
        tool_search_output_format=tool_search_output_format,
    )

    assert captured['request'].timeout == 10
    assert captured['request'].timeout_control_id == 'codex-runtime-timeout'
    assert captured['proxy_kwargs']['request_timeout'] is None
    expected_version = version_kwargs.get('version') or '0.114.0'
    assert captured['install_kwargs']['version'] == expected_version
    assert captured['proxy_kwargs']['codex_version'] == expected_version
    if supports_tool_references is None:
        assert 'responses_to_openai_converter' not in captured['proxy_kwargs']
    else:
        converter = captured['proxy_kwargs']['responses_to_openai_converter']
        params = converter({
            'model': 'model-alias',
            'tools': [{'type': 'function', 'name': 'calculate', 'defer_loading': True}],
        })
        assert params['tools'][0]['function'].get('defer_loading', False) is supports_tool_references
        assert converter.keywords['tool_search_output_format'] == tool_search_output_format, (
            'Runtime must forward the selected tool search result format'
        )
    config_paths = list((tmp_path / runtime_setup.CODEX_HOME_DIR_NAME).glob('*/config.toml'))
    assert len(config_paths) == 1
    config = config_paths[0].read_text()
    tomllib.loads(config)
    has_legacy_apply_patch_config = 'experimental_use_freeform_apply_patch = true' in config
    assert has_legacy_apply_patch_config is expected_legacy_apply_patch_config

    catalog_paths = list((tmp_path / runtime_setup.CODEX_HOME_DIR_NAME).glob('*/model_catalog.json'))
    assert bool(catalog_paths) is (model_catalog_json is not None)
    assert 'model_instructions_file' not in config
    if model_catalog_json is None:
        assert 'model_catalog_json' not in config
        return

    catalog_path = catalog_paths[0]
    assert config.count('model_catalog_json = ') == 1
    assert f'model_catalog_json = {json.dumps(str(catalog_path))}' in config
    assert json.loads(catalog_path.read_text()) == model_catalog_json
    # The top-level key must precede the provider table or TOML nests it under it.
    assert config.index('model_catalog_json') < config.index('[model_providers.nexus]')


async def test_codex_exec_keeps_session_files_for_multi_agent_v2_forks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    js_entry = tmp_path / 'codex.js'
    js_entry.write_text('')
    captured: dict[str, Any] = {}

    async def install_codex_cli(**_kwargs: Any) -> None:
        pass

    async def run_command(request: Any) -> RunCommandResponse:
        captured['request'] = request
        return RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='',
            stderr='',
        )

    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    monkeypatch.setattr('nexus.extensions.agents.codex.installer.install_codex_cli', install_codex_cli)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: _RuntimeControl())
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', run_command)
    monkeypatch.setattr(
        'nexus.runtime.server.services.responses_proxy.ResponsesProxy', lambda **kw: _ResponsesProxy(captured, **kw)
    )
    monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 12345)
    monkeypatch.setattr(runtime_setup, 'find_nexus_runtime_binary', lambda _name: '/bin/node')
    monkeypatch.setattr(runtime_setup, 'resolve_npm_bin_entry', lambda *_args: js_entry)
    monkeypatch.setattr(runtime_setup, 'get_current_span_id', lambda: 'codex-ephemeral-guard')

    await runtime_setup.setup_and_run_codex(
        task='spawn a subagent',
        working_dir=str(tmp_path),
        timeout=10,
        version='0.154.0',
    )

    assert '--ephemeral' not in captured['request'].command


@pytest.mark.parametrize(
    ('version', 'expected'),
    [
        ('0.130.0', True),
        ('0.131.0-alpha.14', True),
        ('0.131.0-alpha.15', False),
        ('0.131.0', False),
        ('0.147.0', False),
        ('latest', False),
    ],
)
def test_legacy_apply_patch_default_follows_codex_version(version: str, expected: bool) -> None:
    assert runtime_setup._should_enable_legacy_freeform_apply_patch(version, None) is expected


def test_unknown_codex_version_preserves_legacy_default(caplog: pytest.LogCaptureFixture) -> None:
    assert runtime_setup._should_enable_legacy_freeform_apply_patch('workspace-build', None) is True
    assert 'Cannot infer legacy apply_patch support' in caplog.text


async def test_windows_binary_runtime_uses_utf8_home_mcp_and_preserves_command_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    from nexus.extensions.agents import installed_common

    captured: dict[str, Any] = {}
    binary = tmp_path / 'native cli' / 'codex.exe'
    binary.parent.mkdir()
    binary.touch()

    async def install_from_url(url: str, install_dir: str, checksum: str | None) -> Path:
        captured['install'] = (url, checksum)
        return binary

    async def run_command(request: Any) -> RunCommandResponse:
        captured['request'] = request
        home = Path(request.extra_env['CODEX_HOME'])
        captured['config'] = tomllib.loads((home / 'config.toml').read_text(encoding='utf-8'))
        (home / 'last_message.txt').write_text('中文验证完成', encoding='utf-8')
        return RunCommandResponse(
            status=CommandRunStatus.Finished, execution_time=0.1, return_code=0, stdout='{"type":"turn.completed"}\n'
        )

    monkeypatch.setattr(runtime_setup, 'is_windows', lambda: True)
    monkeypatch.setattr(installed_common, 'is_windows', lambda: True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.delenv('HOME', raising=False)
    monkeypatch.setattr('nexus.extensions.agents.codex.installer.install_codex_from_url', install_from_url)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: _RuntimeControl())
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', run_command)
    monkeypatch.setattr(
        'nexus.runtime.server.services.responses_proxy.ResponsesProxy', lambda **kw: _ResponsesProxy(captured, **kw)
    )
    monkeypatch.setattr(runtime_setup, 'get_current_span_id', lambda: 'windows-budget')
    monkeypatch.setattr(
        runtime_setup, 'find_nexus_runtime_binary', lambda _name: pytest.fail('URL install must not need Node')
    )
    result = await runtime_setup.setup_and_run_codex(
        task="中文任务\n第二行 'quote'",
        working_dir=str(tmp_path / 'work'),
        version='0.146.1',
        binary_url='https://example.invalid/codex.exe',
        binary_checksum='abc',
        extra_env={'MCP_TEST_VALUE': '中文'},
        mcp_servers=[
            {
                'name': 'test.with.dot',
                'command': 'uv',
                'args': ['run', '中文.py'],
                'env': {'VALUE': '中文'},
                'startup_timeout_sec': 300,
                'required': True,
                'enabled_tools': ['echo'],
            }
        ],
    )
    request = captured['request']
    assert request.command.startswith('$codexProcess = Start-Process')
    assert "''quote''" in request.command
    assert request.command.endswith('; exit $codexProcess.ExitCode')
    assert request.timeout_control_id == 'windows-budget'
    assert request.extra_env['HOME'] == str(tmp_path)
    assert request.extra_env['MCP_TEST_VALUE'] == '中文'
    server = captured['config']['mcp_servers']['test.with.dot']
    assert server['env'] == {'VALUE': '中文'}
    assert server['required'] is True
    assert server['args'] == ['run', '中文.py']
    assert result['success'] is True
    assert result['output'] == '中文验证完成'
    assert result['json_events_stdout'] == '{"type":"turn.completed"}\n'


@pytest.mark.parametrize('server', [{'name': 'missing-command'}, {'name': 'a', 'command': 'uv', 'args': 'invalid'}])
def test_mcp_config_rejects_invalid_stdio_servers(server: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match='requires non-empty|args must be a list'):
        runtime_setup._mcp_config_lines([server])


@pytest.mark.skipif(not runtime_setup.is_windows(), reason='Windows PowerShell argument marshalling')
async def test_windows_codex_command_preserves_arguments_and_exit_code(tmp_path: Path) -> None:
    import sys

    from nexus.runtime.executors.command import RunCommandRequest, run_command

    script = tmp_path / 'argv probe.py'
    script.write_text('import json,sys; print(json.dumps(sys.argv[1:])); sys.exit(7)', encoding='utf-8')
    arguments = ['中文\n第二行', 'a"b', "a'b", 'model_reasoning_effort="high"', 'path with spaces\\', '']
    result = await run_command(
        RunCommandRequest(
            command=runtime_setup._build_codex_command([sys.executable, str(script), *arguments]),
            timeout=30,
        )
    )
    assert result.return_code == 7
    assert json.loads(result.stdout) == arguments
