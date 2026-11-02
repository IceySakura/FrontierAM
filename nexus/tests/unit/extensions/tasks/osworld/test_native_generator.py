"""OSWorld native generator regressions."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core import agent_registry, openai_client_registry
from nexus.extensions.tasks.os_world.generators import generator_native as generator
from nexus.extensions.tasks.os_world.generators.run_parts import osworld_native
from nexus.extensions.tasks.os_world.task import OSWorldTaskInput
from nexus.runner.batch.loader import GeneratorLoader
from nexus.runner.compose import compile_single, parse_ordered_parts
from nexus.runner.single import RunInput, _build_input


MODES = sorted(generator.SUPPORTED)


@pytest.mark.parametrize(('family', 'harness'), MODES)
def test_cold_start_imports_yaml_and_profile(family, harness, tmp_path):
    """Run each mode in a fresh interpreter against the actual loaded package."""
    import nexus

    package_root = Path(nexus.__file__).resolve().parent.parent
    env = dict(os.environ)
    env['PYTHONPATH'] = str(package_root)
    code = r"""
import importlib
import json
import sys
from pathlib import Path

import nexus
from nexus.core import OpenAIConfig, agent_registry, openai_client_registry
from nexus.extensions.tasks.os_world.generators import generator_native
from nexus.runner.compose import compile_single, parse_ordered_parts
from nexus.runner.single import RunInput, _build_input, _get_input_type

family, harness, package_root = sys.argv[1:]
assert Path(nexus.__file__).resolve().is_relative_to(Path(package_root).resolve())
configs = Path(generator_native.__file__).resolve().parents[1] / 'configs'
saved = RunInput.from_file(configs / f'osworld_{family}_{harness}.yaml')
generated = RunInput.from_dict(generator_native.single_config(
    task_id='task-b', model_family=family, harness=harness
).model_dump())
composed = compile_single(parse_ordered_parts([
    '--use', f'osworld_native:model_family={family},harness={harness},task_id=task-b'
]))
assert composed.model_dump() == generated.model_dump()
for config in (saved, generated):
    for name in config.imports:
        importlib.import_module(name)
    block = config.openai_clients['main']
    client_type = openai_client_registry.get(block['name'])
    OpenAIConfig(**block['config'])
    assert callable(client_type)
    agent = agent_registry.get(config.agent_name)
    inp = _build_input(_get_input_type(agent), {
        **config.agent_config,
        'user_prompt': 'cold-start probe',
        'task_context': {
            'osworld_task_id': config.task_config['task_id'],
            'osworld_domain': config.task_config['domain'],
        },
    })
    assert config.runtime_provider_name == 'ags'
    assert config.task_files is not None
    if harness != 'bare':
        package = 'claude_code' if harness == 'cc' else 'codex'
        build = importlib.import_module(
            f'nexus.extensions.agents.{package}.native_bridge.adapters.base'
        ).build_execution_profile
        profile = build(inp.execution_profile, inp.execution_options, inp.task_context, inp.user_prompt)
        if harness == 'cc':
            profile = inp.build_execution_profile()
            assert profile.options.timeout == config.agent_config['timeout']
            assert profile.options.max_turns == config.agent_config['max_turns']
        guide = profile.options.guide_text
        if harness == 'cc' and guide is None:
            from nexus.extensions.tasks.os_world.execution.claude_code import _guide_text
            guide = _guide_text(profile.options.guide_version or '')
        assert guide
        options = profile.bridge_options(None)
        assert 'bridge_extension' not in options
        assert callable(profile.run)
        assert 'hosted_tool_handler' not in options
        if harness == 'codex':
            assert json.loads(profile.options.model_catalog_json)['models']
            if family == 'gpt':
                from types import SimpleNamespace
                native_client = SimpleNamespace(create_responses=lambda: None)
                assert profile.bridge_options(native_client)['api_protocol'] == 'osworld-responses'
        else:
            assert profile.options.max_turns > 0
print(json.dumps({'mode': f'{family}/{harness}', 'package': nexus.__file__, 'status': 'passed'}))
"""
    result = subprocess.run(
        [sys.executable, '-c', code, family, harness, str(package_root)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "passed"' in result.stdout


def test_content_policy_error_contract():
    from nexus.core.openai_client import CONTENT_POLICY_VIOLATION_CODE, ContentPolicyViolationError
    from nexus.runtime.bridge.adapters.ags_osworld import _terminal_error_code
    from nexus.runtime.server.services.rsio_llm_bridge import LLMRequestError, bridge_error_payload

    original = RuntimeError('provider refusal')
    error = ContentPolicyViolationError('content refused', original_error=original)
    assert error.message == str(error) == 'content refused'
    assert error.original_error is original
    assert _terminal_error_code(error) == CONTENT_POLICY_VIOLATION_CODE == 'content_policy_violation'
    status, payload = bridge_error_payload(LLMRequestError(str(error), error_code=_terminal_error_code(error)))
    assert status == 400
    assert payload['error']['code'] == CONTENT_POLICY_VIOLATION_CODE


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(
            (
                'OSWORLD_',
                'OSWV1_',
                'NEXUS_',
                'CC_',
                'CODEX_',
                'HUNYUAN_',
                'AGS_',
                'GPT_',
                'ANTHROPIC_',
                'AIHUB_',
                'CUA_',
            )
        ):
            monkeypatch.delenv(name)
    manifest = tmp_path / 'test.jsonl'
    records = [
        {
            'id': task_id,
            'metadata': {'domain': 'chrome', 'instruction': 'Do a task'},
            'files': {
                'type': 'zip',
                'path': f'blobs/{task_id}.zip',
                'size': 100,
                'remote': {'namespace': 'test', 'dataset': 'osworld', 'version': 'v1', 'split': 'test'},
            },
        }
        for task_id in ('task-a', 'task-b')
    ]
    manifest.write_text('\n'.join(json.dumps(r) for r in records))
    proxy = tmp_path / 'proxy.json'
    proxy.write_text('{}')
    env = {
        'OSWORLD_DATASET_URI': 'test/osworld@v1',
        'OSWORLD_DATASET_MANIFEST': str(manifest),
        'OSWORLD_VALIDATE_EVALUATOR_HOST_DEPS': '0',
        'OSWORLD_AGS_PROVISION_MODE': 'control_plane',
        'AGS_OSWORLD_TOOL_ID': 'a-user-selected-tool',
        'AGS_SECRET_ID': 'secret-id-sentinel',
        'AGS_SECRET_KEY': 'secret-key-sentinel',
        'WEDATA_KEY': '{"id":"fixture","subject":"fixture","key":"wedata-sentinel"}',
        'NEXUS_MODEL': 'arbitrary-routing-name',
        'NEXUS_BASE_URL': 'http://model-fixture/v1',
        'NEXUS_API_KEY': 'model-key-sentinel',
        'HUNYUAN_MODEL': 'hunyuan-service',
        'HUNYUAN_BASE_URL': 'http://hunyuan-fixture/v1',
        'HUNYUAN_API_KEY': 'hunyuan-key-sentinel',
        'OSWORLD_SESSION_NAMESPACE': str(tmp_path / 'run'),
        'OSWORLD_ENABLE_PROXY': 'true',
        'OSWORLD_PROXY_CONFIG_FILE': str(proxy),
        'NEXUS_OSWORLD_CACHE_DIR': str(tmp_path / 'cache'),
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


def choose(monkeypatch, family, harness):
    monkeypatch.setenv('OSWORLD_MODEL_FAMILY', family)
    monkeypatch.setenv('OSWORLD_HARNESS', harness)


@pytest.mark.parametrize(('family', 'harness'), MODES)
def test_generator_loader_single_and_compose_are_identical(monkeypatch, family, harness):
    choose(monkeypatch, family, harness)
    before = dict(os.environ)
    loader = GeneratorLoader(Path(generator.__file__))
    items = list(loader.get_generate_fn()())
    assert len(items) == generator.get_total_count() == 2
    assert generator.get_batch_metadata()['harness'] == harness
    single = generator.single_config(task_id='task-b')
    assert single.model_dump(exclude={'run_id'}) == items[1].config.model_dump(exclude={'run_id'})
    compiled = compile_single(
        parse_ordered_parts(
            [
                '--use',
                f'osworld_native:model_family={family},harness={harness},task_id=task-b',
            ]
        )
    )
    expected = RunInput.from_dict(single.model_dump())
    assert compiled.model_dump() == expected.model_dump()
    assert dict(os.environ) == before
    assert 'OSWV1_AIHUB_RESPONSES_KEY' not in os.environ
    for item in items:
        cfg = item.config
        assert cfg.task_files
        assert not cfg.task_config.get('task_root')
        assert cfg.task_config['enable_proxy'] is True
        assert cfg.task_config['run_setup']
        assert cfg.task_config['run_evaluator']
        assert cfg.task_config['unsupported_evaluator_policy'] == 'fail'
        assert 'traj_dump_dir' not in cfg.task_config
        assert cfg.runner.hooks['run_dir'].name == 'osworld_run_dir'
        dumped = json.dumps(item.model_dump())
        for secret in (
            'secret-id-sentinel',
            'secret-key-sentinel',
            'wedata-sentinel',
            'model-key-sentinel',
            'hunyuan-key-sentinel',
        ):
            assert secret not in dumped


def test_task_and_runtime_do_not_depend_on_model_or_harness(monkeypatch):
    configs = []
    for family, harness in MODES:
        choose(monkeypatch, family, harness)
        configs.append(next(generator.generate_configs()).config)
    for cfg in configs[1:]:
        assert cfg.task_config == configs[0].task_config
        assert cfg.runtime_configs == configs[0].runtime_configs
        assert cfg.runtime_provider_config == configs[0].runtime_provider_config


@pytest.mark.parametrize(('family', 'harness'), MODES)
def test_result_namespace_is_optional_in_all_native_modes(monkeypatch, family, harness):
    choose(monkeypatch, family, harness)
    monkeypatch.delenv('OSWORLD_SESSION_NAMESPACE')
    monkeypatch.delenv('CUA_TRAJ_DUMP_DIR', raising=False)
    before = dict(os.environ)
    item = next(generator.generate_configs())
    assert 'traj_dump_dir' not in item.config.task_config
    assert 'hooks' not in item.config.task_config
    assert set(item.config.runner.hooks) == {'output_contract'}
    assert item.config.runner.hooks['output_contract'].name == 'osworld_output_contract'
    assert dict(os.environ) == before


@pytest.mark.parametrize(('family', 'harness'), MODES)
def test_complete_yaml_export_and_plain_single_cli_match_batch(monkeypatch, family, harness, tmp_path):
    from nexus.cli.commands.run import register_run_single_parser, run_single
    from nexus.extensions.tasks.os_world.scripts.export_single import main

    choose(monkeypatch, family, harness)
    target = tmp_path / 'single.yaml'
    assert main(['--model-family', family, '--harness', harness, '--task-id', 'task-b', '-o', str(target)]) == 0
    assert 'compose:' not in target.read_text()
    assert 'secret-key-sentinel' not in target.read_text()
    expected = RunInput.from_dict(generator.single_config(task_id='task-b').model_dump())
    assert RunInput.from_file(target).model_dump() == expected.model_dump()
    execute = AsyncMock(return_value=SimpleNamespace(task_output={}))
    monkeypatch.setattr(importlib.import_module('nexus.runner.single'), 'run', execute)
    parser = argparse.ArgumentParser()
    register_run_single_parser(parser.add_subparsers(dest='command'))
    args = parser.parse_args(['run-single', str(target)])
    assert run_single(args) == 0
    execute.assert_awaited_once_with(expected)


@pytest.mark.parametrize(('family', 'harness'), MODES)
def test_every_task_and_agent_field_survives_the_runner_filter(monkeypatch, family, harness):
    from nexus.runner.single import _get_input_type

    choose(monkeypatch, family, harness)
    cfg = RunInput.from_dict(next(generator.generate_configs()).config.model_dump())
    assert set(cfg.task_config) <= set(OSWorldTaskInput.model_fields)
    for module in cfg.imports:
        importlib.import_module(module)
    agent_input = _get_input_type(agent_registry.get(cfg.agent_name))
    assert set(cfg.agent_config) <= set(agent_input.model_fields)
    received = _build_input(agent_input, {**cfg.agent_config, 'user_prompt': 'test'})
    if family == 'claude' and harness == 'bare':
        assert received.transport == 'client'
        assert received.client_name == 'main'
        assert received.model is None
    if harness == 'codex':
        assert received.execution_options['model'] == 'gpt-6-astra'
        assert cfg.agent_name == 'codex_native_bridge'
        assert received.execution_options['reasoning_effort'] == 'none'


@pytest.mark.parametrize(('family', 'harness'), MODES)
def test_selected_client_can_be_instantiated_without_network(monkeypatch, family, harness):
    from nexus.core import OpenAIConfig

    choose(monkeypatch, family, harness)
    cfg = RunInput.from_dict(next(generator.generate_configs()).config.model_dump())
    block = cfg.openai_clients['main']
    if block['name'] == 'bridge_litellm':
        pytest.importorskip('nexus_bridge')
    for module in cfg.imports:
        importlib.import_module(module)
    client = openai_client_registry.get(block['name'])(OpenAIConfig(**block['config']))
    expected = 'hunyuan-service' if family == 'hunyuan' else 'arbitrary-routing-name'
    assert client.get_effective_request_args()['model'] == expected
    if harness == 'codex' and family != 'hunyuan':
        assert callable(client.create_responses)


@pytest.mark.parametrize(('family', 'harness'), MODES)
def test_template_provisioning_keeps_callers_template(monkeypatch, family, harness):
    choose(monkeypatch, family, harness)
    monkeypatch.setenv('OSWORLD_AGS_PROVISION_MODE', 'template')
    monkeypatch.setenv('AGS_OSWORLD_TEMPLATE', 'user-template')
    monkeypatch.setenv('AGS_API_KEY', 'template-secret')
    monkeypatch.delenv('AGS_SECRET_ID')
    monkeypatch.delenv('AGS_SECRET_KEY')
    cfg = RunInput.from_dict(next(generator.generate_configs()).config.model_dump())
    assert cfg.runtime_provider_config['api_key'] == 'template-secret'
    assert cfg.runtime_configs['main']['template'] == 'user-template'
    assert cfg.runtime_configs['main']['runtime_kind'] == 'osworld'
    assert 'tool_id' not in cfg.runtime_configs['main']


def test_single_explicit_id_ignores_batch_pagination_without_mutating_env(monkeypatch):
    choose(monkeypatch, 'claude', 'bare')
    monkeypatch.setenv('OSWORLD_SKIP', '100')
    monkeypatch.setenv('OSWORLD_LIMIT', '0')
    monkeypatch.setenv('OSWORLD_TASK_IDS', 'task-a')
    before = dict(os.environ)
    cfg = generator.single_config(task_id='task-b')
    assert cfg.task_config['task_id'] == 'task-b'
    assert dict(os.environ) == before


@pytest.mark.parametrize('task_id', ['', '../task-a', 'task-a,task-b', 'missing'])
def test_invalid_or_missing_single_task_fails(monkeypatch, task_id):
    choose(monkeypatch, 'gpt', 'bare')
    with pytest.raises(ValueError, match='task_id'):
        generator.single_config(task_id=task_id)


def test_no_dataset_never_falls_back_to_local_examples(monkeypatch):
    choose(monkeypatch, 'gpt', 'bare')
    monkeypatch.delenv('OSWORLD_DATASET_URI')
    monkeypatch.setenv('OSWORLD_EXAMPLES_ROOT', '/must-not-read')
    with pytest.raises(ValueError, match='OSWORLD_DATASET_URI'):
        list(generator.generate_configs())


@pytest.mark.parametrize(('family', 'harness'), MODES)
def test_actual_single_cli_dispatches_generator_config(monkeypatch, family, harness):
    from nexus.cli.commands.run import register_run_single_parser, run_single

    choose(monkeypatch, family, harness)
    execute = AsyncMock(return_value=SimpleNamespace(task_output={}))
    monkeypatch.setattr(importlib.import_module('nexus.runner.single'), 'run', execute)
    parser = argparse.ArgumentParser()
    register_run_single_parser(parser.add_subparsers(dest='command'))
    argv = ['run-single', '--use', f'osworld_native:model_family={family},harness={harness},task_id=task-a']
    args = parser.parse_args(argv)
    args._raw_argv = argv
    assert run_single(args) == 0
    execute.assert_awaited_once_with(RunInput.from_dict(generator.single_config(task_id='task-a').model_dump()))


def test_run_part_rejects_unknown_keys():
    with pytest.raises(ValueError, match='does not accept'):
        osworld_native({'task_id': 'task-a', 'unknown': 'value'})


def test_generator_does_not_write_the_environment():
    tree = ast.parse(Path(generator.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            assert not (isinstance(node.value, ast.Attribute) and node.value.attr == 'environ')


@pytest.mark.parametrize(('family', 'harness'), MODES)
def test_actual_batch_cli_loads_native_generator(monkeypatch, family, harness, tmp_path):
    from nexus.cli.commands.run import register_run_batch_parser, run_batch

    choose(monkeypatch, family, harness)
    captured = []

    async def execute(**kwargs):
        loaded = list(GeneratorLoader(kwargs['generator_path']).generate())
        captured.extend(loaded)
        return SimpleNamespace(total=len(loaded), success=len(loaded), failed=0, timeout=0, failed_runs=[])

    monkeypatch.setattr(importlib.import_module('nexus.runner.batch'), 'run_batch', execute)
    parser = argparse.ArgumentParser()
    register_run_batch_parser(parser.add_subparsers(dest='command'))
    args = parser.parse_args(['run-batch', '-g', generator.__file__, '-d', str(tmp_path / 'batch.db')])
    assert run_batch(args) == 0
    assert [i.model_dump() for i in captured] == [i.model_dump() for i in generator.generate_configs()]


def test_template_large_task_requires_explicit_policy_decision(monkeypatch):
    from nexus.extensions.tasks.os_world.data.task_source import OSWorldBatchTask
    from nexus.extensions.tasks.os_world.execution.sandbox_policy import GIMP_UPSCALE_TASK_ID

    choose(monkeypatch, 'gpt', 'bare')
    monkeypatch.setenv('OSWORLD_AGS_PROVISION_MODE', 'template')
    monkeypatch.setenv('AGS_OSWORLD_TEMPLATE', 'user-template')
    monkeypatch.setenv('AGS_API_KEY', 'template-secret')
    task = OSWorldBatchTask('gimp', GIMP_UPSCALE_TASK_ID, 'task')
    with pytest.raises(ValueError, match='large-sandbox policy'):
        generator.build_item(task, generator.selection())


def test_proxy_fields_are_not_lost_on_bare_claude(monkeypatch):
    choose(monkeypatch, 'claude', 'bare')
    monkeypatch.setenv('OSWORLD_FORCE_PROXY', 'true')
    cfg = RunInput.from_dict(next(generator.generate_configs()).config.model_dump())
    assert cfg.task_config['enable_proxy'] is True
    assert cfg.task_config['force_proxy'] is True
    assert cfg.task_config['proxy_config_file'] == os.environ['OSWORLD_PROXY_CONFIG_FILE']


def test_new_path_does_not_consult_old_harness_model_selectors(monkeypatch):
    choose(monkeypatch, 'gpt', 'codex')
    monkeypatch.setenv('AGENT_KIND', 'invalid-legacy')
    monkeypatch.setenv('NEXUS_MAIN_CLIENT_NAME', 'anthropic_messages')
    monkeypatch.setenv('CODEX_MODEL_BACKEND', 'hunyuan')
    cfg = next(generator.generate_configs()).config
    assert cfg.openai_clients['main']['name'] == 'openai_responses'
    assert cfg.agent_config['execution_options']['model'] == 'gpt-6-astra'


@pytest.mark.parametrize(('family', 'harness'), MODES)
@pytest.mark.parametrize('provision_mode', ['control_plane', 'template'])
def test_fixed_batch_entry_and_exported_yaml_share_every_field(monkeypatch, tmp_path, family, harness, provision_mode):
    from nexus.extensions.tasks.os_world.scripts.export_single import main

    choose(monkeypatch, family, harness)
    monkeypatch.setenv('OSWORLD_AGS_PROVISION_MODE', provision_mode)
    monkeypatch.setenv('AGS_API_KEY', 'template-fixture')
    monkeypatch.setenv('AGS_OSWORLD_TEMPLATE', 'template-fixture')
    # Fixed entries must ignore stale selectors without temporarily changing env.
    monkeypatch.setenv('OSWORLD_MODEL_FAMILY', 'wrong')
    monkeypatch.setenv('OSWORLD_HARNESS', 'wrong')
    path = Path(generator.__file__).with_name(f'generator_native_{family}_{harness}.py')
    before = dict(os.environ)
    loader = GeneratorLoader(path)
    items = list(loader.generate())
    assert loader.get_total_count() == len(items) == 2
    assert loader.get_batch_metadata()['model_family'] == family
    target = tmp_path / 'single.yaml'
    main(['--model-family', family, '--harness', harness, '--task-id', 'task-b', '-o', str(target)])
    resolved = RunInput.from_file(target)
    expected = RunInput.from_dict(items[1].config.model_copy(update={'run_id': items[1].run_id}).model_dump())
    assert resolved.model_dump() == expected.model_dump()
    assert resolved.runtime_configs['main']['provision_mode'] == provision_mode
    assert ('api_key' in resolved.runtime_provider_config) == (provision_mode == 'template')
    assert ('secret_id' in resolved.runtime_provider_config) == (provision_mode == 'control_plane')
    assert not resolved.task_config.get('task_root')
    assert not resolved.task_config.get('task_file')
    assert resolved.task_files.remote is not None
    assert dict(os.environ) == before


def test_all_modes_export_uses_identical_data_and_runtime_blocks(monkeypatch, tmp_path):
    import yaml

    from nexus.extensions.tasks.os_world.scripts.export_single import main

    choose(monkeypatch, 'gpt', 'bare')
    assert main(['--all-modes', '--task-id', 'task-b', '-o', str(tmp_path)]) == 0
    configs = [yaml.safe_load((tmp_path / f'osworld_{f}_{h}.yaml').read_text()) for f, h in MODES]
    for config in configs:
        for field in ('task_config', 'task_files', 'runtime_provider_config', 'runtime_configs'):
            assert config[field] == configs[0][field]
        assert 'wedata-sentinel' not in json.dumps(config)


def test_six_checked_in_single_configs_share_data_and_control_plane_settings():
    import yaml

    root = Path(generator.__file__).resolve().parents[1] / 'configs'
    configs = [yaml.safe_load((root / f'osworld_{f}_{h}.yaml').read_text()) for f, h in MODES]
    for config in configs:
        # A user may regenerate one example for another task. Compare common
        # policy without requiring every example to name the same data item.
        for field in ('runtime_provider_config', 'runtime_configs'):
            assert config[field] == configs[0][field]
        assert config['task_files']['path'].endswith(f'/{config["task_config"]["task_id"]}.zip')
        assert config['task_config']['wedata_key'] == '${oc.env:WEDATA_KEY}'
        assert not config['task_config'].get('task_root')
        assert not config['task_config'].get('task_file')
        assert config['runtime_configs']['main']['provision_mode'] == 'control_plane'
        assert config['runtime_provider_config']['secret_id'] == '${oc.env:AGS_SECRET_ID}'
        assert 'api_key' not in config['runtime_provider_config']
        assert 'template' not in config['runtime_configs']['main']


def test_malformed_wedata_key_fails_before_dataset_lookup(monkeypatch):
    from nexus.dataset.exceptions import DatasetConfigurationError

    choose(monkeypatch, 'gpt', 'bare')
    monkeypatch.setenv('WEDATA_KEY', 'shell-damaged-secret')
    monkeypatch.setattr(generator, '_tasks', lambda **kw: pytest.fail('must not query dataset with invalid key'))
    with pytest.raises(DatasetConfigurationError, match='single quotes'):
        list(generator.generate_configs())
    with pytest.raises(DatasetConfigurationError, match='single quotes'):
        generator.single_config(task_id='task-b')


def test_gpt_codex_saved_yaml_and_generator_share_native_client_policy(monkeypatch):
    import yaml

    choose(monkeypatch, 'gpt', 'codex')
    saved = yaml.safe_load(
        (Path(generator.__file__).resolve().parents[1] / 'configs/osworld_gpt_codex.yaml').read_text()
    )
    generated = next(generator.generate_configs()).config
    block = saved['openai_clients']['main']
    live = generated.openai_clients['main']
    assert block['name'] == live['name'] == 'openai_responses'
    assert 'nexus.extensions.openai_clients.openai_responses' in saved['imports']
    assert block['config']['request_args']['reasoning'] == live['config']['request_args']['reasoning']
    for config in (block['config'], live['config']):
        assert 'prompt_cache_task_id' in config['client_args']
        assert 'cache_task_id' not in config['client_args']
        assert 'max_attempts' not in config['client_args']
        assert 'reasoning_summary' not in config['request_args']
    monkeypatch.setenv('CODEX_REASONING_SUMMARY', 'none')
    config = next(generator.generate_configs()).config
    assert config.openai_clients['main']['config']['request_args']['reasoning'] == {'effort': 'none'}


@pytest.mark.parametrize('harness', ['cc', 'codex'])
def test_hunyuan_platform_service_export_does_not_pin_main_client(monkeypatch, harness, tmp_path):
    from nexus.extensions.tasks.os_world.scripts import export_platform_dataset as exporter

    choose(monkeypatch, 'hunyuan', harness)
    before = list(generator.generate_configs())
    out = tmp_path / 'rows.jsonl'
    common_path = tmp_path / 'common.json'
    assert (
        exporter.main(
            [
                '--generator',
                'nexus.extensions.tasks.os_world.generators.generator_native',
                '--model-source',
                'platform-service',
                '-o',
                str(out),
                '--eval-set-out',
                str(common_path),
            ]
        )
        == 0
    )
    common = json.loads(common_path.read_text())
    assert 'openai_clients' not in common
    assert 'main_client_args' not in common
    assert 'nexus_main_client_name' not in common
    assert not any('uda_native' in module for module in common['imports'])
    assert common['agent_name'] == ('claude_code_native_bridge' if harness == 'cc' else 'codex_native_bridge')
    expected_agent = dict(before[0].config.agent_config)
    assert common['agent_config'] == expected_agent
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r['meta_data_']['nexus_run_input']['task_files'] for r in rows] == [
        item.config.task_files.model_dump(exclude_none=True) for item in before
    ]
    assert not (tmp_path / 'client-form.json').exists()


def test_platform_service_does_not_accept_local_client_form_output(tmp_path):
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import main

    with pytest.raises(SystemExit):
        main(['--model-source', 'platform-service', '-o', str(tmp_path / 'rows'), '--client-form-out', 'model.json'])


@pytest.mark.parametrize('harness', ['cc', 'codex'])
def test_hunyuan_yaml_model_placeholder_does_not_change_affinity(monkeypatch, harness):
    choose(monkeypatch, 'hunyuan', harness)
    original = next(generator.generate_configs()).config
    monkeypatch.setenv('HUNYUAN_MODEL', '${oc.env:HUNYUAN_MODEL}')
    template = next(generator.generate_configs()).config
    assert (
        template.openai_clients['main']['config']['request_args']['extra_headers']
        == (original.openai_clients['main']['config']['request_args']['extra_headers'])
    )


def test_native_source_has_no_task_specific_193924_or_uda_dependency():
    text = Path(generator.__file__).read_text()
    assert 'generator_193924' not in text
    assert 'uda_native' not in text
    assert 'osworld_hunyuan_responses' not in text
    # A common agent name is not enough: its harness builder cannot select a
    # historical implementation by model family.
    tree = ast.parse(text)
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_agent')
    harness_tail = ast.get_source_segment(text, builder).split("if choice.harness == 'codex':", 1)[1]
    assert 'model_family' not in harness_tail
    assert '_historical_agent_config' not in text


@pytest.mark.parametrize(('harness', 'families'), [('cc', ('claude', 'hunyuan')), ('codex', ('gpt', 'hunyuan'))])
def test_same_harness_is_exactly_equal_not_just_same_agent_name(monkeypatch, harness, families):
    configs = []
    # Legacy knobs may exist in .env but must not select a different native loop.
    monkeypatch.setenv('CC_TASK_PROTOCOL', 'true')
    monkeypatch.setenv('CC_APP_SKILLS', 'true')
    monkeypatch.setenv('CODEX_MODEL_BACKEND', 'hunyuan')
    monkeypatch.setenv('HUNYUAN_CONTEXT_LIMIT', '123456')
    for family in families:
        choose(monkeypatch, family, harness)
        configs.append(next(generator.generate_configs()).config)
    assert configs[0].agent_name == configs[1].agent_name
    assert configs[0].agent_config == configs[1].agent_config
    assert configs[0].task_config == configs[1].task_config
    assert configs[0].runtime_configs == configs[1].runtime_configs
    assert configs[0].openai_clients != configs[1].openai_clients
    if harness == 'cc':
        assert configs[0].agent_config['execution_profile'] == 'osworld'
        cfg = configs[0].agent_config['execution_options']
        assert 'task_protocol_enabled' not in cfg
        assert 'app_skills_enabled' not in cfg
        assert 'cc_context_window_override' not in cfg
        assert cfg['expected_claude_code_version'] == '2.1.205'
        assert cfg['thinking_preset'] == 'auto'
    else:
        cfg = configs[0].agent_config['execution_options']
        catalog = json.loads(cfg['model_catalog_json'])['models'][0]
        assert catalog['context_window'] == 262144
        assert cfg['auto_compact_token_limit'] == 194048
        assert cfg['reasoning_effort'] == 'none'


@pytest.mark.parametrize('harness', ['cc', 'codex'])
def test_explicit_harness_knobs_apply_to_every_client(monkeypatch, harness):
    if harness == 'cc':
        monkeypatch.setenv('CC_MODEL', 'explicit-facade')
        monkeypatch.setenv('CC_AUTO_COMPACT_WINDOW', '180000')
        families = ('claude', 'hunyuan')
    else:
        monkeypatch.setenv('CODEX_CONTEXT_LIMIT', '400000')
        monkeypatch.setenv('CODEX_REASONING_EFFORT', 'xhigh')
        monkeypatch.setenv('CODEX_REASONING_SUMMARY', 'detailed')
        families = ('gpt', 'hunyuan')
    configs = [generator.single_config(task_id='task-b', model_family=f, harness=harness) for f in families]
    assert configs[0].agent_config == configs[1].agent_config
    if harness == 'cc':
        assert configs[0].agent_config['execution_options']['cc_model'] == 'explicit-facade'
        assert configs[0].agent_config['execution_options']['extra_env']['CLAUDE_CODE_AUTO_COMPACT_WINDOW'] == '180000'
    else:
        assert configs[0].agent_config['execution_options']['auto_compact_token_limit'] == 331904
        assert configs[0].agent_config['execution_options']['reasoning_effort'] == 'xhigh'


@pytest.mark.parametrize(('harness', 'families'), [('cc', ('claude', 'hunyuan')), ('codex', ('gpt', 'hunyuan'))])
def test_saved_harness_configs_match_shared_builder(monkeypatch, harness, families):
    import yaml

    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldExecutionOptions, _guide_text
    from nexus.extensions.tasks.os_world.execution.codex_profile import OSWorldCodexOptions
    from nexus.extensions.tasks.os_world.generators.harness import cc_config, codex_config

    root = Path(generator.__file__).resolve().parents[1] / 'configs'
    expected = cc_config() if harness == 'cc' else codex_config()
    for family in families:
        actual = yaml.safe_load((root / f'osworld_{family}_{harness}.yaml').read_text())['agent_config']
        # Old complete YAMLs can retain explicit nulls for optional input fields.
        actual = {k: v for k, v in actual.items() if v is not None}
        compare = {k: v for k, v in expected.items() if v is not None}
        if harness == 'codex':
            selectors = {
                'guide_version',
                'model_catalog_preset',
                'context_window',
                'max_output_tokens',
                'context_token_margin',
                'user_prompt',
            }
            actual['execution_options'] = OSWorldCodexOptions(user_prompt='', **actual['execution_options']).model_dump(
                exclude=selectors
            )
            compare['execution_options'] = OSWorldCodexOptions(
                user_prompt='', **compare['execution_options']
            ).model_dump(exclude=selectors)
        elif 'guide_version' in actual['execution_options']:
            # A saved YAML can explicitly select a different guide than the generator default.
            options = actual['execution_options']
            selected = _guide_text(options.pop('guide_version'))
            options.setdefault('guide_text', selected)
            compare['execution_options'] = {**compare['execution_options'], 'guide_text': selected}
            # Compare effective defaults, just as Codex above; missing False is
            # equivalent to explicit False after the real profile validation.
            actual['execution_options'] = OSWorldExecutionOptions(
                user_prompt='', **actual['execution_options']
            ).model_dump()
            compare['execution_options'] = OSWorldExecutionOptions(
                user_prompt='', **compare['execution_options']
            ).model_dump()
        assert actual == compare


def test_platform_export_preserves_codex_facade_and_does_not_share_first_task_cache(monkeypatch):
    from nexus.extensions.tasks.os_world.scripts import export_platform_dataset as exporter

    choose(monkeypatch, 'gpt', 'codex')
    item = next(generator.generate_configs())
    common, _ = exporter._eval_set_config(item)
    exporter._drop_derivable(common)
    assert common['agent_config'] == item.config.agent_config
    form = exporter._client_form(item)
    assert 'prompt_cache_task_id' not in form['parameter_configuration']['main_client_args']


@pytest.fixture
def platform_merge():
    path = Path('/data/workspace/hy_unify_eval/sdk/eval_class/text_to_text/nexus/common/param_utils.py')
    if not path.is_file():
        pytest.skip('Platform SDK checkout is not available')

    def merge(default, customize, dataset_version, user):
        program = """
import ast, contextvars, copy, json, logging, sys
from pathlib import Path
from omegaconf import OmegaConf
path=Path('/data/workspace/hy_unify_eval/sdk/eval_class/text_to_text/nexus/common/param_utils.py')
tree=ast.parse(path.read_text())
wanted={'_deep_update','_get_env_var','omega_resolve_config','resolve_env_value'}
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
namespace={
 'copy':copy,'logger':logging.getLogger('platform-sdk-contract'),'OmegaConf':OmegaConf,
 'omega_conf_context':contextvars.ContextVar('platform-contract'),'_ENV_VAR_MISSING':object(),
}
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*functions],type_ignores=[])
exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
OmegaConf.register_new_resolver('oc.env',namespace['_get_env_var'],replace=True)
payload=json.load(sys.stdin)
version=namespace['resolve_env_value'](payload['dataset_version'],payload.get('env_mapping') or {})
result={}
layers=[payload['default'],payload['customize'],version,payload['user']]
for i, layer in enumerate(layers):
    result=namespace['_deep_update'](result,layer,is_clear_key=i==len(layers)-1)
print(json.dumps(result))
"""
        payload = {
            'default': default,
            'customize': customize,
            'dataset_version': dataset_version,
            'user': user,
            'env_mapping': merge.env_mapping,
        }
        result = subprocess.run(
            [sys.executable, '-c', program],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    merge.env_mapping = {}
    return merge


@pytest.mark.parametrize(('family', 'harness'), MODES)
def test_platform_merge_keeps_same_remote_task_files_and_resolves_wedata_only_in_common(
    monkeypatch,
    family,
    harness,
    platform_merge,
):
    from nexus.extensions.tasks.os_world.scripts import export_platform_dataset as exporter

    choose(monkeypatch, family, harness)
    items = list(generator.generate_configs())
    baseline = items[0].config.model_dump(exclude_none=True)
    common, pruned = exporter._eval_set_config(items[0])
    rows = [exporter._dataset_row(item, baseline) for item in items]
    assert 'task_files' not in common
    assert 'openai_clients' not in common
    assert common['task_config']['wedata_key'] == '${oc.env:WEDATA_KEY}'
    assert not (set(exporter.DATASET_LOCAL_PATH_KEYS) & set(common['task_config']))
    for item, row in zip(items, rows, strict=True):
        assert exporter._assert_split_is_total(item, row, common, pruned) == []
        assert '${' not in json.dumps(row)
        assert row['meta_data_']['nexus_run_input']['task_files'] == item.config.task_files.model_dump(
            exclude_none=True
        )

        # Simulate only the documented dataset_version resolver boundary.
        # After resolution, no process env should be needed for the WEDATA field.
        wedata = {'id': 1, 'subject': 'fixture', 'key': 'rainbow-test-key'}
        platform_merge.env_mapping = {'WEDATA_KEY': wedata}
        user = {'agent_config': {'max_iterations': 11}}
        merged = platform_merge(
            {'task_config': {'task_id': 'wrong-default'}, 'agent_config': {'max_iterations': 1}},
            row['meta_data_']['nexus_run_input'],
            common,
            user,
        )
        platform_merge.env_mapping = {}
        assert merged['task_config']['task_id'] == item.config.task_config['task_id']
        assert merged['task_files'] == item.config.task_files.model_dump(exclude_none=True)
        assert merged['agent_config']['max_iterations'] == 11
        task_input = _build_input(OSWorldTaskInput, {**merged['task_config'], 'run_agent': lambda **_: None})
        assert task_input.wedata_key == wedata
        assert task_input.task_root == ''


def test_platform_priority_and_override_use_actual_sdk(platform_merge):
    layers = [
        {'task_config': {'domain': 'default'}, 'agent_config': {'timeout': 1, 'steps': 1}},
        {'task_config': {'domain': 'sample', 'task_id': 'a'}, 'agent_config': {'timeout': 2}},
        {'task_config': {'domain': 'eval-set'}, 'agent_config': {'timeout': 3}},
        {'agent_config': {'timeout': 4}},
    ]
    merged = platform_merge(*layers)
    assert merged['task_config'] == {'domain': 'eval-set', 'task_id': 'a'}
    assert merged['agent_config'] == {'timeout': 4, 'steps': 1}
    layers[-1] = {'agent_config__override__': {'timeout': 5}}
    assert platform_merge(*layers)['agent_config'] == {'timeout': 5}


def test_export_guard_rejects_common_task_files_and_row_interpolation(monkeypatch):
    from nexus.extensions.tasks.os_world.scripts import export_platform_dataset as exporter

    choose(monkeypatch, 'gpt', 'bare')
    item = next(generator.generate_configs())
    common, pruned = exporter._eval_set_config(item)
    row = exporter._dataset_row(item)
    common['task_files'] = row['meta_data_']['nexus_run_input']['task_files']
    assert any('task_files must be per-task' in p for p in exporter._assert_split_is_total(item, row, common, pruned))
    common.pop('task_files')
    row['meta_data_']['nexus_run_input']['task_config']['task_id'] = '${oc.env:TASK_ID}'
    assert any(
        'must not contain interpolation' in p for p in exporter._assert_split_is_total(item, row, common, pruned)
    )


def test_large_sandbox_fix_overrides_public_default_but_user_override_can_replace_it(monkeypatch, platform_merge):
    from nexus.extensions.tasks.os_world.execution import sandbox_policy
    from nexus.extensions.tasks.os_world.scripts import export_platform_dataset as exporter

    choose(monkeypatch, 'gpt', 'bare')
    monkeypatch.setitem(sandbox_policy.LARGE_TASK_PROFILES, ('chrome', 'task-b'), 'fixture-large')
    items = list(generator.generate_configs())
    common, pruned = exporter._eval_set_config(items[0])
    row = exporter._dataset_row(items[1], items[0].config.model_dump(exclude_none=True))
    assert exporter._assert_split_is_total(items[1], row, common, pruned) == []
    platform_merge.env_mapping = {
        'WEDATA_KEY': {'id': 1, 'subject': 'fixture', 'key': 'rainbow-test-key'},
        'AGS_SECRET_ID': 'fixture-id',
        'AGS_SECRET_KEY': 'fixture-key',
        'AGS_OSWORLD_TOOL_ID': 'fixture-tool',
    }
    merged = platform_merge({}, row['meta_data_']['nexus_run_input'], common, {})
    assert merged['runtime_configs']['main']['tool_id'] == sandbox_policy.LARGE_TOOL_ID
    override = {'runtime_configs': {'main': {'tool_id__override__': 'explicit-user-tool'}}}
    merged = platform_merge({}, row['meta_data_']['nexus_run_input'], common, override)
    assert merged['runtime_configs']['main']['tool_id'] == 'explicit-user-tool'
    platform_merge.env_mapping = {}
