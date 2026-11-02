"""Offline platform configuration contracts; no dataset, sandbox or model calls."""

import copy
import importlib.util
import json
import os
import re

from pathlib import Path

import pytest

from omegaconf import OmegaConf

from nexus.extensions.tasks.os_world_v2.execution.claude_code import OSWorldExecutionOptions
from nexus.extensions.tasks.os_world_v2.adapter.codex import OSWorldV2CodexProfile
from nexus.extensions.tasks.os_world_v2.task import OSWorldV2TaskInput, OSWorldV2TaskOutput
from nexus.runner.single import RunInput


ROOT = Path(__file__).resolve().parents[5]
RUNBOOK = ROOT / 'docs/how-to/osworld-v2-runbook.md'


def blocks():
    return [json.loads(block) for block in re.findall(r'```json\n(.*?)```', RUNBOOK.read_text(), re.S)]


def common_config():
    agent = next(block for block in blocks() if block.get('agent_name') == 'claude_code_multiturn')
    hook = next(block for block in blocks() if 'hooks' in block.get('runner', {}))
    return {
        'task_name': 'os_world_v2',
        'task_config': {'v21': {}},
        'runner': hook['runner'],
        'agent_name': agent['agent_name'],
        'agent_config': agent['agent_config__override__'],
    }


@pytest.fixture
def environment(monkeypatch):
    for key, value in {
        'AGS_SECRET_ID': 'fixture-id',
        'AGS_SECRET_KEY': 'fixture-key',
        'AGS_OSWORLD_TOOL_ID': 'sdt-fixture',
        'AGS_REGION': 'ap-guangzhou',
        'OSWORLD_CLIENT_PASSWORD': 'fixture-password',
        'WEBSITE_HOST_SUFFIX': 'localhost',
        'ARTIFACT_TAIHU_PAT': 'fixture-artifact-pat',
        'ARTIFACT_SERVICE_URL': 'https://artifact.invalid',
        'WEDATA_KEY': '{"id":"fixture","subject":"fixture","key":"fixture"}',
    }.items():
        monkeypatch.setenv(key, value)


def row(selected):
    return {
        'task_config': {'task_id': selected, 'domain': 'tasks', 'bundle_sha256': 'a' * 64},
        'task_files': {
            'type': 'zip',
            'path': f'blobs/{selected}.zip',
            'sha256': 'b' * 64,
            'size': 100,
            'remote': {'namespace': 'hy_gui_eval', 'dataset': 'osworld_v2', 'version': 'v2.1.0', 'split': 'test'},
        },
    }


def deep_update(base, extra):
    result = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def assert_merged(common, selected, merge=deep_update):
    # customize then dataset_version, matching the platform's higher common layer.
    config = merge(row(selected), common)
    run = RunInput.model_validate(config)
    task = OSWorldV2TaskInput(run_agent=lambda: None, **run.task_config)
    assert task.task_id == selected
    assert task.bundle_sha256 == 'a' * 64
    assert task.v21.repo_root == task.v21.asset_root == ''
    assert run.task_files.remote.dataset == 'osworld_v2'
    assert run.agent_name == 'claude_code_multiturn'
    OSWorldExecutionOptions(user_prompt='fixture', **run.agent_config['execution_options'])


def test_common_and_rows_are_disjoint_and_schema_valid(environment):
    common = common_config()
    assert common['task_name'] == 'os_world_v2'
    assert not {'task_files', 'run_id', 'processors', 'openai_clients'} & common.keys()
    assert common['runner']['hooks'] == {'output_contract': {'name': 'osworld_v2_output_contract', 'args': {}}}
    assert 'hooks' not in common['task_config']
    assert not {'task_id', 'domain', 'bundle_sha256'} & common['task_config'].keys()
    assert not {'repo_root', 'asset_root', 'dependency_root'} & common['task_config']['v21'].keys()
    assert '${oc.decode:' not in json.dumps(common)
    resolved = OmegaConf.to_container(OmegaConf.create(common), resolve=True)
    for selected in ('001', '020', '108'):
        assert_merged(resolved, selected)


def test_codex_platform_example_supports_v2_interaction():
    config = next(block for block in blocks() if block.get('agent_name') == 'codex_native_bridge')
    agent = config['agent_config__override__']
    assert agent['execution_profile'] == 'osworld_v2'
    for has_simulator in (False, True):
        profile = OSWorldV2CodexProfile(
            agent['execution_options'], {'osworld_user_simulator': has_simulator}, 'fixture'
        )
        assert profile.options.session_mode == 'auto'
        assert profile.options.guide_text
        assert profile.options.model_catalog_json


def test_service_forms_separate_model_sources_and_redact_key():
    pc = next(block for block in blocks() if 'nexus_main_client_name' in block)
    assert pc['nexus_main_client_name'] == 'anthropic_messages'
    assert pc['api_key'].startswith('<')
    assert 'api_key' not in pc['main_client_args']
    hunyuan = next(block for block in blocks() if 'model_call_total_timeout_s' in block.get('main_client_args', {}))
    assert hunyuan['main_client_args']['model_call_total_timeout_s'] > 0
    assert 'openai_clients' not in common_config()


def test_rule_excludes_failed_or_unscored_runs():
    engine = pytest.importorskip('rule_engine')
    rule_text = next(
        block for block in re.findall(r'```text\n(.*?)```', RUNBOOK.read_text(), re.S) if block.startswith('success ==')
    )
    rule = engine.Rule(rule_text.strip())
    task = OSWorldV2TaskOutput(
        completed=True, task_id='020', instruction='fixture', score=0, evaluator_result={'scored': True}
    ).model_dump()
    result = {'success': True, 'error': None, 'task_output': task}
    assert rule.matches(result)  # legitimate zero is valid
    task['evaluator_result']['scored'] = False
    assert not rule.matches(result)
    assert not rule.matches({'success': False, 'error': {'type': 'Error'}, 'task_output': None})


def test_actual_platform_merge_if_local_source_available(environment):
    root = Path(os.getenv('HY_UNIFY_EVAL_ROOT', '/data/workspace/hy_unify_eval'))
    path = root / 'sdk/data_class/text_to_text/nexus/common/param_utils.py'
    if not path.is_file():
        pytest.skip('Local platform source unavailable')
    pytest.importorskip('loguru')
    spec = importlib.util.spec_from_file_location('platform_merge_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = next(block for block in blocks() if block.get('agent_name') == 'claude_code_multiturn')
    resolved = module._deep_update(
        {'agent_config': {'obsolete_fixture_field': True}}, copy.deepcopy(agent), is_clear_key=True
    )
    assert resolved['agent_config'] == agent['agent_config__override__']
    common = OmegaConf.to_container(OmegaConf.create(common_config()), resolve=True)
    for selected in ('020', '069'):
        assert_merged(common, selected, module._deep_update)


def test_all_published_rows_merge_when_local_jsonl_available(environment):
    path = Path('/data/workspace/datasets/osworld_dataset_v2.1.0.jsonl')
    if not path.is_file():
        pytest.skip('Published dataset JSONL not available in this checkout')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 108
    common = OmegaConf.to_container(OmegaConf.create(common_config()), resolve=True)
    for record in rows:
        per_task = record['meta_data_']['nexus_run_input']
        merged = RunInput.model_validate(deep_update(per_task, common))
        task = OSWorldV2TaskInput(run_agent=lambda: None, **merged.task_config)
        assert task.task_id == per_task['task_config']['task_id']
        assert task.bundle_sha256 == per_task['task_config']['bundle_sha256']
        assert (
            merged.task_files
            == RunInput.model_validate(
                {
                    'task_name': 'os_world_v2',
                    'task_files': per_task['task_files'],
                }
            ).task_files
        )
        assert merged.task_files.remote.version == 'v2.1.0'
        assert task.v21.repo_root == ''
