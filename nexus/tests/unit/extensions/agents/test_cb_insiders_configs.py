import importlib.util
import json
import os

import pytest


@pytest.fixture
def hyway_generator(monkeypatch, tmp_path, request):
    root = request.config.rootpath
    path = root / 'docs/examples/configs/cb-insiders/hyway_generator.py'
    spec = importlib.util.spec_from_file_location('cb_insiders_hyway_config_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in list(os.environ):
        if name.startswith(('SWE_', 'GONGFENG_', 'CB_INSIDERS_', 'WORKBUDDY_')):
            monkeypatch.delenv(name)
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture-model-key?provider=aws_third&account_id2=fixture-account')
    monkeypatch.setenv('CB_INSIDERS_NPM_USERNAME', 'fixture-reader')
    monkeypatch.setenv('CB_INSIDERS_NPM_TOKEN', 'fixture-npm-token')
    monkeypatch.setenv('SWE_DISABLE_LANGFUSE', 'true')
    dataset = tmp_path / 'cases.jsonl'
    dataset.write_text(
        json.dumps({
            'instance_id': 'fixture__repo-1',
            'repo': 'fixture/repo',
            'dataset_type': 'swe-bench-multilingual',
            'problem_statement': 'Fix the fixture.',
        })
        + '\n'
    )
    monkeypatch.setenv('SWE_DATASET_PATH', str(dataset))
    return module


@pytest.mark.parametrize('token_name', ['GONGFENG_RUNTIME_TOKEN', 'GONGFENG_TOKEN'])
def test_hyway_accepts_either_sandbox_token_name(hyway_generator, monkeypatch, token_name):
    monkeypatch.setenv(token_name, 'fixture-sandbox-token')
    items = list(hyway_generator.generate_configs())
    assert len(items) == 1
    config = items[0].config
    assert config.runtime_provider_config['token'] == 'fixture-sandbox-token'
    assert config.agent_name == 'cb-insiders'
    assert config.agent_config['npm_username'] == 'fixture-reader'
    assert config.agent_config['effort'] == 'high'
    client = config.openai_clients['main']['config']
    assert client['client_args']['api_key'].endswith('?provider=aws_third&account_id2=fixture-account')
    assert client['enforced_request_args']['reasoning'] == {'effort': 'high'}
    assert config.task_config['enable_anti_hack']
    assert config.task_config['enable_anti_hack_prompt']
    assert config.task_config['enable_reward_hacking_detection']


def test_hyway_requires_a_sandbox_token(hyway_generator):
    with pytest.raises(RuntimeError, match='GONGFENG_RUNTIME_TOKEN or GONGFENG_TOKEN'):
        list(hyway_generator.generate_configs())


def test_hyway_ignores_an_empty_preferred_token(hyway_generator, monkeypatch):
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', '  ')
    monkeypatch.setenv('GONGFENG_TOKEN', ' fixture-sandbox-token ')
    item = next(hyway_generator.generate_configs())
    assert item.config.runtime_provider_config['token'] == 'fixture-sandbox-token'


@pytest.mark.parametrize(
    'key',
    [
        'fixture-model-key',
        'fixture-model-key?provider=aws_third',
        'fixture-model-key?provider=other&account_id2=fixture-account',
        'fixture-model-key?provider=aws_third&account_id2=',
        'fixture-model-key?provider=aws_third&account_id2=first&account_id2=second',
    ],
)
def test_hyway_rejects_ambiguous_or_missing_model_routing(hyway_generator, monkeypatch, key):
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'fixture-sandbox-token')
    monkeypatch.setenv('OPENAI_API_KEY', key)
    with pytest.raises(ValueError, match='GPT-6.1 Sol requires'):
        list(hyway_generator.generate_configs())


def test_swe_smoke_preserves_template_dependencies_and_uses_actual_model(monkeypatch, tmp_path, request):
    path = request.config.rootpath / 'docs/examples/configs/cb-insiders/swe_generator.py'
    spec = importlib.util.spec_from_file_location('cb_insiders_swe_config_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in list(os.environ):
        if name.startswith(('SWE_', 'GONGFENG_', 'CB_INSIDERS_', 'WORKBUDDY_')):
            monkeypatch.delenv(name)
    for name in (
        'GONGFENG_RUNTIME_TOKEN',
        'CB_INSIDERS_NPM_USERNAME',
        'CB_INSIDERS_NPM_TOKEN',
        'OPENAI_API_KEY',
        'LANGFUSE_PUBLIC_KEY',
        'LANGFUSE_SECRET_KEY',
    ):
        monkeypatch.setenv(name, 'fixture-value')
    monkeypatch.setenv('LANGFUSE_BASE_URL', 'http://langfuse.fixture')
    template = module.RunInput.from_file(path.with_name('smoke.yaml'))
    requirements = ['existing-dependency==1.0', 'unidiff>=0.7.5']
    template.runtime_configs['main']['worker_venv']['extra_requirements'] = list(requirements)
    template.openai_clients['main']['config']['enforced_request_args']['model'] = 'fixture-model'
    monkeypatch.setattr(module.RunInput, 'from_file', lambda path: template)
    dataset = tmp_path / 'cases.jsonl'
    dataset.write_text('\n'.join(json.dumps({'instance_id': f'fixture__repo-{i}'}) for i in (1, 2)) + '\n')
    monkeypatch.setenv('SWE_DATASET_PATH', str(dataset))
    items = list(module.generate_configs())
    assert len(items) == 2
    for item in items:
        config = item.config
        for runtime in ('main', 'eval'):
            assert config.runtime_configs[runtime]['worker_venv']['extra_requirements'] == [
                *requirements,
                'dataclasses-json>=0.6.7',
            ]
        langfuse = config.processors['langfuse'].args
        assert langfuse['tags'] == ['cb-insiders', 'swe-smoke', 'fixture-model']
        assert langfuse['trace_metadata']['model'] == 'fixture-model'
    assert template.runtime_configs['main']['worker_venv']['extra_requirements'] == requirements
