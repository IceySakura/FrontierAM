"""Persistent coverage for model YAMLs, instance cache identity and Harbor evidence.

Only external execution/model boundaries are mocked; no credentials or live services.
"""

from __future__ import annotations

import copy
import json

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs

import pytest
import yaml

from nexus.extensions.tasks.artifact import harbor_entry
from nexus.extensions.tasks.artifact.judges import profiles
from nexus.extensions.tasks.artifact.judges.base import JudgeInput, JudgeResult
from nexus.extensions.tasks.artifact.judges.registry import get_judge


# AIHub routes a bare `deepseek-flash` to another vendor's model; api_<provider>_<name> pins DeepSeek's.
DEEPSEEK_MODEL = 'openai/api_deepseek_deepseek-flash'

EXPECTED_ROLES = {
    'web_judger': ('agent', 'vlm'),
    'aesthetic_general': ('agent', 'vlm'),
    'aesthetic': ('vlm',),
    'svg': ('vlm',),
    'visual_anchored': ('vlm',),
    'baseline_check': ('agent',),
    'code_quality': ('agent',),
    'general_agent': ('agent',),
}


@pytest.fixture(autouse=True)
def fake_environment(monkeypatch):
    for name in ('JUDGE_API_KEY', 'JUDGE_CACHE_TASK_ID', 'T3_JUDGE_NAME', 'T3_JUDGE_MODELS',
                 'NEXUS_OPENAI_CLIENTS', 'T3_JUDGE_PROFILE', 'T3_JUDGE_EVIDENCE',
                 'T3_JUDGE_EVIDENCE_UPLOAD', 'T3_AGENT_RESULT_PATH'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('JUDGE_API_KEY', 'fake-key?provider=test&cache_task_id=old')
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    monkeypatch.setenv('NEXUS_AUTO_INSTALL', '0')
    import nexus.extensions.openai_clients.litellm as client_module

    monkeypatch.setattr(client_module.litellm, 'acompletion',
                        AsyncMock(side_effect=AssertionError('unexpected model call')))


@pytest.mark.parametrize(('name', 'roles'), EXPECTED_ROLES.items())
def test_bundled_judges_bind_exact_roles_and_models(name, roles):
    spec = get_judge(name)
    selected = profiles.select_models(name)
    assert spec.requires == roles
    assert set(spec.model_files) == set(roles)
    assert selected.models == {
        role: DEEPSEEK_MODEL if role == 'agent' else 'openai/gemini-3.8-flash'
        for role in roles
    }
    assert set(selected.provenance()['bindings']) == set(roles)


@pytest.mark.parametrize(
    ('stem', 'model'),
    [
        ('deepseek-flash', DEEPSEEK_MODEL),
        ('gemini-3.5-flash', 'openai/gemini-3.5-flash'),
        ('gemini-3.7-flash', 'openai/gemini-3.7-flash'),
        ('gemini-3.8-flash', 'openai/gemini-3.8-flash'),
    ],
)
def test_model_yaml_uses_standard_main_and_preserves_request_policy(stem, model):
    loaded = profiles.load_model(f'{stem}-aihub-2.0.yaml')
    spec = loaded.client_spec
    assert spec['name'] == 'litellm'
    assert 'nexus.extensions.openai_clients.litellm' in loaded.imports
    assert loaded.model == model
    args = spec['config']['request_args']
    assert args['drop_params'] is True
    assert args['max_tokens'] == 65536
    assert spec['config']['enforced_request_args'] == {'model': model, 'max_tokens': 65536}
    assert loaded.agent_config['haiku_model'] == loaded.model
    assert loaded.agent_config['no_thinking_request_args']['model'] == loaded.model
    assert 'cache_task_id' not in spec['config']['client_args']
    assert 'provider' not in spec['config']['client_args']
    assert spec['config']['client_args']['api_key__env'] == 'JUDGE_API_KEY'
    if stem.startswith('gemini'):
        assert args['extra_body'] == {'enable_thinking': True}
    else:
        assert args['extra_body'] == {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}


@pytest.mark.parametrize('role', ['main', 'judge', 'flow2code', 'scoring', 'typo'])
def test_model_overrides_cannot_use_undeclared_roles(role):
    with pytest.raises(profiles.ProfileError, match='declared Judge roles'):
        profiles.select_models('web_judger', {role: 'gemini-3.5-flash-aihub-2.0.yaml'})


def test_one_role_override_does_not_replace_other_role():
    selected = profiles.select_models('web_judger', {'vlm': 'gemini-3.5-flash-aihub-2.0.yaml'})
    assert selected.models == {'agent': DEEPSEEK_MODEL, 'vlm': 'openai/gemini-3.5-flash'}


@pytest.mark.parametrize('filename', ['../private.yaml', '/tmp/private.yaml', '<MODEL>', '', None,
                                      'unknown-aihub-2.0.yaml'])
def test_model_path_is_restricted_to_bundled_files(filename):
    with pytest.raises(profiles.ProfileError):
        profiles.load_model(filename)


@pytest.mark.parametrize('mutation', ['literal-key', 'extra-role', 'runtime-budget', 'placeholder', 'invalid-json-key'])
def test_invalid_model_config_is_rejected(tmp_path, monkeypatch, mutation):
    filename = 'test-aihub-2.0.yaml'
    body = yaml.safe_load((profiles.MODEL_DIR / 'gemini-3.5-flash-aihub-2.0.yaml').read_text())
    spec = body['body']['openai_clients']['main']['config']
    if mutation == 'literal-key':
        spec['client_args']['api_key'] = 'private-invalid-value'
    elif mutation == 'extra-role':
        body['body']['openai_clients']['other'] = copy.deepcopy(body['body']['openai_clients']['main'])
    elif mutation == 'runtime-budget':
        body['body']['agent_config']['timeout'] = 1
    elif mutation == 'placeholder':
        spec['request_args']['model'] = '<UNSET_MODEL>'
    else:
        spec['client_args']['api_key__env'] = 'not a variable name'
    (tmp_path / filename).write_text(yaml.safe_dump(body))
    monkeypatch.setattr(profiles, 'MODEL_DIR', tmp_path)
    with pytest.raises(profiles.ProfileError) as exc:
        profiles.load_model(filename)
    assert 'private-invalid-value' not in str(exc.value)


@pytest.mark.parametrize('secret', [None, '', '   ', '<AIHUB_KEY>'])
def test_missing_or_placeholder_key_is_rejected(monkeypatch, secret):
    if secret is None:
        monkeypatch.delenv('JUDGE_API_KEY')
    else:
        monkeypatch.setenv('JUDGE_API_KEY', secret)
    with pytest.raises(profiles.ProfileError, match='JUDGE_API_KEY'):
        profiles.select_models('web_judger')


def test_cache_identity_is_data_scoped_without_mutating_model_config(monkeypatch):
    secret = 'fake-"quote"-\\backslash?provider=test&cache_task_id=old'
    monkeypatch.setenv('JUDGE_API_KEY', secret)
    selected = profiles.select_models('web_judger')
    original = copy.deepcopy(selected)
    ids = ['case-A', 'case-A', 'case-B', '数据 & case/1']
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(lambda iid: selected.build_clients(instance_id=iid), ids))
    for iid, clients in zip(ids, groups, strict=True):
        assert clients['agent'] is not clients['vlm']
        for role, client in clients.items():
            assert client._client_key == role
            assert client.config.client_args['cache_task_id'] == iid
            key, _, query = client.client_args['api_key'].partition('?')
            assert key == secret.partition('?')[0]
            assert parse_qs(query) == {'provider': ['test'], 'cache_task_id': [iid]}
            assert 'cache_task_id' not in client.client_args
    assert selected == original
    assert selected.provenance() == original.provenance()
    assert selected.bindings['agent'].client_spec['config']['client_args']['api_key__env'] == 'JUDGE_API_KEY'


@pytest.mark.parametrize('iid', [None, False, 12, '', '   ', '<uuid>', '${INSTANCE}'])
def test_missing_or_placeholder_instance_id_cannot_construct_clients(iid):
    with pytest.raises(profiles.ProfileError, match='instance_id'):
        profiles.select_models('web_judger').build_clients(instance_id=iid)


@pytest.mark.parametrize('override', ['', '   ', 'experiment&id=1'])
def test_optional_cache_override_and_fingerprint(monkeypatch, override):
    baseline = profiles.select_models('web_judger').provenance()
    monkeypatch.setenv('JUDGE_CACHE_TASK_ID', override)
    selected = profiles.select_models('web_judger')
    clients = selected.build_clients(instance_id='case-A')
    assert selected.provenance() == baseline
    assert {c.config.client_args['cache_task_id'] for c in clients.values()} == {override.strip() or 'case-A'}


@pytest.mark.parametrize('override', ['<uuid>', '${UNRESOLVED}'])
def test_unresolved_cache_override_is_rejected(monkeypatch, override):
    monkeypatch.setenv('JUDGE_CACHE_TASK_ID', override)
    with pytest.raises(profiles.ProfileError, match='JUDGE_CACHE_TASK_ID'):
        profiles.select_models('web_judger')


def test_same_model_in_two_roles_builds_independent_clients():
    selected = profiles.select_models(
        'web_judger', dict.fromkeys(('agent', 'vlm'), 'gemini-3.5-flash-aihub-2.0.yaml'),
    )
    clients = selected.build_clients(instance_id='sample')
    assert clients['agent'] is not clients['vlm']
    assert clients['agent'].config.client_args['cache_task_id'] == clients['vlm'].config.client_args['cache_task_id']


async def test_agent_turns_and_direct_vlm_use_instance_cache_and_preserve_usage(monkeypatch):
    import nexus.extensions.openai_clients.litellm as client_module

    from nexus.extensions.tasks.artifact.judges.capabilities.agent_direct_gateway import LLMHandlerBridge
    from nexus.extensions.tasks.artifact.judges.capabilities.llm import multimodal_completion

    calls = []

    async def completion(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(model_dump=lambda: {
            'id': 'fake-completion', 'object': 'chat.completion', 'created': 1,
            'model': kwargs['model'],
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}],
            'usage': {'prompt_tokens': 128, 'completion_tokens': 1, 'total_tokens': 129,
                      'prompt_tokens_details': {'cached_tokens': 64}},
        })

    monkeypatch.setattr(client_module.litellm, 'acompletion', completion)
    monkeypatch.setattr(LLMHandlerBridge, '_run_totals', {})
    clients = profiles.select_models('web_judger').build_clients(instance_id='case-A')
    bridge = LLMHandlerBridge(clients['agent'])
    messages = [{'role': 'user', 'content': 'Inspect source'}]
    await bridge.request({'model': 'cli-model', 'messages': messages})
    await bridge.request({'model': 'cli-model', 'messages': [
        *messages, {'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': 'Continue'},
    ]})
    assert await multimodal_completion(clients['vlm'], [{'type': 'text', 'text': 'Judge page'}]) == 'ok'
    assert len(calls) == 3
    for call in calls:
        assert parse_qs(call['api_key'].partition('?')[2])['cache_task_id'] == ['case-A']
        assert 'cache_task_id' not in call
        assert 'provider' not in call
        assert call['drop_params'] is True
    assert [call['model'] for call in calls] == [DEEPSEEK_MODEL, DEEPSEEK_MODEL, 'openai/gemini-3.8-flash']
    assert bridge.usage['calls'] == 2
    assert bridge.usage['cache_read_input_tokens'] == 128  # Synthetic usage, not live cache hits.


async def test_flow2code_applies_model_settings_without_losing_execution_budget(monkeypatch):
    from nexus.extensions.tasks.artifact.judges.capabilities import agent
    from nexus.extensions.tasks.artifact.judges.strategies.web_judger import executor

    model = profiles.load_model('gemini-3.5-flash-aihub-2.0.yaml')
    raw = json.dumps({'scripts': [{'fid': 'F1', 'kind': 'source_code', 'code': 'true'}]})
    run = AsyncMock(return_value=SimpleNamespace(
        raw=raw, agent_error=None, read_error=None,
        cc=SimpleNamespace(execution_time=1, output=raw, trajectories={}),
    ))
    monkeypatch.setattr(agent, 'run_agent_and_read_reward', run)
    client, runtime = object(), object()
    scripts, _ = await executor.flow2code(
        {'flows': [{'fid': 'F1'}]}, runtime=runtime, runtime_name='case-A', archive_b64='',
        problem_statement='Inspect source', file_list=['index.html'], openai_client=client,
        source_dir='/workspace', timeout=123, max_turns=7, model_config=model,
    )
    assert len(scripts) == 1
    kwargs = run.await_args.kwargs
    assert kwargs['openai_client'] is client
    assert kwargs['prepared_runtime'] is runtime
    launch = kwargs['claude_input']
    assert launch.timeout == 123
    assert '--max-turns' in launch.extra_cli_args
    assert launch.extra_cli_args[launch.extra_cli_args.index('--max-turns') + 1] == '7'
    assert launch.pre_start_scripts
    assert launch.thinking_preset == model.agent_config['thinking_preset']
    assert launch.haiku_model == model.model
    assert launch.no_thinking_request_args_mode == 'inherit'


async def test_composite_children_inherit_clients_and_models(monkeypatch):
    from nexus.extensions.tasks.artifact.judges.strategies import aesthetic_general

    inp = JudgeInput(instance_id='case-A', problem_statement='test',
                     openai_clients={'agent': object(), 'vlm': object()},
                     model_configs={'agent': object(), 'vlm': object()})
    aesthetic = AsyncMock(return_value=JudgeResult(judge_name='aesthetic', score=0.6))
    general = AsyncMock(return_value=JudgeResult(judge_name='general_agent', score=0.8))
    monkeypatch.setattr(aesthetic_general, 'get_judge', lambda name: SimpleNamespace(
        fn={'aesthetic': aesthetic, 'general_agent': general}[name],
    ))
    result = await aesthetic_general.compute_score(inp)
    assert result.score == 0.7
    assert aesthetic.await_args.args[0] is inp
    child = general.await_args.args[0]
    assert child.openai_clients is inp.openai_clients
    assert child.model_configs is inp.model_configs
    assert child.instance['general_agent_lite'] is True
    general.reset_mock()
    aesthetic.return_value = JudgeResult(judge_name='aesthetic', score=0)
    result = await aesthetic_general.compute_score(inp)
    assert result.score == 0
    general.assert_not_awaited()


@pytest.mark.parametrize('upload', [None, {'url': 'cos://bucket/evidence.tar.gz'}, {'error': 'upload unavailable'}])
async def test_harbor_persists_verdict_and_instance_bound_clients(tmp_path, monkeypatch, upload):
    from nexus.extensions.tasks.artifact.judges import registry

    tests = tmp_path / 'tests'
    tests.mkdir()
    (tests / 'instruction.md').write_text('Check SVG')
    monkeypatch.setattr(harbor_entry, 'TESTS_DIR', tests)
    monkeypatch.setenv('T3_INSTANCE_ID', 'case-A')
    monkeypatch.setenv('T3_JUDGE_NAME', 'web_judger')
    monkeypatch.setenv('T3_JUDGE_EVIDENCE', '1')
    monkeypatch.setenv('ARTIFACT_SAVE_SCREENSHOTS_DIR', str(tmp_path / 'evidence'))
    runtime = object()
    monkeypatch.setattr('nexus.runtime.client.local.LocalRuntime', lambda: runtime)
    monkeypatch.setattr(harbor_entry, 'collect_archive', AsyncMock(return_value=('', ['index.html'])))
    uploader = Mock(return_value=upload)
    monkeypatch.setattr(harbor_entry, 'upload_evidence', uploader)
    metadata = {'real_url': 'http://real', 'proxy_url': 'http://proxy',
                'verdicts': [{'fid': 'F1', 'evidence_refs': ['F1/main']}]}

    async def judge(name, inp):
        assert name == 'web_judger'
        assert inp.instance_id == 'case-A'
        assert inp.runtime is runtime
        assert set(inp.model_configs) == {'agent', 'vlm'}
        for client in inp.openai_clients.values():
            assert client.config.client_args['cache_task_id'] == inp.instance_id
        return JudgeResult(judge_name=name, score=0.7, metadata=metadata)

    monkeypatch.setattr(registry, 'run_judge', judge)
    assert await harbor_entry.run(workdir='/workspace', log_dir=tmp_path / 'logs') == 0
    reward = json.loads((tmp_path / 'logs/reward.json').read_text())
    evidence = json.loads((tmp_path / 'evidence/case-A/judge_results.json').read_text())
    assert reward['reward'] == evidence['final_score'] == 0.7
    for key, value in metadata.items():
        assert reward['judge_metadata'][key] == evidence['judges'][0]['metadata'][key] == value
    assert reward.get('evidence_archive') == upload
    assert evidence['instance_id'] == 'case-A'
    assert set(reward['judge_profile']['bindings']) == {'agent', 'vlm'}
    assert 'fake-key' not in json.dumps(reward)
    uploader.assert_called_once_with(str(tmp_path / 'evidence/case-A'), 'case-A')
