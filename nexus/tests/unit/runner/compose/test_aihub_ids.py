from __future__ import annotations

import logging
import uuid

from pathlib import Path
from urllib.parse import parse_qsl

import pytest

from nexus.runner.batch import BatchItem
from nexus.runner.compose.compose_hooks import (
    ComposeHookChain,
    ComposeHookError,
    ComposeHookSpec,
    submission_host_env,
)
from nexus.runner.single import RunInput


AIHUB_IDS = str(Path(__file__).resolve().parents[4] / 'docs/examples/configs/compose-hooks/aihub_ids.py')
GLOBAL_ID = 'bf7fe007-e56b-4b5c-9d9e-ac4d800f7ffa'
MAIN_KEY = 'openai_clients.main.config.client_args.api_key'
SUMMARY_KEY = 'tool_configs.openai_clients.summary.config.client_args.api_key'


def _client(api_key):
    return {'name': 'litellm', 'config': {'client_args': {'api_key': api_key, 'max_retries': 10}}}


def _env_client(variable='NEXUS_TEST_HOOK_KEY'):
    return {'name': 'litellm', 'config': {'client_args': {'api_key__env': variable, 'max_retries': 10}}}


def _item(run_id='run-1', metadata=None, main=None, summary=None):
    config = RunInput(task_name='noop', openai_clients={'main': main or _client('credential?cache_task_id=task-a')})
    if summary is not None:
        config.tool_configs = {'openai_clients': {'summary': summary}}
    return BatchItem(run_id=run_id, metadata=metadata or {}, config=config)


def _chain(**params):
    params.setdefault('key', MAIN_KEY)
    return ComposeHookChain([ComposeHookSpec(AIHUB_IDS, params, 'aihub_ids')])


def _args(item):
    return item.config.openai_clients['main']['config']['client_args']


def _summary_args(item):
    return item.config.tool_configs['openai_clients']['summary']['config']['client_args']


def _query(args):
    return dict(parse_qsl(args['api_key'].partition('?')[2]))


def test_check_records_ids_without_changing_the_config():
    original = _item(
        main=_client(f'credential?provider=x&cache_task_id=task-a&acc_global_cache_id={GLOBAL_ID}'), metadata={'s': 1}
    )
    item = _chain(fields='cache_task_id+acc_global_cache_id').apply(original)
    assert item.metadata == {'s': 1, '__extra__': {'cache_task_id': 'task-a', 'acc_global_cache_id': GLOBAL_ID}}
    assert item.config == original.config


def test_check_reads_api_key_env_without_inlining_it(monkeypatch):
    monkeypatch.setenv('NEXUS_TEST_HOOK_KEY', f'app:secret?acc_global_cache_id={GLOBAL_ID}')
    item = _chain(fields='acc_global_cache_id').apply(_item(main=_env_client()))
    assert item.metadata['__extra__'] == {'acc_global_cache_id': GLOBAL_ID}
    assert _args(item) == {'api_key__env': 'NEXUS_TEST_HOOK_KEY', 'max_retries': 10}


@pytest.mark.parametrize(
    ('main', 'params', 'warning'),
    [
        (_client('credential?provider=x'), {}, 'has no cache_task_id'),
        (_client('k?cache_task_id=${oc.env:ID}'), {}, 'resolved only at run time'),
        (_env_client('NEXUS_TEST_UNSET_KEY'), {}, 'reads NEXUS_TEST_UNSET_KEY, which is set neither in --host-env'),
        (_client('k'), {'key': SUMMARY_KEY}, f'{SUMMARY_KEY} is not configured'),
    ],
)
def test_check_only_warns(main, params, warning, caplog):
    chain = _chain(**params)
    with caplog.at_level(logging.WARNING):
        item = chain.apply(_item(main=main))
        chain.finish()
    assert '__extra__' not in item.metadata
    assert item.config.openai_clients['main'] == main
    assert warning in caplog.text
    assert '1 of 1 items' in caplog.text


def test_check_warns_on_ids_shared_across_the_batch(caplog):
    chain = _chain()
    with caplog.at_level(logging.WARNING):
        chain.apply(_item('run-1'))
        chain.apply(_item('run-2'))
        chain.finish()
    assert "cache_task_id is also used by item #0 (run_id='run-1')" in caplog.text
    assert '1 of 2 items' in caplog.text


def test_unique_appends_per_item_ids_to_the_variable_reference(monkeypatch):
    monkeypatch.setenv('NEXUS_TEST_HOOK_KEY', 'app:secret?timeout=600')
    chain = _chain(fields='acc_global_cache_id', mode='unique')
    items = [chain.apply(_item(f'case-{index}', main=_env_client())) for index in range(3)]
    chain.finish()
    ids = [item.metadata['__extra__']['acc_global_cache_id'] for item in items]
    assert len(set(ids)) == 3
    for item, global_id in zip(items, ids, strict=True):
        assert str(uuid.UUID(global_id)) == global_id
        assert _args(item) == {
            'api_key': f'${{oc.env:NEXUS_TEST_HOOK_KEY}}&acc_global_cache_id={global_id}',
            'max_retries': 10,
        }
        assert 'secret' not in item.model_dump_json()


def test_unique_starts_the_query_when_the_variable_has_none(monkeypatch):
    monkeypatch.setenv('NEXUS_TEST_HOOK_KEY', 'app:secret')
    item = _chain(fields='cache_task_id+acc_global_cache_id', mode='unique').apply(_item(main=_env_client()))
    extra = item.metadata['__extra__']
    assert _args(item)['api_key'] == (
        f'${{oc.env:NEXUS_TEST_HOOK_KEY}}?cache_task_id={extra["cache_task_id"]}'
        f'&acc_global_cache_id={extra["acc_global_cache_id"]}'
    )


def test_unique_refuses_to_override_a_field_the_variable_sets_for_the_whole_batch(monkeypatch):
    monkeypatch.setenv('NEXUS_TEST_HOOK_KEY', 'app:secret?timeout=600&acc_global_cache_id=shared')
    with pytest.raises(
        ComposeHookError, match='NEXUS_TEST_HOOK_KEY already sets acc_global_cache_id for the whole batch'
    ):
        _chain(fields='acc_global_cache_id', mode='unique').apply(_item(main=_env_client()))


def test_written_references_resolve_at_run_time(monkeypatch):
    monkeypatch.setenv('NEXUS_TEST_HOOK_KEY', 'app:secret?timeout=600')
    item = _chain(fields='acc_global_cache_id', mode='unique').apply(_item(main=_env_client()))
    resolved = RunInput.from_dict(item.config.model_dump(mode='json'))
    value = item.metadata['__extra__']['acc_global_cache_id']
    assert _args(BatchItem(config=resolved))['api_key'] == f'app:secret?timeout=600&acc_global_cache_id={value}'


def test_unique_replaces_ids_in_literal_keys_and_keeps_other_parameters():
    main = _client('${oc.env:KEY}?provider=x&cache_task_id=shared&acc_allow_evict=1')
    chain = _chain(mode='unique')
    first, second = chain.apply(_item('a', main=main)), chain.apply(_item('b', main=main))
    for item in (first, second):
        assert _args(item)['api_key'] == (
            f'${{oc.env:KEY}}?provider=x&acc_allow_evict=1&cache_task_id={item.metadata["__extra__"]["cache_task_id"]}'
        )
    assert _query(_args(first)) != _query(_args(second))


def test_only_the_named_keys_are_touched(monkeypatch):
    monkeypatch.setenv('NEXUS_TEST_HOOK_KEY', 'app:main')
    summary = _env_client('NEXUS_TEST_UNSET_KEY')
    item = _chain(fields='acc_global_cache_id', mode='unique').apply(_item(main=_env_client(), summary=summary))
    assert 'acc_global_cache_id' in _query(_args(item))
    assert item.config.tool_configs['openai_clients']['summary'] == summary


@pytest.mark.parametrize('path', [SUMMARY_KEY, SUMMARY_KEY + '__env'])
def test_key_paths_select_the_key(monkeypatch, path):
    monkeypatch.setenv('NEXUS_TEST_SUMMARY_KEY', 'app:summary?provider=x')
    item = _item(summary=_env_client('NEXUS_TEST_SUMMARY_KEY'))
    result = _chain(key=path, fields='acc_global_cache_id', mode='unique').apply(item)
    value = result.metadata['__extra__']['acc_global_cache_id']
    assert _summary_args(result) == {
        'api_key': f'${{oc.env:NEXUS_TEST_SUMMARY_KEY}}&acc_global_cache_id={value}',
        'max_retries': 10,
    }
    assert _args(result) == _args(item)


def test_several_keys_share_the_item_id_and_are_recorded_per_key(monkeypatch):
    monkeypatch.setenv('NEXUS_TEST_HOOK_KEY', 'app:main')
    key = f'{MAIN_KEY}+{SUMMARY_KEY}'
    item = _chain(key=key, fields='acc_global_cache_id', mode='unique').apply(
        _item(main=_env_client(), summary=_client('app:summary'))
    )
    value = _query(_args(item))['acc_global_cache_id']
    assert _query(_summary_args(item))['acc_global_cache_id'] == value
    assert item.metadata['__extra__'] == {'acc_global_cache_id': {MAIN_KEY: value, SUMMARY_KEY: value}}


def test_unique_ids_are_deterministic_and_follow_seed_and_source():
    def assigned(run_id='case-0', **params):
        item = _item(run_id, metadata={'instance_id': 'astropy-12907'}, main=_client('credential'))
        return _query(_args(_chain(fields='acc_global_cache_id', mode='unique', **params).apply(item)))

    assert assigned() == assigned()
    assert assigned() != assigned('case-1')
    assert assigned(source='instance_id') == assigned('case-1', source='instance_id')
    assert assigned(source='instance_id') != assigned(source='instance_id', seed='s')
    with pytest.raises(ComposeHookError, match='requires metadata.instance_id'):
        _chain(mode='unique', source='instance_id').apply(_item('case-1'))


@pytest.mark.parametrize(
    ('main', 'params', 'message'),
    [
        (_env_client('NEXUS_TEST_UNSET_KEY'), {}, 'set neither in --host-env'),
        (_client('k'), {'key': SUMMARY_KEY}, 'is not configured'),
    ],
)
def test_unique_fails_when_it_cannot_set_the_id(main, params, message):
    with pytest.raises(ComposeHookError, match=message):
        _chain(mode='unique', **params).apply(_item(main=main))


def test_unique_rejects_items_that_would_share_an_id():
    chain = _chain(mode='unique')
    chain.apply(_item('same'))
    with pytest.raises(ComposeHookError, match=r"cache_task_id is also used by item #0 \(run_id='same'\)"):
        chain.apply(_item('same'))


def test_submissions_read_host_env_before_this_environment(monkeypatch):
    monkeypatch.setenv('NEXUS_TEST_HOOK_KEY', 'local:key')
    chain = _chain(fields='acc_global_cache_id', mode='unique')
    with submission_host_env({'NEXUS_TEST_HOOK_KEY': 'host:key?timeout=1'}):
        from_host = chain.apply(_item('a', main=_env_client()))
    assert _args(from_host)['api_key'].startswith('${oc.env:NEXUS_TEST_HOOK_KEY}&acc_global_cache_id=')
    monkeypatch.delenv('NEXUS_TEST_HOOK_KEY')
    with submission_host_env({'NEXUS_TEST_HOOK_KEY': 'host:key'}):
        assert _args(chain.apply(_item('b', main=_env_client())))['api_key'].startswith(
            '${oc.env:NEXUS_TEST_HOOK_KEY}?acc_global_cache_id='
        )


@pytest.mark.parametrize(
    ('params', 'message'),
    [
        ({'fields': 'cache_task_id+bad name'}, 'query parameter names'),
        ({'mode': 'fill'}, 'mode must be check or unique'),
        ({'key': 'OPENAI_API_KEY'}, 'must be config paths'),
        ({'key': ''}, 'must be "\\+"-separated names'),
        ({'keys': 'OPENAI_API_KEY'}, 'does not accept parameters'),
    ],
)
def test_invalid_parameters_fail_before_any_item(params, message):
    with pytest.raises(ComposeHookError, match=message):
        _chain(**params)


def test_default_key_is_the_main_client_api_key_or_its_variable(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'app:main?timeout=600')
    summary = _env_client('SUMMARY_OPENAI_API_KEY')
    item = _item(main=_env_client('OPENAI_API_KEY'), summary=summary)
    chain = ComposeHookChain([
        ComposeHookSpec(AIHUB_IDS, {'fields': 'acc_global_cache_id', 'mode': 'unique'}, 'aihub_ids')
    ])
    result = chain.apply(item)
    value = result.metadata['__extra__']['acc_global_cache_id']
    assert _args(result) == {'api_key': f'${{oc.env:OPENAI_API_KEY}}&acc_global_cache_id={value}', 'max_retries': 10}
    assert result.config.tool_configs['openai_clients']['summary'] == summary


def test_paths_may_name_any_field():
    item = _item()
    item.config.tool_configs = {'gateway': {'token': 'app:key?provider=x'}}
    result = _chain(key='tool_configs.gateway.token', fields='acc_global_cache_id', mode='unique').apply(item)
    value = result.metadata['__extra__']['acc_global_cache_id']
    assert result.config.tool_configs['gateway']['token'] == f'app:key?provider=x&acc_global_cache_id={value}'


def test_oc_env_references_in_the_query_are_resolved_like_env_fields(monkeypatch):
    monkeypatch.setenv('NEXUS_TEST_CACHE_ID', 'from-env')
    main = _client('${oc.env:KEY}?cache_task_id=${oc.env:NEXUS_TEST_CACHE_ID}')
    assert _chain().apply(_item(main=main)).metadata['__extra__'] == {'cache_task_id': 'from-env'}
    with submission_host_env({'NEXUS_TEST_CACHE_ID': 'from-host-env'}):
        assert _chain().apply(_item('b', main=main)).metadata['__extra__'] == {'cache_task_id': 'from-host-env'}
    assigned = _chain(mode='unique').apply(_item('c', main=main))
    assert _args(assigned)['api_key'] == (
        f'${{oc.env:KEY}}?cache_task_id={assigned.metadata["__extra__"]["cache_task_id"]}'
    )


def _two_hooks(*params):
    return ComposeHookChain([
        ComposeHookSpec(AIHUB_IDS, {'key': MAIN_KEY, **step}, f'aihub_ids #{index}')
        for index, step in enumerate(params)
    ])


def _resolved_key(item):
    return _args(BatchItem(config=RunInput.from_dict(item.config.model_dump(mode='json'))))['api_key']


@pytest.mark.parametrize(
    ('variable', 'separator'), [('credential?provider=x', '&'), ('credential', '?')], ids=['with-query', 'no-query']
)
def test_chained_hooks_keep_extending_the_variable_reference(monkeypatch, variable, separator):
    monkeypatch.setenv('NEXUS_TEST_HOOK_KEY', variable)
    chain = _two_hooks(
        {'fields': 'cache_task_id', 'mode': 'unique'},
        {'fields': 'acc_global_cache_id', 'mode': 'unique'},
    )
    item = chain.apply(_item(main=_env_client()))
    extra = item.metadata['__extra__']
    first, second = extra['cache_task_id'], extra['acc_global_cache_id']
    assert _args(item)['api_key'] == (
        f'${{oc.env:NEXUS_TEST_HOOK_KEY}}{separator}cache_task_id={first}&acc_global_cache_id={second}'
    )
    assert _resolved_key(item) == f'{variable}{separator}cache_task_id={first}&acc_global_cache_id={second}'


def test_a_later_hook_reads_and_replaces_ids_an_earlier_hook_appended(monkeypatch):
    monkeypatch.setenv('NEXUS_TEST_HOOK_KEY', 'credential?provider=x')
    checked = _two_hooks({'fields': 'cache_task_id', 'mode': 'unique'}, {'fields': 'cache_task_id'}).apply(
        _item(main=_env_client())
    )
    assigned = _query({'api_key': _resolved_key(checked)})['cache_task_id']
    assert checked.metadata['__extra__'] == {'cache_task_id': assigned}

    reseeded = _two_hooks(
        {'fields': 'cache_task_id', 'mode': 'unique'},
        {'fields': 'cache_task_id', 'mode': 'unique', 'seed': 'again'},
    ).apply(_item(main=_env_client()))
    value = reseeded.metadata['__extra__']['cache_task_id']
    assert value != assigned
    assert _args(reseeded)['api_key'] == f'${{oc.env:NEXUS_TEST_HOOK_KEY}}&cache_task_id={value}'


def test_chained_hooks_still_refuse_fields_the_variable_sets(monkeypatch):
    monkeypatch.setenv('NEXUS_TEST_HOOK_KEY', 'credential?provider=x&acc_global_cache_id=shared')
    chain = _two_hooks(
        {'fields': 'cache_task_id', 'mode': 'unique'},
        {'fields': 'acc_global_cache_id', 'mode': 'unique'},
    )
    with pytest.raises(ComposeHookError, match='already sets acc_global_cache_id for the whole batch'):
        chain.apply(_item(main=_env_client()))
