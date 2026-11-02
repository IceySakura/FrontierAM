import logging

from typing import Any

import pytest

from nexus.core.registry import EntryNotFoundError
from nexus.dataset.item import DatasetItem, RemoteDatasetRef
from nexus.dataset.layout import create_dataset_template
from nexus.runner.compose import (
    RunComposer,
    RunPart,
    compile_batch_file,
    compile_single,
    compile_single_file,
    parse_ordered_parts,
    run_part,
    run_part_registry,
)
from nexus.runner.compose.model import RawRunPart
from nexus.runner.compose.resolver import resolve_non_data_part


def test_compile_single_from_local_dataset_sets_task_files(tmp_path):
    dataset_root = create_dataset_template(tmp_path / 'my_terminal', template='terminal', item_id='case-001')

    run_input = compile_single(
        parse_ordered_parts(['run', 'single', '--data', str(dataset_root), '--id', 'case-001']),
        item_id='case-001',
    )

    assert run_input.task_name == 'terminal_task'
    assert run_input.task_files is not None
    assert run_input.task_files.type == 'directory'
    assert run_input.task_files.path == str(dataset_root / 'items' / 'case-001' / 'files')
    assert 'task_dir' not in run_input.task_config


def test_compile_single_file_composes_local_dataset_and_inline_input(tmp_path):
    dataset_root = create_dataset_template(tmp_path / 'my_dataset', template='generic', item_id='case-001')
    compose_path = tmp_path / 'run.yaml'
    compose_path.write_text(
        f"""
compose:
  - data:
      uri: {dataset_root.name}
      id: case-001
  - use: worker_venv
  - input:
      task_config:
        message: overridden
""".lstrip(),
        encoding='utf-8',
    )

    run_input = compile_single_file(compose_path)

    assert run_input.task_name == 'echo'
    assert run_input.task_config == {'message': 'overridden', 'run_oracle': True}
    assert run_input.runtime_configs['main']['worker_venv'] is True
    assert run_input.task_files is not None
    assert run_input.task_files.path == str(dataset_root / 'items' / 'case-001' / 'files')


def test_compile_batch_file_expands_local_dataset(tmp_path):
    dataset_root = create_dataset_template(tmp_path / 'my_dataset', template='generic')
    compose_path = tmp_path / 'batch.yaml'
    compose_path.write_text(
        f"""
compose:
  - data:
      uri: {dataset_root.name}
      split: test
  - input:
      task_config:
        source: compose
""".lstrip(),
        encoding='utf-8',
    )

    items = compile_batch_file(compose_path)

    assert [item.run_id for item in items] == ['case-001', 'case-002', 'case-003']
    assert all(item.config.task_config['source'] == 'compose' for item in items)


@pytest.mark.asyncio
async def test_run_composer_compiles_and_caches_remote_dataset_item(tmp_path):
    remote_ref = RemoteDatasetRef(namespace='research', dataset='source_data', version='v3', split='test')
    item = DatasetItem.from_mapping(
        {
            'id': 'sample-42',
            'body': {'task_name': 'identity', 'task_config': {}},
            'files': {'type': 'zip', 'path': 'blobs/sample-42.zip', 'size': 1024},
        },
        remote_ref=remote_ref,
    )
    compose_path = tmp_path / 'remote.yaml'
    compose_path.write_text(
        """
compose:
  - data:
      uri: research/source_data@v3
      split: test
      id: sample-42
""".lstrip(),
        encoding='utf-8',
    )

    class FakeDatasetClient:
        def __init__(self):
            self.calls = 0

        async def resolve_version(self, uri):
            assert uri.display == 'research/source_data@v3'
            return 'v3'

        async def resolve_split(self, uri, *, version, split=None):
            assert uri.display == 'research/source_data@v3'
            assert version == 'v3'
            return split

        async def iter_items(self, uri, *, split=None, item_id=None, limit=None, page_size=500):
            self.calls += 1
            assert uri.display == 'research/source_data@v3'
            assert split == 'test'
            assert item_id == 'sample-42'
            assert limit is None
            assert page_size == 500
            yield item

    client = FakeDatasetClient()
    async with RunComposer(dataset_client=client) as composer:
        first = await composer.compile_single_file(compose_path)
        second = await composer.compile_single_file(compose_path)

    assert first.task_name == second.task_name == 'identity'
    assert first.task_files is not None
    assert first.task_files.remote == remote_ref
    assert client.calls == 1


@pytest.mark.asyncio
async def test_run_composer_limit_zero_stays_empty_after_item_cache_is_warm(tmp_path):
    dataset_root = create_dataset_template(tmp_path / 'dataset', template='generic', item_id='case-001')
    parts = parse_ordered_parts(['--data', str(dataset_root)])

    async with RunComposer() as composer:
        assert len(await composer.compile_batch(parts)) == 1
        assert await composer.compile_batch(parts, item_id='case-001', limit=0) == []


def test_compile_single_logs_dataset_item_loading(tmp_path, caplog):
    dataset_root = create_dataset_template(tmp_path / 'my_terminal', template='terminal', item_id='case-001')

    with caplog.at_level(logging.INFO, logger='nexus.runner.compose.compiler'):
        compile_single(
            parse_ordered_parts(['run', 'single', '--data', str(dataset_root), '--id', 'case-001']),
            item_id='case-001',
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any('Loading dataset items:' in message for message in messages)
    assert any('Loaded dataset items:' in message and 'ids=case-001' in message for message in messages)
    assert any('Applying dataset item: id=case-001 task=terminal_task' in message for message in messages)
    assert any(
        'Dataset item files attached: id=case-001 type=directory' in message and 'remote=<local>' in message
        for message in messages
    )


def test_compile_single_honors_command_line_order(tmp_path):
    dataset_root = create_dataset_template(tmp_path / 'my_dataset', template='generic', item_id='case-001')

    earlier_set = compile_single(
        parse_ordered_parts([
            'run',
            'single',
            '--data',
            str(dataset_root),
            '--set',
            'runtime_configs.main.worker_venv=false',
            '--use',
            'worker_venv',
        ])
    )
    later_set = compile_single(
        parse_ordered_parts([
            'run',
            'single',
            '--data',
            str(dataset_root),
            '--use',
            'worker_venv',
            '--set',
            'runtime_configs.main.worker_venv=false',
        ])
    )

    assert earlier_set.runtime_configs['main']['worker_venv'] is True
    assert later_set.runtime_configs['main']['worker_venv'] is False


def test_run_part_registry_discovers_compose_parts():
    entry = run_part_registry.get_entry('claude_code')
    chat_tracker_entry = run_part_registry.get_entry('chat_tracker')
    openai_entry = run_part_registry.get_entry('openai')
    litellm_entry = run_part_registry.get_entry('litellm')

    assert entry.import_path.split('::', 1)[0] == 'nexus.extensions.agents.claude_code.run_parts'
    assert chat_tracker_entry.import_path.split('::', 1)[0] == 'nexus.trace.processors.chat_tracker'
    assert openai_entry.import_path.split('::', 1)[0] == 'nexus.extensions.openai_clients.openai'
    assert litellm_entry.import_path.split('::', 1)[0] == 'nexus.extensions.openai_clients.litellm'
    with pytest.raises(EntryNotFoundError):
        run_part_registry.get_entry('chat')


def test_compile_single_uses_registered_agent_and_llm_parts(monkeypatch: pytest.MonkeyPatch):
    for name in (
        'GONGFENG_RUNTIME_BASE_URL',
        'GONGFENG_RUNTIME_TOKEN',
        'OPENAI_API_KEY',
        'OPENAI_MODEL',
    ):
        monkeypatch.delenv(name, raising=False)

    run_input = compile_single(
        parse_ordered_parts([
            'run',
            'single',
            '--set',
            'task_name=identity',
            '--use',
            'claude_code',
            '--use',
            'openai',
            '--use',
            'gongfeng',
            '--use',
            'worker_venv',
        ])
    )

    assert run_input.imports == []
    assert run_input.agent_name == 'claude_code'
    assert run_input.agent_config['version'] == '2.1.153'
    assert run_input.agent_config['npm_registry'] == 'https://registry.npmmirror.com/'
    assert run_input.agent_config['extra_cli_args'] == ['--output-format', 'json', '--verbose']
    assert run_input.openai_clients['main']['name'] == 'openai'
    assert run_input.openai_clients['main']['config']['request_args']['model'] == ''
    assert run_input.runtime_provider_name == 'gongfeng'
    assert run_input.runtime_provider_config == {
        'api_base_url': 'http://hyrl-sandbox.prod.woa.com:8052',
        'token': '',
    }
    assert run_input.runtime_configs['main']['worker_venv'] is True


def test_registered_processor_part_uses_static_args():
    part = resolve_non_data_part(
        RawRunPart(
            kind='use',
            value='chat_tracker',
            source='--use chat_tracker',
        )
    )

    assert part.body == {
        'processors': {
            'chat': {
                'name': 'chat_tracker',
                'args': {'tools_mode': 'branch_last'},
            }
        }
    }


def test_registered_run_part_can_return_dict():
    @run_part('dict_return_test')
    def dict_return_test(params: dict[str, Any]) -> dict[str, Any]:
        assert params == {'flag': True}
        return {'task_name': 'identity'}

    part = resolve_non_data_part(
        RawRunPart(
            kind='use',
            value='dict_return_test:flag=true',
            source='--use dict_return_test:flag=true',
        )
    )

    assert isinstance(part, RunPart)
    assert part.source == 'dict_return_test'
    assert part.body == {'task_name': 'identity'}


def test_registered_run_part_rejects_params():
    with pytest.raises(ValueError, match='does not accept parameters'):
        resolve_non_data_part(
            RawRunPart(
                kind='use',
                value='chat_tracker:file_path=/tmp/trace.jsonl',
                source='--use chat_tracker:file_path=/tmp/trace.jsonl',
            )
        )


def test_use_yaml_file_with_run_part_shape(tmp_path):
    path = tmp_path / 'part.yaml'
    path.write_text(
        """
body:
  imports:
    - nexus.extensions.tasks.example.task
  runtime_configs:
    main:
      worker_venv: true
merge:
  imports: append_unique
source: custom-part
""".lstrip(),
        encoding='utf-8',
    )

    part = resolve_non_data_part(RawRunPart(kind='use', value=str(path), source=f'--use {path}'))

    assert part.body == {
        'imports': ['nexus.extensions.tasks.example.task'],
        'runtime_configs': {'main': {'worker_venv': True}},
    }
    assert part.merge == {'imports': 'append_unique'}
    assert part.source == 'custom-part'


def test_use_yaml_file_with_run_input_shape(tmp_path):
    path = tmp_path / 'run_input.yaml'
    path.write_text(
        """
task_name: identity
task_config:
  message: hello
runtime_configs:
  main:
    worker_venv: true
""".lstrip(),
        encoding='utf-8',
    )

    run_input = compile_single(parse_ordered_parts(['run', 'single', '--use', str(path)]))

    assert run_input.task_name == 'identity'
    assert run_input.task_config == {'message': 'hello'}
    assert run_input.runtime_configs['main']['worker_venv'] is True


def test_use_yaml_file_rejects_mapping_that_is_neither_run_part_nor_run_input(tmp_path):
    path = tmp_path / 'invalid.yaml'
    path.write_text(
        """
task_config:
  message: missing task name
""".lstrip(),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='must contain either a RunPart mapping or a RunInput mapping'):
        resolve_non_data_part(RawRunPart(kind='use', value=str(path), source=f'--use {path}'))
