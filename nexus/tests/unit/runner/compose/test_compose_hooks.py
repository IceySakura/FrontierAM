from __future__ import annotations

import argparse
import uuid

from pathlib import Path

import pytest

from nexus.runner.batch import BatchItem
from nexus.runner.batch.runner import _iter_generator_items
from nexus.cli.commands.run import register_run_parser
from nexus.runner.compose import RunComposer, parse_ordered_parts
from nexus.runner.compose.compose_hooks import (
    ComposeHookChain,
    ComposeHookError,
    ComposeHookSpec,
    compose_hook,
    parse_compose_hooks,
    postprocess_generated_items,
)
from nexus.runner.compose.resolver import resolve_non_data_part
from nexus.runner.single import RunInput


DATASET = Path(__file__).resolve().parents[4] / 'docs/examples/datasets/online-service-swe-verified-5'

HOOK_FILE = """
from nexus.runner.batch import BatchItem

CALLS = []

def create_compose_hook(params):
    seen = []

    class Hook:
        def process(self, item):
            seen.append(item.run_id)
            item.metadata.setdefault('__extra__', {})['label'] = params['label']
            item.metadata['__extra__']['timeout'] = item.config.runner.timeout
            item.metadata['__extra__']['position'] = len(seen)
            return item

        def finish(self):
            CALLS.append(list(seen))

    return Hook()
"""


def _item(run_id: str, **config) -> BatchItem:
    return BatchItem(run_id=run_id, config=RunInput(task_name='noop', **config))


def _write_hook(tmp_path: Path) -> Path:
    path = tmp_path / f'hook_{uuid.uuid4().hex}.py'
    path.write_text(HOOK_FILE)
    return path


def _hook_calls(path: Path) -> list:
    import hashlib
    import sys

    return sys.modules[f'nexus_compose_hook_{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]}'].CALLS


def test_hook_values_parse_in_order_and_stay_out_of_the_ordered_run_parts(tmp_path):
    values = [f'{tmp_path}/check.py:mode=fill,limit=3', 'registered:x=1']
    assert parse_compose_hooks(values) == [
        ComposeHookSpec(f'{tmp_path}/check.py', {'mode': 'fill', 'limit': 3}, f'--hook {values[0]}'),
        ComposeHookSpec('registered', {'x': 1}, '--hook registered:x=1'),
    ]
    parts = parse_ordered_parts(['--hook', values[0], '--use', 'worker_venv', '--set', 'runner.timeout=5'])
    assert [(part.kind, part.value) for part in parts] == [('use', 'worker_venv'), ('set', 'runner.timeout=5')]


def test_hooks_cannot_name_a_dataset():
    with pytest.raises(ValueError, match='pass it with --data'):
        parse_compose_hooks(['data:./dataset'])


def test_chain_shares_state_runs_in_order_and_finishes_once():
    events = []

    def counting(params):
        count = 0

        def process(item):
            nonlocal count
            count += 1
            events.append(('first', item.run_id, count))
            return item.model_copy(update={'metadata': {**item.metadata, 'count': count}})

        return process

    def finishing(params):
        class Hook:
            def process(self, item):
                events.append(('second', item.run_id, item.metadata['count']))
                return item

            def finish(self):
                events.append(('finish', params['tag']))

        return Hook()

    first, second = f'first_{uuid.uuid4().hex}', f'second_{uuid.uuid4().hex}'
    compose_hook(first)(counting)
    compose_hook(second)(finishing)
    chain = ComposeHookChain([ComposeHookSpec(first, {}, first), ComposeHookSpec(second, {'tag': 't'}, second)])
    results = [chain.apply(_item(run_id)) for run_id in ('a', 'b')]
    chain.finish()
    chain.finish()

    assert [item.metadata['count'] for item in results] == [1, 2]
    assert events == [('first', 'a', 1), ('second', 'a', 1), ('first', 'b', 2), ('second', 'b', 2), ('finish', 't')]


def test_rejections_name_the_hook_item_and_run_id():
    name = f'reject_{uuid.uuid4().hex}'

    def factory(params):
        def process(item):
            if item.run_id == 'bad':
                raise ValueError('missing cache_task_id')
            return item

        return process

    compose_hook(name)(factory)
    chain = ComposeHookChain([ComposeHookSpec(name, {}, f'--hook {name}')])
    chain.apply(_item('good'))
    with pytest.raises(
        ComposeHookError, match=rf"--hook {name} rejected item #1 \(run_id='bad'\): missing cache_task_id"
    ):
        chain.apply(_item('bad'))


@pytest.mark.parametrize(
    ('result', 'message'),
    [
        (lambda item: {'not': 'an item'}, 'must return a BatchItem'),
        (lambda item: item.model_copy(update={'config': item.config.model_copy(update={'task_name': 3})}), 'invalid'),
    ],
)
def test_hook_results_are_revalidated(result, message):
    name = f'result_{uuid.uuid4().hex}'
    compose_hook(name)(lambda params: result)
    with pytest.raises(ComposeHookError, match=message):
        ComposeHookChain([ComposeHookSpec(name, {}, name)]).apply(_item('a'))


def test_factory_failures_and_missing_files_are_reported(tmp_path):
    with pytest.raises(ComposeHookError, match='could not be created: compose hook file not found'):
        ComposeHookChain([ComposeHookSpec(str(tmp_path / 'missing.py'), {}, 'missing')])
    empty = tmp_path / 'empty.py'
    empty.write_text('')
    with pytest.raises(ComposeHookError, match='must define create_compose_hook'):
        ComposeHookChain([ComposeHookSpec(str(empty), {}, 'empty')])


def test_generated_items_get_static_overrides_before_hooks_and_finish_after_the_last(tmp_path):
    hook = _write_hook(tmp_path)
    parts = parse_ordered_parts(['--set', 'runner.timeout=7'])
    hooks = parse_compose_hooks([f'{hook}:label=gen'])
    items = list(
        postprocess_generated_items(
            [_item('a'), _item('b')], overrides=[resolve_non_data_part(p) for p in parts], hooks=hooks
        )
    )
    assert [item.config.runner.timeout for item in items] == [7, 7]
    assert [item.metadata['__extra__'] for item in items] == [
        {'label': 'gen', 'timeout': 7, 'position': 1},
        {'label': 'gen', 'timeout': 7, 'position': 2},
    ]
    assert _hook_calls(hook) == [['a', 'b']]


def test_generator_limit_is_applied_before_hooks(tmp_path):
    hook = _write_hook(tmp_path)

    class Loader:
        def generate(self):
            return iter([_item(f'case-{index}') for index in range(5)])

    hooks = parse_compose_hooks([f'{hook}:label=x'])
    selected = list(_iter_generator_items(Loader(), None, 2, hooks))
    assert [item.run_id for item in selected] == ['case-0', 'case-1']
    assert _hook_calls(hook) == [['case-0', 'case-1']]


async def test_dataset_compose_runs_hooks_after_all_static_parts(tmp_path):
    hook = _write_hook(tmp_path)
    parts = parse_ordered_parts(['--data', str(DATASET), '--set', 'runner.timeout=9'])
    async with RunComposer() as composer:
        items = await composer.compile_batch(
            parts, limit=3, compose_hooks=parse_compose_hooks([f'{hook}:label=compose'])
        )

    assert len(items) == 3
    assert all(item.config.runner.timeout == 9 for item in items)
    assert [item.metadata['__extra__']['position'] for item in items] == [1, 2, 3]
    assert all(item.metadata['__extra__']['timeout'] == 9 for item in items)
    assert all('dataset_item_id' in item.metadata for item in items)
    assert _hook_calls(hook) == [[item.run_id for item in items]]


@pytest.mark.parametrize(('command', 'accepted'), [('single', False), ('batch', True)])
def test_only_batch_runs_accept_hooks(command, accepted, capsys):
    parser = argparse.ArgumentParser()
    register_run_parser(parser.add_subparsers(dest='command'))
    argv = ['run', command, '--hook', 'check.py', '--set', 'task_name=noop']
    if accepted:
        assert parser.parse_args(argv).hook_values == ['check.py']
    else:
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
        assert 'unrecognized arguments: --hook' in capsys.readouterr().err
