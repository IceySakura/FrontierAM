"""OSWorld platform contract regressions."""

from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any

import pytest


FIXTURE_SECRET = 'fixture-secret-not-real'


FIXTURE_GATEWAY_KEY = f'sk-{FIXTURE_SECRET}'


TASK_IDS = ('task-a', 'task-b', 'task-c')


@pytest.fixture
def exporter_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A three-task V1 checkout, with the developer's own environment out of the way.

    Both entries now have a local .env overlay. A developer's credential file
    must not override the fixture's paths or model settings.
    """
    for key in list(os.environ):
        if key.startswith(('CC_', 'CODEX_', 'OSWORLD_', 'HUNYUAN_', 'NEXUS_', 'AGS_', 'AIHUB_')) or key in {
            'AGENT_KIND',
            'TASK_TIMEOUT',
            'RUNNER_TIMEOUT',
            'EVAL_TIMES',
            'CLAUDE_CODE_VERSION',
        }:
            monkeypatch.delenv(key)

    examples = tmp_path / 'examples'
    (examples / 'chrome').mkdir(parents=True)
    for task_id in TASK_IDS:
        (examples / 'chrome' / f'{task_id}.json').write_text(
            json.dumps({'instruction': f'Do {task_id}'}), encoding='utf-8'
        )
    (tmp_path / 'meta.json').write_text(json.dumps({'chrome': list(TASK_IDS)}), encoding='utf-8')
    (tmp_path / 'proxy.json').write_text('{}', encoding='utf-8')
    manifest = tmp_path / 'test.jsonl'
    manifest.write_text(
        '\n'.join(
            json.dumps({
                'id': task_id,
                'metadata': {'domain': 'chrome', 'instruction': f'Do {task_id}'},
                'files': {
                    'type': 'zip',
                    'path': f'blobs/{task_id}.zip',
                    'size': 100,
                    'remote': {'namespace': 'test', 'dataset': 'osworld', 'version': 'v1', 'split': 'test'},
                },
            })
            for task_id in TASK_IDS
        )
    )

    for key, value in {
        'OSWORLD_MODEL_FAMILY': 'claude',
        'OSWORLD_HARNESS': 'cc',
        'OSWORLD_DATASET_URI': 'test/osworld@v1',
        'OSWORLD_DATASET_MANIFEST': str(manifest),
        'WEDATA_KEY': '{"id":"test","subject":"test","key":"test"}',
        'NEXUS_MODEL': 'anthropic.claude-opus-4-8',
        'NEXUS_BASE_URL': 'http://model-fixture/v1',
        'NEXUS_API_KEY': FIXTURE_SECRET,
        'AGS_OSWORLD_TOOL_ID': 'sdt-47kozq7f',
        'OSWORLD_ENABLE_PROXY': 'true',
        'OSWORLD_TEST_META_PATH': str(tmp_path / 'meta.json'),
        'OSWORLD_PROXY_CONFIG_FILE': str(tmp_path / 'proxy.json'),
        'OSWORLD_VALIDATE_EVALUATOR_HOST_DEPS': '0',
        'OSWORLD_SESSION_NAMESPACE': str(tmp_path / 'new-run'),
        'AGS_SECRET_ID': FIXTURE_SECRET,
        'AGS_SECRET_KEY': FIXTURE_SECRET,
        'AIHUB_API_KEY': FIXTURE_GATEWAY_KEY,
    }.items():
        monkeypatch.setenv(key, value)


def _items() -> list[Any]:
    from nexus.extensions.tasks.os_world.generators import generator_native

    return list(generator_native.generate_configs())


def test_dataset_row_uses_the_platform_required_shape(exporter_env: None) -> None:
    """The platform reads `id` and `meta_data_`; any other top-level naming is
    simply not looked at."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _dataset_row

    row = _dataset_row(_items()[0])

    assert set(row) == {'id', 'meta_data_'}
    # Mirrors the offline run_id so a platform result can be matched against the
    # offline baseline row by row.
    assert row['id'] == 'chrome__task-a'
    run = row['meta_data_']['nexus_run_input']
    assert run['task_config'] == {'domain': 'chrome', 'task_id': 'task-a'}
    assert run['task_files']['path'] == 'blobs/task-a.zip'


def test_instruction_and_domain_are_siblings_of_the_run_input(exporter_env: None) -> None:
    """They feed the platform's detail view. Merged one level down they would
    instead reach ``RunInput`` as unknown task_config keys."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _dataset_row

    row = _dataset_row(_items()[1])

    assert row['meta_data_']['instruction'] == 'Do task-b'
    assert row['meta_data_']['domain'] == 'chrome'
    assert set(row['meta_data_']['nexus_run_input']) == {'task_config', 'task_files'}


def test_eval_set_config_holds_the_batch_common_half(exporter_env: None) -> None:
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _eval_set_config

    common, pruned = _eval_set_config(_items()[0])

    assert common['task_name'] == 'os_world'
    assert common['agent_name'] == 'claude_code_native_bridge'
    assert common['task_config']['run_setup'] is True
    assert common['task_config']['unsupported_evaluator_policy'] == 'fail'
    # The per-task half must not leak in here, or all tasks would run whichever
    # one happened to be exported first.
    assert 'task_id' not in common['task_config']
    assert 'domain' not in common['task_config']
    # Empty containers the model defaults in; noise in a config pasted by hand.
    assert pruned == ['tool_configs', 'gateway_trajectory_observers', 'workflow_states']


def test_split_is_exhaustive_and_non_overlapping(exporter_env: None) -> None:
    """The guard that matters: a newly emitted generator field landing in neither
    half would leave the platform running a default the offline baseline never
    used, with no visible symptom."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        _assert_split_is_total,
        _dataset_row,
        _eval_set_config,
    )

    item = _items()[0]
    common, pruned = _eval_set_config(item)

    assert _assert_split_is_total(item, _dataset_row(item), common, pruned) == []


def test_a_key_in_neither_half_is_reported(exporter_env: None) -> None:
    """Pins the check itself: dropping a task_config key from the common half
    without declaring it local-only must be caught, not absorbed."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        _assert_split_is_total,
        _dataset_row,
        _eval_set_config,
    )

    item = _items()[0]
    common, pruned = _eval_set_config(item)
    common['task_config'].pop('observation_type')

    assert _assert_split_is_total(item, _dataset_row(item), common, pruned) == [
        "task_config keys in neither half: ['observation_type']"
    ]


def test_the_one_item_the_eval_set_is_built_from_speaks_for_all_of_them(exporter_env: None) -> None:
    """The exhaustiveness check compares key spaces and would pass happily while
    every task ran with task-a's value -- 361 identical runs, no error."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _assert_batch_common

    assert _assert_batch_common(_items()) == []


def test_a_value_that_varies_between_tasks_cannot_pass_as_batch_common(exporter_env: None) -> None:
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _assert_batch_common

    items = _items()
    items[1].config.task_config['observation_type'] = 'a11y_tree'

    assert _assert_batch_common(items) == [
        "task_config keys differ between tasks but are not per-task: ['observation_type']"
    ]


@pytest.fixture
def routed_items(exporter_env: None, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """The three-task checkout with ``task-c`` routed to the large sandbox.

    ``sandbox_policy`` routes by (domain, task_id) and the real pair lives under
    ``multi_apps``/``gimp``, which a synthetic ``chrome`` checkout cannot reach.
    That gap is why 19 passing tests coexisted with an export that failed on all
    369 real tasks.
    """
    from nexus.extensions.tasks.os_world.execution import sandbox_policy

    monkeypatch.setitem(sandbox_policy.LARGE_TASK_PROFILES, ('chrome', 'task-c'), 'fixture-large')
    return _items()


def test_the_routed_task_carries_its_own_sandbox(routed_items: list[Any]) -> None:
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _dataset_row

    baseline = routed_items[0].config.model_dump(exclude_none=True)
    run_input = _dataset_row(routed_items[2], baseline)['meta_data_']['nexus_run_input']

    assert run_input['runtime_provider_name__fix__'] == 'osworld_large'
    assert run_input['runtime_configs']['main']['tool_id__fix__'] == 'sdt-ruk8welf'
    assert run_input['runtime_configs']['main']['tool_name__fix__'] == 'osworld-v2-8c-32g'
    assert 'nexus.extensions.tasks.os_world.execution.large_sandbox' in run_input['imports__fix__']
    # The other two rows pay nothing for the escape hatch.
    assert set(_dataset_row(routed_items[1], baseline)['meta_data_']['nexus_run_input']) == {
        'task_config',
        'task_files',
    }


def test_a_plain_override_would_be_overwritten_by_the_eval_set(routed_items: list[Any]) -> None:
    """Why the suffix and not just the value: the row is the ``customize`` layer,
    which loses to the eval set's ``dataset_version``. Without ``__fix__`` the
    routed task runs on the batch's small sandbox while the dataset row says
    otherwise, and nothing reports it."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        _assert_split_is_total,
        _dataset_row,
        _eval_set_config,
    )

    common, pruned = _eval_set_config(routed_items[0])
    baseline = routed_items[0].config.model_dump(exclude_none=True)
    row = _dataset_row(routed_items[2], baseline)
    assert _assert_split_is_total(routed_items[2], row, common, pruned) == []

    run_input = row['meta_data_']['nexus_run_input']
    run_input['runtime_provider_name'] = run_input.pop('runtime_provider_name__fix__')
    assert _assert_split_is_total(routed_items[2], row, common, pruned) == [
        "per-task overrides the eval set would overwrite: ['runtime_provider_name']"
    ]


def test_the_fix_suffix_survives_the_platform_merge(routed_items: list[Any]) -> None:
    """The row's value must beat the eval set's and arrive with the suffix gone.
    Mirrors hy_unify_eval's ``_deep_update`` with ``is_clear_key`` on the last
    round (param_utils.py:156-254)."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        _dataset_row,
        _eval_set_config,
    )

    common, _pruned = _eval_set_config(routed_items[0])
    baseline = routed_items[0].config.model_dump(exclude_none=True)
    row = _dataset_row(routed_items[2], baseline)

    merged = _platform_merge(dataset_version=common, customize=row['meta_data_']['nexus_run_input'])

    assert merged['runtime_provider_name'] == 'osworld_large'
    assert merged['runtime_configs']['main']['tool_id'] == 'sdt-ruk8welf'
    # Sibling keys of the overridden one still come from the eval set.
    assert merged['runtime_configs']['main']['timeout'] == 15300


def test_a_routed_task_no_longer_fails_the_batch_common_check(routed_items: list[Any]) -> None:
    """The check must accept a difference the row can express, and still refuse
    one it cannot."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _assert_batch_common

    assert _assert_batch_common(routed_items) == []

    routed_items[2].config.agent_config['max_turns'] = 7
    assert _assert_batch_common(routed_items) == [
        "top-level keys differ between tasks but the eval set carries only one: ['agent_config']"
    ]


def test_cos_task_files_replace_host_checkout_on_platform(exporter_env: None) -> None:
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _eval_set_config

    item = _items()[0]
    common, _pruned = _eval_set_config(item)

    assert 'task_root' not in common['task_config']
    assert 'task_file' not in common['task_config']
    assert item.config.task_files is not None


def test_the_proxy_path_already_travels_as_a_reference(exporter_env: None) -> None:
    """Why ``LOCAL_ONLY_TASK_CONFIG_KEYS`` is empty: the other host path in V1's
    task_config is emitted as an interpolation by the generator itself, so it
    needs no table entry to survive the split."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        LOCAL_ONLY_TASK_CONFIG_KEYS,
        _eval_set_config,
    )

    common, _pruned = _eval_set_config(_items()[0])

    assert LOCAL_ONLY_TASK_CONFIG_KEYS == ()
    assert common['task_config']['proxy_config_file'] == '${oc.env:OSWORLD_PROXY_CONFIG_FILE}'


def test_a_batch_common_loopback_url_is_refused(exporter_env: None) -> None:
    """The V1 claude harness points its agent at a gateway running beside the
    batch. Exported literally, the platform worker gets a connection refused from
    inside the agent, naming nothing that leads back to this config."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        _assert_no_host_local_urls,
        _eval_set_config,
    )

    common, _pruned = _eval_set_config(_items()[0])
    assert _assert_no_host_local_urls(common) == []

    common['agent_config']['upstream_base_url'] = 'http://127.0.0.1:8787/v1'
    assert _assert_no_host_local_urls(common) == [
        "batch-common values point at this machine only: ['agent_config.upstream_base_url=http://127.0.0.1:8787/v1']"
    ]


def test_the_per_task_key_survives_the_platform_merge() -> None:
    """Why ``domain``/``task_id`` must be ABSENT from the eval-set half rather
    than merely also present in the row: the dataset row is the platform's
    ``customize`` layer, which loses to the eval set's ``dataset_version`` layer.
    Mirrors hy_unify_eval's ``_deep_update`` over ``config_priority``."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        PER_TASK_TASK_CONFIG_KEYS,
        _drop_keys,
        _extract_keys,
    )

    generated = {
        'domain': 'chrome',
        'task_id': 'task-a',
        'observation_type': 'screenshot',
    }
    common_tc = dict(generated)
    _drop_keys(common_tc, PER_TASK_TASK_CONFIG_KEYS)
    # A different task than the one the common half was built from.
    row_tc = _extract_keys({**generated, 'domain': 'gimp', 'task_id': 'task-c'}, PER_TASK_TASK_CONFIG_KEYS)

    merged = _platform_merge(dataset_version={'task_config': common_tc}, customize={'task_config': row_tc})

    assert merged['task_config']['domain'] == 'gimp'
    assert merged['task_config']['task_id'] == 'task-c'
    assert 'task_root' not in merged['task_config']


def test_platform_export_rejects_checkout_only_configs(exporter_env):
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _dataset_row, _eval_set_config

    item = _items()[0]
    item.config.task_files = None
    for export in (_dataset_row, _eval_set_config):
        with pytest.raises(ValueError, match='remote COS task_files'):
            export(item)


def _platform_merge(*, dataset_version: dict, customize: dict) -> dict:
    """Recursive merge over ``config_priority`` low->high, as the platform does.

    Including the ``__fix__`` protocol (param_utils.py:156-254): a fixed key is
    skipped by every later -- i.e. higher-priority -- layer, and the suffix is
    stripped on the last round, which always runs because all four layers are
    present in ``diff_level_param_dict`` (base.py:600-605).
    """
    import copy

    def deep_update(base: dict, update: dict) -> dict:
        for key, value in update.items():
            if not key.endswith('__fix__') and f'{key}__fix__' in base:
                continue
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def strip_suffixes(node: dict) -> dict:
        for key in list(node):
            if isinstance(node[key], dict):
                strip_suffixes(node[key])
            if key.endswith('__fix__'):
                node[key.removesuffix('__fix__')] = node.pop(key)
        return node

    layers = {'customize': customize, 'dataset_version': dataset_version}
    params: dict = {}
    for level in ('default', 'customize', 'dataset_version', 'user'):
        params = deep_update(params, copy.deepcopy(layers.get(level, {})))
    return strip_suffixes(params)


def test_platform_managed_keys_are_not_emitted(exporter_env: None) -> None:
    """Spec 2.3: the platform injects trajectory/token processors and owns the
    runner timeout. ``openai_clients`` is here for a sharper reason -- the
    platform rebuilds ``openai_clients['main']`` from the task's own form, so a
    block left here is either overwritten or the reason the run dies with
    'Failed to build main model config'."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        PLATFORM_MANAGED_KEYS,
        _eval_set_config,
    )

    common, _pruned = _eval_set_config(_items()[0])

    assert PLATFORM_MANAGED_KEYS == ('processors', 'runner', 'run_id', 'openai_clients')
    for key in PLATFORM_MANAGED_KEYS:
        assert key not in common


def test_no_oc_decode_survives_into_the_eval_set(exporter_env: None) -> None:
    """``oc.decode`` only takes strings. 七彩石 returns the raw type, so the day
    somebody helpfully stores ``3600.0`` as a number instead of a string, that
    one key raises InterpolationResolutionError and nothing else does."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _eval_set_config

    common, _pruned = _eval_set_config(_items()[0])

    assert 'oc.decode' not in json.dumps(common, ensure_ascii=False)
    # Numbers stay numbers, which is the whole reason oc.decode is unnecessary.
    assert common['runtime_provider_config']['request_timeout'] == 1800.0
    assert common['task_config']['enable_proxy'] is True


def test_values_the_platforms_own_code_derives_are_not_restated(exporter_env: None) -> None:
    """收敛: a second spelling of a value the runtime already knows is a value
    that can go stale on its own. ``domain`` is ``f'{region}.tencentags.com'``,
    and ``tool_type``/``launch_mode`` both follow ``runtime_kind``."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        _drop_derivable,
        _eval_set_config,
    )

    common, _pruned = _eval_set_config(_items()[0])
    dropped = _drop_derivable(common)

    assert dropped == [
        'runtime_provider_config.domain',
        'runtime_configs.main.tool_type',
    ]
    assert 'domain' not in common['runtime_provider_config']
    assert 'tool_type' not in common['runtime_configs']['main']
    # V2's eval set omits it too; V1's generator is the one that writes it out.
    assert 'launch_mode' not in common['runtime_configs']['main']
    # NOT dropped, unlike V2: V2's agent derives ANTHROPIC_MODEL from cc_model
    # (osworld_v2_cc_agent/agent.py:210) and V1's agent only merges extra_env
    # through, so dropping it here would change what CC calls itself.
    assert common['agent_config']['execution_options']['extra_env']['ANTHROPIC_MODEL'] == 'claude-opus-4-8'
    # The CC harness names its model ``cc_model``, which is the agent's own field
    # rather than the served model -- nothing for _DERIVABLE_AGENT_CONFIG_KEYS to
    # match, and the value is still needed (it is what CC calls itself).
    assert common['agent_config']['execution_options']['cc_model'] == 'claude-opus-4-8'


def test_codex_facade_is_preserved_even_when_equal_to_default(
    exporter_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The codex harness is the one that puts the served model in
    ``agent_config``. Dropping it unconditionally would hand the platform a
    config that silently reverts to the agent's default for a batch that ran on
    something else."""
    from nexus.extensions.tasks.os_world.execution.codex_profile import OSWorldCodexOptions as CodexOSWorldInput
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        _drop_derivable,
        _eval_set_config,
    )

    monkeypatch.setenv('OSWORLD_MODEL_FAMILY', 'gpt')
    monkeypatch.setenv('OSWORLD_HARNESS', 'codex')
    monkeypatch.setenv('CODEX_MODEL', CodexOSWorldInput.model_fields['model'].default)
    common, _pruned = _eval_set_config(_items()[0])
    assert 'agent_config.model' not in _drop_derivable(common)
    assert common['agent_config']['execution_options']['model'] == CodexOSWorldInput.model_fields['model'].default

    monkeypatch.setenv('CODEX_MODEL', 'gpt-5.6-sol')
    common, _pruned = _eval_set_config(_items()[0])
    assert 'agent_config.model' not in _drop_derivable(common)
    assert common['agent_config']['execution_options']['model'] == 'gpt-5.6-sol'


def test_the_sandbox_is_pinned_rather_than_looked_up(exporter_env: None) -> None:
    """With both ``tool_id`` and ``skip_tool_lookup`` set, ``_ensure_tool``
    returns the pinned tool before ``_default_tool_strategy`` is consulted at all
    -- whose ``ensure_builtin`` default would otherwise silently create a blank
    tool. So the pair is what has to be explicit, and ``tool_strategy`` genuinely
    does not belong."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _eval_set_config

    common, _pruned = _eval_set_config(_items()[0])
    main = common['runtime_configs']['main']

    assert main['runtime_kind'] == 'osworld'
    assert main['tool_id'] == '${oc.env:AGS_OSWORLD_TOOL_ID}'
    assert main['skip_tool_lookup'] is True
    # Explicit because the provider's own default (300s) is far short of a run.
    assert main['timeout'] == 15300
    assert main['provision_mode'] == 'control_plane'


def test_no_secret_value_reaches_either_half(exporter_env: None) -> None:
    """The generators keep credentials as ``${oc.env:}`` references resolved long
    after export, so neither half should ever hold the value itself."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        _dataset_row,
        _eval_set_config,
    )

    items = _items()
    common, _pruned = _eval_set_config(items[0])
    rows = [_dataset_row(item) for item in items]

    assert FIXTURE_SECRET not in json.dumps(common, ensure_ascii=False)
    assert FIXTURE_SECRET not in json.dumps(rows, ensure_ascii=False)
    assert common['runtime_provider_config']['secret_id'] == '${oc.env:AGS_SECRET_ID}'
    assert common['runtime_provider_config']['secret_key'] == '${oc.env:AGS_SECRET_KEY}'


def test_the_client_form_carries_a_placeholder_where_the_key_goes(exporter_env: None) -> None:
    """输出二 is a form somebody fills in on the platform, and the file we write
    is a committed artifact -- the last place a real key should appear. ``api_key``
    is a SIBLING of ``main_client_args``: nested it still reaches the client, but
    the platform's own credential-injection path reads the sibling, so the
    symptom is a 401 with a key that is visibly present in the form."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import (
        API_KEY_PLACEHOLDER,
        _client_form,
    )

    form = _client_form(_items()[0])

    assert form['模型名称'] == 'anthropic.claude-opus-4-8'
    assert form['推理服务URL'] == 'http://model-fixture/v1'
    assert form['parameter_configuration']['nexus_main_client_name'] == 'anthropic_messages'
    assert form['parameter_configuration']['api_key'] == API_KEY_PLACEHOLDER
    assert 'api_key' not in form['parameter_configuration']['main_client_args']
    assert FIXTURE_SECRET not in json.dumps(form, ensure_ascii=False)


def test_the_cli_writes_three_artifacts_and_prints_no_secret(
    exporter_env: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End to end, including stdout: a key pasted out of a terminal is a key
    that leaked."""
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import main

    out = tmp_path / 'out'
    assert (
        main([
            '-o',
            str(out / 'dataset.jsonl'),
            '--eval-set-out',
            str(out / 'eval_set.json'),
            '--client-form-out',
            str(out / 'client_form.json'),
        ])
        == 0
    )

    rows = [json.loads(line) for line in (out / 'dataset.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['id'] for row in rows] == [f'chrome__{task_id}' for task_id in TASK_IDS]

    written = '\n'.join(path.read_text(encoding='utf-8') for path in sorted(out.iterdir()))
    captured = capsys.readouterr()
    assert FIXTURE_SECRET not in written
    assert FIXTURE_SECRET not in captured.out + captured.err
