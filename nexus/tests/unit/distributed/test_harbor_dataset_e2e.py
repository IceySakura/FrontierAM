from __future__ import annotations

import json

from pathlib import Path

from tests.e2e.distributed.run_harbor_dataset_batch import (
    HARBOR_DATASET_ITEM_IDS,
    build_harbor_batch_cli_args,
    prepare_harbor_source_root,
    write_harbor_dataset_fixture,
)


def test_harbor_fixture_is_self_contained_and_has_two_nop_tasks(tmp_path: Path) -> None:
    root = write_harbor_dataset_fixture(
        tmp_path / 'harbor-tasks',
        workload_image='mirrors.tencent.com/nexus_utopia/nexus-distributed-workload@sha256:test',
    )

    assert tuple(sorted(path.name for path in root.iterdir())) == HARBOR_DATASET_ITEM_IDS
    for item_id in HARBOR_DATASET_ITEM_IDS:
        task_dir = root / item_id
        task_config = (task_dir / 'task.toml').read_text(encoding='utf-8')
        verifier = task_dir / 'tests' / 'test.sh'
        assert 'DATASET_BARRIER' not in task_config
        assert 'DATASET_BARRIER' not in verifier.read_text(encoding='utf-8')
        assert 'reward.json' in verifier.read_text(encoding='utf-8')
        assert (task_dir / 'fixture' / 'marker.txt').read_text(encoding='utf-8').strip().endswith(item_id)


def test_harbor_batch_cli_args_use_source_package_and_portable_local_data(tmp_path: Path) -> None:
    args = build_harbor_batch_cli_args(
        data_value='t3-harbor:///tmp/harbor?flat=true',
        project_root=tmp_path / 'repo',
        source_root=tmp_path / 'source',
        control_url='https://control.example.test',
        environment_id='tkex-nanjing-test',
        taihu_token_file=tmp_path / 'taihu-token',
        idempotency_key='harbor-e2e-direct',
    )

    assert args[:2] == ['batch', '--generation']
    assert args[2:4] == ['local', '--task-list-version']
    assert '--config-only' not in args
    assert args[args.index('--use') + 1] == 'harbor_bridge:mode=nop'
    assert args[args.index('--data') + 1] == 't3-harbor:///tmp/harbor?flat=true'
    assert args[args.index('--source') + 1] == str(tmp_path / 'source')
    assert args[args.index('--project-root') + 1] == str(tmp_path / 'repo')
    assert args[args.index('--control-url') + 1] == 'https://control.example.test'
    assert args[args.index('--environment') + 1] == 'tkex-nanjing-test'
    assert args[args.index('--taihu-token-file') + 1] == str(tmp_path / 'taihu-token')

    set_json_values = [args[index + 1] for index, value in enumerate(args[:-1]) if value == '--set-json']
    runtime_config = next(value for value in set_json_values if value.startswith('runtime_provider_config='))
    payload = json.loads(runtime_config.split('=', 1)[1])
    assert payload['token__env'] == 'GONGFENG_TOKEN'
    assert payload['api_base_url'] == 'http://hyrl-sandbox.prod.woa.com:8052'
    assert 'token' not in payload


def test_prepare_harbor_source_root_copies_builder_project_inputs(tmp_path: Path) -> None:
    project_root = tmp_path / 'repo'
    source_root = tmp_path / 'source'
    project_root.mkdir()
    for name, content in (
        ('pyproject.toml', '[project]\nname = "nexus-agent"\n'),
        ('uv.lock', 'version = 1\n'),
        ('README.md', 'fixture\n'),
    ):
        (project_root / name).write_text(content, encoding='utf-8')

    prepared = prepare_harbor_source_root(project_root, source_root)

    assert prepared == source_root.resolve()
    assert sorted(path.name for path in prepared.iterdir()) == ['README.md', 'pyproject.toml', 'uv.lock']
    assert (prepared / 'pyproject.toml').read_text(encoding='utf-8') == '[project]\nname = "nexus-agent"\n'
