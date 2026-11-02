"""Local Harbor materialize -> wheelhouse -> offline import, using real package builds.

Run with NEXUS_MATERIALIZE_WHEEL_E2E=1 and the dev, distributed, builder extras,
plus harbor==0.20.0. Package-index access is required; no Cloud or LLM is called.
NEXUS_MATERIALIZE_E2E_OUTPUT optionally preserves artifacts in a chosen directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile

from pathlib import Path

import pytest

from nexus.distributed.batch.data_list import DataRef, read_indexed_frame, scan_data_list, validate_frame
from tests.e2e.distributed.run_harbor_dataset_batch import write_harbor_dataset_fixture


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.getenv('NEXUS_MATERIALIZE_WHEEL_E2E') != '1',
        reason='Set NEXUS_MATERIALIZE_WHEEL_E2E=1 to build and install real wheelhouses',
    ),
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES = PROJECT_ROOT / 'nexus/extensions/harbor_bridge/configs'
PROFILE_NAMES = ('runtime/gongfeng-prod.yaml', 'agents/claude-code.yaml', 'llm/aws-opus48-litellm.yaml')


def _run(output: Path, label: str, command: list[str], *, env: dict[str, str], cwd=PROJECT_ROOT):
    print(f'{label}: started', flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800, check=False)
    (output / f'{label}.stdout.log').write_text(result.stdout)
    (output / f'{label}.stderr.log').write_text(result.stderr)
    (output / f'{label}.command.json').write_text(json.dumps(command, indent=2))
    assert result.returncode == 0, f'{label} failed: {result.stdout[-4000:]}\n{result.stderr[-4000:]}'
    print(f'{label}: passed', flush=True)
    return result


@pytest.fixture(scope='module')
def wheel_build(tmp_path_factory):
    pytest.importorskip('harbor.models.task.task')
    output = Path(os.environ.get('NEXUS_MATERIALIZE_E2E_OUTPUT') or tmp_path_factory.mktemp('materialize-wheel'))
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    env = {
        **os.environ,
        'NEXUS_OTEL_ENABLED': 'false',
        'GONGFENG_RUNTIME_TOKEN': 'fixture-runtime-secret',
        'OPENAI_API_KEY': 'fixture-llm-secret',
    }
    _run(output, 'build-nexus', ['uv', 'build', '--wheel', '--out-dir', str(output / 'dist')], env=env)
    wheels = list((output / 'dist').glob('nexus_agent-*.whl'))
    assert len(wheels) == 1
    return output, wheels[0], env


def _verify_wheelhouse(output, wheel, env, task_list, *, label):
    wheelhouse = output / f'{label}-wheelhouse'
    _run(
        output,
        f'{label}-build-wheelhouse',
        [
            sys.executable,
            '-m',
            'nexus.cli.main',
            'modules',
            'wheel',
            '--task-list',
            str(task_list),
            '--project-root',
            str(PROJECT_ROOT),
            '--lock',
            str(PROJECT_ROOT / 'uv.lock'),
            '--nexus-wheel',
            str(wheel),
            '--output',
            str(wheelhouse),
        ],
        env=env,
    )
    report = json.loads((wheelhouse / 'module-wheel-report.json').read_text())
    metadata = report['wheelhouse']
    assert {'harbor_bridge', 'claude_code', 'gongfeng', 'litellm'} <= set(metadata['module_names'])
    assert 'harbor==0.20.0' in metadata['module_host_requirements']
    assert 'litellm' in metadata['module_host_requirements']
    module_lock = wheelhouse / metadata['module_lock_path']
    assert hashlib.sha256(module_lock.read_bytes()).hexdigest() == metadata['module_lock_sha256']
    assert list((wheelhouse / 'wheels').glob('harbor-0.20.0-*.whl'))

    verify_venv = output / f'{label}-offline-venv'
    _run(
        output,
        f'{label}-create-venv',
        [
            'uv',
            'venv',
            '--python',
            sys.executable,
            str(verify_venv),
        ],
        env=env,
    )
    python = str(verify_venv / 'bin/python')
    _run(
        output,
        f'{label}-offline-install',
        [
            'uv',
            'pip',
            'install',
            '--no-config',
            '--python',
            python,
            '--no-index',
            '--find-links',
            str(wheelhouse / 'wheels'),
            'nexus-agent[distributed,runtime]',
            *metadata['module_host_requirements'],
        ],
        env=env,
    )
    _run(
        output,
        f'{label}-offline-import',
        [
            python,
            '-c',
            'import importlib.metadata, sys; from pathlib import Path; '
            'import nexus; from harbor.models.task.task import Task; '
            'from nexus.extensions.harbor_bridge.compiler import load_harbor_task; '
            'from nexus.extensions.agents.claude_code import agent; '
            'assert Path(nexus.__file__).is_relative_to(Path(sys.prefix)); '
            'assert importlib.metadata.version("harbor") == "0.20.0"; '
            'print("offline Harbor and Claude Code imports passed")',
        ],
        env=env,
        cwd=output,
    )
    return report


def test_harbor_profiles_materialize_to_offline_wheelhouse(wheel_build):
    output, wheel, env = wheel_build
    task_list = output / 'profiles.jsonl'
    command = [sys.executable, '-m', 'nexus.cli.main', 'materialize', '--use', 'harbor_bridge:mode=agent']
    for name in PROFILE_NAMES:
        command.extend(['--use', str(PROFILES / name)])
    command.extend(['--output', str(task_list)])

    result = _run(output, 'profiles-materialize', command, env=env)

    summary = json.loads(result.stdout)
    assert summary['record_version'] == 2
    assert summary['total_items'] == 1
    assert 'data_list' not in summary
    run_input = json.loads(task_list.read_text())['run_input']
    assert run_input['agent_name'] == 'claude_code'
    assert run_input['openai_clients']['main']['name'] == 'litellm'
    assert run_input['runtime_provider_config']['token__env'] == 'GONGFENG_RUNTIME_TOKEN'
    assert 'fixture-runtime-secret' not in task_list.read_text()
    assert 'fixture-llm-secret' not in task_list.read_text()
    _verify_wheelhouse(output, wheel, env, task_list, label='profiles')


def test_harbor_dataset_materializes_portable_files_and_offline_wheelhouse(wheel_build):
    output, wheel, env = wheel_build
    dataset = write_harbor_dataset_fixture(output / 'harbor-tasks', workload_image='example.invalid/fixture:latest')
    experiment = output / 'experiment.yaml'
    experiment.write_text(
        json.dumps({
            'compose': [
                {'data': {'uri': f't3-harbor://{dataset}?flat=true', 'split': 'test', 'id': 'harbor-zip'}},
                {'use': 'harbor_bridge:mode=agent'},
                *({'use': os.path.relpath(PROFILES / name, output)} for name in PROFILE_NAMES),
            ]
        })
    )
    task_list = output / 'harbor.jsonl'
    result = _run(
        output,
        'harbor-materialize',
        [
            sys.executable,
            '-m',
            'nexus.cli.main',
            'materialize',
            str(experiment),
            '--limit',
            '1',
            '--output',
            str(task_list),
        ],
        env=env,
    )

    summary = json.loads(result.stdout)
    assert summary['record_version'] == 3
    assert summary['total_items'] == 1
    assert summary['data_list']['frame_count'] == 1
    records = [json.loads(line) for line in task_list.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record['metadata']['dataset_item_id'] == 'harbor-zip'
    assert record['run_input']['task_name'] == 'harbor_bridge_task'
    assert record['run_input']['agent_name'] == 'claude_code'
    assert 'task_dir' not in record['run_input']['task_config']
    assert 'task_files' not in record['run_input']
    assert str(dataset) not in task_list.read_text()
    assert 'fixture-runtime-secret' not in task_list.read_text()
    assert 'fixture-llm-secret' not in task_list.read_text()
    data_list = Path(summary['data_list']['path'])
    assert data_list == output / 'harbor.data-list.bin'
    assert hashlib.sha256(data_list.read_bytes()).hexdigest() == summary['data_list']['sha256']
    index = scan_data_list(data_list, max_frame_bytes=1024 * 1024, max_bytes=1024 * 1024, max_items=1)
    ref = DataRef.model_validate(record['data_ref'])
    frame = validate_frame(
        read_indexed_frame(
            data_list,
            ref,
            expected_frame=index.by_offset[ref.offset],
            expected_item_id=record['item_id'],
            max_frame_bytes=1024 * 1024,
        ).encoded,
        expected_ref=ref,
        expected_item_id=record['item_id'],
        max_frame_bytes=1024 * 1024,
    )
    assert frame.zip_bytes is not None
    with zipfile.ZipFile(io.BytesIO(frame.zip_bytes)) as archive:
        assert archive.read('fixture/marker.txt') == b'fixture-harbor-zip\n'
        assert {'task.toml', 'instruction.md', 'tests/test.sh'} <= set(archive.namelist())
    _verify_wheelhouse(output, wheel, env, task_list, label='harbor')
