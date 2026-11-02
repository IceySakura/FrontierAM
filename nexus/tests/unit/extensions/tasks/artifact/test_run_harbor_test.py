"""Exercise real Harbor/Compose preflight; replace only uv dispatch and batch execution.

Old clients-file/JSON injection tests now cover declared model overrides, secret
mapping and immutable task data. Instance cache binding is covered by model tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest


REPO = Path(__file__).resolve().parents[5]
SCRIPT = REPO / 'nexus/extensions/tasks/artifact/run_harbor_test.sh'


def write_task(path: Path, *, judge='web_judger', extra_env=None, map_key=True, map_render_token=True):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'instruction.md').write_text('Build an SVG animation', encoding='utf-8')
    tests = path / 'tests'
    tests.mkdir(exist_ok=True)
    (tests / 'instruction.md').write_text('Build an SVG animation', encoding='utf-8')
    (tests / 'test.sh').write_text('#!/bin/sh\nexit 99\n', encoding='utf-8')
    env = {'T3_JUDGE_NAME': judge, 'T3_INSTANCE_ID': path.name}
    if map_key:
        env['JUDGE_API_KEY'] = '${JUDGE_API_KEY}'
    if map_render_token:
        env['GONGFENG_RUNTIME_TOKEN'] = '${GONGFENG_RUNTIME_TOKEN}'
    env.update(extra_env or {})
    text = (
        'version = "T3-RL-v1.0"\n'
        '[metadata]\n'
        f'instance_id = {json.dumps(path.name)}\n'
        '[environment]\n'
        'docker_image = "example.invalid/test:latest"\n'
        'workdir = "/workspace"\n'
        'skip_upload = true\n'
        '[agent]\ntimeout_sec = 90.0\n'
        '[verifier]\ntimeout_sec = 60.0\nenvironment_mode = "shared"\n'
        '[verifier.env]\n'
    )
    text += ''.join(f'{key} = {json.dumps(value)}\n' for key, value in env.items())
    (path / 'task.toml').write_text(text, encoding='utf-8')


@pytest.fixture
def shell_env(tmp_path: Path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    uv = bin_dir / 'uv'
    uv.write_text(
        f'#!{sys.executable}\n'
        'import json, os, sqlite3, sys\n'
        'from pathlib import Path\n'
        'assert sys.argv[1:4] == ["run", "--with", "harbor==0.20.0"]\n'
        'args = sys.argv[4:]\n'
        'if args[0] == "python":\n'
        '    os.execv(sys.executable, [sys.executable, *args[1:]])\n'
        'assert args[:3] == ["nexus", "run", "batch"]\n'
        'Path(os.environ["CAPTURE_ARGS"]).write_text(json.dumps(args))\n'
        'if os.environ.get("NO_DB"):\n'
        '    sys.exit(int(os.environ.get("BATCH_EXIT", "7")))\n'
        'db = sqlite3.connect(args[args.index("-d") + 1])\n'
        'db.execute("create table runs(run_id text, status text, task_output text, error text)")\n'
        'payload = {"score": 0.5, "reward_metadata": {"judge_name": "web_judger"}}\n'
        'db.execute("insert into runs values (?, ?, ?, ?)", ("sample", "success", json.dumps(payload), ""))\n'
        'db.commit()\n'
        'db.close()\n'
        'print("mock batch completed")\n'
        'sys.exit(int(os.environ.get("BATCH_EXIT", "0")))\n',
        encoding='utf-8',
    )
    uv.chmod(0o755)
    tasks = tmp_path / 'tasks with & symbols'
    write_task(tasks / 'sample')
    env = {
        'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
        'NEXUS_REPO': str(REPO),
        'NEXUS_AUTO_INSTALL': '0',
        'PYTHONDONTWRITEBYTECODE': '1',
        'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
        'JUDGE_API_KEY': 'fake-judge-key',
        'OPENAI_API_KEY': 'fake-infer-key',
        'GONGFENG_RUNTIME_TOKEN': 'fake-runtime-token',
        'OUT_ROOT': str(tmp_path / 'outputs'),
        'TAG': 'case',
        'CAPTURE_ARGS': str(tmp_path / 'args.json'),
    }
    return tasks, env


def invoke(tasks: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['bash', str(SCRIPT), str(tasks), *args],
        env=env,
        cwd=tasks,
        text=True,
        capture_output=True,
        timeout=60,
    )


def parsed_models(result):
    return [
        json.loads(line.removeprefix('Judge models: '))
        for line in result.stdout.splitlines()
        if line.startswith('Judge models: ')
    ]


def test_help_requires_no_runtime_configuration():
    result = subprocess.run(
        ['bash', str(SCRIPT), '--help'], env={'PATH': os.environ['PATH']}, text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 0
    assert '--check' in result.stdout


def test_harbor_compose_arguments_and_summary(shell_env):
    tasks, env = shell_env
    env.update({'LIMIT': '2', 'CONCURRENT': '3', 'FLAT': 'false', 'WORKERS': '4'})
    result = invoke(tasks, env, '--set', 'task_config.mode=nop')
    assert result.returncode == 0, result.stderr
    assert 'score=0.500' in result.stdout
    args = json.loads(Path(env['CAPTURE_ARGS']).read_text())
    uri = urlsplit(args[args.index('--data') + 1])
    assert uri.scheme == 't3-harbor'
    assert unquote(uri.path) == str(tasks)
    assert parse_qs(uri.query) == {'flat': ['false'], 'workers': ['4']}
    assert 'harbor_bridge:mode=agent' in args
    assert args[args.index('--limit') + 1] == '2'
    assert args[args.index('--max-concurrent') + 1] == '3'
    for role in ('main', 'verifier'):
        assert f'runtime_configs.{role}.worker_venv.source={REPO}' in args
    assert args[-2:] == ['--set', 'task_config.mode=nop']
    assert '--generator' not in args
    assert parsed_models(result)[0]['models'] == {
        'agent': 'openai/api_deepseek_deepseek-flash',
        'vlm': 'openai/gemini-3.8-flash',
    }


def test_check_only_runs_real_preflight_without_batch(shell_env):
    tasks, env = shell_env
    result = invoke(tasks, env, '--check')
    assert result.returncode == 0, result.stderr
    assert 'Validated 1 task(s)' in result.stdout
    assert parsed_models(result)
    assert not Path(env['CAPTURE_ARGS']).exists()
    assert not Path(env['OUT_ROOT']).exists()


def test_batch_failure_is_preserved_after_summary(shell_env):
    tasks, env = shell_env
    env['BATCH_EXIT'] = '7'
    result = invoke(tasks, env)
    assert result.returncode == 7
    assert 'Results:' in result.stdout
    assert 'score=0.500' in result.stdout


def test_batch_failure_without_database_is_preserved(shell_env):
    tasks, env = shell_env
    env.update({'NO_DB': '1', 'BATCH_EXIT': '9'})
    result = invoke(tasks, env)
    assert result.returncode == 9
    assert 'Results:' not in result.stdout


@pytest.mark.parametrize(('batch_exit', 'expected'), [('0', 6), ('7', 7)])
def test_log_pipeline_failure_is_not_hidden(shell_env, batch_exit, expected):
    tasks, env = shell_env
    env['BATCH_EXIT'] = batch_exit
    tee = Path(env['PATH'].split(os.pathsep)[0]) / 'tee'
    tee.write_text(
        f'#!{sys.executable}\n'
        'import sys\nfrom pathlib import Path\n'
        'text = sys.stdin.read()\n'
        'Path(sys.argv[1]).write_text(text)\n'
        'sys.stdout.write(text)\n'
        'sys.exit(6)\n',
    )
    tee.chmod(0o755)
    result = invoke(tasks, env)
    assert result.returncode == expected
    assert 'score=0.500' in result.stdout


@pytest.mark.parametrize(
    ('key', 'value'), [('LIMIT', '0'), ('CONCURRENT', '-1'), ('WORKERS', 'bad'), ('FLAT', 'maybe')]
)
def test_invalid_batch_controls_fail_before_execution(shell_env, key, value):
    tasks, env = shell_env
    env[key] = value
    result = invoke(tasks, env)
    assert result.returncode == 2
    assert not Path(env['CAPTURE_ARGS']).exists()


@pytest.mark.parametrize('extension', ['db', 'log'])
def test_existing_output_is_not_reused(shell_env, extension):
    tasks, env = shell_env
    output = Path(env['OUT_ROOT'])
    output.mkdir()
    existing = output / f'case.{extension}'
    existing.write_bytes(b'existing run')
    result = invoke(tasks, env)
    assert result.returncode == 2
    assert existing.read_bytes() == b'existing run'
    assert not Path(env['CAPTURE_ARGS']).exists()


@pytest.mark.parametrize('check_only', [False, True])
@pytest.mark.parametrize('key', [None, '', '   ', '<AIHUB_KEY>'])
def test_missing_or_placeholder_secret_fails_before_launch(shell_env, key, check_only):
    tasks, env = shell_env
    if key is None:
        env.pop('JUDGE_API_KEY')
    else:
        env['JUDGE_API_KEY'] = key
    result = invoke(tasks, env, *(['--check'] if check_only else []))
    assert result.returncode != 0
    assert 'JUDGE_API_KEY' in result.stderr
    assert not Path(env['CAPTURE_ARGS']).exists()
    assert not Path(env['OUT_ROOT']).exists()


def test_host_secret_without_verifier_mapping_cannot_satisfy_preflight(shell_env):
    tasks, env = shell_env
    write_task(tasks / 'sample', map_key=False)
    result = invoke(tasks, env)
    assert result.returncode != 0
    assert 'JUDGE_API_KEY' in result.stderr
    assert env['JUDGE_API_KEY'] not in result.stdout + result.stderr
    assert not Path(env['CAPTURE_ARGS']).exists()


def test_render_token_without_verifier_mapping_fails_before_launch(shell_env):
    tasks, env = shell_env
    write_task(tasks / 'sample', map_render_token=False)
    result = invoke(tasks, env)
    assert result.returncode != 0
    assert 'Render credential preflight failed' in result.stderr
    assert env['GONGFENG_RUNTIME_TOKEN'] not in result.stdout + result.stderr
    assert not Path(env['CAPTURE_ARGS']).exists()
    assert not Path(env['OUT_ROOT']).exists()


def test_missing_host_render_token_fails_before_preflight(shell_env):
    tasks, env = shell_env
    env.pop('GONGFENG_RUNTIME_TOKEN')
    result = invoke(tasks, env, '--check')
    assert result.returncode == 2
    assert 'GONGFENG_RUNTIME_TOKEN or GONGFENG_TOKEN is required' in result.stderr
    assert not Path(env['CAPTURE_ARGS']).exists()
    assert not Path(env['OUT_ROOT']).exists()


@pytest.mark.parametrize('field', ['NEXUS_OPENAI_CLIENTS', 'T3_JUDGE_PROFILE'])
def test_legacy_configuration_is_rejected_without_echoing(shell_env, field):
    tasks, env = shell_env
    private = 'private-conflicting-value'
    write_task(tasks / 'sample', extra_env={field: private})
    result = invoke(tasks, env)
    assert result.returncode != 0
    assert 'legacy' in result.stderr
    assert private not in result.stdout + result.stderr
    assert not Path(env['CAPTURE_ARGS']).exists()


@pytest.mark.parametrize(
    'override',
    [
        'private-invalid-json',
        '[]',
        'null',
        '{"flow2code":"gemini-3.5-flash-aihub-2.0.yaml"}',
        '{"scoring":"gemini-3.5-flash-aihub-2.0.yaml"}',
        '{"agent":"../secret.yaml"}',
        '{"agent":"<UNSET_MODEL>"}',
        '{"agent":"missing-aihub-2.0.yaml"}',
    ],
)
def test_invalid_model_overrides_fail_before_batch(shell_env, override):
    tasks, env = shell_env
    write_task(tasks / 'sample', extra_env={'T3_JUDGE_MODELS': override})
    result = invoke(tasks, env)
    assert result.returncode != 0
    assert 'Judge model preflight failed' in result.stderr
    assert 'private-invalid-json' not in result.stdout + result.stderr
    assert not Path(env['CAPTURE_ARGS']).exists()


def test_tasks_select_independent_models_and_are_not_rewritten(shell_env):
    tasks, env = shell_env
    write_task(
        tasks / 'second',
        judge='svg',
        extra_env={
            'T3_JUDGE_MODELS': json.dumps({'vlm': 'gemini-3.5-flash-aihub-2.0.yaml'}),
        },
    )
    before = {path: path.read_bytes() for path in tasks.rglob('*') if path.is_file()}
    result = invoke(tasks, env, '--check')
    assert result.returncode == 0, result.stderr
    models = [item['models'] for item in parsed_models(result)]
    assert len(models) == 2
    assert {'vlm': 'openai/gemini-3.5-flash'} in models
    assert {'agent': 'openai/api_deepseek_deepseek-flash', 'vlm': 'openai/gemini-3.8-flash'} in models
    assert all(path.read_bytes() == data for path, data in before.items())


def test_xtrace_and_batch_arguments_do_not_echo_secret(shell_env):
    tasks, env = shell_env
    secret = 'fake-key-"quotes"-\\backslash-$(exit 99)'
    env['JUDGE_API_KEY'] = secret
    before = (tasks / 'sample/task.toml').read_bytes()
    result = subprocess.run(
        ['bash', '-x', str(SCRIPT), str(tasks)], env=env, cwd=tasks, text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout + result.stderr
    assert secret not in Path(env['CAPTURE_ARGS']).read_text()
    assert secret not in (Path(env['OUT_ROOT']) / 'case.log').read_text()
    assert (tasks / 'sample/task.toml').read_bytes() == before


def test_empty_selection_is_rejected(shell_env):
    tasks, env = shell_env
    empty = tasks / 'empty'
    empty.mkdir()
    result = invoke(empty, env, '--check')
    assert result.returncode != 0
    assert not Path(env['CAPTURE_ARGS']).exists()
