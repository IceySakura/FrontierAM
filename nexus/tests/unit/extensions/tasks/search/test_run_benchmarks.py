import argparse
import json
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.extensions.tasks.search import compute_metrics as compute_metrics_module
from nexus.extensions.tasks.search import run_benchmarks as run_benchmarks_module


DEFAULT_DATASET_NAME = 'demo'
BROKEN_DATASET_NAME = 'broken'
MISSING_DATASET_NAME = 'missing'
ALL_FAILED_DATASET_NAME = 'all_failed'
TEMP_DATASET_NAME = 'demo_temp'
GOOD_BENCHMARK_NAME = 'good'
BAD_BENCHMARK_NAME = 'bad'
PRIMARY_RUN_ID = 'search_00001'
SEARCH_MODEL = 'demo-model'
ENV_SEARCH_MODEL = 'env-model'
OVERRIDE_SEARCH_MODEL = 'override-model'
BATCH_OK = 'batch ok'
FATAL_ERROR = 'fatal error'
BOOM_ERROR = 'boom'
RUN_TRACE = 'ok'
QUESTION_TEXT = 'What is the answer?'
STATUS_SUCCESS = 'success'
STATUS_FAILED = 'failed'
SUCCESS_METRICS = {'accuracy': 1.0, 'correct': 1, 'total': 1, 'timeout': 0}
SUCCESS_METRICS_NO_TIMEOUT = {'accuracy': 1.0, 'correct': 1, 'total': 1}
FINAL_MULTI_RUN_METRICS = {'avg_k_accuracy': 0.75, 'pass_k_accuracy': 1.0, 'instances': 2}
RUN_SUMMARY_JSON = 'RUN_SUMMARY.json'
DB_SNAPSHOT_TEXT = 'db snapshot'
BATCH_COMMAND_PREFIX = ['uv', 'run', 'nexus', 'run-batch']
TRAINING_CONFIG_ENVS = [
    (
        'training-v1-normal',
        {
            'IS_TRAINING': 'true',
            'SEARCH_TOOLSET_VERSION': 'v1',
            'SEARCH_EVAL_MODE': 'llm_judge',
            'SEARCH_MAX_ITERATIONS': '10',
            'SEARCH_AGENT_MAX_LLM_RETRIES': '0',
            'TRAINING_LENGTH_EXCEEDED': 'true',
            'SEARCH_TEMPERATURE': '1',
        },
    ),
    (
        'training-v2-normal',
        {
            'IS_TRAINING': 'true',
            'SEARCH_TOOLSET_VERSION': 'v2',
            'SEARCH_EVAL_MODE': 'llm_judge',
            'SEARCH_MAX_ITERATIONS': '100',
            'SEARCH_AGENT_MAX_LLM_RETRIES': '0',
            'TRAINING_LENGTH_EXCEEDED': 'true',
            'SEARCH_TEMPERATURE': '1',
        },
    ),
]
INFERENCE_CONFIG_ENVS = [
    (
        'inference-v1-normal-llm-judge',
        {
            'IS_TRAINING': 'false',
            'SEARCH_TOOLSET_VERSION': 'v1',
            'SEARCH_EVAL_MODE': 'llm_judge',
            'SEARCH_MAX_ITERATIONS': '20',
            'SEARCH_AGENT_MAX_LLM_RETRIES': '3',
            'SEARCH_TIMEOUT_PER_RUN': '7200',
            'SEARCH_TEMPERATURE': '0',
        },
    ),
    (
        'inference-v2-normal-llm-judge',
        {
            'IS_TRAINING': 'false',
            'SEARCH_TOOLSET_VERSION': 'v2',
            'SEARCH_EVAL_MODE': 'llm_judge',
            'SEARCH_MAX_ITERATIONS': '200',
            'SEARCH_AGENT_MAX_LLM_RETRIES': '10',
            'SEARCH_TIMEOUT_PER_RUN': '10800',
            'SEARCH_TEMPERATURE': '1',
        },
    ),
    (
        'inference-v2-normal-exact-match',
        {
            'IS_TRAINING': 'false',
            'SEARCH_TOOLSET_VERSION': 'v2',
            'SEARCH_EVAL_MODE': 'exact_match',
            'SEARCH_MAX_ITERATIONS': '30',
            'SEARCH_AGENT_MAX_LLM_RETRIES': '5',
            'SEARCH_TIMEOUT_PER_RUN': '7200',
            'SEARCH_TEMPERATURE': '0',
        },
    ),
    (
        'inference-v2-normal-reward-service',
        {
            'IS_TRAINING': 'false',
            'SEARCH_TOOLSET_VERSION': 'v2',
            'SEARCH_EVAL_MODE': 'reward_service',
            'SEARCH_MAX_ITERATIONS': '50',
            'SEARCH_AGENT_MAX_LLM_RETRIES': '6',
            'SEARCH_TIMEOUT_PER_RUN': '10800',
            'SEARCH_TEMPERATURE': '1',
        },
    ),
]


def _make_dataset_config(tmp_path: Path, name: str = DEFAULT_DATASET_NAME) -> run_benchmarks_module.BenchmarkConfig:
    dataset_path = tmp_path / f'{name}.parquet'
    dataset_path.write_text('placeholder', encoding='utf-8')
    return run_benchmarks_module.BenchmarkConfig(name=name, dataset_path=dataset_path)


def _make_main_async_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    args = {
        'benchmarks': None,
        'config': None,
        'name': None,
        'output_dir': None,
        'max_concurrent': 10,
        'timeout': None,
        'benchmark_parallel': 1,
        'resume': False,
        'refresh_db_config': False,
        'rerun_zero_score': False,
        'group_parallel': 1,
        'log_trim': None,
        'env_file': tmp_path / '.env.search',
        'verbose': False,
        'dry_run': True,
        'runs': 1,
        'model': None,
        'llm_max_retries': None,
        'agent_max_llm_retries': None,
        'eval_mode': None,
        'no_table': False,
        'instance_ids': None,
        'run_ids': None,
        'limit': None,
        'db_sync_interval': 0.0,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def _assert_command_contains_flag_value(cmd: list[str], flag: str, expected: str) -> None:
    assert flag in cmd
    flag_index = cmd.index(flag)
    assert flag_index + 1 < len(cmd)
    assert cmd[flag_index + 1] == expected


def test_load_env_file_filters_dataset_path_and_empty_values(tmp_path: Path) -> None:
    env_file = tmp_path / '.env.search'
    env_file.write_text(
        '\n'.join(
            [
                '# comment',
                f'export SEARCH_OPENAI_MODEL="{SEARCH_MODEL}"',
                'SEARCH_DATASET_PATH=/tmp/dataset.parquet',
                'EMPTY_VALUE=',
                'JUDGE_OPENAI_MODEL=judge-model',
            ]
        ),
        encoding='utf-8',
    )

    env_vars = run_benchmarks_module.load_env_file(env_file)

    assert env_vars == {
        'SEARCH_OPENAI_MODEL': SEARCH_MODEL,
        'JUDGE_OPENAI_MODEL': 'judge-model',
    }


def test_load_env_file_returns_empty_when_missing(tmp_path: Path) -> None:
    """When .env.search is absent, fall back to empty env so a self-contained YAML can run."""
    missing_env_file = tmp_path / 'nonexistent.env'
    assert run_benchmarks_module.load_env_file(missing_env_file) == {}


def test_load_benchmarks_without_config_uses_defaults() -> None:
    benchmarks = run_benchmarks_module.load_benchmarks()

    assert benchmarks
    assert benchmarks[0].dataset_path.parent == run_benchmarks_module.DEFAULT_DATA_DIR
    assert {benchmark.name for benchmark in benchmarks} == set(run_benchmarks_module.DEFAULT_BENCHMARKS)


def test_load_benchmarks_reads_search_config_benchmarks() -> None:
    from nexus.extensions.tasks.search.config import SearchConfig

    cfg = SearchConfig(
        task_name='search',
        benchmarks={
            'demo_a': '/tmp/a.parquet',
            'demo_b': '/tmp/b.parquet',
        },
    )

    benchmarks = run_benchmarks_module.load_benchmarks(search_config=cfg)

    assert [(b.name, b.dataset_path) for b in benchmarks] == [
        ('demo_a', Path('/tmp/a.parquet')),
        ('demo_b', Path('/tmp/b.parquet')),
    ]


def test_load_benchmarks_falls_back_to_defaults_when_config_has_no_benchmarks() -> None:
    from nexus.extensions.tasks.search.config import SearchConfig

    cfg = SearchConfig(task_name='search')
    benchmarks = run_benchmarks_module.load_benchmarks(search_config=cfg)

    assert {b.name for b in benchmarks} == set(run_benchmarks_module.DEFAULT_BENCHMARKS)


def test_safe_copy_sqlite_db_copies_database_contents(tmp_path: Path) -> None:
    src = tmp_path / 'source.db'
    dst = tmp_path / 'dest.db'

    import sqlite3

    conn = sqlite3.connect(src)
    conn.execute('CREATE TABLE runs (run_id TEXT, score REAL)')
    conn.execute('INSERT INTO runs VALUES (?, ?)', ('run-1', 1.0))
    conn.commit()
    conn.close()

    copied = run_benchmarks_module.safe_copy_sqlite_db(src, dst)

    assert copied is True
    conn = sqlite3.connect(dst)
    rows = conn.execute('SELECT run_id, score FROM runs').fetchall()
    conn.close()
    assert rows == [('run-1', 1.0)]


@pytest.mark.asyncio
async def test_run_command_starts_child_session_and_watchdog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    create_calls: list[dict[str, object]] = []
    watchdog_calls: list[tuple[int, int]] = []
    stopped_watchdogs: list[object] = []
    fake_watchdog = object()

    class FakeStdout:
        def __init__(self) -> None:
            self._chunks = [b'hello\n', b'']

        async def read(self, _size: int) -> bytes:
            return self._chunks.pop(0)

    class FakeProcess:
        pid = 12345
        returncode = 0
        stdout = FakeStdout()

        async def wait(self) -> int:
            return 0

    async def fake_create_subprocess_exec(*_cmd: str, **kwargs: object) -> FakeProcess:
        create_calls.append(kwargs)
        return FakeProcess()

    async def fake_start_watchdog(parent_pid: int, child_pgid: int) -> object:
        watchdog_calls.append((parent_pid, child_pgid))
        return fake_watchdog

    async def fake_stop_watchdog(watchdog: object) -> None:
        stopped_watchdogs.append(watchdog)

    monkeypatch.setattr(run_benchmarks_module.asyncio, 'create_subprocess_exec', fake_create_subprocess_exec)
    monkeypatch.setattr(run_benchmarks_module, '_start_parent_death_watchdog', fake_start_watchdog)
    monkeypatch.setattr(run_benchmarks_module, '_stop_parent_death_watchdog', fake_stop_watchdog)

    returncode, stdout, stderr = await run_benchmarks_module.run_command(
        ['python', '-c', 'print("hello")'],
        env={},
        cwd=tmp_path,
        stream_output=True,
    )

    assert returncode == 0
    assert stdout == 'hello\n'
    assert stderr == ''
    assert create_calls[0]['start_new_session'] is True
    assert watchdog_calls == [(run_benchmarks_module.os.getpid(), FakeProcess.pid)]
    assert stopped_watchdogs == [fake_watchdog]


@pytest.mark.asyncio
async def test_sync_db_periodically_stops_after_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / 'source.db'
    dst = tmp_path / 'dest.db'
    src.write_text('placeholder', encoding='utf-8')
    copy_calls: list[tuple[Path, Path]] = []

    def fake_copy(src_path: Path, dst_path: Path) -> bool:
        copy_calls.append((src_path, dst_path))
        return True

    monkeypatch.setattr(run_benchmarks_module, 'safe_copy_sqlite_db', fake_copy)
    stop_event = run_benchmarks_module.asyncio.Event()
    task = run_benchmarks_module.asyncio.create_task(
        run_benchmarks_module.sync_db_periodically(src, dst, interval=0.01, stop_event=stop_event)
    )
    await run_benchmarks_module.asyncio.sleep(0.03)
    stop_event.set()
    await task

    assert copy_calls
    assert all(item == (src, dst) for item in copy_calls)


@pytest.mark.asyncio
async def test_run_benchmark_fails_fast_when_dataset_is_missing(tmp_path: Path) -> None:
    config = run_benchmarks_module.BenchmarkConfig(
        name=MISSING_DATASET_NAME,
        dataset_path=tmp_path / f'{MISSING_DATASET_NAME}.parquet',
    )

    result = await run_benchmarks_module.run_benchmark(
        config=config,
        output_dir=tmp_path / 'out',
        env_vars={},
        max_concurrent=1,
        timeout=None,
        db_sync_interval=0.0,
    )

    assert result.status == 'failed'
    assert result.error == f'Dataset not found: {config.dataset_path}'


@pytest.mark.asyncio
async def test_run_benchmark_success_path_exports_trajectory_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_dataset_config(tmp_path, DEFAULT_DATASET_NAME)
    call_record: dict[str, object] = {}

    async def fake_run_command(cmd, env, cwd, stream_output=False):
        call_record['cmd'] = cmd
        call_record['env'] = env
        call_record['cwd'] = cwd
        call_record['stream_output'] = stream_output
        return 0, BATCH_OK, ''

    class FakeBatchDatabase:
        def __init__(self, db_path: Path):
            call_record['db_path'] = db_path

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get_all_runs(self):
            return [
                {
                    'run_id': PRIMARY_RUN_ID,
                    'status': 'success',
                    'start_time': '2026-04-02T00:00:00+00:00',
                    'end_time': '2026-04-02T00:00:03+00:00',
                    'task_output': {'score': 1.0, 'judge_status': 'success'},
                    'agent_output': {'main_trajectory': [{'role': 'assistant', 'content': 'done'}]},
                    'processor_results': {'trace': RUN_TRACE},
                    'metadata': {'question': QUESTION_TEXT},
                }
            ]

    def fake_load_results_from_db(db_path: Path, *, include_agent_output: bool = False):
        call_record['metrics_db_path'] = db_path
        call_record['include_agent_output'] = include_agent_output
        return [{'run_id': PRIMARY_RUN_ID}]

    def fake_compute_metrics(results):
        call_record['metrics_input'] = results
        return SUCCESS_METRICS

    monkeypatch.setattr(run_benchmarks_module, 'run_command', fake_run_command)
    monkeypatch.setattr(
        run_benchmarks_module,
        'print_benchmark_metrics',
        lambda result: call_record.setdefault('printed_result', result),
    )
    monkeypatch.setattr(compute_metrics_module, 'load_results_from_db', fake_load_results_from_db)
    monkeypatch.setattr(compute_metrics_module, 'compute_metrics', fake_compute_metrics)
    monkeypatch.setitem(sys.modules, 'nexus.runner.batch', SimpleNamespace(BatchDatabase=FakeBatchDatabase))

    result = await run_benchmarks_module.run_benchmark(
        config=config,
        output_dir=tmp_path / 'out',
        env_vars={'SEARCH_OPENAI_MODEL': SEARCH_MODEL},
        max_concurrent=3,
        timeout=120,
        db_sync_interval=0.0,
        run_ids=PRIMARY_RUN_ID,
    )

    assert result.status == STATUS_SUCCESS
    assert result.metrics == SUCCESS_METRICS
    assert result.db_path == tmp_path / 'out' / 'search_demo.db'
    assert result.metrics_path == tmp_path / 'out' / 'metrics_demo.json'
    assert result.traj_path == tmp_path / 'out' / 'traj_demo.jsonl'
    cmd = call_record['cmd']
    assert cmd[:4] == BATCH_COMMAND_PREFIX
    _assert_command_contains_flag_value(cmd, '-g', str(run_benchmarks_module.DEFAULT_GENERATOR))
    _assert_command_contains_flag_value(cmd, '-d', str(tmp_path / 'out' / 'search_demo.db'))
    _assert_command_contains_flag_value(cmd, '--max-concurrent', '3')
    _assert_command_contains_flag_value(cmd, '--timeout', '120')
    assert call_record['cwd'] == run_benchmarks_module.PROJECT_ROOT
    assert call_record['stream_output'] is True
    assert call_record['env']['SEARCH_DATASET_PATH'] == str(config.dataset_path)
    assert call_record['env']['SEARCH_OPENAI_MODEL'] == SEARCH_MODEL
    assert call_record['env']['SEARCH_RUN_IDS'] == PRIMARY_RUN_ID
    assert call_record['db_path'] == result.db_path
    assert call_record['metrics_db_path'] == result.db_path
    assert call_record['include_agent_output'] is True
    traj_row = json.loads(result.traj_path.read_text(encoding='utf-8').strip())
    assert traj_row['run_id'] == PRIMARY_RUN_ID
    assert traj_row['task_output']['judge_status'] == STATUS_SUCCESS
    assert 'judge_status' not in traj_row
    assert json.loads(result.metrics_path.read_text(encoding='utf-8'))['accuracy'] == 1.0
    assert call_record['metrics_input'] == [{'run_id': PRIMARY_RUN_ID}]
    assert call_record['printed_result'] is result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('case_name', 'extra_env_vars'),
    TRAINING_CONFIG_ENVS + INFERENCE_CONFIG_ENVS,
    ids=[case_name for case_name, _ in TRAINING_CONFIG_ENVS + INFERENCE_CONFIG_ENVS],
)
async def test_run_benchmark_propagates_search_agent_configuration_env_vars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case_name: str,
    extra_env_vars: dict[str, str],
) -> None:
    config = _make_dataset_config(tmp_path, DEFAULT_DATASET_NAME)
    call_record: dict[str, object] = {}

    async def fake_run_command(cmd, env, cwd, stream_output=False):
        call_record['cmd'] = cmd
        call_record['env'] = env
        call_record['cwd'] = cwd
        call_record['stream_output'] = stream_output
        return 0, BATCH_OK, ''

    class FakeBatchDatabase:
        def __init__(self, db_path: Path):
            call_record['db_path'] = db_path

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get_all_runs(self):
            return [
                {
                    'run_id': PRIMARY_RUN_ID,
                    'status': STATUS_SUCCESS,
                    'task_output': {'score': 1.0},
                    'agent_output': {'main_trajectory': [{'role': 'assistant', 'content': 'done'}]},
                    'processor_results': {'trace': RUN_TRACE},
                    'metadata': {'question': QUESTION_TEXT},
                }
            ]

    monkeypatch.setattr(run_benchmarks_module, 'run_command', fake_run_command)
    monkeypatch.setattr(run_benchmarks_module, 'print_benchmark_metrics', lambda result: None)
    monkeypatch.setattr(
        compute_metrics_module,
        'load_results_from_db',
        lambda _, *, include_agent_output=False: [{'run_id': PRIMARY_RUN_ID}],
    )
    monkeypatch.setattr(compute_metrics_module, 'compute_metrics', lambda _: SUCCESS_METRICS_NO_TIMEOUT)
    monkeypatch.setitem(sys.modules, 'nexus.runner.batch', SimpleNamespace(BatchDatabase=FakeBatchDatabase))

    result = await run_benchmarks_module.run_benchmark(
        config=config,
        output_dir=tmp_path / 'out',
        env_vars={'SEARCH_OPENAI_MODEL': SEARCH_MODEL, **extra_env_vars},
        max_concurrent=2,
        timeout=45,
        db_sync_interval=0.0,
    )

    assert result.status == STATUS_SUCCESS
    assert case_name
    assert call_record['cwd'] == run_benchmarks_module.PROJECT_ROOT
    assert call_record['stream_output'] is True
    assert call_record['db_path'] == result.db_path
    assert call_record['env']['SEARCH_DATASET_PATH'] == str(config.dataset_path)
    assert call_record['env']['SEARCH_OPENAI_MODEL'] == SEARCH_MODEL
    for key, expected_value in extra_env_vars.items():
        assert call_record['env'][key] == expected_value


@pytest.mark.asyncio
async def test_run_benchmark_temp_db_sync_path_uses_snapshot_and_final_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_dataset_config(tmp_path, TEMP_DATASET_NAME)
    sync_record: dict[str, object] = {}

    async def fake_run_command(cmd, env, cwd, stream_output=False):
        sync_record['cmd'] = cmd
        return 0, BATCH_OK, ''

    async def fake_sync_db_periodically(src, dst, interval, stop_event):
        sync_record['sync_src'] = src
        sync_record['sync_dst'] = dst
        sync_record['sync_interval'] = interval
        await stop_event.wait()
        sync_record['sync_stopped'] = True

    def fake_copy2(src, dst):
        sync_record['copy2'] = (src, dst)
        dst.write_text(DB_SNAPSHOT_TEXT, encoding='utf-8')
        return str(dst)

    class FakeBatchDatabase:
        def __init__(self, db_path: Path):
            sync_record['db_path'] = db_path

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get_all_runs(self):
            return [
                {
                    'run_id': PRIMARY_RUN_ID,
                    'status': 'success',
                    'task_output': {'score': 1.0},
                    'agent_output': {},
                    'processor_results': {},
                    'metadata': {},
                }
            ]

    monkeypatch.setattr(run_benchmarks_module, 'run_command', fake_run_command)
    monkeypatch.setattr(run_benchmarks_module, 'sync_db_periodically', fake_sync_db_periodically)
    monkeypatch.setattr(run_benchmarks_module.shutil, 'copy2', fake_copy2)
    monkeypatch.setattr(run_benchmarks_module, 'print_benchmark_metrics', lambda result: None)

    def fake_load_results_from_db(db_path: Path, *, include_agent_output: bool = False):
        sync_record['metrics_db_path'] = db_path
        sync_record['include_agent_output'] = include_agent_output
        return [{'run_id': PRIMARY_RUN_ID}]

    monkeypatch.setattr(compute_metrics_module, 'load_results_from_db', fake_load_results_from_db)
    monkeypatch.setattr(compute_metrics_module, 'compute_metrics', lambda results: SUCCESS_METRICS_NO_TIMEOUT)
    monkeypatch.setitem(sys.modules, 'nexus.runner.batch', SimpleNamespace(BatchDatabase=FakeBatchDatabase))

    result = await run_benchmarks_module.run_benchmark(
        config=config,
        output_dir=tmp_path / 'out',
        env_vars={},
        max_concurrent=2,
        timeout=None,
        db_sync_interval=0.25,
    )

    assert result.status == STATUS_SUCCESS
    assert sync_record['sync_dst'] == tmp_path / 'out' / 'search_demo_temp.db'
    assert sync_record['sync_interval'] == 0.25
    copy_src, copy_dst = sync_record['copy2']
    assert copy_src == sync_record['sync_src']
    assert copy_dst == result.db_path
    assert copy_src != copy_dst
    assert sync_record['sync_stopped'] is True
    assert sync_record['db_path'] == copy_src
    assert sync_record['metrics_db_path'] == copy_src
    assert sync_record['include_agent_output'] is True
    _assert_command_contains_flag_value(sync_record['cmd'], '-d', str(copy_src))
    assert result.db_path.read_text(encoding='utf-8') == DB_SNAPSHOT_TEXT


@pytest.mark.asyncio
async def test_run_benchmark_returns_failed_result_when_command_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_dataset_config(tmp_path, BROKEN_DATASET_NAME)

    async def fake_run_command(cmd, env, cwd, stream_output=False):
        return 7, '', FATAL_ERROR

    monkeypatch.setattr(run_benchmarks_module, 'run_command', fake_run_command)

    result = await run_benchmarks_module.run_benchmark(
        config=config,
        output_dir=tmp_path / 'out',
        env_vars={},
        max_concurrent=1,
        timeout=30,
        db_sync_interval=0.0,
    )

    assert result.status == STATUS_FAILED
    assert result.db_path == tmp_path / 'out' / 'search_broken.db'
    assert result.error == f'Exit code 7: {FATAL_ERROR}'


@pytest.mark.asyncio
async def test_run_benchmark_recovers_metrics_when_batch_has_partial_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_dataset_config(tmp_path, BROKEN_DATASET_NAME)

    async def fake_run_command(cmd, env, cwd, stream_output=False):
        db_path = Path(cmd[cmd.index('-d') + 1])
        db_path.write_text('placeholder db', encoding='utf-8')
        return 1, 'Batch completed: 1/2 succeeded', ''

    class FakeBatchDatabase:
        def __init__(self, db_path: Path):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get_all_runs(self):
            return [
                {
                    'run_id': PRIMARY_RUN_ID,
                    'status': STATUS_SUCCESS,
                    'task_output': {'score': 1.0, 'judge_status': STATUS_SUCCESS},
                    'agent_output': {},
                    'processor_results': {},
                    'metadata': {},
                },
            ]

    monkeypatch.setattr(run_benchmarks_module, 'run_command', fake_run_command)
    monkeypatch.setattr(run_benchmarks_module, 'print_benchmark_metrics', lambda result: None)
    monkeypatch.setattr(
        compute_metrics_module,
        'load_results_from_db',
        lambda _, *, include_agent_output=False: [{'run_id': PRIMARY_RUN_ID}],
    )
    monkeypatch.setattr(compute_metrics_module, 'compute_metrics', lambda _: SUCCESS_METRICS)
    monkeypatch.setitem(sys.modules, 'nexus.runner.batch', SimpleNamespace(BatchDatabase=FakeBatchDatabase))

    result = await run_benchmarks_module.run_benchmark(
        config=config,
        output_dir=tmp_path / 'out',
        env_vars={},
        max_concurrent=1,
        timeout=30,
        db_sync_interval=0.0,
    )

    assert result.status == STATUS_SUCCESS
    assert result.metrics == SUCCESS_METRICS
    assert json.loads(result.metrics_path.read_text(encoding='utf-8')) == SUCCESS_METRICS


@pytest.mark.asyncio
async def test_main_async_dry_run_returns_zero_and_skips_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_dataset_config(tmp_path, DEFAULT_DATASET_NAME)

    monkeypatch.setattr(
        run_benchmarks_module,
        'load_env_file',
        lambda _: {'SEARCH_OPENAI_MODEL': ENV_SEARCH_MODEL},
    )
    monkeypatch.setattr(
        run_benchmarks_module,
        'load_benchmarks',
        lambda **kw: [config],
    )
    monkeypatch.setattr(
        run_benchmarks_module,
        'run_all_benchmarks',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('dry-run should not execute benchmarks')),
    )
    args = _make_main_async_args(tmp_path, model='override/model')

    exit_code = await run_benchmarks_module.main_async(args)

    assert exit_code == 0


@pytest.mark.asyncio
async def test_main_async_validates_limit_and_resume_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_dataset_config(tmp_path, DEFAULT_DATASET_NAME)
    monkeypatch.setattr(run_benchmarks_module, 'load_env_file', lambda _: {'SEARCH_OPENAI_MODEL': ENV_SEARCH_MODEL})
    monkeypatch.setattr(run_benchmarks_module, 'load_benchmarks', lambda **kw: [config])

    negative_limit_args = _make_main_async_args(tmp_path, limit=-1, dry_run=False)
    missing_resume_args = _make_main_async_args(tmp_path, resume=True, dry_run=False)

    assert await run_benchmarks_module.main_async(negative_limit_args) == 1
    assert await run_benchmarks_module.main_async(missing_resume_args) == 1


@pytest.mark.asyncio
async def test_main_async_validates_db_sync_interval_and_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_dataset_config(tmp_path, DEFAULT_DATASET_NAME)
    monkeypatch.setattr(run_benchmarks_module, 'load_env_file', lambda _: {'SEARCH_OPENAI_MODEL': ENV_SEARCH_MODEL})
    monkeypatch.setattr(run_benchmarks_module, 'load_benchmarks', lambda **kw: [config])

    negative_db_sync_args = _make_main_async_args(tmp_path, db_sync_interval=-0.5, dry_run=False)
    invalid_runs_args = _make_main_async_args(tmp_path, runs=0, dry_run=False)

    assert await run_benchmarks_module.main_async(negative_db_sync_args) == 1
    assert await run_benchmarks_module.main_async(invalid_runs_args) == 1


@pytest.mark.asyncio
async def test_main_async_rejects_unknown_toolset_version_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_dataset_config(tmp_path, DEFAULT_DATASET_NAME)
    monkeypatch.setattr(
        run_benchmarks_module,
        'load_env_file',
        lambda _: {
            'SEARCH_OPENAI_MODEL': ENV_SEARCH_MODEL,
            'SEARCH_TOOLSET_VERSION': 'invalid_version',
        },
    )
    monkeypatch.setattr(run_benchmarks_module, 'load_benchmarks', lambda **kw: [config])

    args = _make_main_async_args(tmp_path, dry_run=False)

    assert await run_benchmarks_module.main_async(args) == 1


@pytest.mark.asyncio
async def test_main_async_rejects_unknown_benchmark_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_dataset_config(tmp_path, DEFAULT_DATASET_NAME)
    monkeypatch.setattr(run_benchmarks_module, 'load_env_file', lambda _: {'SEARCH_OPENAI_MODEL': ENV_SEARCH_MODEL})
    monkeypatch.setattr(run_benchmarks_module, 'load_benchmarks', lambda **kw: [config])

    args = _make_main_async_args(tmp_path, benchmarks=['unknown-benchmark'], dry_run=False)

    assert await run_benchmarks_module.main_async(args) == 1


@pytest.mark.asyncio
async def test_main_async_returns_failure_when_all_selected_benchmarks_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_config = run_benchmarks_module.BenchmarkConfig(
        name=MISSING_DATASET_NAME,
        dataset_path=tmp_path / 'missing.parquet',
    )
    monkeypatch.setattr(run_benchmarks_module, 'load_env_file', lambda _: {'SEARCH_OPENAI_MODEL': ENV_SEARCH_MODEL})
    monkeypatch.setattr(run_benchmarks_module, 'load_benchmarks', lambda **kw: [missing_config])

    args = _make_main_async_args(tmp_path, dry_run=False)

    assert await run_benchmarks_module.main_async(args) == 1


@pytest.mark.asyncio
async def test_main_async_uses_plain_summary_and_returns_failure_when_any_benchmark_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_dataset_config(tmp_path, DEFAULT_DATASET_NAME)
    summary_record: dict[str, object] = {}

    monkeypatch.setattr(run_benchmarks_module, 'load_env_file', lambda _: {'SEARCH_OPENAI_MODEL': ENV_SEARCH_MODEL})
    monkeypatch.setattr(run_benchmarks_module, 'load_benchmarks', lambda **kw: [config])
    monkeypatch.setattr(run_benchmarks_module, 'build_execution_metadata', lambda *args: {'built': True})

    async def fake_run_all_benchmarks(*args, **kwargs):
        summary = run_benchmarks_module.RunSummary(output_dir=tmp_path / 'output')
        summary.results.append(
            run_benchmarks_module.BenchmarkResult(
                name=DEFAULT_DATASET_NAME,
                status=STATUS_FAILED,
                start_time=run_benchmarks_module.datetime.now(run_benchmarks_module.timezone.utc),
                end_time=run_benchmarks_module.datetime.now(run_benchmarks_module.timezone.utc),
                error=BOOM_ERROR,
            )
        )
        summary.end_time = run_benchmarks_module.datetime.now(run_benchmarks_module.timezone.utc)
        return summary

    monkeypatch.setattr(run_benchmarks_module, 'run_all_benchmarks', fake_run_all_benchmarks)
    monkeypatch.setattr(
        run_benchmarks_module,
        'print_summary',
        lambda summary: summary_record.setdefault('printed_summary', summary),
    )
    monkeypatch.setattr(run_benchmarks_module, 'save_summary', lambda summary, meta: tmp_path / RUN_SUMMARY_JSON)

    args = _make_main_async_args(tmp_path, dry_run=False, no_table=True)
    exit_code = await run_benchmarks_module.main_async(args)

    assert exit_code == 1
    assert summary_record['printed_summary'].results[0].status == STATUS_FAILED
