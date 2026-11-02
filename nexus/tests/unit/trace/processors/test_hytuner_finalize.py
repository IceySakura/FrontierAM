from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.runner.single import ProcessorConfig, RunInput, run
from nexus.trace.processors import hytuner as hytuner_module
from nexus.trace.processors.hytuner import HytunerProcessor


def read_dump_lines(root: Path, step_id: int) -> list[dict]:
    dump_files = list((root / 'chat_completions' / f'global_step_{step_id}').glob('*.jsonl'))
    assert len(dump_files) == 1
    return [json.loads(line) for line in dump_files[0].read_text().splitlines()]


def test_finalize_leaves_the_dump_to_afinalize(tmp_path):
    processor = HytunerProcessor(file_path=str(tmp_path), trajectory_id='3_task_0')
    processor.finalize()
    assert not (tmp_path / 'chat_completions').exists()

    asyncio.run(processor.afinalize())

    assert [line['trajectory_id'] for line in read_dump_lines(tmp_path, 3)] == ['3_task_0']


def test_afinalize_computes_tool_stats_off_the_event_loop(tmp_path, monkeypatch):
    processor = HytunerProcessor(file_path=str(tmp_path), trajectory_id='3_task_0')
    recompute_threads: list[int] = []
    original_recompute = processor._recompute_tool_call_stats

    def slow_recompute() -> None:
        recompute_threads.append(threading.get_ident())
        time.sleep(0.3)
        original_recompute()

    monkeypatch.setattr(processor, '_recompute_tool_call_stats', slow_recompute)

    async def main() -> int:
        ticks = 0
        finalize_task = asyncio.create_task(processor.afinalize())
        while not finalize_task.done():
            await asyncio.sleep(0.01)
            ticks += 1
        await finalize_task
        return ticks

    loop_thread = threading.get_ident()
    ticks = asyncio.run(main())
    assert ticks >= 10
    assert recompute_threads
    assert all(thread != loop_thread for thread in recompute_threads)
    assert len(read_dump_lines(tmp_path, 3)) == 1


def bash_call(call_id: str, command: str) -> dict:
    function = {'name': 'bash', 'arguments': json.dumps({'command': command})}
    return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': call_id, 'function': function}]}


def training_data(messages: list[dict]) -> dict:
    return {
        'prompt_ids': [1, 2],
        'raw_prompt_ids': [1, 2],
        'response_ids': [3],
        'response_logprobs': [-0.1],
        'messages': messages,
    }


def test_afinalize_covers_the_buffered_last_turn_so_get_result_parses_nothing(tmp_path, monkeypatch):
    processor = HytunerProcessor(file_path=str(tmp_path), trajectory_id='3_task_0')
    first_turn = [{'role': 'user', 'content': 'go'}, bash_call('a', 'ls'), {'role': 'tool', 'tool_call_id': 'a'}]
    last_turn = [*first_turn, bash_call('b', 'grep foo main.py')]
    processor.record_training_data(training_data(first_turn))
    processor.record_training_data(training_data(last_turn))
    stats_windows: list[tuple[int, str]] = []

    def recording_build_turn_stats(window, **_kwargs):
        call_id = window[0]['tool_calls'][0]['id']
        stats_windows.append((threading.get_ident(), call_id))
        return {'call_id': call_id}

    monkeypatch.setattr(hytuner_module, 'build_turn_stats_from_messages', recording_build_turn_stats)

    asyncio.run(processor.afinalize())

    assert [call_id for _, call_id in stats_windows] == ['a', 'b']
    assert threading.get_ident() not in {thread for thread, _ in stats_windows}
    (dump,) = read_dump_lines(tmp_path, 3)
    assert dump['messages'] == last_turn
    assert dump['num_turns'] == 2
    assert dump['tool_call_stats'] == [{'call_id': 'a'}, {'call_id': 'b'}]
    stats_windows.clear()
    assert processor.get_result()['num_turns'] == 2
    assert stats_windows == []


def test_afinalize_without_dump_path_still_precomputes_tool_stats(monkeypatch):
    processor = HytunerProcessor(file_path=None, trajectory_id='3_task_0')
    recompute_calls: list[str] = []
    monkeypatch.setattr(processor, '_recompute_tool_call_stats', lambda: recompute_calls.append('recompute'))
    asyncio.run(processor.afinalize())
    assert recompute_calls == ['recompute']


def test_afinalize_logs_a_failed_dump_instead_of_raising(tmp_path, monkeypatch, caplog):
    processor = HytunerProcessor(file_path=str(tmp_path), trajectory_id='3_task_0')

    def fail(*_args, **_kwargs) -> None:
        raise OSError('disk gone')

    monkeypatch.setattr(processor, 'write_trajectory_to_file', fail)
    asyncio.run(processor.afinalize())
    assert 'finalizing trajectory 3_task_0 failed' in caplog.text


def dump_stem(root: Path) -> str:
    """Finalize one trajectory into ``root`` and return its dump file's name."""
    hytuner_module._node_dump_stem.cache_clear()
    try:
        asyncio.run(HytunerProcessor(file_path=str(root), trajectory_id='3_task_0').afinalize())
        (dump_file,) = (root / 'chat_completions' / 'global_step_3').glob('*.jsonl')
        return dump_file.stem
    finally:
        hytuner_module._node_dump_stem.cache_clear()


def test_dump_file_is_named_after_the_node_address(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCAL_IP', '10.0.3.7')
    assert dump_stem(tmp_path) == '10-0-3-7'


def test_dump_file_stays_unique_without_a_routable_address(tmp_path, monkeypatch):
    # Loopback is what collapsed every node onto one file, and flock cannot keep
    # cross-node appends to a shared file from interleaving.
    monkeypatch.setenv('LOCAL_IP', '127.0.0.1')
    monkeypatch.delenv('NODE_IP', raising=False)
    monkeypatch.setattr(hytuner_module, '_outbound_address', lambda: '')

    stems = [dump_stem(tmp_path / str(index)) for index in range(2)]

    assert not any(stem.startswith('127-') for stem in stems)
    assert stems[0] != stems[1]


class DumpTaskOutput(TaskRunOutput):
    result: str


@pytest.fixture
def task_name() -> str:
    name = f'test_hytuner_dump_task_{uuid.uuid4().hex}'

    @task_registry.register(name)
    async def dump_task(_task_input: TaskRunInput) -> DumpTaskOutput:
        return DumpTaskOutput(result='done')

    return name


def test_single_run_writes_the_dump_before_collecting_results(tmp_path, task_name):
    run_output = asyncio.run(
        run(
            RunInput(
                run_id='3_scope_0',
                task_name=task_name,
                processors={
                    'hytuner': ProcessorConfig(
                        name='hytuner',
                        args={'file_path': str(tmp_path), 'trajectory_id': '3_scope_0'},
                    )
                },
            )
        )
    )

    assert run_output.success
    assert 'hytuner' in run_output.processor_results
    assert [line['trajectory_id'] for line in read_dump_lines(tmp_path, 3)] == ['3_scope_0']


def phase_span(name: str, seconds: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        attributes={'type': 'task_phase', 'phase': name.split('.')[1]},
        start_time_unix_nano=1_000_000_000,
        end_time_unix_nano=1_000_000_000 + seconds * 1_000_000_000,
    )


def test_result_reports_harbor_subphase_durations():
    processor = HytunerProcessor(file_path=None, trajectory_id='3_task_0')
    for span in (
        phase_span('task.handoff.artifact_collect_time', 1),
        phase_span('task.handoff.env_setup_time', 2),
        phase_span('task.handoff.release_main_time', 3),
        phase_span('task.scoring.check_score_time', 5),
    ):
        processor.on_end(span, SimpleNamespace())

    metrics = processor.get_result()['metrics']

    assert metrics['handoff_artifact_collect_time'] == 1.0
    assert metrics['handoff_env_setup_time'] == 2.0
    assert metrics['handoff_release_main_time'] == 3.0
    assert metrics['scoring_check_score_time'] == 5.0
    # Shared verification never acquires a verifier sandbox.
    assert metrics['handoff_sandbox_init_time'] == 0.0
