from __future__ import annotations

import json
import subprocess
import threading

import pytest

from nexus.utils import tool_analyzer as analyzer


@pytest.fixture(autouse=True)
def fresh_parser():
    analyzer._bashlex_parser.close()
    analyzer._parse_command_words_cached.cache_clear()
    yield
    analyzer._bashlex_parser.close()
    analyzer._parse_command_words_cached.cache_clear()


# Same template-literal/escaped-quote pattern as the incident, without task data.
EDIT_ARGS = {
    'file_path': '/tmp/example.ts',
    'old_string': "function quote(value: string): string {\n  return `'${value.replace(/'/g, \"''\")}'`;\n}",
    'new_string': 'function quote(value: string): string { return value; }',
}


def turn(name, args):
    return analyzer.compute_swe_turn_stats(
        {'tool_calls': [{'id': 'call', 'function': {'name': name, 'arguments': args}}]},
        [{'tool_call_id': 'call', 'content': 'ok'}],
    )


@pytest.fixture
def no_parser(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('non-shell input reached the parser')

    monkeypatch.setattr(analyzer, 'extract_sub_commands', unexpected)


@pytest.mark.parametrize(
    ('name', 'args', 'command', 'category'),
    [
        ('apply_patch', {'input': EDIT_ARGS['old_string']}, 'apply_patch', 'text_editing'),
        ('str_replace_editor', {'command': 'view'}, 'view', 'text_editing'),
        ('str_replace_based_edit_tool', {'command': 'str_replace'}, 'str_replace', 'text_editing'),
        ('editor', {'action': 'create'}, 'create', 'text_editing'),
        ('editor', {'command': 'change'}, 'change', 'unknown'),
        ('finish', {}, 'submit', 'control_flow'),
    ],
)
def test_existing_non_shell_tools_keep_labels_without_parsing(no_parser, name, args, command, category):
    result = turn(name, json.dumps(args))
    assert result['parse_total'] == result['exec_total'] == 1
    assert result['commands'][0]['sub_cmds'] == [command]
    assert result['commands'][0]['sub_categories'] == [category]
    assert result['commands'][0]['raw'] == command


def test_empty_editor_action_remains_excluded(no_parser):
    result = turn('editor', {})
    assert result['parse_total'] == result['exec_total'] == 0
    assert result['commands'][0]['excluded_from_stats']


@pytest.mark.parametrize(
    'name', ['Edit', 'agent/Edit', 'MultiEdit', 'Write', 'Read', 'Glob', 'Grep', 'UnknownTool', 'TodoWrite', 'edit']
)
@pytest.mark.parametrize('encode_json', [False, True])
def test_json_fallback_tools_are_excluded(no_parser, name, encode_json):
    result = turn(name, json.dumps(EDIT_ARGS) if encode_json else EDIT_ARGS)
    assert result['parse_total'] == result['exec_total'] == 0
    assert result['commands'][0]['excluded_from_stats']


@pytest.mark.parametrize('name', ['bash', 'Bash', 'agent/bash', 'agent/Bash', 'shell'])
def test_only_explicit_shell_tools_parse(monkeypatch, name):
    seen = []

    def parse(text):
        seen.append(text)
        return [{'name': 'ls', 'category': 'file_management', 'raw': text}]

    monkeypatch.setattr(analyzer, 'extract_sub_commands', parse)
    command = ['bash', '-lc', 'ls -la'] if name == 'shell' else 'ls -la'
    assert turn(name, {'command': command})['exec_total'] == 1
    assert seen == ['ls -la']


def test_non_string_bash_command_is_not_stringified(no_parser):
    result = turn('bash', {'command': EDIT_ARGS})
    assert result['commands'][0]['excluded_from_stats']


@pytest.fixture
def track_processes(monkeypatch):
    processes = []
    original = subprocess.Popen

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(analyzer.subprocess, 'Popen', spawn)
    monkeypatch.setattr(analyzer, 'bashlex', object())
    monkeypatch.setattr(analyzer, '_BASHLEX_AVAILABLE', True)
    yield processes
    analyzer._bashlex_parser.close()
    for process in processes:
        if process.poll() is None:
            process.kill()
            process.wait()
            pytest.fail('parser subprocess survived close()')


@pytest.fixture
def real_parser(monkeypatch):
    pytest.importorskip('bashlex')
    monkeypatch.setattr(analyzer, '_BASHLEX_TIMEOUT_SECONDS', 3.0)


@pytest.mark.parametrize(
    ('text', 'names'),
    [
        ('ls -la && python -m pytest', ['ls', 'python']),
        ('FOO=1 echo "hello world" | cat', ['echo', 'cat']),
        ('x=$(date); echo "$x"', ['date', 'echo']),
        ('for x in a b; do echo "$x"; done', ['echo']),
        ("cat <<'EOF'\nsource code\nEOF\n", ['cat']),
        ('echo "中文"', ['echo']),
        ('FOO=1', []),
        ('echo "unterminated', []),
    ],
)
def test_shell_semantics(real_parser, track_processes, text, names):
    assert [command['name'] for command in analyzer.extract_sub_commands(text)] == names
    assert len(track_processes) == 1


def test_parser_process_is_reused_across_commands(real_parser, track_processes):
    assert analyzer._safe_bashlex_parse('ls -la') == [['ls', '-la']]
    assert analyzer._safe_bashlex_parse('echo ok') == [['echo', 'ok']]
    assert analyzer._safe_bashlex_parse('echo "unterminated') is None
    assert analyzer._safe_bashlex_parse('cat a | wc -l') == [['cat', 'a'], ['wc', '-l']]
    assert len(track_processes) == 1
    assert track_processes[0].poll() is None


def test_parse_from_threads_shares_one_process(real_parser, track_processes):
    results: dict[int, list[list[str]] | None] = {}

    def parse(index: int) -> None:
        results[index] = analyzer._safe_bashlex_parse(f'echo {index}')

    threads = [threading.Thread(target=parse, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == {index: [['echo', str(index)]] for index in range(8)}
    assert len(track_processes) == 1


def test_dead_parser_is_restarted_before_next_request(real_parser, track_processes):
    assert analyzer._safe_bashlex_parse('echo ok') == [['echo', 'ok']]
    track_processes[0].kill()
    track_processes[0].wait()
    assert analyzer._safe_bashlex_parse('ls') == [['ls']]
    assert len(track_processes) == 2


def test_parser_over_rss_limit_is_replaced_between_requests(real_parser, track_processes, monkeypatch):
    monkeypatch.setattr(analyzer, '_BASHLEX_WORKER_MAX_RSS_MB', 0.001)
    assert analyzer._safe_bashlex_parse('echo ok') == [['echo', 'ok']]
    assert track_processes[0].returncode is not None
    assert analyzer._safe_bashlex_parse('ls') == [['ls']]
    assert len(track_processes) == 2


def test_reset_after_fork_starts_a_new_process_without_touching_the_old_one(real_parser, track_processes):
    parser = analyzer._BashlexParserProcess()
    assert parser.parse('echo ok') == [['echo', 'ok']]
    parser.reset_after_fork()
    assert parser.parse('ls') == [['ls']]
    assert len(track_processes) == 2
    assert track_processes[0].poll() is None
    parser.close()
    track_processes[0].kill()
    track_processes[0].wait()


class CountingParser:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.texts: list[str] = []

    def parse(self, text):
        self.texts.append(text)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def counting_parser(monkeypatch):
    def install(*outcomes):
        parser = CountingParser(outcomes)
        monkeypatch.setattr(analyzer, '_bashlex_parser', parser)
        monkeypatch.setattr(analyzer, 'bashlex', object())
        return parser

    return install


def test_repeated_command_is_parsed_once_and_returned_as_a_copy(counting_parser):
    parser = counting_parser([['echo', 'ok']])
    first = analyzer._safe_bashlex_parse('echo ok')
    assert first is not None
    first[0].append('mutated')
    assert analyzer._safe_bashlex_parse('echo ok') == [['echo', 'ok']]
    assert parser.texts == ['echo ok']


def test_rejected_input_is_cached(counting_parser):
    parser = counting_parser(None)
    assert analyzer._safe_bashlex_parse('echo "unterminated') is None
    assert analyzer._safe_bashlex_parse('echo "unterminated') is None
    assert parser.texts == ['echo "unterminated']


def test_command_longer_than_the_cache_limit_is_parsed_every_time(counting_parser, monkeypatch):
    monkeypatch.setattr(analyzer, '_BASHLEX_CACHE_MAX_INPUT_LEN', 20)
    long_command = 'echo ' + 'x' * 16
    short_command = 'echo ' + 'x' * 15
    parser = counting_parser([['echo', 'x' * 16]], [['echo', 'x' * 16]], [['echo', 'x' * 15]])

    assert analyzer._safe_bashlex_parse(long_command) == [['echo', 'x' * 16]]
    assert analyzer._safe_bashlex_parse(long_command) == [['echo', 'x' * 16]]
    assert analyzer._safe_bashlex_parse(short_command) == [['echo', 'x' * 15]]
    assert analyzer._safe_bashlex_parse(short_command) == [['echo', 'x' * 15]]
    assert parser.texts == [long_command, long_command, short_command]
    assert analyzer._parse_command_words_cached.cache_info().currsize == 1


def test_timeout_is_not_cached(counting_parser, caplog):
    parser = counting_parser(analyzer._BashlexParserUnavailableError('timed out', timed_out=True), [['echo', 'ok']])
    assert analyzer._safe_bashlex_parse('echo ok') is None
    assert 'killing parser' in caplog.text
    assert analyzer._safe_bashlex_parse('echo ok') == [['echo', 'ok']]
    assert parser.texts == ['echo ok', 'echo ok']


def test_timeout_kills_reaps_and_next_parse_works(tmp_path, monkeypatch, track_processes, caplog):
    original_worker = analyzer._BASHLEX_WORKER_PATH
    worker = tmp_path / 'blocked.py'
    worker.write_text('import time\ntime.sleep(60)\n')
    monkeypatch.setattr(analyzer, '_BASHLEX_WORKER_PATH', worker)
    monkeypatch.setattr(analyzer, '_BASHLEX_TIMEOUT_SECONDS', 0.1)
    for _ in range(3):
        assert analyzer._safe_bashlex_parse('echo ok') is None
        assert track_processes[-1].returncode is not None
        assert track_processes[-1].returncode != 0
        assert track_processes[-1].stdin.closed
        assert track_processes[-1].stdout.closed
    assert 'killing parser' in caplog.text
    pytest.importorskip('bashlex')
    monkeypatch.setattr(analyzer, '_BASHLEX_WORKER_PATH', original_worker)
    monkeypatch.setattr(analyzer, '_BASHLEX_TIMEOUT_SECONDS', 3)
    assert analyzer._safe_bashlex_parse('echo ok') == [['echo', 'ok']]


def test_timeout_covers_a_request_larger_than_the_pipe(tmp_path, monkeypatch, track_processes):
    worker = tmp_path / 'not_reading.py'
    worker.write_text('import time\ntime.sleep(60)\n')
    monkeypatch.setattr(analyzer, '_BASHLEX_WORKER_PATH', worker)
    monkeypatch.setattr(analyzer, '_BASHLEX_TIMEOUT_SECONDS', 0.1)
    # Within _BASHLEX_MAX_INPUT_LEN, but each astral character is JSON-escaped to 12 bytes, so
    # the ~108 KB request exceeds the 64 KiB pipe buffer of a parser that never reads stdin.
    text = '\U0001f600' * 9000
    results: list[list[list[str]] | None] = []
    parse_thread = threading.Thread(target=lambda: results.append(analyzer._safe_bashlex_parse(text)), daemon=True)
    threads_before = set(threading.enumerate())

    parse_thread.start()
    parse_thread.join(timeout=5)

    assert not parse_thread.is_alive(), 'parse blocked writing the request past its timeout'
    assert results == [None]
    assert track_processes[-1].returncode is not None
    assert track_processes[-1].stdin.closed
    assert not [thread for thread in set(threading.enumerate()) - threads_before if thread.name.startswith('bashlex-')]


def test_request_larger_than_the_pipe_is_parsed(real_parser, track_processes):
    word = '\U0001f600' * 9000

    assert analyzer._safe_bashlex_parse(f'echo {word}') == [['echo', word]]
    assert analyzer._safe_bashlex_parse('ls -la') == [['ls', '-la']]
    assert len(track_processes) == 1


def test_existing_input_limits_do_not_spawn(monkeypatch, track_processes):
    monkeypatch.setattr(analyzer, '_BASHLEX_MAX_INPUT_LEN', 5)
    assert analyzer._safe_bashlex_parse('echo too long') is None
    monkeypatch.setattr(analyzer, '_BASHLEX_MAX_INPUT_LEN', 10000)
    assert analyzer._safe_bashlex_parse('$(' * 51) is None
    assert not track_processes


def test_spawn_failure_excludes_command(monkeypatch):
    monkeypatch.setattr(analyzer, 'bashlex', object())

    def fail(*args, **kwargs):
        raise OSError('cannot spawn')

    monkeypatch.setattr(analyzer.subprocess, 'Popen', fail)
    assert analyzer._safe_bashlex_parse('echo ok') is None


@pytest.mark.parametrize('body', ['import sys\nsys.exit(2)\n', 'print("invalid json")\n'])
def test_worker_failure_is_reaped(tmp_path, monkeypatch, track_processes, body):
    worker = tmp_path / 'failed.py'
    worker.write_text(body)
    monkeypatch.setattr(analyzer, '_BASHLEX_WORKER_PATH', worker)
    monkeypatch.setattr(analyzer, '_BASHLEX_TIMEOUT_SECONDS', 3)
    assert analyzer._safe_bashlex_parse('echo ok') is None
    assert track_processes[-1].returncode is not None
    assert track_processes[-1].stdin.closed
    assert track_processes[-1].stdout.closed
