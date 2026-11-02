"""Fetcher 门面 + 新数据源形式的单元测试（ aidata 重构 Commit 3 ）。

覆盖：
- 传统字符串 'file:...' 形式：与 _gather_source 路径逐字节相同
- 使用主沙箱 + 无解析的字典形式：与传统字符串结果相同
- parse_as ： json / yaml / csv 解析有效内容
- parse_as 遇到缺失/损坏内容：优雅回退
- Trajectory + text_source 门面：与 _gather_trajectory / _gather_source 逐字节相同
- probe() ：运行时缺失时返回 ('', -1) ；接入 stub 运行时可成功执行

此处未覆盖（需真实 sidecar ）：
- FileSource(service=<real-sidecar>) —— 推迟到集成测试
"""

import asyncio

from types import SimpleNamespace

from nexus.extensions.tasks.aidata.rubric._impl import _gather_source, _gather_trajectory
from nexus.extensions.tasks.aidata.rubric.fetch import Fetcher, FileSource, ProbeSpec


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_trajectory_facade_matches_gather():
    state = {
        'trajectory': [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
        ],
        'agent_answer': 'hello',
    }
    got = _run(Fetcher(state).trajectory())
    expected = _gather_trajectory(state)
    assert got == expected


def test_text_source_facade_matches_gather():
    state = {'agent_answer': 'foobar', 'trajectory': []}
    cfg = {'from': 'agent_answer'}
    got = _run(Fetcher(state).text_source(cfg))
    expected = _gather_source(state, 'agent_answer', None)
    assert got == expected


def test_file_string_form_reads_from_sandbox_files():
    state = {'_sandbox_files': {'/tmp/foo': 'contents-42'}, 'agent_answer': '', 'trajectory': []}
    got = _run(Fetcher(state).file('file:/tmp/foo'))
    assert got == 'contents-42', got


def test_file_dict_form_matches_string_form():
    state = {'_sandbox_files': {'/tmp/foo': 'aaa'}, 'agent_answer': '', 'trajectory': []}
    f = Fetcher(state)
    a = _run(f.file('file:/tmp/foo'))
    b = _run(f.file(FileSource(file='/tmp/foo')))
    assert a == b == 'aaa'


def test_file_parse_as_json():
    state = {'_sandbox_files': {'/tmp/j': '{"a": 1, "b": [2,3]}'}, 'agent_answer': '', 'trajectory': []}
    got = _run(Fetcher(state).file(FileSource(file='/tmp/j', parse_as='json')))
    assert got == {'a': 1, 'b': [2, 3]}


def test_file_parse_as_yaml():
    state = {'_sandbox_files': {'/tmp/y': 'a: 1\nb:\n  - 2\n  - 3\n'}, 'agent_answer': '', 'trajectory': []}
    got = _run(Fetcher(state).file(FileSource(file='/tmp/y', parse_as='yaml')))
    assert got == {'a': 1, 'b': [2, 3]}, got


def test_file_parse_as_csv():
    state = {'_sandbox_files': {'/tmp/c': 'a,b\n1,2\n3,4\n'}, 'agent_answer': '', 'trajectory': []}
    got = _run(Fetcher(state).file(FileSource(file='/tmp/c', parse_as='csv')))
    assert got == [{'a': '1', 'b': '2'}, {'a': '3', 'b': '4'}], got


def test_file_parse_as_json_missing_file_returns_none():
    state = {'_sandbox_files': {}, 'agent_answer': '', 'trajectory': []}
    got = _run(Fetcher(state).file(FileSource(file='/nope', parse_as='json')))
    assert got is None, got


def test_file_parse_as_malformed_falls_back_to_raw():
    state = {'_sandbox_files': {'/tmp/bad': '{not json'}, 'agent_answer': '', 'trajectory': []}
    got = _run(Fetcher(state).file(FileSource(file='/tmp/bad', parse_as='json')))
    # 损坏的 JSON 应返回原始字符串，而不是抛出异常
    assert got == '{not json', got


def test_probe_no_runtime_returns_negative_exit():
    state = {'agent_answer': '', 'trajectory': []}
    stdout, code = _run(Fetcher(state).probe(ProbeSpec(script='echo hi')))
    assert stdout == ''
    assert code == -1


def test_probe_stub_runtime_returns_stdout_and_exit_code():
    class StubRuntime:
        async def exec(self, cmd, **kw):
            return SimpleNamespace(stdout='hello world\n', return_code=0)

    state = {'agent_answer': '', 'trajectory': []}
    stdout, code = _run(Fetcher(state, runtime=StubRuntime()).probe(ProbeSpec(script='echo hi')))
    assert stdout == 'hello world\n'
    assert code == 0


def test_fetcher_seeded_into_state_by_run_rubric():
    from nexus.extensions.tasks.aidata.rubric import run_rubric_async

    state = {'agent_answer': 'x', 'trajectory': []}
    rubric = {
        'version': '4.2',
        'checks': [{'id': 'x', 'assert': {'type': 'text_contains', 'from': 'agent_answer', 'contains': 'x'}}],
    }
    _run(run_rubric_async(state, rubric, runtime=None))
    assert '_fetcher' in state, 'run_rubric_async should seed state["_fetcher"]'
    assert isinstance(state['_fetcher'], Fetcher)
