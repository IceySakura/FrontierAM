"""script_probe 原语测试（Commit 3，增量新增）。

覆盖：基本运行、contains 谓词、regex 谓词、非零退出、
无运行时、超时。"""

import asyncio

from types import SimpleNamespace

from nexus.extensions.tasks.aidata.rubric import run_rubric_async


class StubRuntime:
    def __init__(self, stdout='', return_code=0, delay=0):
        self.stdout = stdout
        self.return_code = return_code
        self.delay = delay

    async def exec(self, cmd, **kw):
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(stdout=self.stdout, return_code=self.return_code)


def _score(rubric, state=None, runtime=None):
    return asyncio.run(run_rubric_async(state or {'agent_answer': '', 'trajectory': []}, rubric, runtime=runtime))


def test_probe_pass_default_exit_zero():
    r = {'version': '4.2', 'checks': [{'id': 'p', 'assert': {'type': 'script_probe', 'script': 'echo hi'}}]}
    rows = _score(r, runtime=StubRuntime(stdout='hi\n', return_code=0))
    assert rows[0]['passed'], rows


def test_probe_fail_on_nonzero_exit():
    r = {'version': '4.2', 'checks': [{'id': 'p', 'assert': {'type': 'script_probe', 'script': 'exit 1'}}]}
    rows = _score(r, runtime=StubRuntime(stdout='', return_code=1))
    assert not rows[0]['passed'], rows
    assert rows[0]['verdict'] == 'N', rows


def test_probe_contains_predicate_pass():
    r = {
        'version': '4.2',
        'checks': [{'id': 'p', 'assert': {'type': 'script_probe', 'script': 'echo hello', 'contains': 'hello'}}],
    }
    rows = _score(r, runtime=StubRuntime(stdout='hello world\n', return_code=0))
    assert rows[0]['passed'], rows


def test_probe_contains_predicate_fail():
    r = {
        'version': '4.2',
        'checks': [{'id': 'p', 'assert': {'type': 'script_probe', 'script': 'echo hi', 'contains': 'goodbye'}}],
    }
    rows = _score(r, runtime=StubRuntime(stdout='hi\n', return_code=0))
    assert not rows[0]['passed'], rows


def test_probe_regex_predicate():
    r = {
        'version': '4.2',
        'checks': [{'id': 'p', 'assert': {'type': 'script_probe', 'script': 'echo x', 'regex': r'\bhello\b'}}],
    }
    rows = _score(r, runtime=StubRuntime(stdout='say hello world\n', return_code=0))
    assert rows[0]['passed'], rows


def test_probe_no_runtime_yields_unavailable():
    r = {'version': '4.2', 'checks': [{'id': 'p', 'assert': {'type': 'script_probe', 'script': 'echo hi'}}]}
    rows = _score(r, runtime=None)
    assert rows[0]['verdict'] == 'UNAVAILABLE', rows


def test_probe_missing_script_is_invalid():
    r = {'version': '4.2', 'checks': [{'id': 'p', 'assert': {'type': 'script_probe'}}]}
    rows = _score(r, runtime=StubRuntime())
    assert rows[0]['verdict'] == 'INVALID', rows
