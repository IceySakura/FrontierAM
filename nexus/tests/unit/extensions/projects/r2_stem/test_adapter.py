from types import SimpleNamespace

import pytest

from nexus.extensions.projects.r2_stem.adapter import capture_answer_file, hide_verifier_from_agent


def _command_result(*, stdout: str = '', stderr: str = '', return_code: int = 0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, return_code=return_code)


class FakeRuntime:
    def __init__(self, events, *, captured_answer=''):
        self.events = events
        self.captured_answer = captured_answer

    async def run_command(self, command, **kwargs):
        self.events.append(('run_command', command))
        return _command_result()

    async def upload_file(self, files, overwrite=True):
        self.events.append(('upload_file', files, overwrite))
        return SimpleNamespace(results=[SimpleNamespace(success=True, path=item.path) for item in files])

    async def download_file(self, files):
        self.events.append(('download_file', files))
        return SimpleNamespace(
            files=[SimpleNamespace(content=self.captured_answer)],
            errors=None,
        )


@pytest.mark.asyncio
async def test_hide_verifier_clears_contents_without_unlinking_protected_roots() -> None:
    events = []

    await hide_verifier_from_agent(FakeRuntime(events))

    assert len(events) == 1
    command = events[0][1]
    assert 'find /tests -mindepth 1 -delete' in command
    assert 'find /logs/verifier -mindepth 1 -delete' in command
    assert 'rm -rf /tests' not in command
    assert 'rm -rf /logs/verifier' not in command


@pytest.mark.asyncio
async def test_hide_verifier_fails_closed_when_contents_cannot_be_removed() -> None:
    class FailingRuntime(FakeRuntime):
        async def run_command(self, command, **kwargs):
            self.events.append(('run_command', command))
            return _command_result(stderr='Failed to clear verifier workspace', return_code=1)

    with pytest.raises(RuntimeError, match='Clear verifier artifacts before agent execution failed'):
        await hide_verifier_from_agent(FailingRuntime([]))


@pytest.mark.asyncio
async def test_answer_capture_injects_default_test_shell_and_allows_empty_answer() -> None:
    events = []
    runtime = FakeRuntime(events, captured_answer='')

    answer = await capture_answer_file(runtime=runtime, answer_path='/workspace/answer.txt', test_timeout=45)

    assert answer == ''
    upload_event = next(event for event in events if event[0] == 'upload_file')
    uploaded_file = upload_event[1][0]
    assert uploaded_file.path == '/tests/test.sh'
    assert '/logs/verifier/answer.txt' in uploaded_file.content
    run_event = next(event for event in events if event[0] == 'run_command' and 'bash /tests/test.sh' in event[1])
    assert run_event[1] == 'bash /tests/test.sh /workspace/answer.txt'
