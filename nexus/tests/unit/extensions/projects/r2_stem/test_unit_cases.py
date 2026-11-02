from types import SimpleNamespace

import pytest

import nexus.extensions.projects.r2_stem.evaluator.unit_cases as verifier_module

from nexus.extensions.projects.r2_stem.evaluator.unit_cases import unit_cases_evaluate
from tests.unit.extensions.projects.r2_stem.verifier_outputs import collected_stdout


def _command_result(*, stdout: str = '', stderr: str = '', return_code: int = 0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, return_code=return_code)


class FakeRuntime:
    def __init__(self, events):
        self.events = events

    async def run_command(self, command, **kwargs):
        self.events.append(('run_command', command))
        if 'cd /logs/verifier' in command:
            return _command_result(stdout=collected_stdout(reward='1.0\n', ctrf={'results': []}))
        return _command_result()

    async def invoke(self, target, payload, **kwargs):
        self.events.append(('invoke', target, payload))
        return _command_result()


@pytest.mark.asyncio
async def test_unit_cases_downloads_task_tests_after_workspace_reset(monkeypatch) -> None:
    events = []
    runtime = FakeRuntime(events)

    class FakeTaskFiles:
        async def exists(self, path):
            events.append(('task_files_exists', path))
            return True

        async def download(self, pattern, target, *, runtime):
            events.append(('task_files_download', pattern, target))

    class FakeContext:
        task_files = FakeTaskFiles()

        async def get_runtime(self, name):
            events.append(('get_runtime', name))
            return runtime

        def get_runtime_id(self, target):
            return None

    monkeypatch.setattr(verifier_module, 'require_context', lambda: FakeContext())

    monkeypatch.setenv('VERIFIER_LLM_API_KEY', 'secret-value')
    reward, ctrf = await unit_cases_evaluate(
        runtime_name='main',
        test_timeout=99,
        verifier_env={'STATIC_VALUE': 'configured'},
        verifier_env_refs={'LLM_API_KEY': 'VERIFIER_LLM_API_KEY'},
    )

    assert reward == 1.0
    assert ctrf['results'] == []
    assert ctrf['nexus_verifier']['return_code'] == 0
    assert ctrf['nexus_verifier']['log_tail'] == ''
    event_names = [event[0] for event in events]
    # The reward, CTRF, wrapper result and log tail all come back in the single trailing command.
    assert event_names == [
        'get_runtime',
        'run_command',
        'run_command',
        'task_files_exists',
        'task_files_download',
        'run_command',
        'invoke',
        'run_command',
    ]
    reset_index = next(
        i
        for i, event in enumerate(events)
        if event[0] == 'run_command' and 'find /tests -mindepth 1 -delete' in event[1]
    )
    download_index = next(i for i, event in enumerate(events) if event[0] == 'task_files_download')
    assert reset_index < download_index
    invoke_event = next(event for event in events if event[0] == 'invoke')
    assert invoke_event[2]['extra_env'] == {
        'LLM_API_KEY': 'secret-value',
        'STATIC_VALUE': 'configured',
    }
