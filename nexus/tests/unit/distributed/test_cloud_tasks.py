import importlib

from types import SimpleNamespace


async def test_sleep_echo_task_returns_the_input_payload():
    tasks = importlib.import_module('tests.integration.distributed.cloud_tasks')

    result = await tasks.sleep_echo_task(tasks.SleepEchoInput(index=7, message='ready', sleep_seconds=0))

    assert result.model_dump() == {'index': 7, 'message': 'ready'}


async def test_sandbox_agent_echo_requires_agent_shell_marker_and_holds_ready_runtime(monkeypatch):
    tasks = importlib.import_module('tests.integration.distributed.cloud_tasks')
    state = {'marker': None}
    calls = []

    class FakeRuntime:
        async def run_command(self, command, **kwargs):
            calls.append(command)
            if command.startswith('rm -f '):
                state['marker'] = None
                return SimpleNamespace(return_code=0, stdout='', stderr='')
            return SimpleNamespace(
                return_code=0 if state['marker'] else 1,
                stdout=f'{state["marker"]}\n' if state['marker'] else '',
                stderr='' if state['marker'] else 'missing',
            )

    class FakeProvider:
        def get_runtime_id(self, name):
            assert name == 'main'
            return 'runtime-7'

    class FakeContext:
        runtime_provider = FakeProvider()

        async def get_runtime(self, name):
            assert name == 'main'
            return FakeRuntime()

    async def run_agent(**kwargs):
        prompt = kwargs['user_prompt']
        assert 'swe_agent/bash' in prompt
        assert 'sandbox-agent-smoke-000000000007' in prompt
        state['marker'] = 'sandbox-agent-smoke-000000000007'
        return SimpleNamespace(exit_status='finished')

    times = iter([1000.0, 1360.25])
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(tasks, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(tasks, '_wall_time', lambda: next(times))
    monkeypatch.setattr(tasks.asyncio, 'sleep', fake_sleep)

    output = await tasks.sandbox_agent_echo_task(
        tasks.SandboxAgentEchoInput(
            index=7,
            marker='sandbox-agent-smoke-000000000007',
            hold_seconds=360,
            run_agent=run_agent,
        )
    )

    assert sleeps == [360]
    assert len(calls) == 2
    assert output.model_dump() == {
        'agent_exit_status': 'finished',
        'command_return_code': 0,
        'command_stdout': 'sandbox-agent-smoke-000000000007',
        'index': 7,
        'marker': 'sandbox-agent-smoke-000000000007',
        'runtime_id': 'runtime-7',
        'sandbox_hold_finished_at': 1360.25,
        'sandbox_ready_at': 1000.0,
    }
