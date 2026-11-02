import pytest

from nexus.core import CONTEXT_LENGTH_EXCEEDED_CODE
from nexus.extensions.agents.claude_code import agent as claude_agent
from nexus.extensions.agents.claude_code.agent import ClaudeCodeInput, ClaudeCodeNativeBridgeInput
from nexus.extensions.harbor_bridge.agent_result import agent_termination_reason, build_t3_agent_result
from nexus.runtime.client.base import RuntimeUnhealthyError


def _result(**overrides):
    return {'success': False, 'output': '', 'stderr': 'failed', 'exit_code': -1, 'execution_time': 1, **overrides}


@pytest.fixture
def recorded(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(claude_agent, 'get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr(
        claude_agent, 'update_span', lambda span_id, attributes=None: calls.append((span_id, attributes))
    )
    return calls


async def _run(monkeypatch, result=None, error=None):
    async def fake_call_runtime_agent(_target, /, **_kwargs):
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(claude_agent, 'call_runtime_agent', fake_call_runtime_agent)
    return await claude_agent.run_claude_code(ClaudeCodeInput(user_prompt='test', working_dir='/workspace'))


class TestClaudeCodeTerminationReason:
    @pytest.mark.parametrize('stage', ['install', 'proxy', 'pre_start', 'execution_transport'])
    async def test_setup_failure_is_initialization_error(self, monkeypatch, recorded, stage):
        output = await _run(monkeypatch, _result(failure_kind='infrastructure', failure_stage=stage))

        assert output.failure_stage == stage
        assert recorded == [('span-1', {'hytuner.termination_reason': 'INITIALIZATION_ERROR'})]

    async def test_unclassified_runtime_failure_is_initialization_error(self, monkeypatch, recorded):
        output = await _run(monkeypatch, _result())

        assert output.failure_stage == 'runtime_protocol'
        assert recorded == [('span-1', {'hytuner.termination_reason': 'INITIALIZATION_ERROR'})]

    async def test_other_infrastructure_failure_is_unhandled_error(self, monkeypatch, recorded):
        await _run(monkeypatch, _result(failure_kind='infrastructure', failure_stage='sandbox_exec'))

        assert recorded == [('span-1', {'hytuner.termination_reason': 'UNHANDLED_ERROR'})]

    async def test_unhealthy_runtime_is_env_error(self, monkeypatch, recorded):
        output = await _run(monkeypatch, error=RuntimeUnhealthyError('sandbox stopped'))

        assert output.success is False
        assert output.failure_kind == 'infrastructure'
        assert output.failure_stage == 'runtime_unhealthy'
        assert 'sandbox stopped' in output.stderr
        assert recorded == [('span-1', {'hytuner.termination_reason': 'ENV_ERROR'})]

    @pytest.mark.parametrize(
        'result',
        [
            _result(success=True, exit_code=0),
            _result(exit_code=1, failure_kind='agent', failure_stage='process'),
        ],
    )
    async def test_agent_outcomes_leave_termination_unset(self, monkeypatch, recorded, result):
        output = await _run(monkeypatch, result)

        assert output.exit_status is None
        assert recorded == []


def _refused(**overrides):
    return _result(**{
        'exit_code': 1,
        'failure_kind': 'agent',
        'failure_stage': 'process',
        'last_llm_error_code': CONTEXT_LENGTH_EXCEEDED_CODE,
        **overrides,
    })


class TestClaudeCodeContextLengthTruncation:
    async def test_last_call_refused_on_budget_is_truncation(self, monkeypatch, recorded):
        output = await _run(monkeypatch, _refused())

        assert output.exit_status == 'truncation'
        assert recorded == [('span-1', {'hytuner.termination_reason': 'TRUNCATION'})]

    async def test_harbor_maps_refusal_to_truncation(self, monkeypatch, recorded):
        output = await _run(monkeypatch, _refused())

        result = build_t3_agent_result(mode='agent', agent_output=output)

        assert result.status == 'incomplete'
        assert agent_termination_reason(result) == 'TRUNCATION'

    @pytest.mark.parametrize(
        'result',
        [
            _refused(success=True, exit_code=0),
            _refused(last_llm_error_code=None, llm_request_outcomes={'r1': CONTEXT_LENGTH_EXCEEDED_CODE, 'r2': None}),
            _refused(failure_kind='infrastructure', failure_stage='sandbox_exec'),
            _refused(exit_code=-1),
        ],
        ids=['clean-exit', 'earlier-refusal-only', 'infrastructure', 'cli-never-exited'],
    )
    async def test_other_runs_are_not_relabelled(self, monkeypatch, recorded, result):
        output = await _run(monkeypatch, result)

        assert output.exit_status is None
        assert ('span-1', {'hytuner.termination_reason': 'TRUNCATION'}) not in recorded

    async def test_native_bridge_refusal_is_truncation(self, monkeypatch, recorded):
        async def fake_call_native_bridge_runtime_agent(_target, /, **_kwargs):
            return _result(exit_code=1, last_llm_error_code=CONTEXT_LENGTH_EXCEEDED_CODE)

        monkeypatch.setattr(claude_agent, 'call_native_bridge_runtime_agent', fake_call_native_bridge_runtime_agent)
        output = await claude_agent.run_claude_code_native_bridge(
            ClaudeCodeNativeBridgeInput(user_prompt='test', working_dir='/workspace')
        )

        assert output.exit_status == 'truncation'
        assert recorded == [('span-1', {'hytuner.termination_reason': 'TRUNCATION'})]

    async def test_native_bridge_setup_failure_is_not_relabelled(self, monkeypatch, recorded):
        async def fake_call_native_bridge_runtime_agent(_target, /, **_kwargs):
            return _result(last_llm_error_code=CONTEXT_LENGTH_EXCEEDED_CODE)

        monkeypatch.setattr(claude_agent, 'call_native_bridge_runtime_agent', fake_call_native_bridge_runtime_agent)
        output = await claude_agent.run_claude_code_native_bridge(
            ClaudeCodeNativeBridgeInput(user_prompt='test', working_dir='/workspace')
        )

        assert output.exit_status is None
        assert recorded == []
