"""Host LLMRequestHandler outcomes decide inference/context/output-length statuses."""

import pytest

from nexus.extensions.tasks.os_world_v2.outputs import (
    AgentExitStatus,
    OSWorldAgentOutput,
    apply_host_llm_evidence,
)


def output(status, code):
    return OSWorldAgentOutput(exit_status=status, exit_code=code, error=None if code == 0 else 'cli failed')


@pytest.mark.parametrize(
    ('evidence', 'expected'),
    [
        ({'last_llm_error_code': 'context_length_exceeded'}, AgentExitStatus.CONTEXT_LENGTH_EXCEEDED),
        ({'last_llm_error_code': 'output_length_exceeded'}, AgentExitStatus.OUTPUT_LENGTH_EXCEEDED),
        ({'last_llm_error_code': None, 'last_llm_finish_reason': 'length'}, AgentExitStatus.OUTPUT_LENGTH_EXCEEDED),
        ({'last_llm_error_code': 'upstream_http_502'}, AgentExitStatus.INFERENCE_ERROR),
    ],
)
def test_last_request_outcome_refines_unclassified_failure(evidence, expected):
    result = apply_host_llm_evidence(output(AgentExitStatus.UNKNOWN_ERROR, -1), {'output': 'x', **evidence})
    assert result.exit_status == expected
    assert result.exit_code == (-1 if expected == AgentExitStatus.INFERENCE_ERROR else 0)
    assert result.error


@pytest.mark.parametrize(
    'status',
    [AgentExitStatus.AGENT_TIMEOUT, AgentExitStatus.RUNTIME_ERROR_BY_AGENT, AgentExitStatus.FINISHED],
)
def test_deliberate_termination_outranks_last_request(status):
    code = 0
    result = apply_host_llm_evidence(output(status, code), {'last_llm_error_code': 'upstream_http_502'})
    assert result.exit_status == status
    assert result.model_dump()['last_llm_error_code'] == 'upstream_http_502'


def test_no_host_evidence_keeps_result():
    original = output(AgentExitStatus.UNKNOWN_ERROR, -1)
    assert apply_host_llm_evidence(original, {'output': 'x'}) is original
