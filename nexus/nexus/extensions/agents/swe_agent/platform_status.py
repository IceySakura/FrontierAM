"""AgentBench statuses and structured sandbox attribution.

Status contract: https://doc.weixin.qq.com/doc/w3_AQgA3wbZAIMCNNeIZlvP7SGiV0J0G
Sandbox contract: https://iwiki.woa.com/p/4041018198
"""

from enum import Enum

from nexus.runtime.client.base import AGENT_PROCESS_DEATH_REASONS, RuntimeUnhealthyError
from nexus.runtime.providers.gongfeng import SandboxApiError


CONTRACT_ID = 'wecom:w3_AQgA3wbZAIMCNNeIZlvP7SGiV0J0G'
CONTRACT_SHA256 = 'aa09bf3236aa8b2a1a8539ab8a21df10934322b60d7d01ba21fa09ce9666b152'


class PlatformExitStatus(str, Enum):
    FINISHED = 'finished'
    AGENT_TIMEOUT = 'agent_timeout'
    MAX_ITERATIONS = 'max_iterations'
    INFERENCE_ERROR = 'inference_error'
    CONTEXT_LENGTH_EXCEEDED = 'context_length_exceeded'
    OUTPUT_LENGTH_EXCEEDED = 'output_length_exceeded'
    RUNTIME_ERROR_BY_AGENT = 'runtime_error_by_agent'
    RUNTIME_ERROR_BY_ENV = 'runtime_error_by_env'
    ENVIRONMENT_ERROR = 'environment_error'
    UNKNOWN_ERROR = 'unknown_error'


EXIT_CODES = {
    PlatformExitStatus.FINISHED: 0,
    PlatformExitStatus.AGENT_TIMEOUT: 0,
    PlatformExitStatus.MAX_ITERATIONS: 0,
    PlatformExitStatus.INFERENCE_ERROR: -1,
    PlatformExitStatus.CONTEXT_LENGTH_EXCEEDED: 0,
    PlatformExitStatus.OUTPUT_LENGTH_EXCEEDED: 0,
    PlatformExitStatus.RUNTIME_ERROR_BY_AGENT: 0,
    PlatformExitStatus.RUNTIME_ERROR_BY_ENV: -1,
    PlatformExitStatus.ENVIRONMENT_ERROR: -1,
    PlatformExitStatus.UNKNOWN_ERROR: -1,
}


def exception_details(error: BaseException) -> dict:
    """Keep structured sandbox evidence; diagnostic text is not an attribution signal."""
    details = {'type': type(error).__name__, 'message': str(error)}
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (RuntimeUnhealthyError, SandboxApiError)):
            sandbox = {
                'code': current.code,
                'reason': current.reason,
                'exit_code': current.exit_code,
                'trace_id': current.trace_id,
            }
            # Keep a typed watchdog failure even when the proxy supplied no reason.
            # A wrapped SDK error may still provide the missing structured evidence.
            if 'sandbox' not in details or current.code is not None or current.reason:
                details['sandbox'] = sandbox
            if current.code is not None or current.reason:
                break
        current = current.__cause__ or current.__context__
    return details


def sandbox_failure_status(evidence: dict, *, during_tool: bool = False) -> PlatformExitStatus:
    """Attribute a confirmed sandbox failure using the contract's code/reason pair."""
    code = evidence.get('code')
    reason = evidence.get('reason')
    if (
        during_tool
        and type(code) is int
        and code == 260422
        and isinstance(reason, str)
        and reason in AGENT_PROCESS_DEATH_REASONS
    ):
        return PlatformExitStatus.RUNTIME_ERROR_BY_AGENT
    # Startup failures, Queue.* (including 260422), Process.Killed (260500),
    # missing reasons and unregistered/mismatched pairs must not be scored as agent failures.
    return PlatformExitStatus.RUNTIME_ERROR_BY_ENV


def runtime_failure_status(error: BaseException, *, during_tool: bool = False) -> PlatformExitStatus:
    evidence = exception_details(error).get('sandbox')
    if evidence is None:
        return PlatformExitStatus.UNKNOWN_ERROR
    return sandbox_failure_status(evidence, during_tool=during_tool)
