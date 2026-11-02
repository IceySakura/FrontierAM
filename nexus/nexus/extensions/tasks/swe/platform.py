"""Write the platform status fields into native run-single JSON."""

from typing import Any

from nexus.extensions.agents.swe_agent.platform_status import (
    CONTRACT_ID,
    EXIT_CODES,
    PlatformExitStatus,
    exception_details,
    sandbox_failure_status,
)
from nexus.runner.hooks.base import RunHook, run_hook_registry
from nexus.runner.single import ErrorInfo


STATE_KEY = '_superswe_platform'


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    return dict(value) if isinstance(value, dict) else {}


@run_hook_registry.register('superswe_platform')
class SuperSWEPlatformHook(RunHook):
    """Keep execution status separate from whether grading produced a valid result."""

    def __init__(self) -> None:
        self.result: dict[str, Any] = {}

    async def on_initialization_error(self, run_input, error, output) -> None:
        failure = exception_details(error)
        sandbox = failure.get('sandbox')
        diagnostic = f'{type(error).__name__}: {error}'
        direct_grading = (
            run_input.task_config.get('patch_text') is not None or run_input.task_config.get('run_oracle') is True
        )
        needs_agent = bool(run_input.agent_name) and not direct_grading
        if needs_agent:
            status = (
                PlatformExitStatus.RUNTIME_ERROR_BY_ENV if sandbox is not None else PlatformExitStatus.ENVIRONMENT_ERROR
            )
            output.agent_output = {
                'exit_status': status.value,
                'exit_code': EXIT_CODES[status],
                'error': diagnostic,
                'iterations': 0,
                'agent_started': False,
                'failure_stage': 'setup',
                'failure_kind': 'infrastructure',
                'sandbox': sandbox,
            }
        else:
            output.judge_output = {
                'exit_status': 'judge_error',
                'exit_code': -1,
                'error': diagnostic,
                'tests_executed': False,
            }
        patch = run_input.task_config.get('patch_text')
        output.task_output = {
            'status': 'agent_error' if needs_agent else 'judge_error',
            'score': 0.0,
            'completed': False,
            'resolved': False,
            'patch_text': patch if isinstance(patch, str) else '',
            'error': diagnostic,
            'extra_fields': {},
        }
        self.result = {
            'contract': CONTRACT_ID,
            'stage': 'setup',
            'agent_started': False,
            'judge_started': False,
            'grading_completed': False,
            'terminal_agent_failure': None,
            'native_task_error': None,
            'failure': failure,
        }

    async def on_context_ready(self, ctx) -> None:
        ctx.tool_state[STATE_KEY] = {
            'stage': 'input',
            'agent_started': False,
            'judge_started': False,
            'patch_text': '',
        }

    async def on_run_output(self, ctx, output) -> None:
        state = ctx.tool_state[STATE_KEY]
        native_task = _as_dict(output.task_output)
        agent = _as_dict(output.agent_output)
        judge = _as_dict(output.judge_output)
        root_error = _as_dict(output.error)
        mode = output.extra.get('superswe', {}).get('mode')
        task_config = _as_dict(getattr(ctx, 'task_config', {}))
        direct_grading = task_config.get('patch_text') is not None or task_config.get('run_oracle') is True
        task = dict(native_task)
        failure = _as_dict(state.get('failure'))
        root_exception = getattr(output.error, 'exception', None)
        if isinstance(root_exception, BaseException):
            root_failure = exception_details(root_exception)
            if not failure:
                failure = root_failure
            elif 'sandbox' not in failure and 'sandbox' in root_failure:
                failure['sandbox'] = root_failure['sandbox']
        terminal_failure = _as_dict(state.get('terminal_agent_failure'))
        error = task.get('error') or root_error.get('message') or failure.get('message')
        needs_agent = mode == 'model' or bool(getattr(ctx, 'agent_name', None)) or state.get('agent_started')
        if not agent and needs_agent and not direct_grading:
            stage = state.get('stage', 'input')
            sandbox = failure.get('sandbox')
            if isinstance(sandbox, dict) and not stage.startswith('judge.'):
                status = sandbox_failure_status(
                    sandbox, during_tool=stage == 'agent' and bool(state.get('agent_started'))
                )
            elif stage == 'runtime.create':
                status = PlatformExitStatus.RUNTIME_ERROR_BY_ENV
            elif stage in ('environment.initialize', 'environment.cleanup'):
                status = PlatformExitStatus.ENVIRONMENT_ERROR
            else:
                status = PlatformExitStatus.UNKNOWN_ERROR
            agent = {
                'exit_status': status.value,
                'exit_code': EXIT_CODES[status],
                'error': terminal_failure.get('error')
                or failure.get('message')
                or error
                or 'Task ended before an agent result was available',
                'iterations': 0,
                'agent_started': bool(state.get('agent_started')),
                'failure_kind': 'agent' if status == PlatformExitStatus.RUNTIME_ERROR_BY_AGENT else 'infrastructure',
                'sandbox': sandbox,
            }
            output.agent_output = agent
        if agent:
            try:
                status = PlatformExitStatus(agent['exit_status'])
                if type(agent.get('exit_code')) is not int or agent['exit_code'] != EXIT_CODES[status]:
                    raise ValueError('Agent exit_code does not match its platform exit_status')
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('SuperSWE requires platform status reporting in the configured agent') from exc
        completed = task.get('completed') is True
        if (
            not judge
            and terminal_failure
            and not state.get('judge_started')
            and agent.get('exit_status') == PlatformExitStatus.RUNTIME_ERROR_BY_AGENT
            and isinstance(terminal_failure.get('sandbox'), dict)
            and sandbox_failure_status(terminal_failure['sandbox'], during_tool=True)
            == PlatformExitStatus.RUNTIME_ERROR_BY_AGENT
        ):
            # Record the documented terminal agent failure as a zero-score verdict without reporting test completion.
            judge = {
                'exit_status': 'finished',
                'exit_code': 0,
                'error': None,
                'scoring_method': 'agent_runtime_failure',
                'tests_executed': False,
            }
            output.judge_output = judge
        elif not judge and (
            state.get('judge_started') or completed or direct_grading or bool(agent and agent.get('exit_code') == 0)
        ):
            judge_finished = completed and not error
            judge = {
                'exit_status': 'finished' if judge_finished else 'judge_error',
                'exit_code': 0 if judge_finished else -1,
                'error': None if judge_finished else error or 'Judge did not return a completed result',
            }
            output.judge_output = judge
        if judge:
            expected_code = {'finished': 0, 'judge_error': -1}.get(judge.get('exit_status'))
            if type(judge.get('exit_code')) is not int or judge['exit_code'] != expected_code:
                raise ValueError('Judge exit_code does not match its platform exit_status')
        # Agent failure has priority even when grading also failed. Scores and
        # saved patches remain independent from this execution-validity decision.
        if agent and agent.get('exit_code') != 0:
            task_status = 'agent_error'
            error = agent.get('error')
        elif judge and judge['exit_code'] == 0:
            task_status = 'success'
            # A valid zero-code constraint termination can still carry a real diagnostic.
            error = agent.get('error') or error
        else:
            task_status = 'judge_error'
            error = judge.get('error') or error
        if not task:
            task = {
                'score': 0.0,
                'completed': False,
                'resolved': False,
                'patch_text': state.get('patch_text', ''),
                'extra_fields': {},
            }
        task['status'] = task_status
        task['error'] = error
        output.task_output = task
        if task_status != 'success':
            # A returned error result must also be retryable by the native batch
            # runner, which uses RunOutput.success rather than the platform rule.
            output.success = False
            if output.error is None:
                output.error = ErrorInfo(
                    type='AgentError' if task_status == 'agent_error' else 'JudgeError',
                    message=str(error or f'SuperSWE ended with {task_status}'),
                )
        self.result = {
            'contract': CONTRACT_ID,
            'stage': state.get('stage'),
            'agent_started': bool(state.get('agent_started')),
            'judge_started': bool(state.get('judge_started')),
            'grading_completed': completed,
            'terminal_agent_failure': terminal_failure or None,
            'native_task_error': native_task.get('error'),
            'failure': failure or None,
        }

    def get_result(self) -> dict[str, Any]:
        return self.result
