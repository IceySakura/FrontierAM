import os

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, task_registry
from nexus.extensions.agents.main.agent import MainAgentOutput, PostIterationHookContext, PostIterationHookResult
from nexus.runner.batch import BatchItem
from nexus.runner.single import ProcessorConfig, RunInput
from tests.e2e._llm_account import load_sqlite_model_account
from tests.e2e.test_main_agent_text_thinking import (
    _ACCOUNT_DB_ENV,
    _ACCOUNT_RUN_ID_ENV,
    _CASES,
    _PROTOCOL_FINAL,
    _PROTOCOL_TOOL,
    _make_client,
)


_CASE_ENV = 'NEXUS_MAIN_AGENT_BATCH_E2E_CASE'
_TASK_NAME = 'e2e_main_agent_batch_persistence'
_CLIENT_NAMES = {
    'vertex_anthropic': 'vertex',
    'vertex_openai': 'vertex',
    'openai_responses': 'openai_responses',
    'litellm_openai': 'litellm',
    'litellm_anthropic': 'litellm',
}


def _required_tool_choice(case_name: str) -> str | dict[str, object]:
    if case_name == 'litellm_openai':
        return {'type': 'function', 'function': {'name': 'protocol_step'}}
    return 'required'


class _BatchPersistenceInput(TaskRunInput, WithAgentRunner):
    inference_time_timeout: float = 1200
    first_step_instruction: str = 'On round 1, call protocol_step exactly once with step=1 and checksum=49.'


class _BatchPersistenceOutput(TaskRunOutput):
    final_answer: str
    iterations: int


async def _force_four_round_tool_choice(
    _ctx: object,
    hook_context: PostIterationHookContext,
) -> PostIterationHookResult:
    iteration = hook_context.iteration
    tool_choice = _required_tool_choice(os.environ[_CASE_ENV]) if iteration < 3 else 'none'
    return PostIterationHookResult(next_completion_kwargs={'tool_choice': tool_choice})


@task_registry.register(_TASK_NAME)
async def _batch_persistence_task(task_input: _BatchPersistenceInput) -> _BatchPersistenceOutput:
    output = await task_input.run_agent(
        system_prompt=(
            'You are validating a strict four-round Main Agent protocol. '
            f'{task_input.first_step_instruction} '
            'After each successful tool result, call it exactly once with the next '
            f'integer. After step 3, call no tool and reply exactly {_PROTOCOL_FINAL}. '
            'Never skip, repeat, combine, or parallelize steps.'
        ),
        user_prompt='Begin the strict four-round protocol now.',
        tools=[_PROTOCOL_TOOL],
        max_iterations=4,
        post_iteration_hook='tests.e2e._main_agent_batch_fixture::_force_four_round_tool_choice',
        single_tool_timeout=30,
        inference_time_timeout=task_input.inference_time_timeout,
        observation_time_timeout=120,
    )
    if not isinstance(output, MainAgentOutput):
        raise TypeError(f'Expected MainAgentOutput, got {type(output).__name__}')
    if output.exit_status != 'finished' or output.final_answer is None:
        raise RuntimeError(f'Main Agent batch probe did not finish: {output.exit_status}')
    return _BatchPersistenceOutput(final_answer=output.final_answer, iterations=output.iterations)


def batch_item_from_environment() -> BatchItem:
    account = load_sqlite_model_account(
        os.environ[_ACCOUNT_DB_ENV],
        run_id=os.environ.get(_ACCOUNT_RUN_ID_ENV),
    )
    case_name = os.environ[_CASE_ENV]
    case = next(candidate for candidate in _CASES if candidate.name == case_name)
    client = _make_client(case, account)
    client.config.request_args['tool_choice'] = _required_tool_choice(case.name)
    if case.name == 'vertex_openai':
        # This gateway's Chat Completions route rejects reasoning_effort with function tools.
        # The separate text-thinking E2E covers reasoning; keep this batch probe on ordinary tool replay.
        client.config.request_args['reasoning_effort'] = 'none'
    elif case.protocol == 'anthropic':
        # Anthropic rejects forced tool choice together with thinking. The separate real Main Agent matrix covers
        # signed thinking + tool replay; this batch probe isolates deterministic Task -> SQLite persistence.
        client.config.request_args.pop('thinking', None)
        client.config.request_args.pop('reasoning_effort', None)
    config = client.config
    first_step_instruction = _BatchPersistenceInput.model_fields['first_step_instruction'].default
    return BatchItem(
        run_id=f'main-agent-{case.name}',
        metadata={'client_case': case.name, 'credential_source': 'sqlite'},
        config=RunInput(
            task_name=_TASK_NAME,
            task_config={
                'inference_time_timeout': float(os.environ.get('NEXUS_LLM_E2E_TIMEOUT', '300')) * 4,
                'first_step_instruction': first_step_instruction,
            },
            agent_name='main_agent',
            openai_clients={
                'main': {
                    'name': _CLIENT_NAMES[case.name],
                    'config': {
                        'client_args': config.client_args,
                        'request_args': config.request_args,
                    },
                }
            },
            processors={
                'chat': ProcessorConfig(name='chat_tracker', args={'tools_mode': 'branch_last'}),
            },
        ),
    )


__all__ = ['_CASE_ENV', '_CLIENT_NAMES', 'batch_item_from_environment']
