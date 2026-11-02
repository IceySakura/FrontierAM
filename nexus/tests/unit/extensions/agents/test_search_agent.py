from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core import ToolExecutor
from nexus.extensions.agents.base.errors import NonRetryableProviderError
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.agents.search_agent import agent as search_agent_module
from nexus.extensions.agents.search_agent.agent import (
    OBSERVATION_FOLDED_PLACEHOLDER,
    SearchAgent,
    SearchAgentInput,
    _resolve_tools,
    contains_toolcall_markup,
    get_llm_retry_reason,
    run_search_agent,
)


from nexus.core import ToolContext
from nexus.extensions.agents.search_agent.tools import ExecutePythonInput, execute_python_tool
from nexus.extensions.tools.code_interpreter import jupyter_code_interpreter as jupyter_module


TEST_QUESTION = 'test question'
USER_QUESTION = 'question'
TRAIN_QUESTION = 'train question'
RESEARCH_PROMPT = 'research something'
FINAL_ANSWER = 'final answer'
BEST_EFFORT_ANSWER = 'best effort answer'
RESCUED_ANSWER = 'rescued answer'
OLD_USER_MESSAGE = 'old user message'
OLD_ASSISTANT_MESSAGE = 'old assistant message'
GENERATED_PROMPT = 'generated prompt'


def _make_agent() -> SearchAgent:
    agent_input = SearchAgentInput(
        user_prompt=TEST_QUESTION,
        is_training=False,
        max_llm_retries=3,
        finish_tool_names=['finish'],
    )
    executor = ToolExecutor(_resolve_tools('v2'))
    return SearchAgent.create(agent_input, executor=executor)


def _schema_names(agent: SearchAgent) -> list[str]:
    return [schema['function']['name'] for schema in agent.executor.get_tool_schemas()]


@pytest.mark.asyncio
async def test_force_finish_once_cleans_temporary_message_after_truncation() -> None:
    agent = _make_agent()
    agent.add_message('user', OLD_USER_MESSAGE)
    agent.add_message('assistant', OLD_ASSISTANT_MESSAGE)
    original_retries = agent.max_llm_retries

    async def fake_step() -> None:
        assert _schema_names(agent) == ['finish']
        assert agent.max_llm_retries == 0
        agent.add_message('assistant', FINAL_ANSWER)
        agent.exit_status = TerminationReason.SUCCESS

    agent.step = AsyncMock(side_effect=fake_step)

    finalized = await agent.force_finish_once()

    assert finalized is True
    assert 'finish' in _schema_names(agent)
    assert len(_schema_names(agent)) > 1
    assert agent.max_llm_retries == original_retries
    assert [msg['content'] for msg in agent.messages] == [OLD_USER_MESSAGE, OLD_ASSISTANT_MESSAGE, FINAL_ANSWER]


@pytest.mark.asyncio
async def test_force_finish_once_keeps_existing_identical_prompt_message() -> None:
    agent = _make_agent()
    agent.add_message('user', search_agent_module._FORCE_FINISH_WITH_TOOL)
    original_message = agent.messages[-1]

    async def fake_step() -> None:
        agent.exit_status = TerminationReason.SUCCESS

    agent.step = AsyncMock(side_effect=fake_step)

    finalized = await agent.force_finish_once()

    assert finalized is True
    assert len(agent.messages) == 1
    assert agent.messages[0] is original_message
    assert agent.messages[0]['content'] == search_agent_module._FORCE_FINISH_WITH_TOOL


@pytest.mark.asyncio
async def test_force_finish_once_without_finish_tool_hides_all_tools_temporarily() -> None:
    agent_input = SearchAgentInput(
        user_prompt=TEST_QUESTION,
        is_training=False,
        max_llm_retries=2,
        finish_tool_names=[],
    )
    executor = ToolExecutor(_resolve_tools('v2', include_finish=False))
    agent = SearchAgent.create(agent_input, executor=executor)
    original_retries = agent.max_llm_retries

    async def fake_step() -> None:
        assert _schema_names(agent) == []
        assert agent.max_llm_retries == 0
        agent.add_message('assistant', BEST_EFFORT_ANSWER)
        agent.exit_status = TerminationReason.SUCCESS

    agent.step = AsyncMock(side_effect=fake_step)

    finalized = await agent.force_finish_once()

    assert finalized is True
    assert _schema_names(agent)
    assert agent.max_llm_retries == original_retries
    assert [msg['content'] for msg in agent.messages] == [BEST_EFFORT_ANSWER]


@pytest.mark.asyncio
async def test_inference_ask_llm_retries_bad_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent()
    bad_message = SimpleNamespace(content='<tool_call>', reasoning_content=None, tool_calls=[])
    good_message = SimpleNamespace(content='clean answer', reasoning_content=None, tool_calls=[])
    ask_mock = AsyncMock(side_effect=[bad_message, good_message])
    monkeypatch.setattr(agent, '_ask_llm_with_usage', ask_mock)

    result = await agent.ask_llm()

    assert result is good_message
    assert ask_mock.await_count == 2


@pytest.mark.asyncio
async def test_ask_llm_sets_provider_error_for_non_retryable_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent()
    monkeypatch.setattr(
        'nexus.extensions.agents.base.toolcall.require_context',
        lambda: SimpleNamespace(openai_client=SimpleNamespace(create_completion=AsyncMock(side_effect=NonRetryableProviderError('no_budget_error')))),
    )

    await agent._ask_llm_with_usage()

    assert agent.exit_status == TerminationReason.PROVIDER_ERROR
    assert agent.tool_error_count == 1


@pytest.mark.asyncio
async def test_ask_llm_sets_provider_error_for_marker_in_generic_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent()
    monkeypatch.setattr(
        'nexus.extensions.agents.base.toolcall.require_context',
        lambda: SimpleNamespace(
            openai_client=SimpleNamespace(create_completion=AsyncMock(side_effect=RuntimeError('no_budget_error')))
        ),
    )

    await agent._ask_llm_with_usage()

    assert agent.exit_status == TerminationReason.PROVIDER_ERROR
    assert agent.tool_error_count == 1


def test_contains_toolcall_markup_and_retry_reason_cover_reasoning_only_cases() -> None:
    assert contains_toolcall_markup('<tool_call><function=search></function></tool_call>') is True
    assert contains_toolcall_markup('plain answer only') is False

    reasoning_only_message = SimpleNamespace(content='', reasoning_content='Need more thought', tool_calls=[])
    markup_message = SimpleNamespace(content='clean', reasoning_content='<tool_call>', tool_calls=[])
    clean_message = SimpleNamespace(content=FINAL_ANSWER, reasoning_content='hidden', tool_calls=[{'id': '1'}])

    assert get_llm_retry_reason(reasoning_only_message) is not None
    assert get_llm_retry_reason(markup_message) is not None
    assert get_llm_retry_reason(clean_message) is None


@pytest.mark.asyncio
async def test_run_context_length_exceeded_attempts_force_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent()

    async def fake_step() -> None:
        agent.exit_status = TerminationReason.TRUNCATION

    async def fake_force_finish_once() -> bool:
        agent.exit_status = TerminationReason.SUCCESS
        agent.add_message('tool', RESCUED_ANSWER, tool_call_id='finish-1', name='finish')
        return True

    agent.step = AsyncMock(side_effect=fake_step)
    monkeypatch.setattr(agent, 'force_finish_once', AsyncMock(side_effect=fake_force_finish_once))

    result = await agent.run(USER_QUESTION)

    assert result == RESCUED_ANSWER
    assert agent.exit_status == TerminationReason.SUCCESS
    agent.force_finish_once.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_inference_max_iterations_sets_max_iterations_when_force_finish_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_input = SearchAgentInput(
        user_prompt=TEST_QUESTION,
        is_training=False,
        max_iterations=1,
        finish_tool_names=['finish'],
    )
    agent = SearchAgent.create(agent_input, executor=ToolExecutor(_resolve_tools('v2')))
    agent.step = AsyncMock(return_value=None)
    monkeypatch.setattr(agent, 'force_finish_once', AsyncMock(return_value=False))

    result = await agent.run(USER_QUESTION)

    assert result is None
    assert agent.exit_status == TerminationReason.MAX_ITERATIONS
    agent.force_finish_once.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_training_max_iterations_does_not_force_finish() -> None:
    agent_input = SearchAgentInput(
        user_prompt=TRAIN_QUESTION,
        is_training=True,
        max_iterations=1,
    )
    agent = SearchAgent.create(agent_input, executor=ToolExecutor(_resolve_tools('v2')))
    agent.step = AsyncMock(return_value=None)
    agent.force_finish_once = AsyncMock(return_value=True)  # type: ignore[method-assign]

    result = await agent.run(USER_QUESTION)

    assert result is None
    assert agent.exit_status == TerminationReason.MAX_ITERATIONS
    agent.force_finish_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_marks_unhandled_error_when_step_crashes() -> None:
    agent = _make_agent()
    agent.step = AsyncMock(side_effect=RuntimeError('boom'))

    result = await agent.run(USER_QUESTION)

    assert result is None
    assert agent.exit_status == TerminationReason.UNHANDLED_ERROR


@pytest.mark.asyncio
async def test_run_rejects_second_invocation_after_exit() -> None:
    agent = _make_agent()
    agent.exit_status = TerminationReason.SUCCESS

    with pytest.raises(RuntimeError):
        await agent.run(USER_QUESTION)


def test_resolve_tools_respects_toolset_and_finish_toggle() -> None:
    v1_tool_names = [tool.nexus_meta['name'] for tool in _resolve_tools('v1')]
    v2_without_finish = [tool.nexus_meta['name'] for tool in _resolve_tools('v2', include_finish=False)]

    assert set(v1_tool_names) == {'google_serper_search', 'base/finish', 'visit'}
    assert 'base/finish' not in v2_without_finish
    assert set(v2_without_finish) == {'google_serper_search', 'visit', 'base/write_todos', 'execute_python'}


def test_resolve_tools_rejects_unknown_toolset_and_missing_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.core.registry import EntryNotFoundError

    with pytest.raises(ValueError):
        _resolve_tools('invalid_version')

    original_get = search_agent_module.tool_registry.get

    def fake_get(name: str):
        if name == 'execute_python':
            raise EntryNotFoundError(f"Entry '{name}' not found in registry")
        return original_get(name)

    monkeypatch.setattr('nexus.extensions.agents.search_agent.agent.tool_registry.get', fake_get)

    with pytest.raises(ValueError):
        _resolve_tools('v2')


@pytest.mark.asyncio
async def test_run_search_agent_builds_default_prompt_for_no_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    agent_input = SearchAgentInput(
        user_prompt=RESEARCH_PROMPT,
        system_prompt=None,
        is_training=False,
        finish_tool_names=[],
        toolset_version='v2',
    )

    prompt_call: dict[str, object] = {}

    def fake_get_default_system_prompt(is_training: bool, toolset_version: str, *, include_finish: bool) -> str:
        prompt_call['is_training'] = is_training
        prompt_call['toolset_version'] = toolset_version
        prompt_call['include_finish'] = include_finish
        return GENERATED_PROMPT

    monkeypatch.setattr(search_agent_module, 'get_default_system_prompt', fake_get_default_system_prompt)
    monkeypatch.setattr('nexus.extensions.agents.search_agent.agent._resolve_tools', lambda *args, **kwargs: [])
    run_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(SearchAgent, 'run', run_mock)

    output = await run_search_agent(agent_input)

    assert prompt_call == {
        'is_training': False,
        'toolset_version': 'v2',
        'include_finish': False,
    }
    assert agent_input.system_prompt == GENERATED_PROMPT
    run_mock.assert_awaited_once_with(RESEARCH_PROMPT)
    assert output.exit_status == TerminationReason.SUCCESS


def _make_agent_with_keep_recent_k(k: int) -> SearchAgent:
    agent_input = SearchAgentInput(
        user_prompt=TEST_QUESTION,
        is_training=False,
        keep_recent_k=k,
        finish_tool_names=['finish'],
    )
    executor = ToolExecutor(_resolve_tools('v2'))
    return SearchAgent.create(agent_input, executor=executor)


def test_build_messages_keep_recent_k_folds_old_tool_observations() -> None:
    """keep_recent_k=3 with 5 rounds: rounds 1-2 have tool observations folded,
    rounds 3-5 are kept intact."""
    agent = _make_agent_with_keep_recent_k(k=3)
    agent.system_prompt = 'You are a helpful search assistant.'

    agent.add_message('user', 'What is the population of Tokyo?')

    agent.add_message('assistant', 'I will search for Tokyo population.')
    agent.add_message(
        'tool',
        'Google result: Tokyo has 13.96 million people as of 2024.',
        tool_call_id='tc-1',
        name='google_serper_search',
    )

    agent.add_message('assistant', 'Let me visit the source page for details.')
    agent.add_message(
        'tool',
        '<html><body>Tokyo Metropolitan Government reports 13,960,000 residents...</body></html>',
        tool_call_id='tc-2',
        name='visit',
    )

    agent.add_message('assistant', 'I need to check if this includes the greater metro area.')
    agent.add_message(
        'tool',
        'Google result: Greater Tokyo Area population is 37.4 million.',
        tool_call_id='tc-3',
        name='google_serper_search',
    )

    agent.add_message('assistant', 'Let me verify this number.')
    agent.add_message(
        'tool',
        'Wikipedia: The Greater Tokyo Area is the most populous metropolitan area in the world.',
        tool_call_id='tc-4',
        name='visit',
    )

    agent.add_message('assistant', 'Now I have enough information.')
    agent.add_message(
        'tool', 'Confirmed: city proper ~14M, metro area ~37.4M.', tool_call_id='tc-5', name='google_serper_search'
    )

    built = agent.build_messages()

    assert built[0] == {'role': 'system', 'content': 'You are a helpful search assistant.'}
    assert built[1]['role'] == 'user'
    assert built[1]['content'] == 'What is the population of Tokyo?'

    assert built[2]['role'] == 'assistant'
    assert built[2]['content'] == 'I will search for Tokyo population.'
    assert built[3]['role'] == 'tool'
    assert built[3]['content'] == OBSERVATION_FOLDED_PLACEHOLDER
    assert built[3]['tool_call_id'] == 'tc-1'
    assert built[3]['name'] == 'google_serper_search'

    assert built[4]['role'] == 'assistant'
    assert built[4]['content'] == 'Let me visit the source page for details.'
    assert built[5]['role'] == 'tool'
    assert built[5]['content'] == OBSERVATION_FOLDED_PLACEHOLDER
    assert built[5]['tool_call_id'] == 'tc-2'
    assert built[5]['name'] == 'visit'

    assert built[6]['role'] == 'assistant'
    assert built[7]['content'] == 'Google result: Greater Tokyo Area population is 37.4 million.'
    assert built[7]['tool_call_id'] == 'tc-3'

    assert built[8]['role'] == 'assistant'
    assert built[9]['content'].startswith('Wikipedia:')
    assert built[9]['tool_call_id'] == 'tc-4'

    assert built[10]['role'] == 'assistant'
    assert built[11]['content'] == 'Confirmed: city proper ~14M, metro area ~37.4M.'
    assert built[11]['tool_call_id'] == 'tc-5'

    assert len(built) == 12

    tool_messages = [m for m in agent.messages if m.get('role') == 'tool']
    assert len(tool_messages) == 5
    assert 'Google result: Tokyo has 13.96 million' in tool_messages[0]['content']
    assert '<html>' in tool_messages[1]['content']
    assert 'Greater Tokyo Area' in tool_messages[2]['content']


@pytest.mark.asyncio
async def test_reset_context_discards_all_except_first_user_message() -> None:
    agent = _make_agent()
    agent.max_context_tokens = 1000

    agent.add_message('user', 'Original question')
    agent.add_message('assistant', 'Step 1 reasoning')
    agent.add_message('tool', 'Tool output 1', tool_call_id='tc-1', name='google_serper_search')
    agent.add_message('assistant', 'Step 2 reasoning')
    agent.add_message('tool', 'Tool output 2', tool_call_id='tc-2', name='visit')

    assert len(agent.messages) == 5

    agent._context_token_count = 1500

    dummy_message = SimpleNamespace(content='fresh answer', reasoning_content=None, tool_calls=[])
    agent._ask_llm_with_usage = AsyncMock(return_value=dummy_message)

    result = await agent.ask_llm()

    assert result is dummy_message
    assert len(agent.messages) == 1
    assert agent.messages[0]['role'] == 'user'
    assert agent.messages[0]['content'] == 'Original question'
    assert agent._context_reset_count == 1
    assert agent._context_token_count == 0


def test_build_messages_keep_recent_k_no_folding_when_rounds_within_k() -> None:
    agent = _make_agent_with_keep_recent_k(k=3)
    agent.system_prompt = 'system'

    agent.add_message('user', 'question')

    agent.add_message('assistant', 'Searching...')
    agent.add_message('tool', 'Full result from round 1', tool_call_id='tc-1', name='google_serper_search')
    agent.add_message('assistant', 'Verifying...')
    agent.add_message('tool', 'Full result from round 2', tool_call_id='tc-2', name='visit')

    built = agent.build_messages()

    tool_contents = [m['content'] for m in built if m['role'] == 'tool']
    assert tool_contents == ['Full result from round 1', 'Full result from round 2']
    assert all(c != OBSERVATION_FOLDED_PLACEHOLDER for c in tool_contents)


@pytest.mark.asyncio
async def test_no_reset_when_tokens_under_budget() -> None:
    agent = _make_agent()
    agent.max_context_tokens = 10000

    agent.add_message('user', 'Original question')
    agent.add_message('assistant', 'Step 1')
    agent.add_message('tool', 'Result 1', tool_call_id='tc-1', name='google_serper_search')

    agent._context_token_count = 3000

    dummy_message = SimpleNamespace(content='answer', reasoning_content=None, tool_calls=[])
    agent._ask_llm_with_usage = AsyncMock(return_value=dummy_message)

    await agent.ask_llm()

    assert len(agent.messages) == 3
    assert agent._context_reset_count == 0


def test_build_messages_after_reset_with_keep_recent_k_does_not_error() -> None:
    agent = _make_agent_with_keep_recent_k(k=3)
    agent.system_prompt = 'system'
    agent.max_context_tokens = 1000

    agent.add_message('user', 'My question')
    agent.add_message('assistant', 'thinking...')
    agent.add_message('tool', 'some result', tool_call_id='tc-1', name='google_serper_search')

    agent._context_token_count = 2000
    agent._reset_context()

    assert len(agent.messages) == 1

    built = agent.build_messages()
    assert len(built) == 2
    assert built[0] == {'role': 'system', 'content': 'system'}
    assert built[1]['role'] == 'user'
    assert built[1]['content'] == 'My question'


@pytest.fixture(autouse=True)
async def cleanup_jupyter_runners():
    await jupyter_module._clear_all_kernel_runners()
    yield
    await jupyter_module._clear_all_kernel_runners()


@pytest.mark.asyncio
async def test_execute_python_tool() -> None:
    response = await execute_python_tool(ExecutePythonInput(code='print(1)'), ToolContext(tool_configs={}))

    assert response.meta['success'] is True
    assert response.meta['exit_code'] == 0
    assert response.meta['stdout'] == '1\n'
    assert 'Successfully executed' in response.output[0]['text']
