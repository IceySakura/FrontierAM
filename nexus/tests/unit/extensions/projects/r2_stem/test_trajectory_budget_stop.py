"""The stop that ends a HyCode rollout at the model call filling its trajectory budget."""

from __future__ import annotations

import asyncio
import logging

from types import SimpleNamespace
from typing import Any

import pytest

import nexus.extensions.projects.r2_stem.task as task_module

from nexus.core import CONTEXT_LENGTH_EXCEEDED_CODE, ContextLengthExceededError, OpenAIConfig
from nexus.extensions.openai_clients.hytuner import HyTunerClient
from nexus.extensions.projects.r2_stem.task import (
    ScienceAgentTaskInput,
    _TrajectoryBudgetStop,
    run_science_agent_task,
)
from nexus.runtime.server.services.llm_handler import LLMRequestHandler
from nexus.trace import processor_context
from nexus.trace.controller import SpanController
from nexus.trace.processors.hytuner import HytunerProcessor


TASK_LOGGER = 'nexus.extensions.projects.r2_stem.task'


class _FakeClient:
    supports_request_meta = True

    def __init__(
        self,
        *,
        finish_reason: str = 'length',
        total_tokens: int = 100,
        max_model_len: int | None = 100,
        max_tokens: int | None = None,
        initial_prompt_len: int | None = None,
        auxiliary_models: frozenset[str] = frozenset(),
    ):
        self.max_model_len = max_model_len
        if initial_prompt_len is not None:
            self._initial_prompt_len = initial_prompt_len
        self.auxiliary_models = auxiliary_models
        sampling_params = {} if max_tokens is None else {'max_tokens': max_tokens}
        self.sampling_params = sampling_params
        self.config = SimpleNamespace(
            request_args={'sampling_params': sampling_params} if sampling_params else {},
            client_args={},
        )
        self.finish_reason = finish_reason
        self.total_tokens = total_tokens
        self.calls: list[dict[str, Any]] = []

    def _is_auxiliary_model(self, model: Any) -> bool:
        return model in self.auxiliary_models

    async def create_completion(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=self.finish_reason)],
            usage=SimpleNamespace(
                total_tokens=self.total_tokens,
                prompt_tokens=40,
                completion_tokens=self.total_tokens - 40,
            ),
        )


def _stop(client: Any) -> _TrajectoryBudgetStop:
    return _TrajectoryBudgetStop(client, sample_id='sample-1', trajectory_id='7_traj')


async def _unused_agent(**kwargs: Any) -> None:
    pytest.fail('this input is only used to read configuration')


def _row(*, stop_at_trajectory_budget: bool = True) -> ScienceAgentTaskInput:
    return ScienceAgentTaskInput(
        question='q',
        sample_id='sample-1',
        stop_at_trajectory_budget=stop_at_trajectory_budget,
        run_agent=_unused_agent,
    )


def _ctx(client: Any) -> SimpleNamespace:
    return SimpleNamespace(openai_clients={'main': client}, task_config={'traj_id': '7_traj'})


# --- the stop itself ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_that_fills_the_window_is_kept_and_ends_the_rollout(caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeClient(total_tokens=100)
    stop = _stop(client)

    with caplog.at_level(logging.WARNING, logger=TASK_LOGGER):
        with pytest.raises(ContextLengthExceededError, match='trajectory budget used up'):
            await stop.create_completion(messages=[{'role': 'user', 'content': 'q'}])

    # The client generated the call, so its tokens are already recorded for training.
    assert len(client.calls) == 1
    assert stop.exhausted is True
    assert 'model call filled the trajectory budget' in caplog.text
    assert (
        'sample_id=sample-1 traj_id=7_traj total_tokens=100 budget=100 max_tokens=None max_model_len=100' in caplog.text
    )


@pytest.mark.asyncio
async def test_calls_after_the_stop_never_reach_the_client(caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeClient(total_tokens=100)
    stop = _stop(client)
    with pytest.raises(ContextLengthExceededError):
        await stop.create_completion(messages=[])

    with caplog.at_level(logging.WARNING, logger=TASK_LOGGER):
        for _ in range(2):
            with pytest.raises(ContextLengthExceededError, match='already used up'):
                await stop.create_completion(messages=[], model='claude-haiku')

    assert len(client.calls) == 1
    assert stop.refused_calls == 2
    assert 'refused model call #2 after the trajectory budget was used up' in caplog.text
    assert 'model=claude-haiku' in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('finish_reason', 'total_tokens'),
    [
        # Cut short by a per-call limit while the window still has room.
        ('length', 99),
        ('stop', 100),
        ('tool_calls', 100),
    ],
)
async def test_calls_that_leave_the_rollout_running_pass_through(finish_reason: str, total_tokens: int) -> None:
    client = _FakeClient(finish_reason=finish_reason, total_tokens=total_tokens)
    stop = _stop(client)

    for _ in range(2):
        completion = await stop.create_completion(messages=[])
        assert completion.choices[0].finish_reason == finish_reason

    assert len(client.calls) == 2
    assert stop.exhausted is False


@pytest.mark.asyncio
async def test_tighter_max_tokens_ends_the_rollout_before_the_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _FakeClient(total_tokens=262144, max_model_len=393216, max_tokens=262144)
    stop = _stop(client)

    with caplog.at_level(logging.WARNING, logger=TASK_LOGGER):
        with pytest.raises(ContextLengthExceededError, match='reached budget=262144'):
            await stop.create_completion(messages=[])

    assert stop.exhausted is True
    assert 'budget=262144 max_tokens=262144 max_model_len=393216' in caplog.text


@pytest.mark.asyncio
async def test_length_under_max_tokens_still_passes_through_a_larger_window() -> None:
    client = _FakeClient(finish_reason='length', total_tokens=99, max_model_len=393216, max_tokens=262144)
    stop = _stop(client)

    completion = await stop.create_completion(messages=[])
    assert completion.choices[0].finish_reason == 'length'
    assert stop.exhausted is False


@pytest.mark.asyncio
async def test_window_still_stops_when_max_tokens_is_looser() -> None:
    client = _FakeClient(total_tokens=100, max_model_len=100, max_tokens=256)
    stop = _stop(client)

    with pytest.raises(ContextLengthExceededError, match='reached budget=100'):
        await stop.create_completion(messages=[])
    assert stop.exhausted is True


@pytest.mark.asyncio
async def test_initial_prompt_does_not_count_against_max_tokens() -> None:
    # Window 128, budget 40, initial prompt 25: a per-call length stop at total 55 has used only
    # 30 of the 40 response tokens, so the rollout keeps going.
    client = _FakeClient(total_tokens=55, max_model_len=128, max_tokens=40, initial_prompt_len=25)
    stop = _stop(client)

    completion = await stop.create_completion(messages=[])

    assert completion.choices[0].finish_reason == 'length'
    assert stop.exhausted is False


@pytest.mark.asyncio
async def test_response_budget_after_the_initial_prompt_ends_the_rollout(caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeClient(total_tokens=65, max_model_len=128, max_tokens=40, initial_prompt_len=25)
    stop = _stop(client)

    with caplog.at_level(logging.WARNING, logger=TASK_LOGGER):
        with pytest.raises(ContextLengthExceededError, match='reached budget=65'):
            await stop.create_completion(messages=[])

    assert stop.exhausted is True
    assert 'budget=65 max_tokens=40 max_model_len=128' in caplog.text
    assert 'initial_prompt_len=25' in caplog.text


@pytest.mark.asyncio
async def test_window_caps_max_tokens_plus_the_initial_prompt() -> None:
    client = _FakeClient(total_tokens=100, max_model_len=100, max_tokens=90, initial_prompt_len=25)
    stop = _stop(client)

    with pytest.raises(ContextLengthExceededError, match='reached budget=100'):
        await stop.create_completion(messages=[])
    assert stop.exhausted is True


@pytest.mark.asyncio
async def test_auxiliary_length_does_not_end_the_main_trajectory() -> None:
    client = _FakeClient(total_tokens=100, auxiliary_models=frozenset({'claude-haiku'}))
    stop = _stop(client)

    completion = await stop.create_completion(messages=[], model='claude-haiku')
    assert completion.choices[0].finish_reason == 'length'
    assert stop.exhausted is False

    with pytest.raises(ContextLengthExceededError, match='trajectory budget used up'):
        await stop.create_completion(messages=[], model='claude-opus-4-6')
    assert len(client.calls) == 2
    assert stop.exhausted is True


def test_hytuner_client_keeps_the_auxiliary_routing_check_the_stop_reads() -> None:
    client = HyTunerClient(
        OpenAIConfig(
            client_args={'tokenizer': _Tokenizer(), 'max_model_len': 64, 'auxiliary_model_names': ['helper-model']},
            request_args={},
        )
    )

    assert client._is_auxiliary_model('claude-haiku-4-5') is True
    assert client._is_auxiliary_model('claude-sonnet-4-6[1m]') is True
    assert client._is_auxiliary_model('vendor/helper-model') is True
    assert client._is_auxiliary_model('claude-opus-4-6') is False
    assert client._is_auxiliary_model(None) is False


def test_everything_but_create_completion_is_the_clients() -> None:
    client = _FakeClient()
    stop = _stop(client)

    assert stop.config is client.config
    assert stop.supports_request_meta is True
    assert stop.max_model_len == 100
    assert not hasattr(stop, 'no_such_attribute')


# --- installing it for one agent run --------------------------------------------------------


@pytest.mark.parametrize(
    ('stop_at_trajectory_budget', 'agent_name'),
    [(False, 'hycode'), (True, 'mini_swe_agent_stem'), (True, 'terminus2')],
)
def test_only_hycode_rows_that_ask_get_the_stop(stop_at_trajectory_budget: bool, agent_name: str) -> None:
    client = _FakeClient()
    ctx = _ctx(client)

    assert (
        _TrajectoryBudgetStop.install(ctx, _row(stop_at_trajectory_budget=stop_at_trajectory_budget), agent_name)
        is None
    )
    assert ctx.openai_clients['main'] is client


def test_a_client_without_a_model_window_gets_no_stop() -> None:
    client = _FakeClient(max_model_len=None)
    ctx = _ctx(client)

    assert _TrajectoryBudgetStop.install(ctx, _row(), 'hycode') is None
    assert ctx.openai_clients['main'] is client


@pytest.mark.asyncio
async def test_restore_puts_the_client_back_and_summarises_the_stop(caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeClient(total_tokens=100)
    ctx = _ctx(client)
    stop = _TrajectoryBudgetStop.install(ctx, _row(), 'hycode')
    assert ctx.openai_clients['main'] is stop
    for _ in range(2):
        with pytest.raises(ContextLengthExceededError):
            await stop.create_completion(messages=[])

    with caplog.at_level(logging.WARNING, logger=TASK_LOGGER):
        stop.restore(ctx, 'truncation')

    assert ctx.openai_clients['main'] is client
    assert 'rollout ended at the trajectory budget: sample_id=sample-1 traj_id=7_traj' in caplog.text
    assert 'refused_calls=1 agent_exit_status=truncation' in caplog.text


def test_restore_is_silent_when_the_stop_never_fired(caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeClient()
    ctx = _ctx(client)
    stop = _TrajectoryBudgetStop.install(ctx, _row(), 'hycode')

    with caplog.at_level(logging.WARNING, logger=TASK_LOGGER):
        stop.restore(ctx, 'env_done')

    assert ctx.openai_clients['main'] is client
    assert '[trajectory-budget-stop]' not in caplog.text


# --- the task installs it around run_agent --------------------------------------------------


def _task_context(client: Any) -> Any:
    runtime = SimpleNamespace()

    class FakeProvider:
        async def release(self, name: str) -> None:
            pass

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()
        openai_clients = {'main': client}
        task_config = {'traj_id': '7_traj'}

        async def get_runtime(self, name: str) -> Any:
            return runtime

        def get_runtime_id(self, target: str) -> None:
            return None

    return FakeContext()


def _hycode_row(run_agent: Any, **overrides: Any) -> ScienceAgentTaskInput:
    return ScienceAgentTaskInput(
        question='Solve it.',
        agent_name='hycode',
        ref_answer='reference',
        evaluation_method='rmsys',
        run_agent=run_agent,
        **{'stop_at_trajectory_budget': True, **overrides},
    )


async def _no_verifier(runtime: Any) -> None:
    pass


@pytest.mark.asyncio
async def test_hycode_rollout_runs_behind_the_stop_and_gets_its_client_back(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    ctx = _task_context(client)
    seen: dict[str, Any] = {}

    async def fake_agent(**kwargs: Any) -> SimpleNamespace:
        seen['client'] = ctx.openai_clients['main']
        return SimpleNamespace(termination_reason='TRUNCATION', num_turns=3, output='')

    async def fail_rmsys_evaluate(**kwargs: Any) -> float:
        pytest.fail('a truncated rollout must not be judged')

    monkeypatch.setattr(task_module, 'require_context', lambda: ctx)
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', _no_verifier)
    monkeypatch.setattr(task_module, 'rmsys_evaluate', fail_rmsys_evaluate)

    output = await run_science_agent_task(_hycode_row(fake_agent))

    assert isinstance(seen['client'], _TrajectoryBudgetStop)
    assert seen['client'].client is client
    assert ctx.openai_clients['main'] is client
    assert output.score == 0.0
    assert output.test_results == {'skipped_evaluator_reason': 'TRUNCATION', 'exit_status': 'TRUNCATION'}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['raises', 'times_out'])
async def test_the_stop_is_removed_when_the_agent_does_not_return(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    client = _FakeClient()
    ctx = _task_context(client)

    async def fake_agent(**kwargs: Any) -> None:
        if failure == 'raises':
            raise RuntimeError('sandbox died')
        await asyncio.sleep(3600)

    monkeypatch.setattr(task_module, 'require_context', lambda: ctx)
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', _no_verifier)

    timeout = {'agent_timeout': 0} if failure == 'times_out' else {}
    output = await run_science_agent_task(_hycode_row(fake_agent, **timeout))

    assert ctx.openai_clients['main'] is client
    assert output.exit_status == ('agent_error' if failure == 'raises' else 'timeout')


@pytest.mark.asyncio
async def test_rows_without_the_flag_keep_the_plain_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    ctx = _task_context(client)
    seen: dict[str, Any] = {}

    async def fake_agent(**kwargs: Any) -> SimpleNamespace:
        seen['client'] = ctx.openai_clients['main']
        return SimpleNamespace(termination_reason='TRUNCATION', num_turns=3, output='')

    monkeypatch.setattr(task_module, 'require_context', lambda: ctx)
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', _no_verifier)

    await run_science_agent_task(_hycode_row(fake_agent, stop_at_trajectory_budget=False))

    assert seen['client'] is client


# --- end to end through HyTunerClient, the LLM bridge and the trajectory processor ----------


class _Tokenizer:
    def apply_chat_template(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool, **_: Any) -> str:
        rendered = ''.join(f'<{message["role"]}>{message.get("content") or ""}' for message in messages)
        return rendered + ('<assistant>' if add_generation_prompt else '')

    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], **_: Any) -> str:
        return 'thought</think>answer'


class _BudgetFillingServer:
    """Generates exactly the budget it is given, as a model that never stops on its own."""

    def __init__(self) -> None:
        self.outputs: list[list[int]] = []

    async def generate(self, **kwargs: Any) -> SimpleNamespace:
        limit = int(kwargs['sampling_params']['max_tokens'])
        token_ids = [100_001 + index for index in range(limit)]
        self.outputs.append(token_ids)
        return SimpleNamespace(token_ids=token_ids, log_probs=[-0.1] * limit, stop_reason='length', extra_fields={})


class _Runtime:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.emitted.append((event, payload))


QUESTION = {'role': 'user', 'content': 'question'}
# What HyCode sends after a cut-off turn: the truncated assistant message is gone and a queued
# task-notification (or the max-output recovery prompt) follows, so the cached prefix no longer matches.
REWRITTEN_HISTORY = [QUESTION, {'role': 'user', 'content': 'continue'}]


def _hytuner_client(monkeypatch: pytest.MonkeyPatch, server: _BudgetFillingServer) -> HyTunerClient:
    monkeypatch.setattr(SpanController, '_instance', None)
    client = HyTunerClient(
        OpenAIConfig(
            client_args={'tokenizer': _Tokenizer(), 'server_manager': server, 'max_model_len': 64},
            request_args={},
        )
    )

    async def parse_tool_calls(token_ids: list[int], request: Any = None) -> tuple[str, str, list]:
        return '', 'thought</think>answer', []

    monkeypatch.setattr(client, '_parse_tool_calls', parse_tool_calls)
    return client


async def _send(handler: LLMRequestHandler, runtime: _Runtime, request_id: str, messages: list[dict]) -> dict:
    await handler.handle_llm_request(
        {'request_id': request_id, 'messages': messages, 'rsio': {'request_id': request_id, 'attempt': 1}}
    )
    return next(
        payload
        for event, payload in reversed(runtime.emitted)
        if event == 'llm_response'
        and payload.get('request_id') == request_id
        and (payload.get('rsio') or {}).get('kind') == 'final'
    )


async def _run_cut_off_then_rewrite(monkeypatch: pytest.MonkeyPatch, *, stopped: bool) -> tuple[Any, ...]:
    server = _BudgetFillingServer()
    client = _hytuner_client(monkeypatch, server)
    processor = HytunerProcessor(file_path=None, trajectory_id='7_traj')
    # single.run binds the processor the same way before the task starts.
    client.set_training_data_observer(processor)
    runtime = _Runtime()
    handler = LLMRequestHandler(runtime=runtime, openai_client=_stop(client) if stopped else client)

    with processor_context([processor]):
        cut_off = await _send(handler, runtime, 'cut-off', [QUESTION])
        rewrite = await _send(handler, runtime, 'rewrite', REWRITTEN_HISTORY)
    return server, handler, processor.get_result(), cut_off, rewrite


@pytest.mark.asyncio
async def test_hytuner_client_keeps_the_initial_prompt_length_the_budget_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _BudgetFillingServer()
    client = _hytuner_client(monkeypatch, server)
    assert client._initial_prompt_len is None

    runtime = _Runtime()
    handler = LLMRequestHandler(runtime=runtime, openai_client=client)
    processor = HytunerProcessor(file_path=None, trajectory_id='7_traj')
    client.set_training_data_observer(processor)
    with processor_context([processor]):
        await _send(handler, runtime, 'first', [QUESTION])

    rendered = _Tokenizer().apply_chat_template([QUESTION], add_generation_prompt=True)
    assert client._initial_prompt_len == len(_Tokenizer().encode(rendered))


@pytest.mark.asyncio
async def test_rewrite_after_the_budget_no_longer_masks_the_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    server, handler, result, cut_off, rewrite = await _run_cut_off_then_rewrite(monkeypatch, stopped=True)

    assert cut_off['error_code'] == CONTEXT_LENGTH_EXCEEDED_CODE
    assert rewrite['error_code'] == CONTEXT_LENGTH_EXCEEDED_CODE
    # The rewrite never reached the model, and the cut-off turn trains as recorded.
    assert len(server.outputs) == 1
    assert result['unhandled_prefix_reset'] is False
    assert result['response_ids'] == server.outputs[0]
    assert result['num_turns'] == 1
    # HyCode's classification reads these outcomes to report TRUNCATION.
    assert set(handler.request_outcomes.values()) == {CONTEXT_LENGTH_EXCEEDED_CODE}


@pytest.mark.asyncio
async def test_without_the_stop_the_same_rewrite_masks_the_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    server, handler, result, cut_off, rewrite = await _run_cut_off_then_rewrite(monkeypatch, stopped=False)

    assert cut_off['response']['choices'][0]['finish_reason'] == 'length'
    assert rewrite['success'] is True
    # The rewritten history got a fresh budget and generated again, so the processor masked the rollout.
    assert len(server.outputs) == 2
    assert result['unhandled_prefix_reset'] is True
    assert result['termination_reason'] == 'UNHANDLED_ERROR'
    assert set(handler.request_outcomes.values()) == {None}
