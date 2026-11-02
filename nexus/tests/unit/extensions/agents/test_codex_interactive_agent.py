"""Top-level ``run_codex_interactive`` tests: the real Host multi-turn loop.

The other suites stop at the runtime boundary. These drive the registered agent
through the unmodified ``run_installed_multiturn`` to prove the Host contract
actually holds — above all that questions reach the respond target instead of
being mistaken for the next user prompt.
"""

from __future__ import annotations

import asyncio

from typing import Any

import pytest

from nexus.extensions.agents.codex.agent import CodexInteractiveInput, run_codex_interactive


asyncio_test = pytest.mark.asyncio


class _RuntimeScript:
    """Stands in for the runtime layer, recording how the Host calls it."""

    def __init__(self, turns: list[dict[str, Any]], responses: list[dict[str, Any]] | None = None) -> None:
        self.turns = list(turns)
        self.responses = list(responses or [])
        self.prompts: list[str] = []
        self.turn_kwargs: list[dict[str, Any]] = []
        self.answers: list[tuple[str, str | None]] = []
        self.response_kwargs: list[dict[str, Any]] = []
        self.closed: list[str] = []

    async def call(self, target, **kwargs: Any) -> dict[str, Any]:
        name = target if isinstance(target, str) else getattr(target, '__name__', str(target))
        if 'run_codex_interactive_session_turn' in name:
            self.prompts.append(kwargs['prompt'])
            self.turn_kwargs.append(dict(kwargs))
            return self.turns.pop(0)
        if 'respond_and_wait' in name:
            self.answers.append((kwargs['interaction_id'], kwargs['answer']))
            self.response_kwargs.append(dict(kwargs))
            return self.responses.pop(0)
        if 'close_codex_interactive_session' in name:
            self.closed.append(kwargs.get('session_id', ''))
            return {'success': True, 'stderr': ''}
        raise AssertionError(f'unexpected runtime target: {name}')


def _turn(**kwargs: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        'success': True,
        'status': 'completed',
        'session_id': 's',
        'output': '',
        'stderr': '',
        'exit_code': 0,
        'execution_time': 0.1,
        'events': [],
        'latest_screen': '',
        'recent_events': [],
        'event_cursor': {'after_seq': 0, 'latest_seq': 0, 'count': 0, 'truncated': False, 'dropped_count': 0},
        'session_reusable': True,
    }
    base.update(kwargs)
    return base


def _needs_input(kind: str, interaction_id: str = 'req-1', **kwargs: Any) -> dict[str, Any]:
    return _turn(
        status='needs_user_input',
        output='',
        interaction={
            'interaction_id': interaction_id,
            'kind': kind,
            'method': (
                'item/tool/requestUserInput' if kind == 'user_input' else 'item/commandExecution/requestApproval'
            ),
            'params': {'question': 'may I?'},
        },
        **kwargs,
    )


async def _end_after_first_turn(request: Any) -> str | None:
    """Answer an interaction, then stop after one completed turn."""
    return 'yes' if getattr(request, 'source', None) == 'hook' else None


def _answer_interactions_only(request: Any, answer: str) -> str | None:
    """Answer hook/interaction requests; end the run when asked for a new prompt.

    ``run_installed_multiturn`` reuses one provider for both purposes, so a
    provider that always returns text would silently start another turn.
    """
    source = getattr(request, 'source', None)
    return answer if source == 'hook' else None


@pytest.fixture
def patched_runtime(monkeypatch):
    """Route the agent's runtime calls into a script we can assert on."""

    def _install(script: _RuntimeScript) -> _RuntimeScript:
        from nexus.extensions.agents.installed_common import multiturn

        async def _call(target, /, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return await script.call(target, **kwargs)

        monkeypatch.setattr(multiturn, 'call_runtime_agent', _call)
        return script

    return _install


@asyncio_test
@pytest.mark.parametrize('success', [True, False])
async def test_close_failure_preserves_turn_result(patched_runtime, success):
    class CloseFailure(_RuntimeScript):
        async def call(self, target, **kwargs):
            if 'close_codex_interactive_session' in str(target):
                self.closed.append(kwargs['session_id'])
                raise TimeoutError('close deadline')
            return await super().call(target, **kwargs)

    script = patched_runtime(
        CloseFailure(
            [
                _turn(
                    success=success,
                    exit_code=0 if success else 1,
                    status='completed' if success else 'failed',
                    output='original output',
                    stderr='' if success else 'original failure',
                    events=[{'type': 'original-event'}],
                )
            ]
        )
    )
    result = await run_codex_interactive(CodexInteractiveInput(user_prompt='go', max_turns=1, working_dir='/workspace'))
    assert result.success is success
    assert result.exit_code == (0 if success else 1)
    assert result.status == ('completed' if success else 'failed')
    assert result.output == 'original output'
    assert result.events == [{'type': 'original-event'}]
    assert 'close deadline' in result.stderr
    if not success:
        assert 'original failure' in result.stderr
    assert len(script.closed) == 1


@asyncio_test
@pytest.mark.parametrize('error_type', [RuntimeError, asyncio.CancelledError])
async def test_close_failure_does_not_replace_callback_error(patched_runtime, error_type):
    class CloseFailure(_RuntimeScript):
        async def call(self, target, **kwargs):
            if 'close_codex_interactive_session' in str(target):
                self.closed.append(kwargs['session_id'])
                raise TimeoutError('close deadline')
            return await super().call(target, **kwargs)

    script = patched_runtime(CloseFailure([_turn(output='done')]))

    async def callback(_request):
        raise error_type('original callback error')

    with pytest.raises(error_type, match='original callback error'):
        await run_codex_interactive(
            CodexInteractiveInput(user_prompt='go', working_dir='/workspace', get_user_input=callback)
        )
    assert len(script.closed) == 1


@asyncio_test
async def test_single_turn_returns_the_final_output(patched_runtime):
    script = patched_runtime(_RuntimeScript([_turn(output='HELLO FROM CODEX')]))
    result = await run_codex_interactive(
        CodexInteractiveInput(
            user_prompt='hi',
            working_dir='/workspace',
            close_on_finish=True,
        )
    )
    assert result.success is True
    assert result.output == 'HELLO FROM CODEX'
    assert result.turns == 1
    assert script.prompts == ['hi']
    assert script.turn_kwargs[0]['disable_invoke_wait_timeout'] is True
    assert script.closed  # close_on_finish honoured


@asyncio_test
async def test_user_question_reaches_respond_target_not_the_next_prompt(patched_runtime):
    script = patched_runtime(
        _RuntimeScript(
            turns=[_needs_input('user_input', 'q-1')],
            responses=[_turn(output='ANSWERED')],
        )
    )

    async def provider(request):
        return _answer_interactions_only(request, 'PostgreSQL')

    result = await run_codex_interactive(
        CodexInteractiveInput(user_prompt='ask me', working_dir='/workspace', get_user_input=provider)
    )
    assert script.answers == [('q-1', 'PostgreSQL')]
    assert script.response_kwargs[0]['disable_invoke_wait_timeout'] is True
    assert script.prompts == ['ask me']
    assert result.output == 'ANSWERED'


@asyncio_test
async def test_multiple_interactions_in_one_turn_all_route_to_respond(patched_runtime):
    script = patched_runtime(
        _RuntimeScript(
            turns=[_needs_input('user_input', 'q-1')],
            responses=[_needs_input('user_input', 'q-2'), _turn(output='FINALLY')],
        )
    )

    async def provider(request):
        return _answer_interactions_only(request, 'ok')

    result = await run_codex_interactive(
        CodexInteractiveInput(user_prompt='go', working_dir='/workspace', get_user_input=provider)
    )
    assert script.answers == [('q-1', 'ok'), ('q-2', 'ok')]
    assert script.prompts == ['go']
    assert result.output == 'FINALLY'


@asyncio_test
async def test_two_turns_when_the_provider_supplies_another_prompt(patched_runtime):
    script = patched_runtime(_RuntimeScript([_turn(output='FIRST'), _turn(output='SECOND')]))
    prompts = iter(['second prompt'])

    async def provider(request):
        return next(prompts, None)

    result = await run_codex_interactive(
        CodexInteractiveInput(user_prompt='first prompt', working_dir='/workspace', get_user_input=provider)
    )
    assert script.prompts == ['first prompt', 'second prompt']
    assert result.turns == 2
    assert 'SECOND' in result.output


@asyncio_test
async def test_failed_turn_stops_the_loop_and_reports_failure(patched_runtime):
    script = patched_runtime(
        _RuntimeScript([_turn(success=False, status='timeout', stderr='no boundary', exit_code=-1)])
    )

    async def provider(request):
        raise AssertionError('must not ask for another prompt after a failure')

    result = await run_codex_interactive(
        CodexInteractiveInput(user_prompt='hangs', working_dir='/workspace', get_user_input=provider)
    )
    assert result.success is False
    assert result.status == 'timeout'
    assert script.prompts == ['hangs']


def test_agent_declares_the_codex_gateway_scaffold_policy():
    """HyTuner rejects agents without a declared scaffold policy."""
    from nexus.core import agent_registry

    interactive = agent_registry.get('codex_interactive')
    nexus_meta = getattr(interactive, 'nexus_meta', {}) or {}
    gateway = (nexus_meta.get('meta') or {}).get('gateway') or {}
    assert gateway.get('scaffold_policy') == 'codex'


def test_scaffold_policy_resolves_through_the_public_helper():
    from nexus.core.agent import get_agent_gateway_scaffold_policy

    # Must not raise, and must match the batch agent's declared policy.
    assert get_agent_gateway_scaffold_policy('codex_interactive') == 'codex'
    assert get_agent_gateway_scaffold_policy('codex_interactive') == get_agent_gateway_scaffold_policy('codex')


@asyncio_test
async def test_close_on_finish_false_leaves_the_session_open(patched_runtime):
    script = patched_runtime(_RuntimeScript([_turn(output='x')]))
    await run_codex_interactive(
        CodexInteractiveInput(
            user_prompt='hi',
            working_dir='/workspace',
            close_on_finish=False,
            get_user_input=_end_after_first_turn,
        )
    )
    assert script.closed == []


# --------------------------------------------------------------------------- #
# Optional input provider
# --------------------------------------------------------------------------- #


@asyncio_test
async def test_default_provider_ends_after_one_turn(patched_runtime):
    script = patched_runtime(_RuntimeScript([_turn(output='UNATTENDED OK')]))
    result = await run_codex_interactive(
        CodexInteractiveInput(
            user_prompt='hi',
            working_dir='/workspace',
        )
    )
    assert result.success is True
    assert script.prompts == ['hi']


@asyncio_test
async def test_interaction_payload_reaches_the_provider_with_display_fields(patched_runtime):
    waiting = _needs_input('user_input', 'q-1')
    waiting['interaction']['hint'] = 'Codex asks: Which database? Options: PostgreSQL, MySQL.'
    waiting['interaction']['tool_name'] = 'request_user_input'
    waiting['interaction']['tool_input'] = {'questions': [{'question': 'Which database?'}]}
    patched_runtime(_RuntimeScript(turns=[waiting], responses=[_turn(output='DONE')]))

    seen: list[Any] = []

    async def provider(request):
        seen.append(request)
        return _answer_interactions_only(request, 'yes')

    await run_codex_interactive(
        CodexInteractiveInput(user_prompt='go', working_dir='/workspace', get_user_input=provider)
    )
    assert seen, 'the provider must be consulted'
    rendered = repr(seen[0])
    assert 'Which database?' in rendered or 'request_user_input' in rendered
