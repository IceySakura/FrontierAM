import json

import pytest

from nexus.core import OpenAICompatibleClient, OpenAIConfig
from nexus.core.context import ExecutionContext, require_context
from nexus.extensions.agents.codex.agent import CodexInput, run_codex
from nexus.extensions.harbor_bridge.agent_result import agent_termination_reason, build_t3_agent_result


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runtime_result: dict[str, object],
    span_id: str | None = 'span-123',
) -> dict[str, object]:
    """Patch call_runtime_agent + trace helpers in the codex.agent namespace.

    Returns a dict that records the attributes passed to update_span (if any).
    """
    recorded: dict[str, object] = {}

    async def fake_call_runtime_agent(target: object, /, **kwargs: object) -> dict[str, object]:
        recorded['runtime_kwargs'] = kwargs
        try:
            recorded['context_timeout_id'] = getattr(require_context(), 'runtime_command_timeout_id', None)
        except RuntimeError:
            recorded['context_timeout_id'] = None
        return runtime_result

    def fake_get_current_span_id() -> str | None:
        return span_id

    def fake_update_span(sid: str, *, attributes: dict[str, object]) -> None:
        recorded['span_id'] = sid
        recorded['attributes'] = attributes

    monkeypatch.setattr('nexus.extensions.agents.codex.agent.call_runtime_agent', fake_call_runtime_agent)
    monkeypatch.setattr('nexus.extensions.agents.codex.agent.get_current_span_id', fake_get_current_span_id)
    monkeypatch.setattr('nexus.extensions.agents.codex.agent.update_span', fake_update_span)
    return recorded


class _NoNativeCompactionClient(OpenAICompatibleClient):
    async def _create_completion(self, messages, model=None, **kwargs):  # noqa: ANN001, ANN003, ANN202
        raise AssertionError('rollout must not start when native compaction is unsupported')


@pytest.mark.asyncio
@pytest.mark.parametrize('version', ['0.136.0', '0.147.0', 'latest'])
async def test_native_responses_compaction_client_capability_is_checked_before_rollout(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    recorded = _install_fakes(monkeypatch, runtime_result={'success': True})
    client = _NoNativeCompactionClient(OpenAIConfig(client_args={}, request_args={}))
    context = ExecutionContext(openai_clients={'main': client})

    async with context:
        with pytest.raises(ValueError, match='HyTuner training must set use_azure_provider=false'):
            await run_codex(
                CodexInput(
                    user_prompt='task',
                    working_dir='/workspace',
                    version=version,
                    use_azure_provider=True,
                )
            )

    assert 'runtime_kwargs' not in recorded


@pytest.mark.asyncio
@pytest.mark.parametrize('version_kwargs', [{}, {'version': None}, {'version': '0.114.0'}])
async def test_default_codex_does_not_require_native_compaction(
    monkeypatch: pytest.MonkeyPatch,
    version_kwargs: dict[str, str | None],
) -> None:
    recorded = _install_fakes(monkeypatch, runtime_result={'success': True, 'exit_code': 0})
    client = _NoNativeCompactionClient(OpenAIConfig(client_args={}, request_args={}))
    context = ExecutionContext(openai_clients={'main': client})
    input_data = CodexInput(user_prompt='task', working_dir='/workspace', **version_kwargs)

    assert input_data.version == '0.114.0'
    assert input_data.use_azure_provider is True
    async with context:
        result = await run_codex(input_data)

    assert result.success is True
    assert recorded['runtime_kwargs']['version'] == '0.114.0'


@pytest.mark.asyncio
@pytest.mark.parametrize('supports_tool_references', [None, False, True])
async def test_run_codex_propagates_tool_reference_capability(
    monkeypatch: pytest.MonkeyPatch, supports_tool_references: bool | None
) -> None:
    recorded = _install_fakes(monkeypatch, runtime_result={'success': True, 'exit_code': 0})

    await run_codex(
        CodexInput(user_prompt='task', working_dir='/workspace', supports_tool_references=supports_tool_references)
    )

    if supports_tool_references is None:
        assert 'supports_tool_references' not in recorded['runtime_kwargs']
    else:
        assert recorded['runtime_kwargs']['supports_tool_references'] is supports_tool_references


@pytest.mark.asyncio
async def test_run_codex_propagates_inline_schema_format(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_fakes(monkeypatch, runtime_result={'success': True, 'exit_code': 0})

    await run_codex(
        CodexInput(
            user_prompt='task',
            working_dir='/workspace',
            supports_tool_references=True,
            tool_search_output_format='tool_search_output',
        )
    )

    assert recorded['runtime_kwargs']['supports_tool_references'] is True, 'Inline schemas must remain deferred'
    assert recorded['runtime_kwargs']['tool_search_output_format'] == 'tool_search_output', (
        'Agent must forward the VLM tool search format to its runtime'
    )


@pytest.mark.asyncio
async def test_run_codex_propagates_0147_config_without_changing_default(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
            'timed_out': False,
        },
    )

    model_catalog = {'models': [{'slug': 'gpt-5', 'apply_patch_tool_type': 'freeform'}]}

    await run_codex(
        CodexInput(
            user_prompt='task',
            working_dir='/workspace',
            version='0.147.0',
            codex_home_name='training-parity',
            normalize_exec_command_metadata=True,
            model_context_window=262144,
            reasoning_effort='high',
            auto_compact_token_limit=65536,
            auto_compact_token_limit_scope='total',
            compact_prompt='custom compact prompt',
            model_catalog_json=model_catalog,
            experimental_use_freeform_apply_patch=True,
            images=['/workspace/inputs/0/a.png', '/workspace/inputs/1/b.png'],
        )
    )

    runtime_kwargs = recorded['runtime_kwargs']
    assert runtime_kwargs['version'] == '0.147.0'
    assert runtime_kwargs['codex_home_name'] == 'training-parity'
    assert runtime_kwargs['normalize_exec_command_metadata'] is True
    # One -i per image, ahead of the -c overrides.
    assert runtime_kwargs['extra_cli_args'][:4] == [
        '-i',
        '/workspace/inputs/0/a.png',
        '-i',
        '/workspace/inputs/1/b.png',
    ]
    assert runtime_kwargs['extra_cli_args'][-6:] == [
        '-c',
        'model_context_window=262144',
        '-c',
        'model_reasoning_effort="high"',
        '-c',
        'model_auto_compact_token_limit_scope="total"',
    ]
    assert runtime_kwargs['compact_prompt'] == 'custom compact prompt'
    assert runtime_kwargs['model_catalog_json'] == model_catalog
    assert runtime_kwargs['experimental_use_freeform_apply_patch'] is True
    assert 'emit_json_trace' not in runtime_kwargs
    assert 'wrap_plaintext_reasoning_content' not in runtime_kwargs
    assert CodexInput.model_fields['version'].default == '0.114.0'
    assert CodexInput.model_fields['model_catalog_json'].default is None
    assert CodexInput.model_fields['model_catalog_overrides'].default is None
    assert CodexInput.model_fields['experimental_use_freeform_apply_patch'].default is None

    default_recorded = _install_fakes(monkeypatch, runtime_result={'success': True})
    await run_codex(CodexInput(user_prompt='task', working_dir='/workspace'))
    assert default_recorded['runtime_kwargs']['version'] == '0.114.0'
    assert 'model_catalog_json' not in default_recorded['runtime_kwargs']
    assert 'model_catalog_overrides' not in default_recorded['runtime_kwargs']

    override_recorded = _install_fakes(monkeypatch, runtime_result={'success': True})
    model_catalog_overrides = {'multi_agent_version': 'v1', 'use_responses_lite': False}
    await run_codex(
        CodexInput(
            user_prompt='task',
            working_dir='/workspace',
            model_catalog_overrides=model_catalog_overrides,
        )
    )
    assert override_recorded['runtime_kwargs']['model_catalog_overrides'] == model_catalog_overrides
    assert 'experimental_use_freeform_apply_patch' not in default_recorded['runtime_kwargs']
    assert 'wrap_plaintext_reasoning_content' not in default_recorded['runtime_kwargs']

    wrap_off_recorded = _install_fakes(monkeypatch, runtime_result={'success': True})
    await run_codex(
        CodexInput(
            user_prompt='task',
            working_dir='/workspace',
            wrap_plaintext_reasoning_content=False,
        )
    )
    assert wrap_off_recorded['runtime_kwargs']['wrap_plaintext_reasoning_content'] is False

    disabled_recorded = _install_fakes(monkeypatch, runtime_result={'success': True})
    await run_codex(
        CodexInput(
            user_prompt='task',
            working_dir='/workspace',
            experimental_use_freeform_apply_patch=False,
        )
    )
    assert disabled_recorded['runtime_kwargs']['experimental_use_freeform_apply_patch'] is False


@pytest.mark.asyncio
async def test_run_codex_rejects_comma_in_image_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """`codex exec -i` splits on commas, so such a path would silently become two missing files."""
    _install_fakes(monkeypatch, runtime_result={'success': True, 'output': '', 'exit_code': 0})

    with pytest.raises(ValueError, match='cannot contain a comma'):
        await run_codex(CodexInput(user_prompt='task', working_dir='/workspace', images=['/workspace/inputs/a,b.png']))


@pytest.mark.asyncio
async def test_run_codex_emits_env_done_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 12.5,
            'timed_out': False,
        },
    )

    out = await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert out.success is True
    assert out.exit_status == 'completed'
    assert out.final_answer == 'done'
    assert out.failure_kind is None
    assert out.termination_reason == 'ENV_DONE'
    assert recorded['attributes']['hytuner.termination_reason'] == 'ENV_DONE'
    assert recorded['attributes']['hytuner.observation_time'] == 12.5
    assert 'emit_json_trace' not in recorded['runtime_kwargs']

    handoff = build_t3_agent_result(mode='agent', agent_output=out)
    assert handoff.status == 'success'
    assert handoff.agent_exit_status == 'completed'
    assert handoff.final_answer == 'done'


@pytest.mark.asyncio
async def test_run_codex_passes_app_server_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
            'termination_reason': 'ENV_DONE',
            'event_stream': {
                'version': 1,
                'source': 'app_server',
                'events': [
                    {
                        'seq': 0,
                        'timestamp': 10.0,
                        'type': 'turn.completed',
                        'data': {'turn': {'id': 'turn-1', 'status': 'completed'}},
                    }
                ],
            },
        },
    )

    projected = []
    monkeypatch.setattr('nexus.extensions.agents.codex.agent._emit_codex_timeline_spans', projected.append)
    out = await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace', mode='app_server'))

    assert recorded['runtime_kwargs']['mode'] == 'app_server'
    assert out.exit_status == 'completed'
    assert out.final_answer == 'done'
    assert out.termination_reason == 'ENV_DONE'
    assert projected == [[]]
    assert out.json_events_stdout is None
    assert out.event_stream is not None
    assert out.event_stream.source == 'app_server'
    assert out.event_stream.events[0].type == 'turn.completed'


@pytest.mark.asyncio
async def test_run_codex_emits_task_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': 'unfinished answer',
            'stderr': 'Execution timed out',
            'exit_code': -1,
            'execution_time': 3600.0,
            'timed_out': True,
        },
    )

    out = await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert out.exit_status == 'timeout'
    assert out.final_answer is None
    assert out.failure_kind == 'agent'
    assert out.failure_stage == 'process'
    assert out.termination_reason == 'TASK_TIMEOUT'
    assert recorded['attributes']['hytuner.termination_reason'] == 'TASK_TIMEOUT'
    assert recorded['attributes']['hytuner.observation_time'] == 3600.0


@pytest.mark.asyncio
async def test_run_codex_emits_unhandled_error_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': 'boom',
            'exit_code': 1,
            'execution_time': 5.0,
            'timed_out': False,
        },
    )

    out = await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert out.exit_status == 'error'
    assert out.failure_kind == 'agent'
    assert out.failure_stage == 'process'
    assert recorded['attributes']['hytuner.termination_reason'] == 'UNHANDLED_ERROR'


@pytest.mark.asyncio
async def test_run_codex_skips_update_when_no_span(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': True,
            'output': '',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
            'timed_out': False,
        },
        span_id=None,
    )

    out = await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    # No active span -> update_span must not be called.
    assert 'attributes' not in recorded
    assert out.success is True
    assert out.final_answer is None


@pytest.mark.asyncio
async def test_run_codex_defaults_missing_timed_out_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    # Early-return runtime paths (install/proxy failure) omit 'timed_out'; a nonzero exit must fall
    # back to UNHANDLED_ERROR, not crash.
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': 'Failed to install Codex CLI',
            'exit_code': -1,
            'execution_time': 0.2,
        },
    )

    out = await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert out.exit_status == 'error'
    assert out.failure_kind == 'infrastructure'
    assert out.failure_stage == 'runtime_protocol'
    assert recorded['attributes']['hytuner.termination_reason'] == 'UNHANDLED_ERROR'


@pytest.mark.asyncio
async def test_run_codex_preserves_runtime_failure_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': 'Failed to start Responses Proxy',
            'exit_code': -1,
            'execution_time': 0.2,
            'failure_kind': 'infrastructure',
            'failure_stage': 'proxy',
        },
    )

    out = await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert out.exit_status == 'error'
    assert out.failure_kind == 'infrastructure'
    assert out.failure_stage == 'proxy'
    assert out.stderr == 'Failed to start Responses Proxy'


# --- mode-controlled --json trace path -----------------------------------------

_TURN_FAILED_JSONL = '\n'.join([
    '{"type":"thread.started","thread_id":"t1"}',
    '{"type":"turn.started"}',
    '{"type":"item.completed","item":{"id":"i0","item_type":"command_execution",'
    '"command":"pytest -q","exit_code":1,"status":"failed"}}',
    '{"type":"turn.failed","error":{"message":"boom"}}',
])


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['exec', 'exec_json', 'app_server'])
@pytest.mark.parametrize('retain', [True, False])
async def test_json_trace_output_is_gated_by_mode_and_flag(
    monkeypatch: pytest.MonkeyPatch, mode: str, retain: bool
) -> None:
    raw_jsonl = '{"type":"thread.started","thread_id":"t1"}\n'
    recorded = _install_fakes(
        monkeypatch, runtime_result={'success': True, 'exit_code': 0, 'json_events_stdout': raw_jsonl}
    )
    output = await run_codex(
        CodexInput(
            user_prompt='inspect',
            working_dir='/workspace',
            mode=mode,
            retain_json_events_stdout=retain,
        )
    )
    assert output.json_events_stdout == (raw_jsonl if mode == 'exec_json' and retain else None)
    assert output.event_stream is None
    assert recorded['runtime_kwargs']['mode'] == mode
    assert 'emit_json_trace' not in recorded['runtime_kwargs']


@pytest.mark.asyncio
async def test_run_codex_retains_raw_json_events_from_task_config(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_jsonl = '{"type":"thread.started","thread_id":"t1"}\n'
    _install_fakes(
        monkeypatch,
        runtime_result={
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
            'timed_out': False,
            'json_events_stdout': raw_jsonl,
        },
    )

    class _FakeCtx:
        runtime_command_timeout_id = None
        task_config = {'retain_json_events_stdout': True}

    monkeypatch.setattr('nexus.core.require_context', lambda: _FakeCtx())

    output = await run_codex(CodexInput(user_prompt='inspect', working_dir='/workspace', use_azure_provider=False))

    assert output.json_events_stdout == raw_jsonl


@pytest.mark.asyncio
@pytest.mark.parametrize('retain', [True, False])
async def test_run_codex_json_trace_refines_termination_from_input_flag(
    monkeypatch: pytest.MonkeyPatch,
    retain: bool,
) -> None:
    # exec_json via the CodexInput mode field (per-sample path). A nonzero exit with a
    # turn.failed event refines the coarse UNHANDLED_ERROR into CODEX_TURN_FAILED.
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': '',
            'exit_code': 1,
            'execution_time': 3.0,
            'timed_out': False,
            'json_events_stdout': _TURN_FAILED_JSONL,
        },
    )

    out = await run_codex(
        CodexInput(
            user_prompt='fix bug',
            working_dir='/workspace',
            mode='exec_json',
            retain_json_events_stdout=retain,
        )
    )

    assert recorded['attributes']['hytuner.termination_reason'] == 'CODEX_TURN_FAILED'
    assert out.event_stream is None
    assert out.json_events_stdout == (_TURN_FAILED_JSONL if retain else None)


@pytest.mark.asyncio
async def test_run_codex_json_trace_needs_nobody_to_turn_it_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nothing asks for the trace anywhere -- not the input, not task_config -- and it still refines
    # the reason, because a mixed-scaffold run should not have to carry a codex-only switch.
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': '',
            'exit_code': 1,
            'execution_time': 3.0,
            'timed_out': False,
            'json_events_stdout': _TURN_FAILED_JSONL,
        },
    )

    await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert recorded['attributes']['hytuner.termination_reason'] == 'CODEX_TURN_FAILED'


@pytest.mark.asyncio
async def test_run_codex_json_trace_off_via_input_keeps_coarse_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    # Opting out on the input: even with json stdout present, the exit code is authoritative.
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': '',
            'exit_code': 1,
            'execution_time': 3.0,
            'timed_out': False,
            'json_events_stdout': _TURN_FAILED_JSONL,
        },
    )

    await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace', mode='exec'))

    assert recorded['attributes']['hytuner.termination_reason'] == 'UNHANDLED_ERROR'


@pytest.mark.asyncio
async def test_run_codex_task_mode_does_not_disable_default_json_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': '',
            'exit_code': 1,
            'execution_time': 3.0,
            'timed_out': False,
            'json_events_stdout': _TURN_FAILED_JSONL,
        },
    )

    async with ExecutionContext(task_config={'mode': 'exec'}):
        await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace', use_azure_provider=False))

    assert recorded['attributes']['hytuner.termination_reason'] == 'CODEX_TURN_FAILED'
    assert recorded['runtime_kwargs']['mode'] == 'exec_json'


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['exec', 'exec_json', 'app_server'])
@pytest.mark.parametrize('task_mode', ['agent', 'oracle', 'nop', 'exec', 'exec_json', 'app_server'])
async def test_run_codex_task_mode_does_not_override_input_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str, task_mode: str
) -> None:
    recorded = _install_fakes(monkeypatch, runtime_result={'success': True, 'exit_code': 0})

    async with ExecutionContext(task_config={'mode': task_mode}):
        await run_codex(
            CodexInput(user_prompt='task', working_dir='/workspace', mode=mode, use_azure_provider=False)
        )

    assert recorded['runtime_kwargs']['mode'] == mode


@pytest.mark.asyncio
@pytest.mark.parametrize('task_mode', ['agent', 'exec'])
async def test_run_codex_task_mode_does_not_override_goal_mode(
    monkeypatch: pytest.MonkeyPatch, task_mode: str
) -> None:
    recorded = _install_fakes(monkeypatch, runtime_result={'success': True, 'exit_code': 0})

    async with ExecutionContext(task_config={'mode': task_mode}):
        await run_codex(
            CodexInput(
                user_prompt='goal',
                working_dir='/workspace',
                mode='app_server',
                goal_mode=True,
                goal_token_budget=1000,
                version='0.154.0',
                use_azure_provider=False,
            )
        )

    assert recorded['runtime_kwargs']['mode'] == 'app_server'
    assert recorded['runtime_kwargs']['goal_mode'] is True
    assert recorded['runtime_kwargs']['goal_token_budget'] == 1000


@pytest.mark.parametrize('mode', ['agent', 'oracle', 'nop', 'unknown'])
def test_codex_input_rejects_invalid_execution_mode(mode: str) -> None:
    with pytest.raises(ValueError, match='Input should be'):
        CodexInput(user_prompt='task', working_dir='/workspace', mode=mode)


# --- a model call refused on token budget ----------------------------------------------


def _turn_failed(message: str) -> str:
    return json.dumps({'type': 'turn.failed', 'error': {'message': message}})


# Verbatim shape codex 0.136/0.142 record for the proxy's 400: the whole body lands in the message
# as opaque text. Captured by running `codex exec --json` against a proxy returning that response.
_REFUSAL_MESSAGE = json.dumps({
    'error': {
        'message': '[context_length_exceeded] prompt is 200k tokens over the limit',
        'type': 'invalid_request_error',
        'code': 'context_length_exceeded',
    }
})

_REFUSED_TURN_FAILED_JSONL = '\n'.join([
    '{"type":"thread.started","thread_id":"t1"}',
    '{"type":"turn.started"}',
    _turn_failed(_REFUSAL_MESSAGE),
])

# Refused early, codex compacted and carried on, then died of something else.
_REFUSED_THEN_UNRELATED_JSONL = '\n'.join([
    '{"type":"thread.started","thread_id":"t1"}',
    _turn_failed(_REFUSAL_MESSAGE),
    '{"type":"turn.started"}',
    _turn_failed("We're currently experiencing high demand, which may cause temporary errors."),
])


async def test_run_codex_reports_truncation_when_the_fatal_turn_names_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': '',
            'exit_code': 1,
            'execution_time': 3.0,
            'timed_out': False,
            'json_events_stdout': _REFUSED_TURN_FAILED_JSONL,
        },
    )

    out = await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert out.exit_status == 'truncation'
    assert out.failure_kind == 'agent'
    assert recorded['attributes']['hytuner.termination_reason'] == 'TRUNCATION'


async def test_run_codex_reports_the_failure_it_actually_died_of(monkeypatch: pytest.MonkeyPatch) -> None:
    """An early refusal codex recovered from must not relabel a later, unrelated failure."""
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': '',
            'exit_code': 1,
            'execution_time': 3.0,
            'timed_out': False,
            'json_events_stdout': _REFUSED_THEN_UNRELATED_JSONL,
            'last_llm_error_code': None,
        },
    )

    await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert recorded['attributes']['hytuner.termination_reason'] == 'CODEX_TURN_FAILED'


async def test_run_codex_falls_back_when_the_fatal_turn_is_uninformative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An old in-sandbox proxy still 500s, and codex replaces a 5xx body with a canned message.

    The stream then names a failure without naming its cause, so the reason it yields must not
    shadow what the host knows.
    """
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': '',
            'exit_code': 1,
            'execution_time': 3.0,
            'timed_out': False,
            'json_events_stdout': '\n'.join([
                '{"type":"thread.started","thread_id":"t1"}',
                '{"type":"turn.started"}',
                _turn_failed("We're currently experiencing high demand, which may cause temporary errors."),
            ]),
            'last_llm_error_code': 'context_length_exceeded',
        },
    )

    await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert recorded['attributes']['hytuner.termination_reason'] == 'TRUNCATION'


async def test_run_codex_falls_back_to_the_host_when_the_stream_says_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No --json stream at all (or an old proxy that does not prefix the code): the host's
    # last-settled call is the only remaining evidence.
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': '',
            'exit_code': 1,
            'execution_time': 3.0,
            'timed_out': False,
            'last_llm_error_code': 'context_length_exceeded',
        },
    )

    await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert recorded['attributes']['hytuner.termination_reason'] == 'TRUNCATION'


async def test_run_codex_host_fallback_ignores_a_settled_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression: a refused call followed by a successful one is not a truncated run."""
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': '',
            'exit_code': 1,
            'execution_time': 3.0,
            'timed_out': False,
            'llm_request_outcomes': {'r1': 'context_length_exceeded', 'r2': None},
            'last_llm_error_code': None,
        },
    )

    await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert recorded['attributes']['hytuner.termination_reason'] == 'UNHANDLED_ERROR'


async def test_run_codex_host_fallback_never_recasts_a_clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Codex recovered from the refusal and finished; success stays authoritative.
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 3.0,
            'timed_out': False,
            'last_llm_error_code': 'context_length_exceeded',
        },
    )

    await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert recorded['attributes']['hytuner.termination_reason'] == 'ENV_DONE'


def _clean_exit_result(output: str, **last_call: object) -> dict[str, object]:
    return {
        'success': True,
        'output': output,
        'stderr': '',
        'exit_code': 0,
        'execution_time': 3.0,
        'timed_out': False,
        'last_llm_error_code': None,
        **last_call,
    }


@pytest.mark.parametrize(
    ('last_call', 'termination_reason', 'exit_status'),
    [
        pytest.param(
            {'last_llm_finish_reason': 'stop', 'last_llm_unparsed_tool_call': False},
            'ENV_DONE',
            'completed',
            id='real-answer',
        ),
        pytest.param({}, 'ENV_DONE', 'completed', id='old-runtime-without-last-call-facts'),
        pytest.param(
            {'last_llm_finish_reason': 'length', 'last_llm_unparsed_tool_call': False},
            'TRUNCATION',
            'truncation',
            id='cut-at-max-tokens',
        ),
        pytest.param(
            {'last_llm_finish_reason': 'stop', 'last_llm_unparsed_tool_call': True},
            'FORMAT_ERROR',
            'format_error',
            id='unparsed-tool-call',
        ),
        pytest.param(
            {'last_llm_finish_reason': 'length', 'last_llm_unparsed_tool_call': True},
            'TRUNCATION',
            'truncation',
            id='length-cut-wins-over-the-tool-call-it-interrupted',
        ),
    ],
)
async def test_run_codex_relabels_a_clean_exit_from_its_last_model_call(
    monkeypatch: pytest.MonkeyPatch,
    last_call: dict[str, object],
    termination_reason: str,
    exit_status: str,
) -> None:
    recorded = _install_fakes(monkeypatch, runtime_result=_clean_exit_result('final text', **last_call))

    out = await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert out.termination_reason == termination_reason
    assert out.exit_status == exit_status
    assert out.final_answer == ('final text' if exit_status == 'completed' else None)
    assert recorded['attributes']['hytuner.termination_reason'] == termination_reason
    assert agent_termination_reason(build_t3_agent_result(mode='agent', agent_output=out)) == termination_reason, (
        'Harbor reads only exit_status, so it must recover the same termination reason'
    )


async def test_run_codex_last_call_relabel_leaves_a_failed_run_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            **_clean_exit_result('', last_llm_finish_reason='length', last_llm_unparsed_tool_call=True),
            'success': False,
            'exit_code': 1,
        },
    )

    await run_codex(CodexInput(user_prompt='fix bug', working_dir='/workspace'))

    assert recorded['attributes']['hytuner.termination_reason'] == 'UNHANDLED_ERROR'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('task_config', 'disable_invoke_wait_timeout'),
    [
        pytest.param({'is_validation': False}, True, id='train'),
        pytest.param({'is_validation': True}, False, id='validation'),
        pytest.param({}, False, id='unspecified'),
        pytest.param({'is_validation': None}, False, id='unknown'),
    ],
)
async def test_run_codex_scopes_runtime_timeout_control_to_the_active_call(
    monkeypatch: pytest.MonkeyPatch,
    task_config: dict[str, object],
    disable_invoke_wait_timeout: bool,
) -> None:
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
            'timed_out': False,
        },
    )
    context = ExecutionContext(task_config=task_config)

    async with context:
        await run_codex(CodexInput(user_prompt='fix timeout', working_dir='/workspace', use_azure_provider=False))
        assert context.runtime_command_timeout_id is None

    runtime_trace_context = recorded['runtime_kwargs']['runtime_trace_context']
    timeout_control_id = runtime_trace_context.parent_span_id
    assert isinstance(timeout_control_id, str)
    assert timeout_control_id
    assert recorded['context_timeout_id'] == timeout_control_id
    assert recorded['runtime_kwargs']['disable_invoke_wait_timeout'] is disable_invoke_wait_timeout


@pytest.mark.asyncio
async def test_run_codex_without_context_preserves_invoke_wait_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_fakes(monkeypatch, runtime_result={'success': True, 'output': 'done', 'exit_code': 0})

    await run_codex(CodexInput(user_prompt='fix timeout', working_dir='/workspace'))

    assert recorded['runtime_kwargs']['disable_invoke_wait_timeout'] is False


@pytest.mark.asyncio
async def test_run_codex_clears_runtime_timeout_control_after_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, runtime_result={})

    async def fail_runtime_call(*_args: object, **_kwargs: object) -> dict[str, object]:
        assert require_context().runtime_command_timeout_id is not None
        raise RuntimeError('runtime failed')

    monkeypatch.setattr('nexus.extensions.agents.codex.agent.call_runtime_agent', fail_runtime_call)
    context = ExecutionContext()

    async with context:
        with pytest.raises(RuntimeError, match='runtime failed'):
            await run_codex(CodexInput(user_prompt='fix timeout', working_dir='/workspace', use_azure_provider=False))
        assert context.runtime_command_timeout_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('task_config', 'disable_invoke_wait_timeout'),
    [
        pytest.param({'is_validation': False}, True, id='train'),
        pytest.param({'is_validation': True}, False, id='validation'),
        pytest.param({}, False, id='unspecified'),
        pytest.param({'is_validation': None}, False, id='unknown'),
    ],
)
async def test_app_server_advertises_command_timeout_control(
    monkeypatch: pytest.MonkeyPatch,
    task_config: dict[str, object],
    disable_invoke_wait_timeout: bool,
) -> None:
    recorded = _install_fakes(monkeypatch, runtime_result={'success': True, 'exit_code': 0})
    context = ExecutionContext(task_config=task_config)
    async with context:
        await run_codex(CodexInput(user_prompt='task', working_dir='/workspace', mode='app_server'))
        assert context.runtime_command_timeout_id is None
    assert recorded['context_timeout_id'] == recorded['runtime_kwargs']['runtime_trace_context'].parent_span_id
    assert recorded['runtime_kwargs']['disable_invoke_wait_timeout'] is disable_invoke_wait_timeout


async def test_run_codex_forwards_binary_and_mcp_options(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_fakes(monkeypatch, runtime_result={'success': True, 'exit_code': 0})
    servers = [{'name': 'echo', 'command': 'uv', 'args': ['run', 'server.py']}]
    await run_codex(
        CodexInput(
            user_prompt='task',
            working_dir='/workspace',
            binary_url='https://example.invalid/codex.exe',
            binary_checksum='abc',
            mcp_servers=servers,
            extra_env={'MCP_TEST': 'yes'},
        )
    )
    kwargs = recorded['runtime_kwargs']
    assert kwargs['binary_url'] == 'https://example.invalid/codex.exe'
    assert kwargs['binary_checksum'] == 'abc'
    assert kwargs['mcp_servers'] == servers
    assert kwargs['extra_env'] == {'MCP_TEST': 'yes'}


@pytest.mark.asyncio
@pytest.mark.parametrize('retain', [True, False])
async def test_task_config_overrides_json_output_switches(monkeypatch, retain):
    recorded = _install_fakes(
        monkeypatch, runtime_result={'success': False, 'exit_code': 1, 'json_events_stdout': _TURN_FAILED_JSONL}
    )

    class FakeContext:
        runtime_command_timeout_id = None
        task_config = {'retain_json_events_stdout': retain}

    monkeypatch.setattr('nexus.core.require_context', lambda: FakeContext())
    output = await run_codex(
        CodexInput(
            user_prompt='inspect',
            working_dir='/workspace',
            retain_json_events_stdout=not retain,
        )
    )
    assert output.json_events_stdout == (_TURN_FAILED_JSONL if retain else None)
    assert recorded['attributes']['hytuner.termination_reason'] == 'CODEX_TURN_FAILED'
    assert 'emit_json_trace' not in recorded['runtime_kwargs']


@pytest.mark.asyncio
async def test_json_output_is_not_retained_by_default(monkeypatch):
    _install_fakes(
        monkeypatch, runtime_result={'success': True, 'exit_code': 0, 'json_events_stdout': _TURN_FAILED_JSONL}
    )
    output = await run_codex(CodexInput(user_prompt='inspect', working_dir='/workspace'))
    assert output.json_events_stdout is None


def test_codex_trace_switches_are_replaced_by_mode():
    from nexus.extensions.agents.codex.agent import CodexNativeBridgeInput

    assert 'emit_json_trace' not in CodexInput.model_fields
    assert 'emit_event_stream' not in CodexInput.model_fields
    assert CodexInput.model_fields['mode'].default == 'exec_json'
    assert CodexNativeBridgeInput.model_fields['emit_json_trace'].default is True


@pytest.mark.asyncio
@pytest.mark.parametrize('goal_mode', [False, True])
@pytest.mark.parametrize(
    ('error', 'exit_code', 'timed_out', 'failure_kind', 'reason'),
    [
        ({'codexErrorInfo': 'contextWindowExceeded', 'message': 'out of room'}, 1, False, 'agent', 'TRUNCATION'),
        ({'message': '[context_length_exceeded] rejected'}, 1, False, 'agent', 'TRUNCATION'),
        ({'message': 'tool failed'}, 1, False, 'agent', 'CODEX_TURN_FAILED'),
        ({'codexErrorInfo': 'contextWindowExceeded'}, -1, True, 'agent', 'TASK_TIMEOUT'),
        ({'codexErrorInfo': 'contextWindowExceeded'}, -1, False, 'infrastructure', 'CODEX_ERROR'),
    ],
)
async def test_app_server_classifies_runtime_facts_on_host(
    monkeypatch, goal_mode, error, exit_code, timed_out, failure_kind, reason
):
    _install_fakes(
        monkeypatch,
        runtime_result={
            'success': False,
            'output': '',
            'stderr': 'failure',
            'exit_code': exit_code,
            'execution_time': 1.0,
            'timed_out': timed_out,
            'json_events_stdout': '',
            'failure_kind': failure_kind,
            'failure_stage': 'process' if failure_kind == 'agent' else 'runtime_protocol',
            'event_stream': {
                'source': 'app_server',
                'events': [
                    {'seq': 0, 'type': 'thread.start.response', 'data': {'result': {'thread': {'id': 'root'}}}},
                    *(
                        [
                            {
                                'seq': 1,
                                'type': 'thread.goal.updated',
                                'data': {'threadId': 'root', 'goal': {'status': 'blocked'}},
                            }
                        ]
                        if goal_mode
                        else []
                    ),
                    {
                        'seq': 1 + int(goal_mode),
                        'type': 'turn.completed',
                        'data': {'threadId': 'root', 'turn': {'id': 'turn-1', 'status': 'failed', 'error': error}},
                    },
                    {
                        'seq': 2 + int(goal_mode),
                        'type': 'turn.completed',
                        'data': {
                            'threadId': 'child',
                            'turn': {'id': 'child-turn', 'status': 'completed', 'error': None},
                        },
                    },
                ],
            },
        },
    )
    out = await run_codex(
        CodexInput(user_prompt='test', working_dir='/workspace', mode='app_server', goal_mode=goal_mode)
    )
    if goal_mode and reason == 'CODEX_TURN_FAILED':
        reason = 'CODEX_GOAL_BLOCKED'
    assert out.termination_reason == reason
    assert out.exit_status == ('truncation' if reason == 'TRUNCATION' else 'timeout' if timed_out else 'error')
    assert out.failure_kind == failure_kind
    assert out.final_answer is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('config', 'message'),
    [
        ({'goal_mode': True, 'mode': 'exec_json', 'version': '0.154.0'}, 'requires mode'),
        ({'goal_token_budget': 100}, 'requires goal_mode'),
        ({'mode': 'app_server', 'logs_dir': '/logs/agent'}, 'does not support logs_dir'),
    ],
)
async def test_codex_rejects_unsupported_goal_configuration(monkeypatch, config, message):
    _install_fakes(monkeypatch, runtime_result={})
    with pytest.raises(ValueError, match=message):
        await run_codex(CodexInput(user_prompt='goal', working_dir='/workspace', **config))


@pytest.mark.asyncio
@pytest.mark.parametrize('version', ['0.147.0', '0.154.0'])
async def test_codex_passes_goal_configuration(monkeypatch, version):
    recorded = _install_fakes(monkeypatch, runtime_result={'success': True, 'exit_code': 0})
    await run_codex(
        CodexInput(
            user_prompt='goal',
            working_dir='/workspace',
            mode='app_server',
            version=version,
            goal_mode=True,
            goal_token_budget=1000,
        )
    )
    assert recorded['runtime_kwargs']['goal_mode'] is True
    assert recorded['runtime_kwargs']['goal_token_budget'] == 1000


@pytest.mark.asyncio
@pytest.mark.parametrize('logs_dir', [None, '/logs/agent'])
async def test_run_codex_passes_log_directory_only_when_requested(
    monkeypatch: pytest.MonkeyPatch, logs_dir: str | None
) -> None:
    recorded = _install_fakes(
        monkeypatch,
        runtime_result={
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
            'json_events_stdout': 'raw events',
        },
    )
    output = await run_codex(CodexInput(user_prompt='test', working_dir='/workspace', logs_dir=logs_dir))
    if logs_dir is None:
        assert 'logs_dir' not in recorded['runtime_kwargs']
    else:
        assert recorded['runtime_kwargs']['logs_dir'] == logs_dir
    assert output.json_events_stdout is None


@pytest.mark.asyncio
async def test_native_bridge_rejects_unsupported_log_export() -> None:
    from nexus.extensions.agents.codex.agent import CodexNativeBridgeInput, run_codex_native_bridge

    with pytest.raises(ValueError, match='does not support logs_dir'):
        await run_codex_native_bridge(
            CodexNativeBridgeInput(user_prompt='test', working_dir='/workspace', logs_dir='/logs/agent')
        )
