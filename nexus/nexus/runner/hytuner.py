"""HyTuner training runner for Nexus framework.

This runner provides an interface for HyTuner RL training:
- Takes HyTuner's server_manager and tokenizer as input
- Runs Nexus tasks (which internally run agents)
- Returns trajectory data in AgentLoopOutput-compatible format
- Collects task scores for reward calculation

Usage in HyTuner:
    from nexus.runner.hytuner import run, HyTunerRunInput

    # Example 1 with identity task
    run_input = HyTunerRunInput(
        imports=['nexus.extensions.projects.calculator',
        'nexus.extensions.tasks.identity',
        ],
        task_name='identity',
        agent_name='calculator_agent',
        agent_config={'user_prompt': [{"role": "user", "content": "Compute (7 @ 4 @ 9) @ 4 @ 3"}],},
    )

    # Example 2 with calculator task
    run_input = HyTunerRunInput(
        imports=['nexus.extensions.projects.calculator'],
        task_name='calculator_task',
        task_config={
            'user_prompt': 'Compute (7 @ 4 @ 9) @ 4 @ 3',
            'ground_truth': '159',
        },
        agent_name='calculator_agent',
        agent_config={'max_iterations': 10},
    )

    result = await run(
        run_input,
        tokenizer=tokenizer,
        server_manager=server_manager,
        sampling_params={'temperature': 1.0},
        abort_event=asyncio.Event(),
    )

    # Gateway mode returns a canonical Trajectory. Non-Gateway mode returns
    # the existing HyTunerRunOutput.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import traceback

from typing import Any, Literal

import numpy as np

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nexus.core.agent import get_agent_gateway_scaffold_policy
from nexus.core.context import require_context
from nexus.core.registry import Registry
from nexus.extensions.compaction.compaction_manager import CompactionManager
from nexus.gateway.trajectory import Trajectory
from nexus.gateway.trajectory_store import TrajectoryRecordStore
from nexus.runner import single
from nexus.runner.single import ProcessorConfig
from nexus.utils.token_budget_config import parse_token_budget_include


logger = logging.getLogger(__name__)


class HyTunerRewardPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    mode: Literal['binary']
    threshold: float


class HyTunerTokenBudgetConfig(BaseModel):
    """Per-row trajectory token budget enforced by the Gateway session."""

    model_config = ConfigDict(extra='forbid')

    token_budget: int | None = Field(default=None, gt=0)
    token_budget_include: list[str] | None = None

    @field_validator('token_budget_include')
    @classmethod
    def _validate_token_budget_include(cls, value: list[str] | None) -> list[str] | None:
        if value is not None:
            parse_token_budget_include(value)
        return value


class HyTunerRunInput(single.RunInput):
    token_budget_config: HyTunerTokenBudgetConfig | None = None


def _gateway_token_budget(run_input: HyTunerRunInput) -> tuple[int | None, list[str] | None]:
    """Return the row's trajectory budget and the token sources charged against it."""
    row_budget = run_input.token_budget_config
    if row_budget is None:
        return None, None
    return row_budget.token_budget, row_budget.token_budget_include


class HyTunerSegment(BaseModel):
    """A single segment of a training trajectory.

    When context management is disabled, a trajectory has exactly one segment.
    When enabled, compaction splits the trajectory into multiple segments,
    each with its own prompt/response token-level data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: list[int]
    raw_input_ids: list[int] = []
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None = None
    num_turns: int = 0
    segment_type: str = 'conversation'  # 'conversation' or 'summary_inference'
    # Physical restart cause when the gateway split this segment off without a declared
    # boundary (e.g. 'render_divergence'); None for declared/compaction segments.
    split_cause: str | None = None
    sidechain_info: dict[str, Any] | None = None
    multi_modal_data: dict[str, Any] = {}
    routed_experts: np.ndarray | None = None  # R3: this segment's whole-sequence routing
    extra_fields: dict[str, Any] = Field(default_factory=dict)


class HyTunerRunOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # All segments of the trajectory in chronological order.
    # Without compaction this is a single-element list.
    # With compaction the last element is the "main" (final) segment.
    segments: list[HyTunerSegment] = []

    routed_experts: np.ndarray | None = None  # R3: MoE expert routing decisions from vLLM
    reward_score: float | None = None
    termination_reason: str | None = None
    metrics: dict[str, Any] = {}
    extra_fields: dict[str, Any] = {}

    # Error handling fields (propagated from single.RunOutput)
    success: bool = True
    error: single.ErrorInfo | None = None
    request_id: str | None = None


def _gateway_template_kwargs(main_config: dict[str, Any], defaults: dict[str, Any] | None) -> dict[str, Any]:
    """Use the same per-row template options as standalone OpenAI inference."""
    result = dict(defaults or {})
    for section in ('request_args', 'enforced_request_args'):
        extra_body = (main_config.get(section) or {}).get('extra_body') or {}
        result.update(extra_body.get('chat_template_kwargs') or {})
    return result


def _scoreable_hycode_exit_name(run_input: HyTunerRunInput, result: single.RunOutput) -> str | None:
    if run_input.agent_name not in {'hycode', 'hyness'} or result.agent_output is None:
        return None
    if getattr(result.agent_output, 'termination_reason', None) != 'ENV_DONE':
        return None
    if not getattr(result.agent_output, 'scoreable_process_exit', False):
        return None
    name = getattr(result.agent_output, 'hycode_exit_name', None)
    return name if isinstance(name, str) and name else None


def _task_reward_or_scoreable_zero(result: single.RunOutput, scoreable: bool) -> float | None:
    reward = getattr(result.task_output, 'score', None) if result.task_output else None
    return 0.0 if reward is None and scoreable else reward


def _reward_policy_from_input(run_input: HyTunerRunInput) -> HyTunerRewardPolicy | None:
    raw_policy = run_input.task_config.get('hytuner_reward_policy')
    return None if raw_policy is None else HyTunerRewardPolicy.model_validate(raw_policy)


def _apply_hytuner_reward_policy(
    score: float | int | None,
    policy: HyTunerRewardPolicy | None,
    extra_fields: dict[str, Any] | None = None,
) -> float | None:
    if score is None:
        return None
    numeric_score = float(score)
    if policy is None:
        return numeric_score
    bad_pattern = (extra_fields or {}).get('tool_call_bad_pattern')
    if isinstance(bad_pattern, dict) and bad_pattern.get('penalized'):
        original_score = float(bad_pattern['original_score'])
        penalty = float(bad_pattern['penalty'])
        return max((1.0 if original_score >= policy.threshold else 0.0) - penalty, 0.0)
    return 1.0 if numeric_score >= policy.threshold else 0.0


def _resolve_termination_reason(
    reported_reason: str | None,
    result: single.RunOutput,
    *,
    scoreable_hycode_exit: bool = False,
    gateway_truncated: bool = False,
) -> str:
    """Prefer a task-reported outcome, falling back only when it is absent or contradictory.

    ``gateway_truncated`` is the Gateway's own record that the last main-chain call hit a length
    limit. It applies only when the agent reported nothing, so agents that report their own exit
    (main_agent) keep it, and every other agent is classified without code of its own.
    """
    if reported_reason and (result.success or reported_reason != 'ENV_DONE'):
        return reported_reason
    if reported_reason is None and gateway_truncated:
        return 'TRUNCATION'
    if result.success or scoreable_hycode_exit:
        return reported_reason or 'ENV_DONE'
    if result.error is not None and result.error.type == 'TimeoutError':
        return 'TASK_TIMEOUT'
    return 'UNHANDLED_ERROR'


def _enable_tool_error_wrapping_for_segment_shaping(
    run_input: HyTunerRunInput,
    segment_shaping_config: dict[str, Any],
) -> None:
    policy = segment_shaping_config.get('policy', segment_shaping_config.get('shaping_policy'))
    if run_input.agent_name == 'main_agent' and policy == 'tool_error_turn_penalty_only':
        run_input.agent_config = dict(run_input.agent_config or {})
        run_input.agent_config.setdefault('wrap_failed_tool_response_as_tool_error', True)


def _configure_codex_training_compaction(run_input: HyTunerRunInput) -> None:
    """Keep Codex compaction on HyTuner's trainable local-summary path."""
    if run_input.agent_name != 'codex':
        return
    agent_config = dict(run_input.agent_config or {})
    agent_config.setdefault('use_azure_provider', False)
    run_input.agent_config = agent_config


async def run(
    run_input: HyTunerRunInput,
    *,
    tokenizer: Any,
    server_manager: Any,
    sampling_params: dict[str, Any],
    tool_parser: Any,
    template_parser: Any = None,
    processor: Any = None,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
    use_gateway: bool = False,
    record_store: TrajectoryRecordStore | None = None,
    **kwargs: Any,
) -> HyTunerRunOutput | Trajectory:
    _configure_codex_training_compaction(run_input)
    # Standalone `nexus run --use-gateway` injects gateway_trajectory, which finalizes
    # the session inside the processor scope. This runner finalizes from the rollout
    # key in on_run_output, so that injection must stay off or the trajectory is emptied.
    run_input.runner.use_gateway = False
    run_input.processors = {
        key: processor for key, processor in run_input.processors.items() if processor.name != 'gateway_trajectory'
    }
    # Validation reports the score the policy earns, so task-level reward overrides
    # must be disabled consistently before dispatching to either runner path.
    run_input.task_config['is_validation'] = bool(kwargs.get('validate', False))
    if use_gateway:
        if record_store is None:
            raise ValueError('use_gateway=True requires record_store')
        trajectory, result = await _execute_gateway(
            run_input,
            tokenizer=tokenizer,
            server_manager=server_manager,
            sampling_params=sampling_params,
            tool_parser=tool_parser,
            template_parser=template_parser,
            processor=processor,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            record_store=record_store,
            **kwargs,
        )
        compaction_stats = getattr(run_input.compaction_manager, 'compaction_stats', None)
        token_budget, _ = _gateway_token_budget(run_input)
        _attach_hytuner_execution_metadata(
            trajectory,
            result,
            run_input=run_input,
            compaction_stats=compaction_stats,
            token_budget=token_budget,
        )
        return trajectory

    logger.debug(f'Starting HyTuner runner for task: {run_input.task_name}')
    reward_policy = _reward_policy_from_input(run_input)

    # make sure related modules exist
    Registry.import_module('nexus.trace.processors.hytuner')
    Registry.import_module('nexus.extensions.openai_clients.hytuner')

    process_vision_info_fn = kwargs.get('process_vision_info_fn')

    train_token_budget = kwargs.get('context_management', {}).get('token_budget', None)
    val_token_budget = kwargs.get('context_management', {}).get('val_token_budget', None)
    token_budget = train_token_budget if not kwargs['validate'] else val_token_budget
    token_budget_include = kwargs.get('context_management', {}).get('token_budget_include')

    existing_main_config = run_input.openai_clients.get('main', {}).get('config', {})
    enforced_request_args = existing_main_config.get('enforced_request_args', {})
    run_input.openai_clients['main'] = {
        'name': 'hytuner',
        'config': {
            'client_args': {
                'tokenizer': tokenizer,
                'processor': processor,
                'server_manager': server_manager,
                'tool_parser': tool_parser,
                'template_parser': template_parser,
                'apply_chat_template_kwargs': apply_chat_template_kwargs or {},
                'abort_event': abort_event,
                'priority': kwargs.get('priority', 0),
                'max_model_len': kwargs.get('max_model_len'),
                'token_budget': token_budget,
                'token_budget_include': token_budget_include,
                'enable_rollout_routing_replay': kwargs.get('enable_rollout_routing_replay', False),
                'process_vision_info_fn': process_vision_info_fn,
                **(kwargs.get('extra_hytuner_client_args') or {}),
            },
            'request_args': {
                'sampling_params': sampling_params,
            },
            'enforced_request_args': enforced_request_args,
        },
    }

    # --- Build CompactionManager at runner level ---
    context_management_cfg = kwargs.get('context_management', {})
    max_tool_response_tokens = context_management_cfg.get('max_tool_response_tokens')
    if run_input.agent_name == 'main_agent' and max_tool_response_tokens is not None:
        if max_tool_response_tokens <= 0:
            raise ValueError(
                f'context_management.max_tool_response_tokens must be positive, got {max_tool_response_tokens}'
            )
        run_input.agent_config = dict(run_input.agent_config or {})
        run_input.agent_config.setdefault('max_tool_response_tokens', max_tool_response_tokens)
    if context_management_cfg:

        async def _token_count_fn(messages: list[dict], tools: Any) -> int:
            ctx = require_context()
            result = await ctx.openai_client.count_tokens(messages=messages, tools=tools)
            return result.input_tokens

        condition_kwargs: dict[str, Any] = dict(context_management_cfg.get('condition', {}))
        strategy_kwargs: dict[str, Any] = dict(context_management_cfg.get('strategy', {}))

        run_input.compaction_manager = CompactionManager(
            strategy_name=context_management_cfg.get('compaction_strategy', 'keep_recent'),
            condition_name=context_management_cfg.get('compaction_condition', 'always_not'),
            target_tokens=context_management_cfg.get('target_tokens', 4096),
            max_segments=context_management_cfg.get('max_segments', None),
            token_count_fn=_token_count_fn,
            sampling_params=sampling_params,
            condition_kwargs=condition_kwargs,
            inject_trajectory_budget_state=context_management_cfg.get('inject_trajectory_budget_state', False),
            **strategy_kwargs,
        )
    file_path = kwargs.get('agent_save_path')
    trajectory_id = kwargs.get('trajectory_id')
    progress_state = kwargs.get('progress_state')
    processor_args: dict[str, Any] = {'file_path': file_path, 'trajectory_id': trajectory_id}
    if progress_state is not None:
        processor_args['progress_state'] = progress_state
    # HyTuner enables this only when algorithm.reward_shaping_mode='segment'.
    # When absent, the processor keeps baseline output unchanged.
    segment_shaping_config = kwargs.get('segment_shaping_config') or {}
    if segment_shaping_config:
        processor_args['segment_shaping_config'] = segment_shaping_config
        _enable_tool_error_wrapping_for_segment_shaping(run_input, segment_shaping_config)
    run_input.processors['hytuner'] = ProcessorConfig(name='hytuner', args=processor_args)

    if trajectory_id:
        run_input.task_config['traj_id'] = trajectory_id
    # run
    try:
        result = await single.run(run_input)
    except Exception as e:
        logger.error(f'single.run raised unexpected exception: {e}\n{traceback.format_exc()}')
        raise

    # results
    trajectory = result.processor_results.get('hytuner', {})
    if not trajectory:
        logger.warning(
            f'hytuner processor returned empty result. '
            f'processor_results keys: {list(result.processor_results.keys())}, '
            f'success: {result.success}, error: {result.error}'
        )
    hycode_exit_name = _scoreable_hycode_exit_name(run_input, result)
    scoreable_hycode_exit = hycode_exit_name is not None
    termination_reason = _resolve_termination_reason(
        trajectory.get('termination_reason'),
        result,
        scoreable_hycode_exit=scoreable_hycode_exit,
    )

    extra_fields = {}
    if result.task_output:
        task_extra_fields = getattr(result.task_output, 'extra_fields', None)
        if task_extra_fields:
            extra_fields.update(task_extra_fields)
    if trajectory.get('runtime_id', None):
        extra_fields['runtime_id'] = trajectory['runtime_id']
    if trajectory.get('runtime_ids'):
        extra_fields['runtime_ids'] = trajectory['runtime_ids']
    if trajectory.get('eval_runtime_id', None):
        extra_fields['eval_runtime_id'] = trajectory['eval_runtime_id']

    # Pass through per-tool call statistics from the processor
    extra_fields['tool_stats'] = trajectory.get('tool_stats', {})

    # Pass through agent timeline
    extra_fields['agent_timeline'] = trajectory.get('agent_timeline', [])

    compaction_stats = getattr(run_input.compaction_manager, 'compaction_stats', None)
    if compaction_stats:
        extra_fields['compaction_stats'] = compaction_stats

    llm_summary_traces = getattr(run_input.compaction_manager, 'llm_summary_traces', None)
    if llm_summary_traces:
        extra_fields['llm_summary_traces'] = llm_summary_traces

    trajectory_fields = trajectory.get('extra_fields')
    if trajectory_fields:
        extra_fields.update(trajectory_fields)
    if hycode_exit_name is not None:
        extra_fields['hycode_exit_name'] = hycode_exit_name
        assert result.agent_output is not None
        extra_fields['hycode_exit_code'] = result.agent_output.exit_code

    async def _segment_multi_modal_data(seg_messages: list[dict]) -> dict[str, Any]:
        """Extract per-segment multimodal payload from the segment's messages.

        Each segment carries the messages that fed its last completion, so we
        can reconstruct the exact images/videos/audios that segment's tokens
        reference. This keeps multimodal trajectories correct under context
        compaction: prior segments retain their original images even after the
        live history is rewritten.

        Messages contain the complete history including tool-returned images
        (e.g. base64 image_url from image_zoom_in_tool).
        """
        if not seg_messages or processor is None or process_vision_info_fn is None:
            return {}
        return await process_vision_info_fn(seg_messages)

    # Build segments list: prior segments (from compaction) + main segment (last).
    # Each segment owns its own multimodal payload, derived from the messages
    # snapshot the processor recorded at finalize time. The top-level trajectory
    # `messages` corresponds to the final (main) segment's history.
    # Vision extraction across segments is independent — gather concurrently.
    raw_prior_segments = trajectory.get('segments', [])
    main_messages = trajectory.get('messages') or []
    multi_modal_data_list = await asyncio.gather(
        *(_segment_multi_modal_data(seg.get('messages') or []) for seg in raw_prior_segments),
        _segment_multi_modal_data(main_messages),
    )
    prior_segments: list[HyTunerSegment] = [
        HyTunerSegment(
            prompt_ids=seg.get('prompt_ids', []),
            raw_input_ids=seg.get('raw_input_ids', []),
            response_ids=seg.get('response_ids', []),
            response_mask=seg.get('response_mask', []),
            response_logprobs=seg.get('response_logprobs'),
            num_turns=seg.get('num_turns', 0),
            segment_type=seg.get('segment_type', 'conversation'),
            sidechain_info=seg.get('sidechain_info'),
            multi_modal_data=seg_mm,
            routed_experts=seg.get('routed_experts'),
            extra_fields=seg.get('extra_fields', {}) or {},
        )
        for seg, seg_mm in zip(raw_prior_segments, multi_modal_data_list[:-1], strict=True)
    ]
    main_multi_modal_data = multi_modal_data_list[-1]
    main_segment_extra_fields = {}
    if isinstance(trajectory_fields, dict) and 'effective_response_limit' in trajectory_fields:
        main_segment_extra_fields['effective_response_limit'] = trajectory_fields['effective_response_limit']
    main_segment = HyTunerSegment(
        prompt_ids=trajectory.get('prompt_ids', []),
        raw_input_ids=trajectory.get('raw_input_ids', []),
        response_ids=trajectory.get('response_ids', []),
        response_mask=trajectory.get('response_mask', []),
        response_logprobs=trajectory.get('response_logprobs'),
        num_turns=trajectory.get('num_turns', 0),
        segment_type='conversation',
        sidechain_info=trajectory.get('sidechain_info'),
        multi_modal_data=main_multi_modal_data,
        routed_experts=trajectory.get('routed_experts'),
        extra_fields=main_segment_extra_fields,
    )
    segments = prior_segments
    if len(main_segment.prompt_ids) > 0 or len(prior_segments) == 0:
        segments.append(main_segment)

    return HyTunerRunOutput(
        segments=segments,
        routed_experts=trajectory.get('routed_experts'),
        # -300 is ReasoningReward.TASK_FAILED: a prefix reset desynced the recorded tokens from the task's reward.
        reward_score=-300.0
        if trajectory.get('unhandled_prefix_reset')
        else _apply_hytuner_reward_policy(
            _task_reward_or_scoreable_zero(result, scoreable_hycode_exit),
            reward_policy,
            extra_fields,
        ),
        termination_reason=termination_reason,
        metrics=trajectory.get('metrics', {}),
        extra_fields=extra_fields,
        success=result.success or scoreable_hycode_exit,
        error=result.error,
        request_id=trajectory.get('request_id', None),
    )


async def _execute_gateway(
    run_input: HyTunerRunInput,
    *,
    tokenizer: Any,
    server_manager: Any,
    sampling_params: dict[str, Any],
    tool_parser: Any,
    template_parser: Any = None,
    processor: Any = None,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
    record_store: TrajectoryRecordStore,
    **kwargs: Any,
) -> tuple[Trajectory, single.RunOutput]:
    """Execute and finalize one HyTuner run through a per-trajectory Gateway client."""
    logger.debug(f'Starting HyTuner runner (gateway) for task: {run_input.task_name}')
    reward_policy = _reward_policy_from_input(run_input)

    run_input.task_config['is_validation'] = bool(kwargs.get('validate', False))
    unsupported_generation_fields = [name for name in ('video_data', 'audio_data') if kwargs.get(name)]
    if unsupported_generation_fields:
        raise NotImplementedError(
            'Gateway V2 supports images only; unsupported generation field(s): '
            + ', '.join(unsupported_generation_fields)
        )

    from nexus.gateway.nexus_run_tracker import NexusRunTracker
    from nexus.gateway.session_manager import session_manager

    trajectory_id = kwargs.get('trajectory_id')
    if not trajectory_id:
        raise ValueError('use_gateway=True requires trajectory_id')
    traj_id = str(trajectory_id)
    if kwargs.get('context_management'):
        raise ValueError(
            'Gateway runner does not take context_management; set nexus.agent_config.compaction and '
            'nexus.token_budget_config in the row'
        )
    compaction_cfg = (run_input.agent_config or {}).get('compaction') or {}
    unsupported_context_management_options: list[str] = []
    if compaction_cfg.get('inject_trajectory_budget_state'):
        unsupported_context_management_options.append('agent_config.compaction.inject_trajectory_budget_state')

    if compaction_cfg.get('compaction_strategy', 'keep_recent') == 'self_summarize_v2':
        strategy_cfg = compaction_cfg.get('strategy', {}) or {}
        for option in (
            'self_summary_apply_chat_template_kwargs',
            'summarize_dump_io',
        ):
            if strategy_cfg.get(option):
                unsupported_context_management_options.append(f'agent_config.compaction.strategy.{option}')

    if unsupported_context_management_options:
        options = ', '.join(unsupported_context_management_options)
        raise NotImplementedError(f'Gateway runner does not support these context-management options: {options}')

    if kwargs.get('segment_shaping_config'):
        raise NotImplementedError('Gateway runner does not support segment_shaping_config')

    token_budget, token_budget_include = _gateway_token_budget(run_input)
    existing_main_config = run_input.openai_clients.get('main', {}).get('config', {})
    existing_client_args = existing_main_config.get('client_args', {}) or {}
    configured_policy = existing_client_args.get('scaffold_policy')
    scaffold_policy = (
        get_agent_gateway_scaffold_policy(run_input.agent_name, configured_policy)
        if configured_policy is None or isinstance(configured_policy, str)
        else configured_policy
    )
    enforced_request_args = dict(existing_main_config.get('enforced_request_args') or {})
    if 'logprobs' in sampling_params:
        enforced_request_args['logprobs'] = sampling_params['logprobs']
    run_input.openai_clients['main'] = {
        'name': 'gateway',
        'config': {
            'client_args': {
                'tokenizer': tokenizer,
                'processor': processor,
                'server_manager': server_manager,
                'tool_parser': tool_parser,
                'template_parser': template_parser,
                # ***** FOLLOWS HyTunerClient — rendering parity: it hardcodes
                # is_training=True when applying the chat template (source:
                # nexus/extensions/openai_clients/hytuner.py:712); the gateway
                # Processor must render identically or token streams diverge.
                'apply_chat_template_kwargs': {
                    **_gateway_template_kwargs(existing_main_config, apply_chat_template_kwargs),
                    'is_training': True,
                },
                # ***** END FOLLOWS (rendering parity — the single line above).
                # NOTE(arlenxu): passed whole from the HyTuner side, which owns the
                # model-family details nexus must not import.
                'mm_plugin': kwargs.get('mm_plugin'),
                'priority': kwargs.get('priority', 0),
                'enable_rollout_routing_replay': kwargs.get('enable_rollout_routing_replay', False),
                'trajectory_id': traj_id,
                'record_store': record_store,
                'scaffold_policy': scaffold_policy,
                # The metadata doubles as the per-request ``request_meta`` fallback
                # the client forwards to the generation endpoint on every completion;
                # ``validate`` rides it to select the train/val channel
                # (generate_tokens raises if absent).
                'metadata': {'trajectory_id': traj_id, 'validate': kwargs.get('validate', False)},
                'max_model_len': kwargs.get('max_model_len'),
                'token_budget': token_budget,
                'token_budget_include': token_budget_include,
            },
            'request_args': {'sampling_params': sampling_params},
            'enforced_request_args': enforced_request_args,
        },
    }

    if trajectory_id:
        run_input.task_config['traj_id'] = trajectory_id

    run_tracker = NexusRunTracker(kwargs.get('progress_state'), trajectory_id=traj_id)
    run_tracker.task_name = run_input.task_name
    finalized_trajectory: Trajectory | None = None
    last_run_output: single.RunOutput | None = None

    async def finalize(output: single.RunOutput | None, *, termination_reason: str) -> Trajectory | None:
        nonlocal finalized_trajectory
        if finalized_trajectory is not None:
            return finalized_trajectory
        task_output = output.task_output if output is not None else None
        hycode_exit_name = _scoreable_hycode_exit_name(run_input, output) if output is not None else None
        scoreable_hycode_exit = hycode_exit_name is not None
        task_extra = dict(getattr(task_output, 'extra_fields', None) or {}) if task_output else {}
        if output is not None and hycode_exit_name is not None:
            task_extra['hycode_exit_name'] = hycode_exit_name
            assert output.agent_output is not None
            task_extra['hycode_exit_code'] = output.agent_output.exit_code
        # Unconditional: the scaffold has exited, so no run outcome leaves a consumer behind.
        if session_manager.has_session(traj_id):
            await session_manager.abort_in_flight(traj_id)
        session = session_manager.get_session(traj_id)
        if session is not None:
            await session.finish_pending_publications()
        raw_trajectory = (
            session_manager.end_session(
                traj_id,
                termination_reason=termination_reason,
            )
            if session_manager.has_session(traj_id)
            else None
        )
        finalized_trajectory = run_tracker.finalize_trajectory(
            raw_trajectory,
            traj_id,
            termination_reason=termination_reason,
            final_reward=(
                _apply_hytuner_reward_policy(
                    _task_reward_or_scoreable_zero(output, scoreable_hycode_exit),
                    reward_policy,
                    task_extra,
                )
                if output is not None
                else None
            ),
            extra=task_extra or None,
            allow_empty=output is not None,
        )
        return finalized_trajectory

    def gateway_truncated() -> bool:
        session = session_manager.get_session(traj_id)
        return session is not None and session.last_main_call_truncated

    async def capture_and_finalize(context: Any, output: single.RunOutput) -> None:
        nonlocal last_run_output
        last_run_output = output
        # Agents such as main_agent may install their own manager on the execution context.
        run_input.compaction_manager = run_input.compaction_manager or getattr(context, 'compaction_manager', None)
        try:
            await run_tracker.capture_run_output(context, output)
        finally:
            scoreable = _scoreable_hycode_exit_name(run_input, output) is not None
            termination_reason = _resolve_termination_reason(
                run_tracker.termination_reason,
                output,
                scoreable_hycode_exit=scoreable,
                gateway_truncated=gateway_truncated(),
            )
            await finalize(output, termination_reason=termination_reason)

    async def finalize_on_context_closing(_context: Any) -> None:
        await finalize(None, termination_reason='UNHANDLED_ERROR')

    hooks = single.RunHooks(
        span_processors=[run_tracker],
        on_context_ready=run_tracker.capture_context,
        on_run_output=capture_and_finalize,
        on_context_closing=finalize_on_context_closing,
    )
    try:
        result = await single.run(run_input, hooks=hooks)
    except BaseException as error:
        # Context teardown can fail after on_run_output has already finalized the
        # trajectory, for example when stopping a sandbox times out. Preserve the
        # completed trajectory, but keep cancellation and incomplete finalization fatal.
        if isinstance(error, Exception) and finalized_trajectory is not None and last_run_output is not None:
            logger.exception(
                'Gateway cleanup failed after trajectory %s was finalized; returning the trajectory',
                traj_id,
            )
            return finalized_trajectory, last_run_output
        try:
            await finalize(None, termination_reason='UNHANDLED_ERROR')
        except Exception:
            logger.exception('Failed to finalize Gateway session %s', traj_id)
        raise

    if finalized_trajectory is None:
        scoreable = _scoreable_hycode_exit_name(run_input, result) is not None
        termination_reason = _resolve_termination_reason(
            run_tracker.termination_reason,
            result,
            scoreable_hycode_exit=scoreable,
            gateway_truncated=gateway_truncated(),
        )
        await finalize(result, termination_reason=termination_reason)
    assert finalized_trajectory is not None
    return finalized_trajectory, result


def _attach_hytuner_execution_metadata(
    trajectory: Trajectory,
    result: single.RunOutput,
    *,
    run_input: HyTunerRunInput,
    compaction_stats: Any,
    token_budget: int | None = None,
) -> None:
    if 'hytuner' in trajectory.extra:
        raise ValueError("trajectory.extra['hytuner'] is reserved by the HyTuner runner")
    if compaction_stats is not None:
        trajectory.extra['compaction_stats'] = copy.deepcopy(compaction_stats)
    hytuner_metadata: dict[str, Any] = {
        'success': result.success,
        'request_id': trajectory.session_id,
        'error_type': None if result.error is None else result.error.type,
        'error_message': None if result.error is None else result.error.message,
    }
    if token_budget is not None:
        hytuner_metadata['token_budget'] = token_budget
    hycode_exit_name = _scoreable_hycode_exit_name(run_input, result)
    if hycode_exit_name is not None:
        assert result.agent_output is not None
        exit_code = result.agent_output.exit_code
        trajectory.extra['hycode_exit_name'] = hycode_exit_name
        trajectory.extra['hycode_exit_code'] = exit_code
        hytuner_metadata['hycode_exit_name'] = hycode_exit_name
        hytuner_metadata['hycode_exit_code'] = exit_code
        hytuner_metadata['success'] = True
    trajectory.extra['hytuner'] = hytuner_metadata


__all__ = [
    'HyTunerRewardPolicy',
    'HyTunerRunInput',
    'HyTunerRunOutput',
    'HyTunerSegment',
    'HyTunerTokenBudgetConfig',
    'run',
]
