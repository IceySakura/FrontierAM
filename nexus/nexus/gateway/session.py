"""GatewaySession: per-trajectory interaction container.

Composes Processor + TrajectoryAssembler to manage a single trajectory's model
interactions and trace assembly.
"""

import asyncio
import copy
import logging
import threading
import time

from typing import Any, cast

import numpy as np

from nexus.core import ContextLengthExceededError
from nexus.gateway.model_registry import ModelBinding
from nexus.gateway.observers.base import TrajectoryUpdate
from nexus.gateway.observers.hub import get_current_trajectory_observation_hub
from nexus.gateway.processor import Processor
from nexus.gateway.scaffold_policy import InvocationKind, ScaffoldContext, ScaffoldPolicy
from nexus.gateway.token_budget import TokenBudgetState
from nexus.gateway.trajectory import TokenAccount, Trajectory
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.trajectory_store import NodePayloadRef, RecordRef, TrajectoryRecordStore
from nexus.gateway.types import TOKEN_ID_DTYPE, FrozenPrefix, GatewayCompletionResult
from nexus.utils.token_budget_config import TokenBudgetComponent


logger = logging.getLogger(__name__)


class GatewaySession:
    """Per-trajectory interaction container.

    Each session corresponds to one trajectory_id (e.g., one agent episode).
    It owns a TrajectoryAssembler (token-level trace) and delegates model I/O to a
    shared Processor.

    Budget parameters arrive in metadata['token_budget_config']. With registered models,
    only the global token_budget and token_budget_include are used; max_model_len
    comes from each model. One shared budget state tracks total consumption,
    with model-specific context keys isolating prompt history and segment counters.
    Message-level calls remain uncounted, and explicit requests retain the legacy
    budget bypass. Concurrent requests do not reserve budget before their outputs
    are recorded.
    """

    def __init__(
        self,
        session_id: str,
        processor: Processor,
        record_store: TrajectoryRecordStore,
        metadata: dict | None = None,
        scaffold_policy: ScaffoldPolicy | str | None = None,
    ):
        self.session_id = session_id
        hub = get_current_trajectory_observation_hub()
        if hub is not None:
            hub.begin_trajectory(session_id)
        self._processor = processor
        self._record_store = record_store
        self._segment_assembler = TrajectoryAssembler(scaffold_policy=scaffold_policy, session_id=session_id)
        session_metadata = copy.deepcopy(metadata or {})
        token_budget_config = session_metadata.pop('token_budget_config', {})
        self._metadata = session_metadata
        self._ended = False
        self._termination_reason: str | None = None
        # NOTE(petersheng): whether the latest main-chain agent call ended at a length limit, so a
        # run can be closed as TRUNCATION without every agent reporting it itself.
        self._last_main_call_truncated = False
        self._timeline: list[dict] = []
        self._lock = threading.RLock()
        self._in_flight = 0
        self._pending_completions: dict[asyncio.Task, str] = {}
        self._aborted_request_labels: list[str] = []
        self._record_publication_failure: BaseException | None = None
        self._discard_records = False
        self._pending_publications: list[tuple[list[int], asyncio.Task[list[RecordRef]]]] = []
        self._publication_finalization: asyncio.Task[None] | None = None
        self._max_model_len: int | None = token_budget_config.get('max_model_len')
        self._token_budget_state = TokenBudgetState(
            token_budget=token_budget_config.get('token_budget'),
            token_budget_include=token_budget_config.get('token_budget_include'),
        )

    @property
    def is_ended(self) -> bool:
        return self._ended

    @property
    def last_main_call_truncated(self) -> bool:
        """Whether the latest main-chain agent call hit max_tokens or was refused for context length."""
        return self._last_main_call_truncated

    @property
    def record_store(self) -> TrajectoryRecordStore:
        """The store owning this session's node payloads.

        Finalizers read payloads back through this store, so the session is the single
        owner and no caller has to thread one in alongside the trajectory.
        """
        return self._record_store

    @property
    def segment_assembler(self) -> TrajectoryAssembler:
        return self._segment_assembler

    @staticmethod
    def _get_budget_context_key(context_key: Any, binding: ModelBinding | None) -> Any:
        return (binding.model_id, context_key) if binding is not None else context_key

    async def create_completion(
        self,
        messages: list[dict],
        request_kwargs: dict[str, Any] | None = None,
        priority: int | None = None,
        uses_default_max_tokens: bool = True,
        request_meta: dict[str, Any] | None = None,
        binding: ModelBinding | None = None,
        context: ScaffoldContext | None = None,
    ) -> GatewayCompletionResult:
        """Run a single completion through the processor and record the trace.

        Reuse the client's classification and selected model when supplied.
        Request parameters are already resolved before entering the session.
        """
        processor = binding.processor if binding is not None else self._processor
        model_id = binding.model_id if binding is not None else None
        max_model_len = binding.max_model_len if binding is not None else self._max_model_len
        with self._lock:
            if self._ended:
                raise RuntimeError(f'Session {self.session_id} already ended')
            if self._publication_finalization is not None:
                raise RuntimeError(f'Session {self.session_id} is closing')
            if self._record_publication_failure is not None:
                raise RuntimeError(
                    f'Session {self.session_id} has a failed trajectory publication'
                ) from self._record_publication_failure
            recording_messages = copy.deepcopy(messages)
            execution_messages = copy.deepcopy(messages)
            request_snapshot = copy.deepcopy(request_kwargs)
            execution_request = copy.deepcopy(request_kwargs)
            recording_meta = copy.deepcopy(request_meta)
            execution_meta = copy.deepcopy(request_meta)
            tools = request_snapshot.get('tools') if request_snapshot is not None else None
            request_facts = (
                copy.deepcopy(context)
                if context is not None
                else self._segment_assembler.extract_context(
                    recording_messages,
                    tools=tools,
                    request_meta=recording_meta,
                )
            )
            # NOTE(jerryhrzhou): isolate token histories by model without changing trajectory context identities.
            budget_context_key = self._get_budget_context_key(request_facts.context_key, binding)
            frozen_prefix = self._segment_assembler.frozen_prefix(
                recording_messages,
                tools=tools,
                context=request_facts,
                model_id=model_id,
            )
            self._in_flight += 1
            task = asyncio.current_task()
            if task is not None:
                request_label = (
                    str(request_facts.request_id)
                    if request_facts.request_id is not None
                    else f'{request_facts.agent_id or "main"}#{request_facts.sequence}'
                )
                self._pending_completions[task] = request_label
        t_start = time.time()
        try:
            processor_kwargs: dict[str, Any] = {
                'request_id': self.session_id,
                'messages': execution_messages,
                'request_kwargs': execution_request,
                'priority': priority,
            }
            if isinstance(processor, Processor):

                def prepare_generation(
                    raw_ids: np.ndarray,
                    prompt_ids: np.ndarray,
                    request_kwargs: dict[str, Any],
                    uses_default_max_tokens: bool,
                ) -> dict[str, Any]:
                    with self._lock:
                        self._account_tool_observation_suffix(
                            budget_context_key,
                            execution_messages,
                            raw_ids,
                            prompt_ids,
                            frozen_prefix,
                            execution_meta,
                        )
                        return self._token_budget_state.prepare_generation(
                            budget_context_key,
                            raw_ids,
                            prompt_ids,
                            request_kwargs,
                            uses_default_max_tokens,
                            max_model_len=max_model_len,
                        )

                processor_kwargs['frozen_prefix'] = frozen_prefix
                processor_kwargs['prepare_generation'] = prepare_generation
                processor_kwargs['uses_default_max_tokens'] = uses_default_max_tokens
                processor_kwargs['request_meta'] = execution_meta
            is_main_agent_call = (
                request_facts.membership is None and request_facts.invocation_kind == InvocationKind.AGENT_INFERENCE
            )
            if is_main_agent_call:
                # NOTE(petersheng): clear before the call so an ordinary failure after an earlier
                # length-limited turn is classified as an error, not inherited as TRUNCATION.
                with self._lock:
                    self._last_main_call_truncated = False
            try:
                result = await processor.create_completion(**processor_kwargs)
            except ContextLengthExceededError:
                if is_main_agent_call:
                    with self._lock:
                        self._last_main_call_truncated = True
                raise
            if is_main_agent_call:
                with self._lock:
                    self._last_main_call_truncated = result.stop_reason == 'length'
            t_end = time.time()
            completion = GatewayCompletionResult(
                message=result.message,
                token_ids=result.token_ids,
                log_probs=result.log_probs,
                routed_experts=result.routed_experts,
                stop_reason=result.stop_reason,
                usage=result.usage,
                id=result.id,
                created=result.created,
                model=result.model,
                object=result.object,
                system_fingerprint=result.system_fingerprint,
                service_tier=result.service_tier,
                choice_logprobs=result.choice_logprobs,
                choice_provider_specific_fields=result.choice_provider_specific_fields,
                extra_fields=result.extra_fields,
                inference_time=getattr(result, 'inference_time', None),
                wall_inference_time=getattr(result, 'wall_inference_time', None),
            )
            token_training = None
            if processor.is_token_level:
                token_training = {
                    'prompt_ids': result.prompt_ids,
                    'raw_prompt_ids': result.raw_prompt_ids,
                    'response_ids': result.token_ids or [],
                    'response_logprobs': result.log_probs,
                    'routed_experts': result.routed_experts,
                    'multi_modal_items': result.multi_modal_items,
                    'extra_fields': result.extra_fields,
                }
            backend_abort_wait = float((result.extra_fields or {}).get('_abort_wait_time') or 0.0)
            try:
                with self._lock:
                    if self._record_publication_failure is not None:
                        raise RuntimeError(
                            f'Session {self.session_id} has a failed trajectory publication'
                        ) from self._record_publication_failure
                    if processor.is_token_level:
                        self._token_budget_state.record_generation(
                            budget_context_key,
                            result.raw_prompt_ids
                            if result.raw_prompt_ids is not None
                            else np.empty(0, dtype=TOKEN_ID_DTYPE),
                            result.prompt_ids if result.prompt_ids is not None else np.empty(0, dtype=TOKEN_ID_DTYPE),
                            list(result.token_ids or []),
                            output_component=(
                                TokenBudgetComponent.COMPACTION_OUTPUT
                                if request_facts.invocation_kind == InvocationKind.SUMMARY_INFERENCE
                                else TokenBudgetComponent.AGENT_OUTPUT
                            ),
                        )
                    publications = self._segment_assembler.record_completion(
                        messages=recording_messages,
                        output_message=result.message,
                        request_kwargs=request_snapshot,
                        stop_reason=result.stop_reason,
                        token_training=token_training,
                        response_meta={
                            'model_id': model_id,
                            'model': result.model,
                            'response_id': result.id,
                            'usage': result.usage,
                            'inference_time': getattr(result, 'inference_time', None),
                            'wall_inference_time': getattr(result, 'wall_inference_time', None),
                            'abort_wait_time': backend_abort_wait,
                            'num_preempted': (result.extra_fields or {}).get('_num_preempted'),
                        },
                        completion_text=result.completion_text,
                        prompt_render=result.prompt_render,
                        context=request_facts,
                        request_meta=recording_meta,
                        start_time=t_start,
                        end_time=t_end,
                        frozen_prefix=processor_kwargs.get('frozen_prefix'),
                    )
            except BaseException as exc:
                with self._lock:
                    if self._record_publication_failure is None:
                        self._record_publication_failure = exc
                    self._discard_records = True
                raise
            with self._lock:
                for record in publications:
                    account = cast(
                        TokenAccount,
                        self._segment_assembler.trajectory.nodes[record.node_id].token_account,
                    )
                    account.has_routed_experts = record.routed_experts is not None
                if publications and not self._discard_records:
                    self._pending_publications.append((
                        [record.node_id for record in publications],
                        asyncio.create_task(self._record_store.put_many(publications)),
                    ))
                    # Training payloads belong to the publication task; tokens and wire facts remain live.
                    completion.log_probs = None
                    completion.routed_experts = None
                    for key in ('resp_topk_logprob_indice', 'resp_topk_logprob_value'):
                        completion.extra_fields.pop(key, None)
                        result.extra_fields.pop(key, None)
                    result.log_probs = None
                    result.routed_experts = None
                    result.multi_modal_items = None
                    token_training = None
                record_publication_failure = self._record_publication_failure
                if record_publication_failure is not None:
                    observation_snapshot = None
                else:
                    self._record_timeline_event(
                        'inference',
                        'create_completion',
                        t_start,
                        t_end,
                        request_id=request_facts.request_id,
                        agent_id=request_facts.agent_id,
                        is_sidechain=request_facts.membership is not None,
                        iteration=request_facts.iteration,
                        retry=request_facts.retry,
                        sequence=request_facts.sequence,
                    )
                    observation_snapshot = self._observation_snapshot_locked()
            if record_publication_failure is not None:
                raise RuntimeError(
                    f'Session {self.session_id} has a failed trajectory publication'
                ) from record_publication_failure
            self._publish_observation_snapshot(observation_snapshot)
            return completion
        finally:
            with self._lock:
                self._in_flight -= 1
                if task is not None:
                    self._pending_completions.pop(task, None)

    async def finish_pending_publications(self) -> None:
        """Drain background puts before ending the session, releasing invalid trajectory records."""
        with self._lock:
            if self._publication_finalization is None:
                if self._in_flight:
                    raise RuntimeError(f'Session {self.session_id} has {self._in_flight} in-flight request(s)')
                self._publication_finalization = asyncio.create_task(self._finish_pending_publications())
            finalization = self._publication_finalization
        # NOTE(jerryhrzhou): cancelling a caller must not strand puts or interrupt release after a failed rollout.
        try:
            await asyncio.shield(finalization)
        except asyncio.CancelledError:
            await asyncio.shield(finalization)
            raise

    async def _finish_pending_publications(self) -> None:
        # Measure the remaining wait, excluding puts already overlapped with rollout.
        wait_start = time.perf_counter()
        results = await asyncio.gather(*(task for _, task in self._pending_publications), return_exceptions=True)
        self._segment_assembler.trajectory.metrics['trajectory_store_put_wait_time'] = time.perf_counter() - wait_start
        try:
            for (node_ids, _), refs in zip(self._pending_publications, results, strict=True):
                if isinstance(refs, BaseException):
                    raise refs
                if len(refs) != len(node_ids) or any(not isinstance(ref, NodePayloadRef) for ref in refs):
                    raise RuntimeError('TrajectoryRecordStore returned invalid Node publication refs')
                with self._lock:
                    for node_id, ref in zip(node_ids, refs, strict=True):
                        account = cast(TokenAccount, self._segment_assembler.trajectory.nodes[node_id].token_account)
                        account.payload_ref = cast(NodePayloadRef, ref)
            if self._discard_records:
                # TODO(jerryhrzhou): Verify the record release path when errors occur.
                await self._discard_published_records()
        except BaseException as exc:
            self._record_publication_failure = exc
            raise
        finally:
            self._pending_publications.clear()
        if self._record_publication_failure is not None:
            raise self._record_publication_failure

    async def _discard_published_records(self) -> None:
        with self._lock:
            refs = {
                cast(NodePayloadRef, account.payload_ref)
                for node in self._segment_assembler.trajectory.nodes.values()
                if (account := node.token_account) is not None and account.payload_ref is not None
            }
        if not refs:
            return
        await self._record_store.release_many(tuple(refs))
        with self._lock:
            for node in self._segment_assembler.trajectory.nodes.values():
                account = node.token_account
                if account is not None and account.payload_ref in refs:
                    account.payload_ref = None

    def _account_tool_observation_suffix(
        self,
        context_key: Any,
        messages: list[dict],
        raw_ids: np.ndarray,
        prompt_ids: np.ndarray,
        frozen_prefix: FrozenPrefix | None,
        request_meta: dict[str, Any] | None,
    ) -> bool:
        invocation_kind = request_meta.get('invocation_kind') if isinstance(request_meta, dict) else None
        if frozen_prefix is None or invocation_kind in ('summary_inference', 'compaction_resume'):
            return False
        appended_messages = messages[frozen_prefix.count :]
        if not appended_messages or any(message.get('role') != 'tool' for message in appended_messages):
            return False
        # Measure the entire rendered continuation, including template/control
        # tokens, against the same frozen sampled stream sent to the backend.
        frozen_token_count = sum(len(ids) for ids in frozen_prefix.message_token_ids)
        self._token_budget_state.account_tool_observation_suffix(
            context_key, raw_ids, messages, len(prompt_ids) - frozen_token_count
        )
        return True

    async def update_token_consumption(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        request_meta: dict[str, Any] | None = None,
        *,
        binding: ModelBinding | None = None,
    ) -> None:
        processor = binding.processor if binding is not None else self._processor
        model_id = binding.model_id if binding is not None else None
        if not processor.is_token_level:
            raise RuntimeError('Token consumption tracking requires a token-level processor')
        context = self._segment_assembler.extract_context(messages, tools, request_meta)
        budget_context_key = self._get_budget_context_key(context.context_key, binding)
        # NOTE(arlenxu): accounting must measure the frozen sampled stream + rendered tail the
        # next request actually resends; a fresh canonical render is not a prefix-extension of
        # the spliced stream whenever an emission violates render(parse(x)) == x.
        frozen_prefix = self._segment_assembler.frozen_prefix(
            messages,
            tools=tools,
            context=context,
            model_id=model_id,
        )
        raw_prompt_ids, prompt_ids, _, _ = await processor.render_and_tokenize(
            messages,
            tools,
            frozen_prefix=frozen_prefix,
        )
        with self._lock:
            is_tool_append = self._account_tool_observation_suffix(
                budget_context_key,
                messages,
                raw_prompt_ids,
                prompt_ids,
                frozen_prefix,
                request_meta,
            )
            # Keep explicit updates' existing strict-extension contract for other
            # inputs; classified tools have already been charged through the ledger.
            self._token_budget_state.update_prompt(
                budget_context_key, raw_prompt_ids, prompt_ids, consume=not is_tool_append
            )

    def segment_consumed_tokens(
        self, request_meta: dict[str, Any] | None = None, *, binding: ModelBinding | None = None
    ) -> int:
        context = self._segment_assembler.extract_context([], request_meta=request_meta)
        budget_context_key = self._get_budget_context_key(context.context_key, binding)
        return self._token_budget_state.segment_consumed_tokens(budget_context_key)

    def set_segment_consumed_tokens(
        self,
        value: int,
        request_meta: dict[str, Any] | None = None,
        *,
        binding: ModelBinding | None = None,
    ) -> None:
        context = self._segment_assembler.extract_context([], request_meta=request_meta)
        budget_context_key = self._get_budget_context_key(context.context_key, binding)
        self._token_budget_state.set_segment_consumed_tokens(budget_context_key, value)

    def record_observation(
        self,
        content: str | None = None,
        token_ids: list[int] | None = None,
        role: str = 'tool',
        parent_messages: list[dict] | None = None,
        request_meta: dict[str, Any] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        status: str | None = None,
        message: dict[str, Any] | None = None,
    ) -> None:
        """Record an observation (tool/env output) in the trajectory.

        Called by control flow to inform Gateway about non-LLM tokens. ``parent_messages`` is
        the history the observation follows and is required.
        """
        with self._lock:
            self._segment_assembler.record_observation(
                content=content,
                token_ids=token_ids,
                role=role,
                message=message,
                parent_messages=parent_messages,
                request_meta=request_meta,
                start_time=start_time,
                end_time=end_time,
                status=status,
            )
            observation = copy.deepcopy(message) if message is not None else {'role': role, 'content': content}
            observation.setdefault('role', role)
            observation_snapshot = self._observation_snapshot_locked()
        self._publish_observation_snapshot(observation_snapshot)

    def finalize_segment(self) -> None:
        """Mark a segment boundary (e.g., context compaction)."""
        with self._lock:
            self._segment_assembler.finalize_segment()
            observation_snapshot = self._observation_snapshot_locked()
        self._publish_observation_snapshot(observation_snapshot)

    def get_trajectory(self, *, state: str | None = None, copy_inputs: bool = True) -> Trajectory:
        """Resolve the trajectory, copying live inputs unless explicitly disabled.

        ``copy_inputs=False`` mutates and returns the session's live trajectory.
        """
        from nexus.gateway.trajectory_freeze import freeze_trajectory

        with self._lock:
            inputs = (
                self._segment_assembler.trajectory,
                self._segment_assembler._runtime_state,
                self._timeline,
                self._metadata,
                self._ended,
                self._termination_reason,
            )
            if copy_inputs:
                inputs = copy.deepcopy(inputs)
            (
                live_trajectory,
                runtime_state,
                timeline,
                metadata,
                ended,
                termination_reason,
            ) = inputs
            response_policy = self._segment_assembler._policy.response_policy
            resolved_state = state if state is not None else ('completed' if ended else 'active')

            def materialize() -> Trajectory:
                return freeze_trajectory(
                    live_trajectory,
                    runtime_state,
                    response_policy,
                    state=resolved_state,
                    session_timeline=timeline,
                    termination_reason=termination_reason,
                    metrics=live_trajectory.metrics,
                    extra=metadata,
                    copy_inputs=False,
                )

            if not copy_inputs:
                return materialize()
        return materialize()

    def end(
        self,
        termination_reason: str | None = None,
        *,
        final_reward: float | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Trajectory:
        """End the session and return its canonical trajectory."""
        with self._lock:
            if self._in_flight:
                raise RuntimeError(f'Session {self.session_id} has {self._in_flight} in-flight request(s)')
            if self._record_publication_failure is not None:
                raise RuntimeError(
                    f'Session {self.session_id} has a failed trajectory publication'
                ) from self._record_publication_failure
            if self._pending_publications or (
                self._publication_finalization is not None and not self._publication_finalization.done()
            ):
                raise RuntimeError(f'Session {self.session_id} must await finish_pending_publications() before end()')
            trajectory = self.get_trajectory(state='completed', copy_inputs=False)
            trajectory.final_reward = final_reward
            trajectory.termination_reason = termination_reason
            trajectory.metrics.update(copy.deepcopy(metrics or {}))
            trajectory.extra.update(copy.deepcopy(extra or {}))
            if self._aborted_request_labels:
                trajectory.extra['aborted_in_flight_requests'] = list(self._aborted_request_labels)
            self._ended = True
            self._termination_reason = termination_reason
            return trajectory

    async def abort_in_flight(self) -> list[str]:
        """Seal the session and cancel completions that no longer have a consumer.

        Every caller runs after the scaffold has exited, so a pending result would go
        unread. Sealing makes ``create_completion`` reject a late request instead of
        counting it in flight. Returns the cancelled request labels, which ``end()``
        records on the trajectory as ``extra['aborted_in_flight_requests']``.
        """
        # Seal under the snapshot's lock: end() runs once this returns, and a request
        # arriving in between would leave it rejecting the trajectory as in flight.
        with self._lock:
            self._ended = True
            me = asyncio.current_task()
            orphans = {
                task: label for task, label in self._pending_completions.items() if task is not me and not task.done()
            }
        if not orphans:
            return []
        for task in orphans:
            task.cancel()
        await asyncio.gather(*orphans, return_exceptions=True)
        labels = sorted(orphans.values())
        with self._lock:
            self._aborted_request_labels.extend(labels)
        logger.warning(
            'Session %s: aborted %d in-flight request(s): %s',
            self.session_id,
            len(labels),
            labels,
        )
        return labels

    def _observation_snapshot_locked(self) -> TrajectoryUpdate | None:
        hub = get_current_trajectory_observation_hub()
        if hub is None:
            return None
        revision = hub.reserve_update(self.session_id)
        if revision is None:
            return None
        return TrajectoryUpdate(snapshot=self.get_trajectory(copy_inputs=True), revision=revision)

    @staticmethod
    def _publish_observation_snapshot(update: TrajectoryUpdate | None) -> None:
        hub = get_current_trajectory_observation_hub()
        if update is not None and hub is not None:
            hub.publish_update(update)

    def _record_timeline_event(
        self,
        event_type: str,
        name: str,
        start: float,
        end: float,
        **meta: Any,
    ) -> None:
        clean_meta = {key: value for key, value in meta.items() if value is not None}
        self._timeline.append({
            'type': event_type,
            'name': name,
            'start': start,
            'end': end,
            'duration': end - start,
            **clean_meta,
        })
