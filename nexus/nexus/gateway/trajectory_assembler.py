"""Record Gateway events directly into one live public Trajectory."""

from __future__ import annotations

import logging

from collections.abc import Hashable
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np

from nexus.gateway.reply_alias import ReplyAlias, content_similarity, ledger_view, resend_difference
from nexus.gateway.runtime_state import (
    NodeRuntimeFacts,
    PendingBoundary,
    RuntimeFactOccurrence,
    RuntimeState,
)
from nexus.gateway.scaffold_policy import (
    BoundAnchor,
    BoundaryFact,
    EventFacts,
    InputMessage,
    InvocationKind,
    ObservationEvent,
    ObservationMessage,
    OutputMessage,
    RequestEvent,
    ResponseEvent,
    ScaffoldContext,
    ScaffoldPolicy,
    SidechainMembership,
    build_scaffold_policy,
)
from nexus.gateway.trajectory import (
    InvocationFacts,
    NodeId,
    ProviderFacts,
    SegmentBoundary,
    TokenAccount,
    Trajectory,
    TrajectoryNode,
)
from nexus.gateway.trajectory_store import NodePayloadRecord
from nexus.gateway.types import LOGPROB_DTYPE, TOKEN_ID_DTYPE, AliasCandidate, FrozenPrefix, PromptRender
from nexus.gateway.wire import copy_message, copy_messages, copy_value, copy_wire_value


logger = logging.getLogger(__name__)

_ANCHORED_FACT_FIELDS = (
    'spawn_intents',
    'spawn_bindings',
    'retrieval_intents',
    'delivery_receipts',
    'notification_deliveries',
    'child_prompts',
)


class TrajectoryAssembler:
    """Own and mutate one live Trajectory plus private scalar runtime state."""

    def __init__(
        self,
        scaffold_policy: ScaffoldPolicy | str | None = None,
        *,
        session_id: str = '',
    ):
        self._policy = build_scaffold_policy(scaffold_policy)
        self.trajectory = Trajectory(session_id=session_id, state='active')
        self._runtime_state = RuntimeState()
        self._reply_alias = ReplyAlias.from_env()

    def extract_context(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        request_meta: dict[str, Any] | None = None,
    ) -> ScaffoldContext:
        facts = self._inspect_request(messages, tools, request_meta)
        return ScaffoldContext(
            request_meta=copy_wire_value(request_meta),
            context_key=facts.context_key,
            request_id=facts.request_id,
            agent_id=facts.agent_id,
            membership=facts.membership,
            declared_claim=facts.declared_claim,
            parent_claim=facts.parent_claim,
            iteration=facts.iteration,
            retry=facts.retry,
            sequence=facts.sequence,
            invocation_kind=facts.invocation_kind,
            invocation_source=facts.invocation_source,
        )

    @staticmethod
    def context_key(context: ScaffoldContext) -> Hashable:
        return context.context_key

    def frozen_prefix(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        context: ScaffoldContext | None = None,
        request_meta: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> FrozenPrefix | None:
        """The longest recorded stream the request resends verbatim.

        Only prefixes this context already sent with compatible tools, and got a completion
        for, are reusable; among equally long ones the request's verbatim history, then the
        newest Node, wins. A chain's first resend of its own reply may differ from the sample
        (see ``reply_alias``). The caller must record the completion with the same messages
        and context, since recording attaches them to the returned ``node_ids`` unchecked.
        """
        request_facts = self._event_facts(self._request_event(messages, tools, request_meta), context)
        # NOTE(jerryhrzhou): Reusing another model's sampled prefix would splice foreign tokenization into the prompt.
        last_llm_id = max(
            (
                node_id
                for node_id, node in self.trajectory.nodes.items()
                if node.generated_by == 'llm'
                and self._runtime_state.facts.by_node[node_id].context_key == request_facts.context_key
            ),
            default=None,
        )
        if last_llm_id is not None:
            last_provider = self.trajectory.nodes[last_llm_id].provider_facts
            if (last_provider.model_id if last_provider is not None else None) != model_id:
                return None
        identities = self._input_identities(request_facts, len(messages))
        reusable = self._reusable_nodes(request_facts, tools)
        path, candidates = self._deepest_reusable_path(messages, identities, reusable, request_facts.context_key)
        if not path:
            return None
        frozen = self._frozen_prefix([self.trajectory.nodes[node_id] for node_id in path], candidates)
        if frozen.frozen_routing_length and self._request_starts_new_segment(request_facts, path[-1]):
            frozen = replace(frozen, frozen_routing_length=0)
        return frozen if frozen.text else None

    def _request_starts_new_segment(self, facts: EventFacts, prefix_end_id: NodeId) -> bool:
        if facts.invocation_kind == InvocationKind.SUMMARY_INFERENCE:
            return True
        return bool(self.trajectory.nodes[prefix_end_id].children)

    # ------------------------------------------------------------------
    # Frozen prefix selection
    # ------------------------------------------------------------------

    def _reusable_nodes(self, facts: EventFacts, tools: list[dict] | None = None) -> set[NodeId]:
        """Nodes with sealed tokens on a path this context already sent with compatible tools."""
        sent_completions = [
            node_id
            for node_id, node in self.trajectory.nodes.items()
            if node.generated_by == 'llm'
            and self._runtime_state.facts.by_node[node_id].context_key == facts.context_key
            and self._has_compatible_tools(node_id, facts, tools)
        ]
        on_sent_paths: set[NodeId] = set()
        for node_id in sent_completions:
            ancestor_id: NodeId | None = node_id
            while ancestor_id is not None and ancestor_id not in on_sent_paths:
                on_sent_paths.add(ancestor_id)
                ancestor_id = self.trajectory.nodes[ancestor_id].parent
        reusable: set[NodeId] = set()
        for node_id in on_sent_paths:
            node = self.trajectory.nodes[node_id]
            account = node.token_account
            if node.templated_message is not None and account is not None and account.rollout_token_ids is not None:
                reusable.add(node_id)
        return reusable

    def _deepest_reusable_path(
        self,
        messages: list[dict],
        identities: list[Hashable],
        reusable: set[NodeId],
        context_key: Hashable,
    ) -> tuple[list[NodeId], tuple[AliasCandidate, ...]]:
        """Follow the request's messages down the tree through reusable Nodes as far as they go."""
        nodes = self.trajectory.nodes
        frontier: list[NodeId | None] = [None]
        accepted: list[AliasCandidate] = []
        rejected: list[AliasCandidate] = []
        for message, identity in zip(messages, identities, strict=True):
            children = [
                child_id
                for parent_id in frontier
                for child_id in self._find_children(parent_id, identity)
                if child_id in reusable
            ]
            if not children and self._reply_alias.enabled and message.get('role') == 'assistant':
                candidates = [
                    AliasCandidate(
                        node_id=child_id,
                        identity=identity,
                        rejection=self._reply_alias.rejection(nodes[child_id].original_message, message),
                    )
                    for parent_id in frontier
                    if parent_id is not None
                    for child_id in nodes[parent_id].children
                    if child_id in reusable and self._awaits_first_resend(child_id, context_key)
                ]
                aliases = [candidate for candidate in candidates if candidate.rejection is None]
                accepted += aliases
                children = [candidate.node_id for candidate in aliases]
                if not children:
                    rejected = candidates
            if not children:
                break
            frontier = children
        ends = [node_id for node_id in frontier if node_id is not None]
        if not ends:
            return [], ()
        best_end = max(ends, key=lambda node_id: (self._verbatim_messages(node_id, messages), node_id))
        path = self._node_path(best_end)
        return path, (*(candidate for candidate in accepted if candidate.node_id in path), *rejected)

    def _verbatim_messages(self, node_id: NodeId, messages: list[dict]) -> int:
        """How many messages on the path to ``node_id`` the request resends unchanged."""
        return sum(
            copy_message(message) == self.trajectory.nodes[path_id].original_message
            for path_id, message in zip(self._node_path(node_id), messages, strict=False)
        )

    def _has_compatible_tools(
        self,
        previous_id: NodeId,
        request_facts: EventFacts,
        tools: list[dict] | None,
    ) -> bool:
        """Allow deferred discovery without changing schemas used by the recorded prefix."""
        previous_facts = self._runtime_state.facts.by_node[previous_id]
        if previous_facts.tools_identity == request_facts.tools_identity:
            return True
        if self._policy.tools_key != 'exact-tools-v1':
            return False
        previous_tools = (self.trajectory.nodes[previous_id].request_kwargs or {}).get('tools') or []
        if not isinstance(tools, list) or len(tools) <= len(previous_tools):
            return False
        if tools[: len(previous_tools)] != previous_tools:
            return False
        names: set[str] = set()
        added_names: set[str] = set()
        for index, tool in enumerate(tools):
            function = tool.get('function')
            if tool.get('type') != 'function' or not isinstance(function, dict):
                return False
            name = function.get('name')
            if not isinstance(name, str) or not name or name in names:
                return False
            names.add(name)
            if index >= len(previous_tools):
                if function.get('defer_loading') is not True:
                    return False
                added_names.add(name)
        for node_id in self._node_path(previous_id):
            content = (self.trajectory.nodes[node_id].original_message or {}).get('content')
            if isinstance(content, list) and any(
                isinstance(block, dict)
                and block.get('type') == 'tool_reference'
                and (block.get('name') or block.get('tool_name')) in added_names
                for block in content
            ):
                return False
        return True

    @staticmethod
    def _frozen_prefix(nodes: list[TrajectoryNode], alias_candidates: tuple[AliasCandidate, ...] = ()) -> FrozenPrefix:
        spans: list[str] = []
        token_ids: list[np.ndarray] = []
        raw_token_ids: list[np.ndarray] = []
        has_raw_token_ids = False
        for node in nodes:
            account = node.token_account
            assert node.templated_message is not None
            assert account is not None
            assert account.token_ids is not None
            assert account.rollout_token_ids is not None
            spans.append(node.templated_message)
            # NOTE(jerryhrzhou): Sealed token arrays are read-only inputs to all later renderers.
            token_ids.append(account.token_ids)
            raw_token_ids.append(account.rollout_token_ids)
            has_raw_token_ids |= account.raw_token_ids is not None
        return FrozenPrefix(
            count=len(nodes),
            text=''.join(spans),
            templated_messages=tuple(spans),
            message_token_ids=tuple(token_ids),
            raw_message_token_ids=tuple(raw_token_ids) if has_raw_token_ids else (),
            frozen_routing_length=TrajectoryAssembler._routing_length(nodes),
            node_ids=tuple(node.node_id for node in nodes),
            alias_candidates=alias_candidates,
        )

    def _awaits_first_resend(self, node_id: NodeId, context_key: Hashable) -> bool:
        """Whether this chain has yet to resend ``node_id``: no alias, no chain reply below, no later assistant sibling."""
        nodes, by_node = self.trajectory.nodes, self._runtime_state.facts.by_node
        node = nodes[node_id]
        if node.parent is None or node.generated_by != 'llm' or by_node[node_id].context_key != context_key:
            return False
        if by_node[node_id].alias_identity is not None:
            return False
        pending = list(node.children)
        while pending:
            child_id = pending.pop()
            if nodes[child_id].generated_by == 'llm' and by_node[child_id].context_key == context_key:
                return False
            pending.extend(nodes[child_id].children)
        return not any(
            sibling_id > node_id
            and nodes[sibling_id].generated_by == 'client'
            and nodes[sibling_id].original_message.get('role') == 'assistant'
            and by_node[sibling_id].context_key == context_key
            for sibling_id in nodes[node.parent].children
        )

    def _record_aliases(
        self,
        candidates: tuple[AliasCandidate, ...],
        context_key: Hashable,
        messages: list[dict],
        path: list[NodeId],
        request_sequence: int,
    ) -> None:
        nodes, by_node, metrics = self.trajectory.nodes, self._runtime_state.facts.by_node, self.trajectory.metrics

        def position(candidate: AliasCandidate) -> int:
            return len(self._node_path(candidate.node_id)) - 1

        def describe(candidate: AliasCandidate) -> str:
            index, sampled = position(candidate), nodes[candidate.node_id].original_message
            reason = f' {candidate.rejection}' if candidate.rejection else ''
            return f'node {candidate.node_id}{reason}, message {index} {resend_difference(sampled, messages[index])}'

        def enter(candidate: AliasCandidate) -> None:
            index, sampled = position(candidate), nodes[candidate.node_id].original_message
            resent = messages[index]
            self.trajectory.extra.setdefault('reply_aliases', []).append({
                'request_sequence': request_sequence,
                'sampled_node_id': candidate.node_id,
                'resent_node_id': path[index],
                'rejection': candidate.rejection,
                'content_similarity': content_similarity(sampled, resent),
                'difference': resend_difference(sampled, resent),
                'sampled': ledger_view(sampled),
                'resent': ledger_view(resent),
            })

        rejected = [candidate for candidate in candidates if candidate.rejection is not None]
        for candidate in candidates:
            if candidate.rejection is None and self._awaits_first_resend(candidate.node_id, context_key):
                by_node[candidate.node_id] = replace(by_node[candidate.node_id], alias_identity=candidate.identity)
                enter(candidate)
                metrics['reply_aliased'] = metrics.get('reply_aliased', 0) + 1
                logger.info('gateway reply aliased: %s', describe(candidate))
        if rejected:
            for candidate in rejected:
                enter(candidate)
            metric = f'reply_alias_rejected_{max(rejected, key=lambda candidate: candidate.node_id).rejection}'
            metrics[metric] = metrics.get(metric, 0) + 1
            logger.info('gateway reply alias rejected: %s', '; '.join(describe(candidate) for candidate in rejected))

    def record_completion(
        self,
        messages: list[dict],
        output_message: dict,
        request_kwargs: dict[str, Any] | None = None,
        stop_reason: str | None = None,
        token_training: dict[str, Any] | None = None,
        response_meta: dict[str, Any] | None = None,
        completion_text: str | None = None,
        prompt_render: PromptRender | None = None,
        context: ScaffoldContext | None = None,
        request_meta: dict[str, Any] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        frozen_prefix: FrozenPrefix | None = None,
    ) -> list[NodePayloadRecord]:
        input_messages = copy_messages(messages)
        request_sequence = self._runtime_state.next_request_sequence
        self._runtime_state.next_request_sequence += 1
        tools = (request_kwargs or {}).get('tools')
        request_event = self._request_event(input_messages, tools, request_meta)
        request_facts = self._event_facts(request_event, context)
        normalized_training = self._normalize_token_training(token_training) if token_training is not None else None
        self._validate_prompt_inputs(normalized_training, prompt_render, completion_text)

        current_id, newly_sealed_input_node_ids = self._find_or_create_path(
            input_messages,
            request_facts,
            request_sequence,
            prompt_render,
            frozen_node_ids=frozen_prefix.node_ids if frozen_prefix is not None else (),
        )
        if frozen_prefix is not None:
            self._record_aliases(
                frozen_prefix.alias_candidates,
                request_facts.context_key,
                input_messages,
                self._node_path(current_id),
                request_sequence,
            )

        if prompt_render is not None and normalized_training is not None:
            self._attach_multi_modal_items(
                current_id,
                newly_sealed_input_node_ids,
                normalized_training.get('multi_modal_items'),
                prompt_render.message_media_counts,
            )

        glue_ids = np.empty(0, dtype=TOKEN_ID_DTYPE)
        if normalized_training is not None:
            assert prompt_render is not None
            glue_ids = prompt_render.generation_prompt_ids
            if not np.array_equal(
                np.concatenate((self._input_token_ids(current_id), glue_ids)),
                normalized_training['raw_prompt_ids'],
            ):
                raise ValueError('Assembly path does not reconstruct the model prompt IDs')

        output = copy_message(output_message)
        output.setdefault('role', 'assistant')
        response_attributes = {'type': 'agent_inference'}
        response_attributes.update({
            key: copy_value(value) for key, value in (response_meta or {}).items() if value is not None
        })
        response_facts = self._event_facts(
            ResponseEvent(
                request=request_event,
                message=output,
                response_meta=response_attributes,
                start_time=start_time,
                end_time=end_time,
                status='OK',
            ),
            context,
        )
        boundary = self._request_boundary(request_facts, current_id)
        boundary = self._merge_boundaries(boundary, self._output_boundary(response_facts))
        if request_facts.invocation_kind == InvocationKind.SUMMARY_INFERENCE and (
            boundary is None or boundary.boundary.kind != 'summary_inference'
        ):
            # A current summary invocation takes precedence over a historical
            # resume boundary. Policies need not duplicate metadata detection just
            # to make trajectory segmentation agree with output accounting.
            boundary = self._merge_boundaries(
                PendingBoundary(
                    SegmentBoundary(
                        starts_new_segment=True,
                        kind='summary_inference',
                        causes=('compaction',),
                        evidence={'invocation_source': request_facts.invocation_source},
                    )
                ),
                boundary,
            )
        boundary = self._consume_pending_boundary(boundary)
        physical_boundary = self._physical_boundary(
            request_facts=request_facts,
            current_id=current_id,
            token_training=normalized_training,
            tools=tools,
            model_id=(response_meta or {}).get('model_id'),
        )
        boundary = self._merge_boundaries(boundary, physical_boundary)

        output_identity = self._identity_for(response_facts, OutputMessage())
        output_id = self._new_node(
            parent_id=current_id,
            original_message=output,
            generated_by='llm',
            identity=output_identity,
            event_facts=response_facts,
            templated_message=(prompt_render.generation_prompt + (completion_text or ''))
            if prompt_render is not None
            else completion_text,
        )
        output_node = self.trajectory.nodes[output_id]
        output_node.request_message_overrides = {
            node_id: copy_message(message)
            for node_id, message in zip(self._node_path(current_id), input_messages, strict=True)
            if message != self.trajectory.nodes[node_id].original_message
        }
        wire_kwargs = copy_wire_value(request_kwargs or {})
        output_node.request_kwargs = wire_kwargs or None
        output_node.provider_facts = self._provider_facts(output, stop_reason, response_meta)
        output_node.timeline = [dict(response_facts.timeline)] if response_facts.timeline else []
        output_node.invocation = self._invocation(response_facts)
        if boundary is not None:
            output_node.segment_boundary = copy_value(boundary.boundary)
            output_facts = self._runtime_state.facts.by_node[output_id]
            self._runtime_state.facts.by_node[output_id] = replace(
                output_facts,
                boundary_claims=tuple(
                    (rule_name, output_id if anchor == OutputMessage() else anchor)
                    for rule_name, anchor in boundary.claims
                ),
            )
        if normalized_training is not None:
            output_node.token_account = self._token_account(glue_ids, normalized_training)
        publications: list[NodePayloadRecord] = []
        if normalized_training is not None:
            for node_id in [*newly_sealed_input_node_ids, output_id]:
                account = cast(TokenAccount, self.trajectory.nodes[node_id].token_account)
                publications.append(account.build_store_payload_record(self.trajectory.session_id, node_id))
                account.logprobs = None
                account.topk_indices = None
                account.topk_values = None
                account.routed_experts = None
                account.multi_modal_items = None
        return publications

    def record_observation(
        self,
        content: str | None = None,
        token_ids: list[int] | None = None,
        role: str = 'tool',
        parent_messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        context: ScaffoldContext | None = None,
        request_meta: dict[str, Any] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        status: str | None = None,
        message: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if token_ids:
            raise ValueError('observation token IDs are derived from the next rendered prompt')
        if parent_messages is None:
            raise ValueError('record_observation requires parent_messages to place the observation')
        if message is None:
            if content is None:
                raise ValueError('record_observation requires content or message')
            observation = {'role': role, 'content': content}
        else:
            observation = copy_message(message)
            if content is not None and observation.get('content') != content:
                raise ValueError('content differs from the explicit observation message')
            observation.setdefault('role', role)

        observation_event = ObservationEvent(
            message=observation,
            parent_messages=tuple(copy_messages(parent_messages)),
            tools=copy_wire_value(tools),
            request_meta=copy_wire_value(request_meta),
            start_time=start_time,
            end_time=end_time,
            status=status,
        )
        observation_facts = self._event_facts(observation_event, context)
        parent_event = self._request_event(parent_messages, tools, request_meta)
        parent_facts = self._event_facts(parent_event, context)
        parent_id = self._find_path_node(parent_facts, len(parent_messages))
        if parent_id is None:
            raise ValueError('parent_messages do not match this trajectory')

        identity = self._identity_for(observation_facts, ObservationMessage())
        candidates = self._find_children(parent_id, identity)
        if candidates:
            observation_id = candidates[0]
            self._merge_node_facts(observation_id, observation_facts)
        else:
            observation_id = self._new_node(
                parent_id=parent_id,
                original_message=observation,
                generated_by='observation',
                identity=identity,
                event_facts=observation_facts,
            )
        observation_node = self.trajectory.nodes[observation_id]
        observation_node.generated_by = 'observation'
        if observation_facts.timeline:
            observation_node.timeline.append(dict(observation_facts.timeline))
        return self._wire_messages_for_node(parent_id) + [copy_message(observation)]

    def finalize_segment(self) -> None:
        self._runtime_state.pending_boundary = PendingBoundary(
            SegmentBoundary(starts_new_segment=True, causes=('explicit',))
        )

    def freeze(
        self,
        *,
        session_id: str | None = None,
        state: str | None = None,
        session_timeline: list[dict[str, Any]] | None = None,
        termination_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Trajectory:
        from nexus.gateway.trajectory_freeze import freeze_trajectory

        return freeze_trajectory(
            self.trajectory,
            self._runtime_state,
            self._policy.response_policy,
            session_id=session_id,
            state=state,
            session_timeline=session_timeline,
            termination_reason=termination_reason,
            metrics=metrics,
            extra=extra,
        )

    def _inspect_request(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        request_meta: dict[str, Any] | None,
    ) -> EventFacts:
        return self._event_facts(self._request_event(messages, tools, request_meta))

    def _event_facts(
        self,
        event: RequestEvent | ResponseEvent | ObservationEvent,
        supplied: ScaffoldContext | None = None,
    ) -> EventFacts:
        if supplied is not None:
            if isinstance(event, RequestEvent):
                event = replace(event, request_meta=copy_wire_value(supplied.request_meta))
            elif isinstance(event, ResponseEvent):
                event = replace(
                    event,
                    request=replace(event.request, request_meta=copy_wire_value(supplied.request_meta)),
                )
            else:
                event = replace(event, request_meta=copy_wire_value(supplied.request_meta))
        facts = self._policy.inspect_event(event)
        if supplied is not None:
            facts = replace(
                facts,
                context_key=supplied.context_key,
                request_id=supplied.request_id,
                agent_id=supplied.agent_id,
                membership=supplied.membership,
                declared_claim=supplied.declared_claim,
                parent_claim=supplied.parent_claim,
                iteration=supplied.iteration,
                retry=supplied.retry,
                sequence=supplied.sequence,
                invocation_kind=supplied.invocation_kind,
                invocation_source=supplied.invocation_source,
                timeline=self._timeline_with_context(facts.timeline, supplied),
            )
        self._validate_event_facts(event, facts)
        return facts

    @staticmethod
    def _timeline_with_context(
        timeline: tuple[tuple[str, Hashable], ...],
        context: ScaffoldContext,
    ) -> tuple[tuple[str, Hashable], ...]:
        if not timeline:
            return ()
        projected = dict(timeline)
        for key, value in (
            ('request_id', context.request_id),
            ('agent_id', context.agent_id),
            ('iteration', context.iteration),
            ('retry', context.retry),
            ('sequence', context.sequence),
        ):
            if value is None:
                projected.pop(key, None)
            else:
                projected[key] = value
        projected['is_sidechain'] = context.membership is not None
        canonical_order = (
            'event_type',
            'request_id',
            'agent_id',
            'parent_agent_id',
            'is_sidechain',
            'start_time',
            'end_time',
            'duration',
            'status',
            'span_id',
            'parent_span_id',
            'iteration',
            'retry',
            'sequence',
            'attributes',
        )
        ordered = [(key, projected.pop(key)) for key in canonical_order if key in projected]
        ordered.extend(projected.items())
        return tuple(ordered)

    @staticmethod
    def _validate_event_facts(
        event: RequestEvent | ResponseEvent | ObservationEvent,
        facts: EventFacts,
    ) -> None:
        if isinstance(event, RequestEvent):
            expected_anchors = {InputMessage(index) for index in range(len(event.messages))}
        elif isinstance(event, ResponseEvent):
            expected_anchors = {OutputMessage()}
        else:
            expected_anchors = {ObservationMessage()}
        identity_anchors = [item.anchor for item in facts.message_identities]
        if len(identity_anchors) != len(expected_anchors) or set(identity_anchors) != expected_anchors:
            event_name = type(event).__name__
            raise ValueError(f'{event_name} message identities do not match its messages')

        for field_name in _ANCHORED_FACT_FIELDS:
            for fact in getattr(facts, field_name):
                if fact.anchor not in expected_anchors:
                    raise ValueError(f'{field_name} anchor is not emitted by this event')
        for fact in facts.boundary_facts:
            if fact.anchor not in expected_anchors:
                raise ValueError('boundary_facts anchor is not emitted by this event')

    @staticmethod
    def _request_event(
        messages: list[dict],
        tools: list[dict] | None,
        request_meta: dict[str, Any] | None,
    ) -> RequestEvent:
        return RequestEvent(
            messages=tuple(copy_messages(messages)),
            tools=copy_wire_value(tools),
            request_meta=copy_wire_value(request_meta),
        )

    @staticmethod
    def _identity_for(facts: EventFacts, anchor: InputMessage | OutputMessage | ObservationMessage) -> Hashable:
        return next(item.value for item in facts.message_identities if item.anchor == anchor)

    @staticmethod
    def _input_identities(facts: EventFacts, message_count: int) -> list[Hashable]:
        by_index: dict[int, Hashable] = {}
        for item in facts.message_identities:
            if not isinstance(item.anchor, InputMessage):
                raise ValueError('Request message identities must use InputMessage anchors')
            if item.anchor.index in by_index:
                raise ValueError(f'Duplicate InputMessage identity for index {item.anchor.index}')
            by_index[item.anchor.index] = item.value
        expected = set(range(message_count))
        if set(by_index) != expected:
            raise ValueError('Scaffold policy must emit exactly one identity for every input message')
        return [by_index[index] for index in range(message_count)]

    def _new_node(
        self,
        *,
        parent_id: NodeId | None,
        original_message: dict[str, Any],
        generated_by: Literal['llm', 'observation', 'client'],
        identity: Hashable,
        event_facts: EventFacts,
        message_id: int | None = None,
        templated_message: str | None = None,
        token_account: TokenAccount | None = None,
    ) -> NodeId:
        state = self._runtime_state
        node_id = state.next_node_id
        state.next_node_id += 1
        if message_id is None:
            message_id = state.next_message_id
            state.next_message_id += 1
        self.trajectory.nodes[node_id] = TrajectoryNode(
            node_id=node_id,
            message_id=message_id,
            parent=parent_id,
            generated_by=generated_by,
            children=[],
            original_message=copy_message(original_message),
            templated_message=templated_message,
            token_account=token_account,
        )
        if parent_id is None:
            self.trajectory.roots.append(node_id)
        else:
            self.trajectory.nodes[parent_id].children.append(node_id)
        state.facts.by_node[node_id] = self._bound_node_facts(node_id, identity, event_facts)
        if generated_by == 'llm':
            state.llm_by_message_id[message_id] = node_id
        return node_id

    def _bound_node_facts(
        self,
        node_id: NodeId,
        identity: Hashable,
        event_facts: EventFacts,
        *,
        include_ownership: bool = True,
    ) -> NodeRuntimeFacts:
        bindings: dict[Any, NodeId] = {item.anchor: node_id for item in event_facts.message_identities}
        anchored = {
            field_name: tuple(
                replace(fact, anchor=BoundAnchor(node_id))
                for fact in getattr(event_facts, field_name)
                if bindings.get(fact.anchor) == node_id
            )
            for field_name in _ANCHORED_FACT_FIELDS
        }
        occurrence_facts = [
            fact
            for fact in (
                event_facts.membership if include_ownership else None,
                *anchored['spawn_intents'],
                *anchored['spawn_bindings'],
                *anchored['retrieval_intents'],
                *anchored['delivery_receipts'],
                *anchored['notification_deliveries'],
                event_facts.declared_claim if include_ownership else None,
                event_facts.parent_claim if include_ownership else None,
                *anchored['child_prompts'],
            )
            if fact is not None
        ]
        occurrences: list[RuntimeFactOccurrence] = []
        for fact in occurrence_facts:
            occurrences.append(RuntimeFactOccurrence(self._runtime_state.next_fact_sequence, fact))
            self._runtime_state.next_fact_sequence += 1
        return NodeRuntimeFacts(
            identity=identity,
            tools_identity=event_facts.tools_identity,
            context_key=event_facts.context_key,
            occurrences=tuple(occurrences),
        )

    def _merge_node_facts(self, node_id: NodeId, event_facts: EventFacts, index: int | None = None) -> None:
        current = self._runtime_state.facts.by_node[node_id]
        anchor = InputMessage(index) if index is not None else ObservationMessage()
        selected = replace(
            event_facts,
            message_identities=tuple(item for item in event_facts.message_identities if item.anchor == anchor),
        )
        incoming = self._bound_node_facts(node_id, current.identity, selected, include_ownership=False)
        self._runtime_state.facts.by_node[node_id] = replace(
            current,
            occurrences=current.occurrences + incoming.occurrences,
        )

    def _find_or_create_path(
        self,
        messages: list[dict],
        facts: EventFacts,
        request_sequence: int,
        render: PromptRender | None,
        *,
        frozen_node_ids: tuple[NodeId, ...] = (),
    ) -> tuple[NodeId | None, list[NodeId]]:
        """Attach the request's messages to the tree.

        The first ``len(frozen_node_ids)`` messages follow the frozen prefix's Nodes, so the stream
        the prompt resent and the path it is recorded on are the same Nodes; the rest match
        by identity and tokens, or become new Nodes.
        """
        if render is not None and len(messages) != len(render.templated_messages):
            raise ValueError('PromptRender message count does not match the request')
        logical_ids = self._logical_message_ids(facts, len(messages)) if render is not None else [None] * len(messages)
        templated_messages = render.templated_messages if render is not None else [None] * len(messages)
        # The render's primary partition is the training view, the way TokenAccount keeps it;
        # the engine's own spans ride beside it and fall back to the training ones.
        message_token_ids = render.message_token_ids if render is not None else [None] * len(messages)
        raw_message_token_ids: list[np.ndarray | None] = (
            list(render.rollout_message_token_ids) if render is not None else [None] * len(messages)
        )
        message_media_counts = render.message_media_counts if render is not None else None
        if message_media_counts is not None and len(message_media_counts) != len(messages):
            raise ValueError('PromptRender message media count does not match the request')
        current_id: NodeId | None = None
        newly_sealed_input_node_ids: list[NodeId] = []
        for index, (message, templated, token_ids, raw_token_ids, logical_id) in enumerate(
            zip(
                messages,
                templated_messages,
                message_token_ids,
                raw_message_token_ids,
                logical_ids,
                strict=True,
            )
        ):
            identity = self._identity_for(facts, InputMessage(index))
            semantic_children = self._find_children(current_id, identity)
            node_num_images = (
                int(message_media_counts[index].get('images', 0)) if message_media_counts is not None else 0
            )
            # A message that introduced images carries both views; a media-free message's
            # raw span is its training span, which is what None records.
            raw_token_ids = raw_token_ids if node_num_images > 0 else None
            if index < len(frozen_node_ids):
                child_id = frozen_node_ids[index]
            elif render is None:
                child_id = self._contextual_match(semantic_children, facts.context_key)
            else:
                token_matches = []
                for candidate_id in semantic_children:
                    account = self.trajectory.nodes[candidate_id].token_account
                    if (
                        account is not None
                        and np.array_equal(account.token_ids, token_ids)  # ty: ignore[invalid-argument-type]
                        and np.array_equal(account.raw_token_ids, raw_token_ids)  # ty: ignore[invalid-argument-type]
                    ):
                        token_matches.append(candidate_id)
                child_id = self._contextual_match(
                    token_matches,
                    facts.context_key,
                )
            if child_id is None and render is not None:
                child_id = next(
                    (
                        candidate_id
                        for candidate_id in semantic_children
                        if self.trajectory.nodes[candidate_id].generated_by != 'llm'
                        and self.trajectory.nodes[candidate_id].templated_message is None
                        and self.trajectory.nodes[candidate_id].token_account is None
                        and not self.trajectory.nodes[candidate_id].children
                    ),
                    None,
                )
                if child_id is not None:
                    child = self.trajectory.nodes[child_id]
                    child.templated_message = templated
                    child.token_account = TokenAccount(
                        token_ids=token_ids,
                        raw_token_ids=raw_token_ids,
                        num_images=node_num_images,
                    )
                    newly_sealed_input_node_ids.append(child_id)
            if child_id is None:
                anchor = InputMessage(index)
                child_id = self._new_node(
                    parent_id=current_id,
                    original_message=message,
                    generated_by='client',
                    identity=identity,
                    event_facts=replace(
                        facts,
                        message_identities=tuple(item for item in facts.message_identities if item.anchor == anchor),
                    ),
                    message_id=logical_id,
                    templated_message=templated,
                    token_account=(
                        TokenAccount(
                            token_ids=token_ids,
                            raw_token_ids=raw_token_ids,
                            num_images=node_num_images,
                        )
                        if render is not None
                        else None
                    ),
                )
                if render is not None:
                    newly_sealed_input_node_ids.append(child_id)
            else:
                self._record_variant(child_id, message, request_sequence)
                self._merge_node_facts(child_id, facts, index)
            current_id = child_id
        return current_id, newly_sealed_input_node_ids

    def _attach_multi_modal_items(
        self,
        current_id: NodeId | None,
        newly_sealed_input_node_ids: list[NodeId],
        multi_modal_items: dict[str, list[dict[str, Any]]] | None,
        message_media_counts: list[dict[str, int]] | None,
    ) -> None:
        """Give each message the processor tensors for the images it introduced.

        Only nodes this turn sealed are written: an already-sealed node published its media
        on the turn that sealed it.
        """
        if not multi_modal_items or not message_media_counts:
            return
        image_items = list(multi_modal_items.get('images') or [])
        sealed_input_node_ids = set(newly_sealed_input_node_ids)
        next_image_item_index = len(image_items)
        for node_id, media_counts in zip(
            reversed(self._node_path(current_id)),
            reversed(message_media_counts),
            strict=True,
        ):
            image_count = int(media_counts.get('images', 0))
            image_item_start = next_image_item_index - image_count
            if image_item_start < 0:
                # NOTE(arlenxu): a reused prefix may legitimately under-run, since its images
                # belong to an earlier turn's payload; a node sealed this turn may not.
                if node_id in sealed_input_node_ids:
                    raise ValueError(
                        f'message {node_id} declares {image_count} image(s) but only '
                        f'{next_image_item_index} processor tensor(s) remain for it; a family '
                        'that splits media must return one tensor per image'
                    )
                break
            message_image_items = image_items[image_item_start:next_image_item_index]
            next_image_item_index = image_item_start
            if not message_image_items or node_id not in sealed_input_node_ids:
                continue
            account = cast(TokenAccount, self.trajectory.nodes[node_id].token_account)
            account.multi_modal_items = {'images': message_image_items}

    def _logical_message_ids(self, facts: EventFacts, message_count: int) -> list[int | None]:
        candidates: list[NodeId | None] = [None]
        result: list[int | None] = []
        for identity in self._input_identities(facts, message_count):
            matches = [
                child_id for candidate_id in candidates for child_id in self._find_children(candidate_id, identity)
            ]
            message_ids = {self.trajectory.nodes[node_id].message_id for node_id in matches}
            message_id = next(iter(message_ids)) if len(message_ids) == 1 else None
            if message_id is None:
                contextual = {
                    self.trajectory.nodes[node_id].message_id
                    for node_id in matches
                    if self._runtime_state.facts.by_node[node_id].context_key == facts.context_key
                }
                message_id = next(iter(contextual)) if len(contextual) == 1 else None
            result.append(message_id)
            candidates = matches
        return result

    def _find_children(self, parent_id: NodeId | None, identity: Hashable) -> list[NodeId]:
        children = self.trajectory.roots if parent_id is None else self.trajectory.nodes[parent_id].children
        by_node = self._runtime_state.facts.by_node
        return [
            node_id
            for node_id in children
            if by_node[node_id].identity == identity
            or (by_node[node_id].alias_identity is not None and by_node[node_id].alias_identity == identity)
        ]

    def _contextual_match(self, matches: list[NodeId], context_key: Hashable) -> NodeId | None:
        if not matches:
            return None
        return next(
            (node_id for node_id in matches if self._runtime_state.facts.by_node[node_id].context_key == context_key),
            matches[0],
        )

    def _find_path_node(self, facts: EventFacts, message_count: int) -> NodeId | None:
        candidates: list[NodeId | None] = [None]
        for identity in self._input_identities(facts, message_count):
            candidates = [
                child_id for candidate_id in candidates for child_id in self._find_children(candidate_id, identity)
            ]
            if not candidates:
                return None
        contextual = [
            node_id
            for node_id in candidates
            if node_id is not None and self._runtime_state.facts.by_node[node_id].context_key == facts.context_key
        ]
        if not contextual:
            return None
        if len(contextual) > 1:
            raise ValueError('parent_messages match multiple physical trajectory paths')
        return contextual[0]

    def _node_path(self, node_id: NodeId | None) -> list[NodeId]:
        path: list[NodeId] = []
        current_id = node_id
        while current_id is not None:
            path.append(current_id)
            current_id = self.trajectory.nodes[current_id].parent
        path.reverse()
        return path

    def _path_divergence_message_index(self, previous_id: NodeId, current_id: NodeId | None) -> int | None:
        previous_path = self._node_path(previous_id)
        current_path = self._node_path(current_id)
        for index, (previous_node_id, current_node_id) in enumerate(zip(previous_path, current_path, strict=False)):
            if (
                self._runtime_state.facts.by_node[previous_node_id].identity
                != self._runtime_state.facts.by_node[current_node_id].identity
            ):
                return index
        return len(current_path) if len(current_path) < len(previous_path) else None

    def _record_variant(self, node_id: NodeId, message: dict[str, Any], request_sequence: int) -> None:
        node = self.trajectory.nodes[node_id]
        wire = copy_message(message)
        if wire == node.original_message:
            return
        for variant in node.as_sent_variants:
            if variant['message'] == wire:
                variant['occurrences'] += 1
                return
        node.as_sent_variants.append({'message': wire, 'first_request_seq': request_sequence, 'occurrences': 1})

    def _wire_messages_for_node(self, node_id: NodeId | None) -> list[dict[str, Any]]:
        path = self._node_path(node_id)
        for index in range(len(path) - 1, -1, -1):
            llm = self.trajectory.nodes[path[index]]
            if llm.generated_by != 'llm':
                continue
            inputs = [
                copy_message(
                    llm.request_message_overrides.get(input_id, self.trajectory.nodes[input_id].original_message)
                )
                for input_id in path[:index]
            ]
            return inputs + [copy_message(self.trajectory.nodes[item_id].original_message) for item_id in path[index:]]
        return [copy_message(self.trajectory.nodes[item_id].original_message) for item_id in path]

    @staticmethod
    def _validate_prompt_inputs(
        training: dict[str, Any] | None,
        render: PromptRender | None,
        completion_text: str | None,
    ) -> None:
        if training is not None and render is None:
            raise ValueError('PromptRender is required for token-level recording')
        if render is None:
            return
        if training is None:
            raise ValueError('token_training is required with PromptRender')
        if completion_text is None:
            raise ValueError('completion_text is required with PromptRender')
        if len(training['raw_prompt_ids']) == 0:
            raise ValueError('raw_prompt_ids are required for token-level recording')
        if not np.array_equal(training['raw_prompt_ids'], render.rollout_token_ids):
            raise ValueError('PromptRender token IDs differ from the raw model prompt IDs')
        if len(training['prompt_ids']) < len(training['raw_prompt_ids']):
            raise ValueError(
                f'prompt_ids ({len(training["prompt_ids"])} tokens) are shorter than '
                f'raw_prompt_ids ({len(training["raw_prompt_ids"])}); expansion may only add tokens'
            )
        if not any(render.message_media_counts or []) and not np.array_equal(
            training['prompt_ids'], training['raw_prompt_ids']
        ):
            raise ValueError(
                'expanded and raw prompt IDs differ but no multi-modal payload was recorded; '
                'expansion without media is a token-accounting bug'
            )

    def _request_boundary(self, facts: EventFacts, current_id: NodeId | None) -> PendingBoundary | None:
        path = self._node_path(current_id)
        anchor_bindings = {InputMessage(index): node_id for index, node_id in enumerate(path)}
        matches: list[tuple[BoundaryFact, Hashable]] = []
        for fact in facts.boundary_facts:
            anchor_id = anchor_bindings.get(fact.anchor)
            if anchor_id is None:
                continue
            anchor_context = self._runtime_state.facts.by_node[anchor_id].context_key
            if (
                facts.membership is not None
                and facts.declared_claim is None
                and not fact.from_request_context
                and anchor_context != facts.context_key
            ):
                continue
            if self._path_has_claim(current_id, fact.rule_name, anchor_id):
                continue
            matches.append((fact, anchor_id))
        return self._boundary_from_matches(matches)

    @classmethod
    def _output_boundary(cls, facts: EventFacts) -> PendingBoundary | None:
        return cls._boundary_from_matches([(fact, fact.anchor) for fact in facts.boundary_facts])

    @staticmethod
    def _boundary_from_matches(matches: list[tuple[BoundaryFact, Hashable]]) -> PendingBoundary | None:
        if not matches:
            return None
        consumed = list(matches)
        kinds = {fact.segment_kind for fact, _ in matches}
        if len(kinds) != 1:
            request_matches = [item for item in matches if item[0].from_request_context]
            edge_matches = [item for item in matches if item[0].at_request_edge]
            if request_matches and len({item[0].segment_kind for item in request_matches}) == 1:
                matches = request_matches
            elif edge_matches and len({item[0].segment_kind for item in edge_matches}) == 1:
                matches = edge_matches
            else:
                names = ', '.join(fact.rule_name for fact, _ in matches)
                raise ValueError(f'Boundary rules start incompatible segment kinds: {names}')
        rules = list(dict.fromkeys(fact.rule_name for fact, _ in matches))
        causes = tuple(dict.fromkeys(fact.cause for fact, _ in matches))
        return PendingBoundary(
            SegmentBoundary(
                starts_new_segment=True,
                kind=matches[0][0].segment_kind,
                causes=causes,
                evidence={'boundary_rules': rules},
            ),
            tuple((fact.rule_name, anchor) for fact, anchor in consumed),
        )

    def _path_has_claim(self, node_id: NodeId | None, rule_name: str, anchor_id: NodeId) -> bool:
        current_id = node_id
        while current_id is not None:
            if (rule_name, anchor_id) in self._runtime_state.facts.by_node[current_id].boundary_claims:
                return True
            current_id = self.trajectory.nodes[current_id].parent
        return False

    def _consume_pending_boundary(self, boundary: PendingBoundary | None) -> PendingBoundary | None:
        pending = self._runtime_state.pending_boundary
        self._runtime_state.pending_boundary = None
        return self._merge_boundaries(boundary, pending)

    @staticmethod
    def _merge_boundaries(
        primary: PendingBoundary | None,
        extra: PendingBoundary | None,
    ) -> PendingBoundary | None:
        if extra is None or not extra.boundary.starts_new_segment:
            return primary
        if primary is None or not primary.boundary.starts_new_segment:
            return extra
        primary_boundary = primary.boundary
        extra_boundary = extra.boundary
        evidence = {**extra_boundary.evidence, **primary_boundary.evidence}
        if primary_boundary.evidence.get('physical_only') is not True:
            evidence.pop('physical_only', None)
        causes = tuple(dict.fromkeys(extra_boundary.causes + primary_boundary.causes))
        return PendingBoundary(
            SegmentBoundary(
                starts_new_segment=True,
                kind=primary_boundary.kind or extra_boundary.kind,
                causes=causes,
                evidence=evidence,
            ),
            tuple(dict.fromkeys(extra.claims + primary.claims)),
        )

    def _physical_boundary(
        self,
        *,
        request_facts: EventFacts,
        current_id: NodeId | None,
        token_training: dict[str, Any] | None,
        tools: list[dict] | None = None,
        model_id: str | None = None,
    ) -> PendingBoundary | None:
        nearest_id = next(
            (
                llm_id
                for node_id in reversed(self._node_path(current_id))
                if (llm_id := self._runtime_state.llm_by_message_id.get(self.trajectory.nodes[node_id].message_id))
                is not None
            ),
            None,
        )
        if nearest_id is not None and self._chain_identity(nearest_id) == self._event_chain_identity(request_facts):
            previous_id = nearest_id
        else:
            previous_id = max(
                (
                    node_id
                    for node_id, node in self.trajectory.nodes.items()
                    if node.generated_by == 'llm'
                    and self._runtime_state.facts.by_node[node_id].context_key == request_facts.context_key
                ),
                default=None,
            )
        if previous_id is None:
            return None
        previous = self.trajectory.nodes[previous_id]
        previous_model_id = previous.provider_facts.model_id if previous.provider_facts is not None else None
        if previous_model_id != model_id:
            return self._physical_restart('model_restart', 'model_change')
        previous_account = previous.token_account
        previous_is_token = previous_account is not None
        current_is_token = token_training is not None
        if previous_is_token != current_is_token:
            return self._physical_restart('representation_restart', 'training_representation_discontinuity')
        previous_facts = self._runtime_state.facts.by_node[previous_id]
        if previous_facts.context_key != request_facts.context_key:
            return self._physical_restart('render_restart', 'render_context_discontinuity')
        if not current_is_token:
            if self._has_compatible_tools(previous_id, request_facts, tools):
                return None
            return self._physical_restart('render_restart', 'render_context_discontinuity')

        assert token_training is not None
        assert previous_account is not None
        raw_prompt_ids = token_training['raw_prompt_ids']
        previous_prompt = (
            np.concatenate((self._input_token_ids(previous.parent), previous_account.glue_ids))
            if previous_account.rollout_token_ids is not None
            else self._input_token_ids(previous.parent)
        )
        previous_stream = np.concatenate((
            previous_prompt,
            previous_account.generated_ids,
            previous_account.trailing_ids,
        ))
        if np.array_equal(raw_prompt_ids, previous_prompt) or np.array_equal(
            raw_prompt_ids[: len(previous_stream)], previous_stream
        ):
            return None
        divergence_message_index = self._path_divergence_message_index(previous_id, current_id)
        divergence = self._divergence_attribution(
            prompt_ids=raw_prompt_ids,
            previous_stream=previous_stream,
            previous_tools=previous_facts.tools_identity,
            current_tools=request_facts.tools_identity,
            path_divergence_message_index=divergence_message_index,
        )
        cause = 'history_rewrite' if divergence == 'history-rewrite' else 'render_divergence'
        evidence: dict[str, Any] = {'divergence': divergence, 'physical_only': True}
        if divergence == 'history-rewrite':
            evidence['path_divergence_message_index'] = divergence_message_index
        return PendingBoundary(
            SegmentBoundary(
                starts_new_segment=True,
                kind='render_restart',
                causes=(cause,),
                evidence=evidence,
            )
        )

    @staticmethod
    def _physical_restart(kind: str, cause: str) -> PendingBoundary:
        return PendingBoundary(
            SegmentBoundary(
                starts_new_segment=True,
                kind=kind,
                causes=(cause,),
                evidence={'physical_only': True},
            )
        )

    def _divergence_attribution(
        self,
        *,
        prompt_ids: np.ndarray,
        previous_stream: np.ndarray,
        previous_tools: Hashable,
        current_tools: Hashable,
        path_divergence_message_index: int | None,
    ) -> str:
        divergence_start = 0
        for previous_token, current_token in zip(previous_stream, prompt_ids, strict=False):
            if previous_token != current_token:
                break
            divergence_start += 1
        delta = tuple(prompt_ids[divergence_start:])
        terminator = tuple(self._policy.block_terminator_ids)
        if terminator and delta[: len(terminator)] == terminator:
            return 'terminator-residue'
        if previous_tools != current_tools:
            return 'context-change'
        if path_divergence_message_index is not None:
            return 'history-rewrite'
        return 'assumption-violation'

    def _chain_identity(self, node_id: NodeId) -> tuple[Any, ...]:
        memberships = self._runtime_state.facts.by_node[node_id].facts_of_type(SidechainMembership)
        membership = memberships[0][1] if memberships else None
        if membership is None:
            return ('main',)
        return (membership.chain_id, membership.kind, membership.detector)

    @staticmethod
    def _event_chain_identity(facts: EventFacts) -> tuple[Any, ...]:
        membership = facts.membership
        if membership is None:
            return ('main',)
        return (membership.chain_id, membership.kind, membership.detector)

    @staticmethod
    def _routing_length(nodes: list[TrajectoryNode]) -> int:
        token_length = 0
        routing_length = 0
        for node in nodes:
            if node.token_account is not None and node.token_account.token_ids is not None:
                token_length += len(node.token_account.token_ids)
                if node.generated_by == 'llm':
                    routing_length = token_length - 1
        return routing_length

    def _input_token_ids(self, node_id: NodeId | None) -> np.ndarray:
        """The rollout engine's input stream along the path: each message's own engine view."""
        token_chunks: list[np.ndarray] = []
        for path_id in self._node_path(node_id):
            account = self.trajectory.nodes[path_id].token_account
            rollout_ids = None if account is None else account.rollout_token_ids
            if rollout_ids is not None:
                token_chunks.append(rollout_ids)
        return np.concatenate(token_chunks) if token_chunks else np.empty(0, dtype=TOKEN_ID_DTYPE)

    @staticmethod
    def _provider_facts(
        output: dict[str, Any],
        stop_reason: str | None,
        response_meta: dict[str, Any] | None,
    ) -> ProviderFacts | None:
        meta = response_meta or {}
        facts = ProviderFacts(
            model=copy_value(meta.get('model')),
            model_id=copy_value(meta.get('model_id')),
            response_id=copy_value(meta.get('response_id')),
            usage=copy_value(meta.get('usage')),
            finish_reason=TrajectoryAssembler._finish_reason(output, stop_reason),
            inference_time=copy_value(meta.get('inference_time')),
            wall_inference_time=copy_value(meta.get('wall_inference_time')),
            abort_wait_time=copy_value(meta.get('abort_wait_time')),
            num_preempted=copy_value(meta.get('num_preempted')),
        )
        return facts if any(value is not None for value in vars(facts).values()) else None

    @staticmethod
    def _invocation(facts: EventFacts) -> InvocationFacts | None:
        if facts.iteration is None and facts.retry is None:
            return None
        return InvocationFacts(iteration=facts.iteration, retry=facts.retry)

    @staticmethod
    def _token_account(glue_ids: np.ndarray, training: dict[str, Any]) -> TokenAccount:
        response_ids = training['response_ids']
        extra = training['extra_fields']
        topk_indices = extra.get('resp_topk_logprob_indice')
        topk_values = extra.get('resp_topk_logprob_value')
        return TokenAccount(
            token_ids=np.concatenate((glue_ids, response_ids)),
            gen_start=len(glue_ids),
            gen_len=len(response_ids),
            logprobs=training['response_logprobs'],
            topk_indices=topk_indices,
            topk_values=topk_values,
            routed_experts=copy_value(training.get('routed_experts')),
        )

    @staticmethod
    def _normalize_token_training(token_training: dict[str, Any]) -> dict[str, Any]:
        prompt_ids = token_training.get('prompt_ids')
        raw_prompt_ids = token_training.get('raw_prompt_ids')
        response_ids = token_training.get('response_ids')
        response_logprobs = token_training.get('response_logprobs')
        extra_fields = dict(token_training.get('extra_fields') or {})
        # NOTE(jerryhrzhou): Convert dense top-k fields before copy_value would recurse over every scalar.
        topk_fields: dict[str, np.ndarray] = {}
        for name, dtype in (
            ('resp_topk_logprob_indice', TOKEN_ID_DTYPE),
            ('resp_topk_logprob_value', LOGPROB_DTYPE),
        ):
            if (value := extra_fields.pop(name, None)) is None:
                continue
            try:
                rows = np.array(value, dtype=dtype)
            except ValueError as exc:
                raise ValueError(
                    f'{name} must be a rectangular numeric array with shape [gen_len, K]; '
                    'each generated token must have the same K entries'
                ) from exc
            if rows.shape == (0,):
                rows = rows.reshape(0, 0)
            elif rows.ndim != 2:
                raise ValueError(f'{name} must have shape [gen_len, K], got shape {rows.shape}')
            topk_fields[name] = rows
        return {
            'prompt_ids': np.asarray(prompt_ids if prompt_ids is not None else [], dtype=TOKEN_ID_DTYPE),
            'raw_prompt_ids': np.asarray(raw_prompt_ids if raw_prompt_ids is not None else [], dtype=TOKEN_ID_DTYPE),
            'response_ids': np.array(response_ids if response_ids is not None else [], dtype=TOKEN_ID_DTYPE),
            'response_logprobs': np.array(response_logprobs, dtype=LOGPROB_DTYPE)
            if response_logprobs is not None
            else None,
            'routed_experts': token_training.get('routed_experts'),
            'multi_modal_items': copy_value(token_training.get('multi_modal_items')),
            'extra_fields': {**copy_value(extra_fields), **topk_fields},
        }

    @staticmethod
    def _finish_reason(message: dict[str, Any], stop_reason: str | None) -> str:
        if stop_reason == 'length':
            return 'length'
        if message.get('tool_calls'):
            return 'tool_calls'
        return stop_reason or 'stop'
