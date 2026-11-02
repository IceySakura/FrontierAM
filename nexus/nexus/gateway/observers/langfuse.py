# /// script
# dependencies = ["langfuse>=4.0.0"]
#
# [tool.nexus]
# trajectory_observers = { langfuse = "nexus.gateway.observers.langfuse" }
# ///

"""Export finalized canonical Trajectory Trees to Langfuse."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import heapq
import importlib
import json
import logging
import math
import os
import threading
import uuid

from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any, cast

from nexus.gateway.observers.base import TrajectoryObserver, TrajectoryUpdate
from nexus.gateway.observers.registry import trajectory_observer_registry
from nexus.gateway.trajectory import Chain, SegmentNodeList, Trajectory, TrajectoryNode


logger = logging.getLogger(__name__)

_TAIJI_METADATA_ENV = {
    'instance_id': 'TJ_INSTANCE_ID',
    'task_id': 'TJ_TASK_ID',
    'task_name': 'TJ_TASK_NAME',
    'workspace_id': 'TJ_WORKSPACE_ID',
}
_NANOSECONDS_PER_SECOND = 1_000_000_000
_DISPLAY_TICK_NS = 1_000_000  # Langfuse's graph compares JavaScript millisecond timestamps.
_AGENT_CHAIN_KINDS = {'main', 'subagent', 'thread_spawn'}


@dataclass(frozen=True)
class LangfuseTrajectoryConfig:
    """Configure finalized Trajectory Tree export to Langfuse."""

    public_key: str = ''
    secret_key: str = ''
    host: str = ''
    trace_name: str = ''
    session_id: str = ''
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    final_only: bool = True
    flush_on_final: bool = True
    max_tool_chars: int = 8000
    max_content_chars: int = 0

    def __post_init__(self) -> None:
        if not self.final_only:
            raise ValueError('Langfuse trajectory export supports only final_only=True')

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> LangfuseTrajectoryConfig:
        data = dict(value or {})
        data.pop('enabled', None)
        if 'trace_metadata' in data:
            if 'metadata' in data:
                raise ValueError('Configure either metadata or trace_metadata, not both')
            data['metadata'] = data.pop('trace_metadata')
        if 'tags' in data:
            data['tags'] = tuple(data['tags'] or ())
        return cls(**data)


@dataclass(frozen=True)
class _TimeBounds:
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class _Observation:
    key: str
    parent_key: str | None
    kind: str
    name: str
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    model_parameters: dict[str, Any] | None = None
    usage: dict[str, int] | None = None
    time_bounds: _TimeBounds | None = None

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {'input': self.input, 'output': self.output, 'metadata': self.metadata}
        if self.model:
            payload['model'] = self.model
        if self.model_parameters:
            payload['model_parameters'] = self.model_parameters
        if self.usage and self.usage.get('total', 0) > 0:
            payload['usage_details'] = self.usage
        return payload


@dataclass(frozen=True)
class _Presentation:
    observations: tuple[_Observation, ...]
    session_id: str
    trace_name: str
    tags: tuple[str, ...]
    score: float | None
    score_comment: str


class LangfuseTrajectoryMapper:
    """Map canonical Trajectory objects to a chain-owned Langfuse observation tree."""

    def __init__(self, config: LangfuseTrajectoryConfig) -> None:
        self._config = config

    def map(self, trajectory: Trajectory) -> _Presentation:
        nodes = self._nodes_in_tree_order(trajectory)
        node_chain = {node.node_id: trajectory.node_chain(node.node_id) for node in nodes}
        node_times = {
            node.node_id: self._timeline_bounds(node.timeline, owner=f'node {node.node_id}') for node in nodes
        }
        segments = trajectory.to_segment_node_lists()
        response_segments = {
            node.node_id: segment.segment_id for segment in segments for node in segment.response_nodes
        }
        chain_segment_ids: dict[str, list[str]] = {chain_id: [] for chain_id in trajectory.chains}
        for segment in segments:
            chain_segment_ids[segment.chain_id].append(segment.segment_id)
        trace_metadata = {
            **self._trace_metadata(trajectory),
            'segment_count': len(segments),
            'segment_ids': [segment.segment_id for segment in segments],
        }
        trace_name = (
            self._config.trace_name
            or str(trace_metadata.get('instance_id') or trace_metadata.get('task_name') or '')
            or f'Trajectory {trajectory.session_id}'
        )
        session_id = self._config.session_id or str(trace_metadata.get('episode_id') or trajectory.session_id)
        chains = {chain.chain_id: chain for chain in trajectory.chains.values()}
        chain_node_ids = {
            chain_id: [node.node_id for node in nodes if node_chain[node.node_id] == chain_id] for chain_id in chains
        }
        chain_parents = self._chain_parents(chains)
        time_metadata = self._fill_input_times(nodes, node_times, node_chain, chains, chain_node_ids, chain_parents)
        chain_times = self._chain_time_bounds(chain_node_ids, node_times, chain_parents)
        trajectory_time = self._combine_bounds([
            self._timeline_bounds(trajectory.timeline, owner=f'trajectory {trajectory.session_id}'),
            *node_times.values(),
        ])
        chain_io = {
            chain_id: self._chain_io(trajectory, nodes, chain_id, node_chain)
            for chain_id in dict.fromkeys([*chains, *node_chain.values()])
        }
        root_io = (
            {chain_id: values[0] for chain_id, values in chain_io.items() if chain_parents.get(chain_id) is None},
            {chain_id: values[1] for chain_id, values in chain_io.items() if chain_parents.get(chain_id) is None},
        )
        observations: dict[str, _Observation] = {
            'trace': self._trace_observation(trajectory, trace_name, trace_metadata, trajectory_time, root_io)
        }
        segment_anchors: dict[int, int | None] = {}
        for chain in chains.values():
            display_parent_chain_id = chain_parents[chain.chain_id]
            parent_key = self._chain_key(display_parent_chain_id) if display_parent_chain_id is not None else 'trace'
            observations[self._chain_key(chain.chain_id)] = self._chain_observation(
                chain,
                member_node_ids=chain_node_ids[chain.chain_id],
                display_parent_chain_id=display_parent_chain_id,
                parent_key=parent_key,
                time_bounds=chain_times[chain.chain_id],
                io=chain_io[chain.chain_id],
                segment_ids=chain_segment_ids[chain.chain_id],
            )
        for segment in segments:
            observation = self._segment_observation(segment, node_times)
            observations[observation.key] = observation
        for node in nodes:
            chain_id = node_chain[node.node_id]
            segment_anchors[node.node_id] = (
                node.node_id
                if node.segment_boundary is not None and node.segment_boundary.starts_new_segment
                else segment_anchors.get(node.parent)
                if node.parent is not None and node_chain.get(node.parent) == chain_id
                else None
            )
            # Physical message ancestry belongs in metadata, not the observation hierarchy.
            parent_key = self._chain_key(chain_id) if chain_id in chains else 'trace'
            observations[self._node_key(node.node_id)] = self._node_observation(
                node,
                chain_id=chain_id,
                parent_key=parent_key,
                time_bounds=node_times[node.node_id],
                time_metadata=time_metadata[node.node_id],
                trajectory=trajectory,
                segment_anchor=segment_anchors[node.node_id],
                segment_id=response_segments.get(node.node_id),
            )
        observations = self._add_shared_system_context(trajectory, nodes, node_chain, observations)
        ordered = self._order_by_parent(observations)
        models = [model for node in nodes if (model := self._model(node)[0])]
        tags = self._dedupe([
            *self._config.tags,
            trace_metadata.get('instance_id'),
            trace_metadata.get('task_id'),
            trace_metadata.get('task_name'),
            trace_metadata.get('workspace_id'),
            *(model.removeprefix('openai/') for model in models),
        ])
        return _Presentation(
            observations=tuple(ordered),
            session_id=session_id,
            trace_name=trace_name,
            tags=tuple(tags),
            score=self._finite_number(trajectory.final_reward),
            score_comment=f'trajectory={trajectory.session_id}',
        )

    def _add_shared_system_context(
        self,
        trajectory: Trajectory,
        nodes: tuple[TrajectoryNode, ...],
        node_chain: dict[int, str],
        observations: dict[str, _Observation],
    ) -> dict[str, _Observation]:
        shared_roots: dict[tuple[str, int], list[int]] = {}
        for chain in trajectory.chains.values():
            if chain.kind not in {'subagent', 'thread_spawn'}:
                continue
            for root_id in sorted(chain.roots):
                root = trajectory.nodes[root_id]
                source = trajectory.nodes.get(root.parent) if root.parent is not None else None
                if (
                    root.original_message.get('role') == 'user'
                    and source is not None
                    and source.original_message.get('role') == 'system'
                    and source.generated_by != 'llm'
                    and node_chain[root_id] == chain.chain_id
                    and node_chain[source.node_id] != chain.chain_id
                ):
                    shared_roots.setdefault((chain.chain_id, source.node_id), []).append(root_id)
        if not shared_roots:
            return observations

        copied_roots = {root_id for roots in shared_roots.values() for root_id in roots}
        sources = {source_id for _, source_id in shared_roots}
        owner_consumed = {
            item.node_id
            for node in nodes
            if node.generated_by == 'llm'
            for item in self._input_path(trajectory, node)
            if item.node_id in sources and node_chain[item.node_id] == node_chain[node.node_id]
        }
        # Hide only pure shared prefixes; retain any uncovered branch or owner use.
        hidden = {
            self._node_key(source_id)
            for source_id in sources - owner_consumed
            if set(trajectory.nodes[source_id].children) <= copied_roots
        }
        before: dict[str, list[_Observation]] = {}
        positions = {key: index for index, key in enumerate(observations)}
        for (chain_id, source_id), roots in shared_roots.items():
            source = trajectory.nodes[source_id]
            timed_roots = [
                (bounds.start_ns, root_id)
                for root_id in roots
                if (bounds := observations[self._node_key(root_id)].time_bounds) is not None
            ]
            anchor = min(timed_roots, default=None)
            point = anchor[0] - _DISPLAY_TICK_NS if anchor is not None else None
            metadata: dict[str, Any] = {
                'object_type': 'shared_context',
                'source_node_id': source_id,
                'source_chain_id': node_chain[source_id],
                'chain_id': chain_id,
                'shared_context': True,
                'role': 'system',
                'root_node_ids': roots,
                'source_timeline': copy.deepcopy(source.timeline),
                'timestamp_source': 'inferred_shared_context' if point is not None else 'missing',
                'time_semantics': 'display_point_not_creation_or_execution' if point is not None else 'unknown',
            }
            if anchor is not None:
                metadata.update(
                    anchor_node_id=anchor[1],
                    anchor_time_ns=anchor[0],
                    display_offset_ns=-_DISPLAY_TICK_NS,
                    display_order_resolved=len(timed_roots) == len(roots),
                )
            else:
                metadata['export_timestamp_semantics'] = 'sdk_default_not_execution_time'
            if source.token_account is not None:
                metadata['token_account'] = self._token_account_summary(source)
            # A context reference has no execution, model, usage or reward of its own.
            shared = _Observation(
                key=f'shared:{chain_id}:node:{source_id}',
                parent_key=self._chain_key(chain_id),
                kind='span',
                name=observations[self._node_key(source_id)].name,
                input=self._safe(self._message(source.original_message)),
                metadata=self._safe(metadata),
                time_bounds=_TimeBounds(point, point) if point is not None else None,
            )
            first_root = min((self._node_key(root_id) for root_id in roots), key=positions.__getitem__)
            before.setdefault(first_root, []).append(shared)

        displayed: dict[str, _Observation] = {}
        for key, observation in observations.items():
            for shared in before.get(key, []):
                displayed[shared.key] = shared
            if key not in hidden:
                displayed[key] = observation
        children: dict[str, list[str]] = {key: [] for key in displayed}
        for observation in displayed.values():
            if observation.parent_key is not None:
                children[observation.parent_key].append(observation.key)
        containers = {'trace', *(self._chain_key(chain_id) for chain_id in trajectory.chains)}
        for observation in reversed(self._order_by_parent(displayed)):
            if observation.key not in containers:
                continue
            bounds = self._combine_bounds([
                *(displayed[key].time_bounds for key in children[observation.key]),
                self._timeline_bounds(trajectory.timeline, owner=f'trajectory {trajectory.session_id}')
                if observation.key == 'trace'
                else None,
            ])
            displayed[observation.key] = replace(
                observation,
                time_bounds=bounds,
                metadata={
                    **observation.metadata,
                    'timestamp_source': 'derived_display_bounds' if bounds is not None else 'missing',
                },
            )
        return displayed

    def _trace_observation(
        self,
        trajectory: Trajectory,
        trace_name: str,
        trace_metadata: dict[str, Any],
        time_bounds: _TimeBounds | None,
        io: tuple[dict[str, Any], dict[str, Any]],
    ) -> _Observation:
        metadata = {
            **trace_metadata,
            'object_type': 'trajectory',
            'schema_version': trajectory.schema_version,
            'session_id': trajectory.session_id,
            'state': trajectory.state,
            'root_node_ids': list(trajectory.roots),
            'node_count': len(trajectory.nodes),
            'chain_count': len(trajectory.chains),
            'chain_ids': list(trajectory.chains),
            'latest_message_count': sum(len(trajectory.to_messages(chain_id)) for chain_id in trajectory.chains),
            'timeline': copy.deepcopy(trajectory.timeline),
            'metrics': copy.deepcopy(trajectory.metrics),
            'extra': copy.deepcopy(trajectory.extra),
            'timestamp_source': 'derived_display_bounds' if time_bounds is not None else 'missing',
        }
        output = {
            'state': trajectory.state,
            'termination_reason': trajectory.termination_reason,
            'chains': io[1],
        }
        return _Observation(
            key='trace',
            parent_key=None,
            kind='span',
            name=trace_name,
            input=self._safe({'root_node_ids': trajectory.roots, 'chains': io[0]}),
            output=self._safe(output),
            metadata=self._safe(metadata),
            time_bounds=time_bounds,
        )

    @staticmethod
    def _chain_metadata(chain: Chain) -> dict[str, Any]:
        spawn = None
        if chain.spawn is not None:
            spawn = {
                'node': chain.spawn.node,
                'tool_use_id': chain.spawn.tool_use_id,
                'candidates': list(chain.spawn.candidates),
            }
        join = None
        if chain.join is not None:
            join = {
                'node': chain.join.node,
                'tool_use_id': chain.join.tool_use_id,
                'kind': chain.join.kind,
            }
        return {
            'chain_id': chain.chain_id,
            'kind': chain.kind,
            'parent_chain_id': chain.parent_chain_id,
            'root_node_ids': list(chain.roots),
            'spawn': spawn,
            'join': join,
            'detector': chain.detector,
            'evidence': copy.deepcopy(chain.evidence),
        }

    def _chain_observation(
        self,
        chain: Chain,
        *,
        member_node_ids: list[int],
        display_parent_chain_id: str | None,
        parent_key: str,
        time_bounds: _TimeBounds | None,
        io: tuple[dict[str, Any], dict[str, Any]],
        segment_ids: list[str],
    ) -> _Observation:
        metadata = {
            'object_type': 'trajectory_chain',
            'segment_ids': segment_ids,
            'segment_count': len(segment_ids),
            **self._chain_metadata(chain),
            'display_parent_chain_id': display_parent_chain_id,
            'member_node_ids': list(member_node_ids),
            'member_node_count': len(member_node_ids),
            'timestamp_source': 'derived_display_bounds' if time_bounds is not None else 'missing',
            'time_semantics': 'container_display_extent_not_agent_execution_duration',
        }
        return _Observation(
            key=self._chain_key(chain.chain_id),
            parent_key=parent_key,
            kind='agent' if chain.kind in _AGENT_CHAIN_KINDS else 'span',
            name=f'Chain {chain.chain_id}: {chain.kind}',
            input=self._safe({'root_node_ids': chain.roots, **io[0]}),
            output=self._safe({'join': metadata['join'], **io[1]}),
            metadata=self._safe(metadata),
            time_bounds=time_bounds,
        )

    def _segment_observation(
        self,
        segment: SegmentNodeList,
        node_times: dict[int, _TimeBounds | None],
    ) -> _Observation:
        responses = segment.response_nodes
        source_ids = [node.node_id for node in segment.nodes]
        target_ids = [node.node_id for node in responses]
        boundary = responses[0].segment_boundary
        kind = (boundary.kind or 'boundary') if boundary is not None else 'conversation'
        last = source_ids.index(target_ids[-1])
        target_times = [node_times[node_id] for node_id in target_ids]
        bounds = self._combine_bounds(target_times)
        tokens = self._segment_tokens(segment)
        metadata = {
            'object_type': 'trajectory_segment',
            'segment_id': segment.segment_id,
            'chain_id': segment.chain_id,
            'kind': kind,
            'source_node_ids': source_ids,
            'target_node_ids': target_ids,
            'first_response_node_index': segment.first_response_node_index,
            'boundary': boundary.to_dict() if boundary is not None else None,
            'tokens': tokens,
            'tokens_available': tokens is not None,
            'timestamp_source': 'derived_target_node_bounds' if bounds is not None else 'missing',
            'timestamps_complete': all(item is not None for item in target_times),
            'time_semantics': 'target_llm_extent_not_segment_execution_duration',
        }
        if bounds is None:
            metadata['export_timestamp_semantics'] = 'sdk_default_not_execution_time'
        # Segments reference shared context; they neither own nor duplicate Node observations.
        return _Observation(
            key=f'segment:{segment.segment_id}',
            parent_key=self._chain_key(segment.chain_id),
            kind='span',
            name=f'Segment {segment.segment_id}: {kind}',
            input=self._safe({
                'source_node_ids': source_ids,
                'context_node_ids': source_ids[: segment.first_response_node_index],
                'trailing_node_ids': source_ids[last + 1 :],
            }),
            output=self._safe({'target_node_ids': target_ids, 'tokens': tokens, 'boundary': metadata['boundary']}),
            metadata=self._safe(metadata),
            time_bounds=bounds,
        )

    @staticmethod
    def _segment_tokens(segment: SegmentNodeList) -> dict[str, int] | None:
        last = segment.nodes.index(segment.last_response_node)
        accounts = [node.token_account for node in segment.nodes[: last + 1]]
        if any(account is None or account.token_ids is None for account in accounts):
            return None
        lengths = [
            len(account.token_ids) for account in accounts if account is not None and account.token_ids is not None
        ]
        first = segment.first_response_node_index
        first_account = accounts[first]
        assert first_account is not None
        return {
            'prompt': sum(lengths[:first]) + first_account.gen_start,
            'response': sum(lengths[first:]) - first_account.gen_start,
            'generated': sum(account.gen_len for account in accounts[first:] if account is not None),
        }

    def _node_observation(
        self,
        node: TrajectoryNode,
        *,
        chain_id: str,
        parent_key: str,
        time_bounds: _TimeBounds | None,
        time_metadata: dict[str, Any],
        trajectory: Trajectory,
        segment_anchor: int | None,
        segment_id: str | None,
    ) -> _Observation:
        message = self._message(node.original_message)
        role = str(node.original_message.get('role') or '')
        provider = node.provider_facts
        metadata: dict[str, Any] = {
            'object_type': 'trajectory_node',
            'node_id': node.node_id,
            'message_id': node.message_id,
            'parent_node_id': node.parent,
            'child_node_ids': list(node.children),
            'chain_id': chain_id,
            'generated_by': node.generated_by,
            'role': role or None,
            'timeline': copy.deepcopy(node.timeline),
            'selection': node.selection,
            'reward': node.reward,
            **time_metadata,
            'segment_anchor_node_id': segment_anchor,
        }
        if segment_id is not None:
            metadata['segment_id'] = segment_id
        if node.templated_message is not None:
            metadata['templated_message'] = self._trunc(node.templated_message, self._config.max_content_chars)
        if node.token_account is not None:
            metadata['token_account'] = self._token_account_summary(node)
        if provider is not None:
            provider_metadata = provider.to_dict()
            provider_metadata.pop('model', None)
            provider_metadata.pop('usage', None)
            metadata['provider_facts'] = provider_metadata
        if node.segment_boundary is not None:
            metadata['segment_boundary'] = node.segment_boundary.to_dict()
        if node.invocation is not None:
            metadata['invocation'] = node.invocation.to_dict()
        kind = self._node_kind(node)
        name = self._node_name(node)
        if node.generated_by == 'llm':
            path = self._input_path(trajectory, node)
            request_input: dict[str, Any] = {
                'messages': [
                    self._message(node.request_message_overrides.get(item.node_id, item.original_message))
                    for item in path
                ],
            }
            metadata['input_node_ids'] = [item.node_id for item in path]
            metadata['input_source'] = 'physical_path_with_request_overrides'
            if node.request_kwargs and 'tools' in node.request_kwargs:
                request_input['tools'] = copy.deepcopy(node.request_kwargs['tools'])
            if node.request_kwargs is not None:
                request_input['request_kwargs'] = copy.deepcopy(node.request_kwargs)
            if node.request_message_overrides:
                request_input['request_message_overrides'] = [
                    {'node_id': node_id, 'message': copy.deepcopy(override)}
                    for node_id, override in sorted(node.request_message_overrides.items())
                ]
            if node.as_sent_variants:
                request_input['as_sent_variants'] = copy.deepcopy(node.as_sent_variants)
            observation_input: Any = self._safe(request_input) if request_input else None
            observation_output: Any = self._safe(message)
        else:
            observation_input = self._safe(message)
            observation_output = None
            if kind == 'tool':
                call = self._tool_call(trajectory, node, chain_id)
                observation_input = None if call is None else self._safe(call[1].get('function', call[1]))
                observation_output = self._safe(message)
                metadata['tool_call_id'] = message.get('tool_call_id')
                metadata['tool_call_node_id'] = None if call is None else call[0]
                metadata['tool_semantics'] = 'result_message_not_inferred_execution'
                if call is not None:
                    tool_name = call[1].get('function', call[1]).get('name')
                    if tool_name:
                        name = f'Tool Result: {tool_name} [node {node.node_id}]'
            deliveries = [
                chain.chain_id
                for chain in trajectory.chains.values()
                if chain.kind in _AGENT_CHAIN_KINDS and chain.join is not None and chain.join.node == node.node_id
            ]
            if deliveries:
                metadata['agent_result_chain_ids'] = deliveries
                if all(
                    (join := trajectory.chains[item].join) is not None and join.kind in {'inline', 'notify'}
                    for item in deliveries
                ):
                    kind = 'span'
                    name = f'Agent Result [node {node.node_id}]'
        model, model_source = self._model(node)
        if node.generated_by == 'llm':
            metadata['model_source'] = model_source
        if time_bounds is None:
            metadata['export_timestamp_semantics'] = 'sdk_default_not_execution_time'
        return _Observation(
            key=self._node_key(node.node_id),
            parent_key=parent_key,
            kind=kind,
            name=name,
            input=observation_input,
            output=observation_output,
            metadata=self._safe(metadata),
            model=model,
            model_parameters=self._model_parameters(node.request_kwargs),
            usage=self._usage(None if provider is None else provider.usage),
            time_bounds=time_bounds,
        )

    def _token_account_summary(self, node: TrajectoryNode) -> dict[str, Any]:
        account = node.token_account
        assert account is not None, f'node {node.node_id} token account is missing'
        token_count = len(account.token_ids) if account.token_ids is not None else None
        # Export recorded training-view IDs only; never retokenize messages or fetch
        # remote payloads in this synchronous observer. Missing is not an empty list.
        return {
            'token_ids': account.token_ids.tolist() if account.token_ids is not None else None,
            'gen_start': account.gen_start,
            'gen_len': account.gen_len,
            'token_count': token_count,
            'trailing_token_count': (
                max(0, token_count - account.gen_start - account.gen_len) if token_count is not None else None
            ),
            'has_routed_experts': account.has_routed_experts,
            'has_payload_ref': account.payload_ref is not None,
            'num_images': account.num_images,
        }

    @staticmethod
    def _chain_parents(chains: dict[str, Chain]) -> dict[str, str | None]:
        parents = {
            chain_id: chain.parent_chain_id if chain.parent_chain_id in chains else None
            for chain_id, chain in chains.items()
        }
        cycle_members: set[str] = set()
        settled: set[str] = set()
        for start in chains:
            trail: list[str] = []
            positions: dict[str, int] = {}
            current: str | None = start
            while current is not None and current not in settled:
                if current in positions:
                    cycle_members.update(trail[positions[current] :])
                    break
                positions[current] = len(trail)
                trail.append(current)
                current = parents[current]
            settled.update(trail)
        for chain_id in cycle_members:
            parents[chain_id] = None
        return parents

    @classmethod
    def _fill_input_times(
        cls,
        nodes: tuple[TrajectoryNode, ...],
        node_times: dict[int, _TimeBounds | None],
        node_chain: dict[int, str],
        chains: dict[str, Chain],
        chain_node_ids: dict[str, list[int]],
        chain_parents: dict[str, str | None],
    ) -> dict[int, dict[str, Any]]:
        facts = dict(node_times)
        chain_nodes: dict[str, list[TrajectoryNode]] = {}
        for node in nodes:
            chain_nodes.setdefault(node_chain[node.node_id], []).append(node)
        metadata = {
            node.node_id: {
                'timestamp_source': 'trajectory_tree' if facts[node.node_id] is not None else 'missing',
                'display_order': index,
                'display_order_source': 'chain_node_id_with_physical_precedence',
                'display_offset_ns': 0,
                'time_semantics': (
                    'recorded_first_seen_not_execution'
                    if node.timeline and all(event.get('event_type') == 'input_first_seen' for event in node.timeline)
                    else 'recorded_timeline'
                    if facts[node.node_id] is not None
                    else 'unknown'
                ),
            }
            for members in chain_nodes.values()
            for index, node in enumerate(members)
        }
        first_requests: dict[int, tuple[int, int] | None] = {}
        for node in reversed(nodes):
            candidates = [value for child in node.children if (value := first_requests[child]) is not None]
            bounds = facts[node.node_id]
            if node.generated_by == 'llm' and bounds is not None:
                candidates.append((bounds.start_ns, node.node_id))
            first_requests[node.node_id] = min(candidates, default=None)

        # Only recorded child-agent times constrain delivery, never synthetic
        # container/display times. Include nested agents through chain ancestry.
        chain_facts = cls._chain_time_bounds(chain_node_ids, facts, chain_parents)
        join_ends: dict[int, int] = {}
        for chain in chains.values():
            bounds = chain_facts[chain.chain_id]
            if (
                chain.kind in _AGENT_CHAIN_KINDS
                and chain.join is not None
                and chain.join.node is not None
                and bounds is not None
            ):
                node_id = chain.join.node
                join_ends[node_id] = max(join_ends.get(node_id, bounds.end_ns), bounds.end_ns)
        lower_bounds: dict[int, int | None] = {}
        strict_bounds: dict[int, int | None] = {}
        for node in nodes:
            parent_end = lower_bounds.get(node.parent) if node.parent is not None else None
            parent_strict = strict_bounds.get(node.parent) if node.parent is not None else None
            bounds = facts[node.node_id]
            # A recorded point also needs a distinct millisecond before an input
            # can be claimed to sort strictly after it in Langfuse.
            recorded_end = (
                max(bounds.end_ns, (bounds.start_ns // _DISPLAY_TICK_NS + 1) * _DISPLAY_TICK_NS)
                if bounds is not None
                else None
            )
            lower_bounds[node.node_id] = max(
                (
                    end
                    for end in (parent_end, bounds.end_ns if bounds else None, join_ends.get(node.node_id))
                    if end is not None
                ),
                default=None,
            )
            strict_bounds[node.node_id] = max(
                (end for end in (parent_strict, recorded_end, lower_bounds[node.node_id]) if end is not None),
                default=None,
            )
        display_bounds = dict(chain_facts)
        # _chain_time_bounds visits children before parents, so shared prefixes
        # can use the completed child display extent without changing real times.
        for chain_id in dict.fromkeys([*chain_facts, *chain_nodes]):
            members = chain_nodes.get(chain_id, [])
            preceding_end: int | None = None
            preceding_strict: int | None = None
            runs: list[list[TrajectoryNode]] = []
            for node in members:
                lower = lower_bounds[node.node_id]
                preceding_end = max((end for end in (preceding_end, lower) if end is not None), default=None)
                lower_bounds[node.node_id] = preceding_end
                preceding_strict = max(
                    (end for end in (preceding_strict, strict_bounds[node.node_id]) if end is not None),
                    default=None,
                )
                strict_bounds[node.node_id] = preceding_strict
                anchor = first_requests[node.node_id]
                eligible = facts[node.node_id] is None and node.generated_by != 'llm' and anchor is not None
                if (
                    eligible
                    and runs
                    and facts[runs[-1][-1].node_id] is None
                    and runs[-1][-1].generated_by != 'llm'
                    and first_requests[runs[-1][-1].node_id] == anchor
                ):
                    runs[-1].append(node)
                else:
                    runs.append([node])
            next_start: int | None = None
            for inputs in reversed(runs):
                first = inputs[0]
                anchor = first_requests[first.node_id]
                bounds = facts[first.node_id]
                if bounds is not None:
                    next_start = bounds.start_ns
                    continue
                if first.generated_by == 'llm' or anchor is None:
                    continue
                start, anchor_id = anchor
                upper = min(
                    [start if next_start is None else min(start, next_start)]
                    + [
                        bounds.start_ns
                        for node in inputs
                        for child in node.children
                        if chain_parents.get(node_chain[child]) == chain_id
                        and (bounds := display_bounds.get(node_chain[child])) is not None
                    ]
                )
                points = [upper + (index - len(inputs)) * _DISPLAY_TICK_NS for index in range(len(inputs))]
                can_order = all(
                    (lower := strict_bounds[node.node_id]) is None or point >= lower
                    for node, point in zip(inputs, points, strict=True)
                )
                if not can_order:
                    logger.warning(
                        'Langfuse display order unresolved for chain %s nodes %s: '
                        'insufficient gap or conflicting recorded bounds before request node %s; '
                        'recorded timestamps are unchanged',
                        chain_id,
                        [node.node_id for node in inputs],
                        anchor_id,
                    )
                for node, point in zip(inputs, points, strict=True):
                    lower = lower_bounds[node.node_id]
                    # A tie is preferable to moving real work. If even the tie
                    # violates completion evidence, leave the timestamp missing.
                    point = point if can_order else upper
                    conflict = lower is not None and point < lower
                    details = metadata[node.node_id]
                    details.update({
                        'anchor_node_id': anchor_id,
                        'anchor_time_ns': start,
                        'display_lower_bound_ns': lower,
                        'display_order_resolved': can_order,
                    })
                    if not can_order:
                        details['display_order_reason'] = (
                            'conflicting_recorded_bounds' if conflict else 'insufficient_time_gap'
                        )
                    if not conflict:
                        node_times[node.node_id] = _TimeBounds(point, point)
                        details.update({
                            'timestamp_source': 'inferred_first_request',
                            'display_offset_ns': point - start,
                            'time_semantics': 'display_point_not_creation_or_execution',
                        })
                assigned = [node_times[node.node_id] for node in inputs]
                next_start = min((bounds.start_ns for bounds in assigned if bounds is not None), default=next_start)
            display_bounds[chain_id] = cls._combine_bounds([
                *(node_times[node.node_id] for node in members),
                *(display_bounds[child] for child, parent in chain_parents.items() if parent == chain_id),
            ])
        return metadata

    @classmethod
    def _chain_time_bounds(
        cls,
        chain_node_ids: dict[str, list[int]],
        node_times: dict[int, _TimeBounds | None],
        chain_parents: dict[str, str | None],
    ) -> dict[str, _TimeBounds | None]:
        """Return container bounds in child-before-parent insertion order."""
        children: dict[str, list[str]] = {chain_id: [] for chain_id in chain_node_ids}
        for chain_id, parent_chain_id in chain_parents.items():
            if parent_chain_id is not None:
                children[parent_chain_id].append(chain_id)
        bounds: dict[str, _TimeBounds | None] = {}

        def visit(chain_id: str) -> _TimeBounds | None:
            if chain_id not in bounds:
                bounds[chain_id] = cls._combine_bounds([
                    *(node_times[node_id] for node_id in chain_node_ids[chain_id]),
                    *(visit(child_id) for child_id in children[chain_id]),
                ])
            return bounds[chain_id]

        for chain_id in chain_node_ids:
            visit(chain_id)
        return bounds

    def _order_by_parent(self, observations: dict[str, _Observation]) -> list[_Observation]:
        child_keys: dict[str, list[str]] = {key: [] for key in observations}
        indegree = dict.fromkeys(observations, 0)
        ready: deque[str] = deque()
        for key, observation in observations.items():
            if observation.parent_key is None:
                ready.append(key)
                continue
            if observation.parent_key not in observations:
                raise ValueError(f'Langfuse trajectory mapping references missing parent {observation.parent_key!r}')
            child_keys[observation.parent_key].append(key)
            indegree[key] += 1
        ordered: list[_Observation] = []
        while ready:
            key = ready.popleft()
            ordered.append(observations[key])
            for child_key in child_keys[key]:
                indegree[child_key] -= 1
                if indegree[child_key] == 0:
                    ready.append(child_key)
        if len(ordered) != len(observations):
            cycle_keys = sorted(key for key, degree in indegree.items() if degree > 0)
            raise ValueError(f'Langfuse trajectory mapping contains a parent cycle: {cycle_keys}')
        return ordered

    @staticmethod
    def _nodes_in_tree_order(trajectory: Trajectory) -> tuple[TrajectoryNode, ...]:
        nodes: list[TrajectoryNode] = []
        seen: set[int] = set()
        ready = list(trajectory.roots)
        heapq.heapify(ready)
        while ready:
            node_id = heapq.heappop(ready)
            if node_id in seen:
                raise ValueError(f'Trajectory {trajectory.session_id!r} visits node {node_id} more than once')
            if node_id not in trajectory.nodes:
                raise ValueError(f'Trajectory {trajectory.session_id!r} references missing node {node_id}')
            seen.add(node_id)
            node = trajectory.nodes[node_id]
            nodes.append(node)
            for child_id in node.children:
                heapq.heappush(ready, child_id)
        if len(seen) != len(trajectory.nodes):
            missing = sorted(set(trajectory.nodes) - seen)
            raise ValueError(f'Trajectory {trajectory.session_id!r} has unreachable nodes: {missing}')
        return tuple(nodes)

    def _trace_metadata(self, trajectory: Trajectory) -> dict[str, Any]:
        env = {key: value for key, name in _TAIJI_METADATA_ENV.items() if (value := os.environ.get(name))}
        metadata = {**env, **copy.deepcopy(self._config.metadata)}
        metadata.setdefault('episode', 0)
        if metadata.get('instance_id') and not metadata.get('episode_id'):
            metadata['episode_id'] = f'{metadata["instance_id"]}-ep{metadata["episode"]}'
        metadata.setdefault('trajectory_id', trajectory.session_id)
        return metadata

    @classmethod
    def _timeline_bounds(cls, timeline: list[dict[str, Any]], *, owner: str) -> _TimeBounds | None:
        intervals: list[_TimeBounds] = []
        for event in timeline:
            start = cls._finite_number(event.get('start_time', event.get('start')))
            end = cls._finite_number(event.get('end_time', event.get('end')))
            if start is None or end is None:
                continue
            if end < start:
                raise ValueError(f'{owner} timeline ends before it starts: start={start}, end={end}')
            intervals.append(
                _TimeBounds(
                    start_ns=int(round(start * _NANOSECONDS_PER_SECOND)),
                    end_ns=int(round(end * _NANOSECONDS_PER_SECOND)),
                )
            )
        return cls._combine_bounds(intervals)

    @staticmethod
    def _combine_bounds(bounds: Iterable[_TimeBounds | None]) -> _TimeBounds | None:
        present = [item for item in bounds if item is not None]
        if not present:
            return None
        return _TimeBounds(
            start_ns=min(item.start_ns for item in present),
            end_ns=max(item.end_ns for item in present),
        )

    @staticmethod
    def _node_kind(node: TrajectoryNode) -> str:
        if node.generated_by == 'llm':
            return 'generation'
        if node.original_message.get('role') in {'tool', 'function'} or 'tool_call_id' in node.original_message:
            return 'tool'
        return 'span'

    @staticmethod
    def _node_name(node: TrajectoryNode) -> str:
        role = str(node.original_message.get('role') or node.generated_by)
        label = {'system': 'System Prompt', 'developer': 'Developer Prompt', 'user': 'User Prompt'}.get(role, role)
        if node.generated_by == 'llm':
            boundary = node.segment_boundary
            label = 'Generation'
            if boundary is not None:
                label = {
                    'summary_inference': 'Summarize Context',
                    'compaction_resume': 'Resume After Compaction',
                    'render_restart': 'Render Restart',
                    'context_restart_unknown': 'Context Restart',
                }.get(boundary.kind, label)
        elif role in {'tool', 'function'}:
            label = f'Tool Result: {node.original_message.get("name") or role}'
        return f'{label} [node {node.node_id}]'

    @staticmethod
    def _model(node: TrajectoryNode) -> tuple[str | None, str]:
        if node.generated_by != 'llm':
            return None, 'missing'
        if node.provider_facts is not None and node.provider_facts.model:
            return node.provider_facts.model, 'provider_facts'
        model = (node.request_kwargs or {}).get('model')
        return (model, 'request_kwargs') if isinstance(model, str) and model else (None, 'missing')

    @staticmethod
    def _input_path(trajectory: Trajectory, node: TrajectoryNode) -> list[TrajectoryNode]:
        path: list[TrajectoryNode] = []
        current = node.parent
        seen = {node.node_id}
        while current is not None:
            if current in seen or current not in trajectory.nodes:
                raise ValueError(f'Invalid input ancestry for node {node.node_id}')
            seen.add(current)
            item = trajectory.nodes[current]
            path.append(item)
            current = item.parent
        return list(reversed(path))

    def _tool_call(
        self,
        trajectory: Trajectory,
        node: TrajectoryNode,
        chain_id: str,
    ) -> tuple[int, dict[str, Any]] | None:
        tool_id = node.original_message.get('tool_call_id')
        if not tool_id:
            return None
        for item in reversed(self._input_path(trajectory, node)):
            if trajectory.node_chain(item.node_id) != chain_id:
                break
            for call in item.original_message.get('tool_calls') or []:
                if isinstance(call, dict) and call.get('id') == tool_id:
                    return item.node_id, copy.deepcopy(call)
        return None

    def _chain_io(
        self,
        trajectory: Trajectory,
        nodes: tuple[TrajectoryNode, ...],
        chain_id: str,
        node_chain: dict[int, str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        entries = []
        responses = []
        selected_llm_below: dict[int, bool] = {}
        rejected: dict[int, bool] = {}
        for node in nodes:
            rejected[node.node_id] = node.selection == 'rejected_sibling' or (
                rejected.get(node.parent, False) if node.parent is not None else False
            )
        for node in reversed(nodes):
            selected_llm_below[node.node_id] = any(
                not rejected[child]
                and node_chain[child] == chain_id
                and (trajectory.nodes[child].generated_by == 'llm' or selected_llm_below[child])
                for child in node.children
            )
        for node in nodes:
            if node_chain[node.node_id] != chain_id or node.generated_by != 'llm' or rejected[node.node_id]:
                continue
            path = self._input_path(trajectory, node)
            if not any(item.generated_by == 'llm' and node_chain[item.node_id] == chain_id for item in path):
                entries.append({
                    'node_id': node.node_id,
                    'messages': [
                        self._message(node.request_message_overrides.get(item.node_id, item.original_message))
                        for item in path
                    ],
                })
            if not selected_llm_below[node.node_id]:
                responses.append({'node_id': node.node_id, 'message': self._message(node.original_message)})
        # Several selected physical roots can survive compaction. Do not guess
        # which terminal response is the final answer from node IDs or names.
        return {'entry_requests': entries}, {'terminal_responses': responses}

    @staticmethod
    def _chain_key(chain_id: str) -> str:
        return f'chain:{chain_id}'

    @staticmethod
    def _node_key(node_id: int) -> str:
        return f'node:{node_id}'

    def _message(self, message: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(message)
        limit = self._config.max_tool_chars if message.get('role') == 'tool' else self._config.max_content_chars
        if isinstance(result.get('content'), str):
            result['content'] = self._trunc(result['content'], limit)
        return result

    @staticmethod
    def _model_parameters(request_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        kwargs = request_kwargs or {}
        return {
            key: kwargs[key] for key in ('temperature', 'max_tokens', 'top_p', 'top_k') if kwargs.get(key) is not None
        }

    @staticmethod
    def _usage(value: dict[str, Any] | None) -> dict[str, int]:
        value = value if isinstance(value, dict) else {}
        details = value.get('prompt_tokens_details') or {}
        cached = (
            details.get('gpu_cached_tokens', 0) or details.get('cached_tokens', 0) or 0
            if isinstance(details, dict)
            else 0
        )
        input_tokens = value.get('prompt_tokens', value.get('input_tokens', 0)) or 0
        output_tokens = value.get('completion_tokens', value.get('output_tokens', 0)) or 0
        return {
            'input': int(input_tokens),
            'output': int(output_tokens),
            'total': int(value.get('total_tokens', value.get('total', input_tokens + output_tokens)) or 0),
            'cached_tokens': int(cached),
        }

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _dedupe(items: list[Any]) -> list[str]:
        return list(dict.fromkeys(str(item) for item in items if item is not None and str(item)))

    @staticmethod
    def _trunc(value: Any, limit: int) -> str:
        if value is None:
            return ''
        text = str(value)
        return text if limit <= 0 or len(text) <= limit else text[:limit] + f'...[{len(text)} chars]'

    def _safe(self, value: Any) -> Any:
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            normalized = json.loads(serialized)
        except (TypeError, ValueError):
            return str(value)
        limit = self._config.max_content_chars
        if limit > 0 and len(serialized) > limit:
            return {'truncated_json': serialized[:limit], 'original_chars': len(serialized)}
        return normalized


@trajectory_observer_registry.register('langfuse')
class LangfuseTrajectoryObserver(TrajectoryObserver):
    """Export each finalized canonical Trajectory Tree exactly once."""

    _MAX_FINALIZED_IDS = 4096

    def __init__(
        self,
        config: LangfuseTrajectoryConfig | dict[str, Any] | None = None,
        *,
        client: Any = None,
        **config_values: Any,
    ) -> None:
        if config is not None and config_values:
            raise ValueError('Pass either config or keyword configuration values, not both')
        raw = config if config is not None else config_values
        self.config = raw if isinstance(raw, LangfuseTrajectoryConfig) else LangfuseTrajectoryConfig.from_dict(raw)
        self.mapper = LangfuseTrajectoryMapper(self.config)
        self._client = client
        self._finalized_ids: OrderedDict[str, None] = OrderedDict()
        self._disabled = False
        self._dirty = False
        self._lock = threading.RLock()

    @property
    def observes_updates(self) -> bool:
        """Return whether this observer consumes active trajectory updates."""
        return False

    def on_trajectory_reset(self, trajectory_id: str) -> None:
        with self._lock:
            self._finalized_ids.pop(trajectory_id, None)

    def on_update(self, update: TrajectoryUpdate) -> None:
        return

    def on_final(self, trajectory: Trajectory) -> None:
        with self._lock:
            trajectory_id = trajectory.session_id
            if trajectory_id in self._finalized_ids or not self._ensure_client():
                return
            presentation = self.mapper.map(trajectory)
            trace_id = self._new_trace_id(trajectory_id)
            self._export(presentation, trace_id)
            if presentation.score is not None:
                self._create_score(trace_id, presentation.score, presentation.score_comment)
            if self.config.flush_on_final and self._dirty:
                self._flush()
            self._finalized_ids[trajectory_id] = None
            while len(self._finalized_ids) > self._MAX_FINALIZED_IDS:
                self._finalized_ids.popitem(last=False)

    def close(self) -> None:
        with self._lock:
            self._finalized_ids.clear()
            if self.config.flush_on_final and self._client is not None and self._dirty:
                self._flush()

    def _export(self, presentation: _Presentation, trace_id: str) -> None:
        handles: dict[str, Any] = {}
        specs: dict[str, _Observation] = {}
        order: list[str] = []
        with self._trace_attributes(presentation):
            self._export_observations(presentation, trace_id, handles, specs, order)

    def _export_observations(
        self,
        presentation: _Presentation,
        trace_id: str,
        handles: dict[str, Any],
        specs: dict[str, _Observation],
        order: list[str],
    ) -> None:
        try:
            for spec in presentation.observations:
                parent = handles.get(spec.parent_key or '')
                if spec.parent_key is not None and parent is None:
                    raise RuntimeError(f'Langfuse parent {spec.parent_key!r} was not created before {spec.key!r}')
                if parent is None:
                    client = cast(Any, self._client)
                    handle = client.start_observation(
                        trace_context=self._trace_context(trace_id),
                        name=spec.name,
                        as_type=spec.kind,
                        **spec.payload(),
                    )
                else:
                    handle = parent.start_observation(name=spec.name, as_type=spec.kind, **spec.payload())
                handles[spec.key] = handle
                specs[spec.key] = spec
                order.append(spec.key)
                if spec.time_bounds is not None:
                    self._set_start_time(handle, spec.time_bounds.start_ns, key=spec.key)
                self._dirty = True
        finally:
            for key in reversed(order):
                bounds = specs[key].time_bounds
                try:
                    handles[key].end(end_time=None if bounds is None else bounds.end_ns)
                    self._dirty = True
                except Exception:
                    logger.warning('Failed to end Langfuse trajectory object %s', key, exc_info=True)

    def _create_score(self, trace_id: str, score: float, comment: str) -> None:
        try:
            self._client.create_score(
                trace_id=trace_id,
                name='trajectory_reward',
                value=score,
                comment=comment,
            )
            self._dirty = True
        except Exception:
            logger.warning('Failed to upload Gateway trajectory reward', exc_info=True)

    def _new_trace_id(self, trajectory_id: str) -> str:
        seed = f'{trajectory_id}::{uuid.uuid4().hex}'
        creator = getattr(self._client, 'create_trace_id', None)
        if callable(creator):
            return str(creator(seed=seed))
        return hashlib.blake2b(seed.encode(), digest_size=16).hexdigest()

    @staticmethod
    def _set_start_time(handle: Any, start_time_ns: int, *, key: str) -> None:
        otel_span = getattr(handle, '_otel_span', None)
        if otel_span is None or not hasattr(otel_span, '_start_time'):
            raise RuntimeError(f'Langfuse observation {key!r} cannot preserve its Trajectory start timestamp')
        otel_span._start_time = start_time_ns

    @staticmethod
    def _trace_context(trace_id: str) -> Any:
        try:
            langfuse_types = importlib.import_module('langfuse.types')
            return langfuse_types.TraceContext(trace_id=trace_id, parent_span_id='')
        except ImportError:
            return {'trace_id': trace_id, 'parent_span_id': ''}

    @staticmethod
    def _trace_attributes(presentation: _Presentation) -> Any:
        try:
            langfuse = importlib.import_module('langfuse')
            return langfuse.propagate_attributes(
                tags=list(presentation.tags),
                session_id=presentation.session_id or None,
                trace_name=presentation.trace_name or None,
            )
        except ImportError:
            return contextlib.nullcontext()

    def _ensure_client(self) -> bool:
        if self._disabled:
            return False
        if self._client is not None:
            return True
        public_key = self.config.public_key or os.environ.get('LANGFUSE_PUBLIC_KEY', '')
        secret_key = self.config.secret_key or os.environ.get('LANGFUSE_SECRET_KEY', '')
        host = (
            self.config.host
            or os.environ.get('LANGFUSE_HOST', '')
            or os.environ.get('LANGFUSE_BASE_URL', '')
            or 'https://cloud.langfuse.com'
        )
        if not public_key or not secret_key:
            logger.warning('Langfuse credentials not configured, trajectory observer disabled')
            self._disabled = True
            return False
        try:
            langfuse = importlib.import_module('langfuse')
            self._client = langfuse.Langfuse(public_key=public_key, secret_key=secret_key, host=host)
            return True
        except Exception:
            logger.warning('Failed to initialize Langfuse trajectory observer', exc_info=True)
            self._disabled = True
            return False

    def _flush(self) -> None:
        try:
            self._client.flush()
            self._dirty = False
        except Exception:
            logger.warning('Failed to flush Langfuse trajectory observer', exc_info=True)
