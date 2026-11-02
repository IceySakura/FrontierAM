"""Bridge white-box run spans into trajectory-level facts."""

import copy
import json
import logging

from collections.abc import Iterator
from typing import Any

from nexus.gateway.observers.hub import get_current_trajectory_observation_hub
from nexus.gateway.scaffold_policy import protocol_message_identity
from nexus.gateway.session_manager import session_manager
from nexus.gateway.trajectory import Trajectory, TrajectoryNode
from nexus.trace.events import SpanEndEvent, SpanStartEvent
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.types import Span
from nexus.utils.tool_analyzer import build_turn_stats_from_messages, summarize_tool_call_stats


logger = logging.getLogger(__name__)


def _iter_executed_nodes(trajectory: Trajectory) -> Iterator[tuple[str, TrajectoryNode]]:
    """Yield accepted tree nodes in assembly order, grouped by chain."""
    rejected: set[int] = set()
    pending = [node.node_id for node in trajectory.iter_nodes() if node.selection == 'rejected_sibling']
    while pending:
        node_id = pending.pop()
        if node_id not in rejected:
            rejected.add(node_id)
            pending.extend(trajectory.nodes[node_id].children)
    for node in trajectory.iter_nodes():
        if node.node_id not in rejected:
            yield trajectory.node_chain(node.node_id), node


class NexusRunTracker(SpanProcessor):
    _TIMELINE_TYPES = {'task_phase', 'agent_inference', 'agent_observation', 'agent_compaction'}

    def __init__(
        self,
        progress_state: dict[str, Any] | None = None,
        *,
        trajectory_id: str | None = None,
    ):
        self._progress_state = progress_state
        self._trajectory_id = trajectory_id
        self.task_name: str | None = None
        self.termination_reason: str | None = None
        self.observation_time = 0.0
        self.reward_time = 0.0
        self.sandbox_init_time = 0.0
        self.init_compose_time = 0.0
        self.init_prepare_directories_time = 0.0
        self.init_boot_time = 0.0
        self.handoff_artifact_collect_time = 0.0
        self.handoff_release_main_time = 0.0
        self.handoff_sandbox_init_time = 0.0
        self.handoff_env_setup_time = 0.0
        self.scoring_check_score_time = 0.0
        self.runtime_id: str | None = None
        self.runtime_ids: list[str] = []
        self.eval_runtime_id: str | None = None
        self.agent_timeline: list[dict[str, Any]] = []
        self.context_ready = False

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        del event
        if span.attributes.get('type') == 'task':
            self.task_name = span.attributes.get('name')

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        del event
        attributes = span.attributes
        span_type = attributes.get('type')
        if (
            span_type in self._TIMELINE_TYPES
            and span.end_time_unix_nano is not None
            and (span_type != 'agent_compaction' or attributes.get('compacted'))
        ):
            name = span.name
            timeline_type = span_type
            if span_type == 'agent_observation':
                name = attributes.get('tool_name', 'observation')
            elif span_type == 'agent_inference':
                name = 'gen'
            elif span_type == 'agent_compaction':
                name = 'compaction'
            elif span_type == 'task_phase':
                timeline_type = self.task_name
                name = attributes.get('phase', span.name)
            item = {
                'type': timeline_type,
                'name': name,
                'start_time': span.start_time_unix_nano / 1e9,
                'end_time': span.end_time_unix_nano / 1e9,
            }
            tool_args = attributes.get('hytuner.tool_args')
            if tool_args is not None:
                try:
                    item['tool_args'] = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                except json.JSONDecodeError:
                    item['tool_args'] = tool_args
            if 'hytuner.tool_success' in attributes:
                item['success'] = attributes['hytuner.tool_success']
            if 'hytuner.tool_error' in attributes:
                item['error'] = attributes['hytuner.tool_error']
            if 'hycode.span_transfer' in attributes:
                item['span_transfer'] = attributes['hycode.span_transfer']
            self.agent_timeline.append(item)

        reason = attributes.get('hytuner.termination_reason') or attributes.get('agent.termination_reason')
        if reason:
            self.termination_reason = reason
        observation_time = attributes.get('hytuner.observation_time') or attributes.get('agent.observation_time')
        if observation_time is not None:
            self.observation_time += float(observation_time)
        reward_time = attributes.get('hytuner.reward_time')
        if reward_time is not None:
            self.reward_time += float(reward_time)

        if span_type == 'task_phase' and span.end_time_unix_nano is not None:
            duration = (span.end_time_unix_nano - span.start_time_unix_nano) / 1e9
            if span.name == 'task.init.get_runtime':
                self.sandbox_init_time += duration
            elif span.name == 'task.init.compose':
                self.init_compose_time += duration
            elif span.name == 'task.init.prepare_directories':
                self.init_prepare_directories_time += duration
            elif span.name == 'task.init.boot':
                self.init_boot_time += duration
            elif span.name == 'task.handoff.artifact_collect_time':
                self.handoff_artifact_collect_time += duration
            elif span.name == 'task.handoff.release_main_time':
                self.handoff_release_main_time += duration
            elif span.name == 'task.handoff.sandbox_init_time':
                self.handoff_sandbox_init_time += duration
            elif span.name == 'task.handoff.env_setup_time':
                self.handoff_env_setup_time += duration
            elif span.name == 'task.scoring.check_score_time':
                self.scoring_check_score_time += duration

        self._record_runtime(attributes.get('hytuner.runtime_id'))
        eval_runtime_id = attributes.get('hytuner.eval_runtime_id')
        if eval_runtime_id:
            self.eval_runtime_id = eval_runtime_id
            if self._progress_state is not None:
                self._progress_state['eval_runtime_id'] = eval_runtime_id

    async def capture_context(self, context: Any) -> None:
        self.context_ready = True
        self._record_runtime(context.get_runtime_id('main'))
        eval_runtime_id = context.get_runtime_id('eval')
        if eval_runtime_id:
            self.eval_runtime_id = eval_runtime_id
            if self._progress_state is not None:
                self._progress_state['eval_runtime_id'] = eval_runtime_id

    async def capture_run_output(self, context: Any, output: Any) -> None:
        await self.capture_context(context)
        messages = getattr(getattr(output, 'agent_output', None), 'main_trajectory', None)
        self._import_terminal_observations(messages)

    def _import_terminal_observations(self, messages: Any) -> None:
        if self._trajectory_id is None or not isinstance(messages, list):
            return
        session = session_manager.get_session(self._trajectory_id)
        if session is None:
            return

        recorded = session.get_trajectory().to_messages()
        extends_recorded = len(messages) >= len(recorded) and all(
            isinstance(reported, dict) and protocol_message_identity(reported) == protocol_message_identity(canonical)
            for reported, canonical in zip(messages, recorded, strict=False)
        )
        if not extends_recorded:
            logger.warning(
                'Gateway run output trajectory does not extend the recorded transcript for %s; '
                'terminal observations were not imported',
                self._trajectory_id,
            )
            return
        observations = messages[len(recorded) :]
        if not observations:
            return
        if any(
            not isinstance(message, dict) or message.get('role') not in {'tool', 'user'} for message in observations
        ):
            logger.warning(
                'Gateway run output for %s has an unsupported terminal input suffix; it was not imported',
                self._trajectory_id,
            )
            return

        parent_messages = copy.deepcopy(recorded)
        for observation in observations:
            session.record_observation(message=observation, parent_messages=parent_messages)
            parent_messages.append(copy.deepcopy(observation))

    def finalize_trajectory(
        self,
        trajectory: Trajectory | None,
        trajectory_id: str,
        *,
        termination_reason: str,
        final_reward: float | None = None,
        extra: dict[str, Any] | None = None,
        allow_empty: bool = False,
    ) -> Trajectory | None:
        trajectory_extra = copy.deepcopy(extra or {})
        if self.runtime_id:
            trajectory_extra['runtime_id'] = self.runtime_id
        if self.runtime_ids:
            trajectory_extra['runtime_ids'] = list(self.runtime_ids)
        trajectory_extra['eval_runtime_id'] = self.eval_runtime_id or 'unknown'
        trajectory_extra['agent_timeline'] = copy.deepcopy(self.agent_timeline)
        metrics = {
            'abort_wait_time': 0.0,
            'llm_inference_time': 0.0,
            'true_llm_inference_time': 0.0,
            'observation_time': self.observation_time,
            'reward_time': self.reward_time,
            'sandbox_init_time': self.sandbox_init_time,
            'init_compose_time': self.init_compose_time,
            'init_prepare_directories_time': self.init_prepare_directories_time,
            'init_boot_time': self.init_boot_time,
            'handoff_artifact_collect_time': self.handoff_artifact_collect_time,
            'handoff_release_main_time': self.handoff_release_main_time,
            'handoff_sandbox_init_time': self.handoff_sandbox_init_time,
            'handoff_env_setup_time': self.handoff_env_setup_time,
            'scoring_check_score_time': self.scoring_check_score_time,
        }

        if trajectory is None and (self.context_ready or allow_empty):
            trajectory = Trajectory(
                session_id=trajectory_id,
                final_reward=final_reward,
                termination_reason=termination_reason,
                metrics=metrics,
                extra=trajectory_extra,
            )
        elif trajectory is None:
            return None

        metrics['abort_wait_time'] = sum(
            float(node.provider_facts.abort_wait_time or 0.0)
            for node in trajectory.nodes.values()
            if node.provider_facts is not None
        )
        metrics['llm_inference_time'] = sum(
            float(
                node.provider_facts.wall_inference_time
                if node.provider_facts.wall_inference_time is not None
                else node.provider_facts.inference_time or 0.0
            )
            for node in trajectory.nodes.values()
            if node.provider_facts is not None
        )
        metrics['true_llm_inference_time'] = sum(
            float(node.provider_facts.inference_time or 0.0)
            for node in trajectory.nodes.values()
            if node.provider_facts is not None
        )

        try:
            tool_call_stats = self._build_per_turn_tool_call_stats(trajectory)
            trajectory_extra['tool_stats'] = self._aggregate_tool_stats(trajectory)
        except Exception:
            logger.exception('Failed to build tool statistics for Gateway trajectory %s', trajectory_id)
            tool_call_stats = []
            trajectory_extra['tool_stats'] = {}
        if tool_call_stats:
            trajectory_extra['tool_call_stats'] = tool_call_stats
            trajectory_extra['tool_call_summary'] = summarize_tool_call_stats(tool_call_stats)

        trajectory.metrics.update(copy.deepcopy(metrics))
        trajectory.extra.update(copy.deepcopy(trajectory_extra))
        trajectory.termination_reason = termination_reason
        trajectory.final_reward = final_reward
        hub = get_current_trajectory_observation_hub()
        if hub is not None:
            hub.publish_final(trajectory)
        return trajectory

    def _build_per_turn_tool_call_stats(self, trajectory: Trajectory) -> list[dict[str, Any]] | None:
        turns: list[list[dict[str, Any]]] = []
        turn_by_chain: dict[str, list[dict[str, Any]]] = {}
        call_by_chain_and_id: dict[tuple[str, str], tuple[list[dict[str, Any]], tuple[str, int, int]]] = {}
        seen_result_occurrences: set[tuple[str, int, int]] = set()

        for chain, node in _iter_executed_nodes(trajectory):
            message = node.original_message
            if node.generated_by == 'llm' and message.get('role') == 'assistant':
                turn = [copy.deepcopy(message)]
                turns.append(turn)
                turn_by_chain[chain] = turn
                for index, call in enumerate(message.get('tool_calls') or []):
                    identifier = call.get('id')
                    if identifier is not None:
                        call_by_chain_and_id[(chain, identifier)] = (turn, (chain, node.node_id, index))
                continue

            turn = turn_by_chain.get(chain)
            if turn is None:
                continue
            if message.get('role') == 'tool':
                tool_call_id = message.get('tool_call_id')
                if not isinstance(tool_call_id, str):
                    continue
                matched = call_by_chain_and_id.get((chain, tool_call_id))
                if matched is None:
                    continue
                turn_messages, occurrence = matched
                if occurrence in seen_result_occurrences:
                    continue
                seen_result_occurrences.add(occurrence)
                turn_messages.append(copy.deepcopy(message))
            elif message.get('role') == 'user':
                turn.append(copy.deepcopy(message))

        stats = []
        for turn in turns:
            turn_stats = build_turn_stats_from_messages(turn, task_name=self.task_name)
            if turn_stats is None:
                return None
            stats.append(turn_stats)
        return stats

    def _aggregate_tool_stats(self, trajectory: Trajectory) -> dict[str, dict[str, int]]:
        # Keep the schema and failure classification aligned with HytunerProcessor.
        from nexus.trace.processors.hytuner import _classify_visit_failure, _is_tool_failure

        def _empty_bucket() -> dict[str, int]:
            return {'total': 0, 'success': 0, 'failure': 0, 'failure_readpage': 0, 'failure_summary_llm': 0}

        stats: dict[str, dict[str, int]] = {}
        call_by_chain_and_id: dict[tuple[str, str], tuple[str, tuple[str, int, int]]] = {}
        seen_result_occurrences: set[tuple[str, int, int]] = set()
        for chain, node in _iter_executed_nodes(trajectory):
            message = node.original_message
            if node.generated_by == 'llm' and message.get('role') == 'assistant':
                for index, call in enumerate(message.get('tool_calls') or []):
                    name = call.get('function', {}).get('name', 'unknown')
                    occurrence = (chain, node.node_id, index)
                    stats.setdefault(name, _empty_bucket())['total'] += 1
                    identifier = call.get('id')
                    if identifier is not None:
                        call_by_chain_and_id[(chain, identifier)] = (name, occurrence)
            elif message.get('role') == 'tool':
                tool_call_id = message.get('tool_call_id')
                if not isinstance(tool_call_id, str):
                    continue
                matched = call_by_chain_and_id.get((chain, tool_call_id))
                if matched is None:
                    continue
                name, occurrence = matched
                if occurrence in seen_result_occurrences:
                    continue
                seen_result_occurrences.add(occurrence)
                bucket = stats[name]
                content = str(message.get('content', ''))
                if _is_tool_failure(name, content):
                    bucket['failure'] += 1
                    if name == 'visit':
                        bucket[f'failure_{_classify_visit_failure(content)}'] += 1
                else:
                    bucket['success'] += 1
        return stats

    def _record_runtime(self, runtime_id: str | None) -> None:
        if not runtime_id:
            return
        self.runtime_id = runtime_id
        if runtime_id not in self.runtime_ids:
            self.runtime_ids.append(runtime_id)
        if self._progress_state is not None:
            self._progress_state['runtime_id'] = runtime_id
