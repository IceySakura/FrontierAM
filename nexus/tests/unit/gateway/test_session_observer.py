from types import SimpleNamespace

import pytest

from _fakes import memory_store

from nexus.gateway.nexus_run_tracker import NexusRunTracker
from nexus.gateway.observers import TrajectoryObserver, trajectory_observer_context
from nexus.gateway.session import GatewaySession
from nexus.gateway.trajectory import Trajectory


class StubProcessor:
    is_token_level = False

    async def create_completion(self, **_kwargs):
        return SimpleNamespace(
            message={'role': 'assistant', 'content': 'done'},
            prompt_ids=None,
            raw_prompt_ids=None,
            token_ids=None,
            log_probs=None,
            routed_experts=None,
            stop_reason='stop',
            usage={'prompt_tokens': 2, 'completion_tokens': 1},
            id='response-1',
            created=None,
            model='fake-model',
            object=None,
            system_fingerprint=None,
            service_tier=None,
            choice_logprobs=None,
            choice_provider_specific_fields=None,
            extra_fields={},
            completion_text='done',
            prompt_render=None,
        )


class RecordingObserver(TrajectoryObserver):
    def __init__(self):
        self.updates = []
        self.finals = []
        self.closed = False

    def on_update(self, update):
        self.updates.append(update)

    def on_final(self, trajectory):
        self.finals.append(trajectory)

    def close(self):
        self.closed = True


class BrokenObserver(RecordingObserver):
    def on_update(self, update):
        raise RuntimeError('observer unavailable')


class FinalOnlyRecordingObserver(RecordingObserver):
    @property
    def observes_updates(self) -> bool:
        return False


async def test_session_multicasts_isolated_snapshots_at_stable_boundaries():
    first = RecordingObserver()
    second = RecordingObserver()

    with trajectory_observer_context([first, second], close_on_exit=False):
        session = GatewaySession('run-1', StubProcessor(), memory_store('run-1'), scaffold_policy='generic')
        completion = await session.create_completion([{'role': 'user', 'content': 'question'}])
        session.record_observation(
            message={'role': 'tool', 'tool_call_id': 'call-1', 'content': 'result'},
            parent_messages=[{'role': 'user', 'content': 'question'}, completion.message],
        )
        session.finalize_segment()

    assert [update.revision for update in first.updates] == [1, 2, 3]
    assert [update.revision for update in second.updates] == [1, 2, 3]
    assert all(update.snapshot.state == 'active' for update in first.updates)
    assert first.updates[0].snapshot is not first.updates[1].snapshot
    first.updates[0].snapshot.extra['mutated'] = True
    assert 'mutated' not in session.get_trajectory().extra


async def test_observer_failure_does_not_skip_siblings_or_change_gateway_result(caplog):
    broken = BrokenObserver()
    healthy = RecordingObserver()

    with trajectory_observer_context([broken, healthy], close_on_exit=False):
        session = GatewaySession('run-1', StubProcessor(), memory_store('run-1'), scaffold_policy='generic')
        result = await session.create_completion([{'role': 'user', 'content': 'question'}])

    assert result.message['content'] == 'done'
    assert session.get_trajectory().state == 'active'
    assert len(healthy.updates) == 1
    assert 'Trajectory observer BrokenObserver failed' in caplog.text


async def test_session_does_not_materialize_observation_snapshot_without_observers(monkeypatch):
    session = GatewaySession('run-1', StubProcessor(), memory_store('run-1'), scaffold_policy='generic')

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError('get_trajectory should not be called on the unobserved hot path')

    monkeypatch.setattr(session, 'get_trajectory', fail_if_called)
    result = await session.create_completion([{'role': 'user', 'content': 'question'}])

    assert result.message['content'] == 'done'


async def test_session_does_not_materialize_observation_snapshot_for_final_only_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = FinalOnlyRecordingObserver()

    with trajectory_observer_context([observer], close_on_exit=False):
        session = GatewaySession('run-1', StubProcessor(), memory_store('run-1'), scaffold_policy='generic')

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError('get_trajectory should not be called for a final-only observer')

        monkeypatch.setattr(session, 'get_trajectory', fail_if_called)
        result = await session.create_completion([{'role': 'user', 'content': 'question'}])

    assert result.message['content'] == 'done'
    assert observer.updates == []


def test_shared_finalizer_publishes_enriched_canonical_tree():
    observer = RecordingObserver()
    tracker = NexusRunTracker(trajectory_id='run-1')
    raw = Trajectory(session_id='run-1', state='completed')

    with trajectory_observer_context([observer], close_on_exit=False):
        finalized = tracker.finalize_trajectory(
            raw,
            'run-1',
            termination_reason='ENV_DONE',
            final_reward=0.75,
            extra={'benchmark': 'swe'},
        )

    assert finalized is not None
    assert len(observer.finals) == 1
    published = observer.finals[0]
    assert published is not finalized
    assert published.state == 'completed'
    assert published.final_reward == 0.75
    assert published.termination_reason == 'ENV_DONE'
    assert published.extra['benchmark'] == 'swe'
