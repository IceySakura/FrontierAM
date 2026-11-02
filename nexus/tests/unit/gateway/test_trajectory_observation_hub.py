import asyncio

from typing import cast

import pytest

from nexus.gateway.observers import (
    TrajectoryObservationHub,
    TrajectoryObserver,
    TrajectoryUpdate,
    get_current_trajectory_observation_hub,
    trajectory_observer_context,
)
from nexus.gateway.trajectory import Trajectory


class RecordingObserver(TrajectoryObserver):
    def __init__(self):
        self.updates = []
        self.finals = []
        self.resets = []
        self.close_count = 0

    def on_update(self, update):
        self.updates.append(update)

    def on_final(self, trajectory):
        self.finals.append(trajectory)

    def on_trajectory_reset(self, trajectory_id):
        self.resets.append(trajectory_id)

    def close(self):
        self.close_count += 1


class FinalOnlyRecordingObserver(RecordingObserver):
    @property
    def observes_updates(self) -> bool:
        return False


class StructuralRecordingObserver:
    def __init__(self):
        self.updates = []

    def on_update(self, update):
        self.updates.append(update)

    def on_final(self, trajectory):
        pass

    def on_trajectory_reset(self, trajectory_id):
        pass

    def close(self):
        pass


def _reserve_update(hub, trajectory_id):
    revision = hub.reserve_update(trajectory_id)
    assert revision is not None
    return TrajectoryUpdate(Trajectory(session_id=trajectory_id), revision)


def test_hub_tracks_revision_and_finalization_per_trajectory():
    observer = RecordingObserver()
    hub = TrajectoryObservationHub([observer])

    assert hub.publish_update(_reserve_update(hub, 'run-a'))
    assert hub.publish_update(_reserve_update(hub, 'run-b'))
    assert hub.publish_update(_reserve_update(hub, 'run-a'))
    assert hub.publish_final(Trajectory(session_id='run-a', state='completed'))
    assert not hub.publish_final(Trajectory(session_id='run-a', state='completed'))
    assert hub.reserve_update('run-a') is None

    assert [(update.trajectory_id, update.revision) for update in observer.updates] == [
        ('run-a', 1),
        ('run-b', 1),
        ('run-a', 2),
    ]
    assert [trajectory.session_id for trajectory in observer.finals] == ['run-a']


def test_reserved_revisions_follow_producer_order_even_when_publication_reorders():
    observer = RecordingObserver()
    hub = TrajectoryObservationHub([observer])

    first = _reserve_update(hub, 'run-1')
    second = _reserve_update(hub, 'run-1')
    hub.publish_update(second)
    hub.publish_update(first)

    assert (first.revision, second.revision) == (1, 2)
    assert [update.revision for update in observer.updates] == [2, 1]


def test_hub_begin_trajectory_allows_explicit_id_reuse():
    observer = RecordingObserver()
    hub = TrajectoryObservationHub([observer])

    hub.publish_final(Trajectory(session_id='run-1', state='completed'))
    hub.begin_trajectory('run-1')
    update = _reserve_update(hub, 'run-1')
    assert hub.publish_update(update)

    assert observer.resets == ['run-1']
    assert update.revision == 1


def test_context_binds_run_hub_and_closes_it_on_exit():
    observer = RecordingObserver()

    with trajectory_observer_context([observer]) as hub:
        assert get_current_trajectory_observation_hub() is hub
        update = _reserve_update(hub, 'context-run')
        hub.publish_update(update)

    assert get_current_trajectory_observation_hub() is None
    assert hub.reserve_update('outside-run') is None
    assert [update.trajectory_id for update in observer.updates] == ['context-run']
    assert observer.close_count == 1


def test_nested_context_uses_independent_run_hubs_and_restores_parent():
    parent_observer = RecordingObserver()
    child_observer = RecordingObserver()

    with trajectory_observer_context([parent_observer], close_on_exit=False) as parent_hub:
        parent_update = _reserve_update(parent_hub, 'run-1')
        with trajectory_observer_context([child_observer]) as child_hub:
            assert child_hub is not parent_hub
            assert get_current_trajectory_observation_hub() is child_hub
            child_update = _reserve_update(child_hub, 'run-1')
            child_hub.publish_update(child_update)
        assert get_current_trajectory_observation_hub() is parent_hub
        parent_hub.publish_update(parent_update)

    assert [update.revision for update in parent_observer.updates] == [1]
    assert [update.revision for update in child_observer.updates] == [1]
    assert parent_observer.close_count == 0
    assert child_observer.close_count == 1


def test_hub_deduplicates_observers_by_identity_and_closes_once():
    observer = RecordingObserver()

    with trajectory_observer_context([observer, observer]) as hub:
        hub.publish_update(_reserve_update(hub, 'run-1'))

    assert len(observer.updates) == 1
    assert observer.close_count == 1


def test_hub_keeps_updates_enabled_for_structural_observer_without_capability_property():
    observer = StructuralRecordingObserver()
    hub = TrajectoryObservationHub([cast(TrajectoryObserver, observer)])

    update = _reserve_update(hub, 'run-1')
    assert hub.publish_update(update)

    assert [item.revision for item in observer.updates] == [1]


def test_final_only_observer_skips_updates_and_still_receives_final() -> None:
    observer = FinalOnlyRecordingObserver()
    hub = TrajectoryObservationHub([observer])

    hub.begin_trajectory('run-1')
    assert hub.reserve_update('run-1') is None
    assert hub.publish_final(Trajectory(session_id='run-1', state='completed'))

    assert observer.resets == ['run-1']
    assert observer.updates == []
    assert [trajectory.session_id for trajectory in observer.finals] == ['run-1']


def test_hub_sends_updates_only_to_observers_that_request_them() -> None:
    update_observer = RecordingObserver()
    final_only_observer = FinalOnlyRecordingObserver()
    hub = TrajectoryObservationHub([update_observer, final_only_observer])

    update = _reserve_update(hub, 'run-1')
    assert hub.publish_update(update)
    assert hub.publish_final(Trajectory(session_id='run-1', state='completed'))

    assert [item.revision for item in update_observer.updates] == [1]
    assert final_only_observer.updates == []
    assert len(update_observer.finals) == 1
    assert len(final_only_observer.finals) == 1


@pytest.mark.asyncio
async def test_concurrent_run_contexts_are_isolated():
    first_observer = RecordingObserver()
    second_observer = RecordingObserver()
    ready = asyncio.Barrier(2)

    async def publish(observer, trajectory_id):
        with trajectory_observer_context([observer], close_on_exit=False) as hub:
            await ready.wait()
            assert get_current_trajectory_observation_hub() is hub
            hub.publish_update(_reserve_update(hub, trajectory_id))

    await asyncio.gather(
        publish(first_observer, 'run-a'),
        publish(second_observer, 'run-b'),
    )

    assert [update.trajectory_id for update in first_observer.updates] == ['run-a']
    assert [update.trajectory_id for update in second_observer.updates] == ['run-b']
