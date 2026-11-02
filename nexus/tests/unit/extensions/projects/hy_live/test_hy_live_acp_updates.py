from __future__ import annotations

from typing import Any

from nexus.extensions.projects.hy_live.coordination.acp_updates import AcpUpdateRouter


def _recording_router(received: list[tuple[str, str, dict[str, Any]]]) -> AcpUpdateRouter:
    return AcpUpdateRouter(lambda owner, task, update: received.append((owner, task, update)))


def _frame(channel_id: str, update_id: str, turn_id: str = 'turn-1') -> dict[str, Any]:
    return {'channel_id': channel_id, 'turn_id': turn_id, 'seq': 1, 'update': {'id': update_id}}


def test_frames_reach_the_task_bound_to_their_channel() -> None:
    received: list[tuple[str, str, dict[str, Any]]] = []
    router = _recording_router(received)
    router.bind('chan-a', turn_id='turn-1', owner_id='owner-1', task_id='task_1')
    router.bind('chan-b', turn_id='turn-1', owner_id='owner-1', task_id='task_2')

    router.dispatch(_frame('chan-a', 'c1'))
    router.dispatch(_frame('chan-b', 'c2'))
    router.dispatch(_frame('chan-a', 'c3'))

    assert received == [
        ('owner-1', 'task_1', {'id': 'c1'}),
        ('owner-1', 'task_2', {'id': 'c2'}),
        ('owner-1', 'task_1', {'id': 'c3'}),
    ]


def test_frames_after_unbind_or_for_an_unknown_channel_are_dropped() -> None:
    received: list[tuple[str, str, dict[str, Any]]] = []
    router = _recording_router(received)
    router.bind('chan-a', turn_id='turn-1', owner_id='owner-1', task_id='task_1')
    router.unbind('chan-a')

    router.dispatch(_frame('chan-a', 'c1'))
    router.dispatch(_frame('chan-x', 'c2'))
    router.dispatch('not a frame')

    assert received == []


def test_a_new_binding_routes_the_channel_to_its_task() -> None:
    received: list[tuple[str, str, dict[str, Any]]] = []
    router = _recording_router(received)
    router.bind('chan-a', turn_id='turn-1', owner_id='owner-1', task_id='task_1')
    router.dispatch(_frame('chan-a', 'c1'))
    router.bind('chan-a', turn_id='turn-2', owner_id='owner-1', task_id='task_2')

    router.dispatch(_frame('chan-a', 'c2', turn_id='turn-2'))

    assert [(task, update['id']) for _, task, update in received] == [('task_1', 'c1'), ('task_2', 'c2')]
    assert router.unbind('chan-a') == 1, 'the new binding counts only its own frames'


def test_frames_of_another_turn_on_the_channel_are_dropped() -> None:
    # The CLI behind a Host budget timeout still settles its prompt while the next task is bound.
    received: list[tuple[str, str, dict[str, Any]]] = []
    router = _recording_router(received)
    router.bind('chan-a', turn_id='turn-2', owner_id='owner-1', task_id='task_2')

    router.dispatch(_frame('chan-a', 'c1', turn_id='turn-1'))
    router.dispatch({'channel_id': 'chan-a', 'seq': 1, 'update': {'id': 'c2'}})
    router.dispatch(_frame('chan-a', 'c3', turn_id='turn-2'))

    assert [update['id'] for _, _, update in received] == ['c3']
    assert router.unbind('chan-a') == 1, 'stale frames must not count toward this turn'


def test_retarget_moves_later_frames_and_keeps_the_turn_and_its_count() -> None:
    received: list[tuple[str, str, dict[str, Any]]] = []
    router = _recording_router(received)
    router.bind('chan-a', turn_id='turn-1', owner_id='owner-1', task_id='task_old')
    router.dispatch(_frame('chan-a', 'c1'))

    assert router.retarget('chan-a', task_id='task_new') is True
    router.dispatch(_frame('chan-a', 'c2'))
    router.dispatch(_frame('chan-a', 'c3', turn_id='turn-0'))

    assert [(task, update['id']) for _, task, update in received] == [('task_old', 'c1'), ('task_new', 'c2')]
    assert router.unbind('chan-a') == 2, 'frames before and after the retarget count toward the same turn'


def test_retarget_of_an_unbound_channel_reports_false() -> None:
    router = AcpUpdateRouter(lambda owner, task, update: None)

    assert router.retarget('chan-a', task_id='task_new') is False


def test_listen_subscribes_once_per_runtime(fake_runtime: Any) -> None:
    router = AcpUpdateRouter(lambda owner, task, update: None)
    router.listen(fake_runtime)
    fake_runtime.handlers.clear()

    router.listen(fake_runtime)

    assert fake_runtime.handlers == {}
