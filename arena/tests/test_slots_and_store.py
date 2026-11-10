import json
import multiprocessing
import os
import time

import pytest

from arena import slots
from arena.store import Store


def layout(memory_gib=16):
    return {
        'layout_sha256': 'fixture',
        'memory_budget_bytes': memory_gib * 2**30,
        'docker_hosts': {'dev': 'unix:///dev.sock', 'grade': []},
        'slots': [
            {'id': 'grade-24', 'kind': 'grade', 'cpu': 24, 'host_cpu': 25},
            {'id': 'grade-26', 'kind': 'grade', 'cpu': 26, 'host_cpu': 27},
            {'id': 'dev-04', 'kind': 'dev', 'cpu': 4, 'host_cpu': 5},
        ],
    }


@pytest.fixture(autouse=True)
def no_docker(monkeypatch):
    monkeypatch.setattr(slots, 'orphan_cpus', lambda hosts: set())


def test_leases_are_exclusive_and_released(tmp_path):
    broker = slots.Broker(layout(), tmp_path)
    first = broker.acquire('grade', 2**30, 'a')
    second = broker.acquire('grade', 2**30, 'b')
    assert {first.id, second.id} == {'grade-24', 'grade-26'}
    with pytest.raises(TimeoutError):
        broker.acquire('grade', 2**30, 'c', timeout=0)
    first.release()
    assert broker.acquire('grade', 2**30, 'd', timeout=0).id == 'grade-24'


def test_memory_budget_limits_concurrency(tmp_path):
    broker = slots.Broker(layout(memory_gib=10), tmp_path)
    broker.acquire('grade', 8 * 2**30, 'big')
    with pytest.raises(TimeoutError):
        broker.acquire('grade', 4 * 2**30, 'too much', timeout=0)
    assert broker.acquire('dev', 2 * 2**30, 'fits', timeout=0).id == 'dev-04'


def test_grade_pool_isolation_holds_every_other_grade_slot(tmp_path):
    broker = slots.Broker(layout(), tmp_path)
    exclusive = broker.acquire('grade', 2**30, 'timed', isolation='grade_pool')
    assert sorted(row['slot'] for row in broker.leases()) == ['grade-24', 'grade-26']
    with pytest.raises(TimeoutError):
        broker.acquire('grade', 2**30, 'second grader', timeout=0)
    assert broker.acquire('dev', 2**30, 'development still allowed', timeout=0).id == 'dev-04'
    exclusive.release()
    assert broker.acquire('grade', 2**30, 'after', timeout=0)


def test_exclusive_lease_is_all_or_nothing(tmp_path):
    broker = slots.Broker(layout(), tmp_path)
    development = broker.acquire('dev', 2**30, 'running agent')
    with pytest.raises(TimeoutError):
        broker.acquire('grade', 2**30, 'quiet grade', isolation='quiet', timeout=0)
    assert [row['slot'] for row in broker.leases()] == ['dev-04']
    development.release()
    quiet = broker.acquire('grade', 2**30, 'quiet grade', isolation='quiet', timeout=0)
    assert len(quiet.fds) == 3 and len(broker.leases()) == 3


def _wait_quiet(state, started, dev_drain_after):
    broker = slots.Broker(layout(), state, dev_drain_after=dev_drain_after)
    started.set()
    broker.acquire('grade', 2**30, 'quiet grade', isolation='quiet', timeout=60, poll=0.05)
    time.sleep(60)


def _start_waiter(tmp_path, dev_drain_after):
    started = multiprocessing.Event()
    waiter = multiprocessing.Process(target=_wait_quiet, args=(tmp_path, started, dev_drain_after))
    waiter.start()
    assert started.wait(10)
    time.sleep(0.5)
    return waiter


def test_waiting_quiet_grade_blocks_new_grading_but_not_development_at_first(tmp_path):
    broker = slots.Broker(layout(), tmp_path)
    running = broker.acquire('dev', 2**30, 'running agent')
    waiter = _start_waiter(tmp_path, dev_drain_after=3600)
    with pytest.raises(TimeoutError):
        broker.acquire('grade', 2**30, 'another grader', isolation='grade_pool', timeout=1, poll=0.05)
    running.release()
    # The quiet grade takes the freed dev slot; with a second dev slot a new agent would start meanwhile.
    waiter.kill()
    waiter.join()
    assert broker.acquire('dev', 2**30, 'new agent', timeout=1, poll=0.05).id == 'dev-04'


def test_long_waiting_quiet_grade_finally_drains_development(tmp_path):
    two_dev = layout()
    two_dev['slots'].append({'id': 'dev-06', 'kind': 'dev', 'cpu': 6, 'host_cpu': 7})
    broker = slots.Broker(two_dev, tmp_path)
    running = broker.acquire('dev', 2**30, 'running agent')
    started = multiprocessing.Event()
    waiter = multiprocessing.Process(target=_wait_quiet_two, args=(tmp_path, started))
    waiter.start()
    assert started.wait(10)
    time.sleep(0.5)
    with pytest.raises(TimeoutError):
        broker.acquire('dev', 2**30, 'new agent', timeout=1, poll=0.05)
    running.release()
    waiter.kill()
    waiter.join()


def _wait_quiet_two(state, started):
    two_dev = layout()
    two_dev['slots'].append({'id': 'dev-06', 'kind': 'dev', 'cpu': 6, 'host_cpu': 7})
    broker = slots.Broker(two_dev, state, dev_drain_after=0)
    started.set()
    broker.acquire('grade', 2**30, 'quiet grade', isolation='quiet', timeout=60, poll=0.05)
    time.sleep(60)


def test_waiting_quiet_grade_does_not_block_development_before_the_delay(tmp_path):
    two_dev = layout()
    two_dev['slots'].append({'id': 'dev-06', 'kind': 'dev', 'cpu': 6, 'host_cpu': 7})
    broker = slots.Broker(two_dev, tmp_path)
    running = broker.acquire('dev', 2**30, 'running agent')
    started = multiprocessing.Event()
    waiter = multiprocessing.Process(target=_wait_quiet_delayed_two, args=(tmp_path, started))
    waiter.start()
    assert started.wait(10)
    time.sleep(0.5)
    assert broker.acquire('dev', 2**30, 'new agent', timeout=1, poll=0.05).id == 'dev-06'
    running.release()
    waiter.kill()
    waiter.join()


def _wait_quiet_delayed_two(state, started):
    two_dev = layout()
    two_dev['slots'].append({'id': 'dev-06', 'kind': 'dev', 'cpu': 6, 'host_cpu': 7})
    broker = slots.Broker(two_dev, state, dev_drain_after=3600)
    started.set()
    broker.acquire('grade', 2**30, 'quiet grade', isolation='quiet', timeout=60, poll=0.05)
    time.sleep(60)


def _hold(state, ready):
    broker = slots.Broker(layout(), state)
    broker.acquire('dev', 2**30, 'child')
    ready.set()
    time.sleep(60)


def test_dead_holder_frees_its_slot(tmp_path):
    ready = multiprocessing.Event()
    child = multiprocessing.Process(target=_hold, args=(tmp_path, ready))
    child.start()
    assert ready.wait(10)
    broker = slots.Broker(layout(), tmp_path)
    assert [row['slot'] for row in broker.leases()] == ['dev-04']
    with pytest.raises(TimeoutError):
        broker.acquire('dev', 2**30, 'parent', timeout=0)
    child.kill()
    child.join()
    assert broker.leases() == []
    assert broker.acquire('dev', 2**30, 'parent', timeout=0).id == 'dev-04'


def _claim(root, results):
    job = Store(root).claim()
    results.put(job and job['id'])


def test_exactly_one_process_claims_a_job(tmp_path):
    store = Store(tmp_path)
    submission, directory = store.new_submission('exponential', {'origin': 'test'})
    (directory / 'source.tar').write_bytes(b'')
    job_id = store.enqueue(submission)
    results = multiprocessing.Queue()
    workers = [multiprocessing.Process(target=_claim, args=(tmp_path, results)) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    claimed = [results.get() for _ in workers]
    assert claimed.count(job_id) == 1 and claimed.count(None) == 7


def test_recover_requeues_only_abandoned_jobs(tmp_path):
    store = Store(tmp_path)
    submission, directory = store.new_submission('exponential', {'origin': 'test'})
    (directory / 'source.tar').write_bytes(b'')
    alive = store.claim() if store.enqueue(submission) else None
    assert store.recover() == []
    os.close(alive['lock_fd'])
    (store.result_dir(alive['id']) / 'partial.log').write_text('evidence')
    assert store.recover() == [alive['id']]
    assert json.loads((tmp_path / 'jobs/pending' / f'{alive["id"]}.json').read_text())['requeued_utc']
    archived = list((tmp_path / 'results').glob(f'{alive["id"]}.abandoned-*/partial.log'))
    assert len(archived) == 1


def test_reserved_grade_slot_serves_only_named_leases(tmp_path):
    broker = slots.Broker(layout(), tmp_path)
    with broker.reserve('attempt') as reserved:
        assert broker.reserved() == {reserved}
        other = broker.acquire('grade', 2**30, 'formal', timeout=0)
        assert other.id != reserved
        with pytest.raises(TimeoutError):
            broker.acquire('grade', 2**30, 'formal 2', timeout=0)
        pinned = broker.acquire('grade', 2**30, 'test', slot_id=reserved, timeout=0)
        assert pinned.id == reserved
        pinned.release()
        other.release()
        with pytest.raises(TimeoutError):
            broker.acquire('dev', 2**30, 'quiet grade', timeout=0, isolation='quiet')
    assert broker.reserved() == set()


def test_reservations_are_exclusive(tmp_path):
    broker = slots.Broker(layout(), tmp_path)
    with broker.reserve('a') as first, broker.reserve('b') as second:
        assert first != second
        with pytest.raises(TimeoutError):
            with broker.reserve('c', timeout=0, poll=0):
                pass
