"""Machine-wide slot leases shared by every checkout, worker and agent on this host.

A lease is an exclusive ``flock`` on the slot's lock file plus a JSON record. The kernel drops the lock when the last
process holding the descriptor exits, so crashed holders never leak slots; the descriptor is inherited by the
grading/development child so a slot stays leased while its measurement is still running. Memory is reserved against a
single machine budget so concurrent work cannot push the host into swap-free OOM.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time

from contextlib import contextmanager
from pathlib import Path

from . import machine


# dev_idle (protocol v6.3 timing-sensitive held-out grading): start only while no development slot is leased; gradings
# still run side by side (docs/platform/CONCURRENCY.md section 7).
ISOLATION_HOLDS = {'shared': (), 'grade_pool': ('grade',), 'quiet': ('grade', 'dev'), 'dev_idle': ()}


class Lease:
    def __init__(self, broker: Broker, slot: dict, memory_bytes: int, owner: str, fd: int) -> None:
        self.broker, self.slot, self.memory_bytes, self.owner, self.fd = broker, slot, memory_bytes, owner, fd
        self.held: list[tuple[str, int]] = []
        self.isolation = 'shared'

    @property
    def fds(self) -> tuple[int, ...]:
        return (self.fd, *(fd for _, fd in self.held))

    @property
    def id(self) -> str:
        return self.slot['id']

    @property
    def cpuset(self) -> str:
        return str(self.slot['cpu'])

    @property
    def host_cpu(self) -> int | None:
        return self.slot.get('host_cpu')

    @property
    def cgroup_parent(self) -> str:
        return machine.SLICES[self.slot['kind']]

    def record(self) -> dict:
        return {
            'slot': self.id,
            'kind': self.slot['kind'],
            'cpu': self.slot['cpu'],
            'host_cpu': self.host_cpu,
            'cgroup_parent': self.cgroup_parent,
            'memory_bytes': self.memory_bytes,
            'owner': self.owner,
            'isolation': self.isolation,
            'held_slots': [slot_id for slot_id, _ in self.held],
            'layout_sha256': self.broker.layout['layout_sha256'],
        }

    def release(self) -> None:
        if self.fd < 0:
            return
        for slot_id, fd in self.held:
            self.broker.record_path(slot_id).unlink(missing_ok=True)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        self.held = []
        self.broker.record_path(self.id).unlink(missing_ok=True)
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)
        self.fd = -1

    def __enter__(self) -> Lease:
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class Broker:
    def __init__(self, layout: dict, state: Path = machine.STATE, *, dev_drain_after: float = 9000.0) -> None:
        self.layout = layout
        self.dev_drain_after = dev_drain_after
        self.root = state / 'slots'
        self.root.mkdir(parents=True, exist_ok=True)
        self.slots = {slot['id']: slot for slot in layout['slots']}

    def lock_path(self, slot_id: str) -> Path:
        return self.root / f'{slot_id}.lock'

    def record_path(self, slot_id: str) -> Path:
        return self.root / f'{slot_id}.json'

    @contextmanager
    def _allocation(self):
        fd = os.open(self.root / 'broker.lock', os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _try_lock(self, slot_id: str) -> int | None:
        fd = os.open(self.lock_path(slot_id), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        return fd

    def leases(self) -> list[dict]:
        """Live leases; records whose holder died are removed."""
        live = []
        for slot_id in self.slots:
            path = self.record_path(slot_id)
            fd = self._try_lock(slot_id)
            if fd is not None:
                path.unlink(missing_ok=True)
                os.close(fd)
                continue
            try:
                live.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                live.append({'slot': slot_id, 'memory_bytes': 0, 'owner': 'unknown (record pending)'})
        return live

    def _flag(self, prefix: str, owner: str) -> tuple[int, Path]:
        """A request file held (flock) for as long as its owner needs it; a crashed owner drops it."""
        path = self.root / f'{prefix}-{os.getpid()}-{time.monotonic_ns()}.request'
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, json.dumps({'owner': owner, 'pid': os.getpid(), 'since_unix': time.time()}).encode())
        return fd, path

    def _live(self, prefix: str) -> int:
        """How many request files with this prefix are still held; stale ones are removed."""
        count = 0
        for path in self.root.glob(f'{prefix}-*.request'):
            try:
                fd = os.open(path, os.O_RDWR)
            except FileNotFoundError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                count += 1
            else:
                path.unlink(missing_ok=True)
            finally:
                os.close(fd)
        return count

    @contextmanager
    def iterative_attempt(self, owner: str):
        """Marks a running v6.3 iterative attempt: while any runs, non-priority grading leaves one grade slot free."""
        fd, path = self._flag('attempt', owner)
        try:
            yield
        finally:
            path.unlink(missing_ok=True)
            os.close(fd)

    def acquire(
        self,
        kind: str,
        memory_bytes: int,
        owner: str,
        *,
        slot_id: str | None = None,
        timeout: float | None = None,
        poll: float = 2.0,
        isolation: str = 'shared',
        priority: bool = False,
    ) -> Lease:
        """Lease one slot; isolation also holds every slot of the listed kinds idle for the lease's lifetime.

        priority (v6.3 test evaluations): while a priority grade lease waits, other grade leases wait; while an
        iterative attempt runs, other grade leases never take the last free grade slot.
        """
        if memory_bytes > self.layout['memory_budget_bytes']:
            raise ValueError('Requested memory exceeds the whole machine budget')
        candidates = [s for s in self.layout['slots'] if s['kind'] == kind and (slot_id is None or s['id'] == slot_id)]
        if not candidates:
            raise ValueError(f'No {kind} slot matches {slot_id or "the layout"}')
        holds = ISOLATION_HOLDS[isolation]
        deadline = None if timeout is None else time.monotonic() + timeout
        drain = self._drain_request(owner) if 'dev' in holds else None
        urgent = self._flag('priority', owner) if priority and kind == 'grade' else None
        try:
            while True:
                with self._allocation():
                    used = sum(int(row.get('memory_bytes', 0)) for row in self.leases())
                    # A waiting quiet grade blocks new gradings at once (they are minutes long), but new development
                    # only after dev_drain_after: draining at once would serialise a whole batch of 2-hour attempts.
                    blocked = isolation != 'quiet' and self._draining(kind)
                    if isolation == 'dev_idle' and not blocked:
                        dev = {s['id'] for s in self.layout['slots'] if s['kind'] == 'dev'}
                        blocked = any(row.get('slot') in dev for row in self.leases())
                    if kind == 'grade' and not priority and not blocked:
                        leased = {row['slot'] for row in self.leases()}
                        free = [s for s in self.layout['slots'] if s['kind'] == 'grade' and s['id'] not in leased]
                        blocked = self._live('priority') > 0 or (self._live('attempt') > 0 and len(free) <= 1)
                    if not blocked and used + memory_bytes <= self.layout['memory_budget_bytes']:
                        lease = self._lock_set(candidates, holds, memory_bytes, owner, isolation)
                        if lease is not None:
                            return lease
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f'No free {kind} slot ({isolation}) with {memory_bytes / 2**30:.1f} GiB in {timeout}s'
                    )
                time.sleep(poll)
        finally:
            for request in (drain, urgent):
                if request is not None:
                    request[1].unlink(missing_ok=True)
                    os.close(request[0])

    def _drain_request(self, owner: str) -> tuple[int, Path]:
        """While a quiet grade waits, new gradings wait; new development waits once dev_drain_after has passed."""
        path = self.root / f'drain-{os.getpid()}-{time.monotonic_ns()}.request'
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, json.dumps({'owner': owner, 'dev_after_unix': time.time() + self.dev_drain_after}).encode())
        return fd, path

    def _draining(self, kind: str) -> bool:
        for path in self.root.glob('drain-*.request'):
            try:
                fd = os.open(path, os.O_RDWR)
            except FileNotFoundError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if kind != 'dev':
                    return True
                try:
                    if time.time() >= json.loads(path.read_text())['dev_after_unix']:
                        return True
                except (OSError, ValueError, KeyError):
                    return True
            else:
                path.unlink(missing_ok=True)
            finally:
                os.close(fd)
        return False

    def _lock_set(self, candidates, holds, memory_bytes, owner, isolation) -> Lease | None:
        """All or nothing, under the allocation lock: a partial exclusive lease is never left behind."""
        busy = orphan_cpus(machine.all_docker_hosts(self.layout))
        for slot in candidates:
            if slot['cpu'] in busy:
                continue
            fd = self._try_lock(slot['id'])
            if fd is None:
                continue
            held: list[tuple[str, int]] = []
            ok = True
            for other in self.layout['slots']:
                if other['id'] == slot['id'] or other['kind'] not in holds:
                    continue
                other_fd = self._try_lock(other['id']) if other['cpu'] not in busy else None
                if other_fd is None:
                    ok = False
                    break
                held.append((other['id'], other_fd))
            if not ok:
                for _, other_fd in held:
                    os.close(other_fd)
                os.close(fd)
                # Any other candidate would need the same held slots, which are busy.
                return None
            lease = Lease(self, slot, memory_bytes, owner, fd)
            lease.held, lease.isolation = held, isolation
            for lock in lease.fds:
                os.set_inheritable(lock, True)
            now = time.time()
            record = {**lease.record(), 'pid': os.getpid(), 'acquired_unix': now}
            self.record_path(slot['id']).write_text(json.dumps(record, indent=2) + '\n')
            for other_id, _ in held:
                holder = {'slot': other_id, 'memory_bytes': 0, 'owner': f'held idle by {slot["id"]} ({owner})',
                          'pid': os.getpid(), 'acquired_unix': now}
                self.record_path(other_id).write_text(json.dumps(holder, indent=2) + '\n')
            return lease
        return None


def orphan_cpus(hosts) -> set[int]:
    """CPUs pinned by running containers inside the partition; a dead holder may have left one behind."""
    busy: set[int] = set()
    for host in hosts:
        prefix = ['docker', '--host', host]
        result = subprocess.run([*prefix, 'ps', '-q', '--no-trunc'], capture_output=True, text=True, check=False)
        ids = result.stdout.split()
        if not ids:
            continue
        result = subprocess.run(
            [*prefix, 'inspect', '--format', '{{.HostConfig.CgroupParent}} {{.HostConfig.CpusetCpus}}', *ids],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            parent, _, cpus = line.partition(' ')
            if parent in machine.SLICES.values() and cpus:
                busy.update(machine.parse_cpus(cpus))
    return busy
