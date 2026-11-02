from __future__ import annotations

from datetime import timedelta
from typing import Any, cast

from nexus.distributed.control_plane.leases import ShardLeaseCoordinator
from nexus.distributed.utils.clock import utcnow


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self) -> None:
        return None


class _Repository:
    def __init__(self, active_shards: set[int]) -> None:
        self.active_shards = active_shards
        self.released: list[int] = []
        self.discovery_locks = 0

    async def validate_monotonic_shard_count(self, *, shard_group: str) -> None:
        return None

    async def list_owned(self, *, owner_id: str, shard_group: str, now) -> list[int]:
        return sorted(self.active_shards)

    async def lock_discovery(self, *, shard_group: str) -> None:
        self.discovery_locks += 1

    async def renew(self, *, owner_id: str, shard_group: str, shard_ids, lease_ttl, now) -> list[int]:
        self.active_shards.update(shard_ids)
        return list(shard_ids)

    async def release(self, *, owner_id: str, shard_group: str, shard_ids, now) -> int:
        self.released.extend(shard_ids)
        self.active_shards.difference_update(shard_ids)
        return len(shard_ids)


async def test_refresh_releases_database_leases_above_ownership_limit_after_local_invalidation() -> None:
    repository = _Repository({0, 1, 2, 3})
    coordinator = ShardLeaseCoordinator(
        session_factory=cast(Any, lambda: _Session()),
        shard_count=8,
        owner_id='reconciler-a',
        shard_group='execution-reconciler',
        lease_ttl=timedelta(seconds=20),
        renew_interval=5,
        ownership_limit=2,
        discover_interval=20,
        discover_jitter=0,
        repository_factory=lambda session: repository,
        process_session_id='process-a',
    )
    coordinator._owned_shards = {0, 1}
    coordinator.invalidate()

    owned = await coordinator.refresh(utcnow())

    assert len(owned) == 2
    assert set(repository.active_shards) == set(owned)
    assert set(repository.released) == {0, 1, 2, 3} - set(owned)


async def test_release_recovers_database_leases_after_local_invalidation() -> None:
    repository = _Repository({0, 1})
    coordinator = ShardLeaseCoordinator(
        session_factory=cast(Any, lambda: _Session()),
        shard_count=8,
        owner_id='scheduler-a',
        shard_group='execution-scheduler',
        lease_ttl=timedelta(seconds=20),
        renew_interval=5,
        ownership_limit=2,
        discover_interval=20,
        discover_jitter=0,
        repository_factory=lambda session: repository,
        process_session_id='process-a',
    )
    coordinator._owned_shards = {0, 1}
    coordinator.invalidate()

    await coordinator.release(utcnow())

    assert repository.active_shards == set()
    assert repository.released == [0, 1]


async def test_refresh_serializes_discovery_but_not_steady_renewal() -> None:
    now = utcnow()
    repository = _Repository(set())
    coordinator = ShardLeaseCoordinator(
        session_factory=cast(Any, lambda: _Session()),
        shard_count=8,
        owner_id='scheduler-a',
        shard_group='execution-scheduler',
        lease_ttl=timedelta(seconds=20),
        renew_interval=5,
        ownership_limit=2,
        discover_interval=10,
        discover_jitter=0,
        repository_factory=lambda session: repository,
        process_session_id='process-a',
    )

    await coordinator.refresh(now)
    await coordinator.refresh(now + timedelta(seconds=5))
    await coordinator.refresh(now + timedelta(seconds=10))

    assert repository.discovery_locks == 2
