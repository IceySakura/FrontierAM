from __future__ import annotations

import asyncio
import hashlib
import random
import uuid

from collections.abc import Callable
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nexus.distributed.repositories.shard_lease import ShardLeaseRepository
from nexus.distributed.utils.clock import utcnow


RepositoryFactory = Callable[[AsyncSession], ShardLeaseRepository]


class ShardLeaseCoordinator:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        shard_count: int,
        owner_id: str,
        shard_group: str,
        lease_ttl: timedelta,
        renew_interval: float,
        ownership_limit: int = 0,
        discover_interval: float = 20.0,
        discover_jitter: float = 0.2,
        repository_factory: RepositoryFactory | None = None,
        process_session_id: str | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.shard_count = shard_count
        self.base_owner_id = owner_id
        suffix = process_session_id or uuid.uuid4().hex[:16]
        self.owner_id = f'{owner_id[:110]}:{suffix}'
        self.shard_group = shard_group
        self.lease_ttl = lease_ttl
        self.renew_interval = renew_interval
        self.ownership_limit = ownership_limit
        self.discover_interval = discover_interval
        self.discover_jitter = discover_jitter
        self._repository_factory = repository_factory
        self._owned_shards: set[int] = set()
        self._last_renew_at: datetime | None = None
        self._last_discover_at: datetime | None = None
        self._next_discover_at: datetime | None = None
        self._discover_cursor = 0
        self._recover_owned = False
        self._random = random.Random(f'{self.owner_id}:{shard_group}:shard-discover')
        self._refresh_lock = asyncio.Lock()
        self._validated_shard_count = False

    @property
    def owned_shard_count(self) -> int:
        return len(self._owned_shards)

    @property
    def valid_until(self) -> datetime | None:
        """Read-only expiry for consumers that must not mutate coordinator state."""
        return self._last_renew_at + self.lease_ttl if self._last_renew_at and self._owned_shards else None

    async def refresh(self, now: datetime | None = None) -> list[int]:
        current = now or utcnow()
        async with self._refresh_lock:
            return await self._refresh_locked(current)

    def snapshot(self, now: datetime | None = None) -> list[int]:
        current = now or utcnow()
        if self._last_renew_at is None or current - self._last_renew_at >= self.lease_ttl:
            self.invalidate()
            return []
        retained = self._retained()
        self._owned_shards = set(retained)
        return retained

    async def release(self, now: datetime | None = None) -> None:
        current = now or utcnow()
        async with self._refresh_lock:
            owned = set(self._owned_shards)
            recover_owned = self._recover_owned
            self.invalidate()
            if not owned and not recover_owned:
                return
            try:
                async with self.session_factory() as session:
                    repository = self._repository(session)
                    if recover_owned:
                        owned.update(
                            await repository.list_owned(
                                owner_id=self.owner_id,
                                shard_group=self.shard_group,
                                now=current,
                            )
                        )
                    if owned:
                        await repository.release(
                            owner_id=self.owner_id,
                            shard_group=self.shard_group,
                            shard_ids=sorted(owned),
                            now=current,
                        )
                    await session.commit()
            except BaseException:
                self._recover_owned = True
                raise
            self._recover_owned = False

    def invalidate(self) -> None:
        self._recover_owned = self._recover_owned or bool(self._owned_shards)
        self._owned_shards.clear()
        self._last_renew_at = None
        self._last_discover_at = None
        self._next_discover_at = None

    async def _refresh_locked(self, now: datetime) -> list[int]:
        if not self._should_discover(now) and not self._should_renew(now):
            return self.snapshot(now)
        random_state = self._random.getstate()
        discover = self._should_discover(now)
        try:
            async with self.session_factory() as session:
                repository = self._repository(session)
                if not self._validated_shard_count:
                    await repository.validate_monotonic_shard_count(shard_group=self.shard_group)
                if discover:
                    await repository.lock_discovery(shard_group=self.shard_group)
                recovered: list[int] = []
                if self._recover_owned:
                    recovered = await repository.list_owned(
                        owner_id=self.owner_id,
                        shard_group=self.shard_group,
                        now=now,
                    )
                    self._owned_shards = set(recovered)
                requested, cursor = self._requested_shards(discover=discover)
                excess = sorted(set(recovered) - set(requested))
                if excess:
                    await repository.release(
                        owner_id=self.owner_id,
                        shard_group=self.shard_group,
                        shard_ids=excess,
                        now=now,
                    )
                owned = await repository.renew(
                    owner_id=self.owner_id,
                    shard_group=self.shard_group,
                    shard_ids=requested,
                    lease_ttl=self.lease_ttl,
                    now=now,
                )
                if discover and len(owned) < self._limit():
                    owned, cursor = await self._fill_discovery_window(
                        repository,
                        owned=owned,
                        attempted=requested,
                        cursor=cursor,
                        now=now,
                    )
                await session.commit()
        except BaseException:
            self._random.setstate(random_state)
            self.invalidate()
            raise
        self._owned_shards = set(owned)
        self._recover_owned = False
        self._validated_shard_count = True
        self._last_renew_at = now
        if discover:
            self._discover_cursor = cursor
            self._last_discover_at = now
            self._next_discover_at = now + timedelta(seconds=self._discover_interval_seconds())
        return sorted(owned)

    def _repository(self, session: AsyncSession) -> ShardLeaseRepository:
        if self._repository_factory is not None:
            return self._repository_factory(session)
        return ShardLeaseRepository(session, shard_count=self.shard_count)

    def _requested_shards(self, *, discover: bool) -> tuple[list[int], int]:
        retained = self._retained()
        if not discover:
            return retained, self._discover_cursor
        if self._limit() == self.shard_count:
            return list(range(self.shard_count)), 0
        remaining = self._limit() - len(retained)
        if remaining <= 0:
            return retained, self._discover_cursor
        ordered = self.owner_hash_ordered_shards()
        retained_set = set(retained)
        offset = self._discover_cursor % self.shard_count
        discovered: list[int] = []
        examined = 0
        while examined < self.shard_count and len(discovered) < remaining:
            shard_id = ordered[(offset + examined) % self.shard_count]
            examined += 1
            if shard_id not in retained_set:
                discovered.append(shard_id)
        return [*retained, *discovered], (offset + examined) % self.shard_count

    async def _fill_discovery_window(
        self,
        repository: ShardLeaseRepository,
        *,
        owned: list[int],
        attempted: list[int],
        cursor: int,
        now: datetime,
    ) -> tuple[list[int], int]:
        owned_set = set(owned)
        attempted_set = set(attempted)
        ordered = self.owner_hash_ordered_shards()
        while len(owned_set) < self._limit() and len(attempted_set) < self.shard_count:
            remaining = self._limit() - len(owned_set)
            probe: list[int] = []
            examined = 0
            offset = cursor % self.shard_count
            while examined < self.shard_count and len(probe) < remaining:
                shard_id = ordered[(offset + examined) % self.shard_count]
                examined += 1
                if shard_id in attempted_set:
                    continue
                attempted_set.add(shard_id)
                probe.append(shard_id)
            cursor = (offset + examined) % self.shard_count
            if not probe:
                break
            acquired = await repository.renew(
                owner_id=self.owner_id,
                shard_group=self.shard_group,
                shard_ids=probe,
                lease_ttl=self.lease_ttl,
                now=now,
            )
            owned_set.update(shard_id for shard_id in acquired if shard_id in probe)
        return sorted(owned_set), cursor

    def _retained(self) -> list[int]:
        if len(self._owned_shards) <= self._limit():
            return sorted(self._owned_shards)
        return [shard_id for shard_id in self.owner_hash_ordered_shards() if shard_id in self._owned_shards][
            : self._limit()
        ]

    def _limit(self) -> int:
        if self.ownership_limit <= 0:
            return self.shard_count
        return min(self.ownership_limit, self.shard_count)

    def owner_hash_ordered_shards(self) -> list[int]:
        return sorted(
            range(self.shard_count),
            key=lambda shard_id: (
                int.from_bytes(
                    hashlib.blake2b(
                        f'{self.shard_group}:{self.owner_id}:{shard_id}'.encode(),
                        digest_size=8,
                    ).digest(),
                    'big',
                ),
                shard_id,
            ),
        )

    def _discover_interval_seconds(self) -> float:
        if self.discover_jitter <= 0:
            return self.discover_interval
        return max(
            0.0,
            self.discover_interval * (1 + self._random.uniform(-self.discover_jitter, self.discover_jitter)),
        )

    def _should_discover(self, now: datetime) -> bool:
        return self._next_discover_at is None or now >= self._next_discover_at

    def _should_renew(self, now: datetime) -> bool:
        if not self._owned_shards or self._last_renew_at is None:
            return False
        return (now - self._last_renew_at).total_seconds() >= self.renew_interval


__all__ = ['ShardLeaseCoordinator']
