from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import uuid

from datetime import timedelta

from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.control_plane.leases import ShardLeaseCoordinator
from nexus.distributed.policies.models import PolicyExecutorConfig, PolicyResult, SignalObservation
from nexus.distributed.policies.workers import PolicyWorkerPool
from nexus.distributed.repositories.policy import POLICY_SHARD_GROUP, PolicyRepository, policy_scope_key
from nexus.distributed.utils.clock import utcnow


logger = logging.getLogger(__name__)


class PolicyExecutor:
    def __init__(
        self, *, session_factory: async_sessionmaker, config: PolicyExecutorConfig, shard_count: int, owner_id: str
    ):
        self.session_factory, self.config, self.shard_count = session_factory, config, shard_count
        self.leases = ShardLeaseCoordinator(
            session_factory=session_factory,
            shard_count=shard_count,
            owner_id=owner_id,
            shard_group=POLICY_SHARD_GROUP,
            lease_ttl=timedelta(seconds=30),
            renew_interval=5,
            ownership_limit=0,
            discover_interval=10,
            discover_jitter=0.2,
        )
        self.pool = PolicyWorkerPool(size=config.workers, startup_seconds=config.worker_startup_seconds)
        self.config_hashes = {name: binding.fingerprint(config.imports) for name, binding in config.bindings.items()}
        self.active: dict[str, tuple[str, asyncio.Task]] = {}
        self.stopping = asyncio.Event()

    async def initialize(self) -> None:
        async with self.session_factory() as session:
            await PolicyRepository(session, shard_count=self.shard_count).ensure_scopes(self.config, now=utcnow())
            await session.commit()

    async def run_forever(self) -> None:
        try:
            while not self.stopping.is_set():
                try:
                    await self.tick_once()
                except Exception as exc:
                    logger.warning('Policy executor tick failed: error_type=%s', type(exc).__name__)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.stopping.wait(), timeout=self.config.poll_seconds)
        finally:
            await self.close()

    async def tick_once(self) -> None:
        shards = await self.leases.refresh(utcnow())
        for key, (_, task) in list(self.active.items()):
            if task.done():
                self.active.pop(key)
                if not task.cancelled() and task.exception() is not None:
                    logger.warning('Policy publish failed: error_type=%s', type(task.exception()).__name__)
        free = self.config.workers - len(self.active)
        if free <= 0:
            return
        async with self.session_factory() as session:
            claims = await PolicyRepository(session, shard_count=self.shard_count).due(
                shards,
                now=utcnow(),
                idle_seconds=self.config.idle_seconds,
                limit=512,
                configured_keys=[
                    policy_scope_key(binding.kind, name, scope)
                    for name, binding in self.config.bindings.items()
                    for scope in binding.scopes
                ],
            )
        per_binding = self.config.workers if len(self.config.bindings) == 1 else max(1, self.config.workers // 2)
        for claim in claims:
            binding = self.config.bindings.get(claim['binding_id'])
            if (
                claim['key'] in self.active
                or binding is None
                or binding.mode == 'disabled'
                or claim['scope_id'] not in binding.scopes
                or claim['config_hash'] != self.config_hashes[claim['binding_id']]
            ):
                continue
            if sum(name == claim['binding_id'] for name, _ in self.active.values()) >= per_binding:
                continue
            task = asyncio.create_task(self._evaluate(claim), name=f'policy:{claim["binding_id"]}:{claim["key"][:8]}')
            self.active[claim['key']] = (claim['binding_id'], task)
            free -= 1
            if free == 0:
                break

    async def _evaluate(self, claim: dict) -> None:
        binding = self.config.bindings[claim['binding_id']]
        request = {
            'imports': self.config.imports,
            'kind': binding.kind,
            'policy': binding.policy,
            'source': binding.source,
            'source_config': binding.scopes[claim['scope_id']].source_config,
            'scope_id': claim['scope_id'],
            'params': binding.params,
            'previous_state': claim['state'],
            'max_age_seconds': binding.max_age_seconds,
        }
        result = observation = None
        error = None
        started = utcnow()
        try:
            response = await self.pool.run(request, timeout=binding.timeout_seconds)
            result = PolicyResult.model_validate(response['evaluation'])
            observation = SignalObservation.model_validate(response['observation'])
            now = utcnow()
            if not started <= observation.observed_at <= now or not now < result.valid_until <= observation.valid_until:
                raise ValueError('invalid policy observation time')
            if observation.valid_until > observation.observed_at + timedelta(seconds=binding.max_age_seconds):
                raise ValueError('policy observation exceeds maximum age')
        except Exception as exc:
            result = observation = None
            error = 'policy_timeout' if isinstance(exc, TimeoutError) else 'policy_evaluation_failed'
            logger.warning('Policy evaluation failed: binding=%s reason=%s', claim['binding_id'], error)
        delay = (
            binding.refresh_seconds
            if result is not None
            else min(binding.refresh_seconds, 2 ** min(8, claim['failures']))
        )
        delay *= random.uniform(1, 1.1)
        async with self.session_factory() as session:
            await PolicyRepository(session, shard_count=self.shard_count).publish(
                claim,
                owner_id=self.leases.owner_id,
                binding=binding,
                result=result,
                observation=observation,
                error=error,
                evaluation_id=uuid.uuid4().hex,
                now=utcnow(),
                delay=delay,
            )
            await session.commit()

    async def close(self) -> None:
        self.stopping.set()
        tasks = [task for _, task in self.active.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.active.clear()
        await self.pool.close()
        with contextlib.suppress(Exception):
            await self.leases.release()
