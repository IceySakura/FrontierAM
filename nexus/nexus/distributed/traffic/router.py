from __future__ import annotations

import hashlib

from dataclasses import dataclass


_BASIS_POINTS = 10_000


@dataclass(frozen=True, slots=True)
class TrafficTarget:
    name: str
    weight_bps: int


@dataclass(frozen=True, slots=True)
class TrafficPolicy:
    component: str
    seed: str
    version: int
    targets: tuple[TrafficTarget, ...]

    def __post_init__(self) -> None:
        if not self.component:
            raise ValueError('traffic policy component is required')
        if not self.seed:
            raise ValueError('traffic policy seed is required')
        if isinstance(self.version, bool) or self.version < 1:
            raise ValueError('traffic policy version must be at least 1')
        if not self.targets:
            raise ValueError('traffic policy requires at least one target')

        names: set[str] = set()
        total = 0
        for target in self.targets:
            if not target.name:
                raise ValueError('traffic target name is required')
            if target.name in names:
                raise ValueError('traffic target names must be unique')
            names.add(target.name)
            if isinstance(target.weight_bps, bool) or not isinstance(target.weight_bps, int):
                raise ValueError('traffic target weight must be an integer in range 0..10000')
            if not 0 <= target.weight_bps <= _BASIS_POINTS:
                raise ValueError('traffic target weight is out of range 0..10000')
            total += target.weight_bps

        if not any(target.weight_bps > 0 for target in self.targets):
            raise ValueError('traffic policy requires at least one positive target weight')
        if total != _BASIS_POINTS:
            raise ValueError('traffic target weights must sum to 10000')


class WeightedTrafficRouter:
    """Choose a target deterministically from a stable key and a policy."""

    def __init__(self, policy: TrafficPolicy) -> None:
        self.policy = policy

    def choose(self, stable_key: str) -> str:
        if not stable_key:
            raise ValueError('traffic stable key is required')
        payload = f'{self.policy.seed}\0{stable_key}'.encode()
        digest = hashlib.sha256(payload).digest()
        bucket = int.from_bytes(digest[:8], 'big') % _BASIS_POINTS

        cumulative = 0
        for target in self.policy.targets:
            cumulative += target.weight_bps
            if bucket < cumulative:
                return target.name
        raise AssertionError('validated traffic policy did not select a target')


__all__ = ['TrafficPolicy', 'TrafficTarget', 'WeightedTrafficRouter']
