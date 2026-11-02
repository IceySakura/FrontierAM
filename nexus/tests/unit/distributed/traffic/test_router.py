from __future__ import annotations

import hashlib

import pytest

from nexus.distributed.traffic.router import TrafficPolicy, TrafficTarget, WeightedTrafficRouter


def _policy(*targets: tuple[str, int], seed: str = 'artifact-store-v1', version: int = 1) -> TrafficPolicy:
    return TrafficPolicy(
        component='test',
        seed=seed,
        version=version,
        targets=tuple(TrafficTarget(name=name, weight_bps=weight) for name, weight in targets),
    )


def test_choice_is_stable_across_calls_and_router_instances() -> None:
    policy = _policy(('old', 9500), ('new', 500))
    first = WeightedTrafficRouter(policy)
    second = WeightedTrafficRouter(policy)

    assert [first.choose('owner/object') for _ in range(5)] == ['old'] * 5
    assert second.choose('owner/object') == first.choose('owner/object')


def test_algorithm_uses_sha256_seed_nul_key_bucket_contract() -> None:
    policy = _policy(('first', 1502), ('second', 8498), seed='contract-v1')
    digest = hashlib.sha256(b'contract-v1\0owner/object').digest()
    assert int.from_bytes(digest[:8], 'big') % 10000 == 1502
    assert WeightedTrafficRouter(policy).choose('owner/object') == 'second'


def test_zero_weight_target_never_receives_new_keys_and_full_weight_is_constant() -> None:
    assert WeightedTrafficRouter(_policy(('old', 10000), ('new', 0))).choose('anything') == 'old'
    assert WeightedTrafficRouter(_policy(('old', 0), ('new', 10000))).choose('anything') == 'new'


def test_policy_version_does_not_change_selection() -> None:
    old = WeightedTrafficRouter(_policy(('old', 9500), ('new', 500), version=1))
    new = WeightedTrafficRouter(_policy(('old', 9500), ('new', 500), version=2))
    assert [old.choose(f'key-{index}') for index in range(100)] == [new.choose(f'key-{index}') for index in range(100)]


def test_weighted_distribution_is_close_to_policy() -> None:
    router = WeightedTrafficRouter(_policy(('old', 9500), ('new', 500)))
    results = [router.choose(f'key-{index}') for index in range(10_000)]
    new_ratio = results.count('new') / len(results)
    assert abs(new_ratio - 0.05) <= 0.015


@pytest.mark.parametrize(
    ('targets', 'match'),
    [
        ([('old', 9000), ('new', 900)], 'sum'),
        ([('old', -1), ('new', 10001)], 'range'),
        ([('old', 10000), ('old', 0)], 'unique'),
        ([('old', 0), ('new', 0)], 'positive'),
    ],
)
def test_invalid_weights_are_rejected(targets: list[tuple[str, int]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _policy(*targets)


@pytest.mark.parametrize(
    ('seed', 'key', 'match'),
    [('', 'key', 'seed'), ('seed', '', 'stable key')],
)
def test_empty_seed_or_key_is_rejected(seed: str, key: str, match: str) -> None:
    if not seed:
        with pytest.raises(ValueError, match=match):
            _policy(('only', 10000), seed=seed)
    else:
        router = WeightedTrafficRouter(_policy(('only', 10000), seed=seed))
        with pytest.raises(ValueError, match=match):
            router.choose(key)


@pytest.mark.parametrize(
    ('component', 'version', 'targets', 'match'),
    [
        ('', 1, (('only', 10000),), 'component'),
        ('test', 0, (('only', 10000),), 'version'),
        ('test', 1, (('', 10000),), 'target name'),
    ],
)
def test_policy_metadata_is_validated(
    component: str,
    version: int,
    targets: tuple[tuple[str, int], ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        TrafficPolicy(
            component=component,
            seed='seed',
            version=version,
            targets=tuple(TrafficTarget(name=name, weight_bps=weight) for name, weight in targets),
        )
