"""Trusted fixture plugins used to test the public policy extension contract."""

import time

from datetime import timedelta

from pydantic import BaseModel

from nexus.distributed.policies.models import PolicyResult, SchedulerDecision, SignalObservation
from nexus.distributed.policies.registry import policy_registry, signal_source_registry
from nexus.distributed.utils.clock import utcnow


@signal_source_registry.register('test_signal')
class TestSignal:
    async def read(self, config, *, max_age_seconds):
        now = utcnow()
        return SignalObservation(
            version='observation-1',
            observed_at=now,
            valid_until=now + timedelta(seconds=max_age_seconds),
            values=config,
        )


@policy_registry.register('test_gate')
class TestGate:
    kind = 'scheduler'
    decision_model = SchedulerDecision

    def evaluate(self, inputs, previous_state):
        time.sleep(inputs.params.get('delay', 0))
        return PolicyResult(
            decision={'allow': inputs.observation.values.get('allow', True)},
            next_state={'calls': previous_state.get('calls', 0) + 1},
            valid_until=inputs.observation.valid_until,
            reason_code='test_signal',
        )


class RateDecision(BaseModel):
    qpm: int


@policy_registry.register('test_rate')
class TestRate:
    kind = 'request_rate'
    decision_model = RateDecision

    def evaluate(self, inputs, previous_state):
        return PolicyResult(
            decision={'qpm': inputs.observation.values['qpm']},
            next_state={},
            valid_until=inputs.observation.valid_until,
            reason_code='test_rate',
        )


@policy_registry.register('test_slow_constructor')
class SlowConstructorGate(TestGate):
    def __init__(self):
        time.sleep(30)
