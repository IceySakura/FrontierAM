"""Built-in Tide request_rate signal source and policy for Infer Proxy."""

from __future__ import annotations

import json
import math

from datetime import timedelta
from typing import Any

import httpx

from pydantic import Field, StrictFloat, StrictInt, model_validator

from nexus.distributed.policies.models import (
    REQUEST_RATE_KIND,
    PolicyInput,
    PolicyModel,
    PolicyResult,
    RequestRateDecision,
    SignalObservation,
)
from nexus.distributed.policies.registry import policy_registry, signal_source_registry
from nexus.distributed.utils.clock import utcnow


REQUEST_CAPACITY_SOURCE = 'infer_proxy_request_capacity'
REQUEST_RATE_POLICY = 'infer_proxy_request_rate'
MAX_RESPONSE_BYTES = 64 * 1024

PositiveRate = StrictFloat | StrictInt


class RequestCapacitySourceConfig(PolicyModel):
    model: str = Field(min_length=1, max_length=128)
    url: str = 'http://api.taiji.woa.com/openapi/model/capacity'
    timeout_seconds: float = Field(default=5, gt=0, le=30)
    api_key: str | None = Field(default=None, min_length=1, repr=False)
    quota_qpm: PositiveRate | None = Field(default=None, gt=0)

    @model_validator(mode='after')
    def validate_source(self):
        from urllib.parse import urlsplit

        parsed = urlsplit(self.url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('invalid capacity URL')
        if self.api_key is not None and not self.api_key.strip():
            raise ValueError('capacity credential unavailable')
        return self


class InferProxyRequestRateParams(PolicyModel):
    quota_qpm: PositiveRate | None = Field(default=None, gt=0)
    min_quota_ratio: float = Field(default=0.01, gt=0, le=1)


@signal_source_registry.register(REQUEST_CAPACITY_SOURCE)
class InferProxyRequestCapacitySource:
    config_model = RequestCapacitySourceConfig

    async def read(self, config: dict[str, Any], *, max_age_seconds: float) -> SignalObservation:
        settings = RequestCapacitySourceConfig.model_validate(config)
        key = settings.api_key.strip() if settings.api_key is not None else None
        request_body: dict[str, Any] = {'model': settings.model}
        if key is not None:
            request_body['api_key'] = key
        # Bound the response before decoding; never persist the raw provider response or credential.
        async with httpx.AsyncClient(timeout=settings.timeout_seconds, follow_redirects=False) as client:
            async with client.stream('POST', settings.url, json=request_body) as response:
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ValueError('capacity response too large')
        data = json.loads(body)
        if not isinstance(data, dict) or data.get('success') is False or 'error' in data:
            raise ValueError('capacity response unavailable')
        values: dict[str, Any] = {'keyed': key is not None}
        if settings.quota_qpm is not None:
            values['quota_qpm'] = float(settings.quota_qpm)
        remaining = data.get('key_remaining_raw_qpm')
        if isinstance(remaining, int) and not isinstance(remaining, bool) and remaining >= 0:
            values['remaining'] = remaining
        advice = data.get('scaling_advice')
        if isinstance(advice, dict):
            factor = advice.get('factor')
            hard_brake = advice.get('hard_brake')
            if isinstance(hard_brake, bool):
                values['hard_brake'] = hard_brake
            if (
                isinstance(factor, (int, float))
                and not isinstance(factor, bool)
                and math.isfinite(factor)
                and factor > 0
            ):
                values['factor'] = float(factor)
        now = utcnow()
        return SignalObservation(
            version=now.isoformat(),
            observed_at=now,
            valid_until=now + timedelta(seconds=max_age_seconds),
            values=values,
        )


@policy_registry.register(REQUEST_RATE_POLICY)
class InferProxyRequestRatePolicy:
    kind = REQUEST_RATE_KIND
    decision_model = RequestRateDecision
    params_model = InferProxyRequestRateParams

    def evaluate(self, inputs: PolicyInput, previous_state: dict[str, Any]) -> PolicyResult:
        params = InferProxyRequestRateParams.model_validate(inputs.params)
        signal = inputs.observation
        raw_quota = signal.values.get('quota_qpm', params.quota_qpm)
        if (
            isinstance(raw_quota, bool)
            or not isinstance(raw_quota, (int, float))
            or not math.isfinite(raw_quota)
            or raw_quota <= 0
        ):
            raise ValueError('quota_qpm must be a finite positive number')
        quota_qpm = float(raw_quota)
        ratio = previous_state.get('quota_ratio', 1.0)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
            ratio = 1.0
        ratio = min(1.0, max(params.min_quota_ratio, ratio))
        paused = previous_state.get('paused_below_minimum') is True
        state = {'quota_ratio': ratio, 'paused_below_minimum': paused}
        values = signal.values
        remaining = values.get('remaining')
        factor = values.get('factor')
        reason = None
        if signal.valid_until <= inputs.now:
            reason = 'signal_expired'
        elif remaining == 0 and not isinstance(remaining, bool):
            reason = 'quota_exhausted'
        elif (
            (
                values.get('keyed') is not False
                and (isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0)
            )
            or isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or not math.isfinite(factor)
            or factor <= 0
        ):
            reason = 'signal_incomplete'
        if reason is not None:
            return PolicyResult(
                decision={'allow': False, 'rate_per_minute': 0.0},
                next_state=state,
                valid_until=signal.valid_until,
                reason_code=reason,
            )
        assert isinstance(factor, (int, float))
        next_ratio = min(1.0, ratio * factor)
        if (paused and factor <= 1) or next_ratio < params.min_quota_ratio:
            # Keep the last usable ratio; repeated decreases while paused must not prevent recovery.
            state['paused_below_minimum'] = True
            return PolicyResult(
                decision={'allow': False, 'rate_per_minute': 0.0},
                next_state=state,
                valid_until=signal.valid_until,
                reason_code='below_minimum',
            )
        state['quota_ratio'] = next_ratio
        state['paused_below_minimum'] = False
        return PolicyResult(
            decision={'allow': True, 'rate_per_minute': quota_qpm * next_ratio},
            next_state=state,
            valid_until=signal.valid_until,
            reason_code='resumed' if paused else 'adopted' if next_ratio != ratio else 'unchanged',
        )


class StaticRequestCapacityConfig(PolicyModel):
    rate_per_minute: float = Field(ge=0)
    allow: bool = True

    @model_validator(mode='after')
    def validate_static_rate(self):
        if isinstance(self.rate_per_minute, bool) or not math.isfinite(self.rate_per_minute):
            raise ValueError('rate_per_minute must be a finite non-negative number')
        if self.allow and self.rate_per_minute <= 0:
            raise ValueError('allowed static rate_per_minute must be positive')
        return self


class FixedRequestRateParams(PolicyModel):
    rate_per_minute: float | None = Field(default=None, gt=0)


@signal_source_registry.register('static_request_capacity')
class StaticRequestCapacitySource:
    config_model = StaticRequestCapacityConfig

    async def read(self, config: dict[str, Any], *, max_age_seconds: float) -> SignalObservation:
        settings = StaticRequestCapacityConfig.model_validate(config)
        now = utcnow()
        return SignalObservation(
            version=now.isoformat(),
            observed_at=now,
            valid_until=now + timedelta(seconds=max_age_seconds),
            values={
                'allow': settings.allow,
                'rate_per_minute': settings.rate_per_minute if settings.allow else 0.0,
            },
        )


@policy_registry.register('fixed_request_rate')
class FixedRequestRatePolicy:
    kind = REQUEST_RATE_KIND
    decision_model = RequestRateDecision
    params_model = FixedRequestRateParams

    def evaluate(self, inputs: PolicyInput, previous_state: dict[str, Any]) -> PolicyResult:
        params = FixedRequestRateParams.model_validate(inputs.params)
        obs = inputs.observation.values
        allow = obs.get('allow', True) is not False
        raw_rate = params.rate_per_minute if params.rate_per_minute is not None else obs.get('rate_per_minute', 0.0)
        rate = float(raw_rate) if allow and isinstance(raw_rate, (int, float)) and raw_rate > 0 else 0.0
        effective_allow = allow and rate > 0
        return PolicyResult(
            decision={'allow': effective_allow, 'rate_per_minute': rate if effective_allow else 0.0},
            next_state=dict(previous_state),
            valid_until=inputs.observation.valid_until,
            reason_code='fixed_rate_available' if effective_allow else 'fixed_rate_paused',
        )
