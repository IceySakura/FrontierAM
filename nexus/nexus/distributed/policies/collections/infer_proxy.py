from __future__ import annotations

import math

from datetime import timedelta
from typing import Any

import httpx

from pydantic import Field, model_validator

from nexus.distributed.policies.models import (
    PolicyInput,
    PolicyModel,
    PolicyResult,
    SchedulerDecision,
    SignalObservation,
)
from nexus.distributed.policies.registry import policy_registry, signal_source_registry
from nexus.distributed.utils.clock import utcnow


class CapacitySourceConfig(PolicyModel):
    model: str = Field(min_length=1, max_length=128)
    api_key: str = Field(min_length=1, repr=False)
    url: str = 'http://api.taiji.woa.com/openapi/model/capacity'

    @model_validator(mode='after')
    def validate_source(self):
        from urllib.parse import urlsplit

        parsed = urlsplit(self.url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('invalid capacity URL')
        if not self.api_key.strip():
            raise ValueError('capacity credential unavailable')
        return self


class TideParams(PolicyModel):
    pass


@signal_source_registry.register('infer_proxy_capacity')
class InferProxyCapacitySource:
    config_model = CapacitySourceConfig

    async def read(self, config: dict[str, Any], *, max_age_seconds: float) -> SignalObservation:
        settings = CapacitySourceConfig.model_validate(config)
        key = settings.api_key.strip()
        if not key:
            raise ValueError('capacity credential unavailable')
        # Bound the response before decoding; never persist the raw provider response or credential.
        async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
            async with client.stream('POST', settings.url, json={'model': settings.model, 'api_key': key}) as response:
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 65536:
                        raise ValueError('capacity response too large')
        import json

        data = json.loads(body)
        if not isinstance(data, dict) or data.get('success') is False or 'error' in data:
            raise ValueError('capacity response unavailable')
        values = {}
        remaining = data.get('key_remaining_raw_qpm')
        if isinstance(remaining, int) and not isinstance(remaining, bool) and remaining >= 0:
            values['remaining'] = remaining
        advice = data.get('scaling_advice')
        hard_brake = advice.get('hard_brake') if isinstance(advice, dict) else None
        if isinstance(hard_brake, bool):
            values['hard_brake'] = hard_brake
        factor = advice.get('factor') if isinstance(advice, dict) else None
        if isinstance(factor, (int, float)) and not isinstance(factor, bool) and math.isfinite(factor) and factor > 0:
            values['factor'] = float(factor)
        now = utcnow()
        return SignalObservation(
            version=now.isoformat(),
            observed_at=now,
            valid_until=now + timedelta(seconds=max_age_seconds),
            values=values,
        )


@policy_registry.register('infer_proxy_tide')
class InferProxyTidePolicy:
    params_model = TideParams
    kind = 'scheduler'
    decision_model = SchedulerDecision

    def evaluate(self, inputs: PolicyInput, previous_state: dict[str, Any]) -> PolicyResult:
        TideParams.model_validate(inputs.params)
        signal = inputs.observation
        remaining = signal.values.get('remaining')
        hard_brake = signal.values.get('hard_brake')
        if signal.valid_until <= inputs.now:
            reason = 'signal_expired'
        elif hard_brake is True:
            reason = 'hard_brake'
        elif remaining == 0 and not isinstance(remaining, bool):
            reason = 'quota_exhausted'
        elif not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 0 or hard_brake is not False:
            reason = 'signal_incomplete'
        else:
            reason = 'capacity_available'
        return PolicyResult(
            decision={'allow': reason == 'capacity_available'},
            valid_until=signal.valid_until,
            reason_code=reason,
        )
