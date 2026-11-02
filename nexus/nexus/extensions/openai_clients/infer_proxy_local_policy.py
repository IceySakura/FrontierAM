"""Process-local Tide admission shared by Infer Proxy clients with the same key and model."""

from __future__ import annotations

import logging
import threading

from asyncio import sleep
from time import monotonic
from typing import Any

import httpx

from pydantic import Field, model_validator

from nexus.core.request_admission import BoundRequestAdmission, RequestAttempt, RequestPermit, SmoothRatePacer
from nexus.distributed.policies.collections.infer_proxy_request_rate import (
    InferProxyRequestCapacitySource,
    InferProxyRequestRateParams,
    InferProxyRequestRatePolicy,
    PositiveRate,
    RequestCapacitySourceConfig,
)
from nexus.distributed.policies.models import PolicyInput, RequestRateDecision
from nexus.distributed.utils.clock import utcnow


logger = logging.getLogger(__name__)


class LocalRequestPolicyConfig(InferProxyRequestRateParams):
    quota_qpm: PositiveRate = Field(gt=0)
    poll_interval: float = Field(default=60, ge=60)
    max_age_seconds: float = Field(default=180, gt=0)
    capacity_url: str = 'http://api.taiji.woa.com/openapi/model/capacity'

    @model_validator(mode='after')
    def validate_refresh_window(self):
        if self.max_age_seconds <= self.poll_interval:
            raise ValueError('local_request_policy max_age_seconds must exceed poll_interval')
        return self


class InferProxyLocalAdmission:
    def __init__(self, *, model: str, api_key: str, config: LocalRequestPolicyConfig) -> None:
        self.config = config
        self._source_config = RequestCapacitySourceConfig(
            model=model, api_key=api_key, quota_qpm=config.quota_qpm, url=config.capacity_url
        ).model_dump()
        self._state: dict[str, Any] = {}
        self._rate = 0.0
        self._valid_until = 0.0
        self._next_poll_at = 0.0
        self._refreshing = False
        self._pacer = SmoothRatePacer()
        # Shared across threads/event loops; no loop-bound tasks or locks are retained.
        self._lock = threading.Lock()

    async def _refresh_if_due(self) -> None:
        with self._lock:
            now = monotonic()
            if self._refreshing or now < self._next_poll_at:
                return
            self._refreshing = True
            self._next_poll_at = now + self.config.poll_interval
            previous_state = dict(self._state)
        try:
            observation = await InferProxyRequestCapacitySource().read(
                self._source_config, max_age_seconds=self.config.max_age_seconds
            )
            result = InferProxyRequestRatePolicy().evaluate(
                PolicyInput(
                    scope_id='local',
                    now=utcnow(),
                    params={
                        'quota_qpm': self.config.quota_qpm,
                        'min_quota_ratio': self.config.min_quota_ratio,
                    },
                    observation=observation,
                ),
                previous_state,
            )
            decision = RequestRateDecision.model_validate(result.decision)
            with self._lock:
                now = monotonic()
                # Pauses and expired observations cannot accumulate permits for recovery.
                if now >= self._valid_until or self._rate == 0 or not decision.allow:
                    self._pacer.reset(now)
                self._rate = decision.rate_per_minute
                self._valid_until = now + max(0.0, (result.valid_until - utcnow()).total_seconds())
                self._state = result.next_state
            logger.debug(
                'Local Infer Proxy policy model=%s qpm=%s reason=%s',
                self._source_config['model'],
                decision.rate_per_minute,
                result.reason_code,
            )
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                'Local Infer Proxy capacity query failed (%s); retaining unexpired decision', type(exc).__name__
            )
        except BaseException:
            # Let another caller refresh immediately if the querying task was cancelled.
            with self._lock:
                self._next_poll_at = 0.0
            raise
        finally:
            with self._lock:
                self._refreshing = False

    async def acquire(self, request: RequestAttempt) -> RequestPermit:
        while True:
            await self._refresh_if_due()
            with self._lock:
                now = monotonic()
                delay = 1.0  # Observe refreshes from other loops without sharing asyncio events.
                if now < self._valid_until:
                    delay = min(delay, self._pacer.try_acquire(qpm=self._rate, now=now))
                    if delay == 0:
                        return RequestPermit(
                            execution_id=request.execution_id,
                            attempt_id=request.attempt_id,
                            session_id='local',
                            deadline_at=self._valid_until,
                        )
                if not self._refreshing:
                    delay = min(delay, max(0.0, self._next_poll_at - now))
            await sleep(delay)


# Preserve pacing and policy history between tasks for the lifetime of the process.
_admissions: dict[tuple[str, str], InferProxyLocalAdmission] = {}
_admissions_lock = threading.Lock()


def local_request_admission(*, model: Any, api_key: Any, config: Any, client_id: str) -> BoundRequestAdmission:
    if not isinstance(model, str) or not model.strip():
        raise ValueError('local_request_policy requires a configured model')
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError('local_request_policy requires a non-empty API key')
    model, api_key = model.strip(), api_key.strip()
    settings = LocalRequestPolicyConfig.model_validate(config)
    with _admissions_lock:
        admission = _admissions.get((api_key, model))
        if admission is None:
            admission = InferProxyLocalAdmission(model=model, api_key=api_key, config=settings)
            _admissions[api_key, model] = admission
        elif admission.config != settings:
            raise ValueError('local_request_policy settings must match for clients sharing an API key and model')
    return BoundRequestAdmission(admission=admission, execution_id=client_id, attempt_id='local')
