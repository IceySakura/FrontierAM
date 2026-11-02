from __future__ import annotations

import hashlib
import json
import math

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


REQUEST_RATE_KIND = 'request_rate'


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class PolicySelection(PolicyModel):
    binding: str = Field(min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_-]+$')
    scopes: list[str] = Field(min_length=1, max_length=16)

    @model_validator(mode='after')
    def unique_scopes(self):
        if len(set(self.scopes)) != len(self.scopes) or any(not value or len(value) > 128 for value in self.scopes):
            raise ValueError('policy scopes must be unique, nonempty and at most 128 characters')
        return self


class RequestPolicySelection(PolicyModel):
    binding: str = Field(min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_-]+$')
    scope: str = Field(min_length=1, max_length=128)

    @model_validator(mode='after')
    def validate_scope(self):
        if not self.scope or len(self.scope) > 128:
            raise ValueError('request policy scope must be nonempty and at most 128 characters')
        return self


class RequestBudgetConfig(PolicyModel):
    lease_seconds: float = Field(default=10, gt=0, le=300)
    renew_before_seconds: float = Field(default=4, gt=0, le=120)
    safety_margin_seconds: float = Field(default=0.2, ge=0, le=5)

    @model_validator(mode='after')
    def validate_renewal(self):
        if self.renew_before_seconds >= self.lease_seconds:
            raise ValueError('renew_before_seconds must be less than lease_seconds')
        if self.safety_margin_seconds >= self.lease_seconds:
            raise ValueError('safety_margin_seconds must be less than lease_seconds')
        return self


class PolicyScope(PolicyModel):
    source_config: dict[str, Any] = Field(default_factory=dict)


class PolicyBinding(PolicyModel):
    kind: str = Field(default='scheduler', pattern=r'^[a-zA-Z0-9_-]{1,32}$')
    policy: str = Field(default='infer_proxy_tide', min_length=1, max_length=64)
    source: str = Field(default='infer_proxy_capacity', min_length=1, max_length=64)
    revision: int = Field(default=1, ge=1)
    mode: Literal['shadow', 'enforce', 'disabled'] = 'shadow'
    on_unavailable: Literal['defer', 'default'] = 'defer'
    environments: list[str] = Field(min_length=1, max_length=64)
    principals: list[str] = Field(default_factory=list, max_length=256)
    scopes: dict[str, PolicyScope] = Field(min_length=1, max_length=256)
    params: dict[str, Any] = Field(default_factory=dict)
    refresh_seconds: float = Field(default=60, ge=1, le=3600)
    timeout_seconds: float = Field(default=10, gt=0, le=120)
    max_age_seconds: float = Field(default=180, ge=1, le=86400)
    max_defer_seconds: float = Field(default=30, ge=1, le=300)

    @model_validator(mode='after')
    def validate_scope_names(self):
        if any(not name or len(name) > 128 for name in self.scopes):
            raise ValueError('invalid policy scope name')
        if self.max_age_seconds <= self.refresh_seconds:
            raise ValueError('policy max_age_seconds must exceed refresh_seconds')
        if self.kind == REQUEST_RATE_KIND and self.mode == 'enforce' and self.on_unavailable != 'defer':
            raise ValueError('request_rate enforce mode requires on_unavailable=defer')
        return self

    def fingerprint(self, imports: list[str]) -> str:
        value = {'binding': self.model_dump(mode='json'), 'imports': imports}
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class PolicyExecutorConfig(PolicyModel):
    bindings: dict[str, PolicyBinding] = Field(default_factory=dict, max_length=64)
    defaults: dict[str, PolicySelection] = Field(default_factory=dict, max_length=64)
    request_defaults: dict[str, RequestPolicySelection] = Field(default_factory=dict, max_length=64)
    imports: list[str] = Field(default_factory=list, max_length=16)
    workers: int = Field(default=2, ge=1, le=16)
    poll_seconds: float = Field(default=1, ge=0.1, le=30)
    idle_seconds: float = Field(default=3600, ge=60, le=86400)
    worker_startup_seconds: float = Field(default=30, ge=1, le=120)

    @model_validator(mode='after')
    def validate_bindings(self):
        if sum(len(binding.scopes) for binding in self.bindings.values()) > 512:
            raise ValueError('at most 512 policy scopes are supported per executor')
        for name in self.bindings:
            if len(name) > 64 or not name or any(not (c.isascii() and (c.isalnum() or c in '_-')) for c in name):
                raise ValueError('invalid policy binding name')
        for environment, selection in self.defaults.items():
            binding = self.bindings.get(selection.binding)
            if binding is None or environment not in binding.environments or binding.kind != 'scheduler':
                raise ValueError('default policy must reference a scheduler binding in its environment')
            if not set(selection.scopes) <= binding.scopes.keys():
                raise ValueError('default policy references unknown scopes')
        for environment, selection in self.request_defaults.items():
            binding = self.bindings.get(selection.binding)
            if binding is None or environment not in binding.environments or binding.kind != REQUEST_RATE_KIND:
                raise ValueError('default request policy must reference a request_rate binding in its environment')
            if selection.scope not in binding.scopes:
                raise ValueError('default request policy references unknown scope')
        return self


class SignalObservation(PolicyModel):
    version: str = Field(min_length=1, max_length=128)
    observed_at: datetime
    valid_until: datetime
    values: dict[str, Any]


class PolicyInput(PolicyModel):
    scope_id: str
    now: datetime
    params: dict[str, Any]
    observation: SignalObservation


class PolicyResult(PolicyModel):
    decision: dict[str, Any]
    next_state: dict[str, Any] = Field(default_factory=dict)
    valid_until: datetime
    reason_code: str = Field(min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_-]+$')


class SchedulerDecision(PolicyModel):
    allow: bool
    preferred_managers: list[str] = Field(default_factory=list, max_length=256)

    @model_validator(mode='after')
    def validate_managers(self):
        if len(self.preferred_managers) != len(set(self.preferred_managers)):
            raise ValueError('duplicate preferred managers')
        if any(not item or len(item) > 128 for item in self.preferred_managers):
            raise ValueError('invalid manager identity')
        return self


class RequestRateDecision(PolicyModel):
    allow: bool
    rate_per_minute: float

    @model_validator(mode='after')
    def validate_rate(self):
        if isinstance(self.rate_per_minute, bool) or not math.isfinite(self.rate_per_minute):
            raise ValueError('rate_per_minute must be a finite number')
        if self.allow:
            if self.rate_per_minute <= 0:
                raise ValueError('allowed request rate must be positive')
        elif self.rate_per_minute != 0:
            raise ValueError('denied request rate must be zero')
        return self
