from __future__ import annotations

from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlatformMetricsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)

    enabled: bool = False
    endpoint: str = ''
    app_mark: str = Field(default='', max_length=128)
    environment: str = Field(default='', max_length=64)
    meter_name: str = Field(default='nexus_cloud', min_length=1, max_length=64)
    export_interval_seconds: float = Field(default=60, ge=1, le=3600)
    export_timeout_seconds: float = Field(default=5, gt=0, le=30)

    @model_validator(mode='after')
    def validate_destination(self) -> PlatformMetricsConfig:
        if self.endpoint:
            url = urlsplit(self.endpoint)
            if (
                url.scheme not in {'http', 'https'}
                or not url.hostname
                or url.username is not None
                or url.password is not None
                or url.query
                or url.fragment
                or not url.path.endswith('/v1/metrics')
            ):
                raise ValueError('metrics endpoint must be an HTTP(S) /v1/metrics URL without credentials or query')
        if self.enabled and not all(value.strip() for value in (self.endpoint, self.app_mark, self.environment)):
            raise ValueError('enabled metrics require endpoint, app_mark and environment')
        return self


class StatusSamplingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    enabled: bool = True
    interval_seconds: float = Field(default=30, ge=1, le=300)
    query_timeout_seconds: float = Field(default=1, gt=0, le=5)
    group_timeout_seconds: float = Field(default=3, gt=0, le=10)
    round_timeout_seconds: float = Field(default=10, gt=0, le=30)
    stale_after_seconds: float = Field(default=60, ge=1, le=600)
    live_cache_seconds: float = Field(default=1, ge=0, le=1)
    manager_summary_timeout_seconds: float = Field(default=3, gt=0, le=10)
    manager_summary_cache_seconds: float = Field(default=10, ge=0, le=60)
    release_grace_seconds: float = Field(default=300, ge=1, le=86400)

    @model_validator(mode='after')
    def validate_budgets(self) -> StatusSamplingConfig:
        if not self.query_timeout_seconds <= self.group_timeout_seconds <= self.round_timeout_seconds:
            raise ValueError('status budgets must satisfy query <= group <= round')
        if self.stale_after_seconds < self.interval_seconds:
            raise ValueError('status stale_after_seconds must cover at least one collection interval')
        return self


class BusinessStatisticsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    enabled: bool = True
    query_timeout_seconds: float = Field(default=8, gt=0, le=15)
    request_timeout_seconds: float = Field(default=15, gt=0, le=30)
    queue_timeout_seconds: float = Field(default=1, gt=0, le=5)
    cache_seconds: float = Field(default=300, ge=0, le=300)
    cache_entries: int = Field(default=16, ge=1, le=128)
    max_name_groups: int = Field(default=1000, ge=10, le=5000)
    max_response_bytes: int = Field(default=2 * 1024 * 1024, ge=1024, le=8 * 1024 * 1024)

    @model_validator(mode='after')
    def validate_budgets(self) -> BusinessStatisticsConfig:
        if self.query_timeout_seconds > self.request_timeout_seconds:
            raise ValueError('statistics query budget must not exceed the request budget')
        return self


class MetricsHistoryConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)

    enabled: bool = False
    endpoint: str = 'http://openapi.zhiyan.woa.com/monitor/v2/api/chart/info/query'
    project_name: str = ''
    project_id: int = Field(default=0, ge=0)
    app_mark: str = ''
    environment: str = ''
    group: str = 'nexus_cloud'
    max_segment_hours: int = Field(default=168, ge=1, le=168)
    upstream_requests_per_minute: int = Field(default=750, ge=60, le=900)
    retention_days: int | None = Field(default=None, ge=1, le=3650)
    histogram_aggregation_supported: bool = True

    @model_validator(mode='after')
    def validate_query(self) -> MetricsHistoryConfig:
        url = urlsplit(self.endpoint)
        if (
            url.scheme not in {'http', 'https'}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError('metrics history endpoint must be HTTP(S) without credentials or query')
        if self.enabled and (
            not self.project_id or not all((self.project_name, self.app_mark, self.environment, self.group))
        ):
            raise ValueError('enabled metrics history requires an explicit project, application, environment and group')
        return self
