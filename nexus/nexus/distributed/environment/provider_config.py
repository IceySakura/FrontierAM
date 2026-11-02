from __future__ import annotations

import re

from collections.abc import Collection, Mapping
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, model_validator

from nexus.distributed.errors import DomainConflictError


_CREDENTIAL_REF_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')


class KubernetesProviderConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    provider_type: Literal['kubernetes']
    namespace: str = Field(min_length=1)
    auth_type: Literal['in_cluster', 'secret']
    api_url: str = ''
    credential_ref: str | None = None
    tls_strict: bool = True
    manager_resource_request_ratio: float = Field(default=1.0, gt=0, le=1)

    @model_validator(mode='after')
    def validate_credentials(self) -> KubernetesProviderConfig:
        if self.auth_type == 'in_cluster':
            if self.api_url or self.credential_ref is not None:
                raise ValueError('in_cluster_kubernetes_credentials_must_be_implicit')
            return self
        if not self.api_url or not self.credential_ref:
            raise ValueError('kubernetes_secret_credentials_required')
        if not self.api_url.startswith('https://'):
            raise ValueError('provider_endpoint_must_use_https')
        _validate_credential_ref(self.credential_ref)
        return self


class TkexZhiyanLogConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    log_name: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
    agent_image: str = Field(
        pattern=(
            r'^mirrors\.tencent\.com/zhiyan-agent/agent-integration'
            r'(?::[A-Za-z0-9][A-Za-z0-9._-]{0,127}|@sha256:[0-9a-f]{64})$'
        )
    )
    volume_size_bytes: Annotated[StrictInt, Field(ge=1)]


class TkexProviderConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    provider_type: Literal['tkex']
    namespace: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_account: str = Field(min_length=1)
    credential_ref: str = Field(min_length=1)
    cluster_name: str = Field(min_length=1)
    project_name: str = Field(min_length=1)
    zhiyan_log: TkexZhiyanLogConfig | None = None
    builder_zhiyan_log: TkexZhiyanLogConfig | None = None

    @model_validator(mode='after')
    def validate_credentials(self) -> TkexProviderConfig:
        endpoint = urlsplit(self.base_url)
        if (
            endpoint.scheme not in {'http', 'https'}
            or not endpoint.hostname
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError('provider_endpoint_must_use_http_transport')
        _validate_credential_ref(self.credential_ref)
        return self


ProviderConfig = Annotated[
    KubernetesProviderConfig | TkexProviderConfig,
    Field(discriminator='provider_type'),
]
_PROVIDER_CONFIG_ADAPTER = TypeAdapter(ProviderConfig)


def parse_provider_config(value: Mapping[str, Any]) -> ProviderConfig:
    return _PROVIDER_CONFIG_ADAPTER.validate_python(dict(value))


def parse_provider_snapshot(value: Mapping[str, Any], *, provider_config_ref: str) -> ProviderConfig:
    """Parse the immutable, current-version ProviderConfig allocation snapshot."""

    if {'api_token_file', 'token_file', 'ca_file'}.intersection(value):
        raise ValueError(
            f'legacy provider snapshot {provider_config_ref!r} is unsupported; drain allocations before upgrade'
        )
    return parse_provider_config(value)


def require_provider_type_enabled(provider_type: str, enabled_provider_types: Collection[str]) -> None:
    if provider_type in {'kubernetes', 'tkex'} and provider_type not in enabled_provider_types:
        raise DomainConflictError('provider_type_not_enabled', detail=provider_type)


def _validate_credential_ref(value: str) -> None:
    if _CREDENTIAL_REF_PATTERN.fullmatch(value) is None:
        raise ValueError('invalid_provider_credential_ref')


__all__ = [
    'KubernetesProviderConfig',
    'ProviderConfig',
    'TkexZhiyanLogConfig',
    'TkexProviderConfig',
    'parse_provider_config',
    'parse_provider_snapshot',
    'require_provider_type_enabled',
]
