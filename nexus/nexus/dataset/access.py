"""Task-scoped data access, independent of dataset and file identities."""

from __future__ import annotations

import os

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from nexus.dataset.auth import DatasetAuthenticator, WedataAuthenticator, WedataKey, build_default_authenticator
from nexus.utils.cos import AsyncCosClient, CosConfig, CosCredentials


def credentials_from_env() -> dict[str, dict[str, str]]:
    """Read a fresh submission snapshot, without credential files or global caches."""
    result: dict[str, dict[str, str]] = {}
    names = ('COS_SECRET_ID', 'COS_SECRET_KEY', 'COS_TOKEN')
    if any(name in os.environ for name in names):
        credentials = CosCredentials.from_shared_payload({
            'secret_id': os.environ.get(names[0]),
            'secret_key': os.environ.get(names[1]),
            'token': os.environ.get(names[2]),
        })
        result['cos'] = credentials.to_shared_payload()
    if 'WEDATA_KEY' in os.environ:
        result['wedata'] = WedataKey.from_raw(os.environ['WEDATA_KEY'], source='WEDATA_KEY').to_auth_payload()
    return result


@dataclass
class DataAccessContext:
    """Explicit credentials and owned clients for one reader or task lifetime."""

    credentials: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    authenticator: DatasetAuthenticator | None = field(default=None, repr=False)
    allow_local_auth: bool = False
    _clients: dict[tuple[str, str], AsyncCosClient] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_env(cls) -> DataAccessContext:
        return cls(
            credentials=credentials_from_env(), allow_local_auth=os.getenv('NEXUS_DATA_ACCESS_SUBMISSION') != '1'
        )

    def cos_client(self, bucket: str, region: str) -> AsyncCosClient:
        key = (bucket, region)
        if key not in self._clients:
            payload = self.credentials.get('cos')
            if payload is None:
                raise ValueError('COS credentials are missing for this data access context')
            self._clients[key] = AsyncCosClient(
                CosConfig(bucket=bucket, region=region), CosCredentials.from_shared_payload(payload)
            )
        return self._clients[key]

    def dataset_authenticator(self) -> DatasetAuthenticator | None:
        if self.authenticator is not None:
            return self.authenticator
        if payload := self.credentials.get('wedata'):
            return WedataAuthenticator(key=WedataKey.from_mapping(payload, source='execution credential'))
        if self.allow_local_auth:
            return build_default_authenticator()
        return None

    async def close(self) -> None:
        try:
            for client in self._clients.values():
                await client.close()
        finally:
            self._clients.clear()

    async def __aenter__(self) -> DataAccessContext:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()


_CURRENT_ACCESS: ContextVar[DataAccessContext | None] = ContextVar('dataset_access', default=None)


def current_data_access() -> DataAccessContext | None:
    return _CURRENT_ACCESS.get()


@contextmanager
def bind_data_access(context: DataAccessContext):
    """Bind task access without changing the process environment."""
    token = _CURRENT_ACCESS.set(context)
    try:
        yield context
    finally:
        _CURRENT_ACCESS.reset(token)
