from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.config import (
    DistributedSecrets,
    KubernetesProviderCredential,
    ProviderCredential,
    TkexProviderCredential,
)
from nexus.distributed.config.store import (
    GLOBAL_CONFIG_SCOPE,
    ConfigVersionConflictError,
    get_config_override,
    set_config_override,
)
from nexus.distributed.errors import DomainConflictError


class ProviderCredentialRepository:
    """Manage provider credentials inside the global distributed secret scope."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, credential_ref: str) -> ProviderCredential | None:
        override = await get_config_override(
            self.session,
            role=GLOBAL_CONFIG_SCOPE,
            identity=GLOBAL_CONFIG_SCOPE,
            secret=True,
            include_secret_values=True,
        )
        if override is None:
            return None
        credentials = _parse_provider_credentials(override.values.get('provider_credentials', {}))
        return credentials.get(credential_ref)

    async def configured_types(self) -> dict[str, str]:
        override = await get_config_override(
            self.session,
            role=GLOBAL_CONFIG_SCOPE,
            identity=GLOBAL_CONFIG_SCOPE,
            secret=True,
            include_secret_values=True,
        )
        if override is None:
            return {}
        credentials = _parse_provider_credentials(override.values.get('provider_credentials', {}))
        return {credential_ref: credential.provider_type for credential_ref, credential in credentials.items()}

    async def upsert(
        self,
        credential_ref: str,
        credential: ProviderCredential,
        *,
        actor: str,
    ) -> None:
        override = await get_config_override(
            self.session,
            role=GLOBAL_CONFIG_SCOPE,
            identity=GLOBAL_CONFIG_SCOPE,
            secret=True,
            include_secret_values=True,
        )
        values = dict(override.values) if override is not None else {}
        credentials = dict(values.get('provider_credentials') or {})
        credentials[credential_ref] = _credential_payload(credential)
        values['provider_credentials'] = credentials
        try:
            await set_config_override(
                self.session,
                role=GLOBAL_CONFIG_SCOPE,
                identity=GLOBAL_CONFIG_SCOPE,
                values=values,
                expected_version=override.version if override is not None else 0,
                actor=actor,
                secret=True,
            )
        except ConfigVersionConflictError as exc:
            raise DomainConflictError('provider_credential_version_conflict') from exc


def _parse_provider_credentials(value: object) -> dict[str, ProviderCredential]:
    if not isinstance(value, Mapping):
        raise ValueError('provider_credentials_must_be_an_object')
    return DistributedSecrets.model_validate({'provider_credentials': dict(value)}).provider_credentials


def _credential_payload(credential: ProviderCredential) -> dict[str, str]:
    if isinstance(credential, KubernetesProviderCredential):
        return {
            'provider_type': credential.provider_type,
            'token': credential.token.get_secret_value(),
            'ca_certificate': credential.ca_certificate.get_secret_value(),
        }
    if isinstance(credential, TkexProviderCredential):
        return {
            'provider_type': credential.provider_type,
            'token': credential.token.get_secret_value(),
        }
    raise TypeError(f'unsupported provider credential: {type(credential).__name__}')


__all__ = ['ProviderCredentialRepository']
