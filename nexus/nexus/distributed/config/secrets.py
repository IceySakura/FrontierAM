from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class BasicCredential(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)

    username: str = Field(min_length=1)
    token: SecretStr

    def read(self) -> tuple[str, str]:
        return self.username, self.token.get_secret_value()


class MySQLTraceDatabaseCredential(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)

    provider_type: Literal['mysql'] = 'mysql'
    username: str = Field(min_length=1)
    password: SecretStr

    def read(self) -> tuple[str, str]:
        return self.username, self.password.get_secret_value()


class TencentCloudTraceDatabaseCredential(BaseModel):
    """One API key pair, usable for every Tencent Cloud product it is authorised for.
    ``provider_type`` only records its first product; the real distinction is shape, not product."""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)

    provider_type: Literal['tencent_cloud_dcdb', 'tencent_cloud_tdmysql']
    secret_id: SecretStr = Field(min_length=1)
    secret_key: SecretStr = Field(min_length=1)

    def read(self) -> tuple[str, str]:
        return self.secret_id.get_secret_value(), self.secret_key.get_secret_value()


# Retained so existing imports and isinstance checks keep working; both names now refer to
# the single credential type, which is the point.
TencentCloudDCDBTraceDatabaseCredential = TencentCloudTraceDatabaseCredential
TencentCloudTDMySQLTraceDatabaseCredential = TencentCloudTraceDatabaseCredential


TraceDatabaseCredential = Annotated[
    MySQLTraceDatabaseCredential | TencentCloudTraceDatabaseCredential,
    Field(discriminator='provider_type'),
]


class KubernetesProviderCredential(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)

    provider_type: Literal['kubernetes']
    token: SecretStr
    ca_certificate: SecretStr


class TkexProviderCredential(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)

    provider_type: Literal['tkex']
    token: SecretStr


ProviderCredential = Annotated[
    KubernetesProviderCredential | TkexProviderCredential,
    Field(discriminator='provider_type'),
]


class DistributedSecrets(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)

    internal_token: SecretStr = SecretStr('')
    tai_application_token: SecretStr | None = None
    submit_token_hash_keys: dict[str, SecretStr] = Field(
        default_factory=lambda: {'local': SecretStr('local-dev-submit-token-key')}
    )
    artifact_ticket_keys: dict[str, SecretStr] = Field(
        default_factory=lambda: {'local': SecretStr('local-dev-artifact-upload-key')}
    )
    artifact_writer: BasicCredential | None = None
    artifact_maintenance: BasicCredential | None = None
    # Shared service credential the ``artifact_service`` store presents in the
    # ``X-System-Secret`` header when talking to the backend's passwordless
    # ``/system`` prefix. Optional only outside production; the store factory /
    # production validation requires it whenever ``artifact_service`` is the
    # active provider or a route target.
    artifact_service_shared_secret: SecretStr | None = None
    provider_credentials: dict[str, ProviderCredential] = Field(default_factory=dict)
    trace_cursor_keys: dict[str, SecretStr] = Field(default_factory=dict)
    trace_read_credentials: dict[str, MySQLTraceDatabaseCredential] = Field(default_factory=dict)
    run_trace_credentials: dict[str, SecretStr] = Field(default_factory=dict)
    zhiyan_log_credentials: dict[str, SecretStr] = Field(default_factory=dict)
    metrics_query: BasicCredential | None = None
    trace_database_credentials: dict[str, TraceDatabaseCredential] = Field(default_factory=dict)

    @field_validator('trace_database_credentials', mode='before')
    @classmethod
    def default_legacy_trace_database_credentials_to_mysql(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized: dict[str, Any] = {}
        for credential_ref, credential in value.items():
            if isinstance(credential, dict) and 'provider_type' not in credential:
                credential = {'provider_type': 'mysql', **credential}
            normalized[credential_ref] = credential
        return normalized

    def read_internal_token(self) -> str:
        return self.internal_token.get_secret_value()

    def read_tai_application_token(self) -> str:
        if self.tai_application_token is None:
            raise ValueError('Tai application token is not configured')
        return self.tai_application_token.get_secret_value()

    def read_run_trace_credential(self, credential_ref: str) -> str:
        credential = self.run_trace_credentials.get(credential_ref)
        if credential is None:
            raise ValueError(f'run trace credential {credential_ref!r} is not configured')
        return credential.get_secret_value()

    def read_zhiyan_log_credential(self, credential_ref: str) -> str:
        credential = self.zhiyan_log_credentials.get(credential_ref)
        if credential is None:
            raise ValueError(f'Zhiyan log credential {credential_ref!r} is not configured')
        return credential.get_secret_value()

    def resolve_trace_database_credential(self, credential_ref: str) -> TraceDatabaseCredential | None:
        return self.trace_database_credentials.get(credential_ref)

    def submit_token_keyring(self) -> dict[str, str]:
        return {key_id: value.get_secret_value() for key_id, value in self.submit_token_hash_keys.items()}

    def artifact_ticket_keyring(self) -> dict[str, str]:
        return {key_id: value.get_secret_value() for key_id, value in self.artifact_ticket_keys.items()}

    def read_artifact_service_shared_secret(self) -> str | None:
        if self.artifact_service_shared_secret is None:
            return None
        return self.artifact_service_shared_secret.get_secret_value()


__all__ = [
    'BasicCredential',
    'DistributedSecrets',
    'KubernetesProviderCredential',
    'MySQLTraceDatabaseCredential',
    'ProviderCredential',
    'TencentCloudDCDBTraceDatabaseCredential',
    'TencentCloudTDMySQLTraceDatabaseCredential',
    'TencentCloudTraceDatabaseCredential',
    'TkexProviderCredential',
    'TraceDatabaseCredential',
]
