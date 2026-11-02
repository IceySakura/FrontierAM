from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import urllib.parse

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CommandRunner = Callable[..., Any]
TDSQL_TRACE_CREDENTIAL_REF = 'tdsql-password'
TDSQL_MAINTENANCE_CREDENTIAL_REF = 'tdsql-maintenance'


@dataclass(frozen=True, slots=True)
class TDSQLTraceConfig:
    host: str
    user: str
    password: str = field(repr=False)
    db: str
    port: int = 3306
    read_host: str | None = None
    read_port: int | None = None
    maintenance_user: str | None = None
    maintenance_password: str | None = field(default=None, repr=False)
    account_provider_type: str = 'mysql'
    dcdb_region: str | None = None
    dcdb_instance_id: str | None = None
    dcdb_api_endpoint: str | None = None
    dcdb_secret_id: str | None = field(default=None, repr=False)
    dcdb_secret_key: str | None = field(default=None, repr=False)
    dcdb_account_host: str = '%'

    def __post_init__(self) -> None:
        for name in ('host', 'user', 'password', 'db'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f'TDSQL trace {name} must be a non-empty string')
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError('TDSQL trace port must be between 1 and 65535')
        if (self.read_host is None) != (self.read_port is None):
            raise ValueError('TDSQL trace read host and port must be provided together')
        if self.read_host is not None and not self.read_host:
            raise ValueError('TDSQL trace read_host must be a non-empty string')
        if self.read_port is not None and (
            isinstance(self.read_port, bool) or not isinstance(self.read_port, int) or not 1 <= self.read_port <= 65535
        ):
            raise ValueError('TDSQL trace read_port must be between 1 and 65535')
        if self.read_host is None:
            if self.account_provider_type != 'mysql' or any(
                value is not None
                for value in (
                    self.maintenance_user,
                    self.maintenance_password,
                    self.dcdb_region,
                    self.dcdb_instance_id,
                    self.dcdb_api_endpoint,
                    self.dcdb_secret_id,
                    self.dcdb_secret_key,
                )
            ):
                raise ValueError('TDSQL trace account provider requires a read endpoint')
            return
        if self.account_provider_type == 'mysql':
            if not self.maintenance_user or not self.maintenance_password:
                raise ValueError('TDSQL MySQL account provider requires maintenance credentials')
            if any(
                value is not None
                for value in (
                    self.dcdb_region,
                    self.dcdb_instance_id,
                    self.dcdb_api_endpoint,
                    self.dcdb_secret_id,
                    self.dcdb_secret_key,
                )
            ):
                raise ValueError('TDSQL MySQL account provider cannot use DCDB credentials')
            return
        if self.account_provider_type != 'tencent_cloud_dcdb':
            raise ValueError(f'unsupported TDSQL account provider: {self.account_provider_type}')
        if self.maintenance_user is not None or self.maintenance_password is not None:
            raise ValueError('TDSQL DCDB account provider cannot use MySQL maintenance credentials')
        if not all((self.dcdb_region, self.dcdb_instance_id, self.dcdb_secret_id, self.dcdb_secret_key)):
            raise ValueError('TDSQL DCDB account provider settings must be provided together')
        if self.dcdb_api_endpoint not in {
            None,
            'dcdb.tencentcloudapi.com',
            'dcdb.internal.tencentcloudapi.com',
        }:
            raise ValueError('unsupported TDSQL DCDB API endpoint')
        if not self.dcdb_account_host:
            raise ValueError('TDSQL DCDB account host must be a non-empty string')

    def backend_config(self) -> dict[str, Any]:
        return {
            'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
            'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
            'config': {
                'host': self.host,
                'port': self.port,
                'user': self.user,
                'db': self.db,
            },
            'secretConfig': {'password': TDSQL_TRACE_CREDENTIAL_REF},
        }

    def target_fingerprint(self) -> str:
        identity = json.dumps(
            {'host': self.host, 'port': self.port, 'user': self.user, 'db': self.db},
            sort_keys=True,
            separators=(',', ':'),
        )
        return hashlib.sha256(identity.encode('utf-8')).hexdigest()

    def access_config(self) -> dict[str, Any] | None:
        if self.read_host is None:
            return None
        assert self.read_port is not None
        config: dict[str, Any] = {
            'read_host': self.read_host,
            'read_port': self.read_port,
            'database': self.db,
            'maintenance_credential_ref': TDSQL_MAINTENANCE_CREDENTIAL_REF,
            'enabled': True,
        }
        if self.account_provider_type == 'tencent_cloud_dcdb':
            assert self.dcdb_region is not None
            assert self.dcdb_instance_id is not None
            config['account_provider'] = {
                'provider_type': 'tencent_cloud_dcdb',
                'region': self.dcdb_region,
                'instance_id': self.dcdb_instance_id,
                'api_endpoint': self.dcdb_api_endpoint or 'dcdb.tencentcloudapi.com',
                'account_host': self.dcdb_account_host,
            }
        return config

    def maintenance_credential(self) -> dict[str, str] | None:
        if self.read_host is None:
            return None
        if self.account_provider_type == 'tencent_cloud_dcdb':
            assert self.dcdb_secret_id is not None
            assert self.dcdb_secret_key is not None
            return {
                'provider_type': 'tencent_cloud_dcdb',
                'secret_id': self.dcdb_secret_id,
                'secret_key': self.dcdb_secret_key,
            }
        assert self.maintenance_user is not None
        assert self.maintenance_password is not None
        return {
            'username': self.maintenance_user,
            'password': self.maintenance_password,
        }


def load_tdsql_trace_config(path: Path) -> TDSQLTraceConfig:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'TDSQL trace config is not a file: {path}')
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ValueError('TDSQL trace config must use mode 0600')
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('TDSQL trace config must be valid UTF-8 JSON') from exc
    if not isinstance(payload, dict):
        raise ValueError('TDSQL trace config must be a JSON object')
    base_fields = {'host', 'port', 'user', 'password', 'db'}
    read_fields = {'read_host', 'read_port'}
    mysql_fields = {'maintenance_user', 'maintenance_password'}
    cloud_fields = {
        'account_provider_type',
        'dcdb_region',
        'dcdb_instance_id',
        'dcdb_api_endpoint',
        'dcdb_secret_id',
        'dcdb_secret_key',
    }
    valid_fields = {
        frozenset(base_fields),
        frozenset(base_fields | read_fields | mysql_fields),
        frozenset(base_fields | read_fields | mysql_fields | {'account_provider_type'}),
        frozenset(base_fields | read_fields | cloud_fields),
        frozenset(base_fields | read_fields | cloud_fields | {'dcdb_account_host'}),
    }
    if set(payload) not in valid_fields:
        raise ValueError(
            'TDSQL trace config fields must contain writer fields and one complete account provider configuration'
        )
    try:
        return TDSQLTraceConfig(**payload)
    except TypeError as exc:
        raise ValueError('TDSQL trace config has invalid field types') from exc


def apply_kubernetes_secret(
    run: CommandRunner,
    *,
    kubectl: str,
    namespace: str,
    name: str,
    values: Mapping[str, str],
) -> None:
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
        path = Path(handle.name)
        os.chmod(path, 0o600)
        for key, value in values.items():
            handle.write(f'{key}={value}\n')
    try:
        manifest = run(
            [
                kubectl,
                '-n',
                namespace,
                'create',
                'secret',
                'generic',
                name,
                '--from-env-file',
                str(path),
                '--dry-run=client',
                '-o',
                'yaml',
            ],
            capture=True,
        ).stdout
        run([kubectl, 'apply', '-f', '-'], input_text=manifest)
    finally:
        path.unlink(missing_ok=True)


def database_secret_values(
    *,
    release_name: str,
    namespace: str,
    password: str,
    database_name: str = 'nexus_dist',
) -> dict[str, str]:
    encoded_password = urllib.parse.quote(password, safe='')
    return {
        'database-url': (
            f'mysql+asyncmy://root:{encoded_password}@{release_name}-mysql.{namespace}.svc.cluster.local:3306/'
            f'{database_name}?charset=utf8mb4'
        )
    }


def mysql_secret_values(*, password: str) -> dict[str, str]:
    return {'mysql-root-password': password}


def create_database_bootstrap_secrets(
    run: CommandRunner,
    *,
    kubectl: str,
    namespace: str,
    release_name: str,
    database_secret_name: str,
    mysql_secret_name: str,
    password: str,
) -> None:
    apply_kubernetes_secret(
        run,
        kubectl=kubectl,
        namespace=namespace,
        name=database_secret_name,
        values=database_secret_values(
            release_name=release_name,
            namespace=namespace,
            password=password,
        ),
    )
    apply_kubernetes_secret(
        run,
        kubectl=kubectl,
        namespace=namespace,
        name=mysql_secret_name,
        values=mysql_secret_values(password=password),
    )


def local_database_url(*, port: int, password: str, database_name: str = 'nexus_dist') -> str:
    encoded_password = urllib.parse.quote(password, safe='')
    return f'mysql+asyncmy://root:{encoded_password}@127.0.0.1:{port}/{database_name}?charset=utf8mb4'


def platform_config_values(
    *,
    application_public_base_url: str,
    application_internal_base_url: str,
    workload_image: str,
    builder_image: str,
    namespace: str,
    release_name: str,
    database_secret_name: str,
    enabled_providers: Sequence[str] = (),
    fake_runtime_base_url: str = '',
    environment_extra_env: Sequence[Mapping[str, Any]] = (),
    artifact_provider: str = 'local',
    artifact_base_url: str = '',
    artifact_repository: str = '',
    artifact_routing_enabled: bool = False,
    artifact_routing_seed: str = 'artifact-store-v1',
    artifact_routing_version: int = 1,
    artifact_routes: Sequence[Mapping[str, Any]] = (),
    artifact_service_base_url: str = '',
    artifact_service_system_prefix: str = '/system',
    artifact_service_namespace_prefix: str = '/_nexus_dist',
    tdsql_trace: TDSQLTraceConfig | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        'auth_mode': 'disabled',
        'application_public_base_url': application_public_base_url,
        'application_internal_base_url': application_internal_base_url,
        'environment_base_image': workload_image,
        'environment_builder_image': builder_image,
        'environment_namespace': namespace,
        'environment_release_name': release_name,
        'environment_app_name': 'nexus-distributed',
        'environment_database_url_secret_name': database_secret_name,
        'environment_enabled_providers': list(enabled_providers),
        'environment_fake_runtime_base_url': fake_runtime_base_url,
        'environment_extra_env': [dict(item) for item in environment_extra_env],
        'artifact_store_provider': artifact_provider,
    }
    if artifact_provider == 'generic_repository':
        values.update({
            'artifact_generic_base_url': artifact_base_url,
            'artifact_generic_repository': artifact_repository,
        })
    if artifact_routing_enabled:
        values.update({
            'artifact_store_routing_enabled': True,
            'artifact_store_routing_seed': artifact_routing_seed,
            'artifact_store_routing_version': artifact_routing_version,
            'artifact_store_routes': [dict(route) for route in artifact_routes],
            'artifact_service_base_url': artifact_service_base_url,
            'artifact_service_system_prefix': artifact_service_system_prefix,
            'artifact_service_namespace_prefix': artifact_service_namespace_prefix,
        })
    if tdsql_trace is not None:
        values['run_trace_sinks'] = {'tdsql': tdsql_trace.backend_config()}
        values['default_run_trace_sinks'] = ['tdsql']
        values['batch_task_store_database'] = tdsql_trace.db
        access_config = tdsql_trace.access_config()
        if access_config is not None:
            values['trace_database_access'] = {'tdsql': access_config}
    return values


def platform_secret_values(
    environment: Mapping[str, str],
    *,
    artifact_username: str | None = None,
    artifact_token: str | None = None,
    tdsql_trace: TDSQLTraceConfig | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        'internal_token': environment.get('NEXUS_DIST_INTERNAL_TOKEN', 'local-dynamic-internal-token'),
        'artifact_ticket_keys': _json_object(
            environment.get(
                'NEXUS_DIST_ARTIFACT_UPLOAD_TICKET_KEYS',
                '{"local":"local-dev-artifact-upload-key"}',
            ),
            field='NEXUS_DIST_ARTIFACT_UPLOAD_TICKET_KEYS',
        ),
        'submit_token_hash_keys': _json_object(
            environment.get(
                'NEXUS_DIST_SUBMIT_TOKEN_HASH_KEYS',
                '{"local":"local-dev-submit-token-key"}',
            ),
            field='NEXUS_DIST_SUBMIT_TOKEN_HASH_KEYS',
        ),
        'tai_application_token': environment.get('NEXUS_DIST_TAI_APPLICATION_TOKEN', '0' * 32),
    }
    if (artifact_username is None) != (artifact_token is None):
        raise ValueError('artifact username and token must be provided together')
    if artifact_username is not None and artifact_token is not None:
        credential = {'username': artifact_username, 'token': artifact_token}
        values['artifact_writer'] = credential
        values['artifact_maintenance'] = credential
    if tdsql_trace is not None:
        values['run_trace_credentials'] = {TDSQL_TRACE_CREDENTIAL_REF: tdsql_trace.password}
    return values


def platform_control_gateway_secret_values(*, tdsql_trace: TDSQLTraceConfig | None = None) -> dict[str, Any]:
    if tdsql_trace is None:
        return {}
    credential = tdsql_trace.maintenance_credential()
    if credential is None:
        return {}
    return {
        'trace_database_credentials': {
            TDSQL_MAINTENANCE_CREDENTIAL_REF: credential,
        }
    }


def seed_platform_config(
    run: CommandRunner,
    *,
    python: str,
    database_url: str,
    config_values: Mapping[str, Any],
    secret_values: Mapping[str, Any],
    role_secret_values: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    actor: str = 'system:cluster-e2e',
) -> None:
    writes = [
        ('set', '*', '*', config_values),
        ('secret-set', '*', '*', secret_values),
        *(('secret-set', role, identity, values) for (role, identity), values in (role_secret_values or {}).items()),
    ]
    for command, role, identity, values in writes:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
            path = Path(handle.name)
            os.chmod(path, 0o600)
            json.dump(dict(values), handle, ensure_ascii=False, sort_keys=True)
        try:
            run(
                [
                    python,
                    '-m',
                    'nexus.cli.main',
                    'distributed',
                    'config',
                    command,
                    '--database-url',
                    database_url,
                    '--role',
                    role,
                    '--identity',
                    identity,
                    '--expected-version',
                    '0',
                    '--actor',
                    actor,
                    '--file',
                    str(path),
                ],
                timeout=120,
            )
        finally:
            path.unlink(missing_ok=True)


def _json_object(raw: str, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'{field} must be a JSON object') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{field} must be a JSON object')
    return value


__all__ = [
    'TDSQLTraceConfig',
    'apply_kubernetes_secret',
    'create_database_bootstrap_secrets',
    'database_secret_values',
    'local_database_url',
    'load_tdsql_trace_config',
    'mysql_secret_values',
    'platform_config_values',
    'platform_control_gateway_secret_values',
    'platform_secret_values',
    'seed_platform_config',
]
