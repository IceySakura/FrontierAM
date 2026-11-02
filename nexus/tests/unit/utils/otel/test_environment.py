from __future__ import annotations

import importlib

import pytest

from nexus.utils.otel import redaction


def _load_environment():
    spec = importlib.util.find_spec('nexus.utils.otel.environment')
    assert spec is not None, 'nexus.utils.otel.environment is not implemented'
    return importlib.import_module('nexus.utils.otel.environment')


def test_disabled_managed_environment_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)

    overlay = _load_environment().managed_otel_env(
        process_role='runtime-worker',
        service_name='nexus-runtime-worker',
    )

    assert overlay == {}


def test_managed_environment_preserves_standard_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://collector:4318')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_PROTOCOL', 'http/protobuf')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', 'authorization=credential-sentinel')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE', '/run/secrets/collector-ca.pem')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_TRACES_TIMEOUT', '15')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_METRICS_HEADERS', 'authorization=metrics-secret')
    monkeypatch.setenv('OTEL_BSP_MAX_QUEUE_SIZE', '4096')
    monkeypatch.setenv('OTEL_PYTHON_DISABLED_INSTRUMENTATIONS', 'logging')
    monkeypatch.setenv('OTEL_RESOURCE_ATTRIBUTES', 'deployment.environment=prod')

    overlay = _load_environment().managed_otel_env(
        process_role='runtime-worker',
        service_name='nexus-runtime-worker',
        identifiers={'nexus.runtime.id': 'runtime-1'},
    )

    assert overlay['NEXUS_OTEL_ENABLED'] == 'true'
    assert overlay['NEXUS_OTEL_PROCESS_ROLE'] == 'runtime-worker'
    assert overlay['OTEL_SERVICE_NAME'] == 'nexus-runtime-worker'
    assert overlay['OTEL_EXPORTER_OTLP_ENDPOINT'] == 'http://collector:4318'
    assert overlay['OTEL_EXPORTER_OTLP_PROTOCOL'] == 'http/protobuf'
    assert overlay['OTEL_EXPORTER_OTLP_HEADERS'] == 'authorization=credential-sentinel'
    assert overlay['OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE'] == '/run/secrets/collector-ca.pem'
    assert overlay['OTEL_EXPORTER_OTLP_TRACES_TIMEOUT'] == '15'
    assert overlay['OTEL_EXPORTER_OTLP_METRICS_HEADERS'] == 'authorization=metrics-secret'
    assert overlay['OTEL_BSP_MAX_QUEUE_SIZE'] == '4096'
    assert overlay['OTEL_PYTHON_DISABLED_INSTRUMENTATIONS'] == 'logging'
    assert overlay['OTEL_RESOURCE_ATTRIBUTES'] == 'deployment.environment=prod,nexus.runtime.id=runtime-1'


@pytest.mark.parametrize('process_role', ['runtime-base', 'runtime-worker'])
def test_runtime_roles_use_dedicated_exporter_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    process_role: str,
) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'https://host-collector.invalid:4318')
    monkeypatch.setenv(
        'NEXUS_OTEL_RUNTIME_EXPORTER_OTLP_ENDPOINT',
        'http://runtime-collector.invalid:4318',
    )

    overlay = _load_environment().managed_otel_env(
        process_role=process_role,
        service_name=f'nexus-{process_role}',
    )

    assert overlay['OTEL_EXPORTER_OTLP_ENDPOINT'] == 'http://runtime-collector.invalid:4318'


def test_non_runtime_role_keeps_host_exporter_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'https://host-collector.invalid:4318')
    monkeypatch.setenv(
        'NEXUS_OTEL_RUNTIME_EXPORTER_OTLP_ENDPOINT',
        'http://runtime-collector.invalid:4318',
    )

    overlay = _load_environment().managed_otel_env(
        process_role='mcp',
        service_name='nexus-mcp-shell',
    )

    assert overlay['OTEL_EXPORTER_OTLP_ENDPOINT'] == 'https://host-collector.invalid:4318'


def test_managed_environment_does_not_copy_unrelated_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('UNRELATED_SECRET', 'credential-sentinel')

    overlay = _load_environment().managed_otel_env(process_role='mcp', service_name='nexus-mcp-shell')

    assert 'UNRELATED_SECRET' not in overlay


def test_exporter_credential_helpers_only_redact_connection_credentials() -> None:
    payload = {
        'environment': {
            'OTEL_EXPORTER_OTLP_HEADERS': 'authorization=secret',
            'OTEL_EXPORTER_OTLP_METRICS_HEADERS': 'authorization=metrics-secret',
            'OTEL_EXPORTER_OTLP_ENDPOINT': 'https://user:password@collector.invalid/v1/traces',
            'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT': 'https://collector.invalid/v1/traces',
            'OTEL_RESOURCE_ATTRIBUTES': 'deployment.environment=prod,tps.tenant.id=tenant-secret',
            'NORMAL_VALUE': 'visible',
        },
        'items': [
            {'Name': 'OTEL_EXPORTER_OTLP_TRACES_HEADERS', 'Value': 'authorization=trace-secret'},
            {'Name': 'OTEL_EXPORTER_OTLP_LOGS_HEADERS', 'Value': 'authorization=logs-secret'},
            {
                'Name': 'OTEL_EXPORTER_OTLP_ENDPOINT',
                'Value': 'https://named-user:named-password@collector.invalid/v1/traces',
            },
            {
                'Name': 'OTEL_RESOURCE_ATTRIBUTES',
                'Value': 'tps.tenant.id=nested-tenant-secret,deployment.environment=prod',
            },
        ],
    }

    redacted = redaction.redact_exporter_credentials(payload)

    assert redacted['environment'] == {
        'OTEL_EXPORTER_OTLP_HEADERS': '<redacted>',
        'OTEL_EXPORTER_OTLP_METRICS_HEADERS': '<redacted>',
        'OTEL_EXPORTER_OTLP_ENDPOINT': '<redacted>',
        'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT': 'https://collector.invalid/v1/traces',
        'OTEL_RESOURCE_ATTRIBUTES': '<redacted>',
        'NORMAL_VALUE': 'visible',
    }
    assert redacted['items'][0]['Value'] == '<redacted>'
    assert redacted['items'][1]['Value'] == '<redacted>'
    assert redacted['items'][2]['Value'] == '<redacted>'
    assert redacted['items'][3]['Value'] == '<redacted>'
    assert redaction.exporter_credential_values(payload) == [
        'authorization=secret',
        'authorization=metrics-secret',
        'https://user:password@collector.invalid/v1/traces',
        'deployment.environment=prod,tps.tenant.id=tenant-secret',
        'authorization=trace-secret',
        'authorization=logs-secret',
        'https://named-user:named-password@collector.invalid/v1/traces',
        'tps.tenant.id=nested-tenant-secret,deployment.environment=prod',
    ]
