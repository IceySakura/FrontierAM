from __future__ import annotations

import importlib


def test_redaction_handles_structured_and_shell_rendered_exporter_credentials() -> None:
    spec = importlib.util.find_spec('nexus.utils.otel.redaction')
    assert spec is not None
    redaction = importlib.import_module('nexus.utils.otel.redaction')
    secret = "authorization=Bearer a'b"
    payload = {
        'environment': {
            'OTEL_EXPORTER_OTLP_HEADERS': secret,
            'NORMAL_VALUE': 'visible',
        },
        'items': [
            {'Name': 'OTEL_RESOURCE_ATTRIBUTES', 'Value': 'tps.tenant.id=tenant-secret'},
        ],
    }

    redacted = redaction.redact_exporter_credentials(payload)
    sensitive_values = redaction.exporter_credential_values(payload)
    text = redaction.redact_sensitive_text(
        f"raw={secret}; shell='authorization=Bearer a'\"'\"'b'",
        sensitive_values,
    )

    assert redacted['environment']['OTEL_EXPORTER_OTLP_HEADERS'] == '<redacted>'
    assert redacted['environment']['NORMAL_VALUE'] == 'visible'
    assert redacted['items'][0]['Value'] == '<redacted>'
    assert secret not in text
    assert 'tenant-secret' not in repr(redacted)
