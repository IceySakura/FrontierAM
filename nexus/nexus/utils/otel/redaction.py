from __future__ import annotations

import shlex

from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, urlsplit


def _has_resource_attribute(value: str, expected_key: str) -> bool:
    for item in value.split(','):
        key, separator, _item_value = item.partition('=')
        if separator and unquote(key.strip()).lower() == expected_key:
            return True
    return False


def is_exporter_credential(name: str, value: Any) -> bool:
    normalized = name.upper()
    if normalized.startswith('OTEL_EXPORTER_OTLP_') and normalized.endswith('_HEADERS'):
        return True
    if normalized == 'OTEL_RESOURCE_ATTRIBUTES' and isinstance(value, str):
        return _has_resource_attribute(value, 'tps.tenant.id')
    if not (
        normalized.startswith('OTEL_EXPORTER_OTLP_') and normalized.endswith('_ENDPOINT') and isinstance(value, str)
    ):
        return False
    try:
        endpoint = urlsplit(value)
    except ValueError:
        return False
    return endpoint.username is not None or endpoint.password is not None


def redact_exporter_credentials(value: Any, replacement: str = '<redacted>') -> Any:
    if isinstance(value, Mapping):
        env_name = next(
            (item_value for item_key, item_value in value.items() if str(item_key).lower() == 'name'),
            None,
        )
        return {
            key: replacement
            if is_exporter_credential(str(key), item)
            or (str(key).lower() == 'value' and is_exporter_credential(str(env_name), item))
            else redact_exporter_credentials(item, replacement)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_exporter_credentials(item, replacement) for item in value]
    return value


def exporter_credential_values(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    values = [str(item) for key, item in value.items() if is_exporter_credential(str(key), item)]
    env_name = next(
        (item for key, item in value.items() if str(key).lower() == 'name'),
        None,
    )
    values.extend(
        str(item)
        for key, item in value.items()
        if str(key).lower() == 'value' and is_exporter_credential(str(env_name), item)
    )
    for item in value.values():
        if isinstance(item, Mapping):
            values.extend(exporter_credential_values(item))
        elif isinstance(item, list):
            for nested in item:
                values.extend(exporter_credential_values(nested))
    return list(dict.fromkeys(values))


def redact_sensitive_text(text: str, sensitive_values: list[str], replacement: str = '<redacted>') -> str:
    """Replace sensitive raw values and common POSIX shell renderings."""
    variants: set[str] = set()
    for value in sensitive_values:
        if not value:
            continue
        variants.add(value)
        quoted = shlex.quote(value)
        variants.add(quoted)
        if len(quoted) >= 2 and quoted.startswith("'") and quoted.endswith("'"):
            variants.add(quoted[1:-1])
        variants.add(value.replace("'", "'\\''"))
        variants.add(value.replace("'", "'\"'\"'"))
        variants.add(value.replace("'", "\\'").replace('"', '\\"'))

    redacted = str(text)
    for variant in sorted(variants, key=lambda item: len(item), reverse=True):
        redacted = redacted.replace(variant, replacement)
    return redacted


__all__ = [
    'exporter_credential_values',
    'is_exporter_credential',
    'redact_exporter_credentials',
    'redact_sensitive_text',
]
