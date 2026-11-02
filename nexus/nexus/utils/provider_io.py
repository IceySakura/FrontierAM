from __future__ import annotations

import re

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from nexus.utils.common import serialize_object


REDACTED = '<redacted>'
_INTERNAL_SENSITIVE_SOURCE_KEY = '_provider_sensitive_source'

_SENSITIVE_KEY_MARKERS = (
    'access_key',
    'api_key',
    'apikey',
    'app_key',
    'authorization',
    'client_secret',
    'cookie',
    'credential',
    'password',
    'secret',
)
_TOKEN_METRIC_MARKERS = (
    'audio',
    'cache',
    'cached',
    'completion',
    'input',
    'max',
    'min',
    'output',
    'prediction',
    'prompt',
    'reasoning',
    'total',
)
_URL_PATTERN = re.compile(r'(?i)https?://[^\s<>"\']+')
_URL_USERINFO_PATTERN = re.compile(r'(?i)(https?://)[^/@\s]+@')
_SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r'(?i)(\b(?:access[_-]?key|access[_-]?token|api[_-]?key|apikey|app[_-]?key|auth[_-]?token|'
    r'authorization|client[_-]?secret|cookie|credential|password|refresh[_-]?token|secret[_-]?key|'
    r'session[_-]?token|token)\b\s*[:=]\s*)(?:bearer\s+)?([^\s,;&]+)'
)


def _normalized_key(value: Any) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def _is_sensitive_key(value: Any, *, path: tuple[str, ...] = ()) -> bool:
    normalized = _normalized_key(value)
    if normalized == 'token' and any(item in {'logprobs', 'top_logprobs'} for item in path):
        return False
    if any(marker in normalized for marker in _SENSITIVE_KEY_MARKERS):
        return True
    if any(marker in normalized for marker in ('accesstoken', 'authtoken', 'refreshtoken', 'sessiontoken')):
        return True
    if normalized in {'token', 'tokens'} or normalized.endswith(('_token', '_tokens')):
        return not any(marker in normalized for marker in _TOKEN_METRIC_MARKERS)
    return False


def _credential_string(value: str) -> str:
    stripped = value.strip()
    scheme, separator, credential = stripped.partition(' ')
    if separator and scheme.lower() in {'basic', 'bearer', 'token'}:
        return credential
    return stripped


def _collect_sensitive_strings(value: Any, *, path: tuple[str, ...] = ()) -> set[str]:
    collected: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = _normalized_key(key)
            if _is_sensitive_key(key, path=path) and isinstance(item, str):
                credential = _credential_string(item)
                if credential:
                    collected.add(credential)
                    credential_base, _query_separator, _query = credential.partition('?')
                    if credential_base != credential:
                        collected.add(credential_base)
                    if ':' in credential_base:
                        collected.update(part for part in credential_base.split(':') if part)
                    for component in credential.split(';'):
                        _name, separator, nested_credential = component.strip().partition('=')
                        if separator and nested_credential:
                            collected.add(nested_credential)
            else:
                collected.update(_collect_sensitive_strings(item, path=(*path, normalized_key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            collected.update(_collect_sensitive_strings(item, path=path))
    return collected


def _sanitize_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        safe_netloc = parsed.netloc.rsplit('@', 1)[-1]
        query = urlencode(
            [
                (key, REDACTED if _is_sensitive_key(key) else value)
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            ],
            doseq=True,
            safe='<>',
        )
        return urlunsplit((parsed.scheme.lower(), safe_netloc, parsed.path, query, parsed.fragment))
    except ValueError:
        return url


def _sanitize_text(value: str, sensitive_values: set[str]) -> str:
    sanitized = value
    for secret in sorted(sensitive_values, key=lambda item: len(item), reverse=True):
        if not secret:
            continue
        if len(secret) >= 8:
            sanitized = sanitized.replace(secret, REDACTED)
        else:
            sanitized = re.sub(
                rf'(?<![A-Za-z0-9_]){re.escape(secret)}(?![A-Za-z0-9_])',
                REDACTED,
                sanitized,
            )
    sanitized = _URL_USERINFO_PATTERN.sub(r'\1', sanitized)
    sanitized = _URL_PATTERN.sub(lambda match: _sanitize_url(match.group(0)), sanitized)

    def redact_assignment(match: re.Match[str]) -> str:
        prefix = match.group(1)
        if _normalized_key(prefix).startswith('authorization') and 'bearer' in match.group(0).lower():
            return f'{prefix}Bearer {REDACTED}'
        return f'{prefix}{REDACTED}'

    return _SENSITIVE_ASSIGNMENT_PATTERN.sub(redact_assignment, sanitized)


def _redact_value(
    value: Any,
    sensitive_values: set[str],
    *,
    key: Any = None,
    path: tuple[str, ...] = (),
) -> Any:
    if key is not None and _is_sensitive_key(key, path=path):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact_value(
                item,
                sensitive_values,
                key=item_key,
                path=(*path, _normalized_key(item_key)),
            )
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, sensitive_values, path=path) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item, sensitive_values, path=path) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value, sensitive_values)
    return value


def redact_provider_io(value: Any, *, sensitive_source: Any = None) -> Any:
    """Return provider diagnostic data with credentials removed.

    Sensitive values are collected before key-based redaction so credentials echoed
    inside endpoint paths, provider errors, or response text are removed as well.
    """

    serialized = serialize_object(value)
    internal_sensitive_source = None
    if isinstance(serialized, dict):
        internal_sensitive_source = serialized.pop(_INTERNAL_SENSITIVE_SOURCE_KEY, None)
    sources = [serialized]
    if sensitive_source is not None:
        sources.append(serialize_object(sensitive_source))
    if internal_sensitive_source is not None:
        sources.append(internal_sensitive_source)
    sensitive_values: set[str] = set()
    for source in sources:
        sensitive_values.update(_collect_sensitive_strings(source))
    return _redact_value(serialized, sensitive_values)


__all__ = ['REDACTED', 'redact_provider_io']
