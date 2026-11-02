from __future__ import annotations

import base64
import json
import logging
import os
import re
import runpy
import sys
import types

from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import requests

from .constants import UNIVERSE_IMAGE_ROOT, WORKSPACE_MCP_ROOT, WORKSPACE_ROOT
from .mode import BenchmarkServerMode


LEGACY_KEYS_FILE = WORKSPACE_MCP_ROOT / 'keys.py'
LEGACY_KEYS_FALLBACK_FILE = WORKSPACE_ROOT / 'keys.py'
DEFAULT_CREDENTIALS_JSON = WORKSPACE_ROOT / 'mcp_benchmark_credentials.json'
DEFAULT_CREDENTIALS_JSON_FALLBACKS = (
    WORKSPACE_MCP_ROOT / 'credentials.json',
    WORKSPACE_ROOT / 'credentials.json',
)
DEFAULT_CREDENTIAL_BUNDLE_CANDIDATES = (
    WORKSPACE_MCP_ROOT / 'credential_fils.zip',
    WORKSPACE_ROOT / 'credential_fils.zip',
    WORKSPACE_MCP_ROOT / 'configs',
    WORKSPACE_ROOT / 'configs',
)

EXPLICIT_CREDENTIALS_JSON_PATH_ENV = (
    'NEXUS_MCP_BENCHMARK_CREDENTIALS_JSON_PATH',
    'MCP_BENCHMARK_CREDENTIALS_JSON_PATH',
)
EXPLICIT_CREDENTIALS_JSON_ENV = (
    'NEXUS_MCP_BENCHMARK_CREDENTIALS_JSON',
    'MCP_BENCHMARK_CREDENTIALS_JSON',
)
EXPLICIT_KEYS_FILE_ENV = (
    'NEXUS_MCP_BENCHMARK_CREDENTIALS_KEYS_PY',
    'MCP_BENCHMARK_CREDENTIALS_KEYS_PY',
)
EXPLICIT_CREDENTIAL_BUNDLE_ENV = (
    'NEXUS_MCP_BENCHMARK_CREDENTIAL_BUNDLE_PATH',
    'MCP_BENCHMARK_CREDENTIAL_BUNDLE_PATH',
)

GOOGLE_OAUTH_CREDENTIALS_FILE = 'google_oauth_credentials_file'
GCP_SERVICE_ACCOUNT_FILE = 'gcp_service_account_file'
SNOWFLAKE_PRIVATE_KEY_FILE = 'snowflake_private_key_file'
DEFAULT_EMAIL_CONFIG_FILE = 'default_email_config_file'
DEFAULT_KUBECONFIG_FILE = 'default_kubeconfig_file'
NOTION_MCP_AUTH_DIR = 'notion_mcp_auth_dir'

_FILE_KEY_ALIASES = {
    GOOGLE_OAUTH_CREDENTIALS_FILE: (
        GOOGLE_OAUTH_CREDENTIALS_FILE,
        'google_credentials_file',
        'google_oauth2_credentials_file',
        'google_oauth_credentials_json',
    ),
    GCP_SERVICE_ACCOUNT_FILE: (
        GCP_SERVICE_ACCOUNT_FILE,
        'gcp_service_account_path',
        'gcp_service_account_json',
        'google_service_account_file',
    ),
    SNOWFLAKE_PRIVATE_KEY_FILE: (
        SNOWFLAKE_PRIVATE_KEY_FILE,
        'snowflake_private_key_path',
        'snowflake_private_key',
    ),
    DEFAULT_EMAIL_CONFIG_FILE: (
        DEFAULT_EMAIL_CONFIG_FILE,
        'example_email_config_file',
        'emails_default_config_file',
    ),
    DEFAULT_KUBECONFIG_FILE: (
        DEFAULT_KUBECONFIG_FILE,
        'kubeconfig_file',
        'cluster_kubeconfig_file',
    ),
    NOTION_MCP_AUTH_DIR: (
        NOTION_MCP_AUTH_DIR,
        'mcp_auth_dir',
        'notion_mcp_auth',
    ),
}

_BUNDLE_FILE_CANDIDATES = {
    GOOGLE_OAUTH_CREDENTIALS_FILE: ('google_credentials.json', 'credentials.json', 'gcp-oauth.keys.json'),
    GCP_SERVICE_ACCOUNT_FILE: ('gcp-service_account.keys.json',),
    SNOWFLAKE_PRIVATE_KEY_FILE: ('snowflake_rsa_key.p8',),
    DEFAULT_EMAIL_CONFIG_FILE: ('example_email_config.json',),
    DEFAULT_KUBECONFIG_FILE: ('cluster-inst-alpha1-config.yaml', 'cluster1-config.yaml'),
}

_DIRECT_FILE_ENV_KEYS = {
    GOOGLE_OAUTH_CREDENTIALS_FILE: (
        'NEXUS_MCP_BENCHMARK_GOOGLE_CREDENTIALS_FILE',
        'MCP_BENCHMARK_GOOGLE_CREDENTIALS_FILE',
    ),
    GCP_SERVICE_ACCOUNT_FILE: (
        'NEXUS_MCP_BENCHMARK_GCP_SERVICE_ACCOUNT_FILE',
        'MCP_BENCHMARK_GCP_SERVICE_ACCOUNT_FILE',
    ),
    SNOWFLAKE_PRIVATE_KEY_FILE: (
        'NEXUS_MCP_BENCHMARK_SNOWFLAKE_PRIVATE_KEY_FILE',
        'MCP_BENCHMARK_SNOWFLAKE_PRIVATE_KEY_FILE',
    ),
    DEFAULT_EMAIL_CONFIG_FILE: (
        'NEXUS_MCP_BENCHMARK_EMAIL_CONFIG_FILE',
        'MCP_BENCHMARK_EMAIL_CONFIG_FILE',
    ),
    DEFAULT_KUBECONFIG_FILE: (
        'NEXUS_MCP_BENCHMARK_KUBECONFIG_FILE',
        'MCP_BENCHMARK_KUBECONFIG_FILE',
    ),
    NOTION_MCP_AUTH_DIR: (
        'NEXUS_MCP_BENCHMARK_MCP_AUTH_DIR',
        'MCP_BENCHMARK_MCP_AUTH_DIR',
    ),
}

_TOOLATHLON_DEFAULT_FILE_TARGETS = {
    'emails_config_file': 'configs/example_email_config.json',
    'google_oauth2_credentials_path': 'configs/google_credentials.json',
    'google_oauth2_token_path': 'configs/google_credentials.json',
    'gcp_service_account_path': 'configs/gcp-service_account.keys.json',
    'kubeconfig_path': 'configs/cluster-inst-alpha1-config.yaml',
    'snowflake_private_key_path': 'configs/snowflake_rsa_key.p8',
}
_TOOLATHLON_LITERAL_NULL_KEYS = {
    'github_allowed_repos',
    'google_cloud_allowed_buckets',
    'google_cloud_allowed_bigquery_datasets',
    'google_cloud_allowed_log_buckets',
    'google_cloud_allowed_instances',
    'snowflake_op_allowed_databases',
}
_MISSING_SENTINELS = {'XX', 'NONE', 'NULL', 'N/A'}
_NOTION_PAGE_ID_RE = re.compile(r'([0-9a-fA-F]{32})')


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EmbeddedFile:
    relative_path: str
    content: bytes


def _normalize_secret(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    if not text:
        return None
    if text.upper() in _MISSING_SENTINELS:
        return None
    if 'TO BE FILLED' in text.upper():
        return None
    return text


class _CompatDict(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def _ensure_addict_compat() -> None:
    if 'addict' in sys.modules:
        return
    module = types.ModuleType('addict')
    module.__dict__['Dict'] = _CompatDict
    sys.modules['addict'] = module


def _fallback_namespace_tokens(namespace: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in namespace.items():
        if key.startswith('_') or key == 'all_token_key_session':
            continue
        if isinstance(value, types.ModuleType):
            continue
        if callable(value):
            continue
        values[key] = value
    return values


def _namespace_tokens(namespace: dict[str, Any]) -> dict[str, Any]:
    fallback = _fallback_namespace_tokens(namespace)
    raw = namespace.get('all_token_key_session')
    if raw is None:
        return fallback
    try:
        materialized = dict(raw)
    except Exception:
        return raw
    if not materialized:
        return fallback
    merged = dict(fallback)
    merged.update(materialized)
    return merged


def _load_runpy_tokens(path: Path) -> dict[str, Any]:
    old_cwd = os.getcwd()
    try:
        os.chdir(path.parent)
        _ensure_addict_compat()
        namespace = runpy.run_path(str(path.name))
    finally:
        os.chdir(old_cwd)
    return _namespace_tokens(namespace)


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _path_from_env(*names: str) -> Path | None:
    raw = _env_first(*names)
    if not raw:
        return None
    return Path(raw).expanduser()


def _candidate_legacy_keys_paths() -> list[Path]:
    candidates: list[Path] = []
    if explicit := _path_from_env(*EXPLICIT_KEYS_FILE_ENV):
        candidates.append(explicit)
    candidates.extend((LEGACY_KEYS_FILE, LEGACY_KEYS_FALLBACK_FILE))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _candidate_credentials_json_paths() -> list[Path]:
    candidates: list[Path] = [DEFAULT_CREDENTIALS_JSON]
    candidates.extend(DEFAULT_CREDENTIALS_JSON_FALLBACKS)
    if explicit := _path_from_env(*EXPLICIT_CREDENTIALS_JSON_PATH_ENV):
        candidates.insert(0, explicit)
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _candidate_bundle_paths() -> list[Path]:
    candidates: list[Path] = []
    if explicit := _path_from_env(*EXPLICIT_CREDENTIAL_BUNDLE_ENV):
        candidates.append(explicit)
    candidates.extend(DEFAULT_CREDENTIAL_BUNDLE_CANDIDATES)
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _normalize_file_ref_key(name: str) -> str:
    lowered = name.strip().lower()
    for canonical, aliases in _FILE_KEY_ALIASES.items():
        if lowered in {alias.lower() for alias in aliases}:
            return canonical
    return name.strip()


def _parse_credentials_payload(
    payload: dict[str, Any],
    *,
    base_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, tuple[Path | None, Any]]]:
    values: dict[str, Any] = {}
    file_refs: dict[str, tuple[Path | None, Any]] = {}

    for group_key in ('values',):
        group = payload.get(group_key)
        if isinstance(group, dict):
            values.update(group)

    for group_key in ('files', 'paths', '__files__'):
        group = payload.get(group_key)
        if isinstance(group, dict):
            for key, value in group.items():
                file_refs[_normalize_file_ref_key(key)] = (base_dir, value)

    for key, value in payload.items():
        if key in {'values', 'files', 'paths', '__files__', '__meta__'}:
            continue
        if isinstance(value, dict) and set(value) & {'path', 'content', 'content_base64', 'entries'}:
            file_refs[_normalize_file_ref_key(key)] = (base_dir, value)
            continue
        values[key] = value

    return values, file_refs


def _load_credentials_json_file(path: Path) -> tuple[dict[str, Any], dict[str, tuple[Path | None, Any]]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise TypeError(f'Credentials JSON must contain an object: {path}')
    return _parse_credentials_payload(payload, base_dir=path.parent)


def _load_credentials_json_blob() -> tuple[dict[str, Any], dict[str, tuple[Path | None, Any]]]:
    raw = _env_first(*EXPLICIT_CREDENTIALS_JSON_ENV)
    if not raw:
        return {}, {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError('Credentials JSON blob must contain an object')
    return _parse_credentials_payload(payload, base_dir=None)


def _read_directory_entries(path: Path) -> list[_EmbeddedFile]:
    entries: list[_EmbeddedFile] = []
    for child in sorted(path.rglob('*')):
        if not child.is_file():
            continue
        relative_path = child.relative_to(path).as_posix()
        entries.append(_EmbeddedFile(relative_path=relative_path, content=child.read_bytes()))
    return entries


def _read_path_entries(path: Path) -> list[_EmbeddedFile]:
    if path.is_dir():
        return _read_directory_entries(path)
    return [_EmbeddedFile(relative_path=path.name, content=path.read_bytes())]


def _coerce_ref_path(raw: str, *, base_dir: Path | None) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute() or base_dir is None:
        return path
    return (base_dir / path).resolve()


def _resolve_explicit_file_entries(
    logical_key: str,
    ref: Any,
    *,
    base_dir: Path | None,
) -> list[_EmbeddedFile]:
    if isinstance(ref, str):
        path = _coerce_ref_path(ref, base_dir=base_dir)
        return _read_path_entries(path)
    if not isinstance(ref, dict):
        raise TypeError(f'Unsupported file reference for {logical_key}: {type(ref).__name__}')

    if 'path' in ref:
        path = _coerce_ref_path(str(ref['path']), base_dir=base_dir)
        return _read_path_entries(path)
    if 'entries' in ref:
        entries = ref['entries']
        if not isinstance(entries, dict):
            raise TypeError(f'{logical_key}.entries must be an object')
        resolved: list[_EmbeddedFile] = []
        for relative_path, entry in sorted(entries.items()):
            if not isinstance(entry, dict):
                raise TypeError(f'{logical_key}.entries[{relative_path!r}] must be an object')
            content = entry.get('content')
            content_base64 = entry.get('content_base64')
            if content_base64 is not None:
                resolved.append(
                    _EmbeddedFile(relative_path=str(relative_path), content=base64.b64decode(content_base64))
                )
            elif content is not None:
                resolved.append(_EmbeddedFile(relative_path=str(relative_path), content=str(content).encode('utf-8')))
            else:
                raise ValueError(f'{logical_key}.entries[{relative_path!r}] must contain content or content_base64')
        return resolved
    if 'content_base64' in ref:
        relative_path = str(ref.get('relative_path') or ref.get('filename') or logical_key)
        return [_EmbeddedFile(relative_path=relative_path, content=base64.b64decode(ref['content_base64']))]
    if 'content' in ref:
        relative_path = str(ref.get('relative_path') or ref.get('filename') or logical_key)
        encoding = str(ref.get('encoding') or 'utf8').lower()
        payload = base64.b64decode(str(ref['content'])) if encoding == 'base64' else str(ref['content']).encode('utf-8')
        return [_EmbeddedFile(relative_path=relative_path, content=payload)]
    raise ValueError(f'Unsupported file reference for {logical_key}')


def _load_bundle_bytes(path: Path) -> dict[str, bytes]:
    if not path.exists():
        return {}

    if path.is_dir():
        payload: dict[str, bytes] = {}
        for child in sorted(path.rglob('*')):
            if not child.is_file():
                continue
            relative_name = child.relative_to(path).as_posix()
            if relative_name.startswith('__MACOSX/'):
                continue
            payload.setdefault(relative_name, child.read_bytes())
            payload.setdefault(child.name, child.read_bytes())
        return payload

    if path.suffix.lower() != '.zip':
        return {}

    import zipfile

    payload = {}
    with zipfile.ZipFile(path) as bundle:
        for entry in bundle.infolist():
            if entry.is_dir():
                continue
            if entry.filename.startswith('__MACOSX/'):
                continue
            data = bundle.read(entry.filename)
            payload.setdefault(entry.filename, data)
            payload.setdefault(Path(entry.filename).name, data)
    return payload


def _merge_preserving_existing(target: dict[str, bytes], extra: dict[str, bytes]) -> None:
    for key, value in extra.items():
        target.setdefault(key, value)


def _text_or_base64(content: bytes) -> tuple[str, str]:
    try:
        return 'utf8', content.decode('utf-8')
    except UnicodeDecodeError:
        return 'base64', base64.b64encode(content).decode('ascii')


def _json_from_bytes(content: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(content.decode('utf-8'))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


@cache
def load_toolathlon_task_overrides(task_dir: str) -> dict[str, Any]:
    path = Path(task_dir) / 'token_key_session.py'
    if not path.exists():
        return {}

    try:
        return _load_runpy_tokens(path)
    except Exception as exc:
        return {'__load_error__': str(exc)}


@cache
def _credential_store() -> tuple[dict[str, Any], dict[str, list[_EmbeddedFile]], dict[str, bytes]]:
    values: dict[str, Any] = {}
    explicit_file_refs: dict[str, tuple[Path | None, Any]] = {}
    explicit_legacy_paths: list[Path] = []
    if explicit := _path_from_env(*EXPLICIT_KEYS_FILE_ENV):
        explicit_legacy_paths.append(explicit)

    json_source_found = False

    for path in explicit_legacy_paths:
        if not path.exists():
            continue
        try:
            values.update(_load_runpy_tokens(path))
        except Exception as exc:
            logger.warning('Failed to load legacy credential file %s: %s', path, exc)

    for path in _candidate_credentials_json_paths():
        if not path.exists():
            continue
        json_source_found = True
        try:
            json_values, json_file_refs = _load_credentials_json_file(path)
        except Exception as exc:
            logger.warning('Failed to load credentials JSON %s: %s', path, exc)
            continue
        values.update(json_values)
        explicit_file_refs.update(json_file_refs)

    raw_json_blob = _env_first(*EXPLICIT_CREDENTIALS_JSON_ENV)
    if raw_json_blob:
        json_source_found = True
    try:
        blob_values, blob_file_refs = _load_credentials_json_blob()
    except Exception as exc:
        logger.warning('Failed to load credentials JSON blob: %s', exc)
        blob_values, blob_file_refs = {}, {}
    values.update(blob_values)
    explicit_file_refs.update(blob_file_refs)

    if not json_source_found:
        for path in (LEGACY_KEYS_FILE, LEGACY_KEYS_FALLBACK_FILE):
            if not path.exists():
                continue
            try:
                values.update(_load_runpy_tokens(path))
            except Exception as exc:
                logger.warning('Failed to load legacy credential file %s: %s', path, exc)

    for logical_key, env_names in _DIRECT_FILE_ENV_KEYS.items():
        if raw := _env_first(*env_names):
            explicit_file_refs[logical_key] = (None, raw)

    for key in list(values):
        env_value = _env_first(
            f'NEXUS_MCP_BENCHMARK_CREDENTIAL_{key.upper()}',
            f'MCP_BENCHMARK_CREDENTIAL_{key.upper()}',
        )
        if env_value is not None:
            values[key] = env_value

    resolved_files: dict[str, list[_EmbeddedFile]] = {}
    for logical_key, (base_dir, ref) in explicit_file_refs.items():
        try:
            resolved_files[logical_key] = _resolve_explicit_file_entries(logical_key, ref, base_dir=base_dir)
        except Exception as exc:
            logger.warning('Failed to resolve credential file %s: %s', logical_key, exc)

    bundle_files: dict[str, bytes] = {}
    for path in _candidate_bundle_paths():
        try:
            _merge_preserving_existing(bundle_files, _load_bundle_bytes(path))
        except Exception as exc:
            logger.warning('Failed to inspect credential bundle %s: %s', path, exc)

    return values, resolved_files, bundle_files


@lru_cache(maxsize=1)
def load_workspace_keys() -> dict[str, Any]:
    values, explicit_files, bundle_files = _credential_store()
    materialized = dict(values)

    if (
        not materialized.get('google_client_id')
        or not materialized.get('google_client_secret')
        or not materialized.get('google_refresh_token')
    ):
        google_entries = explicit_files.get(GOOGLE_OAUTH_CREDENTIALS_FILE)
        if not google_entries:
            google_entries = _bundle_entries_for_key(bundle_files, GOOGLE_OAUTH_CREDENTIALS_FILE)
        if google_entries:
            google_payload = _json_from_bytes(google_entries[0].content)
            if google_payload:
                for _gk, _jk in (
                    ('google_client_id', 'client_id'),
                    ('google_client_secret', 'client_secret'),
                    ('google_refresh_token', 'refresh_token'),
                ):
                    if not _normalize_secret(materialized.get(_gk)):
                        _gv = google_payload.get(_jk, '')
                        if _gv:
                            materialized[_gk] = _gv

    if not materialized.get('gcp_project_id'):
        gcp_entries = explicit_files.get(GCP_SERVICE_ACCOUNT_FILE)
        if not gcp_entries:
            gcp_entries = _bundle_entries_for_key(bundle_files, GCP_SERVICE_ACCOUNT_FILE)
        if gcp_entries:
            gcp_payload = _json_from_bytes(gcp_entries[0].content)
            if gcp_payload:
                materialized.setdefault('gcp_project_id', gcp_payload.get('project_id', ''))

    return materialized


def _bundle_entries_for_key(bundle_files: dict[str, bytes], logical_key: str) -> list[_EmbeddedFile]:
    if logical_key == NOTION_MCP_AUTH_DIR:
        prefixes = ('.mcp-auth/', 'mcp-auth/')
        entries: list[_EmbeddedFile] = []
        for name, content in bundle_files.items():
            if '/' not in name:
                continue
            matched_prefix = next((prefix for prefix in prefixes if name.startswith(prefix)), None)
            if matched_prefix is None:
                continue
            entries.append(_EmbeddedFile(relative_path=name[len(matched_prefix) :], content=content))
        return sorted(entries, key=lambda item: item.relative_path)

    entries: list[_EmbeddedFile] = []
    for candidate in _BUNDLE_FILE_CANDIDATES.get(logical_key, ()):
        if candidate in bundle_files:
            entries.append(_EmbeddedFile(relative_path=Path(candidate).name, content=bundle_files[candidate]))
            break
    return entries


@cache
def _github_account_name(token: str) -> str | None:
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'nexus-mcp-benchmark',
    }
    try:
        response = requests.get('https://api.github.com/user', headers=headers, timeout=30)
    except requests.RequestException as exc:
        logger.warning('Failed to resolve GitHub account name: %s', exc)
        return None
    if response.status_code != 200:
        logger.warning(
            'Failed to resolve GitHub account name: status=%s body=%s', response.status_code, response.text[:200]
        )
        return None
    return _normalize_secret(response.json().get('login'))


@cache
def _notion_token_valid(token: str) -> bool:
    headers = {
        'Authorization': f'Bearer {token}',
        'Notion-Version': '2022-06-28',
        'User-Agent': 'nexus-mcp-benchmark',
    }
    try:
        response = requests.get('https://api.notion.com/v1/users/me', headers=headers, timeout=30)
    except requests.RequestException as exc:
        logger.warning('Failed to validate Notion token: %s', exc)
        return False
    return response.status_code == 200


def _notion_page_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    compact = url.replace('-', '')
    match = _NOTION_PAGE_ID_RE.search(compact)
    if not match:
        return None
    return match.group(1)


class WorkspaceKeys:
    def __init__(
        self,
        raw: dict[str, Any] | None = None,
        *,
        explicit_files: dict[str, list[_EmbeddedFile]] | None = None,
        bundle_files: dict[str, bytes] | None = None,
    ) -> None:
        values, store_files, store_bundle_files = _credential_store()
        self._raw = dict(values if raw is None else raw)
        self._explicit_files = {
            key: list(entries) for key, entries in (store_files if explicit_files is None else explicit_files).items()
        }
        self._bundle_files = dict(store_bundle_files if bundle_files is None else bundle_files)

        # Backfill Google OAuth tokens from bundle credentials file when keys.py
        # doesn't provide them (e.g. google_credentials.json not adjacent to keys.py).
        if (
            not _normalize_secret(self._raw.get('google_client_id'))
            or not _normalize_secret(self._raw.get('google_client_secret'))
            or not _normalize_secret(self._raw.get('google_refresh_token'))
        ):
            google_entries = self._file_entries(GOOGLE_OAUTH_CREDENTIALS_FILE)
            if google_entries:
                google_payload = _json_from_bytes(google_entries[0].content)
                if google_payload:
                    for raw_key, json_key in (
                        ('google_client_id', 'client_id'),
                        ('google_client_secret', 'client_secret'),
                        ('google_refresh_token', 'refresh_token'),
                    ):
                        if not _normalize_secret(self._raw.get(raw_key)):
                            val = google_payload.get(json_key, '')
                            if val:
                                self._raw[raw_key] = val

    def get(self, name: str) -> str | None:
        return _normalize_secret(self._raw.get(name))

    def get_any(self, *names: str) -> str | None:
        for name in names:
            value = self.get(name)
            if value is not None:
                return value
        return None

    def _file_entries(self, logical_key: str) -> list[_EmbeddedFile]:
        if logical_key in self._explicit_files:
            return list(self._explicit_files[logical_key])
        return _bundle_entries_for_key(self._bundle_files, logical_key)

    def _has_file_entries(self, logical_key: str) -> bool:
        return bool(self._file_entries(logical_key))

    def merged_toolathlon_raw(
        self,
        task_dir: Path | None = None,
        *,
        case_root: str,
        task_override_values: dict[str, Any] | None = None,
        task_source_path: str | None = None,
        embedded_case_paths: set[str] | None = None,
    ) -> dict[str, str]:
        merged = dict(self._raw)
        if task_override_values is None:
            if task_dir is None:
                task_override_values = {}
            else:
                task_override_values = {
                    k: v for k, v in load_toolathlon_task_overrides(str(task_dir)).items() if not k.startswith('__')
                }
        merged.update({str(key): value for key, value in task_override_values.items() if not str(key).startswith('__')})

        host_task_dir = task_dir.resolve() if task_dir is not None else None
        host_task_dir_hint = task_source_path or (host_task_dir.as_posix() if host_task_dir is not None else None)
        normalized_embedded_case_paths = {PurePosixPath(path).as_posix() for path in (embedded_case_paths or set())}
        rendered: dict[str, str] = {}
        for key, value in merged.items():
            normalized = _normalize_secret(value)
            if normalized is None and key in _TOOLATHLON_LITERAL_NULL_KEYS and str(value).strip().lower() == 'null':
                normalized = 'null'
            if normalized is None:
                continue
            rendered[key] = self._rewrite_task_value(
                key,
                normalized,
                host_task_dir=host_task_dir,
                host_task_dir_hint=host_task_dir_hint,
                case_root=case_root,
                embedded_case_paths=normalized_embedded_case_paths,
            )
        return rendered

    def toolathlon_task_override_error(self, task_dir: Path) -> str | None:
        raw = load_toolathlon_task_overrides(str(task_dir))
        error = raw.get('__load_error__')
        return str(error) if error else None

    @staticmethod
    def _rewrite_task_value(
        key: str,
        value: str,
        *,
        host_task_dir: Path | None,
        host_task_dir_hint: str | None,
        case_root: str,
        embedded_case_paths: set[str] | None = None,
    ) -> str:
        case_prefix = PurePosixPath(case_root).as_posix()
        normalized = value.replace('\\', '/')
        host_prefix = str(host_task_dir_hint or '').replace('\\', '/').rstrip('/')
        if host_prefix and normalized.startswith(host_prefix):
            suffix = normalized[len(host_prefix) :]
            return case_prefix + suffix

        relative_candidate = PurePosixPath(normalized).as_posix()
        if relative_candidate.startswith('./'):
            relative_candidate = relative_candidate[2:]

        if host_task_dir is not None:
            candidate = (host_task_dir / value).resolve()
            if candidate.exists():
                try:
                    relative = candidate.relative_to(host_task_dir)
                except ValueError:
                    return normalized
                return str(PurePosixPath(case_root) / PurePosixPath(relative.as_posix()))

        if embedded_case_paths and relative_candidate in embedded_case_paths:
            return str(PurePosixPath(case_root) / PurePosixPath(relative_candidate))

        default_rel = _TOOLATHLON_DEFAULT_FILE_TARGETS.get(key)
        if default_rel is not None:
            return str(PurePosixPath(case_root) / PurePosixPath(default_rel))

        return normalized

    def toolathlon_template_values(
        self,
        *,
        task_dir: Path | None = None,
        case_root: str,
        agent_workspace: str,
        server_mode: BenchmarkServerMode = BenchmarkServerMode.API,
        task_override_values: dict[str, Any] | None = None,
        task_source_path: str | None = None,
        embedded_case_paths: set[str] | None = None,
    ) -> dict[str, str]:
        values: dict[str, str] = {
            'agent_workspace': agent_workspace,
            'local_servers_paths': '/opt/mcp-benchmarks/local_servers',
            'local_binary_paths': '/opt/mcp-benchmarks/local_bin',
            'token.canvas_domain': self.get('canvas_domain') or str(PurePosixPath(case_root) / 'canvas-domain'),
            'token.timezone': self.get('timezone') or 'UTC',
            'token.emails_config_file': str(PurePosixPath(case_root) / 'configs' / 'example_email_config.json'),
            'token.google_oauth2_credentials_path': str(
                PurePosixPath(case_root) / 'configs' / 'google_credentials.json'
            ),
            'token.google_oauth2_token_path': str(PurePosixPath(case_root) / 'configs' / 'google_credentials.json'),
            'token.gcp_service_account_path': str(
                PurePosixPath(case_root) / 'configs' / 'gcp-service_account.keys.json'
            ),
            'token.kubeconfig_path': str(PurePosixPath(case_root) / 'configs' / 'cluster-inst-alpha1-config.yaml'),
            'token.snowflake_private_key_path': str(PurePosixPath(case_root) / 'configs' / 'snowflake_rsa_key.p8'),
        }

        for key, value in self.merged_toolathlon_raw(
            task_dir,
            case_root=case_root,
            task_override_values=task_override_values,
            task_source_path=task_source_path,
            embedded_case_paths=embedded_case_paths,
        ).items():
            values[f'token.{key}'] = value

        if (snowflake_key := values.get('token.snowflake_private_key_path')) and not Path(snowflake_key).is_absolute():
            values['token.snowflake_private_key_path'] = str(PurePosixPath(case_root) / snowflake_key)

        return values

    def toolathlon_input_files(self, *, case_root: str) -> list[dict[str, str]]:
        files: list[dict[str, str]] = []

        def _append(target_path: str, entries: list[_EmbeddedFile]) -> None:
            if not entries:
                return
            encoding, content = _text_or_base64(entries[0].content)
            files.append({'path': target_path, 'encoding': encoding, 'content': content})

        case_root_path = PurePosixPath(case_root)
        _append(
            str(case_root_path / 'configs' / 'google_credentials.json'),
            self._file_entries(GOOGLE_OAUTH_CREDENTIALS_FILE),
        )
        _append(
            str(case_root_path / 'configs' / 'credentials.json'),
            self._file_entries(GOOGLE_OAUTH_CREDENTIALS_FILE),
        )
        _append(
            str(case_root_path / 'configs' / 'gcp-service_account.keys.json'),
            self._file_entries(GCP_SERVICE_ACCOUNT_FILE),
        )
        _append(
            str(case_root_path / 'configs' / 'snowflake_rsa_key.p8'),
            self._file_entries(SNOWFLAKE_PRIVATE_KEY_FILE),
        )
        _append(
            str(case_root_path / 'configs' / 'example_email_config.json'),
            self._file_entries(DEFAULT_EMAIL_CONFIG_FILE),
        )
        _append(
            str(case_root_path / 'configs' / 'cluster-inst-alpha1-config.yaml'),
            self._file_entries(DEFAULT_KUBECONFIG_FILE),
        )

        for entry in self._file_entries(NOTION_MCP_AUTH_DIR):
            encoding, content = _text_or_base64(entry.content)
            files.append({
                'path': str(case_root_path / 'configs' / '.mcp-auth' / entry.relative_path),
                'encoding': encoding,
                'content': content,
            })

        return files

    def atlas_env_values(self) -> dict[str, str]:
        mapping = {
            'AIRTABLE_API_KEY': self.get_any('airtable_api_key'),
            'ALCHEMY_API_KEY': self.get_any('alchemy_api_key'),
            'BRAVE_API_KEY': self.get_any('brave_api_key', 'brave_search_api_key'),
            'E2B_API_KEY': self.get_any('e2b_api_key'),
            'EXA_API_KEY': self.get_any('exa_api_key'),
            'GITHUB_TOKEN': self.get_any('github_token'),
            'GOOGLE_MAPS_API_KEY': self.get_any('google_cloud_console_api_key', 'google_maps_api_key'),
            'GOOGLE_CLIENT_ID': self.get_any('google_client_id'),
            'GOOGLE_CLIENT_SECRET': self.get_any('google_client_secret'),
            'GOOGLE_REFRESH_TOKEN': self.get_any('google_refresh_token'),
            'LARA_ACCESS_KEY_ID': self.get_any('lara_access_key_id'),
            'LARA_ACCESS_KEY_SECRET': self.get_any('lara_access_key_secret'),
            'MONGODB_CONNECTION_STRING': self.get_any('mongodb_connection_string', 'mongo_connection_string'),
            'NPS_API_KEY': self.get_any('nps_api_key', 'national_parks_api_key'),
            'NOTION_TOKEN': self.get_any('notion_integration_key_eval', 'notion_integration_key'),
            'OXYLABS_USERNAME': self.get_any('oxylabs_username'),
            'OXYLABS_PASSWORD': self.get_any('oxylabs_password'),
            'SLACK_MCP_XOXC_TOKEN': self.get_any('slack_mcp_xoxc_token'),
            'SLACK_MCP_XOXD_TOKEN': self.get_any('slack_mcp_xoxd_token'),
            'TWELVE_DATA_API_KEY': self.get_any('twelve_data_api_key'),
            'WEATHER_API_KEY': self.get_any('weather_api_key', 'weatherapi_api_key'),
        }
        return {key: value for key, value in mapping.items() if value is not None}

    def github_account_name(self) -> str | None:
        configured = (
            self.get('github_personal_account_name')
            or self.get('github_account_name')
            or _normalize_secret(os.getenv('GITHUB_PERSONAL_ACCOUNT_NAME'))
        )
        if configured:
            return configured
        token = self.get('github_token')
        if not token:
            return None
        return _github_account_name(token)

    def notion_root_page(self) -> str | None:
        return _notion_page_id_from_url(self.get('eval_notion_page_url') or self.get('source_notion_page_url'))

    def has_valid_notion_token(self) -> bool:
        token = self.get('notion_integration_key')
        if not token:
            return False
        return _notion_token_valid(token)

    def has_valid_notion_eval_token(self) -> bool:
        token = self.get('notion_integration_key_eval')
        if not token:
            return False
        return _notion_token_valid(token)

    def has_any_valid_notion_token(self) -> bool:
        return self.has_valid_notion_token() or self.has_valid_notion_eval_token()

    def has_google_oauth_credentials_file(self) -> bool:
        return self._has_file_entries(GOOGLE_OAUTH_CREDENTIALS_FILE)

    def has_gcp_service_account_file(self) -> bool:
        return self._has_file_entries(GCP_SERVICE_ACCOUNT_FILE)

    def has_snowflake_private_key_file(self) -> bool:
        return self._has_file_entries(SNOWFLAKE_PRIVATE_KEY_FILE)

    def has_mcp_auth_dir(self) -> bool:
        return self._has_file_entries(NOTION_MCP_AUTH_DIR)

    def universe_input_files(self, *, case_root: str) -> list[dict[str, str]]:
        entries = self._file_entries(GCP_SERVICE_ACCOUNT_FILE)
        if not entries:
            return []
        encoding, content = _text_or_base64(entries[0].content)
        return [
            {
                'path': str(PurePosixPath(case_root) / 'configs' / 'gcp-service_account.keys.json'),
                'encoding': encoding,
                'content': content,
            }
        ]

    def universe_env_values(self, *, case_root: str) -> dict[str, str]:
        mapping = {
            'GITHUB_PERSONAL_ACCESS_TOKEN': self.get_any('github_token'),
            'GITHUB_TOKENS': self.get_any('github_token'),
            'GITHUB_PERSONAL_ACCOUNT_NAME': self.github_account_name(),
            'NOTION_API_KEY': self.get_any('notion_integration_key_eval', 'notion_integration_key'),
            'EVAL_NOTION_API_KEY': self.get_any('notion_integration_key_eval', 'notion_integration_key'),
            'NOTION_ROOT_PAGE': self.notion_root_page(),
            'SERP_API_KEY': self.get_any('serper_api_key'),
            'SERPER_API_KEY': self.get_any('serper_api_key'),
            'GOOGLE_MAPS_API_KEY': self.get_any('google_cloud_console_api_key', 'google_maps_api_key'),
            'GOOGLE_SERVICE_ACCOUNT_PATH': str(PurePosixPath(case_root) / 'configs' / 'gcp-service_account.keys.json'),
            'GOOGLE_DRIVE_FOLDER_ID': self.get_any('google_sheets_folder_id'),
            'FILESYSTEM_DIRECTORY': f'{case_root}/filesystem',
            'FILESYSTEM_TEST_DIR': f'{case_root}/filesystem',
            'MCPUniverse_DIR': UNIVERSE_IMAGE_ROOT,
            'SERPER_BASE_URL': 'https://google.serper.dev/search',
        }
        if not self._has_file_entries(GCP_SERVICE_ACCOUNT_FILE):
            mapping.pop('GOOGLE_SERVICE_ACCOUNT_PATH', None)
        return {key: value for key, value in mapping.items() if value is not None}


__all__ = ['WorkspaceKeys', 'load_workspace_keys', 'load_toolathlon_task_overrides']
