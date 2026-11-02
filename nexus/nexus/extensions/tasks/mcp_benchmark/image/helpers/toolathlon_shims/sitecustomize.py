# NOTE: Do NOT use ``from __future__ import annotations`` here.
# In Python 3.12/3.13 it triggers ``import importlib.abc`` internally, but
# importlib is not fully bootstrapped yet when sitecustomize runs, causing
# ``AttributeError: module 'importlib' has no attribute 'abc'`` which makes
# Python skip this entire file.

import contextlib
import json
import os
import re
import shlex
import subprocess

from datetime import datetime
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# SSL CA bundle fix: force httplib2 (used by Google API client) to use the
# system CA bundle which includes the Gongfeng Squid Proxy CA certificate.
# Without this, httplib2 uses certifi's or its own CA bundle that lacks the
# corporate proxy CA, causing CERTIFICATE_VERIFY_FAILED on Google API calls.
# ---------------------------------------------------------------------------
_SYSTEM_CA_BUNDLE = '/etc/ssl/certs/ca-certificates.crt'
if os.path.exists(_SYSTEM_CA_BUNDLE):
    os.environ.setdefault('SSL_CERT_FILE', _SYSTEM_CA_BUNDLE)
    os.environ.setdefault('REQUESTS_CA_BUNDLE', _SYSTEM_CA_BUNDLE)

    try:
        import httplib2

        httplib2.CA_CERTS = _SYSTEM_CA_BUNDLE
    except Exception:
        pass


def _git_mirror_push_local_dir(command: str) -> str | None:
    try:
        tokens = shlex.split(str(command))
    except Exception:
        return None
    if len(tokens) < 5 or tokens[0] != 'git':
        return None
    try:
        c_index = tokens.index('-C')
        push_index = tokens.index('push')
    except ValueError:
        return None
    if c_index + 1 >= len(tokens) or push_index <= c_index + 1:
        return None
    if '--mirror' not in tokens[push_index + 1 :]:
        return None
    return tokens[c_index + 1]


def _prune_git_mirror_special_refs(local_dir: str) -> None:
    result = subprocess.run(
        ['git', '-C', local_dir, 'for-each-ref', '--format=%(refname)', 'refs/pull'],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return
    for ref in (line.strip() for line in result.stdout.splitlines()):
        if not ref.startswith('refs/pull/'):
            continue
        subprocess.run(
            ['git', '-C', local_dir, 'update-ref', '-d', ref],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


# ---------------------------------------------------------------------------
# gspread fix: ensure AuthorizedSession (used by gspread v6+ for Sheets API
# data I/O) trusts the system CA bundle which includes the Gongfeng Squid Proxy
# CA certificate.  gspread creates its AuthorizedSession before env-var-based
# requests configuration can take effect, so we patch the constructor.
# ---------------------------------------------------------------------------
try:
    from google.auth.transport.requests import AuthorizedSession as _AuthorizedSession

    _real_auth_session_init = _AuthorizedSession.__init__

    def _patched_auth_session_init(self, credentials, *args, **kwargs):
        _real_auth_session_init(self, credentials, *args, **kwargs)
        if os.path.exists(_SYSTEM_CA_BUNDLE):
            self.verify = _SYSTEM_CA_BUNDLE

    _AuthorizedSession.__init__ = _patched_auth_session_init
except Exception:
    pass


# ---------------------------------------------------------------------------
# Google Cloud Logging ConfigServiceV2Client REST shim: the Toolathlon worker
# venv uses google-cloud-logging 3.15.0, whose ConfigServiceV2Client only
# registers gRPC transports. On Gongfeng Squid MITM nodes this fails with
# "missing selected ALPN property". Patch the bucket methods we hit in
# preprocess/eval to use the public REST API through AuthorizedSession.
# ---------------------------------------------------------------------------
try:
    from google.api_core import exceptions as _google_api_exceptions
    from google.auth import credentials as _google_auth_credentials
    from google.auth.transport.requests import AuthorizedSession as _GoogleAuthorizedSession
    from google.cloud.logging_v2.services.config_service_v2 import client as _logging_config_client
    from google.cloud.logging_v2.types import logging_config as _logging_config
    from google.protobuf import json_format as _json_format

    def _gcp_timeout_value(timeout, default=120.0):
        if isinstance(timeout, (int, float)) and timeout > 0:
            return timeout
        return default

    def _gcp_logging_transport(config_client):
        return getattr(config_client, 'transport', None) or getattr(config_client, '_transport', None)

    def _gcp_logging_base_url(config_client):
        host = getattr(_gcp_logging_transport(config_client), '_host', 'logging.googleapis.com')
        host = str(host).removeprefix('https://').removeprefix('http://').rstrip('/')
        return f'https://{host}/v2'

    def _gcp_logging_session(config_client):
        transport = _gcp_logging_transport(config_client)
        credentials = getattr(transport, '_credentials', None)
        if credentials is None:
            raise RuntimeError('ConfigServiceV2Client transport credentials are missing')
        scopes = getattr(transport, 'AUTH_SCOPES', None) or getattr(type(transport), 'AUTH_SCOPES', None)
        credentials = _google_auth_credentials.with_scopes_if_required(credentials, scopes)
        session = _GoogleAuthorizedSession(credentials, default_host=getattr(transport, 'DEFAULT_HOST', None))
        if os.path.exists(_SYSTEM_CA_BUNDLE):
            session.verify = _SYSTEM_CA_BUNDLE
        return session

    def _gcp_logging_request(config_client, method, path, *, params=None, json_payload=None, timeout=None):
        session = _gcp_logging_session(config_client)
        try:
            response = session.request(
                method,
                f'{_gcp_logging_base_url(config_client)}/{path.lstrip("/")}',
                params=params,
                json=json_payload,
                timeout=_gcp_timeout_value(timeout),
            )
        finally:
            with contextlib.suppress(Exception):
                session.close()
        if response.status_code >= 400:
            raise _google_api_exceptions.from_http_response(response)
        if not response.content:
            return {}
        return response.json()

    def _gcp_logging_bucket_payload(bucket):
        if bucket is None:
            return {}
        payload = (
            _json_format.MessageToDict(bucket._pb, preserving_proto_field_name=True)
            if hasattr(bucket, '_pb')
            else dict(bucket)
        )
        return {
            key: value
            for key, value in payload.items()
            if value is not None and value != '' and value != [] and value != {}
        }

    def _gcp_log_bucket_from_rest(payload):
        if not payload:
            return _logging_config.LogBucket()
        message = _json_format.ParseDict(payload, _logging_config.LogBucket.pb(_logging_config.LogBucket()))
        return _logging_config.LogBucket(message)

    def _list_buckets_via_rest(self, request=None, *, parent=None, retry=None, timeout=None, metadata=()):
        del retry, metadata
        has_parent = parent is not None
        if request is not None and has_parent:
            raise ValueError(
                'If the `request` argument is set, then none of the individual field arguments should be set.'
            )
        if not isinstance(request, _logging_config.ListBucketsRequest):
            request = _logging_config.ListBucketsRequest(request or {})
        if parent is not None:
            request.parent = parent
        params = {}
        if request.page_size:
            params['pageSize'] = int(request.page_size)
        next_page_token = request.page_token or ''
        buckets = []
        while True:
            if next_page_token:
                params['pageToken'] = next_page_token
            elif 'pageToken' in params:
                params.pop('pageToken')
            response = _gcp_logging_request(
                self,
                'GET',
                f'{request.parent}/buckets',
                params=params,
                timeout=timeout,
            )
            buckets.extend(_gcp_log_bucket_from_rest(item) for item in response.get('buckets', []))
            next_page_token = response.get('nextPageToken', '')
            if not next_page_token:
                return buckets

    def _get_bucket_via_rest(self, request=None, *, retry=None, timeout=None, metadata=()):
        del retry, metadata
        if not isinstance(request, _logging_config.GetBucketRequest):
            request = _logging_config.GetBucketRequest(request or {})
        response = _gcp_logging_request(self, 'GET', request.name, timeout=timeout)
        return _gcp_log_bucket_from_rest(response)

    def _create_bucket_via_rest(self, request=None, *, retry=None, timeout=None, metadata=()):
        del retry, metadata
        if not isinstance(request, _logging_config.CreateBucketRequest):
            request = _logging_config.CreateBucketRequest(request or {})
        response = _gcp_logging_request(
            self,
            'POST',
            f'{request.parent}/buckets',
            params={'bucketId': request.bucket_id},
            json_payload=_gcp_logging_bucket_payload(request.bucket),
            timeout=timeout,
        )
        return _gcp_log_bucket_from_rest(response)

    def _delete_bucket_via_rest(self, request=None, *, retry=None, timeout=None, metadata=()):
        del retry, metadata
        if not isinstance(request, _logging_config.DeleteBucketRequest):
            request = _logging_config.DeleteBucketRequest(request or {})
        _gcp_logging_request(self, 'DELETE', request.name, timeout=timeout)

    _config_service_cls = _logging_config_client.ConfigServiceV2Client
    if not getattr(_config_service_cls, '_nexus_bucket_rest_patch', False):
        _config_service_cls.list_buckets = _list_buckets_via_rest
        _config_service_cls.get_bucket = _get_bucket_via_rest
        _config_service_cls.create_bucket = _create_bucket_via_rest
        _config_service_cls.delete_bucket = _delete_bucket_via_rest
        _config_service_cls._nexus_bucket_rest_patch = True
except Exception:
    pass


if os.getenv('NEXUS_TOOLATHLON_SERVER_MODE') == 'mock':
    try:
        from googleapiclient import discovery as _discovery
    except Exception:
        _discovery = None

    if _discovery is not None:
        from toolathlon_google_workspace_mock import DriveService, FormsService, SheetsService

        _real_build = _discovery.build

        def _mock_build(service_name, version, *args, **kwargs):
            if service_name == 'drive' and str(version).startswith('v3'):
                return DriveService()
            if service_name == 'forms' and str(version).startswith('v1'):
                return FormsService()
            if service_name == 'sheets' and str(version).startswith('v4'):
                return SheetsService()
            return _real_build(service_name, version, *args, **kwargs)

        _discovery.build = _mock_build

    try:
        from google.oauth2 import credentials as _google_credentials
    except Exception:
        _google_credentials = None

    if _google_credentials is not None:
        _real_refresh = _google_credentials.Credentials.refresh

        def _mock_refresh(self, request):
            del request
            if not getattr(self, 'token', None):
                self.token = 'mock-google-token'
            return

        _google_credentials.Credentials.refresh = _mock_refresh

    try:
        import requests as _requests
    except Exception:
        _requests = None

    if _requests is not None:
        try:
            from toolathlon_notion_store import NotionStoreError, handle_notion_http_request
        except Exception:
            NotionStoreError = None
            handle_notion_http_request = None

        if NotionStoreError is not None and handle_notion_http_request is not None:
            _real_request = _requests.sessions.Session.request

            def _decode_request_payload(kwargs):
                if kwargs.get('json') is not None:
                    return kwargs.get('json')
                data = kwargs.get('data')
                if data is None:
                    return None
                if isinstance(data, bytes):
                    data = data.decode('utf-8', errors='ignore')
                if isinstance(data, str):
                    raw = data.strip()
                    if raw and raw[0] in '{[':
                        try:
                            return json.loads(raw)
                        except json.JSONDecodeError:
                            return None
                return None

            def _mock_request(self, method, url, **kwargs):
                parsed = urlparse(str(url))
                if parsed.netloc == 'api.notion.com' and parsed.path.startswith('/v1/'):
                    try:
                        payload = handle_notion_http_request(
                            method,
                            str(url),
                            json_payload=_decode_request_payload(kwargs),
                            params=kwargs.get('params'),
                        )
                        status_code = 200
                    except NotionStoreError as exc:
                        payload = {
                            'object': 'error',
                            'status': exc.status_code,
                            'code': 'notion_mock_error',
                            'message': exc.message,
                        }
                        status_code = exc.status_code
                    response = _requests.Response()
                    response.status_code = status_code
                    response.encoding = 'utf-8'
                    response.url = str(url)
                    response._content = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                    response.headers['Content-Type'] = 'application/json; charset=utf-8'
                    response.request = _requests.Request(method=method.upper(), url=str(url)).prepare()
                    return response
                return _real_request(self, method, url, **kwargs)

            _requests.sessions.Session.request = _mock_request

    try:
        import toolathlon_snowflake_mock as _snowflake_mock
    except Exception:
        _snowflake_mock = None

    if _snowflake_mock is not None:
        _real_fetch_all_dict = _snowflake_mock.fetch_all_dict

        def _coerce_snowflake_row(row):
            converted = dict(row)
            for key, value in list(converted.items()):
                if not isinstance(value, str):
                    continue
                normalized_key = str(key).upper()
                if 'CURRENT_TIMESTAMP' not in normalized_key and 'CURRENT_TIME' not in normalized_key:
                    continue
                try:
                    converted[key] = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
            return converted

        def _mock_fetch_all_dict(query: str):
            return [_coerce_snowflake_row(row) for row in _real_fetch_all_dict(query)]

        _snowflake_mock.fetch_all_dict = _mock_fetch_all_dict

try:
    from utils.general import helper as _toolathlon_helper
except Exception:
    _toolathlon_helper = None

if _toolathlon_helper is not None:
    _real_run_command = _toolathlon_helper.run_command
    _kill_empty_lsof_re = re.compile(r'^\s*kill\s+-9\s+\$\(\s*lsof\s+-ti:\d+\s*\)\s*2>/dev/null\s*$', re.I)

    # -----------------------------------------------------------------------
    # Notion page duplication: intercept the ``uv run -m
    # utils.app_specific.notion.notion_remove_and_duplicate`` subprocess
    # command and replace it with direct Notion REST API calls.  This avoids
    # the nested subprocess chain that invokes mcp-remote (which needs npm
    # access the container doesn't have).  The intercepted command is called
    # from preprocess/main.py via run_command, where sitecustomize IS active.
    # -----------------------------------------------------------------------
    _NOTION_REMOVE_DUP_MARKER = 'utils.app_specific.notion.notion_remove_and_duplicate'
    _NOTION_DUP_MARKER = 'utils.app_specific.notion.notion_page_duplicator'
    _is_api_mode = os.getenv('NEXUS_TOOLATHLON_SERVER_MODE') != 'mock'

    def _extract_page_id(url):
        slug = url.split('?')[0].split('#')[0].rstrip('/').split('/')[-1]
        compact = ''.join(c for c in slug if c.isalnum())
        if len(compact) < 32:
            raise ValueError(f'Cannot parse page ID from URL: {url}')
        compact = compact[-32:]
        return f'{compact[:8]}-{compact[8:12]}-{compact[12:16]}-{compact[16:20]}-{compact[20:]}'

    def _flag(tokens, name):
        for i, t in enumerate(tokens):
            if t == name and i + 1 < len(tokens):
                return tokens[i + 1]
        return None

    class _NotionAPI:
        """Minimal Notion REST API client using requests (bypasses the shim)."""

        _BASE = 'https://api.notion.com/v1'

        def __init__(self, token):
            import requests as _req

            self._s = _req.Session()
            self._s.headers.update({
                'Authorization': f'Bearer {token}',
                'Notion-Version': '2022-06-28',
                'Content-Type': 'application/json',
            })

        def _get(self, path):
            r = self._s.get(f'{self._BASE}{path}', timeout=60)
            r.raise_for_status()
            return r.json()

        def _post(self, path, payload=None):
            r = self._s.post(f'{self._BASE}{path}', json=payload or {}, timeout=60)
            r.raise_for_status()
            return r.json()

        def _patch(self, path, payload):
            r = self._s.patch(f'{self._BASE}{path}', json=payload, timeout=60)
            r.raise_for_status()
            return r.json()

        def _delete(self, path):
            r = self._s.delete(f'{self._BASE}{path}', timeout=60)
            r.raise_for_status()
            return r.json()

        def get_page(self, page_id):
            return self._get(f'/pages/{page_id}')

        def get_block_children(self, block_id):
            return self._get(f'/blocks/{block_id}/children?page_size=100')

        def create_page(self, parent_id, title):
            return self._post(
                '/pages',
                {
                    'parent': {'page_id': parent_id},
                    'properties': {'title': {'title': [{'text': {'content': title}}]}},
                },
            )

        def append_children(self, block_id, children):
            return self._patch(f'/blocks/{block_id}/children', {'children': children})

        def update_page_title(self, page_id, title):
            return self._patch(
                f'/pages/{page_id}',
                {
                    'properties': {'title': {'title': [{'text': {'content': title}}]}},
                },
            )

        def delete_block(self, block_id):
            return self._delete(f'/blocks/{block_id}')

        def find_child_page(self, parent_id, child_name):
            children = self.get_block_children(parent_id)
            for child in children.get('results', []):
                if child.get('type') != 'child_page':
                    continue
                page = self.get_page(child['id'])
                props = page.get('properties', {})
                tp = props.get('title', {}).get('title') or props.get('Name', {}).get('title')
                if tp:
                    title = ''.join(t.get('plain_text', '') for t in tp).strip()
                    if title == child_name:
                        return child['id']
            return None

    def _run_notion_remove_and_duplicate_via_api(command_str):
        """Intercept ``notion_remove_and_duplicate`` CLI and perform the
        remove-old-page + duplicate-from-source steps via Notion REST API.

        Returns (stdout, stderr, returncode).
        """
        import copy  # deferred – importlib.abc not ready during sitecustomize

        tokens = shlex.split(command_str)
        duplicated_page_id_file = _flag(tokens, '--duplicated_page_id_file')
        needed_subpage_name = _flag(tokens, '--needed_subpage_name')

        if not all([duplicated_page_id_file, needed_subpage_name]):
            raise RuntimeError(f'[nexus shim] Cannot parse notion_remove_and_duplicate args: {command_str}')

        # Load Notion tokens from env var (set by nexus framework)
        token_json = os.getenv('NEXUS_TOOLATHLON_TOKEN_VALUES_JSON', '{}')
        try:
            tokens_dict = json.loads(token_json)
        except Exception as exc:
            raise RuntimeError(f'[nexus shim] Cannot parse NEXUS_TOOLATHLON_TOKEN_VALUES_JSON: {exc}') from exc

        source_url = tokens_dict.get('source_notion_page_url', '')
        eval_url = tokens_dict.get('eval_notion_page_url', '')
        notion_key = tokens_dict.get('notion_integration_key', '')
        notion_key_eval = tokens_dict.get('notion_integration_key_eval', '') or notion_key

        if not all([source_url, eval_url, notion_key]):
            raise RuntimeError(
                f'[nexus shim] Missing Notion tokens in NEXUS_TOOLATHLON_TOKEN_VALUES_JSON '
                f'(have keys: {list(tokens_dict.keys())})'
            )

        eval_api = _NotionAPI(notion_key_eval)
        full_api = _NotionAPI(notion_key)

        eval_parent_id = _extract_page_id(eval_url)
        source_parent_id = _extract_page_id(source_url)

        out_lines = []
        out_lines.append('[nexus shim] Notion remove+duplicate via REST API')

        # --- Step 1: Remove old page from eval parent ---
        out_lines.append(f'Removing old page "{needed_subpage_name}" from eval parent {eval_parent_id}')
        children = eval_api.get_block_children(eval_parent_id)
        for child in children.get('results', []):
            if child.get('type') != 'child_page':
                continue
            try:
                page = eval_api.get_page(child['id'])
                props = page.get('properties', {})
                tp = props.get('title', {}).get('title') or props.get('Name', {}).get('title')
                if tp:
                    title = ''.join(t.get('plain_text', '') for t in tp).strip()
                    if title == needed_subpage_name:
                        eval_api.delete_block(child['id'])
                        out_lines.append(f'Deleted old page: {child["id"]}')
            except Exception as exc:
                out_lines.append(f'Warning: error checking/deleting child {child["id"]}: {exc}')

        # --- Step 2: Find source child page ---
        out_lines.append(f'Looking for child page "{needed_subpage_name}" under source {source_parent_id}')
        child_page_id = full_api.find_child_page(source_parent_id, needed_subpage_name)
        if not child_page_id:
            raise RuntimeError(
                f"[nexus shim] Child page '{needed_subpage_name}' not found under source {source_parent_id}"
            )
        out_lines.append(f'Found source child page: {child_page_id}')

        # --- Step 3: Read blocks + create new page + copy ---
        source_blocks = full_api.get_block_children(child_page_id)
        new_page = full_api.create_page(eval_parent_id, needed_subpage_name)
        new_page_id = new_page['id']
        out_lines.append(f'Created new page (via API): {new_page_id}')

        blocks_to_copy = []
        for block in source_blocks.get('results', []):
            btype = block.get('type', '')
            if btype in ('child_page', 'child_database'):
                continue
            new_block = {'type': btype, 'object': 'block'}
            if btype in block:
                bdata = copy.deepcopy(block[btype])
                bdata.pop('id', None)
                new_block[btype] = bdata
            blocks_to_copy.append(new_block)

        for i in range(0, len(blocks_to_copy), 100):
            batch = blocks_to_copy[i : i + 100]
            try:
                full_api.append_children(new_page_id, batch)
            except Exception as exc:
                out_lines.append(f'Warning: failed to append some blocks: {exc}')

        out_lines.append(f'Copied {len(blocks_to_copy)} blocks')

        # --- Step 4: Write output file ---
        if duplicated_page_id_file:
            import pathlib

            pathlib.Path(duplicated_page_id_file).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(duplicated_page_id_file).write_text(new_page_id)
            out_lines.append(f'Page ID saved to: {duplicated_page_id_file}')

        out_lines.append('✅ Notion remove+duplicate completed successfully!')
        stdout = '\n'.join(out_lines)
        return stdout, '', 0

    def _run_notion_duplicate_via_api(command_str):
        """Intercept ``notion_page_duplicator`` CLI and perform via REST API.

        Returns (stdout, stderr, returncode).
        """
        import copy

        tokens = shlex.split(command_str)
        source_parent_url = _flag(tokens, '--source-parent')
        child_name = _flag(tokens, '--child-name')
        target_parent_url = _flag(tokens, '--target-parent')
        notion_key = _flag(tokens, '--notion-key')
        output_file = _flag(tokens, '--output-file')

        if not all([source_parent_url, child_name, target_parent_url, notion_key]):
            raise RuntimeError(f'[nexus shim] Cannot parse notion_page_duplicator args: {command_str}')

        api = _NotionAPI(notion_key)
        source_parent_id = _extract_page_id(source_parent_url)
        target_parent_id = _extract_page_id(target_parent_url)

        print('[nexus shim] Notion duplicate via REST API')

        child_page_id = api.find_child_page(source_parent_id, child_name)
        if not child_page_id:
            raise RuntimeError(f"[nexus shim] Child page '{child_name}' not found")

        source_blocks = api.get_block_children(child_page_id)
        new_page = api.create_page(target_parent_id, child_name)
        new_page_id = new_page['id']

        blocks_to_copy = []
        for block in source_blocks.get('results', []):
            btype = block.get('type', '')
            if btype in ('child_page', 'child_database'):
                continue
            new_block = {'type': btype, 'object': 'block'}
            if btype in block:
                bdata = copy.deepcopy(block[btype])
                bdata.pop('id', None)
                new_block[btype] = bdata
            blocks_to_copy.append(new_block)

        for i in range(0, len(blocks_to_copy), 100):
            batch = blocks_to_copy[i : i + 100]
            try:
                api.append_children(new_page_id, batch)
            except Exception as exc:
                print(f'Warning: failed to append some blocks: {exc}')

        api.update_page_title(new_page_id, child_name)

        if output_file:
            import pathlib

            pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(output_file).write_text(new_page_id)

        stdout = f'\n✅ Page duplication completed successfully!\nPage ID: {new_page_id}\n'
        return stdout, '', 0

    async def _patched_run_command(command, debug=False, show_output=False):
        cmd_str = str(command)
        mirror_dir = _git_mirror_push_local_dir(cmd_str)
        if mirror_dir is not None:
            _prune_git_mirror_special_refs(mirror_dir)

        # Intercept Notion subprocess commands in API mode
        if _is_api_mode:
            interceptor = None
            if _NOTION_REMOVE_DUP_MARKER in cmd_str:
                interceptor = _run_notion_remove_and_duplicate_via_api
            elif _NOTION_DUP_MARKER in cmd_str:
                interceptor = _run_notion_duplicate_via_api
            if interceptor is not None:
                try:
                    stdout, stderr, rc = interceptor(cmd_str)
                    if debug or show_output:
                        print(stdout)
                    return stdout, stderr, rc
                except Exception as exc:
                    print(f'[nexus shim] REST API Notion operation failed: {exc}')
                    # Fall through to original command as last resort

        try:
            return await _real_run_command(command, debug=debug, show_output=show_output)
        except RuntimeError as exc:
            if not _kill_empty_lsof_re.match(cmd_str.strip()):
                raise
            message = str(exc)
            if 'return_code=2' not in message:
                raise
            if debug:
                with contextlib.suppress(Exception):
                    _toolathlon_helper.print_color('Ignoring empty lsof kill command', 'yellow')
            return '', '', 0

    _toolathlon_helper.run_command = _patched_run_command
