"""Wedata key based authentication for Taiji dataset services."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import random
import socket
import stat
import threading
import uuid

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from nexus.dataset.exceptions import DatasetAuthenticationError, DatasetConfigurationError
from nexus.utils import pydes
from nexus.utils.paths import TAIHU_AUTH_TOKEN_FILE, WEDATA_AUTH_KEY_FILE


logger = logging.getLogger(__name__)

# 换票重试参数：5次尝试 + 指数退避 + 抖动，用于吸收 auth.tdw.oa.com
# 侧的持续性故障（SNAT 端口耗尽回收 / 限流窗口 / 中间层 5xx）。
# 实测他人以 60s 间隔重试 1 次即恢复，说明故障窗口约为分钟级；
# backoff_base=10s 时各次 sleep 约为 10s / 20s / 40s / 80s，
# 前3次累计 ~70s 已超过 60s 故障窗口，整体最长等待约 2.5 分钟。
DEFAULT_TICKET_MAX_RETRY = 5
DEFAULT_TICKET_BACKOFF_BASE = 10.0  # 秒
DEFAULT_TICKET_BACKOFF_JITTER = 5.0  # 秒
DEFAULT_TICKET_TIMEOUT = aiohttp.ClientTimeout(total=10)  # 避免长时间挂起


AUTH_URL_MAPPING = {
    'cn': 'http://auth.tdw.oa.com/api/auth/st2',
    'sg': 'http://sg.auth.tdw.sgwoa.com:8050/api/auth/st2',
}

WEDATA_SECURITY_KEYS_URL = 'https://wedata.woa.com/security/user/keys'
DEFAULT_WEDATA_TARGET = 'unity_catalog'
ENV_WEDATA_KEY = 'WEDATA_KEY'
ENV_WEDATA_KEY_FILE = 'WEDATA_KEY_FILE'

ENV_TAI_PAT = 'TAI_PAT'
# Taiji (太极) is the platform hosting the PAT application page; the PAT itself
# is a Taihu (太湖) token. The TAIJI prefix here is intentional, not a typo for TAIHU.
TAIJI_PAT_APPLY_URL = 'https://tai.it.woa.com/user/pat'


def tai_pat_from_env() -> str | None:
    """Return the Taihu personal access token from the environment, if configured."""

    value = os.getenv(ENV_TAI_PAT)
    if value and value.strip():
        return value.strip()
    return None


def tai_pat_from_file(path: Path) -> str:
    """Read a Taihu PAT from a local file, enforcing strict permissions.

    The file must not be readable or writable by group/other (i.e. at most
    ``chmod 600``).  Raises :class:`DatasetConfigurationError` on permission
    violations or an empty file.
    """

    path = Path(path).expanduser().resolve()
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise DatasetConfigurationError(
            f'Taihu token file {path} must not be readable or writable by group/other'
            f' (current permissions: {oct(mode)}). Fix it with: chmod 600 {path}'
        )
    token = path.read_text(encoding='utf-8').strip()
    if not token:
        raise DatasetConfigurationError(f'Taihu token file {path} is empty')
    return token


def default_taihu_pat() -> str | None:
    """Return the Taihu PAT from the default ``~/.nexus/auth/taihu-pat`` file, if present.

    Returns ``None`` when the file does not exist, so callers can fall back to
    other auth sources.  Permission and empty-file checks still apply when the
    file is present.
    """

    path = TAIHU_AUTH_TOKEN_FILE
    if not path.is_file():
        return None
    return tai_pat_from_file(path)


def wedata_key_available() -> bool:
    """Return ``True`` when a Wedata key is configured via env or the default file."""

    env_raw = os.getenv(ENV_WEDATA_KEY)
    if env_raw and env_raw.strip():
        return True
    path_value = os.getenv(ENV_WEDATA_KEY_FILE)
    path = Path(path_value).expanduser() if path_value else WEDATA_AUTH_KEY_FILE
    return path.is_file()


def resolve_default_tai_pat() -> str | None:
    """Resolve a Taihu PAT from the default sources.

    Priority: ``TAI_PAT`` env var, then the default ``~/.nexus/auth/taihu-pat``
    file.  Returns ``None`` when neither is configured.
    """

    return tai_pat_from_env() or default_taihu_pat()


def format_wedata_auth_error(reason: str | None = None) -> str:
    """Return an actionable Wedata key setup hint for auth failures."""

    prefix = 'Wedata authentication failed'
    if reason:
        prefix = f'{prefix}: {reason}'
    return (
        f'{prefix}. Get a Wedata security key from {WEDATA_SECURITY_KEYS_URL}, then save the raw key JSON to '
        f'{WEDATA_AUTH_KEY_FILE}, or set {ENV_WEDATA_KEY} to the raw key JSON.'
    )


def format_taihu_auth_error(reason: str | None = None) -> str:
    """Return an actionable Taihu PAT setup hint for auth failures."""

    prefix = 'Taihu authentication failed'
    if reason:
        prefix = f'{prefix}: {reason}'
    return (
        f'{prefix}. Check that your Taihu personal access token is valid and authorized for this service '
        f'(a 401 AUDIENCE_MISMATCH means the token is not scoped for it). Set {ENV_TAI_PAT} or create '
        f'{TAIHU_AUTH_TOKEN_FILE} (chmod 600) with a fresh token from {TAIJI_PAT_APPLY_URL}.'
    )


def format_dataset_auth_error(reason: str | None = None, *, taihu: bool) -> str:
    """Return an auth-failure hint tailored to the active authentication mode."""

    return format_taihu_auth_error(reason) if taihu else format_wedata_auth_error(reason)


@dataclass(frozen=True)
class WedataKey:
    """Parsed Wedata security key."""

    subject: str
    key: str
    key_id: str

    @classmethod
    def from_raw(cls, raw: str, *, source: str) -> WedataKey:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetConfigurationError(
                f'Invalid Wedata key from {source}: expected JSON downloaded from {WEDATA_SECURITY_KEYS_URL}'
            ) from exc
        if not isinstance(payload, dict):
            raise DatasetConfigurationError(
                f'Invalid Wedata key from {source}: expected JSON object downloaded from {WEDATA_SECURITY_KEYS_URL}'
            )
        return cls.from_mapping(payload, source=source)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, source: str) -> WedataKey:
        subject = payload.get('subject')
        key = payload.get('key')
        key_id = payload.get('id') or payload.get('key_id')
        missing = [
            name
            for name, value in {'subject': subject, 'key': key, 'id': key_id}.items()
            if value is None or value == ''
        ]
        if missing:
            raise DatasetConfigurationError(
                f'Invalid Wedata key from {source}: missing {", ".join(missing)}. '
                f'Download a fresh key from {WEDATA_SECURITY_KEYS_URL}.'
            )
        if not isinstance(subject, str) or not isinstance(key, str):
            raise DatasetConfigurationError(
                f'Invalid Wedata key from {source}: subject and key must be strings. '
                f'Download a fresh key from {WEDATA_SECURITY_KEYS_URL}.'
            )
        return cls(subject=subject, key=key, key_id=str(key_id))

    def to_auth_payload(self) -> dict[str, str]:
        return {'subject': self.subject, 'key': self.key, 'id': self.key_id}


_CACHED_WEDATA_KEY: WedataKey | None = None
_CACHED_WEDATA_KEY_LOCK = threading.Lock()


def load_wedata_key(*, raw_key: str | None = None, key_file: str | Path | None = None) -> WedataKey:
    """Load the Wedata key from env content, env file, or ``~/.nexus``."""

    if raw_key is not None or key_file is not None:
        return _load_wedata_key_uncached(raw_key=raw_key, key_file=key_file)

    global _CACHED_WEDATA_KEY
    if _CACHED_WEDATA_KEY is not None:
        return _CACHED_WEDATA_KEY

    with _CACHED_WEDATA_KEY_LOCK:
        if _CACHED_WEDATA_KEY is None:
            _CACHED_WEDATA_KEY = _load_wedata_key_uncached()
        return _CACHED_WEDATA_KEY


def clear_wedata_key_cache() -> None:
    """Clear the process-local cached implicit Wedata key."""

    global _CACHED_WEDATA_KEY
    with _CACHED_WEDATA_KEY_LOCK:
        _CACHED_WEDATA_KEY = None


def apply_wedata_key(wedata_key: str | Mapping[str, Any] | None) -> None:
    """Publish a config-supplied Wedata key to the environment.

    Lives here rather than in a task module because every task whose fixtures
    are dataset-carried needs it, and the rule it encodes is about this layer:
    credentials are resolved from ``WEDATA_KEY`` / ``~/.nexus/auth/wedata/key``
    (see ``_load_wedata_key_uncached``), neither of which exists on a platform
    host. ``run_single`` may not read process env for its own configuration, so
    the key travels inside ``task_config`` and is published here, before the
    first ``task_files`` download.

    Accepts both shapes the value arrives in: a raw JSON string (what a local
    ``export WEDATA_KEY=...`` gives) and a mapping (what the platform hands over
    after resolving ``${oc.env:WEDATA_KEY}`` from 七彩石). ``WedataKey.from_raw``
    only parses the string form, so a mapping is re-serialised rather than
    stringified with ``str()`` -- Python's repr uses single quotes and is not
    valid JSON.

    Blank/None is a no-op rather than a clear: a local run passes nothing and
    must keep using its key file or a pre-existing env var.

    The cache flush matters: ``load_wedata_key`` memoises the first key it
    resolves per process, and a platform worker serves many runs. Without it the
    second run would sign with whatever the first one happened to load.
    """

    if not wedata_key:
        return
    raw = json.dumps(dict(wedata_key)) if isinstance(wedata_key, Mapping) else str(wedata_key)
    if not raw.strip():
        return
    # Reject shell-damaged JSON before signing. Otherwise the optional auth
    # lookup can mistake a malformed configured key for anonymous access.
    WedataKey.from_raw(raw.strip(), source='task_config.wedata_key (quote JSON when exporting WEDATA_KEY)')
    os.environ[ENV_WEDATA_KEY] = raw.strip()
    clear_wedata_key_cache()


def _load_wedata_key_uncached(*, raw_key: str | None = None, key_file: str | Path | None = None) -> WedataKey:
    env_raw = raw_key if raw_key is not None else os.getenv(ENV_WEDATA_KEY)
    if env_raw and env_raw.strip():
        return WedataKey.from_raw(env_raw.strip(), source=ENV_WEDATA_KEY)

    path_value = key_file if key_file is not None else os.getenv(ENV_WEDATA_KEY_FILE)
    path = Path(path_value).expanduser() if path_value else WEDATA_AUTH_KEY_FILE
    if path.is_file():
        try:
            return WedataKey.from_raw(path.read_text(encoding='utf-8').strip(), source=str(path))
        except OSError as exc:
            raise DatasetConfigurationError(f'Failed to read Wedata key file {path}: {exc}') from exc

    raise DatasetConfigurationError(
        f'Wedata key missing: set {ENV_WEDATA_KEY}, set {ENV_WEDATA_KEY_FILE}, or create {WEDATA_AUTH_KEY_FILE}. '
        f'Download the key from {WEDATA_SECURITY_KEYS_URL}.'
    )


def default_wedata_subject() -> str | None:
    """Return the configured Wedata subject, or ``None`` when no key is configured."""

    try:
        return load_wedata_key().subject
    except DatasetConfigurationError:
        return None


class WedataAuthenticator:
    """Generates dataset service auth headers, caching tickets until they expire."""

    def __init__(
        self,
        *,
        key: WedataKey | None = None,
        raw_key: str | None = None,
        key_file: str | Path | None = None,
        proxy_user: str | None = None,
        region: str = 'cn',
    ) -> None:
        if region not in AUTH_URL_MAPPING:
            raise DatasetConfigurationError(f'Unsupported Wedata auth region {region!r}')

        self.wedata_key = key or load_wedata_key(raw_key=raw_key, key_file=key_file)
        self.user_name = self.wedata_key.subject
        self.cmk = self.wedata_key.key
        self.target = DEFAULT_WEDATA_TARGET
        self.proxy_user = proxy_user
        self.region = region
        self.request_url = AUTH_URL_MAPPING[region]
        self.ip = self._get_host_ip()
        self.sequence = random.randint(0, 999)
        self.expire_time_stamp = int(datetime.timestamp(datetime.now())) * 1000
        self.session_ticket: str | None = None
        self.service_ticket: str | None = None
        self.session_key: bytes | None = None

        # 并发保护：多协程同时进入 get_headers 时，只让一个协程实际换票，
        # 其余协程等待并复用结果，避免对auth.tdw.oa.com 造成换票风暴。
        self._refresh_lock = asyncio.Lock()

        # 换票统计（可用于观测缓存命中率、并发抖动）。
        self._metric_total = 0
        self._metric_fail = 0
        self._metric_cache_hit = 0

        self.identifier: dict[str, Any] = {
            'user': self.user_name,
            'host': self.ip,
            'target': self.target,
            'epoch': self.wedata_key.key_id,
            'lifetime': 7200000,
        }
        self.client_authenticator: dict[str, Any] = {
            'principle': self.user_name,
            'host': self.ip,
        }

    async def get_headers(self, session: aiohttp.ClientSession) -> dict[str, str]:
        """返回认证 header，ticket 过期时自动刷新。

        并发安全：使用 double-checked locking，多协程同时进入时只会有一个
        协程真正换票，其余协程复用结果，避免对 ``auth.tdw.oa.com`` 造成
        换票风暴（这是生产环境偶发 ``Request session ticket failed`` 的
        主要根因）。
        """
        if self._is_expired():
            async with self._refresh_lock:
                # 拿到锁后再判一次——可能已被别的协程刷过了。
                if self._is_expired():
                    await self._get_session_ticket(session)
                    self._decrypt_client_ticket()
                else:
                    self._metric_cache_hit += 1
        else:
            self._metric_cache_hit += 1
        return self._construct_authentication()

    async def _get_session_ticket_once(self, session: aiohttp.ClientSession) -> None:
        """单次换票；失败抛:class:`DatasetAuthenticationError` 附带诊断上下文。

        注意：必须先判定status 再把 response text赋值给 self.session_ticket，
        否则失败的错误响应体会污染 ticket 状态，导致后续 _decrypt_client_ticket
        报错误，误导排查方向。
        """
        self._metric_total += 1
        self.identifier['timestamp'] = int(datetime.timestamp(datetime.now()))
        identifier_body = base64.b64encode(json.dumps(self.identifier).encode('utf-8')).decode('ascii')
        async with session.get(
            self.request_url,
            params={'ident': identifier_body},
            timeout=DEFAULT_TICKET_TIMEOUT,
        ) as response:
            text = await response.text()
            if response.status != 200:
                self._metric_fail += 1
                raise DatasetAuthenticationError(f'Request session ticket failed: {text}')
            self.session_ticket = text

    async def _get_session_ticket(
        self,
        session: aiohttp.ClientSession,
        max_retry: int = DEFAULT_TICKET_MAX_RETRY,
        backoff_base: float = DEFAULT_TICKET_BACKOFF_BASE,
        backoff_jitter: float = DEFAULT_TICKET_BACKOFF_JITTER,
    ) -> None:
        """带指数退避 + 抖动的换票，用于吸收偶发的瞬时失败。

        单次失败 ~0.06% 的场景下，三次带抖动重试后失败率量级→ 10⁻¹¹。
        抖动很关键：多协程在同一秒失败时，无抖动的重试会同时再次
        撞到 auth 服务，重试反而加剧风暴。
        """
        last_exc: DatasetAuthenticationError | None = None
        for attempt in range(1, max_retry + 1):
            try:
                await self._get_session_ticket_once(session)
                if attempt > 1:
                    logger.info(
                        'TAuth ticket recovered after retry: attempt=%d/%d',
                        attempt,
                        max_retry,
                    )
                return
            except DatasetAuthenticationError as e:
                last_exc = e
                if attempt >= max_retry:
                    break
                sleep_s = backoff_base * (2 ** (attempt - 1)) + random.random() * backoff_jitter
                logger.warning(
                    'TAuth ticket attempt %d/%d failed, retry in %.3fs: %s',
                    attempt,
                    max_retry,
                    sleep_s,
                    e,
                )
                await asyncio.sleep(sleep_s)
        # 所有重试都失败
        assert last_exc is not None
        raise last_exc

    def _decrypt_client_ticket(self) -> None:
        if self.session_ticket is None:
            raise DatasetAuthenticationError('Session ticket is missing')
        session_ticket = json.loads(self.session_ticket)
        self.service_ticket = session_ticket['st']
        client_ticket = base64.b64decode(session_ticket['ct'])
        cmk_epoch = session_ticket['cmkEpoch']

        try:
            cmk = bytes.fromhex(base64.b64decode(self.cmk).decode('utf-8'))
            des = pydes.triple_des(cmk, pydes.ECB, padmode=pydes.PAD_PKCS5)
            ticket = json.loads(des.decrypt(client_ticket).decode('utf-8'))
            self.expire_time_stamp = ticket['timestamp'] + ticket['lifetime'] - 60000
            self.session_key = base64.b64decode(ticket['sessionKey'])
        except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise DatasetAuthenticationError(
                f'Invalid Wedata key for key ID {cmk_epoch}. Download a fresh key from {WEDATA_SECURITY_KEYS_URL}.'
            ) from exc

    def _construct_authentication(self) -> dict[str, str]:
        if self.session_key is None or self.service_ticket is None:
            raise DatasetAuthenticationError('Wedata auth session key is missing')

        self.client_authenticator['timestamp'] = int(datetime.timestamp(datetime.now())) * 1000
        self.client_authenticator['nonce'] = uuid.uuid1().hex
        self.client_authenticator['sequence'] = self.sequence
        self.sequence += 1
        if self.proxy_user:
            self.client_authenticator['proxyUser'] = self.proxy_user

        payload = json.dumps(self.client_authenticator).encode('utf-8')
        des = pydes.triple_des(self.session_key, pydes.ECB, padmode=pydes.PAD_PKCS5)
        authentication = f'tauth.{self.service_ticket}.{base64.b64encode(des.encrypt(payload)).decode("utf-8")}'
        return {'secure-authentication': authentication}

    def _is_expired(self) -> bool:
        return self.expire_time_stamp <= int(datetime.timestamp(datetime.now())) * 1000

    @staticmethod
    def _get_host_ip() -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())
        finally:
            s.close()


class TaiPatAuthenticator:
    """Taihu MCP unified authentication using a personal access token (PAT).

    Sends only ``Authorization: Bearer <pat>`` and never emits the
    ``secure-authentication`` header, so the Taihu gateway does not mistake the
    request for a TAuth call and reject it with a 401.
    """

    def __init__(self, *, pat: str | None = None, pat_file: Path | None = None) -> None:
        if pat and pat_file:
            raise DatasetConfigurationError('pat and pat_file are mutually exclusive')
        if pat:
            resolved: str | None = pat.strip()
        elif pat_file is not None:
            resolved = tai_pat_from_file(pat_file)
        else:
            resolved = resolve_default_tai_pat()
        if not resolved:
            raise DatasetConfigurationError(
                f'Taihu PAT missing: set {ENV_TAI_PAT}, pass pat=, or create {TAIHU_AUTH_TOKEN_FILE} (chmod 600). '
                f'Apply for a token at {TAIJI_PAT_APPLY_URL}.'
            )
        self.pat = resolved

    async def get_headers(self, session: aiohttp.ClientSession) -> dict[str, str]:
        # ``session`` is accepted for interface parity with WedataAuthenticator; PAT auth is stateless.
        del session
        return {'Authorization': f'Bearer {self.pat}'}


DatasetAuthenticator = WedataAuthenticator | TaiPatAuthenticator


def prefer_tai_pat(
    *,
    authenticator: DatasetAuthenticator | None = None,
    auth_token: str | None = None,
) -> bool:
    """Return ``True`` when requests should use Taihu PAT auth and the Taihu gateway URL.

    An explicit ``auth_token`` (raw ``secure-authentication`` value) or a
    :class:`WedataAuthenticator` keeps the legacy TAuth path; a
    :class:`TaiPatAuthenticator` or a configured ``TAI_PAT`` selects Taihu.
    """

    if auth_token is not None:
        return False
    if authenticator is not None:
        return isinstance(authenticator, TaiPatAuthenticator)
    # No explicit authenticator: mirror build_default_authenticator()'s choice.
    if tai_pat_from_env() is not None:
        return True
    if wedata_key_available():
        return False
    return default_taihu_pat() is not None


def _default_tai_pat_authenticator() -> TaiPatAuthenticator:
    """Single construction point for the auto-detected Taihu PAT authenticator.

    Keeping this in one place means a future change to how the default
    ``TaiPatAuthenticator`` is built (new constructor args, extra setup, ...)
    only needs to be made here instead of at every selection branch.
    """

    return TaiPatAuthenticator()


def build_default_authenticator() -> DatasetAuthenticator:
    """Auto-detect the dataset authenticator.

    Selection priority:

    1. ``TAI_PAT`` env var -> Taihu PAT.
    2. A configured Wedata key (env or ``~/.nexus/auth/wedata/key``) -> Wedata
       TAuth, so existing users are unaffected.
    3. The default Taihu token file ``~/.nexus/auth/taihu-pat`` -> Taihu PAT.
    4. Otherwise raise an actionable error guiding the user to configure Taihu,
       which is the going-forward replacement for Wedata TAuth.
    """

    if tai_pat_from_env() is not None:
        return _default_tai_pat_authenticator()
    if wedata_key_available():
        return WedataAuthenticator()
    if default_taihu_pat() is not None:
        return _default_tai_pat_authenticator()
    raise DatasetConfigurationError(
        f'Dataset authentication is not configured. Set {ENV_TAI_PAT} or create {TAIHU_AUTH_TOKEN_FILE} '
        f'(chmod 600) with a Taihu personal access token. Apply for a token at {TAIJI_PAT_APPLY_URL}.'
    )


def authenticator_to_auth_payload(authenticator: DatasetAuthenticator) -> dict[str, Any]:
    """Serialize an already-selected authenticator into a transient auth payload.

    ``task_files`` signing carries credentials as a serializable payload (so it
    can be shipped to the runtime), not as a live authenticator object. This
    lets callers reuse the dataset client's *already chosen* authentication
    context instead of re-reading process-global configuration, which is what
    keeps ``task_files`` consistent with the main client's auth selection.
    """

    if isinstance(authenticator, TaiPatAuthenticator):
        return {'kind': 'tai_pat', 'pat': authenticator.pat}
    if isinstance(authenticator, WedataAuthenticator):
        return {'kind': 'wedata_key', 'key': authenticator.wedata_key.to_auth_payload()}
    raise DatasetConfigurationError(
        f'Unsupported authenticator type for task_files auth payload: {type(authenticator).__name__}'
    )
