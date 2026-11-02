"""Unit tests for WedataAuthenticator: concurrent locking, retry-backoff, timeout."""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import nexus.dataset.auth as auth_module
from nexus.dataset.auth import (
    DEFAULT_TICKET_MAX_RETRY,
    WedataAuthenticator,
    WedataKey,
)
from nexus.dataset.exceptions import DatasetAuthenticationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_KEY = WedataKey(subject='test_user', key='dGVzdF9rZXk=', key_id='1')


def _make_auth(**kwargs: object) -> WedataAuthenticator:
    """Create a WedataAuthenticator with a fake key, bypassing socket lookup."""
    with patch.object(WedataAuthenticator, '_get_host_ip', return_value='127.0.0.1'):
        return WedataAuthenticator(key=_FAKE_KEY, **kwargs)  # type: ignore[arg-type]


def _make_ticket_response(status: int = 200, body: str = '{}') -> MagicMock:
    """Return an async context-manager mock that simulates aiohttp response."""
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _fake_session(responses: list[MagicMock]) -> MagicMock:
    """Return a fake aiohttp.ClientSession whose .get() yields responses in order."""
    session = MagicMock()
    session.get = MagicMock(side_effect=responses)
    return session


# ---------------------------------------------------------------------------
# _get_session_ticket_once
# ---------------------------------------------------------------------------

class TestGetSessionTicketOnce:
    @pytest.mark.asyncio
    async def test_success_sets_session_ticket(self) -> None:
        auth = _make_auth()
        resp_cm = _make_ticket_response(200, 'some_ticket_body')
        session = _fake_session([resp_cm])

        await auth._get_session_ticket_once(session)

        assert auth.session_ticket == 'some_ticket_body'
        assert auth._metric_total == 1
        assert auth._metric_fail == 0

    @pytest.mark.asyncio
    async def test_non_200_raises_and_does_not_set_ticket(self) -> None:
        auth = _make_auth()
        resp_cm = _make_ticket_response(500, 'internal error')
        session = _fake_session([resp_cm])

        with pytest.raises(DatasetAuthenticationError, match='Request session ticket failed'):
            await auth._get_session_ticket_once(session)

        # ticket不应被污染
        assert auth.session_ticket is None
        assert auth._metric_fail == 1

    @pytest.mark.asyncio
    async def test_empty_body_on_failure_raises_clearly(self) -> None:
        """auth.tdw.oa.com 偶发返回空 body 的 5xx，错误消息应包含状态信息。"""
        auth = _make_auth()
        resp_cm = _make_ticket_response(503, '')
        session = _fake_session([resp_cm])

        with pytest.raises(DatasetAuthenticationError, match='Request session ticket failed'):
            await auth._get_session_ticket_once(session)


# ---------------------------------------------------------------------------
# _get_session_ticket (retry wrapper)
# ---------------------------------------------------------------------------

class TestGetSessionTicketRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self) -> None:
        auth = _make_auth()
        resp_cm = _make_ticket_response(200, 'ticket_ok')
        session = _fake_session([resp_cm])

        await auth._get_session_ticket(session)

        assert auth.session_ticket == 'ticket_ok'
        assert session.get.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure_and_succeeds(self) -> None:
        """第1次失败，第2次成功——应恢复并只计一次 fail。"""
        auth = _make_auth()
        fail_cm = _make_ticket_response(500, 'err')
        ok_cm = _make_ticket_response(200, 'ticket_ok')
        session = _fake_session([fail_cm, ok_cm])

        with patch('asyncio.sleep', new_callable=AsyncMock):
            await auth._get_session_ticket(session, max_retry=3, backoff_base=0, backoff_jitter=0)

        assert auth.session_ticket == 'ticket_ok'
        assert session.get.call_count == 2
        assert auth._metric_fail == 1

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self) -> None:
        auth = _make_auth()
        fail_cms = [_make_ticket_response(500, 'err') for _ in range(DEFAULT_TICKET_MAX_RETRY)]
        session = _fake_session(fail_cms)

        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(DatasetAuthenticationError):
                await auth._get_session_ticket(session, max_retry=DEFAULT_TICKET_MAX_RETRY, backoff_base=0, backoff_jitter=0)

        assert session.get.call_count == DEFAULT_TICKET_MAX_RETRY
        assert auth._metric_fail == DEFAULT_TICKET_MAX_RETRY

    @pytest.mark.asyncio
    async def test_backoff_sleep_is_called_between_retries(self) -> None:
        auth = _make_auth()
        fail_cms = [_make_ticket_response(500, 'err') for _ in range(2)]
        session = _fake_session(fail_cms)

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(DatasetAuthenticationError):
                await auth._get_session_ticket(session, max_retry=2, backoff_base=0.1, backoff_jitter=0)

        #2次尝试应有1次 sleep
        assert mock_sleep.call_count == 1
        sleep_duration = mock_sleep.call_args[0][0]
        assert sleep_duration >= 0.1  # 至少 backoff_base


# ---------------------------------------------------------------------------
# get_headers: double-checked locking / concurrency
# ---------------------------------------------------------------------------

class TestGetHeadersConcurrency:
    def _make_expired_auth(self) -> WedataAuthenticator:
        auth = _make_auth()
        auth.expire_time_stamp =0  # 强制过期
        return auth

    @pytest.mark.asyncio
    async def test_only_one_ticket_refresh_under_concurrency(self) -> None:
        """多协程并发调用 get_headers，只应换票一次（其余命中缓存）。"""
        auth = self._make_expired_auth()

        call_count = 0

        async def fake_get_session_ticket(session: object) -> None:
            nonlocal call_count
            call_count += 1
            # 模拟换票后设置一个未来的过期时间
            auth.expire_time_stamp = int(10**13)  # 很远的将来
            auth.session_ticket = json.dumps({'st': 'svc', 'ct': base64.b64encode(b'x' * 16).decode(), 'cmkEpoch': '1'})

        with patch.object(auth, '_get_session_ticket', side_effect=fake_get_session_ticket):
            with patch.object(auth, '_decrypt_client_ticket'):
                with patch.object(auth, '_construct_authentication', return_value={'secure-authentication': 'x'}):
                    session = MagicMock()
                    tasks = [auth.get_headers(session) for _ in range(10)]
                    await asyncio.gather(*tasks)

        assert call_count == 1, f'Expected 1 ticket refresh, got {call_count}'
        # 其余 9 个协程命中缓存
        assert auth._metric_cache_hit >= 9

    @pytest.mark.asyncio
    async def test_cache_hit_when_not_expired(self) -> None:
        auth = _make_auth()
        auth.expire_time_stamp = int(10**13)  # 未过期

        with patch.object(auth, '_construct_authentication', return_value={'secure-authentication': 'y'}):
            session = MagicMock()
            result = await auth.get_headers(session)

        assert result == {'secure-authentication': 'y'}
        assert auth._metric_cache_hit == 1

    @pytest.mark.asyncio
    async def test_double_check_prevents_second_refresh(self) -> None:
        """拿到锁后如果已被刷过（不再过期），不应再次换票。"""
        auth = self._make_expired_auth()
        refresh_count = 0

        original_lock_cls = asyncio.Lock

        async def mock_get_session_ticket(session: object) -> None:
            nonlocal refresh_count
            refresh_count += 1
            auth.expire_time_stamp = int(10**13)

        with patch.object(auth, '_get_session_ticket', side_effect=mock_get_session_ticket):
            with patch.object(auth, '_decrypt_client_ticket'):
                with patch.object(auth, '_construct_authentication', return_value={'secure-authentication': 'z'}):
                    session = MagicMock()
                    # 第一次调用触发换票
                    await auth.get_headers(session)
                    # 第二次调用不应再换票（double-check 命中缓存）
                    await auth.get_headers(session)

        assert refresh_count == 1
        assert auth._metric_cache_hit >= 1
