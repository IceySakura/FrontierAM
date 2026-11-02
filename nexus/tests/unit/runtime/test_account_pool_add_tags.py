from typing import Any
from unittest.mock import AsyncMock

import pytest

from nexus.runtime.accounts import AccountPoolManager, AsyncAccountPoolClient


@pytest.mark.asyncio
async def test_client_add_account_sends_nonempty_tags() -> None:
    client = AsyncAccountPoolClient('http://account-pool.test')
    call = AsyncMock(return_value={'success': True, 'account_id': '42'})
    client._call = call  # type: ignore[method-assign]

    account_id = await client.add_account(
        'gui-sandbox',
        'ags-waa-1',
        credentials={'instance_id': 'sandbox-1'},
        tags=['ags', '', 'waa'],
    )

    assert account_id == 42
    call.assert_awaited_once_with(
        'AddAccount',
        {
            'pool': 'gui-sandbox',
            'account_key': 'ags-waa-1',
            'credentials': '{"instance_id": "sandbox-1"}',
            'tokens': '{}',
            'shared_cap': 0,
            'note': '',
            'tags': ['ags', 'waa'],
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('tags', [None, [], ['']])
async def test_client_add_account_omits_tags_without_nonempty_values(tags: list[str] | None) -> None:
    client = AsyncAccountPoolClient('http://account-pool.test')
    payloads: list[dict[str, Any]] = []

    async def call(method: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert method == 'AddAccount'
        payloads.append(payload)
        return {'success': True, 'account_id': 42}

    client._call = call  # type: ignore[method-assign]

    await client.add_account('gui-sandbox', 'ags-waa-1', tags=tags)

    assert 'tags' not in payloads[0]


@pytest.mark.asyncio
async def test_manager_add_account_forwards_tags() -> None:
    manager = AccountPoolManager('http://account-pool.test')
    manager._client.add_account = AsyncMock(return_value=42)  # type: ignore[method-assign]

    account_id = await manager.add_account('gui-sandbox', 'ags-waa-1', tags=['ags', 'waa'])

    assert account_id == 42
    manager._client.add_account.assert_awaited_once_with(
        'gui-sandbox',
        'ags-waa-1',
        credentials=None,
        tokens=None,
        shared_cap=0,
        note='',
        tags=['ags', 'waa'],
    )
