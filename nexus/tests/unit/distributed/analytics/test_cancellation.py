from unittest.mock import AsyncMock, MagicMock

import pytest

from sqlalchemy.exc import OperationalError

from nexus.distributed.analytics.reader import BusinessStatisticsReader
from nexus.distributed.config.observability import BusinessStatisticsConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('info', 'kill'),
    [
        ('SELECT /* nexus_statistics:owned */ SLEEP(20)', True),
        ('SELECT /* nexus_statistics:another_request */ SLEEP(20)', False),
        ('UPDATE table_name SET value=1 /* nexus_statistics:owned */', False),
        (None, False),
    ],
)
async def test_cancel_only_stops_the_marked_select_on_the_captured_connection(info, kill):
    reader = object.__new__(BusinessStatisticsReader)
    reader.config = BusinessStatisticsConfig()
    session = MagicMock()
    session.scalar = AsyncMock(return_value=info)
    session.execute = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock()
    reader.factory = MagicMock(return_value=session)
    await reader._stop_query(123, 'owned')
    assert session.scalar.await_args.args[1] == {'id': 123}
    assert session.execute.await_count == int(kill)
    if kill:
        assert str(session.execute.await_args.args[0]) == 'KILL QUERY 123'


@pytest.mark.asyncio
async def test_cancel_tolerates_the_owned_connection_finishing_before_kill():
    reader = object.__new__(BusinessStatisticsReader)
    reader.config = BusinessStatisticsConfig()
    session = MagicMock()
    session.scalar = AsyncMock(return_value='SELECT /* nexus_statistics:owned */ SLEEP(20)')
    session.execute = AsyncMock(side_effect=OperationalError('KILL QUERY 123', {}, Exception(1094, 'Unknown thread id')))
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock()
    reader.factory = MagicMock(return_value=session)
    await reader._stop_query(123, 'owned')
