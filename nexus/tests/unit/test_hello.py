"""Sample tests to verify pytest setup."""

import asyncio


def test_hello_sync():
    """Simple synchronous test to verify pytest works."""
    assert 1 + 1 == 2


async def test_hello_async():
    """Simple async test to verify pytest-asyncio works."""
    await asyncio.sleep(0.01)
    assert True


def test_sample_data_fixture(sample_data: dict):
    """Test that fixtures from conftest.py are available."""
    assert sample_data['name'] == 'test'
    assert sample_data['value'] == 42
