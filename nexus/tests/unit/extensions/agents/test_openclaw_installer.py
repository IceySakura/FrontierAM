import pytest

from nexus.extensions.agents.openclaw import installer


async def test_openclaw_installer_requires_node_22_19(monkeypatch) -> None:
    async def fake_run_checked_command(*args, **kwargs):
        return 'v22.18.0'

    monkeypatch.setattr(installer, '_run_checked_command', fake_run_checked_command)

    with pytest.raises(RuntimeError, match='Node >= 22.19.0'):
        await installer.ensure_supported_node_version()


async def test_openclaw_installer_accepts_node_22_19(monkeypatch) -> None:
    async def fake_run_checked_command(*args, **kwargs):
        return 'v22.19.0'

    monkeypatch.setattr(installer, '_run_checked_command', fake_run_checked_command)

    assert await installer.ensure_supported_node_version() == 'v22.19.0'
