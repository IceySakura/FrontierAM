import pytest

from nexus.extensions.tasks.waa.getters import edge


class FakeRuntime:
    def __init__(self, output: str = ''):
        self.output = output

    async def execute(self, *args, **kwargs):
        return {'output': self.output}


@pytest.mark.asyncio
async def test_edge_safety_pref_legacy(monkeypatch):
    async def fake_load_preferences(runtime):
        return {'edge': {'super_duper_secure_mode': {'enabled': True}}}

    monkeypatch.setattr(edge, '_load_edge_preferences', fake_load_preferences)

    score = await edge.get_enable_enhanced_safety_browsing_from_edge(FakeRuntime(''), {})

    assert score == 1.0


@pytest.mark.asyncio
async def test_edge_safety_pref_missing_returns_zero(monkeypatch):
    async def fake_load_preferences(runtime):
        return {'edge': {}}

    monkeypatch.setattr(edge, '_load_edge_preferences', fake_load_preferences)

    score = await edge.get_enable_enhanced_safety_browsing_from_edge(FakeRuntime(''), {})

    assert score == 0.0


@pytest.mark.asyncio
async def test_edge_safety_honors_smart_screen_policy(monkeypatch):
    async def fail_if_preferences_loaded(runtime):
        raise AssertionError('policy should be checked before Preferences')

    monkeypatch.setattr(edge, '_load_edge_preferences', fail_if_preferences_loaded)

    score = await edge.get_enable_enhanced_safety_browsing_from_edge(FakeRuntime('1'), {})

    assert score == 1.0
