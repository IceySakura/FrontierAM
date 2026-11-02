from types import SimpleNamespace

import pytest

from nexus.core.agent import agent_registry, get_agent_gateway_scaffold_policy


def test_get_agent_gateway_scaffold_policy_reads_registry_metadata(monkeypatch):
    agent = SimpleNamespace(nexus_meta={'meta': {'gateway': {'scaffold_policy': 'test_policy'}}})
    monkeypatch.setattr(agent_registry, 'get', lambda name: agent)

    assert get_agent_gateway_scaffold_policy('test_agent') == 'test_policy'


def test_get_agent_gateway_scaffold_policy_prefers_explicit_policy(monkeypatch):
    agent = SimpleNamespace(nexus_meta={'meta': {'gateway': {'scaffold_policy': 'metadata_policy'}}})
    monkeypatch.setattr(agent_registry, 'get', lambda name: agent)

    assert get_agent_gateway_scaffold_policy('test_agent', 'explicit_policy') == 'explicit_policy'


def test_get_agent_gateway_scaffold_policy_auto_uses_registry_metadata(monkeypatch):
    agent = SimpleNamespace(nexus_meta={'meta': {'gateway': {'scaffold_policy': 'metadata_policy'}}})
    monkeypatch.setattr(agent_registry, 'get', lambda name: agent)

    assert get_agent_gateway_scaffold_policy('test_agent', 'auto') == 'metadata_policy'


def test_single_turn_declares_nexus_main_agent_gateway_policy():
    import nexus.extensions.agents.single_turn.agent  # noqa: F401

    agent = agent_registry.get('single_turn')
    assert agent.nexus_meta['meta']['gateway']['scaffold_policy'] == 'nexus_main_agent'


def test_calculator_declares_nexus_main_agent_gateway_policy():
    import nexus.extensions.projects.calculator.agent  # noqa: F401

    agent = agent_registry.get('calculator_agent')
    assert agent.nexus_meta['meta']['gateway']['scaffold_policy'] == 'nexus_main_agent'


@pytest.mark.parametrize(
    ('agent_name', 'nexus_meta'),
    [
        (None, None),
        ('test_agent', {}),
        ('test_agent', {'meta': {}}),
        ('test_agent', {'meta': {'gateway': {}}}),
        ('test_agent', {'meta': {'gateway': {'scaffold_policy': ''}}}),
    ],
)
def test_get_agent_gateway_scaffold_policy_raises_by_default_when_policy_is_missing(
    monkeypatch,
    agent_name,
    nexus_meta,
):
    monkeypatch.delenv('NEXUS_GATEWAY_USE_GENERIC_POLICY', raising=False)
    if agent_name is not None:
        monkeypatch.setattr(agent_registry, 'get', lambda name: SimpleNamespace(nexus_meta=nexus_meta))

    with pytest.raises(ValueError, match='NEXUS_GATEWAY_USE_GENERIC_POLICY=true'):
        get_agent_gateway_scaffold_policy(agent_name)


def test_get_agent_gateway_scaffold_policy_falls_back_to_generic_when_error_is_disabled(
    monkeypatch,
    caplog,
):
    agent = SimpleNamespace(nexus_meta={})
    monkeypatch.setattr(agent_registry, 'get', lambda name: agent)
    monkeypatch.setenv('NEXUS_GATEWAY_USE_GENERIC_POLICY', ' TRUE ')

    with caplog.at_level('WARNING'):
        assert get_agent_gateway_scaffold_policy('test_agent') == 'generic'

    assert 'falling back to generic' in caplog.text.lower()
