"""The swe_cosmos WorkBuddy run part.

The generator no longer builds agent config in Python; ``AGENT_CONFIG`` names a run-part YAML
under ``configs/agents/`` that :func:`load_config` merges. These tests therefore go through the
profile and the agent's own input model, which is what actually consumes it.
"""

from typing import Any

import pytest

from omegaconf import OmegaConf

from nexus.extensions.agents.workbuddy.agent import WorkBuddyInput
from nexus.extensions.tasks.swe.swe_cosmos.load_config import load_config


# Only the agent axis is under test; a stray runtime/LLM profile in the ambient environment would
# merge into the same tree. AGENT_VERSION is the sharp one: the profile falls back to it, so a
# batch that ran another scaffold first can silently pin WorkBuddy to that scaffold's version.
_ISOLATED_ENV = ('RUNTIME_CONFIG', 'LLM_CONFIG', 'AGENT_VERSION', 'WORKBUDDY_VERSION')


@pytest.fixture
def agent_config(monkeypatch: pytest.MonkeyPatch):
    """Resolve ``configs/agents/workbuddy.yaml`` into a plain ``agent_config`` dict."""

    def _load(**env: str) -> dict[str, Any]:
        for name in _ISOLATED_ENV:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv('AGENT_CONFIG', 'workbuddy')
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        config = load_config()
        assert config['agent_name'] == 'workbuddy'
        # load_config leaves ${oc.*} unresolved; RunInput.from_dict would normally do this.
        resolved = OmegaConf.to_container(OmegaConf.create(config['agent_config']), resolve=True)
        assert isinstance(resolved, dict)
        return resolved

    return _load


def test_workbuddy_profile_imports_the_agent_module(agent_config) -> None:
    agent_config()
    config = load_config()
    assert 'nexus.extensions.agents.workbuddy' in config['imports']


def test_agent_config_forwards_workbuddy_tarball(agent_config) -> None:
    cfg = agent_config(
        AGENT_WORKING_DIR='/testbed',
        AGENT_TIMEOUT='14400',
        NPM_REGISTRY='http://mirrors.tencent.com/npm/',
        WORKBUDDY_TARBALL='/data/wb-5.7.0-linux-x64.tar',
        WORKBUDDY_EFFORT='xhigh',
        WORKBUDDY_PRESET='code',
    )
    assert cfg['working_dir'] == '/testbed'
    assert cfg['timeout'] == 14400
    assert cfg['npm_registry'] == 'http://mirrors.tencent.com/npm/'
    assert cfg['tarball_path'] == '/data/wb-5.7.0-linux-x64.tar'
    assert cfg['effort'] == 'xhigh'
    assert cfg['preset'] == 'code'


def test_agent_config_validates_as_workbuddy_input(agent_config) -> None:
    model = WorkBuddyInput(user_prompt='fix it', **agent_config())
    assert model.version == '5.9.0'
    assert model.timeout == 14400.0


def test_headless_defaults_survive_resolution(agent_config) -> None:
    """The three settings that a headless run is wrong without.

    ``fullAccess`` plus the compat patch because nobody answers a permission prompt; ``SHELL``
    because the Bash tool otherwise falls back to dash and mangles ``${PIPESTATUS[0]}``.
    """
    model = WorkBuddyInput(user_prompt='fix it', **agent_config())
    assert model.permission_mode == 'fullAccess'
    assert model.compat_permission_mode_patch is True
    assert (model.extra_env or {})['SHELL'] == '/bin/bash'


def test_cloud_memory_polling_is_disabled(agent_config) -> None:
    """wb reads generateMemoryEnabled from app-config.json, not settings.json, and only as a
    JSON boolean. Getting either wrong puts it back to polling the cloud every 20s on a token
    it cannot use."""
    model = WorkBuddyInput(user_prompt='fix it', **agent_config())
    patches = {patch.path: patch.content for patch in model.config_file_patches}
    assert patches['~/.workbuddy/app/app-config.json'] == {'generateMemoryEnabled': False}
