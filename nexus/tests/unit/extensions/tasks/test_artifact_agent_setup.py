import pytest

from nexus.extensions.tasks.artifact import config
from nexus.extensions.tasks.artifact.agent_setup import claude_hooks, harness
from nexus.extensions.tasks.artifact.agent_setup.claude_config import default_judge_claude_input
from nexus.extensions.tasks.artifact.agent_setup.clients import build_client_spec


# Both the primary and legacy name of every hook knob, so a developer's exported value cannot
# decide the outcome of these tests.
_HOOK_ENV_NAMES = (
    'ARTIFACT_CC_HOOKS',
    'ARTIFACT_STOP_HOOK',
    'NEXUS_STOP_HOOK_ENABLE',
    'ARTIFACT_STOP_HOOK_PERSONA',
    'NEXUS_STOP_HOOK_PERSONA',
    'ARTIFACT_STOP_HOOK_MODEL',
    'NEXUS_STOP_HOOK_MODEL',
    'ARTIFACT_STOP_HOOK_MAX_REPLIES',
    'NEXUS_STOP_HOOK_MAX_REPLIES',
    'ARTIFACT_STOP_HOOK_ITERATIVE_TURNS',
    'NEXUS_STOP_HOOK_ITERATIVE_TURNS',
    'ARTIFACT_STOP_HOOK_SHOW_CODE',
    'NEXUS_STOP_HOOK_SHOW_CODE',
    'ARTIFACT_ASKUSER_HOOK_VERSION',
    'NEXUS_ASKUSER_HOOK_VERSION',
    'ARTIFACT_ASKUSER_HOOK_DISABLE_LLM',
    'NEXUS_ASKUSER_HOOK_DISABLE_LLM',
    'ARTIFACT_ASKUSER_HOOK_MODEL',
    'NEXUS_ASKUSER_HOOK_MODEL',
)

_MINIMAL_SPEC = {'working_dir': '/workspace', 'timeout': 1800.0}


@pytest.fixture
def hook_env(monkeypatch: pytest.MonkeyPatch):
    for name in _HOOK_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    config.reset()
    yield monkeypatch
    config.reset()


def _build_with_hooks(monkeypatch: pytest.MonkeyPatch, **env: str) -> dict:
    """Build a claude_code agent_config with hooks on, plus any extra knobs.

    ``hooks_enabled`` and the hook knobs are config-driven (not HarnessSpec fields), so a test
    that wants them has to set env and reset the cache before building.
    """
    monkeypatch.setenv('ARTIFACT_CC_HOOKS', '1')
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config.reset()
    return harness.build_infer_agent_config('claude_code', harness.HarnessSpec(**_MINIMAL_SPEC))


def test_stop_hook_off_by_default(hook_env) -> None:
    """Default: no Stop entry and no 1109-line script shipped into the container."""
    agent_config = _build_with_hooks(hook_env)

    hooks = agent_config['config_file_patches'][0]['content']['hooks']
    assert 'Stop' not in hooks
    writer = next(s for s in agent_config['pre_start_scripts'] if s['interpreter'] == 'python')
    assert '_nexus_stop_hook' not in writer['code']
    assert 'NEXUS_STOP_HOOK_ENABLE' not in agent_config['extra_env']


def test_stop_hook_on_installs_script_and_enable_env(hook_env) -> None:
    """The switch has to do both halves: install the script AND enable it.

    The hook used to be unreachable because only the first half was wired — the script sat in
    /tmp while its own NEXUS_STOP_HOOK_ENABLE gate stayed false, so every Stop was a no-op.
    """
    agent_config = _build_with_hooks(
        hook_env,
        ARTIFACT_STOP_HOOK='1',
        ARTIFACT_STOP_HOOK_PERSONA='iterative_user',
        ARTIFACT_STOP_HOOK_MAX_REPLIES='12',
    )

    hooks = agent_config['config_file_patches'][0]['content']['hooks']
    assert hooks['Stop'] == [{'hooks': [{'type': 'command', 'command': 'python3 /tmp/_nexus_stop_hook.py'}]}]
    writer = next(s for s in agent_config['pre_start_scripts'] if s['interpreter'] == 'python')
    assert '_nexus_stop_hook' in writer['code']
    assert agent_config['extra_env']['NEXUS_STOP_HOOK_ENABLE'] == 'true'
    assert agent_config['extra_env']['NEXUS_STOP_HOOK_PERSONA'] == 'iterative_user'
    assert agent_config['extra_env']['NEXUS_STOP_HOOK_MAX_REPLIES'] == '12'


def test_unset_stop_hook_knobs_are_not_emitted(hook_env) -> None:
    """An unset knob must not be emitted, so the script's own default stays authoritative."""
    hook_env.setenv('ARTIFACT_STOP_HOOK', '1')
    config.reset()

    env = claude_hooks.build_hook_env()

    assert env == {'NEXUS_STOP_HOOK_ENABLE': 'true'}


def test_askuser_hook_version_selects_the_script_file(hook_env) -> None:
    """v1 and v2 are separate files; the version switch picks which one is injected."""
    hook_env.setenv('ARTIFACT_ASKUSER_HOOK_VERSION', 'v1')
    config.reset()
    v1_code = claude_hooks.build_askuser_hook_writer()['code']

    hook_env.setenv('ARTIFACT_ASKUSER_HOOK_VERSION', 'v2')
    config.reset()
    v2_code = claude_hooks.build_askuser_hook_writer()['code']

    assert v1_code != v2_code
    # v2 calls the proxy over urllib; v1 is a pure-stdlib first-option pick.
    assert 'urllib' in _decode_written_script(v2_code)
    assert 'urllib' not in _decode_written_script(v1_code)


def test_askuser_disable_llm_keeps_v2_but_switches_off_the_call(hook_env) -> None:
    """disable_llm is not the same switch as version=v1: v2 still ships, it just skips the call."""
    hook_env.setenv('ARTIFACT_ASKUSER_HOOK_DISABLE_LLM', '1')
    config.reset()

    assert claude_hooks.build_hook_env()['NEXUS_ASKUSER_HOOK_DISABLE_LLM'] == '1'
    assert 'urllib' in _decode_written_script(claude_hooks.build_askuser_hook_writer()['code'])


def _decode_written_script(writer_code: str) -> str:
    """Pull the base64 payload back out of a pre_start_scripts writer."""
    import base64
    import re

    return base64.b64decode(re.search(r"b64decode\('([^']+)'\)", writer_code).group(1)).decode()


def test_gemini_client_requests_plaintext_thinking():
    spec = build_client_spec(
        'gemini',
        model='openai/google/gemini-3.5-flash',
        base_url='http://example.test',
        api_key='test-key',
        effort='low',
    )

    request_args = spec['config']['request_args']
    assert request_args['model'] == 'gemini/gemini-3.5-flash'
    assert request_args['reasoning_effort'] == 'low'


def test_gemini_client_defaults_to_low_reasoning_effort():
    spec = build_client_spec(
        'gemini',
        model='gemini-3.5-flash',
        base_url='http://example.test',
        api_key='test-key',
    )

    assert spec['config']['request_args']['reasoning_effort'] == 'low'


def test_judge_claude_input_preserves_structured_thinking_blocks():
    claude_input = default_judge_claude_input(with_playwright=False)

    assert claude_input.thinking_preset == 'bedrock-claude'
