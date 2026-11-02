"""Guard the infer agent_config against silent drift during the harness extraction.

The golden files in ``golden/`` were frozen from generator.py's inline builders before
any harness code moved (see ``docs/developing/plan-artifact-harness-refactor.md`` §8.1 step 0).
These tests assert the current code still produces them key-for-key.

Why golden rather than hand-written expectations: a moved builder that forgets one dict entry
raises nothing, passes ruff, and only surfaces as odd agent behaviour inside a sandbox 10
minutes into a rollout. ``==`` on the whole dict is the cheapest thing that catches it.

Everything routes through ``_build_agent_config`` below, so the suite followed the
implementation into ``agent_setup/harness/`` by changing that one helper.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import pytest


GOLDEN_DIR = Path(__file__).parent / 'golden'

# The goldens were frozen when every knob was a HarnessSpec argument. The builders now read
# the global ones from config, so each combo below says which *layer* it drives:
#   spec   -> per-row values still passed through HarnessSpec
#   config -> ARTIFACT_* env, set for the duration of one test
# The frozen golden JSON is unchanged; only how the same inputs are supplied moved.
BASE_SPEC: dict[str, Any] = {
    'working_dir': '/workspace',
    'timeout': 1800.0,
    'thinking_preset': 'hunyuan',
    'extra_pre_start_scripts': [],
    'mcp_servers': [],
}

COMBOS: dict[str, dict[str, dict[str, Any]]] = {
    'plain': {},
    'mcp': {'spec': {'mcp_servers': [{'name': 'playwright', 'transport': 'stdio'}]}},
    'prestart': {'spec': {'extra_pre_start_scripts': [{'interpreter': 'bash', 'timeout': 240.0, 'code': 'echo diag'}]}},
    'router': {
        'config': {
            'ARTIFACT_LLM_ROUTER_ENABLE': '1',
            'ARTIFACT_NO_THINKING_MODEL': 'bedrock/x',
            'ARTIFACT_HAIKU_MODEL': 'bedrock/haiku',
        }
    },
    'bedrock_preset': {'spec': {'thinking_preset': 'bedrock-claude'}},
    'hooks': {'config': {'ARTIFACT_CC_HOOKS': '1'}},
}

# Keys the RL runner injects into agent_config after the builder has run
# (``nexus/runner/hytuner.py`` setdefault calls). A builder that emits any of these would
# silently win over the runner, because setdefault does not overwrite.
RUNNER_OWNED_KEYS = frozenset({
    'wrap_failed_tool_response_as_tool_error',
    'max_tool_response_tokens',
})

CASES = [(agent, combo) for agent in ('claude_code', 'hycode', 'hycode_dynamicagent') for combo in COMBOS]


# Fields HarnessSpec takes as tuples where the frozen golden was built from lists.
_TUPLE_FIELDS = ('mcp_servers', 'extra_pre_start_scripts', 'plugin_names', 'skills_dirs')

# Config-backed knobs the builders read directly (hook switches, hycode flags). The goldens
# were frozen at their defaults, so a developer's exported ARTIFACT_*/NEXUS_* would otherwise
# change the output and fail these tests for the wrong reason.
_CONFIG_ENV_NAMES = (
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
    'ARTIFACT_HYCODE_VERSION',
    'HYCODE_PACKAGE_VERSION',
    'ARTIFACT_HYCODE_NPM_REGISTRY',
    'HYCODE_NPM_REGISTRY',
    'ARTIFACT_HYCODE_THINKING_PRESET',
    'HYCODE_THINKING_PRESET',
    'ARTIFACT_HYCODE_ALLOW_NATIVE_WEB',
    'HYCODE_ALLOW_NATIVE_WEB',
    'ARTIFACT_HYCODE_ALLOW_PLAN_MODE',
    'ARTIFACT_HYCODE_EXTRA_CLI_ARGS',
    'ARTIFACT_DYNAMICAGENT_VERSION',
    'ARTIFACT_DYNAMICAGENT_NPM_REGISTRY',
    'ARTIFACT_DYNAMICAGENT_THINKING_PRESET',
    'ARTIFACT_DYNAMICAGENT_ALLOW_NATIVE_WEB',
    'ARTIFACT_DYNAMICAGENT_EXTRA_CLI_ARGS',
    'ARTIFACT_DYNAMICAGENT_DEFAULT_PRESET',
    'ARTIFACT_DYNAMICAGENT_BOOTSTRAP_TEMPLATES',
    'ARTIFACT_DYNAMICAGENT_TEE_LOG_DIR',
    'IMAGE_GEN_APP_ID',
    'IMAGE_GEN_APP_KEY',
)


@pytest.fixture(autouse=True)
def _default_config(monkeypatch: pytest.MonkeyPatch):
    """Pin the config to its built-in defaults for every test in this module."""
    from nexus.extensions.tasks.artifact import config

    for name in _CONFIG_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    config.reset()
    yield
    config.reset()


def _build_agent_config(agent_name: str, combo_id: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build one agent_config through the harness registry.

    Applies the combo's ``config`` half as env before building, since the builders read those
    knobs from :func:`config.get` rather than taking them as arguments.
    """
    from nexus.extensions.tasks.artifact import config
    from nexus.extensions.tasks.artifact.agent_setup import harness

    combo = COMBOS[combo_id]
    for name, value in combo.get('config', {}).items():
        monkeypatch.setenv(name, value)
    config.reset()

    kwargs = {**BASE_SPEC, **combo.get('spec', {})}
    spec_kwargs = {k: (tuple(v) if k in _TUPLE_FIELDS else v) for k, v in kwargs.items()}
    return harness.build_infer_agent_config(agent_name, harness.HarnessSpec(**spec_kwargs))


def _load_golden(agent_name: str, combo_id: str) -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / f'{agent_name}__{combo_id}.json').read_text())


@pytest.mark.non_e2e
@pytest.mark.parametrize(('agent_name', 'combo_id'), CASES)
def test_agent_config_matches_golden(agent_name: str, combo_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    got = _build_agent_config(agent_name, combo_id, monkeypatch)
    want = _load_golden(agent_name, combo_id)
    # Report the symmetric key difference first: a missing/extra key is the common failure and
    # is far easier to read than a full dict diff.
    assert set(got) == set(want), f'key drift: {sorted(set(got) ^ set(want))}'
    assert got == want


@pytest.mark.non_e2e
@pytest.mark.parametrize(('agent_name', 'combo_id'), CASES)
def test_agent_config_is_json_serializable(
    agent_name: str, combo_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # parquet rows carry this dict, so a non-JSON value makes it unusable for RL.
    json.dumps(_build_agent_config(agent_name, combo_id, monkeypatch))


@pytest.mark.non_e2e
@pytest.mark.parametrize(('agent_name', 'combo_id'), CASES)
def test_agent_config_leaves_runner_owned_keys_alone(
    agent_name: str, combo_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    got = _build_agent_config(agent_name, combo_id, monkeypatch)
    assert not set(got) & RUNNER_OWNED_KEYS, f'builder claims runner keys: {sorted(set(got) & RUNNER_OWNED_KEYS)}'


@pytest.mark.non_e2e
@pytest.mark.parametrize('combo_id', list(COMBOS))
def test_hycode_config_keeps_every_key_hycode_accepts(combo_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """HyCodeInput has pydantic extra='ignore', so unknown keys vanish without an error.

    Passing claude_code-shaped config to hycode is exactly how mcp_servers got silently
    dropped during the initial integration; assert the builder never relies on that.
    """
    from nexus.extensions.agents.hycode.agent import HyCodeInput

    got = _build_agent_config('hycode', combo_id, monkeypatch)
    HyCodeInput(user_prompt='x', **got)
    dropped = set(got) - set(HyCodeInput.model_fields)
    assert not dropped, f'HyCodeInput would silently drop: {sorted(dropped)}'


@pytest.mark.non_e2e
@pytest.mark.parametrize('combo_id', list(COMBOS))
def test_dynamicagent_config_keeps_every_key(combo_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same guard as the hycode variant — no silent key drops."""
    from nexus.extensions.agents.hycode_dynamicagent.agent import HyCodeDynamicAgentInput

    got = _build_agent_config('hycode_dynamicagent', combo_id, monkeypatch)
    HyCodeDynamicAgentInput(user_prompt='x', **got)
    dropped = set(got) - set(HyCodeDynamicAgentInput.model_fields)
    assert not dropped, f'HyCodeDynamicAgentInput would silently drop: {sorted(dropped)}'


@pytest.mark.non_e2e
def test_hycode_cli_flags_cover_the_hycode_agents_own_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every flag ``agents/hycode`` ships must survive, in order, as a prefix of ours.

    A drifting --output-format (json vs stream-json) or a missing --no-session-persistence
    changes how the CLI reports back, and the failure shows up as an empty archive rather
    than an error.

    Not equality: artifact additionally denies EnterPlanMode and pre-authorizes ExitPlanMode.
    """
    from nexus.extensions.agents.hycode.generator import _extra_cli_args

    got = _build_agent_config('hycode', 'plain', monkeypatch)['extra_cli_args']
    upstream = _extra_cli_args()
    assert got[: len(upstream)] == upstream, 'upstream hycode flags changed or were reordered'
    assert set(got) - set(upstream) == {'--allowedTools', 'EnterPlanMode', 'ExitPlanMode'}


@pytest.mark.non_e2e
def test_plan_mode_exit_is_never_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    """ExitPlanMode must stay allowed for every harness and every plan-mode setting.

    Denying the exit alongside the entry is the one combination that hangs a rollout: whatever
    still puts the agent in plan mode leaves it with no way out, and it waits for an approval
    that never comes until the timeout kills it.
    """

    def values_after(args: list[str], flag: str) -> list[str]:
        """Values belonging to *flag* — up to the next flag, not to the end of the list."""
        if flag not in args:
            return []
        out = []
        for value in args[args.index(flag) + 1 :]:
            if value.startswith('--'):
                break
            out.append(value)
        return out

    for agent_name in ('hycode', 'hycode_dynamicagent'):
        for allow_plan in ('0', '1'):
            monkeypatch.setenv('ARTIFACT_HYCODE_ALLOW_PLAN_MODE', allow_plan)
            args = _build_agent_config(agent_name, 'plain', monkeypatch)['extra_cli_args']
            where = f'{agent_name} (allow_plan={allow_plan})'
            assert 'ExitPlanMode' in values_after(args, '--allowedTools'), f'{where} does not allow the exit'
            assert 'ExitPlanMode' not in values_after(args, '--disallowedTools'), f'{where} denies the exit'


@pytest.mark.non_e2e
def test_claude_code_keeps_mcp_and_playwright_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    """claude_code's browser access rides on these three; hycode has none of them."""
    got = _build_agent_config('claude_code', 'mcp', monkeypatch)
    assert got['mcp_servers'], 'mcp_servers dropped'
    assert 'PLAYWRIGHT_MCP_CONFIG' in got['extra_env']
    assert got['pre_start_scripts'], 'playwright config pre-start dropped'


@pytest.mark.non_e2e
def test_unknown_harness_is_rejected() -> None:
    """Falling back to a default harness hides a config typo until the sandbox misbehaves.

    Asserted through the registry, which is now the only path -- generator.py no longer has a
    builder to fall through to.
    """
    from nexus.extensions.tasks.artifact.agent_setup import harness

    with pytest.raises(ValueError, match='Unsupported harness'):
        harness.build_infer_agent_config('does_not_exist', harness.HarnessSpec())


@pytest.mark.non_e2e
def test_registry_covers_both_harnesses() -> None:
    from nexus.extensions.tasks.artifact.agent_setup import harness

    assert set(harness.HARNESS_BUILDERS) == {'claude_code', 'hycode', 'hycode_dynamicagent'}
