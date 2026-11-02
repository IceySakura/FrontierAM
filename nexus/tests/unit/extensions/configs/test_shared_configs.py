"""The shared run-part layer under ``nexus/extensions/configs``.

Every test goes through :func:`load_config` or a generator template, which is how a task consumes these
files. ``${oc.*}`` stays unresolved after merging; tests resolve it with OmegaConf the way ``RunInput.from_dict``
would.
"""

import importlib
import json
import re

from pathlib import Path
from typing import Any

import pytest

from omegaconf import OmegaConf

from nexus.extensions.configs import env_table, validate
from nexus.extensions.configs.load_config import SHARED_DIR, load_config


_AXIS_ENV = ('RUNTIME_CONFIG', 'AGENT_CONFIG', 'LLM_CONFIG')
# Read by many files; an ambient value from another run would change what the tests see.
_AMBIENT_ENV = (
    'AGENT_VERSION',
    'WORKBUDDY_VERSION',
    'WORKBUDDY_ROLLOUT_MODE',
    'AGENT_WORKING_DIR',
    'AGENT_TIMEOUT',
    'THINKING_EFFORT',
    'LANGFUSE_TRACE_NAME',
    'RUNNER_TIMEOUT',
    'LANGFUSE_PUBLIC_KEY',
    'LANGFUSE_SECRET_KEY',
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    """Clear the axis and ambient variables, then set the given ones."""

    def _set(**values: str) -> None:
        for name in (*_AXIS_ENV, *_AMBIENT_ENV):
            monkeypatch.delenv(name, raising=False)
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    return _set


def _resolve(node: Any) -> Any:
    return OmegaConf.to_container(OmegaConf.create(node), resolve=True)


def test_task_dir_wins_over_shared_dir(env, tmp_path: Path) -> None:
    (tmp_path / 'configs' / 'agents').mkdir(parents=True)
    (tmp_path / 'configs' / 'agents' / 'kilo.yaml').write_text(
        'body:\n  agent_name: kilo\n  agent_config:\n    version: task-local\nmerge: {}\nsource: test\n'
    )
    env(AGENT_CONFIG='kilo')
    assert load_config(task_dir=tmp_path)['agent_config'] == {'version': 'task-local'}
    assert load_config()['agent_config']['version'] != 'task-local'


def test_missing_name_lists_both_search_dirs(env, tmp_path: Path) -> None:
    env(AGENT_CONFIG='no_such_agent')
    with pytest.raises(FileNotFoundError) as exc:
        load_config(task_dir=tmp_path)
    message = str(exc.value)
    assert message.index(str(tmp_path / 'configs' / 'agents')) < message.index(str(SHARED_DIR / 'agents'))


def test_the_scaffold_folds_the_family_section_of_the_selected_llm(env) -> None:
    env(AGENT_CONFIG='claude_code_eval', LLM_CONFIG='claude-opus-48-aws_third-aihub2.0.litellm')
    config = load_config()
    agent_config = config['agent_config']
    # The llm file states the fact; claude_code_eval's `claude:` section decides to use it.
    assert agent_config['extra_env']['ANTHROPIC_MODEL'] == 'claude-opus-4-8[1m]'
    assert agent_config['thinking_preset'] == 'bedrock-claude'
    assert 'version' in agent_config  # the agent file's own keys survive the fold
    # Both helper keys are stripped: neither is a RunInput field.
    assert 'llm' not in config
    assert 'agent_config_by_llm' not in config


def test_a_scaffold_takes_nothing_from_a_family_it_says_nothing_about(env) -> None:
    # kilo has no `gpt:` section, so a GPT wire leaves its own agent_config untouched.
    env(AGENT_CONFIG='kilo', LLM_CONFIG='gpt-6-sol.openai_responses')
    agent_config = load_config()['agent_config']
    assert 'thinking_preset' not in agent_config
    assert 'ANTHROPIC_MODEL' not in (agent_config.get('extra_env') or {})


def test_an_optional_fact_the_wire_lacks_leaves_the_scaffold_default_standing(env) -> None:
    # hy4-preview-offline declares no no_thinking_request_args; handy asks for it with `?`.
    env(AGENT_CONFIG='handy', LLM_CONFIG='hy4-preview-offline.openai')
    agent_config = load_config()['agent_config']
    assert agent_config['thinking_preset'] == 'hunyuan'
    assert 'no_thinking_request_args' not in agent_config


def test_a_flat_llm_ref_resolves_from_the_whole_agent_config_not_just_the_family_section(env) -> None:
    # codex's reasoning_effort is a flat top-level agent_config key holding `${llm:...}`, not a
    # per-family entry -- it must still resolve. Before the flat pass in _apply_llm_family it
    # stayed as literal `${llm:reasoning_effort?}` text (the old family-section-only limitation).
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG='gpt-6-sol.openai_responses', HR_API_KEY='x')
    agent_config = load_config()['agent_config']
    assert agent_config['reasoning_effort'] == '${oc.env:REASONING_EFFORT,xhigh}'
    assert 'reasoning_effort' in agent_config  # resolved via the flat pass, not left literal


def test_a_flat_optional_llm_ref_is_omitted_when_the_wire_lacks_the_fact(env, tmp_path: Path) -> None:
    # A wire that declares no reasoning_effort fact must drop the key (not leave `${llm:...}`
    # text), so codex's own default_reasoning_level stands rather than a literal string.
    (tmp_path / 'configs' / 'llms').mkdir(parents=True)
    (tmp_path / 'configs' / 'llms' / 'no_effort.yaml').write_text(
        'body:\n'
        '  imports: [nexus.extensions.openai_clients.litellm]\n'
        '  llm: {family: claude, anthropic_model: claude-opus-5}\n'
        '  openai_clients:\n'
        '    main:\n'
        '      name: litellm\n'
        '      config:\n'
        "        client_args: {api_key: 'x'}\n"
        "        request_args: {model: 'anthropic/claude-opus-5'}\n"
        'merge: {imports: append_unique}\n'
        "source: 'test fixture'\n"
    )
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG='no_effort')
    agent_config = load_config(task_dir=tmp_path)['agent_config']
    assert 'reasoning_effort' not in agent_config


def test_a_key_no_scaffold_reads_fails_at_load_time(env, tmp_path: Path) -> None:
    # The whole point of the fold: pydantic inputs are extra='ignore', so this used to vanish.
    bad = tmp_path / 'bad.yaml'
    bad.write_text(
        'body:\n'
        '  imports: [nexus.extensions.agents.kilo]\n'
        '  agent_name: kilo\n'
        '  agent_config: {thinking_preset: bedrock-claude}\n'
        'merge: {imports: append_unique}\n'
        "source: 'test fixture'\n"
    )
    env(AGENT_CONFIG=str(bad))
    with pytest.raises(ValueError, match='dropped silently'):
        load_config()


@pytest.mark.parametrize('family', ['clawd', 'GPT'])
def test_an_unknown_family_is_rejected_on_the_scaffold_side(env, tmp_path: Path, family: str) -> None:
    bad = tmp_path / 'bad.yaml'
    bad.write_text(
        'body:\n'
        '  imports: [nexus.extensions.agents.kilo]\n'
        '  agent_name: kilo\n'
        f'  agent_config_by_llm: {{{family}: {{extra_env: {{A: b}}}}}}\n'
        'merge: {imports: append_unique}\n'
        "source: 'test fixture'\n"
    )
    env(AGENT_CONFIG=str(bad))
    with pytest.raises(ValueError, match='unknown families'):
        load_config()


@pytest.mark.parametrize(
    'path',
    sorted(p for axis in validate.AXES for p in (SHARED_DIR / axis).glob('*.yaml')),
    ids=lambda p: f'{p.parent.name}/{p.name}',
)
def test_every_shared_file_follows_the_schema(path: Path) -> None:
    assert validate.check_file(path) == []


def test_validator_rejects_a_key_the_agent_would_silently_drop(tmp_path: Path) -> None:
    (tmp_path / 'agents').mkdir()
    bad = tmp_path / 'agents' / 'kilo_typo.yaml'
    text = (SHARED_DIR / 'agents' / 'kilo.yaml').read_text().replace('    timeout:', '    timeoutt:')
    bad.write_text(text)
    errors = validate.check_file(bad)
    assert any('timeoutt' in e for e in errors)


def test_readme_env_table_is_current() -> None:
    readme = (SHARED_DIR / 'README.md').read_text()
    region = readme.split(env_table.START, 1)[1].split(env_table.END, 1)[0].strip()
    assert region == env_table.render(), 'run `python nexus/extensions/configs/env_table.py --write`'


def test_env_table_lists_every_variable_the_yaml_reads() -> None:
    """Independent oracle: a plain grep for names, not ``env_table``'s own scanner.

    ``test_readme_env_table_is_current`` compares the README against ``render()``, so both
    sides share one scanner and a variable it cannot see is missing from both and the
    comparison still passes. That hid five variables -- four of them credentials -- whose
    ``oc.env`` sits inside another default (``${oc.env:A,${oc.env:B,...}}``) or spans lines.
    """
    expected: set[str] = set()
    for axis in env_table.AXES:
        for path in sorted((SHARED_DIR / axis).glob('*.yaml')):
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.lstrip().startswith('#'):
                    continue
                expected |= set(re.findall(r'\$\{oc\.env:([A-Z][A-Z0-9_]*)', line))
    listed = set(env_table.collect())
    assert not expected - listed, f'env_table 漏了 {sorted(expected - listed)}'
    assert not listed - expected, f'env_table 多出 {sorted(listed - expected)}'


def test_workbuddy_rollout_mode_defaults_on(env) -> None:
    """Smoke using this profile must pin the WorkBuddy prompt, not fall back to CodeBuddy.

    ``oc.decode`` turns ``1`` and ``0`` into ints. The agent field is a bool, so the
    value has to survive that coercion, which is what the runner actually does.
    """
    from nexus.extensions.agents.workbuddy.agent import WorkBuddyInput

    def enabled() -> bool:
        raw = _resolve(load_config()['agent_config'])['rollout_mode']
        return WorkBuddyInput(user_prompt='x', working_dir='/testbed', rollout_mode=raw).rollout_mode

    env(AGENT_CONFIG='workbuddy')
    assert enabled() is True
    env(AGENT_CONFIG='workbuddy', WORKBUDDY_ROLLOUT_MODE='0')
    assert enabled() is False
    env(AGENT_CONFIG='workbuddy', WORKBUDDY_ROLLOUT_MODE='1')
    assert enabled() is True


def test_workbuddy_permission_mode_is_actually_delivered(env) -> None:
    """``permission_mode`` is inert unless the compat patch is on, so pin both together.

    wb 5.9.0's CLI has no ``--permission-mode``; the only way the value reaches wb is the
    compat patch rewriting the minified bundle. With the patch off, ``fullAccess`` is
    declared but never sent, wb falls back to its own ``bypassPermissions``, and that still
    prompts on HIGH/CRITICAL tools -- which in headless means the pending call is denied and
    the whole turn is cancelled (``taskOutcome=PERMISSION_DENIED``). That is not a loud
    failure; the run just ends early with an empty patch.

    This test used to assert the patch was ``False`` and so froze exactly that bug.
    Production (``nexus/extensions/tasks/swe/swe_cosmos/configs/agents/workbuddy.yaml``)
    has defaulted it to ``true`` ever since 5.9.0 support landed; keep the two in step.
    """
    env(AGENT_CONFIG='workbuddy')
    agent_config = _resolve(load_config()['agent_config'])
    assert agent_config['version'] == '5.9.0'
    assert agent_config['compat_permission_mode_patch'] is True
    assert agent_config['permission_mode'] == 'fullAccess'

    # The pair has to be deliverable for the declared version, not merely present.
    from nexus.extensions.agents.workbuddy.compat import (
        VERIFIED_PERMISSION_MODE_PATCH_VERSIONS,
        build_compatibility_patches,
    )

    assert agent_config['version'] in VERIFIED_PERMISSION_MODE_PATCH_VERSIONS
    patches = build_compatibility_patches(agent_config['version'], agent_config['permission_mode'])
    assert patches['extra_cli_args'] == ['--permission-mode', 'fullAccess']


def test_workbuddy_permission_default_matches_production(env) -> None:
    """Read the default straight out of the production config rather than restating it."""
    import re
    import subprocess

    prod_path = 'nexus/extensions/tasks/swe/swe_cosmos/configs/agents/workbuddy.yaml'
    try:
        prod = subprocess.run(
            ['git', 'show', f'origin/main:{prod_path}'],
            capture_output=True,
            text=True,
            cwd=SHARED_DIR.parents[2],
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env without git
        pytest.skip(f'cannot read production config: {exc}')
    if prod.returncode != 0:  # pragma: no cover - shallow clone or missing remote
        pytest.skip(f'cannot read {prod_path} from origin/main')

    match = re.search(r'WORKBUDDY_COMPAT_PERMISSION_PATCH,(\w+)\}', prod.stdout)
    assert match, f'production config no longer declares the flag the same way:\n{prod.stdout[:400]}'
    env(AGENT_CONFIG='workbuddy')
    ours = _resolve(load_config()['agent_config'])['compat_permission_mode_patch']
    assert ours is (match.group(1) == 'true'), (
        f'production defaults WORKBUDDY_COMPAT_PERMISSION_PATCH to {match.group(1)!r}, ours resolves to {ours!r}'
    )


#: Shared agent profile -> its production counterpart under
#: ``nexus/extensions/tasks/swe/swe_cosmos/configs/agents``. Ours splits some profiles into
#: eval/rollout and codemode/toolmode variants, so several map onto one production file.
_PRODUCTION_AGENT_COUNTERPART = {
    'claude_code_eval': 'claude_code',
    'claude_code_rollout': 'claude_code',
    'codebuddy_eval': 'codebuddy',
    'codebuddy_rollout': 'codebuddy',
    'codex_0.160.0_codemode': 'codex_0.160.0',
    'codex_0.160.0_toolmode': 'codex_0.160.0',
    **{
        n: n
        for n in (
            'cline',
            'codeact',
            'codebuddy_server',
            'dsh',
            'handy',
            'hermes',
            'kilo',
            'mini_swe_agent',
            'openclaw',
            'opencode',
            'pi',
            'swe_agent',
            'terminus2',
            'workbuddy',
        )
    },
}

#: Env-var defaults allowed to differ from production, with the reason. Anything not listed
#: here is a finding: it means a value looks configured but behaves differently than the
#: config that actually shipped.
_DIVERGENCE_ALLOWED = {
    'AGENT_VERSION': 'we pin newer harness versions than the production snapshot',
    'AGENT_TIMEOUT': 'owned by the runtime axis, which is shared across swe and frontend',
    'AGENT_WORKING_DIR': 'frontend does not use /testbed',
}


def test_no_shared_agent_profile_silently_diverges_from_production(env) -> None:
    """Every env default we share with a production profile must agree with it.

    This is a guard for a whole bug class rather than one value. Twice now a default in this
    layer drifted from the config that actually shipped, and in both cases nothing failed --
    the value simply stopped doing anything:

    * ``compat_permission_mode_patch`` false made workbuddy's ``fullAccess`` undeliverable, so
      headless runs were cancelled mid-turn with an empty patch;
    * ``wrap_plaintext_reasoning_content`` flat false stopped claude/hunyuan reasoning from
      being minted into a reasoning item at all.

    Both were found by diffing against production, so do the diff in CI instead.
    """
    import subprocess

    prod_dir = 'nexus/extensions/tasks/swe/swe_cosmos/configs/agents'
    repo = SHARED_DIR.parents[2]

    def production(name: str) -> str | None:
        try:
            r = subprocess.run(
                ['git', 'show', f'origin/main:{prod_dir}/{name}.yaml'],
                capture_output=True,
                text=True,
                cwd=repo,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            pytest.skip(f'cannot read production configs: {exc}')
        return r.stdout if r.returncode == 0 else None

    env_default = re.compile(r'\$\{oc\.env:([A-Z0-9_]+),([^}]*)\}')

    def defaults(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for var, dflt in env_default.findall(text):
            out.setdefault(var, dflt.strip())
        return out

    # Skip only when origin/main itself is unreachable (shallow clone, no remote). Probing a
    # specific file instead would turn "production renamed workbuddy.yaml" into a silent skip
    # that drops the drift check for all 20 profiles -- the exact failure this test exists to
    # prevent. With main reachable, a missing counterpart is a finding, asserted per profile.
    reachable = subprocess.run(
        ['git', 'rev-parse', '--verify', '--quiet', 'origin/main'],
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=30,
    )
    if reachable.returncode != 0:  # pragma: no cover - shallow clone or missing remote
        pytest.skip('origin/main is not available, so production configs cannot be read')

    findings = []
    for ours_name, prod_name in sorted(_PRODUCTION_AGENT_COUNTERPART.items()):
        prod_text = production(prod_name)
        assert prod_text is not None, (
            f'{prod_dir}/{prod_name}.yaml is not on origin/main any more. If production renamed or '
            f'removed it, update _PRODUCTION_AGENT_COUNTERPART -- do not delete this test.'
        )
        ours = defaults((SHARED_DIR / 'agents' / f'{ours_name}.yaml').read_text(encoding='utf-8'))
        theirs = defaults(prod_text)
        for var in sorted(set(ours) & set(theirs)):
            if ours[var] != theirs[var] and var not in _DIVERGENCE_ALLOWED:
                findings.append(f'{ours_name}: {var} prod={theirs[var]!r} ours={ours[var]!r}')

    assert not findings, 'shared profile defaults disagree with the shipped production config:\n  ' + '\n  '.join(
        findings
    )


def test_codebuddy_rollout_passes_each_disallowed_tool_as_one_argument(env) -> None:
    # `--disallowedTools A B` is variadic and swallows the prompt that follows it.
    env(AGENT_CONFIG='codebuddy_rollout')
    assert load_config()['agent_config']['extra_cli_args'] == [
        '--disallowedTools=EnterPlanMode',
        '--disallowedTools=ExitPlanMode',
    ]


async def test_codebuddy_prompt_suffix_is_appended_to_the_task_not_substituted() -> None:
    """The yaml sets prompt_suffix; prove the agent really appends it to the task it runs.

    Declaring the field only buys "the key is not dropped at merge time". What reaches the CLI
    is a separate question, and this is the test that answers it.
    """
    import nexus.extensions.agents.codebuddy.agent as codebuddy_agent

    captured: dict[str, Any] = {}

    async def _fake_call_runtime_agent(_target, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {'success': True, 'output': '', 'error': None}

    original = codebuddy_agent.call_runtime_agent
    codebuddy_agent.call_runtime_agent = _fake_call_runtime_agent
    try:
        await codebuddy_agent.run_codebuddy(
            codebuddy_agent.CodeBuddyInput(
                user_prompt='fix the bug',
                working_dir='/testbed',
                prompt_suffix='Never ask the user questions.',
            )
        )
    finally:
        codebuddy_agent.call_runtime_agent = original

    assert captured['task'] == 'fix the bug\n\nNever ask the user questions.'


async def test_codebuddy_multiturn_rejects_prompt_suffix_instead_of_ignoring_it() -> None:
    # CodeBuddyMultiturnInput inherits the field, so _require_consumers cannot catch a yaml
    # that sets it here -- but this entry point builds turns from the input provider and has
    # nothing to append it to. Silent no-op is exactly what this layer exists to prevent.
    import nexus.extensions.agents.codebuddy.agent as codebuddy_agent

    async def _no_more_input(_request: Any) -> str | None:
        return None

    assert 'prompt_suffix' in codebuddy_agent.CodeBuddyMultiturnInput.model_fields
    with pytest.raises(ValueError, match='not supported by codebuddy_multiturn'):
        await codebuddy_agent.run_codebuddy_multiturn(
            codebuddy_agent.CodeBuddyMultiturnInput(
                user_prompt='task',
                working_dir='/testbed',
                get_user_input=_no_more_input,
                prompt_suffix='Never ask the user questions.',
            )
        )


# Transcribed from frontend/scaffold runtime_setup.py `_override_tool_mode_in_catalog`, the
# `catalog_profile == 'gpt6_v7'` branch the frontend rollout actually runs. These values are
# deliberately NOT Codex's own defaults: the goal is trajectories that train HY models well.
_GPT6_V7 = {
    'use_responses_lite': False,
    'apply_patch_tool_type': 'freeform',
    'supports_search_tool': False,
    'supports_image_detail_original': False,
    'default_reasoning_level': 'max',
    'context_window': 1_000_000,
    'max_context_window': 1_000_000,
    'auto_compact_token_limit': 900_000,
    'multi_agent_version': 'v2',
    'experimental_supported_tools': [],
}


@pytest.mark.parametrize(
    ('agent', 'expected_tool_mode'),
    [
        ('codex_0.160.0_codemode', 'code_mode_only'),
        ('codex_0.160.0_toolmode', 'direct'),
    ],
)
def test_codex_catalog_overrides_match_the_frontend_rollout_profile(env, agent: str, expected_tool_mode: str) -> None:
    # model_catalog_overrides (MR !1535) shallow-merges onto Codex's own entry, so only the
    # fields we deliberately differ on are written here -- persona and the rest stay native.
    env(AGENT_CONFIG=agent, LLM_CONFIG='gpt-6-sol.openai_responses')
    agent_config = _resolve(load_config()['agent_config'])
    assert agent_config.get('model_catalog_json') is None, 'the 614-line hand-written clone is gone'
    overrides = agent_config['model_catalog_overrides']
    assert {k: overrides.get(k) for k in _GPT6_V7} == _GPT6_V7
    assert overrides['tool_mode'] == expected_tool_mode
    assert overrides['supports_parallel_tool_calls'] is True
    assert 'slug' not in overrides  # runtime_setup.py rejects overriding the slug
    # gpt6_v7 pops base_instructions; a shallow merge cannot delete, so null was the stand-in.
    # Real runs scored 1.0 with and without it (swe/smoke/codex_gpt6v7_20261015/{v7,v7_nullbase}),
    # so it was dropped as unnecessary -- the override is absent, not explicitly null.
    assert 'base_instructions' not in overrides
    # config.toml's own copies: codex clamps the compaction trigger to
    # min(auto_compact_token_limit, context_window * 90%), so both must be set together.
    assert agent_config['model_context_window'] == 1_000_000
    assert agent_config['auto_compact_token_limit'] == 900_000


@pytest.mark.parametrize(
    ('llm', 'expected'),
    [
        # Real Azure-backed AIHub deployments: the llm file states the fact, Codex trusts it.
        ('gpt-6-sol.openai_responses', True),
        ('gpt-61-sol.openai_responses', True),
        ('gpt-6-astra.openai_responses', True),
        # Same family (gpt), different wire: aihub2's 'standard' endpoint is not Azure-backed.
        # If this were keyed by family instead of by llm file, it would wrongly come out True.
        ('gpt-56-sol.openai_responses', False),
        # Non-gpt families never declare the fact; the scaffold's own safe default stands.
        ('claude-opus-5.litellm', False),
    ],
)
def test_use_azure_provider_follows_the_specific_llm_wire_not_the_family(env, llm: str, expected: bool) -> None:
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG=llm)
    assert _resolve(load_config()['agent_config'])['use_azure_provider'] is expected


@pytest.mark.parametrize(
    ('llm', 'expected'),
    [
        # openai_responses wires deliver reasoning as native reasoning_items, so the Responses
        # converter must NOT also mint one from plaintext reasoning_content.
        ('gpt-6-sol.openai_responses', False),  # gpt, azure, openai_responses
        ('gpt-56-sol.openai_responses', False),  # gpt, non-azure, openai_responses
        ('deepseek-v41-flash.openai_responses', False),  # family=gpt but a different vendor
        # Every other client carries reasoning only as plaintext reasoning_content. Leaving the
        # flag false there means no reasoning item is minted at all and the signal is dropped --
        # which is the whole point of the SFT pipeline.
        ('claude-opus-5.litellm', True),  # claude via litellm
        ('claude-opus-5.openai', True),  # claude via the chat-completions client
        ('claude-opus-5.vertex_anthropic', True),  # claude via vertex_anthropic
        ('hy4-taiji-infer-proxy', True),  # hunyuan, infer_proxy -- the swe-codex-tool-hy4-* pairing
        ('hy3.openai', True),  # hunyuan via the chat-completions client
    ],
)
@pytest.mark.parametrize('agent', ['codex_0.160.0_codemode', 'codex_0.160.0_toolmode'])
def test_wrap_plaintext_reasoning_follows_the_wire_protocol_not_the_family(
    env, agent: str, llm: str, expected: bool
) -> None:
    """False only where the wire has native reasoning_items; true everywhere else.

    This is the production rule, read off origin/main rather than restated: the production
    codex agent yaml defaults the key to ``true`` and exactly the four ``openai_responses``
    llm profiles override it to ``false`` (the other 13 claude/hunyuan profiles never set it).

    An earlier version of this test asserted ``False`` for *every* pairing, on the strength of
    "no experiment cell overrides it". That checked the wrong layer -- production overrides it
    from the llm profile, not from the experiment input -- and so it froze a flat default that
    silently dropped reasoning on every claude and hunyuan wire.
    """
    env(AGENT_CONFIG=agent, LLM_CONFIG=llm)
    assert _resolve(load_config()['agent_config'])['wrap_plaintext_reasoning_content'] is expected


def test_wrap_plaintext_reasoning_matches_production_for_every_shared_wire(env) -> None:
    """Derive the expectation from production instead of hardcoding the wire list.

    Production marks the flag false on exactly its ``openai_responses`` llm profiles. Rather
    than trust that reading, read it back: for every llm file we share, the effective value must
    be ``client != 'openai_responses'``.
    """
    import subprocess

    prod_llm_dir = 'nexus/extensions/tasks/swe/swe_cosmos/configs/llms'
    repo = SHARED_DIR.parents[2]
    try:
        listing = subprocess.run(
            ['git', 'ls-tree', '-r', '--name-only', 'origin/main', prod_llm_dir],
            capture_output=True,
            text=True,
            cwd=repo,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env without git
        pytest.skip(f'cannot read production llm configs: {exc}')
    if listing.returncode != 0 or not listing.stdout.strip():  # pragma: no cover
        pytest.skip(f'cannot list {prod_llm_dir} from origin/main')

    # Confirm production really splits it by client before relying on that rule.
    for path in listing.stdout.split():
        body = subprocess.run(
            ['git', 'show', f'origin/main:{path}'], capture_output=True, text=True, cwd=repo, timeout=30
        ).stdout
        declares_false = 'wrap_plaintext_reasoning_content: false' in body
        is_responses = path.endswith('.openai_responses.yaml')
        assert declares_false is is_responses, (
            f'production {path}: declares false={declares_false} but openai_responses={is_responses};'
            ' the client-based rule this test relies on no longer holds'
        )

    for llm_file in sorted((SHARED_DIR / 'llms').glob('*.yaml')):
        wire = llm_file.name[: -len('.yaml')]
        client = wire.rsplit('.', 1)[-1]
        env(AGENT_CONFIG='codex_0.160.0_toolmode', LLM_CONFIG=wire)
        actual = _resolve(load_config()['agent_config'])['wrap_plaintext_reasoning_content']
        assert actual is (client != 'openai_responses'), (
            f'{wire} (client={client}) resolved wrap_plaintext_reasoning_content={actual!r}'
        )


@pytest.mark.parametrize(
    ('llm', 'window', 'compact'),
    [
        # litellm wires: codex sends the catalog value as max_tokens; left at 1000000 the backend
        # rejects it (real sandbox 400 on claude-opus-5.litellm). Same client, same path for opus-4.8.
        ('claude-opus-5.litellm', 128000, 115000),
        ('claude-opus-48.litellm', 128000, 115000),
        ('claude-opus-48-aws_third-aihub2.0.litellm', 128000, 115000),
        # The chat-completions wire ran green in a real sandbox at the 1000000 default (codex_toolmode x
        # claude-opus-5.openai, Resolved=True). Do not "harmonise" it: that would move its compaction
        # trigger from 900k to 115k on a pairing that is known to work.
        ('claude-opus-5.openai', 1000000, 900000),
    ],
)
def test_codex_context_window_per_claude_wire(env, llm: str, window: int, compact: int, monkeypatch) -> None:
    for name in ('APP_ID', 'APP_KEY'):
        monkeypatch.setenv(name, 'x')
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG=llm)
    agent_config = _resolve(load_config()['agent_config'])
    assert (agent_config['model_context_window'], agent_config['auto_compact_token_limit']) == (window, compact)


def test_codex_context_window_caps_to_the_real_model_limit_for_claude(env) -> None:
    # Real sandbox run: codex sends max_output_tokens straight from this catalog value. Left at
    # the gpt-oriented 1000000 default, claude-opus-5 rejects it with a 400 ("max_tokens:
    # 1000000 > 128000, which is the maximum allowed number of output tokens for
    # anthropic.claude-opus-5") -- confirmed by replaying the exact real request through the
    # installed openai SDK. Not a family fact (a different claude model could have a different
    # real cap), so it comes from the llm file, not a hardcoded claude-family value.
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG='claude-opus-5.litellm')
    agent_config = _resolve(load_config()['agent_config'])
    assert agent_config['model_context_window'] == 128000
    assert agent_config['model_catalog_overrides']['context_window'] == 128000
    assert agent_config['model_catalog_overrides']['max_context_window'] == 128000
    assert agent_config['auto_compact_token_limit'] == 115000
    assert agent_config['model_catalog_overrides']['auto_compact_token_limit'] == 115000


def test_codex_context_window_stays_gpt_default_for_gpt_wires(env) -> None:
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG='gpt-6-sol.openai_responses')
    assert _resolve(load_config()['agent_config'])['model_context_window'] == 1_000_000


def test_codex_reasoning_effort_comes_from_the_llm_files_effort_fact(env) -> None:
    # codex reads the protocol-agnostic `llm.reasoning_effort` fact, so it never has to know
    # whether the wire spells the field `reasoning.effort` (openai_responses) or
    # `output_config.effort` (litellm/Anthropic). The llm file is the one place that knows.
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG='claude-opus-5.litellm', HR_API_KEY='x')
    assert _resolve(load_config())['agent_config']['reasoning_effort'] == 'max'
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG='gpt-6-sol.openai_responses', HR_API_KEY='x')
    assert _resolve(load_config())['agent_config']['reasoning_effort'] == 'xhigh'
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG='claude-opus-48.litellm', APP_ID='x', APP_KEY='y')
    assert _resolve(load_config())['agent_config']['reasoning_effort'] == 'xhigh'


def _wire_effort(request_args: dict) -> object:
    """The effort value a wire actually sends, wherever that wire spells it."""
    extra = request_args.get('extra_body') or {}
    for candidate in (
        (request_args.get('reasoning') or {}).get('effort'),
        (request_args.get('output_config') or {}).get('effort'),
        (extra.get('output_config') or {}).get('effort'),
        (extra.get('chat_template_kwargs') or {}).get('reasoning_effort'),
        request_args.get('reasoning_effort'),
    ):
        if candidate is not None:
            return candidate
    return None


@pytest.mark.parametrize('llm', sorted(p.stem for p in (SHARED_DIR / 'llms').glob('*.yaml')))
def test_every_llm_file_declares_a_reasoning_effort_matching_its_wire(env, monkeypatch, llm: str) -> None:
    # The fact exists so a scaffold never has to know how a wire spells the field. That only holds if
    # the fact and the field cannot disagree: set both env vars to a non-default value and compare.
    for name, value in {
        'HR_API_KEY': 'x',
        'OPENAI_API_KEY': 'x',
        'APP_ID': 'a',
        'APP_KEY': 'b',
        'OPENAI_MODEL': 'm',
        'REASONING_EFFORT': 'low',
        'THINKING_EFFORT': 'low',
    }.items():
        monkeypatch.setenv(name, value)
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG=llm, RUNTIME_CONFIG='swe_runtime')
    resolved = _resolve(load_config())
    wire = _wire_effort(resolved['openai_clients']['main']['config']['request_args'])
    assert wire is not None, f'{llm}: no effort field found on the wire'
    assert resolved['agent_config']['reasoning_effort'] == wire


@pytest.mark.parametrize('llm', sorted(p.stem for p in (SHARED_DIR / 'llms').glob('claude-*.yaml')))
def test_every_claude_llm_default_max_tokens_meets_the_wire_pin(env, monkeypatch, llm: str) -> None:
    # Production and eval are pinned to max_tokens >= 65535 (prod-eval-wire-alignment). The 4.8 files
    # used to default to 16000, copied from an old Bedrock generator, so a run that forgot MAX_TOKENS
    # silently broke the pin. No env override here: the default itself has to hold.
    for name, value in {
        'HR_API_KEY': 'x',
        'OPENAI_API_KEY': 'x',
        'APP_ID': 'a',
        'APP_KEY': 'b',
        'OPENAI_MODEL': 'm',
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('MAX_TOKENS', raising=False)
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG=llm, RUNTIME_CONFIG='swe_runtime')
    args = _resolve(load_config())['openai_clients']['main']['config']['request_args']
    assert args.get('max_tokens') >= 65535, f'{llm}: default max_tokens {args.get("max_tokens")} < 65535'


# The eval-table profile sends no summary on purpose: the evalmx_gpt61sol_aws_xhigh runs it must stay comparable with did not send one.
_SUMMARY_EXEMPT = {'gpt-61-sol-aws_third-eval.openai_responses'}


@pytest.mark.parametrize(
    'llm',
    sorted(p.stem for p in (SHARED_DIR / 'llms').glob('gpt-*.openai_responses.yaml') if p.stem not in _SUMMARY_EXEMPT),
)
def test_every_gpt_responses_llm_default_keeps_reasoning_summary_on(env, monkeypatch, llm: str) -> None:
    # Wire pin "summarized thinking": the Responses summary must be on, never off/None. The pin names
    # `auto`; some files default to `detailed`, which is also on, so the test pins "on" and not the spelling.
    for name, value in {
        'HR_API_KEY': 'x',
        'OPENAI_API_KEY': 'x',
        'APP_ID': 'a',
        'APP_KEY': 'b',
        'OPENAI_MODEL': 'm',
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('REASONING_SUMMARY', raising=False)
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG=llm, RUNTIME_CONFIG='swe_runtime')
    summary = _resolve(load_config())['openai_clients']['main']['config']['request_args']['reasoning'].get('summary')
    assert summary in {'auto', 'concise', 'detailed'}, f'{llm}: reasoning.summary={summary!r} is off'


@pytest.mark.parametrize(
    ('client', 'client_args', 'api_key', 'expect'),
    [
        # AI Hub + a client that takes the argument -> set it; the client folds it into the key.
        (
            'openai_responses',
            {'endpoint': 'http://api.aihub.woa.com/openai/v1/responses'},
            'k?provider=azure',
            {'cache_task_id': 'ID'},
        ),
        # AI Hub with a BARE key still gets one: claude-opus-5.litellm looks exactly like this, and the
        # old "does the key contain ?" rule silently skipped it.
        ('litellm', {'base_url': 'http://api.aihub.woa.com/standard'}, 'bare', {'cache_task_id': 'ID'}),
        # AI Hub + a client with no such argument -> write the key query ourselves.
        (
            'openai',
            {'base_url': 'http://api.aihub.woa.com/openapi/v2'},
            'k?provider=taiji',
            {'api_key': 'k?provider=taiji&cache_task_id=ID'},
        ),
        # Polaris: not AI Hub, but the id is now required everywhere that is not explicitly
        # excluded, so these carry one. This is a deliberate divergence from the Opus production
        # configs, which send none here.
        (
            'litellm',
            {'base_url': 'http://trpc-gpt-eval.production.polaris:8080'},
            'id:key?provider=aws_third',
            {'cache_task_id': 'ID'},
        ),
        (
            'vertex_anthropic',
            {'base_url': 'http://trpc-gpt-eval.production.polaris:8080'},
            'id:key?provider=google',
            {'api_key': 'id:key?provider=google&cache_task_id=ID'},
        ),
        # Excluded by gateway: bare keys on the Taiji / infer endpoints must not be rewritten.
        ('openai', {'base_url': 'http://api.taiji.woa.com/openapi/inner/v2'}, 'bare-key', None),
        ('litellm', {'base_url': 'http://infer-api.taiji.woa.com/api/v1'}, 'bare-key', None),
        # Never overwrite a caller's id, wherever it sits.
        ('openai', {'base_url': 'http://api.aihub.woa.com/openapi/v2'}, 'k?cache_task_id=mine', None),
        ('openai_responses', {'endpoint': 'http://api.aihub.woa.com/v1/responses', 'cache_task_id': 'mine'}, 'k', None),
        ('openai', {'base_url': 'http://api.aihub.woa.com/openapi/v2'}, '', None),
    ],
)
def test_cache_task_overlay_follows_the_gateway_not_the_key_shape(client, client_args, api_key, expect) -> None:
    from nexus.extensions.configs.cache_id import cache_task_overlay

    out, _ = cache_task_overlay(client, client_args, api_key, 'ID')
    assert out == ({'client_args': expect} if expect else None)


def test_an_id_in_the_key_query_would_be_replaced_by_the_derived_one() -> None:
    """Why the id is set as a client argument for clients that accept one.

    The client folds `cache_task_id` into the key by replacing any existing parameter, and the
    runner derives one with `setdefault` -- so an id the generator wrote into the query would be
    swapped out, leaving `metadata.cache_task_id` naming an id the wire never carried.
    """
    from nexus.extensions.openai_clients.cache_task_id import normalize_cache_task_id

    assert normalize_cache_task_id('k?provider=aws_third&cache_task_id=GENERATOR', 'DERIVED').endswith('=DERIVED')


@pytest.mark.parametrize('llm', sorted(p.stem for p in (SHARED_DIR / 'llms').glob('*.yaml')))
def test_every_wire_carries_a_cache_task_id_except_the_named_exclusions(env, monkeypatch, llm: str) -> None:
    """Carrying the id is the default; each omission has to be a named gateway.

    The downstream SFT pipeline requires an id on every request, so this is an exclusion list,
    not an allowlist: a newly added wire gets one without anyone remembering to opt in, and
    dropping one means editing GATEWAYS_WITHOUT_CACHE_TASK_ID on purpose.
    """
    from nexus.extensions.configs.cache_id import GATEWAYS_WITHOUT_CACHE_TASK_ID, _gateway, cache_task_overlay

    for name, value in {
        'HR_API_KEY': 'k',
        'APP_ID': 'a',
        'APP_KEY': 'b',
        'OPENAI_MODEL': 'm',
        'GONGFENG_RUNTIME_TOKEN': 'x',
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)  # let each file's own default key apply
    env(RUNTIME_CONFIG='swe_runtime', AGENT_CONFIG='kilo', LLM_CONFIG=llm)
    main = _resolve(load_config())['openai_clients']['main']
    config = main['config']
    client_args = config.get('client_args') or {}
    overlay, _ = cache_task_overlay(main['name'], client_args, str(client_args.get('api_key') or ''), 'ID')
    gateway = _gateway(client_args)
    excluded = any(host in gateway for host in GATEWAYS_WITHOUT_CACHE_TASK_ID)
    assert (overlay is None) is excluded, (
        f'{llm}: gateway={gateway!r} excluded={excluded} overlay={overlay!r}. '
        'Every wire must carry a cache_task_id unless its gateway is named in '
        'GATEWAYS_WITHOUT_CACHE_TASK_ID.'
    )


def _swe_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env, llm: str, *, key_env: dict[str, str]):
    from nexus.extensions.configs import generator_swe_template as generator

    dataset = tmp_path / 'swe.jsonl'
    dataset.write_text(''.join(json.dumps({'instance_id': f'repo__{n}'}) + '\n' for n in ('a', 'b')))
    env(RUNTIME_CONFIG='swe_runtime', AGENT_CONFIG='kilo', LLM_CONFIG=llm)
    for name, value in {'DATASET_PATH': str(dataset), 'GONGFENG_RUNTIME_TOKEN': 'x', **key_env}.items():
        monkeypatch.setenv(name, value)
    return list(generator.generate_configs())


def _wire_cache_id(item) -> str | None:
    """The id the request will actually carry, read the way the client reads it."""
    from urllib.parse import parse_qs

    client_args = item.config.openai_clients['main']['config']['client_args']
    if client_args.get('cache_task_id'):
        return client_args['cache_task_id']
    query = parse_qs(str(client_args.get('api_key', '')).partition('?')[2])
    return (query.get('cache_task_id') or [None])[0]


@pytest.mark.parametrize(
    ('llm', 'key_env'),
    [
        ('gpt-6-sol.openai_responses', {'HR_API_KEY': 'k'}),  # responses client: explicit argument
        ('gpt-61-sol-aws_third.openai_responses', {'HR_API_KEY': 'k'}),
        ('claude-opus-5.openai', {'HR_API_KEY': 'k'}),  # chat client: key query
        ('hy4-preview.openai', {'HR_API_KEY': 'k'}),
    ],
)
def test_swe_metadata_cache_task_id_is_the_id_on_the_wire(tmp_path, monkeypatch, env, llm, key_env) -> None:
    first, second = _swe_items(tmp_path, monkeypatch, env, llm, key_env=key_env)
    for item in (first, second):
        on_wire = _wire_cache_id(item)
        assert on_wire, f'{llm}: no cache_task_id reached the client'
        assert item.metadata['cache_task_id'] == on_wire, 'HyWay would pass the CoT hack a different id'
    assert first.metadata['cache_task_id'] != second.metadata['cache_task_id'], 'one id per trajectory'


def test_swe_does_not_record_an_id_the_wire_never_carried(tmp_path, monkeypatch, env) -> None:
    # api.taiji.woa.com takes a bare key and must not be rewritten, so there is nothing to record.
    items = _swe_items(tmp_path, monkeypatch, env, 'hy4-preview-offline.openai', key_env={'OPENAI_API_KEY': 'bare-key'})
    for item in items:
        assert _wire_cache_id(item) is None
        assert 'cache_task_id' not in item.metadata


@pytest.mark.parametrize(
    ('sent', 'floor', 'expected'),
    [
        ('none', 'low', 'low'),  # the case this exists for: a no-thinking side request
        ('minimal', 'low', 'low'),  # also rejected by gpt-6.1-sol / gpt-6-astra
        ('low', 'low', 'low'),  # already legal, untouched
        ('xhigh', 'low', 'xhigh'),  # the agent loop must never be weakened
        ('max', 'low', 'max'),
    ],
)
def test_the_effort_floor_lifts_only_values_below_it(sent: str, floor: str, expected: str) -> None:
    from nexus.extensions.openai_clients.openai_responses import _raise_reasoning_effort_to_floor

    request = {'reasoning': {'effort': sent, 'summary': 'auto'}}
    _raise_reasoning_effort_to_floor(request, floor)
    assert request['reasoning']['effort'] == expected
    assert request['reasoning']['summary'] == 'auto', 'sibling keys must survive'


def test_a_wire_without_a_declared_floor_is_left_alone() -> None:
    # gpt-6-sol / gpt-5.6-sol / deepseek accept 'none' (measured), so they declare no
    # floor and their side requests must keep costing nothing.
    from nexus.extensions.openai_clients.openai_responses import _raise_reasoning_effort_to_floor

    request = {'reasoning': {'effort': 'none'}}
    assert _raise_reasoning_effort_to_floor(request, None) is None
    assert request['reasoning']['effort'] == 'none'


def test_the_effort_floor_refuses_a_value_it_cannot_order() -> None:
    from nexus.extensions.openai_clients.openai_responses import _raise_reasoning_effort_to_floor

    with pytest.raises(ValueError, match='min_reasoning_effort'):
        _raise_reasoning_effort_to_floor({'reasoning': {'effort': 'none'}}, 'turbo')
    # An unknown effort is the caller's business; guessing its rank would be wrong.
    request = {'reasoning': {'effort': 'ultra'}}
    assert _raise_reasoning_effort_to_floor(request, 'low') is None
    assert request['reasoning']['effort'] == 'ultra'


@pytest.mark.parametrize(
    'llm',
    ['gpt-61-sol.openai_responses', 'gpt-61-sol-aws_third.openai_responses', 'gpt-6-astra.openai_responses'],
)
def test_the_wires_measured_to_reject_none_declare_the_floor(env, monkeypatch, llm: str) -> None:
    for name, value in {'HR_API_KEY': 'x', 'OPENAI_API_KEY': 'x', 'APP_ID': 'a', 'APP_KEY': 'b'}.items():
        monkeypatch.setenv(name, value)
    env(AGENT_CONFIG='codex_0.160.0_codemode', LLM_CONFIG=llm, RUNTIME_CONFIG='swe_runtime')
    client_args = _resolve(load_config())['openai_clients']['main']['config']['client_args']
    assert client_args.get('min_reasoning_effort') == 'low', f'{llm} rejects effort=none and needs the floor'


def test_an_llm_file_can_clamp_a_scaffold_sent_reasoning_effort() -> None:
    """`enforced_request_args` is the layer's lever over requests the scaffold builds itself.

    Scaffolds periodically issue a no-thinking side request carrying
    `reasoning.effort=none`. That value is a per-model fact: gpt-6-sol accepts it,
    us.openai.gpt-6.1-sol answers 400 and wants `low` (measured 2026-10-16), which
    costs ~4% of calls on that wire. An llm file cannot reach the construction site
    of that request, but it can clamp the value on the way out -- documented here so
    the option is not rediscovered by reading the client.
    """
    from nexus.core.openai_client import apply_enforced_request_args
    from nexus.extensions.openai_clients.openai_responses import _reapply_enforced_nested_controls

    scaffold_request = {'model': 'us.openai.gpt-6.1-sol', 'reasoning': {'effort': 'none'}}
    policy = {'reasoning': {'effort': 'xhigh'}}
    clamped = apply_enforced_request_args(scaffold_request, policy)
    _reapply_enforced_nested_controls(clamped, policy)
    assert clamped['reasoning'] == {'effort': 'xhigh'}

    # The converter's flat spelling is clamped by the same policy key.
    flat = apply_enforced_request_args({'reasoning_effort': 'none'}, {'reasoning_effort': 'xhigh'})
    assert flat['reasoning_effort'] == 'xhigh'


def test_a_null_llm_fact_drops_the_key_in_the_flat_pass_too(tmp_path: Path) -> None:
    from nexus.extensions.configs.load_config import _apply_llm_family

    config = {'llm': {'family': 'claude', 'effort': None}, 'agent_config': {'a': '${llm:effort}', 'b': 1}}
    assert _apply_llm_family(config)['agent_config'] == {'b': 1}


def test_a_flat_ref_error_names_its_real_location() -> None:
    from nexus.extensions.configs.load_config import _apply_llm_family

    config = {'llm': {'family': 'claude'}, 'agent_config': {'x': '${llm:missing}'}}
    with pytest.raises(ValueError, match=r'^agent_config\.x needs llm\.missing'):
        _apply_llm_family(config)


def test_opus48_effort_default_is_one_litellm_accepts_for_opus_4_8(env) -> None:
    # litellm rejects xhigh for every Claude and max for anything but Opus 4.6, before sending.
    env(LLM_CONFIG='claude-opus-48-aws_third-aihub2.0.litellm')
    request_args = _resolve(load_config()['openai_clients']['main']['config']['request_args'])
    assert request_args['output_config']['effort'] in {'high', 'medium', 'low'}
    # kilo sends a non-1 temperature, which Bedrock rejects while thinking is on.
    assert {'reasoning_effort', 'temperature'} <= set(request_args['additional_drop_params'])


def test_mini_wall_time_follows_agent_timeout(env) -> None:
    env(AGENT_CONFIG='mini_swe_agent')
    assert _resolve(load_config()['agent_config'])['wall_time_limit_seconds'] == 0
    env(AGENT_CONFIG='mini_swe_agent', AGENT_TIMEOUT='12000')
    assert _resolve(load_config()['agent_config'])['wall_time_limit_seconds'] == 12000


@pytest.mark.parametrize(
    ('runtime', 'request_timeout', 'cpu', 'memory'),
    [
        # frontend/scaffold generator_rollout.py: request_timeout 10000.0, cpu 8, memory 16.
        # Its agent limit is also 10000 (FRONTEND_AGENT_TIMEOUT or 10000), i.e. frontend production
        # lets the outer fuse equal the agent limit -- unlike SWE, where runner.timeout must exceed it.
        # This file used to say 12900, which is the SWE number and appears in neither frontend source.
        ('frontend_prod_runtime', 10000, 8, 16),
        # frontend/eval/runs/humaneval_*/generator.py: request_timeout 13500 with agent 12000.
        ('frontend_eval_runtime', 13500, 4, 16),
    ],
)
def test_frontend_runtimes_keep_the_timeouts_their_sources_use(env, runtime, request_timeout, cpu, memory) -> None:
    # Both production profiles make Langfuse mandatory (no default), so the keys go through `env`,
    # which clears ambient variables before setting the ones it is given.
    env(
        RUNTIME_CONFIG=runtime,
        AGENT_CONFIG='kilo',
        LLM_CONFIG='gpt-6-sol.openai_responses',
        LANGFUSE_PUBLIC_KEY='x',
        LANGFUSE_SECRET_KEY='x',
        HR_API_KEY='k',
        GONGFENG_RUNTIME_TOKEN='t',
    )
    provider = _resolve(load_config())['runtime_provider_config']
    assert provider['request_timeout'] == request_timeout
    assert provider['resource_spec'] == {'cpu': cpu, 'memory': memory}


def test_swe_prod_runtime_pins_the_production_sandbox(env) -> None:
    # The prod profile deliberately gives langfuse no default: a production run without a
    # trajectory should fail loudly. So a test that resolves it has to supply the keys.
    env(RUNTIME_CONFIG='swe_prod_runtime', LANGFUSE_PUBLIC_KEY='pk', LANGFUSE_SECRET_KEY='sk')
    body = _resolve(load_config())
    assert body['runtime_provider_config']['resource_spec'] == {'cpu': 4, 'memory': 16}
    assert body['runtime_provider_config']['request_timeout'] == 15000
    assert body['task_config']['hack_reward_override'] == 0.0
    assert len(body['task_config']['pre_agent_script']) == 2


def test_swe_prod_runtime_serves_eval_too_through_runner_timeout(env) -> None:
    """One file for prod and eval: they used to be two, differing by this single line.

    The default is the safe side -- the agent axis defaults to 14400, so 14500 lets the agent
    hit its own limit first and terminate cleanly. Production pins the smaller value.
    """
    env(RUNTIME_CONFIG='swe_prod_runtime', LANGFUSE_PUBLIC_KEY='pk', LANGFUSE_SECRET_KEY='sk')
    assert _resolve(load_config())['runner'] == {'timeout': 14500}

    env(
        RUNTIME_CONFIG='swe_prod_runtime',
        RUNNER_TIMEOUT='12900',
        LANGFUSE_PUBLIC_KEY='pk',
        LANGFUSE_SECRET_KEY='sk',
    )
    assert _resolve(load_config())['runner'] == {'timeout': 12900}


def test_runner_timeout_stays_inside_the_request_timeout(env) -> None:
    # The invariant the header states: AGENT_TIMEOUT < runner.timeout < request_timeout.
    env(RUNTIME_CONFIG='swe_prod_runtime', LANGFUSE_PUBLIC_KEY='pk', LANGFUSE_SECRET_KEY='sk')
    body = _resolve(load_config())
    assert body['runner']['timeout'] < body['runtime_provider_config']['request_timeout']


def test_frontend_template_aligns_agent_working_dir_and_names_traces(
    env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nexus.extensions.configs import generator_frontend_template as generator

    dataset = tmp_path / 'cases.jsonl'
    dataset.write_text(json.dumps({'instance_id': 'case-1', 'prompt': 'draw a pelican'}) + '\n')
    env(RUNTIME_CONFIG='frontend_runtime', AGENT_CONFIG='kilo', LLM_CONFIG='gpt-6-sol.openai_responses')
    for name, value in {'DATASET_PATH': str(dataset), 'HR_API_KEY': 'x', 'GONGFENG_RUNTIME_TOKEN': 'x'}.items():
        monkeypatch.setenv(name, value)

    (item,) = list(generator.generate_configs())
    assert item.config.agent_config['working_dir'] == item.config.task_config['working_dir'] == '/workspace'
    langfuse = item.config.processors['langfuse'].args
    assert langfuse['trace_name'] == 'case-1'
    assert langfuse['trace_metadata'] == {'instance_id': 'case-1', 'agent_name': 'kilo', 'model': 'gpt-6-sol'}


def _frontend_cell(env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = tmp_path / 'cases.jsonl'
    dataset.write_text(json.dumps({'instance_id': 'case-1', 'prompt': 'draw a pelican'}) + '\n')
    env(RUNTIME_CONFIG='frontend_runtime', AGENT_CONFIG='kilo', LLM_CONFIG='gpt-6-sol.openai_responses')
    for name, value in {'DATASET_PATH': str(dataset), 'HR_API_KEY': 'x', 'GONGFENG_RUNTIME_TOKEN': 'x'}.items():
        monkeypatch.setenv(name, value)


def test_frontend_task_config_extra_reaches_every_instance(
    env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nexus.extensions.configs import generator_frontend_template as generator

    _frontend_cell(env, monkeypatch, tmp_path)
    monkeypatch.setenv('FRONTEND_TASK_CONFIG_EXTRA', '{"test_timeout": 123}')
    (item,) = list(generator.generate_configs())
    assert item.config.task_config['test_timeout'] == 123
    assert item.config.task_config['instance_id'] == 'case-1'  # per-instance values still win


def test_frontend_task_config_extra_cannot_smuggle_a_field_this_checkout_drops(
    env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # tar_exclude_paths exists only on the frontend checkout's FrontendTaskInput. On any checkout
    # that lacks it the run must stop at generate time, not finish having ignored the setting.
    from nexus.extensions.configs import generator_frontend_template as generator

    _frontend_cell(env, monkeypatch, tmp_path)
    monkeypatch.setenv('FRONTEND_TASK_CONFIG_EXTRA', '{"definitely_not_a_task_field": ["a"]}')
    with pytest.raises(ValueError, match='would be dropped silently'):
        list(generator.generate_configs())


def test_frontend_image_by_agent_overrides_only_the_listed_agent(
    env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nexus.extensions.configs import generator_frontend_template as generator

    _frontend_cell(env, monkeypatch, tmp_path)  # agent = kilo
    (default,) = list(generator.generate_configs())
    monkeypatch.setenv('FRONTEND_IMAGE_BY_AGENT', '{"codebuddy": "reg/node22:1"}')
    (unlisted,) = list(generator.generate_configs())
    assert unlisted.config.runtime_provider_config['image'] == default.config.runtime_provider_config['image']
    monkeypatch.setenv('FRONTEND_IMAGE_BY_AGENT', '{"kilo": "reg/node22:1"}')
    (listed,) = list(generator.generate_configs())
    assert listed.config.runtime_provider_config['image'] == 'reg/node22:1'


def test_frontend_image_by_agent_rejects_a_malformed_map(env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from nexus.extensions.configs import generator_frontend_template as generator

    _frontend_cell(env, monkeypatch, tmp_path)
    monkeypatch.setenv('FRONTEND_IMAGE_BY_AGENT', '{"kilo": 3}')
    with pytest.raises(ValueError, match='FRONTEND_IMAGE_BY_AGENT'):
        list(generator.generate_configs())


def test_frontend_eval_times_repeats_each_instance_with_distinct_run_ids_and_cache_scopes(
    env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nexus.extensions.configs import generator_frontend_template as generator

    _frontend_cell(env, monkeypatch, tmp_path)
    (single,) = list(generator.generate_configs())
    assert single.run_id == 'case-1'  # EVAL_TIMES=1 keeps the plain id

    monkeypatch.setenv('EVAL_TIMES', '3')
    items = list(generator.generate_configs())
    assert [i.run_id for i in items] == ['case-1__try1', 'case-1__try2', 'case-1__try3']
    assert generator.get_total_count() == 3
    ids = [_wire_cache_id(i) for i in items]
    assert None not in ids, 'each attempt must get its own cache scope'
    assert len(set(ids)) == 3, 'each attempt must get its own cache scope'
    assert [i.metadata['cache_task_id'] for i in items] == ids


def test_frontend_eval_times_rejects_a_non_positive_value(env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from nexus.extensions.configs import generator_frontend_template as generator

    _frontend_cell(env, monkeypatch, tmp_path)
    monkeypatch.setenv('EVAL_TIMES', '0')
    with pytest.raises(ValueError, match='EVAL_TIMES'):
        list(generator.generate_configs())


def _frontend_two_cases(env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm: str) -> None:
    dataset = tmp_path / 'cases.jsonl'
    dataset.write_text(
        json.dumps({'instance_id': 'case-1', 'prompt': 'a'})
        + '\n'
        + json.dumps({'instance_id': 'case-2', 'prompt': 'b'})
        + '\n'
    )
    env(RUNTIME_CONFIG='frontend_runtime', AGENT_CONFIG='kilo', LLM_CONFIG=llm)
    for name, value in {'DATASET_PATH': str(dataset), 'HR_API_KEY': 'bare', 'GONGFENG_RUNTIME_TOKEN': 'x'}.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize('llm', ['gpt-61-sol-aws_third.openai_responses', 'gpt-6-sol.openai_responses'])
def test_frontend_cache_task_id_is_per_cell_and_matches_the_metadata(env, monkeypatch, tmp_path, llm) -> None:
    from nexus.extensions.configs import generator_frontend_template as generator

    _frontend_two_cases(env, monkeypatch, tmp_path, llm)
    first, second = list(generator.generate_configs())
    for item in (first, second):
        assert _wire_cache_id(item), f'{llm}: no cache_task_id reached the client'
        assert item.metadata['cache_task_id'] == _wire_cache_id(item)
    assert first.metadata['cache_task_id'] != second.metadata['cache_task_id']
    # account_id2 is retired: nothing may send or record one.
    for item in (first, second):
        assert 'account_id' not in item.metadata
        assert 'account_id2' not in str(item.config.openai_clients['main']['config']['client_args'])


@pytest.mark.parametrize('domain', ['swe', 'frontend'])
@pytest.mark.parametrize(
    ('llm', 'supplied_args', 'expected'),
    [
        ('gpt-6-sol.openai_responses', {'cache_task_id': 'caller'}, 'caller'),
        ('gpt-6-sol.openai_responses', {'cache_task_id': '${oc.env:CALLER_CACHE_TASK_ID}'}, 'from-env'),
        ('gpt-6-sol.openai_responses', {'api_key': 'k?provider=azure&cache_task_id=caller%26one'}, 'caller&one'),
        ('claude-opus-5.litellm', {'api_key': 'k?cache_task_id=caller'}, 'caller'),
        ('claude-opus-5.openai', {'api_key': 'k?cache%5Ftask_id=caller'}, 'caller'),
        ('gpt-6-sol.openai_responses', {'cache_task_id': ''}, None),
        ('gpt-6-sol.openai_responses', {'cache_task_id': None}, None),
        ('gpt-6-sol.openai_responses', {'api_key': 'k?cache_task_id='}, None),
        ('gpt-6-sol.openai_responses', {'cache_task_id': 'argument', 'api_key': 'k?cache_task_id=query'}, 'argument'),
        ('gpt-6-sol.openai_responses', {'cache_task_id': '', 'api_key': 'k?cache_task_id=query'}, 'query'),
        ('gpt-6-sol.openai_responses', {'api_key': 'k?cache_task_id=first&cache_task_id=last'}, 'last'),
    ],
)
def test_generator_records_the_caller_id_after_runner_and_client_preparation(
    env, monkeypatch, tmp_path, domain, llm, supplied_args, expected
) -> None:
    from urllib.parse import parse_qsl

    from nexus.extensions.configs import generator_frontend_template, generator_swe_template
    from nexus.extensions.openai_clients.cache_task_id import prepare_cache_task_id
    from nexus.runner.single import _resolve_dynamic_client_args

    generator = generator_swe_template if domain == 'swe' else generator_frontend_template
    original_load = generator.load_config

    def load_with_caller_id():
        config = original_load()
        config['openai_clients']['main']['config']['client_args'].update(supplied_args)
        return config

    monkeypatch.setattr(generator, 'load_config', load_with_caller_id)
    monkeypatch.setenv('CALLER_CACHE_TASK_ID', 'from-env')
    if domain == 'swe':
        items = _swe_items(tmp_path, monkeypatch, env, llm, key_env={'HR_API_KEY': 'k'})
    else:
        _frontend_two_cases(env, monkeypatch, tmp_path, llm)
        items = list(generator.generate_configs())
    for item in items:
        main = item.config.openai_clients['main']
        run_input = item.config.model_copy(update={'run_id': item.run_id})
        args = _resolve_dynamic_client_args(
            run_input, client_name=main['name'], config_dict=main['config'], run_id_provided=True
        )
        wire_args = prepare_cache_task_id(args)
        wire_id = dict(parse_qsl(wire_args['api_key'].partition('?')[2])).get('cache_task_id')
        assert wire_id == expected
        assert item.metadata.get('cache_task_id') == wire_id
        assert ('cache_task_id' in item.metadata) is (expected is not None)


def test_frontend_no_longer_reads_an_account_pool(env, monkeypatch, tmp_path) -> None:
    """FRONTEND_ACCOUNT_IDS2 was removed with account_id2; setting it must not change the cells."""
    from nexus.extensions.configs import generator_frontend_template as generator

    _frontend_two_cases(env, monkeypatch, tmp_path, 'gpt-61-sol-aws_third.openai_responses')
    before = [(_wire_cache_id(i) is not None, sorted(i.metadata)) for i in generator.generate_configs()]
    monkeypatch.setenv('FRONTEND_ACCOUNT_IDS2', '77638,77874')
    after = [(_wire_cache_id(i) is not None, sorted(i.metadata)) for i in generator.generate_configs()]
    assert before == after
    assert not hasattr(generator, '_aws_third_account_pool')


def _questions_fixture(tmp_path: Path) -> Path:
    """A two-row question set in the eval format: one plain, one with assets and a repo."""
    root = tmp_path / 'questions'
    (root / 'assets' / 'Q-2').mkdir(parents=True)
    (root / 'assets' / 'Q-2' / 'spec.md').write_text('spec')
    path = root / 'q.jsonl'
    path.write_text(
        json.dumps({'id': 'Q-1', 'title': '标题一', 'language': 'zh', 'turns': ['做这个']})
        + '\n'
        + json.dumps({
            'id': 'Q-2',
            'title': '标题二',
            'language': 'zh',
            'turns': ['第一轮', '第二轮'],
            'assets': ['spec.md'],
            'repo': {'url': 'https://example.invalid/r.git', 'commit': 'deadbeef'},
        })
        + '\n'
    )
    return path


def test_frontend_question_mode_assembles_prompt_and_pre_start(
    env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nexus.extensions.configs import generator_frontend_template as generator

    _frontend_cell(env, monkeypatch, tmp_path)
    monkeypatch.setenv('FRONTEND_QUESTIONS', str(_questions_fixture(tmp_path)))
    items = list(generator.generate_configs())
    assert [i.run_id for i in items] == ['Q-1', 'Q-2']

    # One turn: bare text, no "第 N 轮" wrapper.
    assert items[0].config.task_config['prompt'] == '# 标题一\n做这个'
    # Two turns are labelled; assets and repo become notes and pre-start steps.
    prompt2 = items[1].config.task_config['prompt']
    assert '## 第 1 轮' in prompt2
    assert '## 第 2 轮' in prompt2
    assert '本题附件已放在工作区' in prompt2
    assert '仓库已检出到 /workspace' in prompt2
    pre = items[1].config.agent_config['pre_start_scripts']
    assert pre[0]['code'] == 'mkdir -p /workspace'
    assert any('git clone' in (s.get('code') or '') for s in pre)
    assert any('nexus_assets.b64' in (s.get('code') or '') for s in pre)
    # pre_start_scripts belongs to the agent input, not the task's.
    assert 'pre_start_scripts' not in items[1].config.task_config


def test_frontend_question_mode_follows_the_runtime_working_dir(
    env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The scripts and the prompt must follow ``working_dir``, not stay pinned to /workspace.

    The task and the agent already read it from the runtime profile. The questions-mode
    pre-start scripts and prompt had it hardcoded, so setting ``AGENT_WORKING_DIR`` left the
    repo and the assets in /workspace while the agent ran elsewhere and the prompt named the
    wrong directory -- every script still exits 0, so the eval scored it as usual.
    """
    from nexus.extensions.configs import generator_frontend_template as generator

    _frontend_cell(env, monkeypatch, tmp_path)
    monkeypatch.setenv('FRONTEND_QUESTIONS', str(_questions_fixture(tmp_path)))
    monkeypatch.setenv('AGENT_WORKING_DIR', '/srv/app')

    item = list(generator.generate_configs())[1]
    pre = item.config.agent_config['pre_start_scripts']
    codes = '\n'.join(s.get('code') or '' for s in pre)

    assert item.config.task_config['working_dir'] == '/srv/app'
    assert item.config.agent_config['working_dir'] == '/srv/app'
    assert '仓库已检出到 /srv/app' in item.config.task_config['prompt']
    assert pre[0]['code'] == 'mkdir -p /srv/app'
    assert '/workspace' not in codes
    assert '/workspace' not in item.config.task_config['prompt']


def test_a_working_dir_the_pre_start_would_wipe_is_refused() -> None:
    """``rm -rf <working_dir>/*`` is templated, so the value cannot be the root."""
    from nexus.extensions.configs import generator_frontend_template as generator

    for bad in ('/', 'workspace', 'relative/path'):
        with pytest.raises(ValueError, match='absolute path below the root'):
            generator._checked_working_dir({'task_config': {'working_dir': bad}})

    # Unset falls back to the frontend default rather than erroring.
    assert generator._checked_working_dir({}) == '/workspace'


def test_frontend_question_mode_rejects_a_row_without_id_or_turns(
    env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nexus.extensions.configs import generator_frontend_template as generator

    _frontend_cell(env, monkeypatch, tmp_path)
    bad = tmp_path / 'bad.jsonl'
    bad.write_text(json.dumps({'id': 'Q-1', 'turns': []}) + '\n')
    monkeypatch.setenv('FRONTEND_QUESTIONS', str(bad))
    with pytest.raises(ValueError, match='turns'):
        list(generator.generate_configs())


_LLM_YAML = """body:
  imports: [{module}]
  llm:
    family: claude
    not_a_fact: 1
merge:
  imports: append_unique
source: 'test fixture'
"""


def test_a_missing_nexus_module_is_still_a_hard_failure(tmp_path: Path) -> None:
    (tmp_path / 'llms').mkdir()
    bad = tmp_path / 'llms' / 'no_such_module.yaml'
    bad.write_text(_LLM_YAML.format(module='nexus.extensions.openai_clients.definitely_not_a_module_xyz'))
    skips: list[str] = []
    errors = validate.check_file(bad, skips)
    assert skips == [], (errors, skips)
    assert any('definitely_not_a_module_xyz' in e for e in errors), (errors, skips)


def test_a_missing_optional_dependency_is_a_skip_not_a_failure(tmp_path: Path, monkeypatch) -> None:
    # Lean interpreters have no langfuse/litellm installed. That is not a yaml mistake, so only the
    # checks that need the registry are skipped; everything checkable without importing still runs.
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'needs_absent_dep.py').write_text('import absent_third_party_xyz\n')
    monkeypatch.syspath_prepend(str(tmp_path / 'pkg'))

    (tmp_path / 'llms').mkdir()
    path = tmp_path / 'llms' / 'dep_absent.yaml'
    path.write_text(_LLM_YAML.format(module='needs_absent_dep'))
    skips: list[str] = []
    errors = validate.check_file(path, skips)
    assert len(skips) == 1, skips
    assert 'absent_third_party_xyz' in skips[0], skips
    # The registry lookup is skipped, the static checks are not.
    assert any('not_a_fact' in e for e in errors), errors
    assert any('openai_clients.main' in e for e in errors), errors


def _absent_dep_llm(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'needs_absent_dep.py').write_text('import absent_third_party_xyz\n')
    monkeypatch.syspath_prepend(str(tmp_path / 'pkg'))
    (tmp_path / 'llms').mkdir()
    path = tmp_path / 'llms' / 'dep_absent.yaml'
    path.write_text(_LLM_YAML.format(module='needs_absent_dep'))
    return path


def test_strict_turns_a_partly_checked_file_into_a_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    # Without --strict a file checked only in part still ends green. That is what let a run look validated
    # while the llm and runtime files it actually used had only had their outer shell read.
    path = _absent_dep_llm(tmp_path, monkeypatch)
    assert validate.main([str(path)]) == 1  # the fixture has real yaml errors, so it fails either way
    capsys.readouterr()

    good = tmp_path / 'llms' / 'only_skipped.yaml'
    good.write_text(path.read_text())
    monkeypatch.setattr(validate, 'check_file', lambda p, skips=None: skips.append('x: skipped') or [])
    assert validate.main([str(good)]) == 0
    assert validate.main([str(good), '--strict']) == 1
    assert '--strict' in capsys.readouterr().out


def test_include_instances_raises_on_an_id_the_dataset_lacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A named subset that quietly runs one question short still gets scored as the full subset.
    from nexus.extensions.configs import generator_swe_template as generator

    dataset = tmp_path / 'swe.jsonl'
    dataset.write_text(''.join(json.dumps({'instance_id': f'repo__{n}'}) + '\n' for n in ('a', 'b', 'c')))
    monkeypatch.setenv('DATASET_PATH', str(dataset))

    monkeypatch.setenv('INCLUDE_INSTANCES', 'repo__a,repo__b')
    assert [i for i, _, _ in generator._iter_selected_instances()] == ['repo__a', 'repo__b']

    monkeypatch.setenv('INCLUDE_INSTANCES', 'repo__a,repo__nope')
    with pytest.raises(ValueError, match='repo__nope'):
        list(generator._iter_selected_instances())


def test_claude_code_eval_and_rollout_differ_only_in_the_web_tool_deny() -> None:
    import yaml

    agents = SHARED_DIR / 'agents'
    bodies = {n: yaml.safe_load((agents / f'claude_code_{n}.yaml').read_text()) for n in ('eval', 'rollout')}
    deny = ['--disallowedTools', 'WebSearch', 'WebFetch']

    def cli(n: str) -> list[str]:
        return bodies[n]['body']['agent_config']['extra_cli_args']

    assert cli('rollout')[-3:] == deny
    assert not any(a in cli('eval') for a in deny)
    assert cli('rollout')[:-3] == cli('eval')
    for n in bodies:  # strip the one intended difference and the label; nothing else may drift
        bodies[n]['body']['agent_config']['extra_cli_args'] = cli(n)[:4]
        bodies[n].pop('source')
    assert bodies['eval'] == bodies['rollout']


def test_a_caller_supplied_cache_task_id_is_never_overwritten_even_when_falsy() -> None:
    """The docstring promises this; truthiness checking broke it for an explicit empty string."""
    from nexus.extensions.configs.cache_id import cache_task_overlay

    aihub = {'base_url': 'http://api.aihub.woa.com/standard', 'cache_task_id': ''}
    assert cache_task_overlay('openai_responses', aihub, 'KEY', 'ID') == (None, None)


@pytest.mark.parametrize(
    ('api_key', 'expect'),
    [
        ('KEY', 'KEY?cache_task_id=ID'),
        ('KEY?provider=azure', 'KEY?provider=azure&cache_task_id=ID'),
        # A key already ending in a separator must not produce `KEY?&...` or `KEY??...`.
        ('KEY?', 'KEY?cache_task_id=ID'),
        ('KEY?provider=azure&', 'KEY?provider=azure&cache_task_id=ID'),
    ],
)
def test_the_key_query_stays_well_formed(api_key: str, expect: str) -> None:
    from nexus.extensions.configs.cache_id import cache_task_overlay

    # `openai` is not in CLIENTS_TAKING_CACHE_ARG, so the id goes into the key query.
    out, _ = cache_task_overlay('openai', {'base_url': 'http://api.aihub.woa.com/standard'}, api_key, 'ID')
    assert out == {'client_args': {'api_key': expect}}


def test_an_id_with_query_characters_cannot_forge_extra_parameters() -> None:
    """Ids are md5 hex today, so this is defensive -- but the id is an argument, not a constant."""
    from nexus.extensions.configs.cache_id import cache_task_overlay

    out, _ = cache_task_overlay('openai', {'base_url': 'http://api.aihub.woa.com/standard'}, 'KEY', 'x&provider=evil')
    assert out == {'client_args': {'api_key': 'KEY?cache_task_id=x%26provider%3Devil'}}


def test_a_list_cannot_hold_an_llm_ref_that_may_vanish_and_the_message_names_the_cause() -> None:
    """_OMIT has two sources; the error used to blame the optional-ref one for both."""
    from nexus.extensions.configs.load_config import _fill_llm_refs

    with pytest.raises(ValueError, match=r'does not declare cannot sit in a list'):
        _fill_llm_refs(['${llm:nope?}'], {'family': 'gpt'}, path='agent_config.x')

    with pytest.raises(ValueError, match=r'llm\.ctx is declared null'):
        _fill_llm_refs(['${llm:ctx}'], {'family': 'gpt', 'ctx': None}, path='agent_config.x', drop_null=True)


@pytest.mark.parametrize(('limit', 'expect'), [('0', []), ('1', ['repo__a']), ('2', ['repo__a', 'repo__b'])])
def test_swe_limit_zero_selects_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit, expect) -> None:
    """The bound was checked after the yield, so SWE_LIMIT=0 shipped one instance."""
    from nexus.extensions.configs import generator_swe_template as generator

    dataset = tmp_path / 'swe.jsonl'
    dataset.write_text(''.join(json.dumps({'instance_id': f'repo__{n}'}) + '\n' for n in ('a', 'b', 'c')))
    monkeypatch.setenv('DATASET_PATH', str(dataset))
    monkeypatch.setenv('SWE_LIMIT', limit)
    assert [i for i, _, _ in generator._iter_selected_instances()] == expect


@pytest.mark.parametrize('module', ['generator_swe_template', 'generator_frontend_template'])
@pytest.mark.parametrize('name', ['SWE_LIMIT', 'SWE_SKIP', 'FRONTEND_LIMIT', 'EVAL_TIMES'])
def test_a_non_integer_limit_is_an_error_not_the_default(
    monkeypatch: pytest.MonkeyPatch, module: str, name: str
) -> None:
    """``SWE_LIMIT=10x`` used to read as "no limit" and ship the whole dataset.

    ``_env_int`` swallowed the ``ValueError`` and returned ``default``, which for the
    limits is ``None`` -- the unbounded case. The error has to name the variable so the
    message says which knob to fix.
    """
    generator = importlib.import_module(f'nexus.extensions.configs.{module}')

    monkeypatch.setenv(name, '10x')
    with pytest.raises(ValueError, match=rf'{name}=.10x.'):
        generator._env_int(name)


def test_the_validate_tally_never_claims_every_file_passed_above_a_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Under --strict the tally was printed before half-checked files were counted as failures."""
    from nexus.extensions.configs import validate

    good = SHARED_DIR / 'agents' / 'mini_swe_agent.yaml'
    skipped = 'pretend-missing-dependency.yaml'

    real_check_file = validate.check_file

    def check_file(path, skips):
        skips.append(skipped)
        return real_check_file(path, [])

    validate.check_file = check_file
    try:
        assert validate.main([str(good), '--strict']) == 1
    finally:
        validate.check_file = real_check_file

    out = capsys.readouterr().out
    tally = next(line for line in out.splitlines() if line.startswith('files:'))
    assert tally == 'files: 0/1 通过', f'tally must already account for the strict skip, got {tally!r}'
    assert 'FAIL --strict' in out


@pytest.mark.parametrize('llm', sorted(p.stem for p in (SHARED_DIR / 'llms').glob('*.yaml')))
def test_every_aihub_wire_pins_its_vendor(llm: str) -> None:
    """Unpinned, AIHub 2.0 picks a vendor per request (gpt-5.6-sol: azure or openai; opus-5 has
    routed to google-vertex), so one trajectory can mix vendors. validate.py rejects it."""
    import yaml

    from nexus.extensions.configs.validate import _aihub_provider_errors

    body = yaml.safe_load((SHARED_DIR / 'llms' / f'{llm}.yaml').read_text())['body']
    for client in (body.get('openai_clients') or {}).values():
        assert _aihub_provider_errors(client) == []


@pytest.mark.parametrize(
    ('client_args', 'pinned'),
    [
        ({'api_key': '${oc.env:HR_API_KEY}', 'base_url': 'http://api.aihub.woa.com/standard'}, False),
        ({'api_key': '${oc.env:HR_API_KEY}?provider=aws_third', 'base_url': 'http://api.aihub.woa.com/standard'}, True),
        (
            {
                'api_key': 'k',
                'endpoint': 'http://api.aihub.woa.com/v1/responses',
                'gateway_routing': {'provider': 'deepseek'},
            },
            True,
        ),
        ({'api_key': 'k', 'base_url': 'http://infer-api.taiji.woa.com/api/v1'}, True),
    ],
)
def test_aihub_vendor_pin_check(client_args: dict, pinned: bool) -> None:
    from nexus.extensions.configs.validate import _aihub_provider_errors

    assert (_aihub_provider_errors({'config': {'client_args': client_args}}) == []) is pinned
