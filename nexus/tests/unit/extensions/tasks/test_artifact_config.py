"""Tests for the artifact config single source of truth.

The ``clean_env`` fixture is load-bearing: a developer shell that has sourced
``artifact/.env`` would otherwise poison every default assertion and make the suite
order-dependent.
"""

from __future__ import annotations

import json
import re

from pathlib import Path

import pytest

from nexus.extensions.tasks.artifact import config


# Every env name the config reads (primary + legacy), so a real .env cannot leak in.
_ALL_ENV_NAMES = sorted({
    *config.LEGACY_ALIASES.keys(),
    *config.LEGACY_ALIASES.values(),
    'ARTIFACT_DATASET_PATH',
    'ARTIFACT_INSTANCE_IDS',
    'ARTIFACT_SKIP',
    'ARTIFACT_LIMIT',
    'ARTIFACT_IMAGE',
    'ARTIFACT_INFER_DB',
    'ARTIFACT_SKIP_INFER',
    'ARTIFACT_SKIP_EVAL',
    'ARTIFACT_RUNTIME_PROVIDER_NAME',
    'ARTIFACT_RUNTIME_PROVIDER',
    'ARTIFACT_RUNTIME_TIMEOUT',
    'ARTIFACT_AGENT_TIMEOUT',
    'ARTIFACT_MCP_SERVERS',
    'ARTIFACT_CC_HOOKS',
    'ARTIFACT_CLAUDE_PLUGINS',
    'ARTIFACT_SKILLS_DIR',
    'ARTIFACT_SKILL_NAME',
    'ARTIFACT_INFER_MODEL',
    'ARTIFACT_JUDGE_MODEL',
    'ARTIFACT_VLM_MODEL',
    'ARTIFACT_LLM_ROUTER_ENABLE',
    'ARTIFACT_NO_THINKING_MODEL',
    'ARTIFACT_HAIKU_MODEL',
    'ARTIFACT_JUDGE_TIMEOUT',
    'ARTIFACT_JUDGE_MAX_TURNS',
    'ARTIFACT_JUDGE_WORKING_DIR',
    'ARTIFACT_DEPLOY_EXPIRED_TIME',
    'ARTIFACT_DEPLOY_POLL_TIMEOUT',
    'ARTIFACT_DEPLOY_POLL_INTERVAL',
    'ARTIFACT_DISABLE_LANGFUSE',
    'ARTIFACT_LANGFUSE_EXPERIMENT',
    'ARTIFACT_RUN_TAG',
    'ARTIFACT_SAVE_SCREENSHOTS_DIR',
    'ARTIFACT_BATCH_TIMEOUT',
    'LANGFUSE_PUBLIC_KEY',
    'LANGFUSE_SECRET_KEY',
    'LANGFUSE_BASE_URL',
    'LANGFUSE_HOST',
    'NEXUS_CONVERSION_LOG_LEVEL',
})


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in _ALL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    config.reset()
    yield monkeypatch
    config.reset()


def test_defaults_snapshot(clean_env):
    """Regression net: these are the values the pipelines behaved with before the
    config refactor, so any accidental default change fails here."""
    cfg = config.load()
    assert (cfg.runtime.runtime_timeout, cfg.runtime.agent_timeout) == (4000, 1800)
    assert (cfg.judge.timeout, cfg.judge.max_turns) == (7200.0, 200)
    assert cfg.judge.working_dir == '/workspace'
    assert cfg.judge.conversion_log_level == 'DEBUG'
    assert cfg.runtime.infer_conversion_log_level == 'INFO'  # 两个 harness 共用,故归 runtime
    assert (cfg.deploy.expired_time, cfg.deploy.poll_timeout, cfg.deploy.poll_interval) == (3600, 300.0, 5.0)
    assert cfg.models.litellm_timeout == 1800.0
    assert cfg.models.openapi_temp == 0.9
    assert cfg.models.summarized_thinking is True
    assert cfg.models.max_tokens is None
    assert cfg.runtime.provider_name == 'gongfeng'
    assert cfg.obs.langfuse_experiment == 'artifact'
    assert cfg.data.limit is None
    assert cfg.parse_errors == ()


@pytest.mark.parametrize(
    ('new_name', 'legacy_name', 'read'),
    [
        ('ARTIFACT_LITELLM_TIMEOUT', 'LITELLM_TIMEOUT', lambda c: c.models.litellm_timeout),
        ('ARTIFACT_BATCH_TIMEOUT', 'ATOMIC_RUN_TIMEOUT', lambda c: c.budget.batch_timeout),
    ],
)
def test_alias_precedence(clean_env, new_name, legacy_name, read):
    clean_env.setenv(legacy_name, '111')
    assert read(config.load()) == 111.0, 'legacy name alone must still work'

    clean_env.setenv(new_name, '222')
    assert read(config.load()) == 222.0, 'new name must win when both are set'

    clean_env.delenv(new_name)
    clean_env.delenv(legacy_name)
    cfg = config.load()
    assert read(cfg) not in (111.0, 222.0), 'neither set → default'


def test_legacy_name_is_recorded_and_warned(clean_env):
    clean_env.setenv('LITELLM_TIMEOUT', '1200')
    cfg = config.load()
    assert cfg.provenance['models.litellm_timeout'] == 'LITELLM_TIMEOUT'
    messages = [p.message for p in config.validate(cfg) if p.field == 'aliases']
    assert any('LITELLM_TIMEOUT→ARTIFACT_LITELLM_TIMEOUT' in m for m in messages)


def test_parse_error_is_recorded_not_raised(clean_env):
    clean_env.setenv('ARTIFACT_AGENT_TIMEOUT', 'abc')
    cfg = config.load()
    assert cfg.runtime.agent_timeout == 1800, 'falls back to the default'
    assert any('ARTIFACT_AGENT_TIMEOUT' in e for e in cfg.parse_errors)
    problems = config.validate(cfg)
    assert any(p.level == 'error' and 'ARTIFACT_AGENT_TIMEOUT' in p.message for p in problems)


@pytest.mark.parametrize(('raw', 'expected'), [('2400', 2400), ('2400.0', 2400), (' 5000', 5000), ('-5', -5)])
def test_int_accepts_float_and_whitespace(clean_env, raw, expected):
    """The old ``_env_int`` used ``isdigit()``, so '2400.0' and '-5' were silently
    dropped and fell back to the default — a real "I set it and it was ignored" bug."""
    clean_env.setenv('ARTIFACT_AGENT_TIMEOUT', raw)
    assert config.load().runtime.agent_timeout == expected


def test_zero_limit_means_no_limit(clean_env):
    """run_batch.artifact.sh documents ``0 = no limit``; the old code treated 0 as
    "stop immediately" in generate_configs while get_total_count returned everything."""
    clean_env.setenv('ARTIFACT_LIMIT', '0')
    assert config.effective_limit(config.load()) is None


def test_validate_warns_judge_timeout_exceeds_batch(clean_env):
    clean_env.setenv('ARTIFACT_BATCH_TIMEOUT', '3600')
    problems = config.validate(config.load())
    assert any(p.level == 'warning' and p.field == 'judge.timeout' for p in problems)


def test_validate_flags_agent_outliving_runtime(clean_env):
    clean_env.setenv('ARTIFACT_AGENT_TIMEOUT', '5000')
    clean_env.setenv('ARTIFACT_RUNTIME_TIMEOUT', '4000')
    problems = config.validate(config.load())
    assert any(p.level == 'error' and p.field == 'runtime.agent_timeout' for p in problems)


def test_validate_warns_host_nexus_conversion_log_level(clean_env):
    clean_env.setenv('NEXUS_CONVERSION_LOG_LEVEL', 'INFO')
    problems = config.validate(config.load())
    assert any('has no effect' in p.message for p in problems)


def test_validate_rejects_unknown_role_alias(clean_env):
    clean_env.setenv('ARTIFACT_JUDGE_MODEL', 'nope')
    problems = config.validate(config.load(), model_aliases={'gemini35', 'opus48'})
    errors = [p for p in problems if p.field == 'models.judge_alias']
    assert errors
    assert 'gemini35' in errors[0].message, 'must list the known aliases'


def test_get_is_cached_and_reset_reloads(clean_env):
    clean_env.setenv('ARTIFACT_JUDGE_TIMEOUT', '111')
    assert config.get().judge.timeout == 111.0
    clean_env.setenv('ARTIFACT_JUDGE_TIMEOUT', '222')
    assert config.get().judge.timeout == 111.0, 'cached until reset'
    config.reset()
    assert config.get().judge.timeout == 222.0


def test_hook_env_table_is_complete():
    """Mechanically enforce "one place to look": the container-injected hook scripts
    cannot read this module, so their env names must at least be INDEXED here."""
    root = Path(config.__file__).parent / 'agent_setup'
    documented = {doc.name for doc in config.HOOK_ENV}
    prefixes = {name.rstrip('*') for name in documented if name.endswith('*')}
    missing: set[str] = set()
    for filename in ('stop_hook.py', 'askuser_hook_v2.py'):
        path = root / filename
        if not path.is_file():
            continue
        for found in re.findall(r"os\.environ(?:\.get)?[\[(]'([A-Z_]+)'", path.read_text()):
            if not found.startswith(('NEXUS_STOP_HOOK', 'NEXUS_ASKUSER_HOOK')):
                continue  # ANTHROPIC_* etc. are documented in EXTERNAL_ENV
            if found in documented or any(found.startswith(p) for p in prefixes):
                continue
            missing.add(found)
    assert not missing, f'hook env vars missing from config.HOOK_ENV: {sorted(missing)}'


def test_cli_report_redacts_secrets(clean_env):
    sentinel = 'SUPERSECRETVALUE123'
    clean_env.setenv('LANGFUSE_SECRET_KEY', sentinel)
    clean_env.setenv('ARTIFACT_GONGFENG_TOKEN', sentinel)
    cfg = config.load()
    report = config.format_report(cfg, config.validate(cfg))
    assert sentinel not in report, 'CLI output gets pasted into chat; never echo secrets'
    assert 'set (len=' in report
    for tier, _note, _groups in config._TIERS:
        assert tier in report


def test_cli_json_round_trips(clean_env, capsys):
    assert config.main(['--json']) in (0, 1)
    payload = json.loads(capsys.readouterr().out)
    assert 'config' in payload
    assert 'problems' in payload


def test_agent_budgets_override_the_core_judge_params(clean_env):
    """预算靠给唯一那个 factory 传参来覆盖 —— 不给每条链路加自己的 timeout/max_turns 配置，
    也不手改 extra_cli_args 里 ``--max-turns`` 的下一个元素（那段代码曾被复制 4 份）。

    原来测三个 probe planner 的常量，它们随 probe/atomic_probe/orbit_probe 一起删了；
    约束本身仍然有效，改测存活的调用方 web_judger.flow2code。"""
    from nexus.extensions.tasks.artifact.agent_setup.claude_config import default_judge_claude_input
    from nexus.extensions.tasks.artifact.judges.strategies.web_judger import executor

    config.reset()
    claude_input = default_judge_claude_input(
        timeout=executor.FLOW2CODE_TIMEOUT, max_turns=executor.FLOW2CODE_MAX_TURNS
    )
    args = claude_input.extra_cli_args
    assert claude_input.timeout == executor.FLOW2CODE_TIMEOUT
    assert args[args.index('--max-turns') + 1] == str(executor.FLOW2CODE_MAX_TURNS)

    # unset → inherit the core judge defaults
    assert default_judge_claude_input().timeout == config.get().judge.timeout


# ── YAML profiles ─────────────────────────────────────────────────────────────


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / 'p.yaml'
    path.write_text(body, encoding='utf-8')
    return path


def test_profile_and_equivalent_env_agree(clean_env, tmp_path):
    """The migration gate: a profile must produce the same config as the env it replaces.

    Compares every group rather than a sample, so a reader that handles one type wrongly
    (bool / csv / int) cannot slip through.
    """
    clean_env.setenv('ARTIFACT_AGENT_NAME', 'hycode')
    clean_env.setenv('ARTIFACT_HYCODE_VERSION', '0.2.0')
    clean_env.setenv('ARTIFACT_JUDGE_MAX_TURNS', '77')
    clean_env.setenv('ARTIFACT_JUDGE_TIMEOUT', '123.5')
    clean_env.setenv('ARTIFACT_MCP_SERVERS', 'playwright,figma')
    clean_env.setenv('ARTIFACT_HYCODE_ALLOW_PLAN_MODE', '1')
    clean_env.setenv('ARTIFACT_LIMIT', '5')
    config.reset()
    from_env = config.load()

    from_yaml = config.load(
        _write(
            tmp_path,
            """
            runtime: {agent_name: hycode}
            hycode: {version: '0.2.0', allow_plan_mode: true}
            judge: {max_turns: 77, timeout: 123.5}
            claude_code: {mcp_servers: [playwright, figma]}
            data: {limit: 5}
            """,
        )
    )

    for group in ('data', 'runtime', 'claude_code', 'hycode', 'models', 'judge', 'deploy', 'obs', 'budget'):
        assert getattr(from_yaml, group) == getattr(from_env, group), group


def test_set_overrides_profile_which_overrides_env(clean_env, tmp_path):
    clean_env.setenv('ARTIFACT_HYCODE_VERSION', 'from-env')
    profile = _write(tmp_path, 'hycode: {version: from-yaml}\n')
    config.reset()

    assert config.load().hycode.version == 'from-env'
    assert config.load(profile).hycode.version == 'from-yaml'
    assert config.load(profile, ['hycode.version=from-set']).hycode.version == 'from-set'


def test_profile_records_which_env_it_shadowed(clean_env, tmp_path):
    """A leftover `export` must not look like it is still in effect."""
    clean_env.setenv('ARTIFACT_HYCODE_VERSION', 'from-env')
    profile = _write(tmp_path, 'hycode: {version: from-yaml}\n')
    config.reset()

    cfg = config.load(profile)

    assert cfg.shadowed_env['hycode.version'] == 'ARTIFACT_HYCODE_VERSION'
    assert 'ARTIFACT_HYCODE_VERSION' in config.format_report(cfg, config.validate(cfg))


def test_unknown_profile_key_is_reported_not_ignored(clean_env, tmp_path):
    """A typo'd key would otherwise leave the default in place and look like a no-op."""
    cfg = config.load(_write(tmp_path, 'hycode: {verison: 0.2.0}\n'))

    assert any('verison' in e for e in cfg.parse_errors)
    assert any(p.level == 'error' and 'verison' in p.message for p in config.validate(cfg))


def test_missing_profile_is_reported(clean_env, tmp_path):
    cfg = config.load(tmp_path / 'nope.yaml')

    assert any('not found' in e for e in cfg.parse_errors)


def test_omitted_key_keeps_the_dataclass_default(clean_env, tmp_path):
    """Profiles are allowed to be partial; anything absent stays at the declared default."""
    cfg = config.load(_write(tmp_path, 'hycode: {version: 0.2.0}\n'))

    assert cfg.hycode.version == '0.2.0'
    assert cfg.judge.max_turns == config.JudgeCfg().max_turns
    assert cfg.hycode.npm_registry == config.HyCodeCfg().npm_registry


def test_shipped_profile_examples_have_no_unknown_keys(clean_env):
    """The committed templates double as the schema doc, so a stale key there is a real bug."""
    profiles = Path(config.__file__).parent / 'configs' / 'legacy'
    examples = sorted(profiles.glob('*.yaml.example'))
    assert {example.name for example in examples} == {
        'claude_code.yaml.example', 'hycode.yaml.example',
        'dynamicagent_html.yaml.example', 'dynamicagent_react.yaml.example',
    }
    for example in examples:
        _values, _sources, problems = config.load_profile(example)
        assert problems == [], f'{example.name}: {problems}'


def test_batch_args_come_from_the_profile(clean_env, tmp_path, capsys):
    """``batch.*`` reaches run-batch as CLI flags: cli/commands/run.py builds BatchConfig from
    argv before the generator loads, so the generator cannot feed these back."""
    profile = _write(tmp_path, 'batch: {worker_processes: 3, max_concurrent: 96, db: /tmp/x.db}\n')

    assert config.main(['-p', str(profile), '--emit-batch-args']) == 0

    flags = capsys.readouterr().out.split()
    assert flags[flags.index('--worker-processes') + 1] == '3'
    assert flags[flags.index('--max-concurrent') + 1] == '96'
    assert flags[flags.index('-d') + 1] == '/tmp/x.db'
    assert '--timeout' not in flags  # batch_timeout=0 means "not wired"


def test_batch_timeout_is_forwarded_only_when_set(clean_env, tmp_path, capsys):
    profile = _write(tmp_path, 'budget: {batch_timeout: 300}\n')

    config.main(['-p', str(profile), '--emit-batch-args'])

    flags = capsys.readouterr().out.split()
    assert flags[flags.index('--timeout') + 1] == '300.0'


def test_generator_subprocess_picks_up_profile_and_overrides(clean_env, tmp_path):
    """``run.sh`` resolves the profile but ``nexus run-batch`` spawns the generator, so both the
    profile path and the ``--set`` values have to survive a process boundary as env."""
    profile = _write(tmp_path, 'hycode: {version: from-yaml}\njudge: {max_turns: 5}\n')
    clean_env.setenv(config.PROFILE_ENV, str(profile))
    clean_env.setenv(config.OVERRIDES_ENV, 'hycode.version=from-set\njudge.max_turns=9')
    config.reset()

    cfg = config.get()

    assert cfg.hycode.version == 'from-set'
    assert cfg.judge.max_turns == 9


def test_provider_and_provider_name_are_different_concepts(clean_env):
    """``provider_name`` picks the provider class; ``provider`` is gongfeng's own scheduling
    strategy. The names look interchangeable, so an edit meant to switch sandboxes lands on the
    wrong one — validate() has to say which is which."""
    cfg = config.load(None, ['runtime.provider=docker'])
    errors = [p for p in config.validate(cfg) if p.field == 'runtime.provider']
    assert errors
    assert errors[0].level == 'error'
    assert 'provider_name' in errors[0].message, 'must point at the right field'

    for ok in ('gongfeng', 'suanli', ''):
        cfg = config.load(None, [f'runtime.provider={ok}'])
        assert not [p for p in config.validate(cfg) if p.field == 'runtime.provider' and p.level == 'error']


def test_provider_is_flagged_as_ignored_for_non_gongfeng(clean_env):
    """generator only forwards it when provider_name is gongfeng, so anywhere else it is inert."""
    cfg = config.load(None, ['runtime.provider_name=docker', 'runtime.provider=suanli'])
    warnings = [p for p in config.validate(cfg) if p.field == 'runtime.provider' and p.level == 'warning']
    assert warnings
    assert 'ignored' in warnings[0].message


def test_model_table_from_yaml_needs_no_env(clean_env, tmp_path):
    """A profile carries the whole model table, so a run needs no sourced .env at all."""
    from nexus.extensions.tasks.artifact.agent_setup import clients

    for name in [k for k in list(__import__('os').environ) if k.startswith('M_')]:
        clean_env.delenv(name, raising=False)
    profile = _write(
        tmp_path,
        """
        models:
          infer_alias: g35
          judge_alias: o48
          table:
            g35: {provider: gemini, model: openai/google/gemini-3.5-flash, base: 'http://a', key: k1, effort: low}
            o48: {provider: opus, model: bedrock/anthropic.claude-opus-4-8, base: 'http://b', key: k2, effort: max}
        """,
    )
    clean_env.setenv(config.PROFILE_ENV, str(profile))
    config.reset()

    assert sorted(clients.parse_model_table_from_env()) == ['g35', 'o48']
    built = clients.build_named_clients_from_env()
    assert built['main']['config']['request_args']['model'] == 'gemini/gemini-3.5-flash'
    assert built['judge']['config']['request_args']['model'] == 'bedrock/anthropic.claude-opus-4-8'


def test_model_table_falls_back_to_env_groups(clean_env):
    """An existing ``M_<alias>_*`` .env keeps working when the profile has no table."""
    from nexus.extensions.tasks.artifact.agent_setup import clients

    clean_env.setenv('M_fromenv_PROVIDER', 'gemini')
    clean_env.setenv('M_fromenv_MODEL', 'openai/google/gemini-3.5-flash')
    clean_env.setenv('M_fromenv_BASE', 'http://a')
    clean_env.setenv('M_fromenv_KEY', 'k')
    config.reset()

    assert 'fromenv' in clients.parse_model_table_from_env()


def test_table_is_kept_whole_not_flattened(clean_env, tmp_path):
    """``models.table`` is a mapping field: flattening it per leaf would turn each entry into
    an unknown key and drop the table entirely."""
    cfg = config.load(_write(tmp_path, 'models:\n  table:\n    a: {provider: gemini, model: m}\n'))

    assert cfg.models.table == {'a': {'provider': 'gemini', 'model': 'm'}}
    assert cfg.parse_errors == ()


def test_alias_check_uses_the_loaded_profile_not_the_env(clean_env, tmp_path):
    """Validating a profile has to resolve aliases from that profile's own table.

    Reading them from the cached global instead reported every alias in the profile as
    "no M_<alias>_PROVIDER env group" — a wall of errors on a perfectly good profile.
    """
    profile = _write(
        tmp_path,
        """
        models:
          judge_alias: only_in_yaml
          table:
            only_in_yaml: {provider: gemini, model: m, base: 'http://a', key: k}
        """,
    )
    cfg = config.load(profile)

    assert config.model_aliases(cfg) == {'only_in_yaml'}
    assert not [p for p in config.validate(cfg, model_aliases=config.model_aliases(cfg)) if p.level == 'error']


def test_runtime_image_without_registry_prefix_is_rejected(clean_env):
    """A bare image name resolves against docker.io: the sandbox dies with ErrImagePull ~40s
    into a real run, so this has to fail at validate time instead."""
    cfg = config.load(None, ['runtime.runtime_image=nexus-runtime:feat-cc-support-proxy-convert'])
    errors = [p for p in config.validate(cfg) if p.field == 'runtime.runtime_image']
    assert errors
    assert errors[0].level == 'error'

    for ok in ('current', 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.4.0'):
        cfg = config.load(None, [f'runtime.runtime_image={ok}'])
        assert not [p for p in config.validate(cfg) if p.field == 'runtime.runtime_image']
