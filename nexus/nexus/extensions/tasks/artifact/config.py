"""Artifact configuration schema, environment bindings, and validation.

Precedence: overrides > YAML profile > environment > dataclass defaults.
HOOK_ENV and EXTERNAL_ENV index independently consumed settings. Dynamic
model groups are resolved by clients; runtime-image constants live in
claude_config. Keep module-level imports lightweight.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar


# ── env reading ───────────────────────────────────────────────────────────────
# Every reader takes the dotted config field name first (for provenance), then one or
# more env names: THE FIRST NAME THAT IS SET WINS. That is how the new ``ARTIFACT_*``
# name takes precedence while a legacy name keeps working.


class _Reader:
    """Env + YAML reader that records provenance and parse errors.

    Layer precedence per field: ``--set`` > YAML profile > env > dataclass default. The
    overlay (YAML + ``--set``) is consulted first; env is the legacy layer, so an existing
    all-env run behaves exactly as before while a profile takes over once one is passed.

    Instance-scoped (not global) so it is reentrant and thread-safe, and so the CLI and
    a live run trace independently.
    """

    def __init__(self, overlay: dict[str, Any] | None = None, overlay_sources: dict[str, str] | None = None) -> None:
        self.provenance: dict[str, str] = {}
        self.parse_errors: list[str] = []
        self.shadowed_env: dict[str, str] = {}
        self._overlay = overlay or {}
        self._overlay_sources = overlay_sources or {}

    def _overlay_hit(self, field_name: str, names: tuple[str, ...]) -> tuple[Any, bool]:
        """Return ``(value, True)`` when a YAML/--set layer supplies *field_name*."""
        if field_name not in self._overlay:
            return None, False
        self.provenance[field_name] = self._overlay_sources.get(field_name, 'yaml')
        # Note which env vars the profile is overriding, so a stale `export` cannot look
        # like it is still in effect.
        for name in names:
            if os.getenv(name, '').strip():
                self.shadowed_env[field_name] = name
                break
        return self._overlay[field_name], True

    def _raw(self, field_name: str, names: tuple[str, ...]) -> tuple[str | None, str]:
        for name in names:
            value = os.getenv(name)
            if value is not None and value.strip():
                self.provenance[field_name] = name
                return value.strip(), name
        return None, names[0] if names else ''

    def s(self, field_name: str, *names: str) -> str | None:
        value, found = self._overlay_hit(field_name, names)
        if found:
            return None if value is None else str(value)
        raw, _ = self._raw(field_name, names)
        return raw

    def i(self, field_name: str, *names: str) -> int | None:
        value, found = self._overlay_hit(field_name, names)
        if found:
            return self._coerce_int(field_name, value)
        raw, name = self._raw(field_name, names)
        if raw is None:
            return None
        return self._coerce_int(name, raw)

    def f(self, field_name: str, *names: str) -> float | None:
        value, found = self._overlay_hit(field_name, names)
        if not found:
            value, name = self._raw(field_name, names)
            if value is None:
                return None
        else:
            name = field_name
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            self.parse_errors.append(f'{name}={value!r} is not a number (falling back to the default)')
            return None

    def i_opt(self, field_name: str, *names: str) -> int | None:
        """An int that may legitimately be absent (e.g. a limit of "no limit")."""
        value, found = self._overlay_hit(field_name, names)
        if found:
            return self._coerce_int(field_name, value)
        raw, name = self._raw(field_name, names)
        if raw is None:
            return None
        return self._coerce_int(name, raw)

    def b(self, field_name: str, *names: str) -> bool | None:
        value, found = self._overlay_hit(field_name, names)
        if found:
            if isinstance(value, bool) or value is None:
                return value
            return str(value).strip().lower() in _TRUTHY
        raw, _ = self._raw(field_name, names)
        if raw is None:
            return None
        return raw.lower() in _TRUTHY

    def csv(self, field_name: str, *names: str) -> tuple[str, ...] | None:
        value, found = self._overlay_hit(field_name, names)
        if found:
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return tuple(str(v).strip() for v in value if str(v).strip())
            return tuple(p.strip() for p in str(value).split(',') if p.strip())
        raw, _ = self._raw(field_name, names)
        if raw is None:
            return None
        return tuple(part.strip() for part in raw.split(',') if part.strip())

    def mapping(self, field_name: str) -> dict[str, dict[str, str]] | None:
        """A whole mapping supplied by YAML (no env equivalent — env cannot nest).

        Values are stringified so a YAML ``effort: 1`` behaves like the env path's ``'1'``.
        """
        value, found = self._overlay_hit(field_name, ())
        if not found or value is None:
            return None
        if not isinstance(value, dict):
            self.parse_errors.append(f'{field_name} must be a mapping, got {type(value).__name__}')
            return None
        out: dict[str, dict[str, str]] = {}
        for key, entry in value.items():
            if not isinstance(entry, dict):
                self.parse_errors.append(f'{field_name}.{key} must be a mapping, got {type(entry).__name__}')
                continue
            out[str(key)] = {str(k): '' if v is None else str(v) for k, v in entry.items()}
        return out

    def _coerce_int(self, label: str, value: Any) -> int | None:
        if value is None:
            return None
        try:
            # int(float(...)) so '2400.0' and ' 5000' work — the old isdigit() check
            # silently dropped both and fell back to the default.
            return int(float(value))
        except (TypeError, ValueError):
            self.parse_errors.append(f'{label}={value!r} is not a number (falling back to the default)')
            return None


_TRUTHY = {'1', 'true', 'yes', 'y', 'on'}

# Accepted values of runtime.provider — gongfeng's own scheduling strategies, per
# runtime/providers/gongfeng.py:364. Not provider class names.
_GONGFENG_STRATEGIES = {'gongfeng', 'suanli'}

# Accepted values of judge.aesthetic_mode. Kept here (not imported from the verifier) so
# validating a profile never pulls the judging stack in.
AESTHETIC_MODES = {'absolute', 'pairwise'}


_CfgT = TypeVar('_CfgT')


def _build(cls: type[_CfgT], **kwargs: Any) -> _CfgT:
    """Construct a config group, dropping unset (``None``) values.

    Dropping rather than passing them through is what makes the dataclass field default the
    single definition of "the default". A ``None`` here always means "no env set it" — the two
    fields where ``None`` is a real value (``limit``, ``max_tokens``) default to ``None``
    anyway, so omitting them lands on the same value.
    """
    return cls(**{k: v for k, v in kwargs.items() if v is not None})


# ── config groups ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DataCfg:
    """Tier 1 — which rows to run. Every run script sets these."""

    dataset_path: str = ''
    """JSONL dataset, one instance per line. Required for an offline batch."""

    instance_ids: tuple[str, ...] = ()
    """Run only these instance_ids. Empty = the whole dataset."""

    skip: int = 0
    """Skip the first N rows, for paging through a large dataset."""

    limit: int | None = None
    """Cap the number of rows. ``None`` and ``0`` both mean no limit."""

    image: str = ''
    """Global business-container override. '' = use each row's own ``image`` field.

    Distinct from ``runtime.runtime_image``, which is the sandbox runtime — both have to be
    right, and they fail in different ways.
    """

    infer_db: str = ''
    """A previous run's db to read ``archive_b64`` from. Pairs with ``skip_infer``."""

    skip_infer: bool = False
    """Skip generation and judge an existing archive (needs ``infer_db``)."""

    skip_eval: bool = False
    """Run infer only and store the archive, to judge later."""


@dataclass(frozen=True, slots=True)
class RuntimeCfg:
    """Tier 1/2 — sandbox provider + the outer agent/runtime timeouts."""

    provider_name: str = 'gongfeng'
    """**Which provider class** starts the sandbox: ``gongfeng`` / ``docker`` / ``local`` / …

    Looked up in ``runtime_provider_registry`` (``runner/single.py:434``). Not the same thing
    as ``provider`` below — the similar names are historical.
    """

    provider: str = 'gongfeng'
    """**Resource scheduling strategy inside the gongfeng backend**: ``gongfeng`` / ``suanli``,
    or '' to let the backend choose.

    Only read when ``provider_name == 'gongfeng'``; it becomes the ``runtime_provider`` field of
    the gongfeng payload (``runtime/providers/gongfeng.py:1123``). Setting it to a provider
    class name like ``docker`` is a config error, not a way to switch sandboxes — validate()
    warns. Defaults to mirroring ``provider_name`` only because both spell 'gongfeng'.
    """

    agent_name: str = 'claude_code'
    """Which infer harness to launch: ``claude_code`` or ``hycode``.

    Selects which of the ``claude_code`` / ``hycode`` config groups the builder reads.
    """

    gongfeng_token: str = ''
    """Sandbox API token. Only read when the provider is gongfeng."""

    runtime_image: str = 'current'
    """Sandbox runtime image. ``'current'`` resolves to the installed package version and
    does **not** carry the Anthropic-proxy protocol conversion that hycode needs; that wants
    a ``nexus-runtime:feat-cc-support-proxy-convert``-style image. Getting this wrong shows
    up as auth / IAM errors from the LLM call, which looks like a credential problem.
    """

    runtime_timeout: int = 4000
    """Sandbox wall clock, in seconds. Must exceed ``agent_timeout`` — validate() checks it."""

    agent_timeout: int = 1800
    """The agent's own execution budget, in seconds."""

    infer_conversion_log_level: str = 'INFO'
    """Anthropic<->OpenAI conversion log level for the infer agent. Applies to whichever
    harness runs — hycode is claude_code's fork and reuses the same switch."""


@dataclass(frozen=True, slots=True)
class ClaudeCodeCfg:
    """Tier 3/4 — claude-code version + opt-in scaffolding (all default off).

    The hook knobs below are forwarded to the in-container scripts as ``NEXUS_*`` env by
    ``claude_hooks.build_hook_env``. An empty / zero value means "not set": the env var is
    not emitted and the script's own default stands, so each default lives in exactly one
    place (the script) instead of being restated here.
    """

    version: str = ''  # '' → claude_config.DEFAULT_CLAUDE_CODE_VERSION
    binary_url: str = ''
    mcp_servers: tuple[str, ...] = ()
    figma_token: str = ''

    hooks_enabled: bool = False
    """Master switch for the PreToolUse hooks. Every knob below needs it on."""

    askuser_hook_version: str = 'v2'
    """Which AskUserQuestion script to inject: ``v1`` always picks the first option,
    ``v2`` asks a model to pick a context-aware one (falling back to v1 behaviour)."""

    askuser_hook_disable_llm: bool = False
    """Inject v2 but skip its model call (equivalent to v1's behaviour). Distinct from
    ``askuser_hook_version='v1'``, which injects the other file entirely."""

    askuser_hook_model: str = ''
    """Model for v2's simulated user. '' → the script's own default."""

    stop_hook: bool = False
    """Multi-turn rollout: on each Stop event a model plays the user and pushes the agent
    on. Needs ``hooks_enabled``; when off the script is not injected at all."""

    stop_hook_persona: str = ''
    """``nudger`` (push to deliver code) or ``iterative_user`` (keep giving design
    feedback for genuine multi-turn data). '' → the script's own default."""

    stop_hook_model: str = ''
    stop_hook_max_replies: int = 0
    """Cap on auto-replies. 0 → the script's own default."""

    stop_hook_iterative_turns: int = 0
    """Runaway guard on iterative_user rounds, not a target. 0 → the script's default."""

    stop_hook_show_code: bool = False
    """Let the iterative_user proposer read the code on disk, so feedback can name real
    defects instead of only reacting to the agent's closing text."""

    plugins: tuple[str, ...] = ()
    """Plugin names to offline-install from the local marketplace."""

    skills_dirs: tuple[str, ...] = ()
    """Local skill dirs to sync into the sandbox's ``~/.claude/skills/``."""

    skill_name: str = ''
    """Prefix the prompt with ``/<skill> `` to trigger a skill."""


@dataclass(frozen=True, slots=True)
class HyCodeCfg:
    """Tier 3/4 — HyCode infer agent knobs (read only when runtime.agent_name='hycode')."""

    version: str = '0.1.8'
    """@tencent/hycode npm release. claude_code's version number here makes npm fail with
    ETARGET and the run yields an empty archive. Bump freely."""

    npm_registry: str = 'https://mirrors.tencent.com/npm/'
    """Internal mirror."""

    thinking_preset: str = ''
    """'' inherits the preset derived from the row's model provider."""

    allow_native_web: bool = False
    """Off on purpose: a frontend artifact should come from the model itself, and web search
    makes a rollout depend on live pages."""

    allow_plan_mode: bool = False
    """Off on purpose: in plan mode the agent writes a plan and waits for a human, so the
    working dir stays empty and the archive comes back with 0 project files."""

    extra_cli_args: tuple[str, ...] = ()
    """Appended after the built-in flags, which they never replace."""


DYNAMICAGENT_SUPPORTED_PRESETS = ('code-arena-html', 'code-arena-react', 'code-arena-nextjs-fullstack')


@dataclass(frozen=True, slots=True)
class DynamicAgentCfg:
    """Tier 4 — hycode_dynamicagent (code-arena preset) knobs."""

    version: str = '0.1.20'
    """@tencent/hycode-dynamicagent npm version. Pinned for RL reproducibility."""

    npm_registry: str = 'https://mirrors.tencent.com/npm/'
    thinking_preset: str = ''
    allow_native_web: bool = False
    extra_cli_args: tuple[str, ...] = ()

    default_preset: str = 'code-arena-html'
    """Fallback when the row has no preset field."""

    bootstrap_templates: bool = True
    """Pre-scaffold react/nextjs project before agent starts."""

    tee_log_dir: str = ''
    """Tee CLI stdout/stderr to this dir inside the sandbox. Set to working_dir for archive."""

    image_gen_app_id: str = ''
    image_gen_app_key: str = ''
    serper_base_url: str = 'http://trpc-gpt-eval.production.polaris:8080'
    image_gen_naci_url: str = 'http://trpc-gpt-eval.production.polaris:8080/v1/images/generations'
    image_gen_naci_timeout_seconds: int = 120


@dataclass(frozen=True, slots=True)
class ModelsCfg:
    """Model-role aliases and provider request settings; clients resolves dynamic groups."""

    infer_alias: str = ''
    """Default model alias for infer (main). A row's ``openai_client_name`` wins over it."""

    judge_alias: str = ''
    """Default alias for the ``judge`` role."""

    vlm_alias: str = ''
    """Default alias for the ``vlm`` role (screenshot scoring)."""

    enable_thinking: bool = False
    max_tokens: int | None = None
    """``None`` leaves each provider spec's own default in place."""

    litellm_timeout: float = 1800.0
    litellm_stream: bool = False
    openapi_temp: float = 0.9
    summarized_thinking: bool = True
    vertex_anthropic_version: str = 'vertex-2023-10-16'
    vertex_thinking_type: str = 'adaptive'
    vertex_thinking_budget_tokens: int | None = None
    llm_router_enable: bool = False
    """Route no-thinking auxiliary calls (title / commit message / Task-Explore subagents) to
    a cheaper model so they do not burn the main one."""

    no_thinking_model: str = ''
    """The cheaper model used when ``llm_router_enable`` is on."""

    haiku_model: str = ''
    """Model to serve the CLI's haiku-class requests."""

    table: dict[str, dict[str, str]] = field(default_factory=dict)
    """``{alias: {provider, model, base, key, effort}}`` — the models this run can address.

    Each alias becomes a named client, and a dataset row (or a judge dict) picks one by
    ``openai_client_name``; that is what makes per-row / per-judge model comparison possible.
    The three role fields above name the defaults for rows that omit it.

    Empty falls back to scanning ``M_<alias>_*`` env groups, so a pre-YAML ``.env`` keeps
    working unchanged.
    """


@dataclass(frozen=True, slots=True)
class JudgeCfg:
    """Shared judge execution settings."""

    timeout: float = 7200.0
    """Judge agent wall clock. An exploration planner needs far less than a rubric verifier,
    so it passes a smaller value to ``default_judge_claude_input(timeout=...)``."""

    max_turns: int = 200
    """Tool-call rounds allowed per judge agent."""

    working_dir: str = '/workspace'
    """Judge agent's working directory in the eval container. Read via a ``judge_cfg`` alias in
    ``claude_config.default_judge_claude_input``."""

    conversion_log_level: str = 'DEBUG'

    aesthetic_prompt: str = ''
    """web_judger 美学 prompt 路由。空 = 所有品类共用 ``webjudger_aesthetic.md``（现状）；
    ``by_type`` = 按 ``flow.task_type`` 路由；``web``/``game``/``3d``/``svg`` = 强制某一份。
    认不出的取值回落共用那份 —— 新品类不该让整条样本判不出分。"""

    aesthetic_mode: str = 'absolute'
    """web_judger 美学判分模式。``absolute`` = VLM 逐维 0-5 绝对打分（现状）；
    ``pairwise`` = 与数据行的 ``anchor_screenshot`` 对比出 win/tie/lose（1.0/0.5/0.0）。

    **数据行的 ``aesthetic_mode`` 优先于本字段。** RL 走 parquet，判分时读不到 profile
    （容器侧判分读不到 profile，所以 RL 要改这类开关只能写在数据里，不传
    ``ARTIFACT_*`` 进 rollout），所以 RL 要开 pairwise 必须写在数据行里。"""


@dataclass(frozen=True, slots=True)
class DeployCfg:
    """Tier 2 — web-sandbox deploy budget + project lifetime."""

    expired_time: int = 3600
    """Deployed project lifetime, in seconds. Must exceed ``poll_timeout``, or polling
    outlives the sandbox — validate() checks it."""

    poll_timeout: float = 300.0
    """轮询上限。要盖过服务端自己的阶段上限(build 90s + install)，否则拿不到它算好的失败原因。"""

    poll_interval: float = 5.0
    """Seconds between deploy status polls. Must be > 0."""


@dataclass(frozen=True, slots=True)
class ObsCfg:
    """Tier 5 — credentials + observability. The CLI redacts every secret here."""

    langfuse_public_key: str = ''
    langfuse_secret_key: str = ''
    langfuse_host: str = ''
    langfuse_disabled: bool = False
    """Kill switch. Tracing also needs both keys — see ``langfuse_enabled``."""

    langfuse_experiment: str = 'artifact'
    run_tag: str = ''
    """Free-text tag attached to the run, for grouping traces."""

    save_screenshots_dir: str = ''
    """Root for screenshots + judge evidence. '' = do not persist (debug aid)."""

    @property
    def langfuse_enabled(self) -> bool:
        return not self.langfuse_disabled and bool(self.langfuse_public_key and self.langfuse_secret_key)


@dataclass(frozen=True, slots=True)
class CosCfg:
    """Tier 4 — the archive-backup switch.

    Only the switch is a config field. Bucket / region / prefix / endpoint / credentials stay
    literals in ``agent_setup/cos_io.py``: they are set once for the cluster and never tuned
    per run, so surfacing them here would add knobs nobody turns.
    """

    enabled: bool = True
    """Upload each rollout's archive to COS. Backup only — eval reads ``archive_b64`` from
    memory in the same call, so a failed upload never fails the rollout. Turn it off for
    debug runs that do not need the deliverable kept."""


@dataclass(frozen=True, slots=True)
class BudgetCfg:
    """Tier 1 — mirror of ``nexus run-batch --timeout``.

    A Python module cannot read a sibling CLI flag, so the run script must be the single
    source: ``--timeout "$ARTIFACT_BATCH_TIMEOUT"``. 0 = not wired (validate warns).
    """

    batch_timeout: float = 0.0


@dataclass(frozen=True, slots=True)
class BatchCfg:
    """Tier 1 — ``nexus run-batch`` CLI arguments.

    Declared here so a profile can carry the whole run, but consumed by ``run.sh`` rather than
    by the generator: ``cli/commands/run.py`` builds ``BatchConfig`` from argv *before* it loads
    the generator, so the generator cannot feed these back. The run script reads them out of the
    profile with ``--emit-batch-args`` and passes them as flags.
    """

    worker_processes: int = 8
    """Subprocess count. Each one owns a GIL + event loop; keep per-process concurrency at
    32–64 and set this to target_total / that."""

    max_concurrent: int = 512
    """Total concurrency, split evenly across the workers by the framework."""

    db: str = './outputs/artifact_eval_batch.db'
    """Run database. Sharded mode writes ``<db>.shards/000N.db`` alongside it."""

    export_path: str = './outputs/artifact_eval/batch_results.jsonl'
    """Results JSONL. Overwritten and re-exported in full from the db each time."""

    retry_failed: int = 1
    """Retry attempts for failed rows."""


@dataclass(frozen=True, slots=True)
class ArtifactCfg:
    data: DataCfg = field(default_factory=DataCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    claude_code: ClaudeCodeCfg = field(default_factory=ClaudeCodeCfg)
    hycode: HyCodeCfg = field(default_factory=HyCodeCfg)
    dynamicagent: DynamicAgentCfg = field(default_factory=DynamicAgentCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    judge: JudgeCfg = field(default_factory=JudgeCfg)
    deploy: DeployCfg = field(default_factory=DeployCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    batch: BatchCfg = field(default_factory=BatchCfg)
    cos: CosCfg = field(default_factory=CosCfg)
    provenance: dict[str, str] = field(default_factory=dict)
    parse_errors: tuple[str, ...] = ()
    shadowed_env: dict[str, str] = field(default_factory=dict)
    """``field path -> env name`` for env vars a profile overrode. Surfaced by the dump so a
    leftover ``export`` cannot be mistaken for the value in effect."""


# Legacy env names still honoured, mapped to their new primary name. Extensions add
# their own via register_extension(legacy_aliases=...).
LEGACY_ALIASES: dict[str, str] = {
    'MAX_TOKENS': 'ARTIFACT_MAX_TOKENS',
    'LITELLM_TIMEOUT': 'ARTIFACT_LITELLM_TIMEOUT',
    'OPENAPI_TEMP': 'ARTIFACT_OPENAPI_TEMP',
    'ENABLE_THINKING': 'ARTIFACT_ENABLE_THINKING',
    'IS_SUMMARIZED_THINKING': 'ARTIFACT_SUMMARIZED_THINKING',
    'VERTEX_ANTHROPIC_VERSION': 'ARTIFACT_VERTEX_ANTHROPIC_VERSION',
    'VERTEX_THINKING_TYPE': 'ARTIFACT_VERTEX_THINKING_TYPE',
    'VERTEX_THINKING_BUDGET_TOKENS': 'ARTIFACT_VERTEX_THINKING_BUDGET_TOKENS',
    'CLAUDE_CODE_VERSION': 'ARTIFACT_CLAUDE_CODE_VERSION',
    'HYCODE_PACKAGE_VERSION': 'ARTIFACT_HYCODE_VERSION',
    'HYCODE_NPM_REGISTRY': 'ARTIFACT_HYCODE_NPM_REGISTRY',
    'HYCODE_THINKING_PRESET': 'ARTIFACT_HYCODE_THINKING_PRESET',
    'HYCODE_ALLOW_NATIVE_WEB': 'ARTIFACT_HYCODE_ALLOW_NATIVE_WEB',
    'CLAUDE_CODE_BINARY_URL': 'ARTIFACT_CLAUDE_CODE_BINARY_URL',
    'CLAUDE_CODE_CONV_LOG_LEVEL': 'ARTIFACT_INFER_CONVERSION_LOG_LEVEL',
    'ARTIFACT_CONVERSION_LOG_LEVEL': 'ARTIFACT_JUDGE_CONVERSION_LOG_LEVEL',
    'GONGFENG_TOKEN': 'ARTIFACT_GONGFENG_TOKEN',
    'FIGMA_TOKEN': 'ARTIFACT_FIGMA_TOKEN',
    'NEXUS_ASKUSER_HOOK_VERSION': 'ARTIFACT_ASKUSER_HOOK_VERSION',
    'ATOMIC_RUN_TIMEOUT': 'ARTIFACT_BATCH_TIMEOUT',
}


# ── env NOT owned here (indexed so this file still answers "what knobs exist") ──


@dataclass(frozen=True, slots=True)
class EnvDoc:
    name: str
    default: str
    owner: str
    effect: str


# Injected into the sandbox container as base64 pre-start scripts; they re-read
# os.environ inside the container with stdlib only, so they cannot read this module.
# ``test_hook_env_table_is_complete`` greps the owning files and fails if this drifts.
HOOK_ENV: tuple[EnvDoc, ...] = (
    EnvDoc('NEXUS_ASKUSER_HOOK_VERSION', 'v2', 'agent_setup/claude_hooks.py', 'v1|v2 hook script choice (host-side)'),
    EnvDoc('NEXUS_ASKUSER_HOOK_TIMEOUT', '30', 'agent_setup/askuser_hook_v2.py', 'LLM answer timeout'),
    EnvDoc(
        'NEXUS_ASKUSER_HOOK_MODEL',
        'claude-opus-4-7',
        'agent_setup/askuser_hook_v2.py',
        'model answering AskUserQuestion',
    ),
    EnvDoc('NEXUS_ASKUSER_HOOK_MAX_TOKENS', '64000', 'agent_setup/askuser_hook_v2.py', 'max tokens'),
    EnvDoc('NEXUS_ASKUSER_HOOK_CONTEXT_CHARS', '3000', 'agent_setup/askuser_hook_v2.py', 'context budget'),
    EnvDoc('NEXUS_ASKUSER_HOOK_RECENT_TURNS', '2', 'agent_setup/askuser_hook_v2.py', 'assistant turns kept'),
    EnvDoc('NEXUS_ASKUSER_HOOK_DISABLE_LLM', '0', 'agent_setup/askuser_hook_v2.py', 'fall back to first option'),
    EnvDoc('NEXUS_STOP_HOOK_ENABLE', '0', 'agent_setup/stop_hook.py', 'opt-in gate for the Stop hook'),
    EnvDoc(
        'NEXUS_STOP_HOOK_*', 'see file', 'agent_setup/stop_hook.py', '~20 more knobs (model/replies/persona/budgets)'
    ),
)

# Read by other layers or by a dynamic scan; documented here, owned elsewhere.
EXTERNAL_ENV: tuple[EnvDoc, ...] = (
    EnvDoc(
        'M_<alias>_PROVIDER/_MODEL/_BASE/_KEY/_EFFORT',
        '—',
        'agent_setup/clients.py',
        'dynamic scan: every group registers a named client',
    ),
    EnvDoc(
        'NEXUS_CONVERSION_LOG_LEVEL',
        'DEBUG',
        'runtime/server/services/anthropic_proxy',
        'read INSIDE the container; setting it on the host does nothing',
    ),
    EnvDoc(
        'ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY',
        '—',
        'claude_code/runtime_setup.py',
        'injected to point at the in-container proxy',
    ),
    EnvDoc(
        'LANGFUSE_PUBLIC_KEY / _SECRET_KEY / _BASE_URL',
        '—',
        'this module (legacy primary name)',
        'ecosystem-standard names, shared with hycode in one .env',
    ),
)


# ── load / cache ──────────────────────────────────────────────────────────────


# Fields whose value is itself a mapping, so flattening must stop there and hand the whole
# dict to the field instead of splintering it into one key per leaf.
_OPAQUE_MAPPING_FIELDS = frozenset({'models.table'})


def _flatten(node: Mapping[Any, Any], prefix: str = '') -> dict[str, Any]:
    """Flatten nested profile keys to dotted paths; empty mappings contribute no values."""
    out: dict[str, Any] = {}
    for key, value in node.items():
        path = f'{prefix}{key}'
        if path in _OPAQUE_MAPPING_FIELDS:
            out[path] = value
        elif isinstance(value, dict) and value:
            out.update(_flatten(value, f'{path}.'))
        elif not isinstance(value, dict):
            out[path] = value
    return out


def _known_field_paths() -> set[str]:
    """Every ``group.field`` path an :class:`ArtifactCfg` accepts."""
    paths: set[str] = set()
    for group in fields(ArtifactCfg):
        factory = getattr(group, 'default_factory', None)
        default = factory() if callable(factory) else None
        if not is_dataclass(default):
            continue
        paths.update(f'{group.name}.{f.name}' for f in fields(default))
    return paths


def load_profile(path: str | Path, overrides: Sequence[str] = ()) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Read profile values, provenance, and validation issues; report unknown keys."""
    from omegaconf import OmegaConf

    problems: list[str] = []
    path = Path(path)
    if not path.exists():
        return {}, {}, [f'profile not found: {path}']

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        return {}, {}, [f'profile {path} must be a mapping at the top level']
    values = _flatten(raw)
    sources = dict.fromkeys(values, f'yaml {path.name}')

    if overrides:
        raw_over = OmegaConf.to_container(OmegaConf.from_dotlist(list(overrides)))
        over = _flatten(raw_over) if isinstance(raw_over, dict) else {}
        values.update(over)
        sources.update(dict.fromkeys(over, '--set'))

    known = _known_field_paths()
    for key in sorted(set(values) - known):
        problems.append(f'unknown config key {key!r} — not a field of ArtifactCfg')
        values.pop(key)
        sources.pop(key, None)
    return values, sources, problems


def load(profile: str | Path | None = None, overrides: Sequence[str] = ()) -> ArtifactCfg:
    """Build a fresh config from (optionally) a YAML profile, then the environment.

    Precedence per field: ``overrides`` > *profile* > env > dataclass default. Never raises on
    bad values — they land in ``parse_errors`` and are surfaced by :func:`validate`.
    """
    overlay: dict[str, Any] = {}
    overlay_sources: dict[str, str] = {}
    profile_problems: list[str] = []
    if profile:
        overlay, overlay_sources, profile_problems = load_profile(profile, overrides)
    elif overrides:
        from omegaconf import OmegaConf

        raw_over = OmegaConf.to_container(OmegaConf.from_dotlist(list(overrides)))
        overlay = _flatten(raw_over) if isinstance(raw_over, dict) else {}
        overlay_sources = dict.fromkeys(overlay, '--set')

    r = _Reader(overlay, overlay_sources)
    r.parse_errors.extend(profile_problems)
    data = _build(
        DataCfg,
        dataset_path=r.s('data.dataset_path', 'ARTIFACT_DATASET_PATH'),
        instance_ids=r.csv('data.instance_ids', 'ARTIFACT_INSTANCE_IDS'),
        skip=r.i('data.skip', 'ARTIFACT_SKIP'),
        limit=r.i_opt('data.limit', 'ARTIFACT_LIMIT'),
        image=r.s('data.image', 'ARTIFACT_IMAGE'),
        infer_db=r.s('data.infer_db', 'ARTIFACT_INFER_DB'),
        skip_infer=r.b('data.skip_infer', 'ARTIFACT_SKIP_INFER'),
        skip_eval=r.b('data.skip_eval', 'ARTIFACT_SKIP_EVAL'),
    )
    provider_name = r.s('runtime.provider_name', 'ARTIFACT_RUNTIME_PROVIDER_NAME')
    runtime = _build(
        RuntimeCfg,
        provider_name=provider_name,
        provider=r.s('runtime.provider', 'ARTIFACT_RUNTIME_PROVIDER'),
        agent_name=r.s('runtime.agent_name', 'ARTIFACT_AGENT_NAME'),
        runtime_image=r.s(
            'runtime.runtime_image',
            'ARTIFACT_RUNTIME_IMAGE',
            'GONGFENG_RUNTIME_IMAGE',
        ),
        gongfeng_token=r.s('runtime.gongfeng_token', 'ARTIFACT_GONGFENG_TOKEN', 'GONGFENG_TOKEN'),
        runtime_timeout=r.i('runtime.runtime_timeout', 'ARTIFACT_RUNTIME_TIMEOUT'),
        agent_timeout=r.i('runtime.agent_timeout', 'ARTIFACT_AGENT_TIMEOUT'),
        infer_conversion_log_level=r.s(
            'runtime.infer_conversion_log_level',
            'ARTIFACT_INFER_CONVERSION_LOG_LEVEL',
            'CLAUDE_CODE_CONV_LOG_LEVEL',
        ),
    )
    claude_code = _build(
        ClaudeCodeCfg,
        version=r.s('claude_code.version', 'ARTIFACT_CLAUDE_CODE_VERSION', 'CLAUDE_CODE_VERSION'),
        binary_url=r.s('claude_code.binary_url', 'ARTIFACT_CLAUDE_CODE_BINARY_URL', 'CLAUDE_CODE_BINARY_URL'),
        mcp_servers=r.csv('claude_code.mcp_servers', 'ARTIFACT_MCP_SERVERS'),
        figma_token=r.s('claude_code.figma_token', 'ARTIFACT_FIGMA_TOKEN', 'FIGMA_TOKEN'),
        hooks_enabled=r.b('claude_code.hooks_enabled', 'ARTIFACT_CC_HOOKS'),
        askuser_hook_version=r.s(
            'claude_code.askuser_hook_version',
            'ARTIFACT_ASKUSER_HOOK_VERSION',
            'NEXUS_ASKUSER_HOOK_VERSION',
        ),
        askuser_hook_disable_llm=r.b(
            'claude_code.askuser_hook_disable_llm',
            'ARTIFACT_ASKUSER_HOOK_DISABLE_LLM',
            'NEXUS_ASKUSER_HOOK_DISABLE_LLM',
        ),
        askuser_hook_model=r.s(
            'claude_code.askuser_hook_model', 'ARTIFACT_ASKUSER_HOOK_MODEL', 'NEXUS_ASKUSER_HOOK_MODEL'
        ),
        stop_hook=r.b('claude_code.stop_hook', 'ARTIFACT_STOP_HOOK', 'NEXUS_STOP_HOOK_ENABLE'),
        stop_hook_persona=r.s('claude_code.stop_hook_persona', 'ARTIFACT_STOP_HOOK_PERSONA', 'NEXUS_STOP_HOOK_PERSONA'),
        stop_hook_model=r.s('claude_code.stop_hook_model', 'ARTIFACT_STOP_HOOK_MODEL', 'NEXUS_STOP_HOOK_MODEL'),
        stop_hook_max_replies=r.i(
            'claude_code.stop_hook_max_replies',
            'ARTIFACT_STOP_HOOK_MAX_REPLIES',
            'NEXUS_STOP_HOOK_MAX_REPLIES',
        ),
        stop_hook_iterative_turns=r.i(
            'claude_code.stop_hook_iterative_turns',
            'ARTIFACT_STOP_HOOK_ITERATIVE_TURNS',
            'NEXUS_STOP_HOOK_ITERATIVE_TURNS',
        ),
        stop_hook_show_code=r.b(
            'claude_code.stop_hook_show_code', 'ARTIFACT_STOP_HOOK_SHOW_CODE', 'NEXUS_STOP_HOOK_SHOW_CODE'
        ),
        plugins=r.csv('claude_code.plugins', 'ARTIFACT_CLAUDE_PLUGINS'),
        skills_dirs=r.csv('claude_code.skills_dirs', 'ARTIFACT_SKILLS_DIR'),
        skill_name=r.s('claude_code.skill_name', 'ARTIFACT_SKILL_NAME'),
    )
    hycode = _build(
        HyCodeCfg,
        version=r.s('hycode.version', 'ARTIFACT_HYCODE_VERSION', 'HYCODE_PACKAGE_VERSION'),
        npm_registry=r.s(
            'hycode.npm_registry',
            'ARTIFACT_HYCODE_NPM_REGISTRY',
            'HYCODE_NPM_REGISTRY',
        ),
        thinking_preset=r.s('hycode.thinking_preset', 'ARTIFACT_HYCODE_THINKING_PRESET', 'HYCODE_THINKING_PRESET'),
        allow_native_web=r.b('hycode.allow_native_web', 'ARTIFACT_HYCODE_ALLOW_NATIVE_WEB', 'HYCODE_ALLOW_NATIVE_WEB'),
        allow_plan_mode=r.b('hycode.allow_plan_mode', 'ARTIFACT_HYCODE_ALLOW_PLAN_MODE'),
        extra_cli_args=r.csv('hycode.extra_cli_args', 'ARTIFACT_HYCODE_EXTRA_CLI_ARGS'),
    )
    dynamicagent = _build(
        DynamicAgentCfg,
        version=r.s('dynamicagent.version', 'ARTIFACT_DYNAMICAGENT_VERSION'),
        npm_registry=r.s('dynamicagent.npm_registry', 'ARTIFACT_DYNAMICAGENT_NPM_REGISTRY'),
        thinking_preset=r.s('dynamicagent.thinking_preset', 'ARTIFACT_DYNAMICAGENT_THINKING_PRESET'),
        allow_native_web=r.b('dynamicagent.allow_native_web', 'ARTIFACT_DYNAMICAGENT_ALLOW_NATIVE_WEB'),
        extra_cli_args=r.csv('dynamicagent.extra_cli_args', 'ARTIFACT_DYNAMICAGENT_EXTRA_CLI_ARGS'),
        default_preset=r.s('dynamicagent.default_preset', 'ARTIFACT_DYNAMICAGENT_DEFAULT_PRESET'),
        bootstrap_templates=r.b('dynamicagent.bootstrap_templates', 'ARTIFACT_DYNAMICAGENT_BOOTSTRAP_TEMPLATES'),
        tee_log_dir=r.s('dynamicagent.tee_log_dir', 'ARTIFACT_DYNAMICAGENT_TEE_LOG_DIR'),
        image_gen_app_id=r.s('dynamicagent.image_gen_app_id', 'IMAGE_GEN_APP_ID'),
        image_gen_app_key=r.s('dynamicagent.image_gen_app_key', 'IMAGE_GEN_APP_KEY'),
        serper_base_url=r.s('dynamicagent.serper_base_url', 'SERPER_BASE_URL'),
        image_gen_naci_url=r.s('dynamicagent.image_gen_naci_url', 'IMAGE_GEN_NACI_URL'),
        image_gen_naci_timeout_seconds=r.i(
            'dynamicagent.image_gen_naci_timeout_seconds', 'IMAGE_GEN_NACI_TIMEOUT_SECONDS'
        ),
    )
    models = _build(
        ModelsCfg,
        infer_alias=r.s('models.infer_alias', 'ARTIFACT_INFER_MODEL'),
        judge_alias=r.s('models.judge_alias', 'ARTIFACT_JUDGE_MODEL'),
        vlm_alias=r.s('models.vlm_alias', 'ARTIFACT_VLM_MODEL'),
        enable_thinking=r.b('models.enable_thinking', 'ARTIFACT_ENABLE_THINKING', 'ENABLE_THINKING'),
        max_tokens=r.i_opt('models.max_tokens', 'ARTIFACT_MAX_TOKENS', 'MAX_TOKENS'),
        litellm_timeout=r.f('models.litellm_timeout', 'ARTIFACT_LITELLM_TIMEOUT', 'LITELLM_TIMEOUT'),
        litellm_stream=r.b('models.litellm_stream', 'ARTIFACT_LITELLM_STREAM', 'LITELLM_STREAM'),
        openapi_temp=r.f('models.openapi_temp', 'ARTIFACT_OPENAPI_TEMP', 'OPENAPI_TEMP'),
        summarized_thinking=r.b('models.summarized_thinking', 'ARTIFACT_SUMMARIZED_THINKING', 'IS_SUMMARIZED_THINKING'),
        vertex_anthropic_version=r.s(
            'models.vertex_anthropic_version',
            'ARTIFACT_VERTEX_ANTHROPIC_VERSION',
            'VERTEX_ANTHROPIC_VERSION',
        ),
        vertex_thinking_type=r.s(
            'models.vertex_thinking_type', 'ARTIFACT_VERTEX_THINKING_TYPE', 'VERTEX_THINKING_TYPE'
        ),
        vertex_thinking_budget_tokens=r.i_opt(
            'models.vertex_thinking_budget_tokens',
            'ARTIFACT_VERTEX_THINKING_BUDGET_TOKENS',
            'VERTEX_THINKING_BUDGET_TOKENS',
        ),
        llm_router_enable=r.b('models.llm_router_enable', 'ARTIFACT_LLM_ROUTER_ENABLE'),
        no_thinking_model=r.s('models.no_thinking_model', 'ARTIFACT_NO_THINKING_MODEL'),
        haiku_model=r.s('models.haiku_model', 'ARTIFACT_HAIKU_MODEL'),
        table=r.mapping('models.table'),
    )
    judge = _build(
        JudgeCfg,
        timeout=r.f('judge.timeout', 'ARTIFACT_JUDGE_TIMEOUT'),
        max_turns=r.i('judge.max_turns', 'ARTIFACT_JUDGE_MAX_TURNS'),
        working_dir=r.s('judge.working_dir', 'ARTIFACT_JUDGE_WORKING_DIR'),
        conversion_log_level=r.s(
            'judge.conversion_log_level',
            'ARTIFACT_JUDGE_CONVERSION_LOG_LEVEL',
            'ARTIFACT_CONVERSION_LOG_LEVEL',
        ),
        aesthetic_prompt=r.s('judge.aesthetic_prompt', 'ARTIFACT_JUDGE_AESTHETIC_PROMPT'),
        aesthetic_mode=r.s('judge.aesthetic_mode', 'ARTIFACT_JUDGE_AESTHETIC_MODE'),
    )
    deploy = _build(
        DeployCfg,
        expired_time=r.i('deploy.expired_time', 'ARTIFACT_DEPLOY_EXPIRED_TIME'),
        poll_timeout=r.f('deploy.poll_timeout', 'ARTIFACT_DEPLOY_POLL_TIMEOUT'),
        poll_interval=r.f('deploy.poll_interval', 'ARTIFACT_DEPLOY_POLL_INTERVAL'),
    )
    obs = _build(
        ObsCfg,
        langfuse_public_key=r.s('obs.langfuse_public_key', 'LANGFUSE_PUBLIC_KEY', 'ARTIFACT_LANGFUSE_PUBLIC_KEY'),
        langfuse_secret_key=r.s('obs.langfuse_secret_key', 'LANGFUSE_SECRET_KEY', 'ARTIFACT_LANGFUSE_SECRET_KEY'),
        langfuse_host=r.s('obs.langfuse_host', 'LANGFUSE_BASE_URL', 'LANGFUSE_HOST', 'ARTIFACT_LANGFUSE_BASE_URL'),
        langfuse_disabled=r.b('obs.langfuse_disabled', 'ARTIFACT_DISABLE_LANGFUSE'),
        langfuse_experiment=r.s('obs.langfuse_experiment', 'ARTIFACT_LANGFUSE_EXPERIMENT'),
        run_tag=r.s('obs.run_tag', 'ARTIFACT_RUN_TAG'),
        save_screenshots_dir=r.s('obs.save_screenshots_dir', 'ARTIFACT_SAVE_SCREENSHOTS_DIR'),
    )
    budget = _build(
        BudgetCfg,
        batch_timeout=r.f('budget.batch_timeout', 'ARTIFACT_BATCH_TIMEOUT', 'ATOMIC_RUN_TIMEOUT'),
    )
    cos = _build(
        CosCfg,
        enabled=r.b('cos.enabled', 'ARTIFACT_COS_UPLOAD', 'FRONTEND_UPLOAD_TO_COS'),
    )
    batch = _build(
        BatchCfg,
        worker_processes=r.i('batch.worker_processes', 'ARTIFACT_WORKER_PROCESSES'),
        max_concurrent=r.i('batch.max_concurrent', 'ARTIFACT_MAX_CONCURRENT', 'MAX_CONCURRENT'),
        db=r.s('batch.db', 'ARTIFACT_BATCH_DB', 'DB'),
        export_path=r.s('batch.export_path', 'ARTIFACT_EXPORT_PATH', 'EXPORT_PATH'),
        retry_failed=r.i('batch.retry_failed', 'ARTIFACT_RETRY_FAILED'),
    )
    return ArtifactCfg(
        data=data,
        runtime=runtime,
        claude_code=claude_code,
        hycode=hycode,
        dynamicagent=dynamicagent,
        models=models,
        judge=judge,
        deploy=deploy,
        obs=obs,
        budget=budget,
        batch=batch,
        cos=cos,
        provenance=dict(r.provenance),
        parse_errors=tuple(r.parse_errors),
        shadowed_env=dict(r.shadowed_env),
    )


_CACHE: ArtifactCfg | None = None

PROFILE_ENV = 'ARTIFACT_PROFILE'
"""Path to the active YAML profile. The one env var that survives the YAML migration: the
run script has to tell the generator subprocess which profile to load, and that cannot itself
live in the profile."""

OVERRIDES_ENV = 'ARTIFACT_SET'
"""``--set`` overrides for the generator subprocess, newline-separated ``a.b=c`` entries.

Same reason as :data:`PROFILE_ENV`: ``run.sh`` resolves the overrides but ``nexus run-batch``
spawns the generator, so they have to cross a process boundary. Newline-separated because a
value may legitimately contain commas (e.g. a comma-joined list)."""


def get() -> ArtifactCfg:
    """Cached config. Call :func:`reset` after mutating the environment (tests/CLI).

    Honours ``$ARTIFACT_PROFILE`` / ``$ARTIFACT_SET`` so a run script can select a profile and
    its overrides for the whole process tree without every call site threading them through.
    """
    global _CACHE
    if _CACHE is None:
        raw = os.getenv(OVERRIDES_ENV) or ''
        overrides = [line.strip() for line in raw.splitlines() if line.strip()]
        _CACHE = load(os.getenv(PROFILE_ENV) or None, overrides)
    return _CACHE


def reset() -> None:
    """Drop the cache so the next :func:`get` re-reads the environment."""
    global _CACHE
    _CACHE = None


def effective_limit(cfg: ArtifactCfg | None = None) -> int | None:
    """Return the row limit, treating None and non-positive values as unlimited."""
    cfg = get() if cfg is None else cfg
    limit = cfg.data.limit
    if limit is None or limit <= 0:
        return None
    return limit


# ── validation ────────────────────────────────────────────────────────────────

# Legacy env names still honoured, mapped to their new primary name (for the "you are
# using an old name" notice). Keys are what a user may have set.

_SECRET_FIELDS = {
    'runtime.gongfeng_token',
    'claude_code.figma_token',
    'obs.langfuse_public_key',
    'obs.langfuse_secret_key',
}


@dataclass(frozen=True, slots=True)
class Problem:
    level: Literal['error', 'warning']
    field: str
    message: str


def model_aliases(cfg: ArtifactCfg | None = None) -> set[str] | None:
    """Resolve aliases from the configured table or environment; return None on failure."""
    cfg = get() if cfg is None else cfg
    if cfg.models.table:
        return {alias for alias, entry in cfg.models.table.items() if (entry.get('provider') or '').strip()}
    try:  # the env scan lives in clients.py; importing it here keeps config stdlib-only at import
        from nexus.extensions.tasks.artifact.agent_setup import clients

        return set(clients.parse_model_table_from_env())
    except Exception:  # pragma: no cover - diagnostics only
        return None


def validate(cfg: ArtifactCfg | None = None, *, model_aliases: set[str] | None = None) -> list[Problem]:
    """Check the config for contradictions. ``model_aliases`` is the set discovered by
    ``clients.parse_model_table_from_env()``; pass it to also check role aliases.
    """
    cfg = get() if cfg is None else cfg
    out: list[Problem] = []

    for message in cfg.parse_errors:
        out.append(Problem('error', 'parse', message))

    batch = cfg.budget.batch_timeout
    if batch > 0:
        if cfg.judge.timeout > batch:
            out.append(
                Problem(
                    'warning',
                    'judge.timeout',
                    f'judge.timeout {cfg.judge.timeout:.0f}s > batch timeout {batch:.0f}s — the batch kills '
                    f'the run first, so raising ARTIFACT_JUDGE_TIMEOUT has no effect.',
                )
            )
    else:
        out.append(
            Problem(
                'warning',
                'budget.batch_timeout',
                'no batch wall clock: set budget.batch_timeout for legacy profile validation; '
                'the caller must also pass the corresponding --timeout to the batch runner. '
                'The legacy JSONL launcher has been retired.',
            )
        )

    if cfg.runtime.agent_timeout > cfg.runtime.runtime_timeout:
        out.append(
            Problem(
                'error',
                'runtime.agent_timeout',
                f'agent_timeout {cfg.runtime.agent_timeout}s > runtime_timeout {cfg.runtime.runtime_timeout}s — '
                f'the sandbox dies under the agent.',
            )
        )
    if cfg.deploy.poll_timeout >= cfg.deploy.expired_time:
        out.append(
            Problem(
                'error',
                'deploy.poll_timeout',
                f'poll_timeout {cfg.deploy.poll_timeout:.0f}s >= expired_time {cfg.deploy.expired_time}s — '
                f'polling outlives the sandbox.',
            )
        )
    if cfg.deploy.poll_interval <= 0:
        out.append(Problem('error', 'deploy.poll_interval', 'must be > 0 (would spin forever).'))

    # pairwise 需要数据行提供 anchor_screenshot，认错的取值会让整批判不出美学分。
    if cfg.judge.aesthetic_mode not in AESTHETIC_MODES:
        out.append(
            Problem(
                'error',
                'judge.aesthetic_mode',
                f'{cfg.judge.aesthetic_mode!r} 不是有效模式（{" / ".join(sorted(AESTHETIC_MODES))}）。'
                f'pairwise 还要求数据行带 anchor_screenshot。',
            )
        )

    # Require a fully qualified runtime image unless using the current package version.
    image_ref = cfg.runtime.runtime_image
    if image_ref and image_ref != 'current' and '/' not in image_ref:
        out.append(
            Problem(
                'error',
                'runtime.runtime_image',
                f'{image_ref!r} has no registry prefix, so it resolves against docker.io and the '
                f'sandbox fails with ErrImagePull. Use the full '
                f'mirrors.tencent.com/hunyuan_yanguan/... path, or the literal "current".',
            )
        )

    # runtime.provider selects a gongfeng scheduling strategy, not a provider class.
    if cfg.runtime.provider and cfg.runtime.provider not in _GONGFENG_STRATEGIES:
        out.append(
            Problem(
                'error',
                'runtime.provider',
                f'{cfg.runtime.provider!r} is not a gongfeng scheduling strategy '
                f'({" / ".join(sorted(_GONGFENG_STRATEGIES))} / empty for auto). To change which '
                f'provider starts the sandbox, set runtime.provider_name instead.',
            )
        )
    if cfg.runtime.provider and cfg.runtime.provider_name != 'gongfeng':
        out.append(
            Problem(
                'warning',
                'runtime.provider',
                f'ignored: only read when provider_name is gongfeng (it is {cfg.runtime.provider_name!r}).',
            )
        )

    # Claude Code scaffolding applies only to its matching harness.
    if cfg.runtime.agent_name != 'claude_code':
        inert = [
            name
            for name, is_set in (
                ('hooks_enabled', cfg.claude_code.hooks_enabled),
                ('stop_hook', cfg.claude_code.stop_hook),
                ('plugins', bool(cfg.claude_code.plugins)),
                ('skills_dirs', bool(cfg.claude_code.skills_dirs)),
                ('mcp_servers', bool(cfg.claude_code.mcp_servers)),
                ('version', bool(cfg.claude_code.version)),
            )
            if is_set
        ]
        if inert:
            out.append(
                Problem(
                    'warning',
                    'claude_code',
                    f'{", ".join(inert)} set but runtime.agent_name={cfg.runtime.agent_name!r} — only the '
                    f'claude_code harness reads these, so they have no effect.',
                )
            )

    # The hook scripts are wired in through hooks_enabled's settings.json patch, so any hook
    # knob set without it is silently inert — which is exactly how the stop hook sat unusable.
    if not cfg.claude_code.hooks_enabled:
        for name, is_set in (
            ('claude_code.stop_hook', cfg.claude_code.stop_hook),
            ('claude_code.askuser_hook_disable_llm', cfg.claude_code.askuser_hook_disable_llm),
            ('claude_code.askuser_hook_model', bool(cfg.claude_code.askuser_hook_model)),
        ):
            if is_set:
                out.append(
                    Problem(
                        'error',
                        name,
                        'set but claude_code.hooks_enabled is off, so no hook is installed and this has no '
                        'effect. Enable hooks (ARTIFACT_CC_HOOKS=1) or unset this.',
                    )
                )
    if cfg.claude_code.stop_hook_persona and cfg.claude_code.stop_hook_persona not in (
        'nudger',
        'iterative_user',
    ):
        out.append(
            Problem(
                'error',
                'claude_code.stop_hook_persona',
                f'{cfg.claude_code.stop_hook_persona!r} is not a known persona '
                f'(nudger / iterative_user); stop_hook.py would fall back silently.',
            )
        )
    if cfg.claude_code.askuser_hook_version not in ('v1', 'v2'):
        out.append(
            Problem(
                'error',
                'claude_code.askuser_hook_version',
                f'{cfg.claude_code.askuser_hook_version!r} is neither v1 nor v2 — the builder would '
                f'silently inject v2.',
            )
        )

    # dynamicagent preset validation
    if cfg.runtime.agent_name == 'hycode_dynamicagent':
        da = cfg.dynamicagent
        if da.default_preset not in DYNAMICAGENT_SUPPORTED_PRESETS:
            out.append(
                Problem(
                    'error',
                    'dynamicagent.default_preset',
                    f'{da.default_preset!r} not in {DYNAMICAGENT_SUPPORTED_PRESETS}',
                )
            )
        if not da.image_gen_app_key:
            out.append(
                Problem(
                    'warning',
                    'dynamicagent.image_gen_app_key',
                    'empty — the bundled image generation tool will fail at runtime.',
                )
            )

    if model_aliases is not None:
        for name, alias in (
            ('models.infer_alias', cfg.models.infer_alias),
            ('models.judge_alias', cfg.models.judge_alias),
            ('models.vlm_alias', cfg.models.vlm_alias),
        ):
            if alias and alias not in model_aliases:
                known = ', '.join(sorted(model_aliases)) or '(none)'
                out.append(
                    Problem('error', name, f'{alias!r} has no M_{alias}_PROVIDER env group. Known aliases: {known}')
                )

    if os.getenv('NEXUS_CONVERSION_LOG_LEVEL', '').strip():
        out.append(
            Problem(
                'warning',
                'judge.conversion_log_level',
                'NEXUS_CONVERSION_LOG_LEVEL is set on the host but the anthropic proxy runs INSIDE the '
                'container — it has no effect. Use ARTIFACT_JUDGE_CONVERSION_LOG_LEVEL instead.',
            )
        )

    legacy_in_use = sorted(
        {name: new for name, new in LEGACY_ALIASES.items() if name in cfg.provenance.values()}.items()
    )
    if legacy_in_use:
        pairs = ', '.join(f'{old}→{new}' for old, new in legacy_in_use)
        out.append(Problem('warning', 'aliases', f'legacy env names in use (still honoured): {pairs}'))
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

_TIERS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ('Tier 1', 'set by every run script', ('data', 'batch', 'budget')),
    ('Tier 2', 'timeout budget — tune together, validate() guards you', ('runtime', 'deploy', 'judge')),
    ('Tier 3', 'rarely touched', ('models',)),
    ('Tier 4', 'opt-in scaffolding (default off)', ('claude_code', 'hycode', 'dynamicagent', 'cos')),
    ('Tier 5', 'credentials / observability (redacted)', ('obs',)),
)


def _redact(field_path: str, value: Any) -> str:
    if field_path in _SECRET_FIELDS and value:
        return f'set (len={len(str(value))})'
    return repr(value)


def format_report(cfg: ArtifactCfg, problems: list[Problem], model_aliases: set[str] | None = None) -> str:
    lines: list[str] = ['=== artifact config (effective) ===', '']
    for tier, note, groups in _TIERS:
        lines.append(f'── {tier}: {note} ──')
        for group_name in groups:
            group = getattr(cfg, group_name)
            if not is_dataclass(group):
                continue
            for f in fields(group):
                path = f'{group_name}.{f.name}'
                value = getattr(group, f.name)
                via = cfg.provenance.get(path)
                mark = '*' if via else ' '
                suffix = ''
                if via and via in LEGACY_ALIASES:
                    suffix = f'  [via legacy {via}]'
                elif via:
                    suffix = f'  [via {via}]'
                lines.append(f'  {mark} {path:42} = {_redact(path, value)}{suffix}')
        lines.append('')
    lines.append('  (* = set from the environment; everything else is the built-in default)')
    lines.append('')
    if cfg.shadowed_env:
        lines.append('── env overridden by the profile (these `export`s have NO effect) ──')
        for path, env_name in sorted(cfg.shadowed_env.items()):
            lines.append(f'  {env_name:38} shadowed by {path}')
        lines.append('')
    if model_aliases is not None:
        source = 'models.table' if cfg.models.table else 'M_*_PROVIDER env scan'
        lines.append(f'── model aliases ({source}) ──\n  {", ".join(sorted(model_aliases)) or "(none)"}\n')
    lines.append('── NOT owned here (container-injected hooks) ──')
    for doc in HOOK_ENV:
        lines.append(f'  {doc.name:38} default={doc.default:16} {doc.owner}')
    lines.append('')
    lines.append('── NOT owned here (external / dynamic) ──')
    for doc in EXTERNAL_ENV:
        lines.append(f'  {doc.name}\n      {doc.effect} ({doc.owner})')
    lines.append('')
    errors = [p for p in problems if p.level == 'error']
    warnings = [p for p in problems if p.level == 'warning']
    if errors:
        lines.append('=== ERRORS ===')
        lines += [f'  [{p.field}] {p.message}' for p in errors]
        lines.append('')
    if warnings:
        lines.append('=== WARNINGS ===')
        lines += [f'  [{p.field}] {p.message}' for p in warnings]
        lines.append('')
    if not errors and not warnings:
        lines.append('=== OK: no problems found ===')
    return '\n'.join(lines)


def _to_jsonable(cfg: ArtifactCfg) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        if is_dataclass(value):
            out[f.name] = {
                sub.name: (
                    _redact(f'{f.name}.{sub.name}', getattr(value, sub.name))
                    if f'{f.name}.{sub.name}' in _SECRET_FIELDS
                    else getattr(value, sub.name)
                )
                for sub in fields(value)
            }
        else:
            out[f.name] = value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m nexus.extensions.tasks.artifact.config')
    parser.add_argument('--json', action='store_true', help='machine-readable dump')
    parser.add_argument('--check', action='store_true', help='validate only, no dump')
    parser.add_argument(
        '-p',
        '--profile',
        default=os.getenv(PROFILE_ENV) or None,
        help=f'YAML profile to load (default: ${PROFILE_ENV})',
    )
    parser.add_argument(
        '--set',
        dest='overrides',
        action='append',
        default=[],
        metavar='a.b=c',
        help='override one field; repeatable',
    )
    parser.add_argument(
        '--emit-batch-args',
        action='store_true',
        help='print resolved legacy batch.* as `nexus run-batch` flags; the JSONL launcher is retired',
    )
    args = parser.parse_args(argv)

    reset()
    cfg = load(args.profile, args.overrides)

    if args.emit_batch_args:
        # run.sh evals this. batch.* cannot be read by the generator: cli/commands/run.py
        # builds BatchConfig from argv before the generator is loaded.
        flags = [
            '--worker-processes',
            str(cfg.batch.worker_processes),
            '--max-concurrent',
            str(cfg.batch.max_concurrent),
            '-d',
            cfg.batch.db,
            '--retry-failed',
            str(cfg.batch.retry_failed),
            '--export-format',
            'jsonl',
            '--export-path',
            cfg.batch.export_path,
        ]
        if cfg.budget.batch_timeout > 0:
            flags += ['--timeout', str(cfg.budget.batch_timeout)]
        print(' '.join(shlex.quote(f) for f in flags))
        return 0

    aliases = model_aliases(cfg)
    problems = validate(cfg, model_aliases=aliases)
    errors = [p for p in problems if p.level == 'error']

    if args.json:
        print(
            json.dumps(
                {
                    'config': _to_jsonable(cfg),
                    'provenance': cfg.provenance,
                    'problems': [{'level': p.level, 'field': p.field, 'message': p.message} for p in problems],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    elif args.check:
        for p in problems:
            print(f'[{p.level}] {p.field}: {p.message}')
        if not problems:
            print('OK: no problems found')
    else:
        print(format_report(cfg, problems, aliases))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())


__all__ = [
    'ArtifactCfg',
    'BudgetCfg',
    'ClaudeCodeCfg',
    'DataCfg',
    'DeployCfg',
    'EXTERNAL_ENV',
    'EnvDoc',
    'HOOK_ENV',
    'JudgeCfg',
    'LEGACY_ALIASES',
    'ModelsCfg',
    'ObsCfg',
    'Problem',
    'RuntimeCfg',
    'effective_limit',
    'format_report',
    'get',
    'model_aliases',
    'load',
    'reset',
    'validate',
]
