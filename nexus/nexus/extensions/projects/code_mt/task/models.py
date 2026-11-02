"""Data models for code_mt_task."""

from typing import Any, Literal

from pydantic import BaseModel

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
)


# === Scoring ===


class ScoreDimension(BaseModel):
    """A single scoring dimension (rule-based or LLM judge).

    `config` may carry an inline `LLMJudgeConfig` dict for llm_judge dimensions;
    if omitted, the dimension falls back to `task_input.llm_judge`.
    """

    name: str
    type: Literal['rule', 'llm_judge'] = 'rule'
    weight: float = 1.0
    config: dict[str, Any] = {}


class LLMJudgeConfig(BaseModel):
    """Generic LLM Judge configuration.

    The judge is fully field-agnostic: the prompt template is rendered with all
    scalar fields from reward.json + task_input.extra (plus a few code_mt
    standards: instance_id, task_type, attempt, max_turns, original_message).
    Datasets pick their own template and reference whatever fields they wrote.

    Extraction is also dataset-defined. Order of precedence (first match wins):
      1. `parser_import` — module::func(content, config) -> (score, reason);
         full custom parser, e.g. for non-JSON rubric outputs.
      2. `value_map` — explicit enum→float mapping for the value pulled at
         `score_field`. E.g. {"A":1.0,"B":0.5,"C":0.0} for letter grades.
      3. `score_scale` — (min, max) linear rescale to [0,1] for non 0-1
         numeric scales (e.g. (0,10) for a 0-10 rubric).
      4. Built-in coercion — bool / number clamp / pass|fail|yes|no enums.
    """

    enabled: bool = False
    template: str = ''
    template_path: str = ''
    score_field: str = 'score'
    reason_field: str = 'reason'
    model: str = ''
    api_base: str = ''
    api_key: str = ''
    # Explicit, self-documenting judge backend. Replaces the old implicit
    # (use_self bool + has_env_judge inference) selection so a dataset states
    # WHERE its judge runs instead of relying on env-var side effects:
    #   'self'    — score with the policy model via ctx.openai_client (host-side,
    #               always reachable). AUTHORITATIVE: a stray JUDGE_* exported in
    #               the launch env can NEVER flip this off. Judge tokens are kept
    #               out of training_data (see scoring._create_completion).
    #   'host'    — dedicated external judge on the host (LiteLLMClient →
    #               JUDGE_OPENAI_BASE_URL/MODEL/API_KEY gateway).
    #   'sandbox' — judging runs IN-SANDBOX (harbor Verifier test.sh → judge.py →
    #               /logs/verifier/reward.json). The host-side judge is a no-op;
    #               a rule dimension surfaces the score from reward.json.
    #   ''        — auto/legacy: 'self' when use_self else 'host'. use_self is now
    #               authoritative here too (the has_env_judge flip is gone), so a
    #               builder self-judge and a repoqa sandbox judge coexist in one
    #               run without the host env leaking between them.
    judge_backend: Literal['', 'self', 'host', 'sandbox'] = ''
    use_self: bool = True
    max_tokens: int = 1024
    temperature: float = 0.0
    # Standard [verifier.env] JUDGE_* wiring (see scoring._resolve_judge_params).
    # These carry the dedicated third-party judge's request knobs; when the
    # dataset injects them via [verifier.env] they override these defaults.
    timeout: float = 1200.0
    max_retries: int = 3
    extra_body: dict[str, Any] = {}
    # Dataset-defined extraction overrides (see docstring for precedence).
    value_map: dict[str, float] = {}
    score_scale: tuple[float, float] | None = None
    parser_import: str = ''
    # Optional prompt builder for judges whose input is NOT on disk (no grader /
    # reward.json field) but must be pulled from the agent's runtime trajectory —
    # e.g. RepoQA/rubric judges that score the model's own answer or its whole
    # conversation. "module.path::func" with signature
    #   (template, ctx, task_input, eval_result, judge_config) -> str
    # returning the fully-rendered prompt. Only invoked when set, so judges that
    # rely on the default {key} render (e.g. builder anti_hack) are unaffected.
    prompt_builder_import: str = ''


# === Task Input/Output ===


class CodeMTTaskInput(TaskRunInput, WithAgentRunner):
    instance_id: str = ''
    task_type: str = ''

    message: str = ''
    system_prompt: str = ''
    workdir: str = '/app'

    # Runtime — harbor-native runtime topology, same names as harbor_bridge.
    # main runtime hosts the agent; a separate 'verifier' runtime is acquired
    # only for verifier_mode='separate' (SWE) grading.
    infer_runtime_name: str = 'main'
    verifier_runtime_name: str = 'verifier'

    # Setup
    startup_jitter_ms: int = 0
    # Legacy pre-0720 setup_commands (e.g. git-history truncation). 0720 datasets
    # carry startup entirely in [environment.boot]; kept only so pre-0720 dirs
    # that ship [other].setup_commands still run them (after boot).
    setup_commands: list[str] = []
    task_dir: str = ''

    # Agent
    agent_timeout: float = 7200
    agent_timeout_mode: Literal['soft', 'hard'] = 'soft'

    # Oracle — when True, the agent loop is swapped for the golden-solution
    # executor: {task_dir}/solution is uploaded to /solution and
    # `bash /solution/solve.sh` is run instead of run_agent. Everything after
    # (artifact collection, verifier, scoring) is UNCHANGED, exactly like
    # harbor_bridge — oracle validates the dataset by making a correct row score
    # reward=1.0. Oracle is single-shot (forces one turn).
    run_oracle: bool = False

    # === Harbor-native verifier (identical vocabulary to harbor_bridge) ===
    # verifier_mode replaces code_mt's old proprietary eval_mode. Resolved from
    # [verifier].environment_mode via harbor's resolve_task_verifier_mode:
    #   'shared'   — grade in the SAME runtime as the agent (builder / repoqa).
    #   'separate' — collect artifacts (e.g. /tmp/model.patch via
    #                [[verifier.collect]]) from main, grade them in a fresh
    #                'verifier' runtime (SWE). The dataset owns collection +
    #                artifacts, so the task carries NO SWE-specific git logic.
    # Grading itself is harbor's Verifier: it uploads tests/ -> /tests, runs
    # test.sh, and reads /logs/verifier/reward.json — the SAME contract for all
    # datasets. code_mt only layers the MT loop + host-side score mixing on top.
    verifier_mode: Literal['shared', 'separate'] = 'shared'
    test_timeout: float = 1800.0
    skip_tests_upload: bool = False
    verifier_tests_prebaked: bool = False

    # Network isolation capability flags — MUST match the requested harbor
    # network policy or harbor's BaseEnvironment rejects the task with
    # "network_mode='no-network' is not supported". The generator derives these
    # from task.toml [environment].allow_internet / network_mode (allow_internet
    # =false -> no-network -> network_isolation_enabled=True). Default False
    # means "internet allowed" (public), the correct default for tasks whose
    # task.toml carries NO allow_internet key (e.g. all 0720 terminal rows).
    # Threaded into GongfengEnvironment as capabilities.disable_internet,
    # identical to harbor_bridge (task.py network_isolation_enabled=...).
    network_isolation_enabled: bool = False
    verifier_network_isolation_enabled: bool = False

    # Image boot hook ([environment.boot]). gongfeng replaces the image
    # ENTRYPOINT, so in-image init (repo checkout, service start) never runs on
    # its own. harbor's Task model has NO boot field, so we read it from raw
    # task.toml and drive it via harbor_bridge._boot_image_services (launch
    # detached, then poll [environment.healthcheck] readiness). Same fields as
    # HarborBridgeTaskInput so the shared helper is reused verbatim.
    boot_command: str = ''
    boot_env: dict[str, str] = {}
    # Wall-clock for the detached boot launch command (seconds). Set via the
    # parquet index overlay (like per_observation_timeout), NOT task.toml, so
    # slow-booting images get more room without editing every task. Boot
    # launch/healthcheck failure surfaces as INITIALIZATION_ERROR, not
    # UNHANDLED_ERROR. Default 900s so slow-booting images boot without a
    # per-task/parquet override (the parquet overlay still narrows it per category).
    boot_timeout: float = 900.0
    healthcheck_command: str = ''
    healthcheck_url: str = ''
    healthcheck_retries: int = 180
    healthcheck_interval_sec: float = 3.0
    healthcheck_timeout_sec: float = 3.0
    healthcheck_start_period_sec: float = 0.0

    # Standard [verifier.env] passthrough. The generator (offline) and the
    # hytuner dataset loader / index overlay (training) both populate this from
    # task.toml's [verifier.env]. It is the SINGLE carrier both paths share, so
    # the dedicated LLM judge sources its JUDGE_OPENAI_* model config from here
    # regardless of who built the task_config. Empty for swe/tb (no llm_judge).
    verifier_env: dict[str, str] = {}

    # MT
    mt_max_turns: int = 1
    mt_feedback_template: str = ''
    mt_feedback_max_length: int = 3000

    # NOTE: the INITIAL prompt + system prompt are owned by the AGENT now
    # (code_mt_agent loads instruction.md / system_prompt.md). The task no longer
    # carries a prompt_template — it hands the agent user_prompt + working_dir.
    # Turn-0 env setup (setup_strategy) also moved to the AGENT (CodeMTAgentInput),
    # so the task no longer orchestrates env prep and can drive any agent.

    # === Scoring ===
    # Multi-dimension scoring is the only path. If empty, raw verifier score
    # is returned as-is (binary by default — >=1 → 1, else 0; see task.py).
    score_dimensions: list[ScoreDimension] = []
    score_merge: Literal['multiply', 'weighted_sum', 'min', 'sequential'] = 'sequential'
    llm_judge: LLMJudgeConfig = LLMJudgeConfig()
    score_kind: Literal['binary', 'partial'] = 'binary'

    # Hooks
    pre_agent_hook: str | None = None
    post_agent_hook: str | None = None
    # Per-turn eval cleanup strategy (`module::func`), run in the MT loop AFTER
    # scoring and BEFORE the next retry turn. Builder tasks set this to
    # ``...cleanup::builder_cleanup`` to reset /testbed (the judge may have applied
    # golden.patch) and re-materialize /tmp/*.patch; build_cc sets it to
    # ``...build_cc_upload::finalize`` to publish + wipe the inner DinD image.
    # Empty for swe/tb/repoqa — the loop short-circuits and never imports it.
    eval_cleanup_strategy: str | None = None

    # === Reward-hacking / baseline monitoring (scoring plug-in) ===
    # Scaffold-agnostic plug-in (see reward_hacking/). Monitors FLAG by default;
    # penalties default to 0 (no-op) and anti-hack termination is OFF. Overlaid
    # via the parquet task_config, never task.toml. Penalties subtract from the
    # reward but never push it below reward_lower_bound (hard floor 0.0).
    # Multi-select whitelist of monitor families to run + penalize. Empty = ALL
    # (backward compatible). Allowed values: codeswitch, output_format_bad_pattern,
    # str_replace_noop, let_me_loop, anti_hack, baseline. A family not listed is
    # neither flagged nor penalized.
    monitors: list[str] = []
    enable_output_monitoring: bool = True
    enable_reward_hacking_detection: bool = True
    enable_anti_hack: bool = False
    # Validation split marker. Detection and every hack metric stay ON; only the
    # CONSEQUENCE is withheld (no CM-1 score zeroing, no CODE_REWARD_HACKING
    # flip). Aligns SWE, which gates the consequence alone:
    # `tasks/swe/task.py` -> `hack_reward_override=None if is_validation else ...`.
    # Without this an eval sample that trips one detector is flipped to
    # CODE_REWARD_HACKING and, on any run listing that reason in
    # `additional_terminate_reason`, dropped outright — corrupting the eval
    # denominator and thus the evaluation verdict itself.
    is_validation: bool = False
    # PREVENTION half: when True, the task appends the anti-hack notice to the
    # turn-0 message (the ONLY prompt injection in code_mt). Detection half is
    # enable_anti_hack above. Kept as an explicit task field so the switch lives
    # in the task layer (the agent never touches the prompt).
    enable_anti_hack_prompt: bool = False
    codeswitch_penalty: float = 0.0
    codeswitch_penalty_reward_threshold: float = 1.0
    output_format_bad_pattern_penalty: float = 0.0
    output_format_bad_pattern_reward_threshold: float = 1.0
    str_replace_old_eq_new_penalty: float = 0.0
    str_replace_old_eq_new_reward_threshold: float = 1.0
    let_me_loop_penalty: float = 0.0
    let_me_loop_reward_threshold: float = 1.0
    baseline_penalties: dict[str, float] = {}
    baseline_penalty_reward_threshold: float = 1.0
    reward_lower_bound: float = 0.0

    # === Extra fields (dataset-specific) ===
    # Free-form dataset state. All scalar values are auto-exposed to prompt /
    # mt_feedback / llm_judge templates via {key} substitution.
    extra: dict[str, Any] = {}


class CodeMTTaskOutput(TaskRunOutput, WithScore):
    instance_id: str = ''
    task_type: str = ''
    # Non-empty only when score < 1.0. Classifies WHY a code_mt run failed,
    # derived from eval_result so the DB carries a failure signal for SWE tasks
    # (which otherwise only expose resolved / f2p_rate / p2p_rate). Empty on pass.
    failure_type: str = ''
    exit_status: str = ''
    iterations: int = 0
    eval_result: dict[str, Any] = {}
    score_details: dict[str, float] = {}
    patch_text: str = ''
    # MT loop bookkeeping — surfaced into db's task_output column so callers can
    # filter without scanning trajectories for submit tool positions.
    termination_reason: str = ''  # Canonical hytuner vocab (ENV_DONE / MAX_STEPS / ...)
    mt_max_turns: int = 0  # Configured ceiling
    mt_turns_executed: int = 0  # Turns actually run (incl. the successful one)
    mt_successful_turn: int = 0  # 1-based turn INDEX that hit score>=1.0; 0 if all turns failed
    # NOTE: the two fields below are PER-TURN agent-iteration COUNTS, not turn indices.
    # The whole-run total is `iterations` (== mt_successful_turn_iters + sum(mt_failed_turn_iters)).
    mt_successful_turn_iters: int = 0  # agent iterations spent inside the turn that passed (0 if never)
    mt_failed_turn_iters: list[int] = []  # agent iterations spent inside each failed turn (one entry per failed turn)
    # Reward-hacking / baseline monitor detection metadata (flag mirrors +
    # reward_extra_info). Empty by default; consumed by hytuner's runner, which
    # reads task_output.extra_fields. Never affects score unless penalties are on.
    extra_fields: dict[str, Any] = {}
    # Log-only monitor report: a human-readable mirror of which monitor families
    # were selected, which triggered, and how much each deducted (base_score ->
    # final_score). Built in task.py AFTER penalties are applied — it carries NO
    # scoring logic of its own, it just surfaces the already-computed deductions
    # into the DB so task_config + this field fully explain the final reward.
    monitor: dict[str, Any] = {}
