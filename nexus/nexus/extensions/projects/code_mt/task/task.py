"""Code MT task — harbor-native multi-turn executor.

This is, by design, harbor_bridge + a multi-turn loop + host-side score mixing.
It drives the SAME 0720 harbor datasets as harbor_bridge and grades them with the
SAME machinery: harbor's `Verifier` running `tests/test.sh -> /logs/verifier/
reward.json` inside a `GongfengEnvironment`, with `[verifier].environment_mode`
(shared / separate) resolved by harbor's `resolve_task_verifier_mode`. There is
NO code_mt-private eval_mode and NO code_mt-private verifier — the harbor-native
leaf helpers are imported verbatim from harbor_bridge so there is ONE
implementation of boot / collect / verify / deliverable persistence.

code_mt owns ONLY what harbor_bridge (single-turn) does not:
- the MT retry loop (mt_max_turns turns, dataset-owned mt_feedback between turns),
- host-side score mixing: `score_dimensions` / `llm_judge` (e.g. builder's
  anti_hack, which is NOT in judge.sh and must be judged host-side) merged via
  `score_merge`, then snapped to 0/1 or kept float per `[judge].reward_kind`
  (score_kind),
- training monitoring (spans, termination_reason, per-turn iteration bookkeeping),
- the golden-solution oracle executor (run_oracle).

Runtime topology mirrors harbor_bridge, except the main runtime is kept ALIVE
across MT turns (the agent keeps editing it); for separate mode a fresh
'verifier' runtime is acquired and torn down per grading turn.
"""

import asyncio
import base64
import contextlib
import json
import logging
import shutil
import tempfile
import time

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from nexus.core import require_context, task_registry
from nexus.core.registry import import_from_path
from nexus.extensions.harbor_bridge.environment import GongfengEnvironment
from nexus.extensions.harbor_bridge.task import (
    _boot_image_services,
    _collect_artifacts,
    _configure_git_safe_directory,
    _run_boot_healthcheck,
    _run_collect_hooks,
    _run_harbor_verifier,
)
from nexus.extensions.projects.code_mt.task import loglib
from nexus.extensions.projects.code_mt.task.models import CodeMTTaskInput, CodeMTTaskOutput
from nexus.extensions.projects.code_mt.task.reward_hacking import (
    apply_code_mt_penalties,
    build_monitor_report,
    resolve_reward_hacking_config,
    run_code_mt_monitors,
)
from nexus.extensions.projects.code_mt.task.scoring import (
    build_format_vars,
    compute_score,
    extract_agent_trajectory,
    render_template,
)
from nexus.runtime.executors import FileItem
from nexus.trace import get_current_span_id, update_span
from nexus.utils.common import with_random_jitter


if TYPE_CHECKING:
    # Type-only: the runtime import stays lazy inside scan_trajectory_for_anti_hack
    # (kept local to mirror the existing deferred-import pattern in this module).
    from nexus.extensions.projects.code_mt.task.anti_hack import AntiHackViolation


logger = logging.getLogger(__name__)
AGENT_DELIVERABLE_DIR = '/tmp/agent_deliverable'

# L3 = the repo-blind wide-net regex sweep inherited from hytuner's SWE agent
# loop (detect_local_command_hack / detect_local_output_hack). Disabled: it ran
# under `if cv is None`, and L1 returns None BOTH for "clean" and for
# "deliberately exempted", so L3 silently revoked every L1 exemption — `--help`,
# `--dry-run`, cargo vendor paths, clones of non-target repos, `git ls-remote`
# against a blocked network, `git merge-base`, `git show <tag>^`. Its one unique
# capability, `git_commit_probe`, is reimplemented correctly as L4 (ancestor
# semantics, SWE reward_hacking v1.3.0).
#
# The code and the pattern table stay in place: summarize_anti_hack_events()
# iterates _ANTI_HACK_LOCAL_COMMAND_PATTERNS DIRECTLY, so the swanlab counters
# (anti_hack_commit_probe / _external_source / _git_log) keep reporting exactly
# as before — only the violation-producing path is off.
_L3_ENABLED = False


async def _persist_code_mt_deliverable(runtime: Any, ctx: Any) -> None:
    """Stage the Code MT answer and trajectory for its in-sandbox repo-QA judge."""
    answer, transcript, _ = extract_agent_trajectory(ctx)
    encoded_answer = base64.b64encode(answer.encode()).decode()
    encoded_transcript = base64.b64encode(transcript.encode()).decode()
    try:
        await runtime.run_command(command=f'mkdir -p {AGENT_DELIVERABLE_DIR}', timeout=60)
        await runtime.run_command(
            command=f'base64 -d > {AGENT_DELIVERABLE_DIR}/answer.txt',
            stdin=encoded_answer,
            timeout=60,
        )
        await runtime.run_command(
            command=f'base64 -d > {AGENT_DELIVERABLE_DIR}/trajectory.txt',
            stdin=encoded_transcript,
            timeout=60,
        )
        logger.info(
            'Persisted Code MT deliverable: answer=%s chars, transcript=%s chars',
            len(answer),
            len(transcript),
        )
    except Exception as exc:
        logger.warning('Failed to persist Code MT deliverable: %s', exc)


# ============================================================================
# Full-blood anti-hack: TASK-side post-processing (truncation-exempt)
# ============================================================================
#
# Decoupling contract (user hard requirement):
#   * TASK-side anti-hack is gated by CodeMTTaskInput.enable_anti_hack (task_config).
#   * The AGENT is deliberately NOT touched: detection is 100% task-side
#     post-processing of the completed trajectory, so code_mt/bash needs no extra
#     logic and there is no agent_config switch (a switch gating nothing would be
#     dead config). This is exactly the user's "bash tool 的检测放到 task 内进行
#     后处理，truncation 可以豁免" — the truncation trade-off is accepted.
#
# The detector itself is a byte-for-byte copy of hytuner tasks/swe/anti_hack.py
# (all 14 channels) → the mask set is identical to hytuner.


def _observation_text(content: Any) -> str:
    """Flatten a role='tool' ``content`` to text for the regex output detectors.

    ``_code_mt_messages`` always stored a plain string, but the chat_tracker
    trajectory (the fallback that makes codex / hycode / mini_swe visible) keeps
    tool content verbatim, so it can be a list of OpenAI content parts or a dict.
    detect_*_output are ``re``-based and raise TypeError on anything but str/bytes,
    which would drop the entire output tier for those scaffolds.
    """
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                # {'type': 'text', 'text': ...} and similarly-shaped variants
                parts.append(str(p.get('text') or p.get('content') or p.get('output') or ''))
            else:
                parts.append(str(p))
        return '\n'.join(x for x in parts if x)
    if isinstance(content, dict):
        for key in ('text', 'content', 'output', 'stdout'):
            val = content.get(key)
            if isinstance(val, str) and val:
                return val
        try:
            return json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(content)
    return str(content)


def scan_trajectory_for_anti_hack(
    messages: list[dict] | None,
    *,
    repo: str | None,
    working_dir: str | None,
    instance_id: str | None,
    base_commit: str | None = None,
    history_truncated: bool | None = None,
) -> 'tuple[list[AntiHackViolation], dict]':
    """Re-scan a completed code_mt trajectory for source-download hacks.

    Walks every ``bash`` tool_call (structuring text_parse trajectories first via
    ``normalize_text_parse_messages``) and runs, per command + its following
    observation, the precise repo-aware full-blood detector
    (detect_source_download_hack[_output]), plus — when ``base_commit`` is known
    — the L4 ancestor verifier that replaces L3's blunt ``git_commit_probe``.

    Corrections to the original hytuner-derived behaviour:

    * **L3 is gated OFF** (``_L3_ENABLED``). Its trigger was ``if cv is None``,
      but L1 returns ``None`` both for "clean" and for "explicitly exempted", so
      the repo-blind sweep silently overrode every L1 exemption (``--help``,
      ``--dry-run``, cargo vendor paths, non-target clones).
    * **Network failures are exempted per violation.** A single refused fetch is
      not a hack, while a compound command cannot hide another successful read
      behind one blocked-network message.
    * **Tool results are paired by tool_call_id.** Missing IDs use the same
      positional fallback as the SWE trajectory adapter.
    * **L4 replaces ``git_commit_probe``** with SWE v1.3.0's ancestor semantics:
      reading ``base_commit`` or any commit already visible in ancestor-walk
      output is legitimate; only unverified/remote-tracking reads are flagged.

    ``base_commit``/``history_truncated`` are optional and default to None, so a
    task without them (Cyber / Terminal / RepoQA / StackOverflow / NL2repo) keeps
    working — L4 simply self-disables there, matching SWE's own
    "skipped without a base_commit" contract.

    Returns ``(deduped_violations, summary)`` where summary carries the
    observability counters (anti_hack_commit_probe_count / _external_source /
    _git_log ...). Pure/side-effect-free so it is unit-testable without a
    sandbox. Never raises: detector failures on any single command are swallowed
    (fail-open per command).
    """
    from nexus.extensions.projects.code_mt.task.anti_hack import (
        AntiHackViolation,
        ancestor_commits,
        detect_local_command_hack,
        detect_local_output_hack,
        detect_source_download_hack,
        detect_source_download_hack_output,
        detect_unverified_commit_reads,
        shell_command_events,
        should_exempt_blocked_network_violation,
        summarize_anti_hack_events,
    )
    from nexus.extensions.projects.code_mt.task.reward_hacking.text_parse_bridge import (
        normalize_text_parse_messages,
    )

    struct = normalize_text_parse_messages(messages) or (messages or [])
    violations: list[AntiHackViolation] = []
    # A task is "repo-anchored" when it actually has a target repository. Both
    # repo and base_commit are sunk from {task_dir}/task.toml [other] by
    # loader.py, so this is True for SWE/SWEBuilder and Setup and False for
    # Cyber / Terminal / RepoQA / StackOverflow / NL2repo. It gates only the
    # unknown-repo WIDENING inside detect_source_download_hack (see its
    # docstring) — repo-aware rules are unaffected either way.
    _repo_anchored = bool(repo) or bool(base_commit)
    # (command, observation) pairs for the observability summary (git_commit_probe
    # / external-source / git_log counts), mirroring hytuner's loop events.
    events: list[tuple[str, str]] = []
    # The local-history pass receives every event. Network failure applies only
    # to network violations and cannot suppress a local read in the same shell.
    l4_events: list[tuple[str, str]] = []

    for cmd, obs in shell_command_events(struct):
        events.append((cmd, obs))
        l4_events.append((cmd, obs))

        # --- L1/L2: precise repo-aware source-download detector (command). ---
        cv = None
        try:
            cv = detect_source_download_hack(
                cmd,
                repo=repo,
                working_dir=working_dir,
                instance_id=instance_id,
                repo_anchored=_repo_anchored,
            )
            if cv and not should_exempt_blocked_network_violation(cmd, obs, cv['reason']):
                violations.append(cv)
        except Exception:
            logger.exception('anti-hack command scan failed for one command')
        # --- L3: repo-blind wide-net sweep. DISABLED — see _L3_ENABLED. Its
        #     `cv is None` trigger could not tell "clean" from "exempted", so
        #     it silently revoked every L1 exemption. Kept behind the flag
        #     (not deleted) so the patterns remain available to
        #     summarize_anti_hack_events, which reads the table directly and
        #     therefore keeps reporting the swanlab counters unchanged. ---
        if _L3_ENABLED and cv is None:
            try:
                lcv = detect_local_command_hack(cmd)
                if lcv:
                    violations.append(lcv)
            except Exception:
                logger.exception('anti-hack local command scan failed for one command')

        if obs:
            # --- L1/L2 output detector, then L3 output fallback. ---
            ov = None
            try:
                ov = detect_source_download_hack_output(
                    obs, command=cmd, repo=repo, working_dir=working_dir, instance_id=instance_id
                )
                if ov:
                    violations.append(ov)
            except Exception:
                logger.exception('anti-hack output scan failed for one command')
            if _L3_ENABLED and ov is None:
                # --- L1/L2 output detector, then L3 output fallback. ---
                try:
                    lov = detect_local_output_hack(obs, command=cmd)
                    if lov:
                        violations.append(lov)
                except Exception:
                    logger.exception('anti-hack local output scan failed for one command')

    # --- L4: git-commit ancestor verifier (SWE reward_hacking v1.3.0 semantics).
    #     Restores the legitimate-read distinction that L3's `git_commit_probe`
    #     lacked. Self-disables without a base_commit, and when the history is
    #     known to be truncated (ancestor set would be unknowable -> would
    #     over-flag). `history_truncated is not True` keeps an UNMEASURED
    #     truncation (None) behaving as before, matching SWE. ---
    if base_commit and history_truncated is not True:
        try:
            _ancestors = ancestor_commits(l4_events)
            violations.extend(detect_unverified_commit_reads(l4_events, base_commit=base_commit, ancestors=_ancestors))
        except Exception:
            logger.exception('anti-hack L4 commit-read scan failed')

    # Dedup on (reason, command[:200]) — stable order.
    seen: set = set()
    deduped: list[AntiHackViolation] = []
    for v in violations:
        if not isinstance(v, dict):
            continue
        key = (v.get('reason'), (v.get('command') or '')[:200])
        if key not in seen:
            seen.add(key)
            deduped.append(v)
    # Observability summary (git_commit_probe / external-source / git_log
    # counts) — does not affect masking, mirrors hytuner's loop metrics.
    try:
        summary = summarize_anti_hack_events(events, cast('list[dict]', deduped))
    except Exception:
        logger.exception('anti-hack summary failed')
        summary = {}
    return deduped, summary


# ============================================================================
# File transfer helpers (skills upload + oracle solution)
# ============================================================================


def _read_local_file_to_file_item(local_file: Path, container_path: str) -> FileItem:
    if not local_file.is_file():
        raise ValueError(f'Path is not a file: {local_file}')
    try:
        content = local_file.read_text(encoding='utf-8')
        encoding = 'utf8'
    except UnicodeDecodeError:
        content = base64.b64encode(local_file.read_bytes()).decode('utf-8')
        encoding = 'base64'
    return FileItem(path=container_path, content=content, encoding=encoding)


def _read_local_directory_to_file_items(local_dir: Path, container_base_path: str) -> list[FileItem]:
    if not local_dir.is_dir():
        raise NotADirectoryError(f'Path is not a directory: {local_dir}')
    items: list[FileItem] = []
    for fp in local_dir.rglob('*'):
        if not fp.is_file():
            continue
        rel = fp.relative_to(local_dir)
        items.append(_read_local_file_to_file_item(fp, f'{container_base_path.rstrip("/")}/{rel.as_posix()}'))
    return items


async def _upload_dir(runtime: Any, local_dir: Path, target: str) -> int:
    if not local_dir.exists() or not local_dir.is_dir():
        return 0
    await runtime.run_command(command=f'mkdir -p {target}', timeout=30)
    items = _read_local_directory_to_file_items(local_dir, target)
    if items:
        await runtime.upload_file(files=items, overwrite=True)
    return len(items)


async def _upload_agent_visible_files(runtime: Any, task_dir: str) -> None:
    """Upload `{task_dir}/environment/skills/` to `/skills` if present.

    Generic dataset convention: skills are agent-visible reference files (builder
    ships them). No-op when the directory is absent. tests/ is NEVER uploaded here
    — harbor's Verifier uploads it to /tests only at grading time (post-agent).
    """
    if not task_dir:
        return
    n = await _upload_dir(runtime, Path(task_dir) / 'environment' / 'skills', '/skills')
    if n:
        logger.debug('Uploaded %d skill files', n)


async def _run_oracle_solution(runtime: Any, task_input: CodeMTTaskInput) -> str:
    """Golden-solution executor — the agent-loop replacement for run_oracle.

    Uploads {task_dir}/solution to /solution and runs `bash /solution/solve.sh`.
    For separate-mode SWE the solve.sh mutates /testbed (and/or writes
    /tmp/model.patch); the harbor collect hook then captures the golden diff, so
    grading is UNCHANGED — oracle just proves a correct row scores reward=1.0.
    Returns exit_status ('submitted' on rc==0, else 'error').
    """
    solution_dir = Path(task_input.task_dir) / 'solution'
    n = await _upload_dir(runtime, solution_dir, '/solution')
    if n == 0:
        loglib.phase_fail(logger, 'ORACLE', f'no solution/ dir under {task_input.task_dir}')
        return 'error'
    await runtime.run_command(command='chmod -R +x /solution', timeout=30)
    loglib.phase_start(logger, 'ORACLE', f'Running /solution/solve.sh ({n} file(s) uploaded)')
    result = await runtime.run_command(command='bash /solution/solve.sh', cwd=task_input.workdir, timeout=9999)
    if result.return_code == 0:
        loglib.phase_ok(logger, 'ORACLE', 'solve.sh rc=0')
        return 'submitted'
    loglib.phase_fail(logger, 'ORACLE', f'solve.sh rc={result.return_code}')
    stderr = (result.stderr or '').strip()
    if stderr:
        loglib.phase_info(logger, 'ORACLE', f'stderr: {stderr[:400]}')
    return 'error'


# ============================================================================
# Legacy setup_commands (pre-0720 only)
# ============================================================================


async def _run_setup_commands(runtime: Any, commands: list[str], task_config: dict[str, Any]) -> None:
    loglib.banner(logger, 'cyan', 'SETUP')
    loglib.phase_start(logger, 'SETUP', f'Running {len(commands)} setup command(s)')
    ok_count = 0
    fail_count = 0
    for i, cmd in enumerate(commands):
        try:
            formatted = cmd.format(**task_config)
        except (KeyError, IndexError):
            formatted = cmd
        t0 = time.monotonic()
        # Repo checkout (git clone --filter=blob:none + fetch + reset) is the
        # heaviest setup command; 300s was too tight for large repos. Allow
        # task_config to override (setup_timeout), else 900s to match the old
        # boot_timeout budget. Timeout raises -> UNHANDLED_ERROR (fail-fast),
        # which is the intended behavior for a failed environment build.
        setup_timeout = float(task_config.get('setup_timeout') or 900.0)
        result = await runtime.run_command(command=formatted, timeout=setup_timeout)
        elapsed = time.monotonic() - t0
        short = formatted if len(formatted) <= 120 else formatted[:117] + '...'
        if result.return_code == 0:
            ok_count += 1
            loglib.phase_ok(logger, 'SETUP', f'cmd[{i}] rc=0 ({elapsed:.1f}s): {short}')
        else:
            fail_count += 1
            loglib.phase_fail(logger, 'SETUP', f'cmd[{i}] rc={result.return_code} ({elapsed:.1f}s): {short}')
            stderr = (result.stderr or '').strip()
            if stderr:
                loglib.phase_info(logger, 'SETUP', f'stderr: {stderr[:400]}')
    if fail_count == 0:
        loglib.phase_ok(logger, 'SETUP', f'Setup complete ({ok_count}/{len(commands)} ok)')
    else:
        loglib.phase_fail(logger, 'SETUP', f'Setup had {fail_count} failure(s) ({ok_count}/{len(commands)} ok)')


# ============================================================================
# Prompt / feedback rendering
# ============================================================================


def _read_task_file(task_dir: str, fname: str) -> str:
    """Read {task_dir}/{fname} VERBATIM (no rendering). '' if absent.

    Used by the task to source the turn-0 instruction / system_prompt from the
    dataset's on-disk files when the parquet does not override them. Verbatim on
    purpose: standard-harbor instruction.md is plain text, and SWEBuilder's baked
    instruction embeds literal braces (e.g. {host}) — any str.format() here would
    crash, so we never format.
    """
    if not task_dir:
        return ''
    p = Path(task_dir) / fname
    return p.read_text(encoding='utf-8') if p.is_file() else ''


def _render_mt_feedback(
    task_input: CodeMTTaskInput,
    eval_result: dict[str, Any],
    score: float,
    attempt: int,
) -> str:
    """Render the retry-turn prompt.

    No baked-in template. If the dataset didn't provide mt_feedback_template,
    fall back to the original message — the agent simply gets another attempt
    with whatever state remains in the runtime.
    """
    if not task_input.mt_feedback_template:
        return task_input.message
    test_output = str(eval_result.get('test_output', ''))[: task_input.mt_feedback_max_length]
    fv = build_format_vars(
        task_input,
        eval_result,
        attempt=attempt,
        original_message=task_input.message,
        extra_overrides={'score': score, 'test_output': test_output},
    )
    return render_template(task_input.mt_feedback_template, fv)


# ============================================================================
# Monitoring helpers
# ============================================================================


def _extract_runtime_id(runtime: Any) -> str:
    base_url = getattr(runtime, 'base_url', '')
    if base_url:
        parts = str(base_url).rstrip('/').split('/')
        return parts[-1] if parts else ''
    return ''


# Canonical hytuner termination_reason vocabulary. The agent-loop statuses are
# owned by CodeMTExitStatus (agent/models.py) so the agent and task never drift;
# this dict only adds task-level statuses the agent enum does not model (e.g.
# 'not_started' before the agent ever ran). Hytuner treats any value outside the
# vocabulary as 'UNKNOWN'.
_TASK_ONLY_TERMINATION_REASON: dict[str, str] = {
    'not_started': 'INITIALIZATION_ERROR',
}


def _canonical_termination_reason(exit_status: str) -> str:
    from nexus.extensions.projects.code_mt.agent.models import CodeMTExitStatus

    if exit_status in _TASK_ONLY_TERMINATION_REASON:
        return _TASK_ONLY_TERMINATION_REASON[exit_status]
    reason = CodeMTExitStatus.termination_reason_for(exit_status)
    return reason if reason != 'UNKNOWN' else 'UNHANDLED_ERROR'


def _classify_failure(eval_result: dict[str, Any] | None, score: float) -> str:
    """Derive a failure_type for a run that did not reach score>=1.0.

    SWE-SPECIFIC BY DESIGN: reads only SWE grading keys (resolved / f2p_rate /
    p2p_rate / stage). If eval_result carries none of them (repoqa llm_judge, or
    any non-SWE task) it returns '' so the field never conflicts with builder's
    own failure_type vocabulary. Returns '' on pass too.
    """
    if score >= 1.0:
        return ''
    er = eval_result if isinstance(eval_result, dict) else {}
    if not any(k in er for k in ('resolved', 'f2p_rate', 'p2p_rate', 'stage')):
        return ''
    stage = er.get('stage')
    if stage and stage != 'graded':
        err = (er.get('error') or '').lower()
        if 'timeout' in err or 'timed out' in err:
            return 'eval_timeout'
        if 'patch' in err and 'apply' in err:
            return 'patch_apply_failed'
        return f'eval_error:{stage}'
    f2p = er.get('f2p_rate')
    p2p = er.get('p2p_rate')
    if f2p is None and p2p is None:
        return 'unresolved' if er.get('resolved') is False else 'graded_fail'
    if p2p is not None and p2p == 0.0 and (f2p or 0.0) == 0.0:
        return 'build_broken'
    if p2p is not None and p2p < 1.0:
        return 'p2p_regression'
    if p2p == 1.0 and f2p is not None and f2p < 1.0:
        return 'f2p_unresolved'
    return 'graded_fail'


async def _run_hook(hook_path: str, **kwargs: Any) -> Any:
    # Hook paths (pre_agent_hook / post_agent_hook / eval_cleanup_strategy) come
    # straight from task.toml and may still carry the pre-move prefixes
    # (nexus.extensions.tasks.code_mt.* / nexus.extensions.agents.code_mt.*).
    # import_from_path bypasses the loader, so a stale prefix raises
    # ModuleNotFoundError — and every caller wraps this in contextlib.suppress,
    # which would SILENTLY skip Builder reset / build_cc image upload. Canonicalize
    # to the current projects.code_mt.* location first (idempotent; leaves
    # already-current and unrelated paths untouched).
    from nexus.extensions.projects.code_mt.task._module_aliases import canonicalize_import_path

    resolved = canonicalize_import_path(hook_path)
    assert resolved is not None  # canonicalize returns non-None for non-empty input
    fn = import_from_path(resolved)
    return await fn(**kwargs)


# ============================================================================
# Harbor-native grading (shared / separate) — reuses harbor_bridge helpers
# ============================================================================


def _read_reward_payload(trial_paths: Any, test_stdout: str) -> tuple[float, dict[str, Any]]:
    """Read the reward the dataset's test.sh wrote (host side).

    harbor's Verifier types `rewards` as numeric-only, but code_mt's MT feedback
    + failure classification need the non-numeric fields too (stage / failure_type
    / output_tail for builder, reason for repoqa). So we read the whole payload
    from trial_paths.reward_json_path and keep everything except `reward`.

    Fallback: if there's no reward.json but a plain reward.txt exists, read the
    numeric score from it. This mirrors harbor's own Verifier.verify() (reward.json
    first, reward.txt second) so a task that only writes reward.txt still scores
    instead of silently returning 0.0. reward.txt carries only the number, so the
    rich `extra` fields stay empty there (every downstream consumer reads them with
    .get(), so an empty extra is safe).
    """
    raw_score = 0.0
    extra: dict[str, Any] = {}
    p = trial_paths.reward_json_path
    txt = getattr(trial_paths, 'reward_text_path', None)
    if p.exists():
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            payload = json.loads(p.read_text(encoding='utf-8') or '{}')
            if isinstance(payload, dict):
                raw_score = float(payload.get('reward', 0.0))
                extra = {k: v for k, v in payload.items() if k != 'reward'}
    elif txt is not None and txt.exists():
        with contextlib.suppress(ValueError):
            raw_score = float((txt.read_text(encoding='utf-8') or '').strip() or 0.0)
    if test_stdout:
        extra['test_output'] = test_stdout[-5000:]
    return raw_score, extra


def _read_collected_patch(trial_paths: Any) -> str:
    """Best-effort read of the collected model patch from host artifact staging.

    Only for CodeMTTaskOutput.patch_text (DB/logging); grading itself uses the
    artifacts the harbor ArtifactHandler already re-uploaded to the verifier env.
    """
    with contextlib.suppress(Exception):
        artifacts_dir = Path(trial_paths.artifacts_dir)
        if artifacts_dir.exists():
            for fp in sorted(artifacts_dir.rglob('*')):
                if fp.is_file() and fp.name.endswith(('.patch', '.diff')):
                    return fp.read_text(encoding='utf-8', errors='replace')
    return ''


async def _grade_turn(
    ctx: Any,
    task_input: CodeMTTaskInput,
    harbor_task: Any,
    trial_paths: Any,
    main_environment: GongfengEnvironment,
    span_id: str | None,
) -> tuple[float, dict[str, Any], str]:
    """Run harbor's Verifier for one turn and return (raw_score, eval_result, patch).

    shared:   grade in the agent's runtime (tests uploaded to /tests, test.sh ->
              reward.json). No artifacts.
    separate: collect [[verifier.collect]] artifacts from main (main stays alive
              for the next MT turn), acquire a FRESH 'verifier' runtime, upload
              artifacts, grade there, then tear the verifier runtime down.
    """
    from harbor.models.task.verifier_mode import (
        VerifierEnvironmentMode,
        resolve_effective_verifier_env_config,
    )
    from harbor.models.trial.paths import EnvironmentPaths

    # Run harbor's [[verifier.collect]] hooks FIRST — for BOTH shared and separate,
    # exactly like harbor_bridge (task.py: _run_collect_hooks before the mode branch).
    # SWE (separate) needs `git diff <base> > /tmp/model.patch` to exist before it is
    # collected; a shared task with collect hooks needs them run too. Dropping this is
    # what made every SWE (separate) get no_model_patch -> reward=0.
    await _run_collect_hooks(main_environment, harbor_task)

    if task_input.verifier_mode != VerifierEnvironmentMode.SEPARATE.value:
        # --- shared -------------------------------------------------------
        shared_paths = EnvironmentPaths.for_os(main_environment.os)
        reset_dirs = [shared_paths.verifier_dir]
        if not task_input.skip_tests_upload:
            reset_dirs.append(shared_paths.tests_dir)
        await main_environment.empty_dirs(reset_dirs, chmod=True)
        _, _, test_stdout = await _run_harbor_verifier(
            harbor_task=harbor_task,
            trial_paths=trial_paths,
            environment=main_environment,
            skip_tests_upload=task_input.skip_tests_upload,
            timeout_sec=task_input.test_timeout,
        )
        raw_score, eval_result = _read_reward_payload(trial_paths, test_stdout)
        return raw_score, eval_result, ''

    # --- separate ---------------------------------------------------------
    artifact_handler, main_paths = await _collect_artifacts(main_environment, harbor_task, trial_paths)
    patch_text = _read_collected_patch(trial_paths)

    verifier_runtime = await ctx.get_runtime(task_input.verifier_runtime_name)
    if span_id:
        vid = _extract_runtime_id(verifier_runtime)
        if vid:
            update_span(span_id, attributes={'hytuner.eval_runtime_id': vid})
    verifier_environment: GongfengEnvironment | None = None
    try:
        verifier_env_config = resolve_effective_verifier_env_config(harbor_task.config, step_cfg=None)
        if verifier_env_config is None:
            raise RuntimeError('Separate verifier did not resolve an effective environment')
        verifier_phase = harbor_task.config.verifier.explicit_phase_policy()
        verifier_baseline = verifier_env_config.resolve_baseline()
        # See the agent-phase note: only hand harbor a policy when this runtime was
        # actually provisioned isolated. Otherwise the stale source no-network policy
        # would fail Harbor validation against an open runtime.
        verifier_runtime_policy = (
            (verifier_phase or verifier_baseline) if task_input.verifier_network_isolation_enabled else None
        )
        verifier_environment = GongfengEnvironment(
            runtime=verifier_runtime,
            environment_dir=harbor_task.paths.tests_dir,
            environment_name=harbor_task.name,
            session_id=f'nexus_{task_input.instance_id or harbor_task.name}__verifier',
            trial_paths=trial_paths,
            task_env_config=verifier_env_config,
            default_timeout_sec=task_input.test_timeout,
            network_isolation_enabled=task_input.verifier_network_isolation_enabled,
            harbor_network_policy=verifier_runtime_policy,
            gpus_enabled=bool(verifier_env_config.gpus),
        )
        await verifier_environment.start(force_build=False)
        verifier_paths = EnvironmentPaths.for_os(verifier_environment.os)
        skip_tests_upload = (
            task_input.verifier_tests_prebaked
            or harbor_task.paths.discovered_test_path_for(verifier_environment.os) is None
        )
        reset_dirs = [verifier_paths.verifier_dir, verifier_paths.artifacts_dir]
        if not skip_tests_upload:
            reset_dirs.append(verifier_paths.tests_dir)
        await verifier_environment.empty_dirs(reset_dirs, chmod=True)
        await artifact_handler.upload_artifacts(
            verifier_environment,
            trial_paths.artifacts_dir,
            source_artifacts_dir=main_paths.artifacts_dir,
            target_artifacts_dir=verifier_paths.artifacts_dir,
        )
        _, _, test_stdout = await _run_harbor_verifier(
            harbor_task=harbor_task,
            trial_paths=trial_paths,
            environment=verifier_environment,
            skip_tests_upload=skip_tests_upload,
            timeout_sec=task_input.test_timeout,
        )
        raw_score, eval_result = _read_reward_payload(trial_paths, test_stdout)
        return raw_score, eval_result, patch_text
    finally:
        if verifier_environment is not None:
            with contextlib.suppress(Exception):
                await verifier_environment.stop(delete=True)
        with contextlib.suppress(Exception):
            await ctx.runtime_provider.release(name=task_input.verifier_runtime_name)
        ctx._runtimes.pop(task_input.verifier_runtime_name, None)


# ============================================================================
# Main task
# ============================================================================


@task_registry.register('code_mt_task')
async def code_mt_task(task_input: CodeMTTaskInput) -> CodeMTTaskOutput:
    """Unified multi-turn task for code agent training (harbor-native)."""
    import shlex

    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import VerifierEnvironmentMode, resolve_task_verifier_mode
    from harbor.models.trial.paths import TrialPaths

    ctx = require_context()
    span_id = get_current_span_id()

    # === Phase 0: Initialization ===
    if task_input.startup_jitter_ms > 0:
        await with_random_jitter(0, task_input.startup_jitter_ms)

    # Field-sinking: read every business/grading knob from {task_dir}/task.toml
    # HERE, so the generator (and hytuner loader) only pass a path + provisioning.
    # Precedence: any key already in the incoming task_config (index/generator
    # override) wins; task.toml fills the rest.
    if task_input.task_dir:
        from nexus.extensions.projects.code_mt.task.loader import apply_task_spec, load_task_spec

        apply_task_spec(task_input, ctx.task_config, load_task_spec(task_input.task_dir))

    # Load the harbor Task — the single source of truth for verifier topology,
    # collect hooks, artifacts, tests, instruction and verifier.env. Reconcile
    # verifier_mode with harbor's own resolution so the two never drift.
    task_dir_path = Path(task_input.task_dir)
    if not task_dir_path.exists():
        raise FileNotFoundError(f'Task directory not found: {task_dir_path}')
    harbor_task = Task(task_dir=task_dir_path)
    verifier_mode = resolve_task_verifier_mode(harbor_task.config)
    task_input.verifier_mode = verifier_mode.value

    dumped = task_input.model_dump(exclude={'run_agent'})
    task_config_dict = {**ctx.task_config, **dumped, **task_input.extra}
    ctx.task_config.update(task_config_dict)

    workdir = task_input.workdir or '/app'
    safe_name = harbor_task.name.replace('/', '_').replace('\\', '_')
    trial_dir = Path(tempfile.mkdtemp(prefix=f'code_mt_trial_{safe_name}_'))
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    main_environment: GongfengEnvironment | None = None
    score = 0.0
    agent_output = None
    eval_result: dict[str, Any] = {}
    exit_status = 'not_started'
    iterations = 0
    patch_text = ''
    score_details: dict[str, float] = {}
    attempt = 0
    mt_successful_turn = 0
    mt_successful_turn_iters = 0
    mt_failed_turn_iters: list[int] = []

    try:
        infer_runtime = await ctx.get_runtime(task_input.infer_runtime_name)
        runtime_id = _extract_runtime_id(infer_runtime)
        if span_id and runtime_id:
            update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})

        # Reset harbor's log/verifier/artifact staging + ensure workdir exists,
        # exactly as harbor_bridge does before boot.
        reset_dirs = ['/logs/agent', '/logs/verifier', '/logs/artifacts']
        if verifier_mode == VerifierEnvironmentMode.SEPARATE or not task_input.skip_tests_upload:
            reset_dirs.append('/tests')
        quoted = ' '.join(shlex.quote(p) for p in reset_dirs)
        await infer_runtime.run_command(
            command=f'rm -rf {quoted} && mkdir -p {shlex.quote(workdir)} {quoted}',
            timeout=60,
        )
        # Align to harbor_bridge: allow git metadata access when gongfeng's runtime
        # user differs from image ownership (else git ops hit "dubious ownership").
        await _configure_git_safe_directory(infer_runtime, workdir)

        # task_input reflects the static policy actually provisioned for this
        # runtime. Converters deliberately open networking for CLI agents
        # (hycode/codex/mini), even when the source task.toml is offline.
        # Do not pass the stale source no-network policy to Harbor validation.
        agent_runtime_policy = (
            (harbor_task.config.agent.explicit_phase_policy() or harbor_task.config.environment.resolve_baseline())
            if task_input.network_isolation_enabled
            else None
        )
        main_environment = GongfengEnvironment(
            runtime=infer_runtime,
            environment_dir=task_dir_path / 'environment',
            environment_name=harbor_task.name,
            session_id=f'nexus_{task_input.instance_id or harbor_task.name}__main',
            trial_paths=trial_paths,
            task_env_config=harbor_task.config.environment,
            default_timeout_sec=task_input.test_timeout,
            network_isolation_enabled=task_input.network_isolation_enabled,
            harbor_network_policy=agent_runtime_policy,
            gpus_enabled=bool(harbor_task.config.environment.gpus),
        )
        await main_environment.start(force_build=False)

        # Image boot ([environment.boot]) runs ONCE before the agent — gongfeng
        # bypasses the image ENTRYPOINT, so repo checkout / service start must be
        # launched here, then healthchecked. Reuses harbor_bridge verbatim.
        # _boot_image_services reads task_input via getattr (duck-typed on
        # boot_command/boot_env/boot_timeout), which CodeMTTaskInput provides.
        await _boot_image_services(infer_runtime, cast('Any', task_input))
        # Align to harbor_bridge: wait for [environment.healthcheck] readiness AFTER
        # boot, before the agent runs (else the agent starts before repo checkout /
        # MCP servers are up). harbor_bridge calls run_healthcheck() right after boot.
        # _run_boot_healthcheck widens harbor's 3-retry/5s default (~10s, far too
        # short for a real boot) to 10s x 90 unless the task set its own policy.
        await _run_boot_healthcheck(main_environment)
        # Legacy pre-0720 setup_commands (git-history truncation etc.) run after.
        if task_input.setup_commands:
            await _run_setup_commands(infer_runtime, task_input.setup_commands, task_config_dict)

        # Agent-visible skills/ only. tests/ is uploaded by harbor's Verifier at
        # grading time (post-agent) — the agent never sees it.
        await _upload_agent_visible_files(infer_runtime, task_input.task_dir)
        ctx.task_config['working_dir'] = workdir
        # Seed the SAME working_dir the agent's tools read (tools.py _get_working_dir
        # reads tool_state['working_dir'], NOT task_config). Without this, tasks whose
        # workdir is not the tool default (/testbed) — e.g. terminal /app — would run
        # every bash/editor/submit in the wrong dir. setdefault mirrors swe/task.py:467
        # so a tool that already pinned its cwd (e.g. after switch_runtime_image) wins.
        ctx.tool_state.setdefault('working_dir', workdir)

        # Load the dataset-owned MT retry prompt if not configured explicitly.
        if not task_input.mt_feedback_template and task_input.task_dir:
            fb = Path(task_input.task_dir) / 'mt_feedback.md'
            if fb.is_file():
                task_input.mt_feedback_template = fb.read_text(encoding='utf-8')

        if task_input.pre_agent_hook:
            await _run_hook(task_input.pre_agent_hook, ctx=ctx, runtime=infer_runtime, task_input=task_input)

        # === Phase 1: MT Loop ===
        # agent_timeout is the TOTAL agent budget across ALL MT turns. The agent's
        # own inference/observation segment timeouts enforce the active-time budget
        # (aligned with swe_agent). The outer runner.timeout (pause-aware via
        # hytuner's ctx.timeout_handle reschedule) provides the hard wall-clock cap.
        # The deadline below is informational (logging) only — we no longer wrap
        # run_agent in asyncio.wait_for, because that would count update_actor pause
        # against the agent's budget (see P7 of the Phase-1 plan).
        agent_deadline = time.monotonic() + task_input.agent_timeout
        # max_iterations is ALSO a TOTAL agent-step budget across ALL mt turns
        # (not per-turn): mt_max_turns is a retry count only and must not multiply
        # the iteration budget. The ceiling is the agent's own configured
        # max_iterations (read from agent_config, the same dict the runner merges
        # into run_agent) so there is ONE knob, no duplicate task field. 0/absent
        # => legacy per-turn behavior (task passes no override).
        try:
            # getattr keeps this cross-core-safe: an older ExecutionContext without
            # ``agent_config`` falls back to 0 => legacy per-turn behavior (as the
            # comment above documents), instead of raising AttributeError.
            _agent_config = getattr(ctx, 'agent_config', None) or {}
            total_max_iterations = int(_agent_config.get('max_iterations') or 0)
        except (TypeError, ValueError, AttributeError):
            total_max_iterations = 0
        for turn in range(task_input.mt_max_turns):
            attempt = turn + 1
            # Clear per-turn exit-control state so a retry turn's first tool call
            # doesn't immediately bail on the previous turn's submit signal.
            ctx.tool_state.pop('exit_status', None)
            ctx.tool_state.pop('should_exit', None)

            # Turn-0 message: the TASK owns all prompt composition (the agent is a
            # verbatim passthrough now). Source precedence:
            #   1. task_config['instruction'] — parquet override (SWE/SWEBuilder,
            #      whose prompt can't be expressed as plain harbor; convertor bakes
            #      the swe_agent-templated instruction here).
            #   2. {task_dir}/instruction.md — standard harbor: read the dataset's
            #      instruction file VERBATIM (no {placeholder} render).
            #   3. task_config['problem_statement'] — sunk from task.toml
            #      [other].description by loader.py. NOT in 0824 (which stops at
            #      task_input.message); retained here so a task-dir launch with a
            #      description but no instruction.md keeps its old prompt instead of
            #      silently falling through to an empty message.
            #   4. task_input.message — raw fallback.
            # This is SOURCE SELECTION + VERBATIM read, NOT injection. The ONLY
            # prompt injection anywhere in code_mt is the anti-hack notice below.
            # Retry turns render MT feedback instead.
            if turn == 0 and not task_input.run_oracle:
                message = (
                    ctx.task_config.get('instruction')
                    or _read_task_file(ctx.task_config.get('task_dir', ''), 'instruction.md')
                    or ctx.task_config.get('problem_statement')
                    or task_input.message
                )
                # === ONLY prompt injection in code_mt: anti-hack (task-owned) ===
                # PREVENTION half of enable_anti_hack. Without it the detector
                # penalizes a constraint the agent was never told about — the
                # unfairness 0824 calls out explicitly.
                if task_input.enable_anti_hack_prompt:
                    from nexus.extensions.projects.code_mt.task.anti_hack_prompt import (
                        append_anti_hack_notice,
                    )

                    message = append_anti_hack_notice(message)
            elif turn == 0 and task_input.run_oracle:
                message = ''
            else:
                message = _render_mt_feedback(task_input, eval_result, score, attempt)

            loglib.banner(
                logger,
                'blue',
                f'TURN {attempt}/{task_input.mt_max_turns}  {"ORACLE" if task_input.run_oracle else "AGENT"}',
            )
            # Time left in the TOTAL agent budget for this turn (oracle runs are
            # not agent-timed, so they ignore this).
            turn_agent_timeout = agent_deadline - time.monotonic()
            if not task_input.run_oracle and turn_agent_timeout <= 0:
                exit_status = 'timeout'
                loglib.phase_fail(
                    logger,
                    'AGENT',
                    f'total agent budget ({task_input.agent_timeout:.0f}s) exhausted before turn {attempt}',
                )
                mt_failed_turn_iters.append(0)
                break

            # Each turn gets only the iterations remaining in the TOTAL budget;
            # once the total is spent we stop launching new turns (-> MAX_STEPS),
            # exactly like the agent_deadline above. 0 keeps legacy per-turn
            # behavior (turn_max_iterations stays 0 => the task passes no override,
            # so the agent's own configured max_iterations applies each turn).
            turn_max_iterations = 0
            if total_max_iterations > 0:
                turn_max_iterations = total_max_iterations - iterations
                if not task_input.run_oracle and turn_max_iterations <= 0:
                    exit_status = 'max_iterations'
                    loglib.phase_fail(
                        logger,
                        'AGENT',
                        f'total iteration budget ({total_max_iterations}) exhausted before turn {attempt}',
                    )
                    mt_failed_turn_iters.append(0)
                    break

            loglib.phase_start(
                logger,
                'AGENT',
                f'Running agent (timeout={turn_agent_timeout:.0f}s remaining of '
                f'{task_input.agent_timeout:.0f}s total, mode={task_input.agent_timeout_mode})',
            )

            observation_start = time.monotonic()
            turn_iterations = 0
            try:
                if task_input.run_oracle:
                    agent_output = None
                    exit_status = await _run_oracle_solution(infer_runtime, task_input)
                else:
                    _agent_kwargs: dict[str, Any] = {'user_prompt': message, 'working_dir': workdir}
                    # system_prompt: task reads it (parquet override or
                    # {task_dir}/system_prompt.md, VERBATIM) and passes it through.
                    # The agent no longer reads files, so this is the ONLY path that
                    # keeps SWE/SWEBuilder's system_prompt (e.g. "You are an expert
                    # DevOps engineer...") in the loop. Only sent on turn 0; MT retry
                    # turns resume the stored conversation (system already present).
                    if turn == 0:
                        sp = ctx.task_config.get('system_prompt') or _read_task_file(
                            ctx.task_config.get('task_dir', ''), 'system_prompt.md'
                        )
                        if sp:
                            _agent_kwargs['system_prompt'] = sp
                        # setup_strategy is an AGENT-input field (turn-0 env hook,
                        # e.g. SWEBuilder's builder_pre_agent). The dataset declares
                        # it in [other]; t3_source carries it into task_config. The
                        # task forwards it to the agent here (only turn 0) so the
                        # hook fires under BOTH harbor_bridge_task and code_mt_task.
                        _ss = ctx.task_config.get('setup_strategy')
                        if _ss:
                            _agent_kwargs['setup_strategy'] = _ss
                    if turn_max_iterations > 0:
                        # Cap this turn to the remaining total budget; task_kwargs
                        # override the agent's configured max_iterations in the
                        # runner's run_agent merge.
                        _agent_kwargs['max_iterations'] = turn_max_iterations
                    agent_output = await task_input.run_agent(**_agent_kwargs)
                    exit_status = getattr(agent_output, 'exit_status', None)
                    if exit_status is None:
                        exit_status = 'finished' if getattr(agent_output, 'success', True) else 'unknown'
                    turn_iterations = getattr(agent_output, 'iterations', 0)
                    iterations += turn_iterations
                elapsed = time.monotonic() - observation_start
                if exit_status in ('submitted', 'finished'):
                    loglib.phase_ok(
                        logger, 'AGENT', f'exit_status={exit_status}, iterations={iterations} ({elapsed:.1f}s)'
                    )
                else:
                    loglib.phase_warn(
                        logger, 'AGENT', f'exit_status={exit_status}, iterations={iterations} ({elapsed:.1f}s)'
                    )
            except asyncio.TimeoutError:
                # Runner-level timeout (pause-aware via hytuner's ctx.timeout_handle
                # reschedule). The agent's own active-time budget exhaustion returns
                # exit_status='timeout' normally (not an exception).
                exit_status = 'timeout'
                loglib.phase_warn(
                    logger,
                    'AGENT',
                    f'runner-level timeout after {turn_agent_timeout:.0f}s remaining budget '
                    f'(total {task_input.agent_timeout:.0f}s, mode={task_input.agent_timeout_mode})',
                )
                if task_input.agent_timeout_mode == 'hard':
                    score = 0.0
                    mt_failed_turn_iters.append(turn_iterations)
                    break
                logger.warning(
                    'Runner-level timeout (total agent budget %.0fs), continuing to eval',
                    task_input.agent_timeout,
                )

            if span_id:
                update_span(span_id, attributes={'hytuner.observation_time': time.monotonic() - observation_start})

            # Re-fetch the infer runtime handle after the agent loop — if the
            # agent called switch_runtime_image, the old pod was replaced.
            infer_runtime = await ctx.get_runtime(task_input.infer_runtime_name)
            main_environment._runtime = infer_runtime  # keep the harbor env bound to the live pod

            if task_input.post_agent_hook:
                await _run_hook(
                    task_input.post_agent_hook,
                    ctx=ctx,
                    runtime=infer_runtime,
                    task_input=task_input,
                    agent_output=agent_output,
                )

            # Code MT owns this repo-QA compatibility output. Generic Harbor Bridge
            # exposes only T3AgentResult and does not write benchmark-private files.
            await _persist_code_mt_deliverable(infer_runtime, ctx)

            # === Eval ===
            loglib.banner(logger, 'magenta', f'TURN {attempt}  VERIFIER + SCORING')
            loglib.phase_start(
                logger,
                'VERIFIER',
                f'Running {task_input.verifier_mode} harbor verifier (timeout={task_input.test_timeout:.0f}s)',
            )
            v_t0 = time.monotonic()
            raw_score, eval_result, turn_patch = await _grade_turn(
                ctx,
                task_input,
                harbor_task,
                trial_paths,
                main_environment,
                span_id,
            )
            if turn_patch:
                patch_text = turn_patch
            eval_result['attempt'] = attempt
            v_elapsed = time.monotonic() - v_t0
            (loglib.phase_ok if raw_score >= 1.0 else loglib.phase_fail)(
                logger,
                'VERIFIER',
                f'raw_score={raw_score:.2f} ({v_elapsed:.1f}s)',
            )
            if span_id:
                update_span(span_id, attributes={'hytuner.reward_time': time.monotonic() - observation_start})

            score, score_details = await compute_score(
                raw_score=raw_score,
                eval_result=eval_result,
                task_input=task_input,
                ctx=ctx,
            )
            eval_result['merged_score'] = score

            details_str = ', '.join(f'{k}={v:.2f}' for k, v in score_details.items())
            final_marker = loglib.OK if score >= 1.0 else loglib.FAIL
            logger.info(
                '%s %s %s → final=%s (merge=%s)',
                loglib._bold('magenta', '[SCORE]'),
                final_marker,
                details_str or '(no dimensions)',
                loglib._bold('green' if score >= 1.0 else 'red', f'{score:.2f}'),
                task_input.score_merge,
            )

            # Per-turn eval cleanup (builder /testbed reset + /tmp/*.patch rewrite,
            # or build_cc image publish + DinD wipe). Runs on EVERY turn — including
            # the final/successful one — because build_cc's finalize uploads on
            # success. Best-effort: it must never change the score or abort the loop
            # (the strategies are internally guarded too), so suppress here as well.
            if task_input.eval_cleanup_strategy:
                with contextlib.suppress(Exception):
                    await _run_hook(
                        task_input.eval_cleanup_strategy,
                        ctx=ctx,
                        runtime=infer_runtime,
                        task_input=task_input,
                        eval_result=eval_result,
                    )

            if score >= 1.0:
                if mt_successful_turn == 0:
                    mt_successful_turn = attempt
                mt_successful_turn_iters = turn_iterations
                break
            mt_failed_turn_iters.append(turn_iterations)
            # Oracle is single-shot — a <1.0 oracle score means the row is broken.
            if task_input.run_oracle:
                break
            if turn == task_input.mt_max_turns - 1:
                break
    finally:
        # === Phase 2: Teardown ===
        if main_environment is not None:
            with contextlib.suppress(Exception):
                await main_environment.stop(delete=True)
        with contextlib.suppress(Exception):
            if ctx.runtime_provider is not None:
                await ctx.runtime_provider.release(name=task_input.infer_runtime_name)
        shutil.rmtree(trial_dir, ignore_errors=True)

    termination_reason = _canonical_termination_reason(exit_status)

    # === Reward-hacking / baseline monitoring (fail-open; default no-op) ===
    # Runs for BOTH the tool_call and text_parse loops — the trajectory is read
    # from ctx.tool_state via extract_agent_trajectory. Default config = monitors
    # ON (flag-only), all penalties 0, anti_hack OFF, floor 0.0, so this block
    # only writes mirror metadata into extra_fields and leaves score +
    # termination_reason untouched until enabled via the parquet task_config.
    reward_hack_extra: dict[str, Any] = {}
    rh_config = None
    base_score = score
    try:
        rh_config = resolve_reward_hacking_config(task_input)
        rh_answer, _, rh_messages = extract_agent_trajectory(ctx)
        rh_repo = ''
        if isinstance(task_input.extra, dict):
            rh_repo = str(task_input.extra.get('repo', '') or '')
        reward_hack_extra = run_code_mt_monitors(
            rh_messages,
            config=rh_config,
            eval_result=eval_result,
            patch_text=patch_text,
            instance_id=task_input.instance_id,
            repo=rh_repo,
            score=score,
            answer=rh_answer,
            task_type=str(getattr(task_input, 'task_type', '') or ''),
            base_commit=str(ctx.task_config.get('base_commit') or ''),
            history_truncated=ctx.task_config.get('history_truncated'),
        )
        # --- Full-blood anti-hack: TASK-side post-processing, gated by task_config.
        #     Runs BEFORE penalties so the CODE_REWARD_HACKING flip + hard-zero are
        #     visible to _normalize_hack_observe_always. Re-scans the completed
        #     trajectory (truncation-exempt) with the verbatim detector. Feeding
        #     reward_hack_extra['anti_hack_violations'] is the ENTIRE mask wiring:
        #     apply_code_mt_penalties -> _normalize_hack_observe_always flips
        #     termination to CODE_REWARD_HACKING, which the run script's
        #     additional_terminate_reason + maskout_terminate_sample drops from loss.
        # Validation split: SWE withholds only the CONSEQUENCE on eval samples
        # (`hack_reward_override=None if is_validation`), never the detection.
        # CM-1's consequence is `score = 0.0` below, which would zero a correct
        # eval solution outright, so the whole CM-1 pass is skipped here. CM-2
        # (monitors.py) keeps detecting and still reports `is_hack`, so eval hack
        # rate stays observable — see penalties.py `suppress_flip`.
        if getattr(task_input, 'enable_anti_hack', False) and not getattr(task_input, 'is_validation', False):
            try:
                wdir = ctx.task_config.get('working_dir') or task_input.workdir
                anti_hack_violations, anti_hack_summary = scan_trajectory_for_anti_hack(
                    rh_messages,
                    repo=rh_repo,
                    working_dir=wdir,
                    instance_id=task_input.instance_id,
                    # Sunk from {task_dir}/task.toml [other] by loader.py. Present
                    # for SWE/SWEBuilder + Setup, absent for Cyber / Terminal /
                    # RepoQA / StackOverflow / NL2repo — where L4 self-disables.
                    base_commit=ctx.task_config.get('base_commit'),
                    history_truncated=ctx.task_config.get('history_truncated'),
                )
                # Observability counters (git_commit_probe / external-source /
                # git_log) — always mirrored, independent of whether a violation
                # fired, so the swanlab metrics have the correct denominator.
                if isinstance(anti_hack_summary, dict) and anti_hack_summary:
                    reward_hack_extra.update(anti_hack_summary)
                if anti_hack_violations:
                    reward_hack_extra['anti_hack_violations'] = anti_hack_violations
                    score = 0.0  # strictest control: any source-download hack -> 0
                    # Flip termination here too so the mask does NOT depend on the
                    # 'anti_hack' monitor being in the parquet whitelist — under
                    # enable_anti_hack the gate is authoritative. _normalize_hack_
                    # observe_always (if monitor_on) re-affirms the same value.
                    from nexus.extensions.projects.code_mt.task.reward_hacking.penalties import (
                        CODE_REWARD_HACKING,
                    )

                    termination_reason = CODE_REWARD_HACKING
                    logger.warning(
                        'code_mt anti-hack detected (%d): %s',
                        len(anti_hack_violations),
                        anti_hack_violations[:3],
                    )
            except Exception:
                logger.exception('code_mt task-side anti-hack scan failed; score left as-is')

        # Capture the pre-penalty reward so the log-only report can show the
        # base_score -> final_score deduction.
        base_score = score
        score, termination_reason = apply_code_mt_penalties(
            score,
            reward_hack_extra,
            termination_reason,
            config=rh_config,
        )
    except Exception:
        logger.exception('reward-hacking monitor/penalty failed; score/term left as-is')

    # apply_code_mt_penalties may hand back None; the output field is a plain str.
    termination_reason = termination_reason or ''

    if span_id:
        update_span(span_id, attributes={'hytuner.termination_reason': termination_reason})

    # NOTE: the binary 0/1 verdict is now applied PRE-PENALTY inside
    # scoring.compute_score (on the merged rule+llm base). Re-snapping here would
    # be POST-PENALTY and erase the penalty deduction (turning 0.98 back into 0),
    # so `score` is passed through unchanged — penalties already produced the
    # final continuous reward on top of the 0/1 base.
    final_score = score if score is not None else 0.0

    # Log-only monitor report: mirror the already-computed deductions (no scoring
    # logic here) so the DB explains the final reward from task_config alone.
    monitor_report: dict[str, Any] = {}
    if rh_config is not None:
        try:
            monitor_report = build_monitor_report(
                rh_config,
                reward_hack_extra,
                base_score=base_score,
                final_score=final_score,
                termination_reason=termination_reason,
            )
        except Exception:
            logger.exception('build_monitor_report failed; monitor left empty')
    mt_turns_executed = attempt
    loglib.banner(logger, 'green' if final_score >= 1.0 else 'red', 'RESULT')
    (loglib.phase_ok if final_score >= 1.0 else loglib.phase_fail)(
        logger,
        'RESULT',
        f'instance={task_input.instance_id} score={final_score:.2f} '
        f'turns={mt_turns_executed}/{task_input.mt_max_turns} '
        f'success_turn={mt_successful_turn} '
        f'success_iters={mt_successful_turn_iters} failed_iters={mt_failed_turn_iters} '
        f'exit={exit_status} term={termination_reason} iters={iterations}',
    )
    return CodeMTTaskOutput(
        instance_id=task_input.instance_id,
        task_type=task_input.task_type,
        failure_type=_classify_failure(eval_result, final_score),
        score=final_score,
        exit_status=exit_status,
        iterations=iterations,
        eval_result=eval_result,
        score_details=score_details,
        patch_text=patch_text,
        termination_reason=termination_reason,
        mt_max_turns=task_input.mt_max_turns,
        mt_turns_executed=mt_turns_executed,
        mt_successful_turn=mt_successful_turn,
        mt_successful_turn_iters=mt_successful_turn_iters,
        mt_failed_turn_iters=mt_failed_turn_iters,
        extra_fields=reward_hack_extra,
        monitor=monitor_report,
    )
