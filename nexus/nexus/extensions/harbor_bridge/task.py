"""Harbor Bridge Task: Nexus task that uses harbor's Verifier for evaluation.

This task is agent-agnostic — it works with ANY nexus agent (terminus2, claude_code,
codex, swe_agent, etc.). The agent is configured via RunInput.agent_name, NOT here.

The task's responsibility is:
1. Initialize the runtime environment
2. Delegate to the configured agent (via run_agent)
3. Run harbor's Verifier to grade the result (upload tests, execute test.sh, collect reward)

This replaces the old approach of bundling a custom agent inside the bridge.
Use existing nexus agents that already support claude_code, swe_agent, terminus2, etc.
"""

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import posixpath
import re
import shlex
import shutil
import tempfile
import time
import tomllib

from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.dataset.task_files import TaskFiles
from nexus.extensions.agents.installed_common.execution import ComposeExecutionTarget, agent_execution_target
from nexus.extensions.harbor_bridge.agent_result import (
    T3AgentResult,
    agent_termination_reason,
    build_t3_agent_result,
    extract_messages,
    extract_reported_messages,
    upload_t3_agent_result,
    verifier_env_with_t3_agent_result,
    write_t3_agent_result,
)
from nexus.extensions.harbor_bridge.agent_trajectory import (
    HarborAgentTrajectoryTracker,
    build_t3_agent_trajectory,
    chat_tracker_fallback_record,
    upload_t3_agent_trajectory,
    verifier_env_with_t3_agent_trajectory,
    write_t3_agent_trajectory,
)
from nexus.extensions.harbor_bridge.anti_hack.strategies.anthropic_blocks import (
    normalize_anthropic_block_messages,
)
from nexus.extensions.harbor_bridge.bad_pattern_operator.diagnostics import (
    FINAL_TURN_LANGUAGE_CH_RULE,
    final_turn_language_ch_diagnostic,
)
from nexus.extensions.harbor_bridge.bad_pattern_operator.policy import ToolCallBadPatternPolicy
from nexus.extensions.harbor_bridge.bad_pattern_operator.schemas import merge_tool_schemas, normalize_tool_schema
from nexus.extensions.harbor_bridge.browser import (
    require_browser_session_opened,
    verifier_env_with_browser_session,
)
from nexus.extensions.harbor_bridge.compiler import load_harbor_task, resolve_env_mapping
from nexus.extensions.harbor_bridge.compose import (
    ComposePlan,
    compose_config_from_task,
    topology_spec_from_plan,
)
from nexus.extensions.harbor_bridge.upload import raise_for_upload_failures
from nexus.runtime.topology import ComposeTopology
from nexus.trace import get_current_span_id, get_processor, update_span
from nexus.trace.phases import phase
from nexus.utils.provider_io import redact_provider_io


logger = logging.getLogger(__name__)
_ENV_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_AGENT_WATCHDOG_GRACE_SECONDS = 300.0
_AGENT_WATCHDOG_PAUSE_POLL_SECONDS = 1.0
_VERIFIER_DIAGNOSTIC_CHARS = 2000
_VERIFIER_TEST_OUTPUT_TAIL_CHARS = 2000
_VERIFIER_TEST_STDOUT_BYTES = 10000


class AntiHackPolicy(BaseModel):
    """Versioned anti-hack policy.

    ``delivery='none'`` disables only prompt delivery.  A configured
    ``trajectory_strategy`` still runs after the Agent and provides
    detection-only anti-hack.
    """

    model_config = ConfigDict(extra='forbid')

    id: str
    # ``none`` means detection-only, not "disable anti-hack".
    delivery: Literal['instruction_prefix', 'instruction_suffix', 'none'] = 'instruction_suffix'
    prompt: str = ''
    trajectory_strategy: str | None = None

    @field_validator('id')
    @classmethod
    def _require_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError('must be a non-empty string')
        return normalized

    @field_validator('trajectory_strategy')
    @classmethod
    def _normalize_optional_strategy(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError('must be a non-empty string when provided')
        return normalized


class HarborBridgeTaskInput(TaskRunInput, WithAgentRunner):
    """Input for harbor bridge task.

    This is intentionally minimal — agent-specific config goes in agent_config,
    not here. The task only handles harbor-specific concerns (verifier, task loading).
    """

    # Harbor task identification
    task_dir: str | None = None  # Legacy live path; Dataset runs use ExecutionContext.task_files.
    instance_id: str = ''  # Task instance ID for tracking

    # Task configuration (passed to agent as user_prompt)
    instruction: str = ''  # Override instruction (if empty, read from task dir)
    workdir: str  # Explicit absolute working directory compiled from task.toml
    # Normalized at Dataset compile time so Harnesses never re-read live task.toml.
    mcp_servers: list[dict[str, Any]] = []
    # Optional [environment.browser] declaration. A browser-capable Agent creates
    # the live session; shared Verifier tests receive its session ID.
    browser: dict[str, Any] | None = None
    # Frozen native Compose contract. Agent mode supplies a trusted service target
    # only to Agents that explicitly implement the socket-v1 Compose transport.
    compose_plan: ComposePlan | None = None

    # Execution parameters
    agent_timeout: float = 600.0  # Agent execution timeout (seconds)
    test_timeout: float = 600.0  # End-to-end verifier timeout (seconds)

    # Harbor-specific
    mode: Literal['agent', 'oracle', 'nop'] = 'agent'
    verifier_mode: Literal['shared', 'separate'] = 'shared'
    skip_tests_upload: bool = False  # Skip uploading tests (pre-baked in image)
    # Materialize T3AgentTrajectory for this task. Off unless [verifier] says otherwise,
    # so a task family that grades deliverables sees no new file and no new env var.
    needs_agent_trajectory: bool = False
    skip_environment_upload: bool = True  # Skip default environment/workspace overlay onto workdir
    environment_uploads: list[dict[str, str]] = []  # Items from [environment.upload] to copy into the sandbox
    network_isolation_enabled: bool = False  # True only when runtime network_policy is active
    verifier_network_isolation_enabled: bool = False
    # Optional, reusable policy selected by a Compose Policy Profile. The prompt
    # may provide guidance; delivery='none' intentionally leaves the instruction
    # unchanged while a named trajectory_strategy still performs Host-side
    # detection and reports facts without altering Harbor rewards.
    anti_hack_policy: AntiHackPolicy | None = None
    # Optional Host-side bad-pattern checks. Only explicitly enabled rules run;
    # fixed penalties affect the training-facing score.
    tool_call_bad_pattern_policy: ToolCallBadPatternPolicy | None = None

    # Image boot hook: gongfeng replaces the image ENTRYPOINT with nexus-runtime, so
    # any in-image init (e.g. Canyon's /omnibus/start_all.sh that brings up the MCP
    # HTTP servers) never runs on its own. When boot_command is set we launch it in
    # the background (the script may block forever) and inject boot_env (e.g.
    # ENABLED_APPS from task.toml [environment.env]). Harbor's native
    # BaseEnvironment.run_healthcheck() handles readiness before the agent runs.
    boot_command: str = ''  # e.g. "/omnibus/start_all.sh"
    boot_env: dict[str, str] = {}  # env for the boot process (ENABLED_APPS, DOJO_FAKE_NOW, ...)

    # Separate Verifier images use the same optional boot contract because
    # Gongfeng also replaces their ENTRYPOINT with nexus-runtime.
    verifier_boot_command: str = ''
    verifier_boot_env: dict[str, str] = {}


class HarborBridgeTaskOutput(TaskRunOutput, WithScore):
    """Output from harbor bridge task."""

    reward: float = 0.0
    rewards: dict[str, float] = {}
    reward_metadata: dict[str, Any] = {}  # non-numeric reward.json keys (e.g. check accumulators)
    test_stdout: str = ''
    # Optional task-owned metadata forwarded by nexus.runner.hytuner to training consumers.
    extra_fields: dict[str, Any] = Field(default_factory=dict)


class AgentInfrastructureError(RuntimeError):
    """The selected Agent could not make a valid task attempt."""


def _update_task_span(span_id: str | None, attributes: dict[str, Any]) -> None:
    """Update optional training metadata without making tracing a task dependency."""
    if not span_id or not attributes:
        return
    try:
        update_span(span_id, attributes=dict(attributes))
    except Exception:
        logger.warning('Failed to update Harbor task span', exc_info=True)


def _context_runtime_id(ctx: Any, name: str) -> str | None:
    """Read a provider-native Runtime ID when the embedding Context exposes it."""
    get_runtime_id = getattr(ctx, 'get_runtime_id', None)
    if not callable(get_runtime_id):
        return None
    try:
        runtime_id = get_runtime_id(name)
    except Exception:
        logger.warning('Failed to read Harbor Runtime ID for %s', name, exc_info=True)
        return None
    return str(runtime_id) if runtime_id else None


def _agent_watchdog_timeout(agent_timeout: float) -> float:
    """Allow runtime setup and process cleanup outside the Agent's own budget."""
    return float(agent_timeout) + _AGENT_WATCHDOG_GRACE_SECONDS


async def _run_agent_with_watchdog(agent_run: Awaitable[Any], watchdog_timeout: float, task_timeout_handle: Any) -> Any:
    """Run the Agent under a watchdog that shifts with the runner task deadline.

    The runner deadline is paused while rollout is aborted for training and is
    extended by LLM time that is not inference. Keeping the watchdog at a fixed
    offset from that deadline gives both the same accounting, so training pauses
    no longer count against the Agent budget. Waking up never cancels the Agent
    by itself: the runner deadline is read again first, so a pause or extension
    that landed while sleeping is always honored.

    Raises:
        TimeoutError: The Agent did not finish within its active watchdog budget.
    """
    task_deadline = None if task_timeout_handle is None else task_timeout_handle.when()
    if task_deadline is None:
        # Without a runner deadline nothing records pauses, so keep a wall-clock watchdog.
        return await asyncio.wait_for(agent_run, timeout=watchdog_timeout)

    loop = asyncio.get_running_loop()
    deadline_offset = task_deadline - (loop.time() + watchdog_timeout)
    agent_task = asyncio.ensure_future(agent_run)
    try:
        while True:
            task_deadline = task_timeout_handle.when()
            if task_deadline is None:
                # The abort gate paused the runner deadline and signals no resume, so poll until it is re-armed.
                wait_seconds = _AGENT_WATCHDOG_PAUSE_POLL_SECONDS
            else:
                wait_seconds = max(0.0, task_deadline - deadline_offset - loop.time())
            done, _ = await asyncio.wait({agent_task}, timeout=wait_seconds)
            if done:
                return agent_task.result()
            task_deadline = task_timeout_handle.when()
            if task_deadline is not None and task_deadline - deadline_offset <= loop.time():
                break
        agent_task.cancel()
        await asyncio.wait({agent_task})
        if not agent_task.cancelled() and agent_task.exception() is not None:
            logger.warning('Agent raised while being cancelled by the watchdog', exc_info=agent_task.exception())
        raise TimeoutError(f'Agent did not finish within {watchdog_timeout:g}s of active time')
    except BaseException:
        # Runner cancellation reaches here while the Agent still runs; stop it before propagating.
        if not agent_task.done():
            agent_task.cancel()
            await asyncio.wait({agent_task})
        raise


def _build_agent_instruction(instruction: str, *, anti_hack_policy: AntiHackPolicy | None) -> str:
    """Apply an optional anti-hack policy without changing Oracle instructions."""
    if anti_hack_policy is None or anti_hack_policy.delivery == 'none':
        return instruction
    if anti_hack_policy.delivery == 'instruction_prefix':
        return f'{anti_hack_policy.prompt}\n\n{instruction.lstrip()}'
    return f'{instruction.rstrip()}\n\n{anti_hack_policy.prompt}'


def _mcp_tool_schemas(ctx: Any) -> list[dict[str, Any]]:
    """Serialize Host-loaded MCP tools for schema-aware bad-pattern rules."""
    schemas: list[dict[str, Any]] = []
    for fallback_name, tool in getattr(ctx, 'mcp_tools', {}).items():
        schema = normalize_tool_schema({
            'name': str(getattr(tool, 'full_name', None) or fallback_name),
            'description': getattr(tool, 'description', None),
            'input_schema': getattr(tool, 'input_schema', None),
        })
        if schema is not None:
            schemas.append(schema)
    return schemas


def _json_safe_metadata(value: Any) -> Any:
    """Normalize TOML metadata to the JSON-only strategy boundary."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list | tuple):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    return str(value)


def _anti_hack_task_metadata(harbor_task: Any, task_input: HarborBridgeTaskInput) -> dict[str, Any]:
    """Build the small, secret-free task metadata exposed to strategies."""
    config = getattr(harbor_task, 'config', None)
    if isinstance(config, dict):
        metadata = config.get('metadata')
        other = config.get('other')
    else:
        metadata = getattr(config, 'metadata', None)
        other = getattr(config, 'other', None)
    if not isinstance(other, dict):
        config_path = getattr(getattr(harbor_task, 'paths', None), 'config_path', None)
        if config_path is not None:
            raw_config = tomllib.loads(Path(config_path).read_text())
            other = raw_config.get('other')
    harbor_metadata = _json_safe_metadata(metadata) if isinstance(metadata, dict) else {}
    harbor_other = _json_safe_metadata(other) if isinstance(other, dict) else {}
    return {
        'instance_id': task_input.instance_id,
        'task_name': str(getattr(harbor_task, 'name', '') or ''),
        'workdir': task_input.workdir,
        'harbor': harbor_metadata,
        'other': harbor_other,
    }


async def _chat_tracker_fallback(
    runtime: Any,
    *,
    final_answer: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Snapshot one unambiguous dedicated ChatTracker conversation and its tools."""
    drain_trace = getattr(runtime, 'drain_trace', None)
    if callable(drain_trace):
        try:
            await drain_trace(timeout=5.0)
        except Exception as exc:
            logger.warning('Failed to drain Runtime trace before Agent trajectory snapshot: %s', exc)

    processor = get_processor(HarborAgentTrajectoryTracker)
    if processor is None:
        logger.warning('Agent trajectory requested but the dedicated Harbor ChatTracker is not registered')
        return [], []

    snapshot = copy.deepcopy(processor.snapshot_result() or {})
    messages, tools, candidate_count = chat_tracker_fallback_record(snapshot, final_answer=final_answer)
    if candidate_count > 1:
        if messages:
            logger.debug('Selected one authoritative ChatTracker trajectory from %d branches', candidate_count)
        else:
            logger.warning(
                'Agent trajectory ChatTracker fallback found %d candidate branches without a unique main role; '
                'refusing to guess the authoritative branch',
                candidate_count,
            )
    return messages, tools


def _split_reward_payload(payload: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    """Split a reward.json payload into (numeric_rewards, extra_metadata).

    harbor's VerifierResult types ``rewards`` as ``dict[str, float | int]``, so a rich
    reward.json like ``{"reward": 0.5, "metadata": {...}}`` (the shape Canyon/Chakra
    reward functions emit) fails validation on the ``metadata`` dict. We keep only the
    numeric top-level keys for harbor and preserve everything else as metadata.
    """
    numeric: dict[str, float] = {}
    extra: dict[str, Any] = {}
    for key, value in (payload or {}).items():
        if isinstance(value, bool):  # bools are ints in Python; treat as metadata, not score
            extra[key] = value
        elif isinstance(value, (int, float)):
            numeric[key] = value
        else:
            extra[key] = value
    return numeric, extra


def _non_numeric_reward_entries(exc: ValidationError) -> set[str]:
    """Name non-numeric ``rewards`` entries, or return nothing for structural failures.

    Individual entries are the expected shape for a reward.json carrying extras such as
    ``answer_found`` (bool) or ``judge_reason`` (str) next to the score, and
    :func:`_split_reward_payload` recovers from them without losing anything. Any error
    outside ``rewards`` is a real VerifierResult problem, so an empty result means the
    caller must let the exception stand.
    """
    entries: set[str] = set()
    for error in exc.errors():
        loc = error.get('loc', ())
        # len > 1 keeps this to individual entries: a bare ('rewards',) means the field
        # itself is the wrong shape, which is structural and must stay a warning.
        if len(loc) <= 1 or loc[0] != 'rewards':
            return set()
        entries.add(str(loc[1]))
    return entries


def _read_verifier_output_tail(trial_paths: Any) -> str | None:
    """Return the useful end of Harbor's merged verifier stdout/stderr capture."""
    path = getattr(trial_paths, 'test_stdout_path', None)
    if path is None or not path.exists():
        return None
    try:
        output = path.read_text(encoding='utf-8', errors='replace').strip()
    except OSError as exc:
        return f'<failed to read verifier output: {type(exc).__name__}: {exc}>'
    if not output:
        return None
    if len(output) <= _VERIFIER_TEST_OUTPUT_TAIL_CHARS:
        return output
    return f'<tail of {len(output)} chars>…{output[-_VERIFIER_TEST_OUTPUT_TAIL_CHARS:]}'


def _read_optional_verifier_stdout(path: Path, *, max_bytes: int = _VERIFIER_TEST_STDOUT_BYTES) -> str:
    """Read a bounded verifier log preview without making reward delivery depend on it."""
    try:
        with path.open('rb') as stream:
            return stream.read(max_bytes).decode('utf-8', errors='replace')
    except OSError:
        logger.warning('Unable to read optional verifier stdout: %s', path, exc_info=True)
        return ''


async def _capture_remote_verifier_output_tail(environment: Any, trial_paths: Any) -> None:
    """Best-effort capture of a bounded remote verifier diagnostic after failure."""
    from harbor.models.trial.paths import EnvironmentPaths

    download_tail = getattr(environment, 'download_file_tail', None)
    if not callable(download_tail):
        return

    environment_paths = EnvironmentPaths.for_os(environment.os)
    remote_stdout = environment_paths.verifier_dir / trial_paths.test_stdout_path.name
    try:
        await download_tail(
            str(remote_stdout),
            trial_paths.test_stdout_path,
            max_bytes=_VERIFIER_TEST_STDOUT_BYTES,
        )
    except Exception:
        logger.warning('Unable to capture remote verifier output tail: %s', remote_stdout, exc_info=True)


def _verifier_failure_detail(
    trial_paths: Any,
    *,
    reward_metadata: dict[str, Any] | None = None,
    sensitive_source: Any = None,
) -> str:
    """Build a bounded, credential-redacted diagnostic before the trial directory is deleted."""
    detail: dict[str, Any] = {}
    if reward_metadata:
        detail['reward_metadata'] = reward_metadata
    test_output_tail = _read_verifier_output_tail(trial_paths)
    if test_output_tail:
        detail['test_output_tail'] = test_output_tail
    if not detail:
        return 'no verifier diagnostic output was captured'

    redacted = redact_provider_io(detail, sensitive_source=sensitive_source)
    serialized = json.dumps(redacted, ensure_ascii=False, default=str)
    if len(serialized) <= _VERIFIER_DIAGNOSTIC_CHARS:
        return serialized
    return f'{serialized[:_VERIFIER_DIAGNOSTIC_CHARS]}… <truncated; {len(serialized)} diagnostic chars total>'


def _validate_reward_contract(
    rewards: dict[str, float | int],
    *,
    raw_payload: dict[str, Any] | None = None,
) -> None:
    """Require one finite primary reward in ``{-1} ∪ [0, 1]``."""
    if 'reward' not in rewards:
        raise ValueError("Verifier reward payload must contain a top-level numeric 'reward'")

    raw_reward = raw_payload.get('reward') if raw_payload is not None else rewards['reward']
    if isinstance(raw_reward, bool) or not isinstance(raw_reward, (int, float)):
        raise ValueError(f"Verifier 'reward' must be a number, got {raw_reward!r}")

    reward = float(raw_reward)
    if not math.isfinite(reward):
        raise ValueError(f"Verifier 'reward' must be finite, got {raw_reward!r}")
    if reward != -1.0 and not 0.0 <= reward <= 1.0:
        raise ValueError(f"Verifier 'reward' must be -1 or within [0, 1], got {reward}")

    for key, value in rewards.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'Verifier numeric reward field {key!r} is invalid: {value!r}')
        if not math.isfinite(float(value)):
            raise ValueError(f'Verifier numeric reward field {key!r} must be finite, got {value!r}')


def _raise_for_command_failure(result: Any, operation: str) -> None:
    """Raise when a sandbox preparation command failed."""
    if result is None or result.return_code != 0:
        details = getattr(result, 'stderr', None) or getattr(result, 'stdout', None) or 'no output'
        return_code = getattr(result, 'return_code', None)
        raise RuntimeError(f'{operation} failed with code {return_code}: {details}')


def _reset_runtime_directories_command(
    reset_dirs: list[str],
    writable_dirs: list[str],
    *,
    runtime_bin: str = '/nix/runtime/nix-env/bin',
) -> str:
    """Build a mount-safe reset command with remove/recreate and content-only fallback."""
    reset_args = ' '.join(shlex.quote(path) for path in reset_dirs)
    writable_args = ' '.join(shlex.quote(path) for path in writable_dirs)
    return (
        'resolve_tool() { '
        'name="$1"; '
        f'candidate={runtime_bin}/"$name"; '
        'if [ -x "$candidate" ]; then printf "%s\\n" "$candidate"; else command -v "$name"; fi; '
        '}; '
        'rm_bin="$(resolve_tool rm)" || { printf "%s\\n" "Required tool not found: rm" >&2; exit 127; }; '
        'mkdir_bin="$(resolve_tool mkdir)" || { printf "%s\\n" "Required tool not found: mkdir" >&2; exit 127; }; '
        'find_bin="$(resolve_tool find)" || { printf "%s\\n" "Required tool not found: find" >&2; exit 127; }; '
        'chmod_bin="$(resolve_tool chmod)" || { printf "%s\\n" "Required tool not found: chmod" >&2; exit 127; }; '
        'reset_dir() { '
        'dir="$1"; '
        'if [ -L "$dir" ]; then '
        'printf "%s\\n" "Refusing to reset symlink directory: $dir" >&2; return 1; '
        'fi; '
        'if "$rm_bin" -rf -- "$dir" 2>/dev/null && '
        '"$mkdir_bin" -p -- "$dir"; then return 0; fi; '
        'if [ ! -d "$dir" ]; then "$mkdir_bin" -p -- "$dir" || return 1; fi; '
        '"$find_bin" "$dir" -mindepth 1 -maxdepth 1 '
        '-exec "$rm_bin" -rf -- {} +; '
        '}; '
        f'for dir in {reset_args}; do reset_dir "$dir" || exit 1; done; '
        f'"$chmod_bin" 0777 {writable_args}'
    )


async def _prepare_runtime_directories(
    environment: Any,
    *,
    workdir: str,
    reset_dirs: list[str],
    writable_dirs: list[str],
) -> None:
    """Validate the workdir and manage Harbor-owned directories as root."""
    quoted_workdir = shlex.quote(workdir)
    missing_workdir_message = shlex.quote(f'T3-Harbor workdir does not exist in task image: {workdir}')
    reset_command = _reset_runtime_directories_command(reset_dirs, writable_dirs)
    result = await environment.exec(
        (
            f'if [ ! -d {quoted_workdir} ]; then '
            f'printf "%s\\n" {missing_workdir_message} >&2; exit 1; fi; '
            f'{reset_command}'
        ),
        timeout_sec=60,
        user='root',
    )
    _raise_for_command_failure(result, 'Prepare Harbor runtime directories')


async def _configure_git_safe_directory(runtime: Any, workdir: str) -> None:
    """Allow Git metadata access when Gongfeng's runtime user differs from image ownership."""
    quoted_workdir = shlex.quote(workdir)
    result = await runtime.run_command(
        command=(
            'if command -v git >/dev/null 2>&1 && '
            f'[ -e {quoted_workdir}/.git ]; then '
            f'git config --global --add safe.directory {quoted_workdir}; '
            'fi'
        ),
        timeout=30,
    )
    _raise_for_command_failure(result, f'Configure Git safe.directory for {workdir}')


async def _upload_path_to_runtime(runtime: Any, source_path: Path, target_path: str) -> None:
    """Upload a local file or directory to a container path."""
    from nexus.extensions.tasks.terminal.utils import read_local_directory_to_file_items, read_local_file_to_file_item

    if source_path.is_dir():
        file_items = read_local_directory_to_file_items(
            local_dir=source_path,
            container_base_path=target_path,
        )
    elif source_path.is_file():
        file_items = [read_local_file_to_file_item(source_path, target_path)]
    else:
        raise FileNotFoundError(f'Upload source not found: {source_path}')

    if file_items:
        response = await runtime.upload_file(files=file_items, overwrite=True)
        raise_for_upload_failures(response, str(source_path))


def _default_upload_target(workdir: str, source_path: Path) -> str:
    """Default upload target is workdir / basename(source)."""
    return posixpath.join(workdir.rstrip('/') or '/', source_path.name)


def _validate_env_mapping(env: dict[str, str]) -> dict[str, str]:
    """Normalize environment values after rejecting invalid variable names."""
    invalid_names = [name for name in env if not _ENV_NAME_RE.fullmatch(name)]
    if invalid_names:
        raise ValueError(f'Invalid environment variable name(s): {invalid_names}')
    return {name: str(value) for name, value in env.items()}


def _build_boot_launch_command(
    boot_command: str,
    *,
    log_dir: str = '/logs',
    startup_grace_sec: float = 1.0,
) -> str:
    """Build a detached launcher with a portable shell and immediate failure propagation."""
    quoted_log_dir = shlex.quote(log_dir)
    quoted_log_path = shlex.quote(posixpath.join(log_dir, 'boot.log'))
    quoted_pid_path = shlex.quote(posixpath.join(log_dir, 'boot.pid'))
    quoted_rc_path = shlex.quote(posixpath.join(log_dir, 'boot.rc'))
    quoted_started_path = shlex.quote(posixpath.join(log_dir, 'boot.started'))
    quoted_grace = shlex.quote(str(max(startup_grace_sec, 0.0)))
    # Detect fast failures via an exit-code sentinel written by the boot shell itself.
    # Probing ``$!`` is unreliable: ``setsid`` detaches the boot process (so it is no longer
    # our child), and an exited-but-unreaped child still answers ``kill -0`` successfully,
    # which silently swallowed boot failures.
    # The boot command runs in a subshell so that a bare ``exit N`` inside it does not skip
    # the sentinel write. The trailing newline keeps a final comment or here-doc well formed.
    inner_command = (
        f'printf "" > {quoted_started_path}; '
        f'( {boot_command}\n)\nboot_rc=$?; printf "%s\\n" "$boot_rc" > {quoted_rc_path}; exit "$boot_rc"'
    )
    return (
        f'mkdir -p {quoted_log_dir} || exit $?; '
        f'rm -f {quoted_pid_path} {quoted_rc_path} {quoted_started_path}; '
        'if [ -n "${BASH:-}" ] && [ -x "$BASH" ]; then '
        'boot_shell="$BASH"; boot_shell_args="-lc"; '
        'elif [ -x "$(command -v bash 2>/dev/null)" ]; then '
        'boot_shell="$(command -v bash)"; boot_shell_args="-lc"; '
        'elif [ -x "$(command -v sh 2>/dev/null)" ]; then '
        'boot_shell="$(command -v sh)"; boot_shell_args="-c"; '
        'else echo "No usable bash or sh found for Harbor boot command" >&2; exit 127; fi; '
        f'setsid nohup "$boot_shell" "$boot_shell_args" {shlex.quote(inner_command)} '
        f'> {quoted_log_path} 2>&1 < /dev/null & '
        f'boot_pid=$!; printf "%s\\n" "$boot_pid" > {quoted_pid_path} || exit $?; '
        # The grace period must measure the boot command itself, not the login shell's
        # startup cost (sourcing profiles can easily exceed a 1s grace), so wait for the
        # boot shell to signal that it reached the command before starting the clock.
        'boot_waited=0; '
        f'while [ ! -f {quoted_started_path} ] && [ ! -f {quoted_rc_path} ] && [ "$boot_waited" -lt 300 ]; do '
        'sleep 0.1; boot_waited=$((boot_waited+1)); '
        'done; '
        f'sleep {quoted_grace}; '
        f'if [ -f {quoted_rc_path} ]; then '
        f'boot_rc="$(cat {quoted_rc_path} 2>/dev/null)"; '
        'if [ -n "$boot_rc" ] && [ "$boot_rc" -ne 0 ]; then '
        'echo "boot process exited during startup with code $boot_rc"; '
        f'echo "--- boot.log tail ---"; tail -30 {quoted_log_path} 2>/dev/null; '
        'exit "$boot_rc"; '
        'fi; '
        'fi; '
        'echo "boot pid=$boot_pid shell=$boot_shell"'
    )


async def _apply_environment_uploads(
    runtime: Any,
    *,
    task_dir_path: Path,
    workdir: str,
    task_input: HarborBridgeTaskInput,
) -> None:
    """Apply explicit ``[environment.upload]`` items and optional workspace overlay."""
    workspace_dir = task_dir_path / 'environment' / 'workspace'
    if not task_input.skip_environment_upload and workspace_dir.exists() and any(workspace_dir.iterdir()):
        logger.debug(f'Overlaying workspace files from {workspace_dir} to {workdir}')
        await _upload_path_to_runtime(runtime, workspace_dir, workdir)

    for item in task_input.environment_uploads or []:
        source = str(item.get('source', '')).strip()
        if not source:
            continue
        source_path = (task_dir_path / source).resolve()
        try:
            source_path.relative_to(task_dir_path.resolve())
        except ValueError as exc:
            raise ValueError(f'environment.upload source must stay inside task_dir: {source}') from exc

        target = str(item.get('target', '')).strip() or _default_upload_target(workdir, source_path)
        logger.debug(f'Uploading explicit environment item {source_path} to {target}')
        await _upload_path_to_runtime(runtime, source_path, target)


async def _boot_image_services(
    runtime: Any,
    task_input: HarborBridgeTaskInput,
    *,
    prefix: str = '',
) -> None:
    """Run the image's in-container init when Gongfeng bypasses ENTRYPOINT.

    gongfeng runs nexus-runtime as PID 1, so scripts like Canyon's
    ``/omnibus/start_all.sh`` (which start the MCP HTTP servers) never execute. We
    launch boot_command detached (it may ``sleep infinity``), passing boot_env
    explicitly. The launcher waits briefly and propagates the real exit status
    when the command fails during startup instead of returning ``echo``'s status.
    """
    boot_command = getattr(task_input, f'{prefix}boot_command')
    boot_env = getattr(task_input, f'{prefix}boot_env')
    if not boot_command:
        return

    # Launch detached with nohup so a long-running service survives this command.
    # After a short grace period, wait on an already-exited child so its real status
    # is returned. A still-running child is left for Harbor's healthcheck to probe.
    normalized_boot_env = _validate_env_mapping(boot_env or {})
    # Humanlaya's trusted service launcher expands these names inside its uploaded
    # docker-compose.yaml. Keep the passthrough narrowly scoped so arbitrary task
    # boot commands cannot receive Judge credentials merely by naming the variables.
    if boot_command.strip() == 'bash /environment/start-services.sh' and 'TASK_MCP_URL' in normalized_boot_env:
        for variable_name in ('YOUR_API_KEY', 'YOUR_BASE_URL', 'JUDGE_MODEL'):
            value = os.environ.get(variable_name)
            if value and variable_name not in normalized_boot_env:
                normalized_boot_env[variable_name] = value
    launch = _build_boot_launch_command(boot_command)
    logger.debug(f'Booting image services: {boot_command} (env keys: {list(normalized_boot_env)})')
    res = await runtime.run_command(command=launch, timeout=60, extra_env=normalized_boot_env)
    logger.info(f'Boot launch: rc={res.return_code} out={(res.stdout or "").strip()[:200]}')
    if res.return_code != 0:
        raise RuntimeError(
            f'Image boot command failed with code {res.return_code}: {res.stderr or res.stdout or "no output"}'
        )


BOOT_HEALTHCHECK_INTERVAL_SEC = 10.0
BOOT_HEALTHCHECK_RETRIES = 90


async def _run_boot_healthcheck(environment: Any) -> None:
    """Run readiness checks with a 900s default while preserving task overrides."""
    task_env_config = getattr(environment, 'task_env_config', None)
    healthcheck = getattr(task_env_config, 'healthcheck', None) if task_env_config is not None else None
    if healthcheck is None:
        await environment.run_healthcheck()
        return

    explicit_fields = getattr(healthcheck, 'model_fields_set', set())
    updates: dict[str, Any] = {}
    if 'interval_sec' not in explicit_fields:
        updates['interval_sec'] = BOOT_HEALTHCHECK_INTERVAL_SEC
    if 'retries' not in explicit_fields:
        updates['retries'] = BOOT_HEALTHCHECK_RETRIES

    effective_healthcheck = healthcheck.model_copy(update=updates) if updates else healthcheck
    logger.debug(
        'Boot healthcheck: interval=%ss retries=%s (~%ss retry budget) for %r',
        effective_healthcheck.interval_sec,
        effective_healthcheck.retries,
        int(effective_healthcheck.interval_sec * effective_healthcheck.retries),
        effective_healthcheck.command,
    )
    await environment.run_healthcheck(effective_healthcheck)


async def _run_collect_hooks(environment: Any, harbor_task: Any) -> None:
    """Run required Harbor verifier.collect hooks in their declared services."""
    from harbor.constants import MAIN_SERVICE_NAME

    for hook in harbor_task.config.verifier.collect:
        if hook.service != MAIN_SERVICE_NAME and not environment.capabilities.docker_compose:
            raise ValueError(
                f'Harbor Bridge collect service={hook.service!r} requires native Compose; '
                'non-Compose tasks support only service="main"'
            )
        try:
            if hook.service == MAIN_SERVICE_NAME:
                result = await environment.exec(
                    command=hook.command,
                    timeout_sec=int(hook.timeout_sec),
                    user=hook.user,
                )
            else:
                result = await environment.service_exec(
                    hook.command,
                    service=hook.service,
                    timeout_sec=int(hook.timeout_sec),
                    user=hook.user,
                )
        except Exception as exc:
            raise RuntimeError(f'Verifier collect hook failed: {hook.command}: {exc}') from exc
        if result.return_code != 0:
            details = result.stderr or result.stdout or 'no output'
            raise RuntimeError(
                f'Verifier collect hook exited with code {result.return_code}: {hook.command}: {details}'
            )


async def _release_runtime_with_retry(
    ctx: Any,
    name: str,
    *,
    force: bool = False,
    attempts: int = 2,
) -> bool:
    """Release a named runtime, retrying transient provider stop failures."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await ctx.release_runtime(name, force=force)
        except Exception as exc:
            last_error = exc
            logger.warning(
                'Failed to release runtime %s (attempt %s/%s): %s',
                name,
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts:
                await asyncio.sleep(0)
    assert last_error is not None
    raise last_error


async def _collect_artifacts(
    environment: Any,
    harbor_task: Any,
    trial_paths: Any,
) -> tuple[Any, Any]:
    """Collect Harbor artifacts from the agent runtime into host staging."""
    from harbor.constants import MAIN_SERVICE_NAME
    from harbor.models.trial.paths import EnvironmentPaths
    from harbor.trial.artifact_handler import ArtifactHandler

    artifact_handler = ArtifactHandler(
        artifacts=harbor_task.config.artifacts,
        logger=logger,
    )
    artifact_handler.begin_collection()
    source_paths = EnvironmentPaths.for_os(environment.os)
    sidecar_services = set(artifact_handler.sidecar_services())
    await artifact_handler.download_artifacts(
        environment,
        trial_paths.artifacts_dir,
        source_artifacts_dir=source_paths.artifacts_dir,
        services={MAIN_SERVICE_NAME},
    )
    if sidecar_services:
        if not environment.capabilities.docker_compose:
            raise ValueError('Sidecar artifact collection requires native Compose')
        # Harbor determines artifact file-vs-directory shape with an in-service
        # exec. Collect main while it is still running, then freeze it before
        # reading evidence from sidecars.
        await environment.stop_service(MAIN_SERVICE_NAME)
        await artifact_handler.download_artifacts(
            environment,
            trial_paths.artifacts_dir,
            source_artifacts_dir=source_paths.artifacts_dir,
            services=sidecar_services,
        )
    return artifact_handler, source_paths


def _restore_tests_executable_bits(tests_dir: Path) -> None:
    """Re-add the execute bit to staged test scripts before Harbor uploads them.

    Nexus's TaskFiles staging is byte-oriented (``Path.write_bytes``) and drops
    file modes, so a ``tests/`` helper that ``test.sh`` runs directly (e.g.
    ``test.source.sh``) is materialized as 0644. Harbor's Verifier only re-chmods
    the main test script, so the helper then fails with ``Permission denied`` on
    direct exec — even for root, since executing a file with no execute bit is
    EACCES regardless of ownership. Restore +x here (Harbor's ``upload_dir``
    preserves mode), mirroring the source tree's own executable scripts.
    """
    if not tests_dir.is_dir():
        return
    for path in tests_dir.rglob('*'):
        if not path.is_file():
            continue
        try:
            mode = path.stat().st_mode
            path.chmod(mode | 0o111)
        except OSError:
            logger.debug('Could not restore +x on staged test file %s', path, exc_info=True)


async def _run_harbor_verifier(
    *,
    harbor_task: Any,
    trial_paths: Any,
    environment: Any,
    skip_tests_upload: bool,
    timeout_sec: float,
    needs_agent_trajectory: bool = False,
    browser: dict[str, Any] | None = None,
    tool_state: dict[str, Any] | None = None,
    include_test_stdout: bool = True,
) -> tuple[dict[str, float], dict[str, Any], str]:
    """Run Harbor Verifier and normalize rich reward.json payloads."""
    from harbor.verifier.verifier import (
        RewardFileEmptyError,
        RewardFileNotFoundError,
        Verifier,
        VerifierOutputParseError,
    )

    rewards: dict[str, float] = {}
    reward_metadata: dict[str, Any] = {}
    raw_reward_payload: dict[str, Any] | None = None
    resolved_verifier_env = resolve_env_mapping(harbor_task.config.verifier.env or {})
    verifier_env = verifier_env_with_t3_agent_result(
        resolved_verifier_env,
        environment_os=environment.os,
    )
    if needs_agent_trajectory:
        verifier_env = verifier_env_with_t3_agent_trajectory(verifier_env, environment_os=environment.os)
    verifier_env = verifier_env_with_browser_session(
        verifier_env,
        browser=browser,
        tool_state=tool_state or {},
    )
    # Staging drops file modes; restore +x on the uploaded test scripts so a
    # directly-exec'd helper (e.g. test.source.sh) is not EACCES in the sandbox.
    # Best-effort: never let this side fix break verification itself.
    if not skip_tests_upload:
        tests_dir = getattr(getattr(harbor_task, 'paths', None), 'tests_dir', None)
        if tests_dir is not None:
            _restore_tests_executable_bits(Path(tests_dir))
    with environment.with_default_user(harbor_task.config.verifier.user):
        verifier = Verifier(
            task=harbor_task,
            trial_paths=trial_paths,
            environment=environment,
            skip_tests_upload=skip_tests_upload,
            verifier_env=verifier_env,
            include_logs=[trial_paths.test_stdout_path.name] if include_test_stdout else None,
            exclude_logs=None if include_test_stdout else ['*'],
        )
        try:
            result = await asyncio.wait_for(
                verifier.verify(),
                timeout=timeout_sec,
            )
            rewards = result.rewards or {}
        except asyncio.TimeoutError as timeout_exc:
            if not include_test_stdout:
                await _capture_remote_verifier_output_tail(environment, trial_paths)
            detail = _verifier_failure_detail(
                trial_paths,
                sensitive_source=resolved_verifier_env,
            )
            raise TimeoutError(f'Verifier timed out after {timeout_sec}s; diagnostics: {detail}') from timeout_exc
        except ValidationError as verify_exc:
            non_numeric = _non_numeric_reward_entries(verify_exc)
            if not non_numeric or not trial_paths.reward_json_path.exists():
                if not include_test_stdout:
                    await _capture_remote_verifier_output_tail(environment, trial_paths)
                detail = _verifier_failure_detail(
                    trial_paths,
                    sensitive_source=resolved_verifier_env,
                )
                logger.error('Verifier result validation failed; diagnostics: %s', detail)
                raise
            # Non-numeric extras are the normal shape for these tasks and the recovery is
            # lossless below, so this stays out of the batch log: a few thousand tasks
            # of it would bury the validation failures that do need attention. Only entry
            # names are logged so values such as judge credentials cannot leak.
            logger.debug(
                'VerifierResult rejected non-numeric rewards (%s); normalizing rich reward.json directly',
                ', '.join(sorted(non_numeric)),
            )
        except (RewardFileNotFoundError, RewardFileEmptyError, VerifierOutputParseError) as verify_exc:
            if not include_test_stdout:
                await _capture_remote_verifier_output_tail(environment, trial_paths)
            safe_error = redact_provider_io(
                f'{type(verify_exc).__name__}: {verify_exc}',
                sensitive_source=resolved_verifier_env,
            )
            detail = _verifier_failure_detail(
                trial_paths,
                sensitive_source=resolved_verifier_env,
            )
            raise RuntimeError(
                f'Verifier did not produce a usable reward: {safe_error}; diagnostics: {detail}'
            ) from None
        except Exception as verify_exc:
            if not include_test_stdout:
                await _capture_remote_verifier_output_tail(environment, trial_paths)
            safe_error = redact_provider_io(
                f'{type(verify_exc).__name__}: {verify_exc}',
                sensitive_source=resolved_verifier_env,
            )
            detail = _verifier_failure_detail(
                trial_paths,
                sensitive_source=resolved_verifier_env,
            )
            logger.error('Verifier failed before reward normalization: %s; diagnostics: %s', safe_error, detail)
            raise

    if trial_paths.reward_json_path.exists():
        payload = json.loads(trial_paths.reward_json_path.read_text() or '{}')
        if not isinstance(payload, dict):
            raise ValueError(f'Verifier reward.json must contain an object, got {type(payload).__name__}')
        raw_reward_payload = payload
        rewards, reward_metadata = _split_reward_payload(payload)
    try:
        _validate_reward_contract(rewards, raw_payload=raw_reward_payload)
    except ValueError as contract_exc:
        if not include_test_stdout:
            await _capture_remote_verifier_output_tail(environment, trial_paths)
        safe_contract_error = redact_provider_io(
            str(contract_exc),
            sensitive_source=resolved_verifier_env,
        )
        detail = _verifier_failure_detail(
            trial_paths,
            reward_metadata=reward_metadata,
            sensitive_source=resolved_verifier_env,
        )
        raise ValueError(f'{safe_contract_error}; verifier diagnostics: {detail}') from None
    reward = float(rewards['reward'])
    if reward == -1.0:
        if not include_test_stdout:
            await _capture_remote_verifier_output_tail(environment, trial_paths)
        detail = _verifier_failure_detail(
            trial_paths,
            reward_metadata=reward_metadata,
            sensitive_source=resolved_verifier_env,
        )
        raise RuntimeError(f'Verifier reported a system error with reward=-1: {detail}')

    test_stdout = _read_optional_verifier_stdout(trial_paths.test_stdout_path) if include_test_stdout else ''
    return rewards, reward_metadata, test_stdout


@task_registry.register('harbor_bridge_task')
async def harbor_bridge_task(task_input: HarborBridgeTaskInput) -> HarborBridgeTaskOutput:
    """Run a harbor benchmark task through nexus, using ANY configured agent.

    Flow:
    1. Get nexus runtime (gongfeng sandbox)
    2. Load harbor Task from task_dir (for verifier metadata)
    3. Initialize environment directories
    4. Run the configured agent (terminus2, claude_code, etc.) via run_agent()
    5. Run harbor Verifier to grade results
    6. Return reward

    The agent is NOT defined here — it's configured in RunInput.agent_name.
    Any nexus agent that can operate in a terminal works here.
    """
    from harbor.models.task.verifier_mode import (
        VerifierEnvironmentMode,
        resolve_effective_verifier_env_config,
        resolve_task_verifier_mode,
    )
    from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

    from nexus.extensions.harbor_bridge.compose.environment import ComposeHarborEnvironment
    from nexus.extensions.harbor_bridge.environment import GongfengEnvironment

    span_id = get_current_span_id()
    phase('task.init')
    task_termination_reason = 'INITIALIZATION_ERROR'
    task_span_attributes: dict[str, Any] = {}
    trajectory_strategy = (
        task_input.anti_hack_policy.trajectory_strategy if task_input.anti_hack_policy is not None else None
    )
    bad_pattern_policy = task_input.tool_call_bad_pattern_policy
    agent_info = getattr(task_input.run_agent, 'nexus_meta', {})
    agent_name = str(agent_info.get('name') or '')
    logger.debug(f'Starting harbor bridge task: {task_input.instance_id}')
    try:
        if task_input.anti_hack_policy is not None and task_input.mode != 'agent':
            raise ValueError('anti_hack_policy is supported only for Agent runs')
        if (
            task_input.compose_plan is not None
            and task_input.mode == 'agent'
            and not agent_info.get('meta', {}).get('supports_execution_target', False)
        ):
            raise ValueError(
                f'Agent {agent_name or "<unknown>"!r} does not support Native Compose execution; '
                'select an Agent that supports execution_target'
            )
        if trajectory_strategy:
            from nexus.extensions.harbor_bridge.anti_hack import validate_trajectory_strategy

            validate_trajectory_strategy(trajectory_strategy)
        if bad_pattern_policy is not None and task_input.mode != 'agent':
            raise ValueError('tool_call_bad_pattern_policy is supported only for Agent runs')
        ctx = require_context()
    except Exception:
        task_span_attributes['hytuner.termination_reason'] = task_termination_reason
        _update_task_span(span_id, task_span_attributes)
        raise

    if not ctx.runtime_provider:
        task_span_attributes['hytuner.termination_reason'] = task_termination_reason
        _update_task_span(span_id, task_span_attributes)
        raise RuntimeError('Harbor bridge task requires a runtime provider (gongfeng)')

    # Resolve the task bundle before allocating a sandbox. Cloud assignments provide
    # a standard TaskFiles object backed by the assignment ZIP; local legacy runs
    # may continue to pass a live task_dir.
    task_scope_dir = Path(tempfile.mkdtemp(prefix='harbor_task_'))
    try:
        task_dir_path = await _materialize_task_directory(
            getattr(ctx, 'task_files', None),
            serialized_task_dir=task_input.task_dir,
            # Harbor derives a task's fallback name from the directory
            # basename when task.toml has no explicit [task].name.  Preserve
            # the serialized dataset item identity instead of making every
            # cloud assignment appear as a task named "task".
            destination=task_scope_dir / _task_scope_name(task_input.instance_id),
        )
        loaded_task = load_harbor_task(task_dir_path)
        has_compose_declaration = compose_config_from_task(loaded_task.config) is not None
        if has_compose_declaration != (task_input.compose_plan is not None):
            raise RuntimeError(
                'Serialized compose_plan does not match the authoritative task declaration; regenerate the RunInput'
            )
        harbor_task = loaded_task.task
        logger.debug(f'Loaded harbor task: {harbor_task.name}')
        verifier_mode = resolve_task_verifier_mode(harbor_task.config)
        if task_input.verifier_mode != verifier_mode.value:
            raise ValueError(
                f'Serialized verifier_mode={task_input.verifier_mode!r} differs from '
                f'Harbor task mode={verifier_mode.value!r}; regenerate the RunInput.'
            )
        if task_input.browser and verifier_mode != VerifierEnvironmentMode.SHARED:
            raise ValueError(
                'T3-Harbor browser tasks require a shared Verifier because the live '
                'browser session belongs to the main runtime'
            )
        instruction = task_input.instruction or harbor_task.instruction
        if not instruction:
            raise ValueError(f'No instruction found for task {harbor_task.name}')
    except BaseException:
        # The task scope is allocated before Harbor validates the serialized
        # metadata.  Keep validation failures from leaking a worker-local
        # directory even when no runtime has been acquired yet.
        shutil.rmtree(task_scope_dir, ignore_errors=True)
        task_span_attributes['hytuner.termination_reason'] = task_termination_reason
        _update_task_span(span_id, task_span_attributes)
        raise

    safe_name = harbor_task.name.replace('/', '_').replace('\\', '_')
    trial_dir = Path(tempfile.mkdtemp(prefix=f'harbor_trial_{safe_name}_'))
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()
    main_environment: GongfengEnvironment | None = None
    verifier_environment: GongfengEnvironment | None = None
    compose_topology: ComposeTopology | None = None
    verifier_runtime_acquired = False
    extra_fields: dict[str, Any] = {}
    task_error: BaseException | None = None
    try:
        with phase('task.init.get_runtime'):
            main_runtime = await ctx.get_runtime('main')
        logger.debug('Got main nexus runtime for harbor bridge task')
        runtime_id = _context_runtime_id(ctx, 'main')
        if runtime_id:
            task_span_attributes['hytuner.runtime_id'] = runtime_id
        workdir = task_input.workdir
        agent_baseline = harbor_task.config.environment.resolve_baseline()
        agent_phase = harbor_task.config.agent.explicit_phase_policy()
        # task_input reflects the static policy actually provisioned for this
        # runtime. Converters deliberately open networking for CLI agents
        # (hycode/codex/mini), even when the source task.toml is offline.
        # Do not pass the stale source no-network policy to Harbor validation.
        agent_runtime_policy = (agent_phase or agent_baseline) if task_input.network_isolation_enabled else None
        if task_input.compose_plan is not None:
            with phase('task.init.compose'):
                compose_topology = ComposeTopology(
                    runtime=main_runtime,
                    spec=topology_spec_from_plan(
                        task_input.compose_plan,
                        agent_transport='socket-v1' if task_input.mode == 'agent' else 'none',
                    ),
                    task_dir=task_dir_path,
                    project_name=_compose_project_name(
                        task_input.instance_id or harbor_task.name,
                        str(getattr(ctx, 'run_id', '') or trial_dir.name),
                    ),
                    agent_helper_dir=_compose_agent_helper_dir() if task_input.mode == 'agent' else None,
                )
                await compose_topology.stage()
                await compose_topology.start()
                await compose_topology.wait_ready()
                main_environment = ComposeHarborEnvironment(
                    topology=compose_topology,
                    primary_service=task_input.compose_plan.primary_service,
                    persistent_environment=task_input.compose_plan.environment,
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
                execution_runtime = main_environment.primary_runtime
        else:
            main_environment = GongfengEnvironment(
                runtime=main_runtime,
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
            execution_runtime = main_runtime

        reset_dirs = ['/logs/agent', '/logs/verifier', '/logs/artifacts']
        writable_log_dirs = list(reset_dirs)
        if verifier_mode == VerifierEnvironmentMode.SEPARATE or not task_input.skip_tests_upload:
            reset_dirs.append('/tests')
        with phase('task.init.prepare_directories'):
            await _prepare_runtime_directories(
                main_environment,
                workdir=workdir,
                reset_dirs=reset_dirs,
                writable_dirs=writable_log_dirs,
            )
            if compose_topology is None or task_input.mode != 'agent':
                await _configure_git_safe_directory(execution_runtime, workdir)
            await _apply_environment_uploads(
                execution_runtime,
                task_dir_path=task_dir_path,
                workdir=workdir,
                task_input=task_input,
            )
        with phase('task.init.boot'):
            await main_environment.start(force_build=False)
            await _boot_image_services(execution_runtime, task_input)
            await _run_boot_healthcheck(main_environment)

        phase('task.agent')
        task_termination_reason = 'UNHANDLED_ERROR'
        ctx.task_config['working_dir'] = workdir
        ctx.tool_state.setdefault('working_dir', workdir)
        agent_result: T3AgentResult
        agent_output: Any = None
        if task_input.mode == 'oracle':
            task_termination_reason = 'ENV_ERROR'
            await _run_oracle(
                main_environment,
                harbor_task,
                trial_paths,
                instruction=instruction,
                timeout_sec=task_input.agent_timeout,
            )
            agent_result = build_t3_agent_result(mode='oracle', status='success')
        elif task_input.mode == 'nop':
            logger.debug('RUN_NOP enabled: skipping agent and oracle before verification')
            agent_result = build_t3_agent_result(mode='nop', status='skipped')
        else:
            agent_instruction = _build_agent_instruction(
                instruction,
                anti_hack_policy=task_input.anti_hack_policy,
            )
            watchdog_timeout = _agent_watchdog_timeout(task_input.agent_timeout)
            execution_target = None
            agent_kwargs: dict[str, Any] = {
                'user_prompt': agent_instruction,
                'working_dir': '/nexus/control' if compose_topology is not None else workdir,
                'timeout': float(task_input.agent_timeout),
                'mcp_servers': copy.deepcopy(task_input.mcp_servers),
                **({'browser': copy.deepcopy(task_input.browser)} if task_input.browser else {}),
            }
            if compose_topology is not None:
                execution_target = ComposeExecutionTarget(
                    project_name=compose_topology.project_name,
                    project_directory=compose_topology.controller_dir,
                    compose_files=list(compose_topology.compose_files),
                    service=compose_topology.spec.primary_service,
                    transport_host_dir=compose_topology.agent_transport_host_dir,
                    transport_container_dir=compose_topology.agent_transport_container_dir,
                    socket_bridge_host_path=compose_topology.socket_bridge_host_path,
                    socket_bridge_container_path=compose_topology.socket_bridge_container_path,
                    user=harbor_task.config.agent.user,
                )
                agent_kwargs['working_dir'] = workdir
            try:
                with agent_execution_target(execution_target):
                    agent_output = await _run_agent_with_watchdog(
                        task_input.run_agent(**agent_kwargs),
                        watchdog_timeout,
                        getattr(ctx, 'timeout_handle', None),
                    )
            except asyncio.TimeoutError:
                agent_result = build_t3_agent_result(
                    mode='agent',
                    status='timeout',
                    failure_stage='watchdog',
                    failure_message=(
                        f'Agent watchdog timed out after {watchdog_timeout:g}s of active time '
                        f'({task_input.agent_timeout}s Agent budget + '
                        f'{_AGENT_WATCHDOG_GRACE_SECONDS:g}s cleanup grace)'
                    ),
                )
                logger.warning('%s; continuing with verification...', agent_result.failure_message)
            except Exception as exc:
                raw_failure_stage = getattr(exc, 'failure_stage', None)
                agent_result = build_t3_agent_result(
                    mode='agent',
                    status='error',
                    failure_kind='infrastructure',
                    failure_stage=(str(raw_failure_stage).strip() or None if raw_failure_stage is not None else None),
                    failure_message=f'{type(exc).__name__}: {exc}',
                )
                logger.exception('Agent infrastructure failure before a valid result')
            else:
                agent_result = build_t3_agent_result(mode='agent', agent_output=agent_output)

        agent_term_reason = agent_termination_reason(agent_result)
        task_termination_reason = agent_term_reason
        phase('task.handoff')
        with phase('task.handoff.artifact_collect_time'):
            agent_result_host_path = write_t3_agent_result(trial_paths, agent_result)
            capture_agent_trajectory = (
                task_input.needs_agent_trajectory or bool(trajectory_strategy) or bad_pattern_policy is not None
            )
            reported_messages = extract_reported_messages(agent_output)
            # Preserve the existing fast path for ordinary trajectory/anti-hack
            # consumers. Bad-pattern detection additionally needs the tool schemas
            # recorded by ChatTracker, even when the Agent reports its own messages.
            needs_tracker_snapshot = capture_agent_trajectory and (
                not reported_messages or bad_pattern_policy is not None
            )
            tracker_fallback, tracker_tools = (
                await _chat_tracker_fallback(execution_runtime, final_answer=agent_result.final_answer)
                if needs_tracker_snapshot
                else ([], [])
            )
            raw_quality_messages = (
                extract_messages(agent_output, fallback_messages=tracker_fallback) if capture_agent_trajectory else []
            )
            quality_messages = normalize_anthropic_block_messages(raw_quality_messages) or raw_quality_messages
            agent_trajectory = (
                build_t3_agent_trajectory(
                    mode=agent_result.mode,
                    status=agent_result.status,
                    agent_output=agent_output,
                    fallback_messages=tracker_fallback,
                )
                if capture_agent_trajectory
                else None
            )
            agent_trajectory_host_path = (
                write_t3_agent_trajectory(trial_paths, agent_trajectory)
                if task_input.needs_agent_trajectory and agent_trajectory is not None
                else None
            )
            if agent_result.failure_kind == 'infrastructure':
                logger.error('Agent infrastructure result: %s', agent_result.model_dump_json())
                raise AgentInfrastructureError(agent_result.failure_message or 'Agent infrastructure failure')
            task_termination_reason = 'ENV_ERROR'
            require_browser_session_opened(task_input.browser, ctx.tool_state)
            environment = main_environment
            skip_tests_upload = task_input.skip_tests_upload
            await _run_collect_hooks(main_environment, harbor_task)
        if verifier_mode == VerifierEnvironmentMode.SHARED:
            with phase('task.handoff.env_setup_time'):
                await upload_t3_agent_result(
                    execution_runtime,
                    agent_result_host_path,
                    environment_os=main_environment.os,
                )
                if agent_trajectory_host_path is not None:
                    await upload_t3_agent_trajectory(
                        execution_runtime,
                        agent_trajectory_host_path,
                        environment_os=main_environment.os,
                    )
        if verifier_mode == VerifierEnvironmentMode.SEPARATE:
            with phase('task.handoff.artifact_collect_time'):
                artifact_handler, main_paths = await _collect_artifacts(
                    main_environment,
                    harbor_task,
                    trial_paths,
                )
            with phase('task.handoff.release_main_time'):
                await main_environment.stop(delete=True)
                # A Compose topology is owned by the outer main runtime. Tear it down
                # before releasing that runtime; otherwise deferred cleanup can hang
                # while trying to execute `docker compose down` through a dead client.
                if compose_topology is not None:
                    await compose_topology.delete()
                    compose_topology = None
                if not await _release_runtime_with_retry(ctx, 'main', force=True):
                    raise RuntimeError('Separate verifier requires the main runtime to be released')
            main_environment = None

            task_termination_reason = 'INITIALIZATION_ERROR'
            with phase('task.handoff.sandbox_init_time'):
                verifier_runtime = await ctx.get_runtime('verifier')
            verifier_runtime_acquired = True
            eval_runtime_id = _context_runtime_id(ctx, 'verifier')
            if eval_runtime_id:
                task_span_attributes['hytuner.eval_runtime_id'] = eval_runtime_id
            with phase('task.handoff.env_setup_time'):
                verifier_environment_config = resolve_effective_verifier_env_config(
                    harbor_task.config,
                    step_cfg=None,
                )
                if verifier_environment_config is None:
                    raise RuntimeError('Separate verifier did not resolve an effective environment')
                verifier_baseline = verifier_environment_config.resolve_baseline()
                verifier_phase = harbor_task.config.verifier.explicit_phase_policy()
                # Same reasoning as the agent phase above: only hand Harbor a policy
                # when this runtime was actually provisioned with network isolation.
                verifier_runtime_policy = (
                    (verifier_phase or verifier_baseline) if task_input.verifier_network_isolation_enabled else None
                )
                verifier_environment = GongfengEnvironment(
                    runtime=verifier_runtime,
                    environment_dir=harbor_task.paths.tests_dir,
                    environment_name=harbor_task.name,
                    session_id=f'nexus_{task_input.instance_id or harbor_task.name}__verifier',
                    trial_paths=trial_paths,
                    task_env_config=verifier_environment_config,
                    default_timeout_sec=task_input.test_timeout,
                    network_isolation_enabled=task_input.verifier_network_isolation_enabled,
                    harbor_network_policy=verifier_runtime_policy,
                    gpus_enabled=bool(verifier_environment_config.gpus),
                )
                environment = verifier_environment
                await environment.start(force_build=False)
                await _boot_image_services(verifier_runtime, task_input, prefix='verifier_')
                await _run_boot_healthcheck(environment)
                task_termination_reason = 'ENV_ERROR'
                verifier_paths = EnvironmentPaths.for_os(environment.os)
                verifier_reset_dirs = [
                    verifier_paths.agent_dir,
                    verifier_paths.verifier_dir,
                    verifier_paths.artifacts_dir,
                ]
                if not skip_tests_upload:
                    verifier_reset_dirs.append(verifier_paths.tests_dir)
                reset_result = await environment.empty_dirs(verifier_reset_dirs, chmod=True)
                _raise_for_command_failure(reset_result, 'Reset separate verifier directories')
                await upload_t3_agent_result(
                    verifier_runtime,
                    agent_result_host_path,
                    environment_os=environment.os,
                )
                if agent_trajectory_host_path is not None:
                    await upload_t3_agent_trajectory(
                        verifier_runtime,
                        agent_trajectory_host_path,
                        environment_os=environment.os,
                    )
                await artifact_handler.upload_artifacts(
                    environment,
                    trial_paths.artifacts_dir,
                    source_artifacts_dir=main_paths.artifacts_dir,
                    target_artifacts_dir=verifier_paths.artifacts_dir,
                )
        else:
            with phase('task.handoff.env_setup_time'):
                shared_paths = EnvironmentPaths.for_os(environment.os)
                shared_reset_dirs = [shared_paths.verifier_dir]
                if not skip_tests_upload:
                    shared_reset_dirs.append(shared_paths.tests_dir)
                reset_result = await environment.empty_dirs(shared_reset_dirs, chmod=True)
                _raise_for_command_failure(reset_result, 'Reset shared verifier directories')

        phase('task.scoring')
        task_termination_reason = 'REWARD_ERROR'
        verifier_error: BaseException | None = None
        reward_started = time.monotonic()
        try:
            rewards, reward_metadata, test_stdout = await _run_harbor_verifier(
                harbor_task=harbor_task,
                trial_paths=trial_paths,
                environment=environment,
                skip_tests_upload=skip_tests_upload,
                timeout_sec=task_input.test_timeout,
                needs_agent_trajectory=task_input.needs_agent_trajectory,
                browser=task_input.browser,
                tool_state=ctx.tool_state,
                include_test_stdout=False,
            )
        except BaseException as exc:
            verifier_error = exc
        finally:
            task_span_attributes['hytuner.reward_time'] = time.monotonic() - reward_started

        with phase('task.scoring.check_score_time'):
            cleanup_error: Exception | None = None
            if verifier_mode == VerifierEnvironmentMode.SEPARATE:
                try:
                    await environment.stop(delete=True)
                    if verifier_runtime_acquired:
                        await _release_runtime_with_retry(ctx, 'verifier')
                        verifier_runtime_acquired = False
                    verifier_environment = None
                except Exception as exc:
                    cleanup_error = exc
                    logger.error('Separate verifier cleanup failed; deferring to context shutdown: %s', exc)

            if verifier_error is not None:
                if cleanup_error is not None:
                    logger.error('Preserving verifier error despite cleanup failure: %s', cleanup_error)
                raise verifier_error

            reward = float(rewards['reward'])
            if trajectory_strategy:
                from nexus.extensions.harbor_bridge.anti_hack import AntiHackContext, run_trajectory_strategy

                if agent_trajectory is None:  # pragma: no cover - capture is derived from trajectory_strategy
                    raise AssertionError('Anti-hack trajectory capture was not initialized')
                anti_hack_result = await run_trajectory_strategy(
                    trajectory_strategy,
                    AntiHackContext(
                        trajectory=agent_trajectory,
                        task_metadata=_anti_hack_task_metadata(harbor_task, task_input),
                    ),
                )
                extra_fields['is_hack'] = anti_hack_result.detected
                extra_fields['anti_hack_status'] = anti_hack_result.status
                logger.info(
                    'Harbor anti-hack strategy=%s status=%s detected=%s violations=%d reasons=%s',
                    trajectory_strategy,
                    anti_hack_result.status,
                    anti_hack_result.detected,
                    len(anti_hack_result.violations),
                    [violation.get('reason') for violation in anti_hack_result.violations],
                )
            score = reward
            if bad_pattern_policy is not None:
                from nexus.extensions.harbor_bridge.bad_pattern_operator import (
                    apply_tool_call_bad_pattern_penalty,
                    detect_tool_call_bad_patterns,
                    legacy_deferred_metrics,
                    run_legacy_deferred_monitors,
                )

                if agent_trajectory is None:  # pragma: no cover - capture is derived from the policy
                    raise AssertionError('Bad-pattern trajectory capture was not initialized')
                language_ch_enabled = FINAL_TURN_LANGUAGE_CH_RULE in bad_pattern_policy.enabled_rules_for_agent(
                    agent_name
                )
                language_ch_logged = False
                original_score = score
                try:
                    detection = detect_tool_call_bad_patterns(
                        quality_messages,
                        merge_tool_schemas(_mcp_tool_schemas(ctx), tracker_tools),
                        termination_reason=agent_term_reason,
                        **bad_pattern_policy.detection_kwargs(agent_name=agent_name),
                    )
                    score, penalty_metrics = apply_tool_call_bad_pattern_penalty(score, detection, bad_pattern_policy)
                    language_ch_event = final_turn_language_ch_diagnostic(
                        detection,
                        bad_pattern_policy,
                        agent_name=agent_name,
                        instance_id=task_input.instance_id,
                        original_score=original_score,
                        final_score=score,
                    )
                    if language_ch_event is not None:
                        logger.info('CODESWITCH_CH %s', json.dumps(language_ch_event, sort_keys=True))
                        language_ch_logged = True
                    legacy_results = run_legacy_deferred_monitors(
                        quality_messages,
                        enabled_families=bad_pattern_policy.enabled_legacy_deferred(),
                        extra_fields={
                            'messages': quality_messages,
                            'tool_stats': getattr(agent_output, 'tool_stats', None),
                        },
                    )
                    penalty_metrics.update(legacy_deferred_metrics(legacy_results))
                    extra_fields['tool_call_bad_pattern'] = detection
                    extra_fields.update(legacy_results)
                    extra_fields['reward_extra_info'] = {
                        **penalty_metrics,
                        'extra_reward_metrics': dict(penalty_metrics),
                    }
                    logger.info(
                        'Harbor bad-pattern policy=%s detected=%s issues=%d penalty=%s score=%s',
                        bad_pattern_policy.id,
                        detection['is_bad_pattern'],
                        detection['issue_count'],
                        detection['penalty'],
                        score,
                    )
                except Exception as exc:
                    if language_ch_enabled and not language_ch_logged:
                        logger.warning(
                            'CODESWITCH_CH %s',
                            json.dumps(
                                {
                                    'event': FINAL_TURN_LANGUAGE_CH_RULE,
                                    'status': 'error',
                                    'entered': True,
                                    'instance_id': task_input.instance_id,
                                    'agent_name': agent_name,
                                    'policy_id': bad_pattern_policy.id,
                                    'rule_enabled': True,
                                    'error_type': type(exc).__name__,
                                    'original_score': float(original_score),
                                    'final_score': float(score),
                                },
                                sort_keys=True,
                            ),
                        )
                    logger.warning('Harbor bad-pattern detection skipped (fail-open): %s', exc, exc_info=True)
                    extra_fields['tool_call_bad_pattern'] = {
                        'status': 'error',
                        'error': f'{type(exc).__name__}: {exc}',
                        'is_bad_pattern': False,
                        'issue_count': 0,
                        'issue_counts': {},
                        'hits': [],
                    }
        task_termination_reason = agent_term_reason
        logger.info(f'Verifier result: reward={reward}, score={score}, rewards={rewards}')
    except Exception as exc:
        task_error = exc
        logger.exception('Harbor Bridge task failed')
        raise
    except BaseException as exc:
        task_error = exc
        raise
    finally:
        # On success cleanup replaces the completed phase. During exception unwinding
        # it nests under the failed phase so @observe can still close that phase with
        # ERROR rather than phase() prematurely marking it OK.
        with phase('task.cleanup', close_previous=task_error is None):
            task_span_attributes['hytuner.termination_reason'] = task_termination_reason
            _update_task_span(span_id, task_span_attributes)
            if verifier_environment is not None:
                try:
                    await verifier_environment.stop(delete=True)
                except Exception:
                    logger.exception('Failed to stop verifier environment during deferred cleanup')
            if verifier_runtime_acquired:
                try:
                    await _release_runtime_with_retry(ctx, 'verifier')
                except Exception:
                    logger.exception('Failed to release verifier runtime; context shutdown will retry')
            if main_environment is not None:
                try:
                    await main_environment.stop(delete=True)
                except Exception:
                    logger.exception('Failed to stop main environment')
            if compose_topology is not None:
                try:
                    await compose_topology.delete()
                except Exception:
                    logger.exception('Failed to delete Compose topology; context shutdown will release the Controller')
            shutil.rmtree(trial_dir, ignore_errors=True)
            shutil.rmtree(task_scope_dir, ignore_errors=True)

    score = float(score)
    return HarborBridgeTaskOutput(
        score=score,
        reward=reward,
        rewards=rewards,
        reward_metadata=reward_metadata,
        test_stdout=test_stdout,
        extra_fields=extra_fields,
    )


async def _materialize_task_directory(
    task_files: TaskFiles | None,
    *,
    serialized_task_dir: str | None,
    destination: Path,
) -> Path:
    """Resolve a Harbor task from portable task files or a legacy local path."""

    if task_files is not None:
        destination.mkdir(parents=True, exist_ok=False)
        await task_files.download('**/*', destination)
        return destination

    if not serialized_task_dir:
        raise ValueError('Harbor task requires task_files or a legacy task_dir')
    task_dir = Path(serialized_task_dir).expanduser().resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f'Task directory not found: {task_dir}')
    return task_dir


def _task_scope_name(instance_id: str | None) -> str:
    """Return a safe temporary directory basename for a Harbor item identity."""

    raw = str(instance_id or '').strip()
    safe = raw.replace('/', '_').replace('\\', '_').strip()
    if safe in {'', '.', '..'}:
        return 'task'
    return safe


def _compose_project_name(instance_id: str, attempt_id: str) -> str:
    """Build a bounded, attempt-unique Docker Compose project name."""
    raw = f'{instance_id}-{attempt_id}'
    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:10]
    prefix = re.sub(r'[^a-z0-9_-]+', '-', raw.lower()).strip('_-') or 'nexus'
    return f'{prefix[:52].rstrip("_-") or "nexus"}-{digest}'


def _compose_agent_helper_dir() -> Path:
    """Locate Runtime helper scripts in a source tree or packaged wheel."""
    source_path = Path(__file__).resolve().parents[3] / 'deployment' / 'docker' / 'runtime-bin'
    if source_path.is_dir():
        return source_path

    from importlib import resources

    bundled = resources.files('nexus._bundle').joinpath('deployment').joinpath('docker').joinpath('runtime-bin')
    bundled_path = Path(str(bundled))
    if bundled_path.is_dir():
        return bundled_path
    raise FileNotFoundError('Nexus package does not contain Compose Agent Runtime helpers')


async def _run_oracle(
    environment: Any,
    harbor_task: Any,
    trial_paths: Any,
    *,
    instruction: str,
    timeout_sec: float,
) -> None:
    """Run Harbor's native OracleAgent against the Gongfeng environment."""
    from harbor.agents.oracle import OracleAgent
    from harbor.models.agent.context import AgentContext

    agent_user = harbor_task.config.agent.user
    if isinstance(agent_user, int):
        raise ValueError('Harbor Oracle numeric agent UID is not supported by Gongfeng Bridge; use a username.')

    oracle = OracleAgent(
        logs_dir=trial_paths.agent_dir,
        task_dir=harbor_task.task_dir,
        trial_paths=trial_paths,
        agent_timeout_sec=float(timeout_sec),
        logger=logger,
    )
    with environment.scoped_exec_env(oracle.extra_env):
        await oracle.setup(environment)
        try:
            with environment.with_default_user(agent_user):
                await asyncio.wait_for(
                    oracle.run(
                        instruction=instruction,
                        environment=environment,
                        context=AgentContext(),
                    ),
                    timeout=float(timeout_sec),
                )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f'Oracle timed out after {timeout_sec}s') from exc
